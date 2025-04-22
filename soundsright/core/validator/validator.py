import copy
import argparse
from datetime import datetime, timedelta, timezone
from typing import List
import os
import traceback
import secrets
import time
import bittensor as bt
import numpy as np
import asyncio
import sys
import logging
import pickle

# Import custom modules
import soundsright.base.benchmarking as Benchmarking
import soundsright.base.data as Data 
import soundsright.base.utils as Utils 
import soundsright.base.models as Models 
import soundsright.base as Base

class SuppressPydanticFrozenFieldFilterDereverberation_16kHz_Protocol(logging.Filter):
    def filter(self, record):
        return 'Ignoring error when setting attribute: 1 validation error for Dereverberation_16kHz_Protocol' not in record.getMessage()

class SuppressPydanticFrozenFieldFilterDenoising_16kHz_Protocol(logging.Filter):
    def filter(self, record):
        return 'Ignoring error when setting attribute: 1 validation error for Denoising_16kHz_Protocol' not in record.getMessage()

class SubnetValidator(Base.BaseNeuron):
    """
    Main class for the SoundsRight subnet validator.
    """

    def __init__(self, parser: argparse.ArgumentParser):

        super().__init__(parser=parser, profile="validator")
        
        self.version = Utils.config["module_version"]
        self.neuron_config = None
        self.wallet = None
        self.subtensor = None
        self.dendrite = None
        self.metagraph: bt.metagraph | None = None
        self.scores = None
        self.hotkeys = None
        self.load_validator_state = None
        self.query = None
        self.debug_mode = True
        self.dataset_size = 2000
        self.weights_objects = []
        self.sample_rates = [16000]
        self.tasks = ['DENOISING','DEREVERBERATION']
        self.miner_models = {
            'DENOISING_16000HZ':[],
            'DEREVERBERATION_16000HZ':[],
        }
        self.best_miner_models = {
            'DENOISING_16000HZ':[],
            'DEREVERBERATION_16000HZ':[],
        }
        self.blacklisted_miner_models = {
            "DENOISING_16000HZ":[],
            "DEREVERBERATION_16000HZ":[],
        }
        self.competition_max_scores = {
            'DENOISING_16000HZ':50,
            'DEREVERBERATION_16000HZ':50,
        }
        self.metric_proportions = {
            "DENOISING_16000HZ":{
                "PESQ":0.3,
                "ESTOI":0.25,
                "SI_SDR":0.15,
                "SI_SIR":0.15,
                "SI_SAR":0.15,
            },
            "DEREVERBERATION_16000HZ":{
                "PESQ":0.3,
                "ESTOI":0.25,
                "SI_SDR":0.15,
                "SI_SIR":0.15,
                "SI_SAR":0.15,
            },
        }
        self.competition_scores = {
            'DENOISING_16000HZ':None,
            'DEREVERBERATION_16000HZ':None,
        }
        self.sgmse_benchmarks = {
            "DENOISING_16000HZ":None,
            "DEREVERBERATION_16000HZ":None,
        }
        self.models_evaluated_today = {
            "DENOISING_16000HZ":[],
            "DEREVERBERATION_16000HZ":[],
        }
        self.model_cache = {
            "DENOISING_16000HZ":[],
            "DEREVERBERATION_16000HZ":[],
        }

        self.remote_logging_interval = 3600
        self.last_remote_logging_timestamp = 0

        self.apply_config(bt_classes=[bt.subtensor, bt.logging, bt.wallet])
        self.initialize_neuron()        
        self.TTSHandler = Data.TTSHandler(
            tts_base_path=self.tts_path, 
            sample_rates=self.sample_rates
        )
        dataset_download_outcome = Data.dataset_download(
            wham_path = self.noise_data_path,
            arni_path = self.rir_data_path,
            log_level = self.log_level
        )
        if not dataset_download_outcome: 
            sys.exit()
        
        self.generate_new_dataset(override=False)
        
        if not self.check_wav_files():
            self.benchmark_sgmse_for_all_competitions()

    def check_wav_files(self):
        directories = [self.tts_path, self.reverb_path, self.noise_path]
        
        for dir_path in directories:
            if not os.path.isdir(dir_path):
                return False
            
            wav_files = [f for f in os.listdir(dir_path) if f.endswith('.wav')]
            if not wav_files:
                return False

        return True

    def generate_new_dataset(self, override=True) -> None:

        # Check to see if we need to generate a new dataset
        if override or not self.check_wav_files():

            self.neuron_logger(
                severity="INFO",
                message=f"Generating new dataset."
            )

            # Clear existing datasets
            Data.reset_all_data_directories(
                tts_base_path=self.tts_path,
                reverb_base_path=self.reverb_path,
                noise_base_path=self.noise_path
            )

            # Generate new TTS data
            self.TTSHandler.create_openai_tts_dataset_for_all_sample_rates(n=(3 if self.debug_mode else self.dataset_size))
            
            tts_16000 = os.path.join(self.tts_path, "16000")
            tts_files_16000 = [f for f in os.listdir(tts_16000)]
            
            self.neuron_logger(
                severity="DEBUG",
                message=f"TTS files generated in directory: {tts_16000} are: {tts_files_16000}"
            )

            # Generate new noise/reverb data
            Data.create_noise_and_reverb_data_for_all_sampling_rates(
                tts_base_path=self.tts_path,
                arni_dir_path=self.rir_data_path,
                reverb_base_path=self.reverb_path,
                wham_dir_path=self.noise_data_path,
                noise_base_path=self.noise_path,
                tasks=self.tasks,
                log_level=self.log_level,
            )

            noise_16000 = os.path.join(self.noise_path, "16000")
            noise_files_16000 = [f for f in os.listdir(noise_16000)]
            
            self.neuron_logger(
                severity="DEBUG",
                message=f"Noise files generated in directory: {noise_16000}: {noise_files_16000}"
            )
            
            reverb_16000 = os.path.join(self.reverb_path, "16000")
            reverb_files_16000 = [f for f in os.listdir(reverb_16000)]
            
            self.neuron_logger(
                severity="DEBUG",
                message=f"Reverb files generated in directory: {reverb_16000}: {reverb_files_16000}"
            )

            self.healthcheck_api.append_metric(metric_name="datasets_generated", value=1)

    def get_next_competition_timestamp(self) -> int:
        """
        Finds the Unix timestamp for the next day at 9:00 AM GMT.
        """
        # Current time in GMT
        now = datetime.now(timezone.utc)

        # Find the next day at 9:00 AM
        next_day = now + timedelta(days=1)
        next_day_at_nine = next_day.replace(hour=9, minute=0, second=0, microsecond=0)

        # Return Unix timestamp
        return int(next_day_at_nine.timestamp())

    def update_next_competition_timestamp(self) -> None:
        """
        Updates the next competition timestamp to the 9:00 AM GMT of the following day.
        """
        # Add 1 day to the current competition time
        next_competition_time = datetime.fromtimestamp(self.next_competition_timestamp, tz=timezone.utc)
        next_competition_time += timedelta(days=1)

        # Set the new timestamp
        self.next_competition_timestamp = int(next_competition_time.timestamp())

        self.neuron_logger(
            severity="INFO",
            message=f"Next competition will be at {datetime.fromtimestamp(self.next_competition_timestamp, tz=timezone.utc)}"
        )

        self.healthcheck_api.append_metric(metric_name="competitions_judged", value=1)

    def apply_config(self, bt_classes) -> bool:
        """This method applies the configuration to specified bittensor classes"""
        try:
            self.neuron_config = self.config(bt_classes=bt_classes)
        except AttributeError as e:
            self.neuron_logger(
                severity="ERROR",
                message=f"Unable to apply validator configuration: {e}"
            )
            raise AttributeError from e
        except OSError as e:
            self.neuron_logger(
                severity="ERROR",
                message=f"Unable to create logging directory: {e}"
            )
            raise OSError from e

        return True

    def validator_validation(self, metagraph, wallet, subtensor) -> bool:
        """This method validates the validator has registered correctly"""
        if wallet.hotkey.ss58_address not in metagraph.hotkeys:
            self.neuron_logger(
                severity="ERROR",
                message=f"Your validator: {wallet} is not registered to chain connection: {subtensor}. Run btcli register and try again"
            )
            return False

        return True

    def setup_bittensor_objects(self, neuron_config) -> tuple[bt.wallet, bt.subtensor, bt.dendrite, bt.metagraph]:
        """Setups the bittensor objects"""
        try:
            wallet = bt.wallet(config=neuron_config)
            subtensor = bt.subtensor(config=neuron_config)
            dendrite = bt.dendrite(wallet=wallet)
            metagraph = subtensor.metagraph(neuron_config.netuid)
        except AttributeError as e:
            self.neuron_logger(
                severity="ERROR",
                message=f"Unable to setup bittensor objects: {e}"
            )
            raise AttributeError from e

        self.hotkeys = copy.deepcopy(metagraph.hotkeys)

        self.wallet = wallet
        self.subtensor = subtensor
        self.dendrite = dendrite
        self.metagraph = metagraph

        return self.wallet, self.subtensor, self.dendrite, self.metagraph

    def initialize_neuron(self) -> bool:
        """This function initializes the neuron.

        The setup function initializes the neuron by registering the
        configuration.

        Args:
            None

        Returns:
            Bool:
                A boolean value indicating success/failure of the initialization.
        Raises:
            AttributeError:
                AttributeError is raised if the neuron initialization failed
            IndexError:
                IndexError is raised if the hotkey cannot be found from the metagraph
        """
        # Read command line arguments and perform actions based on them
        args = self._parse_args(parser=self.parser)
        self.log_level = args.log_level

        # Setup logging
        bt.logging(config=self.neuron_config, logging_dir=self.neuron_config.full_path)
        if args.log_level in ("DEBUG", "DEBUGX"):
            bt.logging.enable_debug()
        elif args.log_level in ("TRACE", "TRACEX"):
            bt.logging.enable_trace()
        else:
            bt.logging.enable_default()
        
        # Suppress specific validation errors from pydantic
        bt.logging._logger.addFilter(SuppressPydanticFrozenFieldFilterDereverberation_16kHz_Protocol())
        bt.logging._logger.addFilter(SuppressPydanticFrozenFieldFilterDenoising_16kHz_Protocol())
        
        self.neuron_logger(
            severity="INFO",
            message=f"Initializing validator for subnet: {self.neuron_config.netuid} on network: {self.neuron_config.subtensor.chain_endpoint} with config: {self.neuron_config}"
        )

        # Setup the bittensor objects
        self.setup_bittensor_objects(self.neuron_config)

        self.neuron_logger(
            severity="INFO",
            message=f"Bittensor objects initialized:\nMetagraph: {self.metagraph}\nSubtensor: {self.subtensor}\nWallet: {self.wallet}"
        )

        if not args.debug_mode:
            # Validate that the validator has registered to the metagraph correctly
            if not self.validator_validation(self.metagraph, self.wallet, self.subtensor):
                raise IndexError("Unable to find validator key from metagraph")

            # Get the unique identity (UID) from the network
            validator_uid = self.metagraph.hotkeys.index(self.wallet.hotkey.ss58_address)
            self.neuron_logger(
                severity="INFO",
                message=f"Validator is running with UID: {validator_uid}"
            )

            # Disable debug mode
            self.debug_mode = False
            
        self.dataset_size = args.dataset_size
            
        self.neuron_logger(
            severity="INFO",
            message=f"Debug mode: {self.debug_mode}"
        )
        
        self.next_competition_timestamp = self.get_next_competition_timestamp()
        
        if args.load_state == "False":
            self.load_validator_state = False
        else:
            self.load_validator_state = True

        if self.load_validator_state:
            self.load_state()
        else:
            self.init_default_scores()
        
        # Healthcheck API 
        self.healthcheck_api = Utils.HealthCheckAPI(
            host=args.healthcheck_host, port=args.healthcheck_port, is_validator = True, current_models=self.miner_models, best_models=self.best_miner_models
        )

        # Run healthcheck API
        self.healthcheck_api.run()

        self.neuron_logger(
            severity="INFO",
            message=f"HealthCheck API running at: http://{args.healthcheck_host}:{args.healthcheck_port}"
        )

        return True

    def _parse_args(self, parser) -> argparse.Namespace:
        return parser.parse_args()

    def check_hotkeys(self) -> None:
        """Checks if some hotkeys have been replaced in the metagraph"""
        if self.hotkeys is not None and np.size(self.hotkeys) > 0:
            # Check if known state len matches with current metagraph hotkey length
            if len(self.hotkeys) == len(self.metagraph.hotkeys):
                current_hotkeys = self.metagraph.hotkeys
                for i, hotkey in enumerate(current_hotkeys):
                    if self.hotkeys[i] != hotkey:
                        self.neuron_logger(
                            severity="DEBUG",
                            message=f"Index '{i}' has mismatching hotkey. Old hotkey: '{self.hotkeys[i]}', new hotkey: '{hotkey}. Resetting score to 0.0"
                        )
                        self.neuron_logger(
                            severity="DEBUG",
                            message=f"Score before reset: {self.scores[i]}, competition scores: {self.competition_scores}"
                        )
                        self.reset_hotkey_scores(i)
                        
                        self.neuron_logger(
                            severity="DEBUG",
                            message=f"Score after reset: {self.scores[i]}, competition scores: {self.competition_scores}"
                        )
            # Case that there are more/less hotkeys in metagraph 
            else:
                # Add new zero-score values 
                self.neuron_logger(
                    severity="INFO",
                    message=f"State and metagraph hotkey length mismatch. Metagraph: {len(self.metagraph.hotkeys)} State: {len(self.hotkeys)}. Adjusting scores accordingly."
                )
                self.adjust_scores_length(
                    metagraph_len=len(self.metagraph.hotkeys),
                    state_len=len(self.hotkeys)
                )
        
        self.hotkeys = copy.deepcopy(self.metagraph.hotkeys)
        
        for competition in self.miner_models:
            self.miner_models[competition] = Benchmarking.filter_models_for_deregistered_miners(
                miner_models=self.miner_models[competition],
                hotkeys=self.hotkeys
            )

    def reset_hotkey_scores(self, hotkey_index) -> None:
        self.scores[hotkey_index] = 0.0
        for competition in self.competition_scores: 
            self.competition_scores[competition][hotkey_index] = 0.0

    def adjust_scores_length(self, metagraph_len, state_len) -> None:
        if metagraph_len > state_len:
            additional_zeros = np.zeros(
                    (metagraph_len-state_len),
                    dtype=np.float32,
                )

            self.scores = np.concatenate((self.scores, additional_zeros))
            for competition in self.competition_scores: 
                self.competition_scores[competition] = np.concatenate((self.competition_scores[competition], additional_zeros))

    async def send_competition_synapse(self, uid_to_query: int, sample_rate: int, task: str, timeout: int = 5) -> List[bt.synapse]:
        # Broadcast query to valid Axons
        
        self.neuron_logger(
            severity="DEBUG",
            message=f"Sent competition synapse for {task} at {sample_rate/1000} kHz to UID: {uid_to_query}."
        )
        
        axon_to_query = self.metagraph.axons[uid_to_query]
            
        if sample_rate == 16000 and task == 'DENOISING':
            return await self.dendrite.forward(
                axon_to_query,
                Base.Denoising_16kHz_Protocol(subnet_version=self.subnet_version),
                timeout=timeout,
                deserialize=True,
            )
            
        elif sample_rate == 16000 and task == 'DEREVERBERATION':
            return await self.dendrite.forward(
                axon_to_query,
                Base.Dereverberation_16kHz_Protocol(subnet_version=self.subnet_version),
                timeout=timeout,
                deserialize=True,
            )

    def save_state(self) -> None:
        """Saves the state of the validator to a file."""
        
        state_filename = os.path.join(self.cache_path, "state.npz")
        
        self.neuron_logger(
            severity="INFO",
            message=f"Saving validator state to file: {state_filename}."
        )

        # Save the state of the validator to file.
        np.savez_compressed(
            state_filename,
            step=self.step,
            scores=self.scores,
            competition_scores_DENOISING_16000HZ=self.competition_scores['DENOISING_16000HZ'],
            competition_scores_DEREVERBERATION_16000HZ=self.competition_scores['DEREVERBERATION_16000HZ'],
            hotkeys=self.hotkeys,
            last_updated_block=self.last_updated_block,
            next_competition_timestamp=self.next_competition_timestamp,
        )

        self.neuron_logger(
            severity="INFOX",
            message=f"Saved the following state to file: {state_filename} step: {self.step}, scores: {self.scores}, competition_scores: {self.competition_scores}, hotkeys: {self.hotkeys}, last_updated_block: {self.last_updated_block}"
        )
    
        miner_models_pickle_filename = os.path.join(self.cache_path, "miner_models.pickle")
        
        self.neuron_logger(
            severity="DEBUG",
            message=f"Saving miner models to pickle file: {miner_models_pickle_filename}"
        )
        
        with open(miner_models_pickle_filename, "wb") as pickle_file:
            pickle.dump(self.miner_models, pickle_file)
   
        best_miner_models_pickle_filename = os.path.join(self.cache_path, "best_miner_models.pickle")
        
        self.neuron_logger(
            severity="DEBUG",
            message=f"Saving best miner models to pickle file: {best_miner_models_pickle_filename}"
        )

        with open(best_miner_models_pickle_filename, "wb") as pickle_file:
            pickle.dump(self.best_miner_models, pickle_file)
            
        blacklisted_miner_models_pickle_filename = os.path.join(self.cache_path, "blacklisted_miner_models.pickle")
        
        self.neuron_logger(
            severity="DEBUG",
            message=f"Saving blacklisted miner models to pickle file: {blacklisted_miner_models_pickle_filename}"
        )
        
        with open(blacklisted_miner_models_pickle_filename, "wb") as pickle_file:
            pickle.dump(self.blacklisted_miner_models, pickle_file)

    def init_default_scores(self) -> None:
        """Validators without previous validation knowledge should start
        with default score of 0.0 for each UID. The method can also be
        used to reset the scores in case of an internal error"""

        self.neuron_logger(
            severity="INFO",
            message="Initiating validator with default overall scores for each UID"
            )
        self.scores = np.zeros_like(self.metagraph.S, dtype=np.float32)
        self.neuron_logger(
            severity="INFO",
            message=f"Overall weights for validation have been initialized: {self.scores}"
        )
        self.competition_scores = {
            "DENOISING_16000HZ":None,
            "DEREVERBERATION_16000HZ":None,
        }
        for competition in self.competition_scores.keys():
            self.competition_scores[competition] = np.zeros_like(self.metagraph.S, dtype=np.float32)
            self.neuron_logger(
                severity="INFO",
                message=f"Scores for competition: {competition} have been initialized: {self.competition_scores[competition]}"
            )

    def reset_validator_state(self, state_path) -> None:
        """Inits the default validator state. Should be invoked only
        when an exception occurs and the state needs to reset."""

        # Rename current state file in case manual recovery is needed
        os.rename(
            state_path,
            f"{state_path}-{int(datetime.now().timestamp())}.autorecovery",
        )

        self.init_default_scores()
        self.step = 0
        self.last_updated_block = 0
        self.hotkeys = None
        self.next_competition_timestamp = self.get_next_competition_timestamp()

    def load_state(self) -> None:
        """Loads the state of the validator from a file."""

        # Load the state of the validator from file.
        state_path = os.path.join(self.cache_path, "state.npz")

        if os.path.exists(state_path):
            try:
                self.neuron_logger(
                    severity="INFO",
                    message="Loading validator state."
                )
                state = np.load(state_path, allow_pickle=True)
                self.neuron_logger(
                    severity="DEBUG",
                    message=f"Loaded the following state from file: {state}"
                )
                
                self.step = state["step"]
                self.neuron_logger(
                    severity="INFOX",
                    message=f"Step loaded from file: {self.step}"
                )
                
                self.scores = state["scores"]
                self.neuron_logger(
                    severity="INFOX",
                    message=f"Scores loaded from saved file: {self.scores}"
                )
                
                self.competition_scores = {
                    "DENOISING_16000HZ": state['competition_scores_DENOISING_16000HZ'],
                    "DEREVERBERATION_16000HZ": state['competition_scores_DEREVERBERATION_16000HZ']
                }
                
                self.neuron_logger(
                    severity="INFOX",
                    message=f"Competition scores loaded from file: {self.competition_scores}"
                )
                
                self.hotkeys = state["hotkeys"]
                self.neuron_logger(
                    severity="INFOX",
                    message=f"Hotkeys loaded from file: {self.hotkeys}"
                )
                
                self.last_updated_block = state["last_updated_block"]
                self.neuron_logger(
                    severity="INFOX",
                    message=f"Last updated block loaded from file: {self.last_updated_block}"
                )
                
                self.next_competition_timestamp = state['next_competition_timestamp']
                self.neuron_logger(
                    severity="INFOX",
                    message=f"Next competition timestamp loaded from file: {self.next_competition_timestamp}"
                )
                
            except Exception as e:
                self.neuron_logger(
                    severity="ERROR",
                    message=f"Validator state reset because an exception occurred: {e}"
                )
                self.reset_validator_state(state_path=state_path)
        else:
            self.init_default_scores()
            self.step = 0
            self.last_updated_block = 0
            self.hotkeys = None
            self.next_competition_timestamp = self.get_next_competition_timestamp()
            
        miner_models_filepath = os.path.join(self.cache_path, "miner_models.pickle")
        if os.path.exists(miner_models_filepath):
            try:
                with open(miner_models_filepath, "rb") as pickle_file:
                    self.miner_models = pickle.load(pickle_file)
                    
                self.neuron_logger(
                    severity="INFO",
                    message=f"Loaded miner models from {miner_models_filepath}: {self.miner_models}"
                )
            except Exception as e:
                self.neuron_logger(
                    severity="ERROR",
                    message=f"Could not load miner models from {miner_models_filepath} because: {e}"
                )
                self.miner_models = {
                    'DENOISING_16000HZ':[],
                    'DEREVERBERATION_16000HZ':[],
                }
        else:
            self.miner_models = {
                'DENOISING_16000HZ':[],
                'DEREVERBERATION_16000HZ':[],
            }
                
        best_miner_models_filepath = os.path.join(self.cache_path, "best_miner_models.pickle")
        if os.path.exists(best_miner_models_filepath):
            try:
                with open(best_miner_models_filepath, "rb") as pickle_file:
                    self.best_miner_models = pickle.load(pickle_file)
                    
                self.neuron_logger(
                    severity="INFO",
                    message=f"Loaded best miner models from {best_miner_models_filepath}: {self.best_miner_models}"
                )
            except Exception as e:
                self.neuron_logger(
                    severity="ERROR",
                    message=f"Could not load best miner models from {best_miner_models_filepath} because: {e}"
                )
                self.best_miner_models = {
                    'DENOISING_16000HZ':[],
                    'DEREVERBERATION_16000HZ':[],
                }
        else:
            self.best_miner_models = {
            'DENOISING_16000HZ':[],
            'DEREVERBERATION_16000HZ':[],
        }
            
        blacklisted_miner_models_filepath = os.path.join(self.cache_path, "blacklisted_miner_models.pickle")
        if os.path.exists(blacklisted_miner_models_filepath):
            try:
                with open(blacklisted_miner_models_filepath, "rb") as pickle_file:
                    self.blacklisted_miner_models = pickle.load(pickle_file)
                    
                self.neuron_logger(
                    severity="INFO",
                    message=f"Loaded blacklisted miner models from {blacklisted_miner_models_filepath}: {self.blacklisted_miner_models}"
                )
            except Exception as e:
                self.neuron_logger(
                    severity="ERROR",
                    message=f"Could not load blacklisted miner models from {blacklisted_miner_models_filepath} because: {e}"
                )
                self.blacklisted_miner_models = {
                    'DENOISING_16000HZ':[],
                    'DEREVERBERATION_16000HZ':[],
                }
        else:
            self.blacklisted_miner_models = {
                'DENOISING_16000HZ':[],
                'DEREVERBERATION_16000HZ':[],
            }

    @Utils.timeout_decorator(timeout=30)
    async def sync_metagraph(self) -> None:
        """Syncs the metagraph"""

        self.neuron_logger(
            severity="INFOX",
            message=f"Attempting sync of metagraph: {self.metagraph} with subtensor: {self.subtensor}"
        )

        # Sync the metagraph
        self.metagraph.sync(subtensor=self.subtensor)

    def handle_metagraph_sync(self) -> None:
        try:
            asyncio.run(self.sync_metagraph())
            self.neuron_logger(
                severity="INFOX",
                message=f"Metagraph synced: {self.metagraph}"
            )
        except TimeoutError as e:
            self.neuron_logger(
                severity="ERROR",
                message=f"Metagraph sync timed out: {e}"
            )   

    def handle_weight_setting(self) -> None:
        """
        Checks if setting/committing/revealing weights is appropriate, triggers the process if so.
        """
        # Check if it's time to set/commit new weights
        if self.subtensor.get_current_block() >= self.last_updated_block + 100 and not self.debug_mode: 

            # Try set/commit weights
            try:
                asyncio.run(self.commit_weights())
                self.last_updated_block = self.subtensor.get_current_block()

            except TimeoutError as e:
                self.neuron_logger(
                    severity="ERROR", 
                    message=f"Committing weights timed out: {e}"
                )

        # If commit reveal is enabled, reveal weights in queue
        if self.subtensor.get_subnet_hyperparameters(netuid=self.neuron_config.netuid).commit_reveal_weights_enabled:

            # Reveal weights stored in queue
            self.reveal_weights_in_queue()

    @Utils.timeout_decorator(timeout=30)
    async def commit_weights(self) -> None:
        """Sets the weights for the subnet"""

        def normalize_weights_list(weights):
            max_value = self.subtensor.get_subnet_hyperparameters(netuid=self.neuron_config.netuid).max_weight_limit
            if all(x==1 for x in weights):
                return [(x/max_value) for x in weights]
            elif all(x==0 for x in weights):
                return weights
            else:
                return [(x/max(weights)) for x in weights]
            
        self.healthcheck_api.update_metric(metric_name='weights.targets', value=np.count_nonzero(self.scores))

        weights = self.scores
        salt=secrets.randbelow(2**16)
        block = self.subtensor.get_current_block()
        uids = [int(uid) for uid in self.metagraph.uids]
        
        self.neuron_logger(
            severity="INFO",
            message=f"Committing weights: {weights}"
        )
        if not self.debug_mode:
            # Commit reveal if it is enabled
            if self.subtensor.get_subnet_hyperparameters(netuid=self.neuron_config.netuid).commit_reveal_weights_enabled:

                self.neuron_logger(
                    severity="DEBUGX",
                    message=f"Committing weights with the following parameters: netuid={self.neuron_config.netuid}, wallet={self.wallet}, uids={uids}, weights={weights}, version_key={self.subnet_version}"
                )
                # This is a crucial step that updates the incentive mechanism on the Bittensor blockchain.
                # Miners with higher scores (or weights) receive a larger share of TAO rewards on this subnet.
                result, msg = self.subtensor.commit_weights(
                    netuid=self.neuron_config.netuid,  # Subnet to set weights on.
                    wallet=self.wallet,  # Wallet to sign set weights using hotkey.
                    uids=uids,  # Uids of the miners to set weights for.
                    weights=weights,  # Weights to set for the miners.
                    salt=[salt],
                    max_retries=5,
                )
                # For successful commits
                if result:

                    self.neuron_logger(
                        severity="SUCCESS",
                        message=f"Successfully committed weights: {weights}. Message: {msg}"
                    )

                    self.healthcheck_api.update_metric(metric_name='weights.last_committed_timestamp', value=time.strftime("%H:%M:%S", time.localtime()))
                    self.healthcheck_api.append_metric(metric_name="weights.total_count_committed", value=1)

                    self._store_weight_metadata(
                        salt=salt,
                        uids=uids,
                        weights=weights,
                        block=block
                    )

                # For unsuccessful commits
                else:
                    self.neuron_logger(
                        severity="ERROR",
                        message=f"Failed to commit weights: {weights}. Message: {msg}"
                    )
            else:
                self.neuron_logger(
                    severity="DEBUGX",
                    message=f"Setting weights with the following parameters: netuid={self.neuron_config.netuid}, wallet={self.wallet}, uids={self.metagraph.uids}, weights={weights}, version_key={self.subnet_version}"
                )

                weights = normalize_weights_list(weights)

                # This is a crucial step that updates the incentive mechanism on the Bittensor blockchain.
                # Miners with higher scores (or weights) receive a larger share of TAO rewards on this subnet.
                result = self.subtensor.set_weights(
                    netuid=self.neuron_config.netuid,  # Subnet to set weights on.
                    wallet=self.wallet,  # Wallet to sign set weights using hotkey.
                    uids=self.metagraph.uids,  # Uids of the miners to set weights for.
                    weights=weights,  # Weights to set for the miners.
                    wait_for_inclusion=False,
                    version_key=self.subnet_version,
                )
                if result:
                    self.neuron_logger(
                        severity="SUCCESS",
                        message=f"Successfully set weights: {weights}"
                    )
                    
                    self.healthcheck_api.update_metric(metric_name='weights.last_set_timestamp', value=time.strftime("%H:%M:%S", time.localtime()))
                    self.healthcheck_api.append_metric(metric_name="weights.total_count_set", value=1)

                else:
                    self.neuron_logger(
                        severity="ERROR",
                        message=f"Failed to set weights: {weights}"
                    )
        else:
            self.neuron_logger(
                severity="INFO",
                message=f"Skipped setting weights due to debug mode"
            )

    def _store_weight_metadata(self, salt, uids, weights, block) -> None:
        """Stores weight metadata as part of the SubnetValidator.weights_objects attribute

        Args:
            salt (int): Unique salt for weights.
            uids (list): Uids to set weights for
            weights (np.ndarray)): Weights array
            block (int): What block weights were initially committed to chain
        """
        # Construct weight object
        data = {
            "salt": salt,
            "uids": uids,
            "weights": weights,
            "block": block
        }

        # Store weight object
        self.weights_objects.append(data)

        self.neuron_logger(
            severity='TRACE',
            message=f'Weight data appended to weights_objects for future reveal: {data}'
        )

    @Utils.timeout_decorator(timeout=30)
    async def reveal_weights(self, weight_object) -> bool:
        """
        Reveals weights (in the case that commit reveal is enabled for the subnet)
        
        Args: 
            :param weight_object: (dict): Validator's local log of weights to be revealed
            
        Returns: 
            bool: True if weights were revealed successfully, False otherwise
        """
        self.neuron_logger(
            severity="INFO",
            message=f"Revealing weights: {weight_object}"
        )

        status, msg = self.subtensor.reveal_weights(
            wallet=self.wallet,
            netuid=self.neuron_config.netuid,
            uids=weight_object["uids"],
            weights=weight_object["weights"],
            salt=np.array([weight_object["salt"]]),
            max_retries=5
        )

        if status: 
            self.neuron_logger(
                severity="SUCCESS",
                message=f'Weight reveal succeeded for weights: {weight_object} Status message: {msg}'
            )
            self.healthcheck_api.update_metric(metric_name='weights.last_revealed_timestamp', value=time.strftime("%H:%M:%S", time.localtime()))
            self.healthcheck_api.append_metric(metric_name="weights.total_count_revealed", value=1)

        else:
            self.neuron_logger(
                severity="ERROR",
                message=f'Weight reveal failed. Status message: {msg}'
            )

        return status

    def reveal_weights_in_queue(self) -> None:
        """
        Looks through queue, sees if any weight objects are at/past the time to reveal them. Reveals them if this is the case
        """
        current_block = self.subtensor.get_current_block()
        commit_reveal_weights_interval = self.subtensor.get_subnet_hyperparameters(netuid=self.neuron_config.netuid).commit_reveal_weights_interval
        new_weights_objects = []

        for weight_object in self.weights_objects:
            if (current_block - weight_object['block']) >= commit_reveal_weights_interval:
                try: 
                    status = asyncio.run(self.reveal_weights(weight_object=weight_object))
                    if not status: 
                        new_weights_objects.append(weight_object)
                except TimeoutError as e:
                    self.neuron_logger(
                        severity="ERROR", 
                        message=f"Revealing weights timed out: {e}"
                    )
                    new_weights_objects.append(weight_object)

            else:
                new_weights_objects.append(weight_object)

        self.weights_objects = new_weights_objects

        self.neuron_logger(
            severity="TRACE",
            message=f"Weights objects in queue to be revealed: {self.weights_objects}"
        )

    def handle_remote_logging(self) -> None:
        """
        References last updated timestamp and specified interval
        to see if remote logging needs to be done for best models 
        from previous competition and current best models. If 
        logging is successful it updates the timestamp         
        """
        current_timestamp = int(time.time())
        
        if (self.last_remote_logging_timestamp + self.remote_logging_interval) <= current_timestamp and not self.debug_mode:
           
            # Log models for current competition
            current_models_outcome = Benchmarking.miner_models_remote_logging(
                hotkey=self.wallet.hotkey,
                current_miner_models=self.miner_models,
                log_level=self.log_level,
            )
            
            sgmse_outcome = Benchmarking.sgmse_remote_logging(
                hotkey=self.wallet.hotkey,
                sgmse_benchmarks=self.sgmse_benchmarks,
                log_level=self.log_level,
            )
            
            if current_models_outcome and sgmse_outcome:
                self.last_remote_logging_timestamp = int(time.time())

    def get_uids_to_query(self) -> List[int]:
        """This function determines valid axon to send the query to--
        they must have valid ips """
        axons = self.metagraph.axons
        # Clear axons that do not have an IP
        axons_with_valid_ip = [axon for axon in axons if axon.ip != "0.0.0.0"]

        # Clear axons with duplicate IP/Port 
        axon_ips = set()
        filtered_axons = [
            axon
            for axon in axons_with_valid_ip
            if axon.ip_str() not in axon_ips and not axon_ips.add(axon.ip_str())
        ]

        self.neuron_logger(
            severity="TRACEX",
            message=f"Filtered out axons. Original list: {len(axons)}, filtered list: {len(filtered_axons)}"
        )

        self.healthcheck_api.append_metric(metric_name="axons.total_filtered_axons", value=len(filtered_axons))

        return [self.hotkeys.index(axon.hotkey) for axon in filtered_axons]

    def find_dict_by_hotkey(self, dict_list, hotkey) -> dict | None:
        """_summary_

        Args:
            :param dict_list: (List[dict]): List of dictionaries
            :param hotkey: (str): ss58_adr

        Returns:
            dict: if hotkey in dict_list. None otherwise
        """
        for d in dict_list:
            if d.get('hotkey') == hotkey:
                return d
        return {}
        
    def benchmark_sgmse(self, sample_rate: int, task: str) -> None:
        """Runs benchmarking for SGMSE for competition based on current dataset

        Args:
            sample_rate (int): Sample rate
            task (str): DENOISING/DEREVERBERATION
        """
        
        competition = f"{task}_{sample_rate}HZ"
        task_path = os.path.join(self.noise_path, str(sample_rate)) if task == "DENOISING" else os.path.join(self.reverb_path, str(sample_rate))
        
        sgmse_handler = Models.SGMSEHandler(
            task = task,
            sample_rate = sample_rate,
            task_path = task_path,
            sgmse_path = self.sgmse_path,
            sgmse_output_path = self.sgmse_output_path,
            log_level=self.log_level,
        )
        
        sgmse_benchmarking_outcome = sgmse_handler.download_start_and_enhance()
        
        # Calculate metrics
        if sgmse_benchmarking_outcome:
            metrics_dict = Benchmarking.calculate_metrics_dict(
                clean_directory=os.path.join(self.tts_path, str(sample_rate)),
                enhanced_directory=self.sgmse_output_path,
                noisy_directory=task_path,
                sample_rate=sample_rate,
                log_level=self.log_level,
            )
        
            # Append metrics to dict
            self.sgmse_benchmarks[competition] = metrics_dict
            
            self.neuron_logger(
                severity="TRACE",
                message=f"Determined SGMSE+ benchmarks for {competition} competition: {self.sgmse_benchmarks[competition]}"
            )
            
    def benchmark_sgmse_for_all_competitions(self) -> None:
        """Runs benchmarking for SGMSE+ for all competitions, sends results to remote logger
        """
        self.neuron_logger(
            severity="INFO",
            message="Benchmarking SGMSE+ on today's dataset."
        )
        
        # Reset benchmarking dic    
        for competition_key in self.sgmse_benchmarks.keys():
            self.sgmse_benchmarks[competition_key] = None
        
        for sample_rate in self.sample_rates: 
            for task in self.tasks:                
                # Benchmark SGMSE+ on dataset
                self.benchmark_sgmse(sample_rate=sample_rate, task=task)
                
        self.neuron_logger(
            severity="INFO",
            message=f"SGMSE+ benchmarks: {self.sgmse_benchmarks}"
        )
    
    def benchmark_model(self, model_metadata: dict, sample_rate: int, task: str, hotkey: str) -> dict:
        """Runs benchmarking for miner-submitted model using Models.ModelEvaluationHandler 

        Args:
            :param model_metadata: (dict): Model metadata submitted by miner via synapse
            :param sample_rate: (int): Sample rate
            :param task: (str): DENOISING/DEREVERBERATIOn
            :param hotkey: (str): ss58_address

        Returns:
            dict: model benchmarking results. If model benchmarking could not be performed, returns an empty (no-response) dict
        """
        # Validate that miner data is formatted correctly
        if not Utils.validate_miner_response(model_metadata):
            
            self.neuron_logger(
                severity="INFO",
                message=f"Miner with hotkey: {hotkey} has response that was not properly formatted, cannot benchmark: {model_metadata}"
            )
            
            return {
                'hotkey':hotkey,
                'hf_model_name':'',
                'hf_model_namespace':'',
                'hf_model_revision':'',
                'model_hash':'',
                'block':10000000000000000,
                'metrics':{}
            }
        
        # Initialize model evaluation handler
        eval_handler = Models.ModelEvaluationHandler(
            tts_base_path=self.tts_path,
            noise_base_path=self.noise_path,
            reverb_base_path=self.reverb_path,
            model_output_path=self.model_output_path,
            model_path=self.model_path,
            sample_rate=sample_rate,
            task=task,
            hf_model_namespace=model_metadata['hf_model_namespace'],
            hf_model_name=model_metadata['hf_model_name'],
            hf_model_revision=model_metadata['hf_model_revision'],
            log_level=self.log_level,
            subtensor=self.subtensor,
            subnet_netuid=self.neuron_config.netuid,
            miner_hotkey=hotkey,
            miner_models=self.miner_models[f'{task}_{sample_rate}HZ']
        )
        
        metrics_dict, model_hash, model_block = eval_handler.download_run_and_evaluate()
        
        model_benchmark = {
            'hotkey':hotkey,
            'hf_model_name':model_metadata['hf_model_name'],
            'hf_model_namespace':model_metadata['hf_model_namespace'],
            'hf_model_revision':model_metadata['hf_model_revision'],
            'model_hash':model_hash,
            'block':model_block,
            'metrics':metrics_dict,
        }
        
        if not Utils.validate_model_benchmark(model_benchmark):
            self.neuron_logger(
                severity="INFO",
                message=f"Model benchmark: {model_benchmark} for task: {task} and sample rate: {sample_rate} is invalidly formatted."
            )
            
            return {
                'hotkey':hotkey,
                'hf_model_name':'',
                'hf_model_namespace':'',
                'hf_model_revision':'',
                'model_hash':'',
                'metrics':{},
                'block':10000000000000000,
            }
        
        self.neuron_logger(
            severity="INFO",
            message=f"Model benchmark for task: {task} and sample rate: {sample_rate}: {model_benchmark}"
        )
        
        return model_benchmark
    
    async def get_miner_response(self, uid_to_query, sample_rate, task):
        response = await self.send_competition_synapse(
            uid_to_query=uid_to_query,
            sample_rate=sample_rate, 
            task=task
        )
        return response

    def query_competitions(self, sample_rates, tasks) -> None:
        """
        Runs a competition (a competition is a unique combination of sample rate and task).
        
        1. Queries all miners for their models.
        2. If miner submits a new model, benchmarks it with SubnetValidator.benchmark_model 
        3. Updates knowledge of miner model benchmarking results.
        """
        self.model_cache = {
            "DENOISING_16000HZ":[],
            "DEREVERBERATION_16000HZ":[],
        }
        
        # Initialize asyncio loop
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:
            
            # Iterate through sample rates
            for sample_rate in sample_rates:
                # Iterate through tasks
                for task in tasks:
                    
                    self.neuron_logger(
                        severity="INFO",
                        message=f"Obtaining models for competition: {task}_{sample_rate}HZ"
                    )
                    
                    # Obtain existing list of miner model data for this competition
                    competition_miner_models = self.miner_models[f"{task}_{sample_rate}HZ"]
                    blacklisted_miner_models = self.blacklisted_miner_models[f"{task}_{sample_rate}HZ"]
                    
                    # Iterate through UIDs to query
                    for uid_to_query in self.get_uids_to_query():

                        if Utils.validate_uid(uid_to_query):

                            # Send synapse
                            response = loop.run_until_complete(self.get_miner_response(
                                uid_to_query=uid_to_query,
                                sample_rate=sample_rate,
                                task=task,
                            ))

                            # Add this data to the HealthCheck API
                            self.healthcheck_api.append_metric(metric_name="axons.total_queried_axons", value=1)

                            # Check that the miner has responded with a model for this competition. If not, skip it 
                            if response and response.data:
                                
                                self.neuron_logger(
                                    severity="TRACE",
                                    message=f"Recieved response from miner with UID: {uid_to_query}: {response.data}"
                                )
                                
                                # Add this data to HealthCheck API 
                                self.healthcheck_api.append_metric(metric_name="responses.total_valid_responses", value=1)
                                
                                # Check that the synapse response is validly formatted
                                valid_model=False
                                if isinstance(response.data, dict) and 'hf_model_namespace' in response.data and 'hf_model_name' in response.data and 'hf_model_revision' in response.data and response.data['hf_model_namespace'] != "temp":
                                    valid_model=True
                                
                                # In case that synapse response is not formatted correctly and no known historical data:
                                if not valid_model:
                                    self.neuron_logger(
                                        severity="DEBUG",
                                        message=f"Miner response is invalid: {response.data}"
                                    )
                                    continue
                                    
                                # If the model in the synapse is validly formatted, has not been evaluated today and is not blacklisted:
                                if (miner_model_data not in blacklisted_miner_models) and valid_model and (response.data not in self.models_evaluated_today[f"{task}_{sample_rate}HZ"]):
                                    
                                    # Append it to cache of models to evaluate
                                    self.model_cache[f"{task}_{sample_rate}HZ"].append(
                                        {
                                            "uid":uid_to_query,
                                            "response_data":response.data,
                                        }
                                    )

                            # In the case of empty rersponse:
                            else: 

                                # Add this data to the HealthCheck API 
                                self.healthcheck_api.append_metric(metric_name="responses.total_invalid_responses", value=1)

                                # Find miner model data
                                miner_model_all_data = self.find_dict_by_hotkey(competition_miner_models, self.hotkeys[uid_to_query])
                                
                                # In case of no known historical data:
                                if not miner_model_all_data:
                                    self.neuron_logger(
                                        severity="DEBUG",
                                        message=f"Miner model data is invalid."
                                    )
                                    continue

                                miner_model_data = {} 
                                if miner_model_all_data and 'hf_model_namespace' in miner_model_all_data.keys() and 'hf_model_name' in miner_model_all_data.keys() and 'hf_model_revision' in miner_model_all_data.keys():
                                    for k in ['hf_model_namespace','hf_model_name','hf_model_revision']:
                                        miner_model_data[k] = miner_model_all_data[k]
                                    
                                # If the model in the synapse is validly formatted and not blacklisted:
                                if (miner_model_data not in blacklisted_miner_models) and (miner_model_data not in self.models_evaluated_today[f"{task}_{sample_rate}HZ"]):
                                    
                                    # Append it to cache of models to evaluate
                                    self.model_cache[f"{task}_{sample_rate}HZ"].append(
                                        {
                                            "uid":uid_to_query,
                                            "response_data":miner_model_data,
                                        }
                                    )
            
        finally:
            loop.close()
            
    def run_competitions(self, sample_rates, tasks) -> None:
            
        # Iterate through sample rates
        for sample_rate in sample_rates:
            # Iterate through tasks
            for task in tasks:
                
                self.neuron_logger(
                    severity="INFO",
                    message=f"Evaluating for competition: {task}_{sample_rate}HZ"
                )

                # Create new list which we will gradually append to and eventually replace self.miner_models with
                new_competition_miner_models = []
                
                # Obtain competition models
                models_to_evaluate = self.model_cache[f"{task}_{sample_rate}HZ"]
                
                # Iterate through models to evaluate
                for model_to_evaluate in models_to_evaluate:
                    
                    # Obtain uid and response data
                    uid, response_data = model_to_evaluate['uid'], model_to_evaluate['response_data']
                    
                    # Create a dictionary logging miner model metadata & benchmark values
                    model_data = self.benchmark_model(
                        model_metadata = response_data,
                        sample_rate = sample_rate,
                        task = task,
                        hotkey = self.hotkeys[uid],
                    )

                    # Append to the list
                    new_competition_miner_models.append(model_data)
                    
                    # Append to daily cache
                    self.models_evaluated_today[f"{task}_{sample_rate}HZ"].append(response_data)
                
                # In the case that multiple models have the same hash, we only want to include the model with the earliest block when the metadata was uploaded to the chain
                hash_filtered_new_competition_miner_models, same_hash_blacklist = Benchmarking.filter_models_with_same_hash(
                    new_competition_miner_models=new_competition_miner_models
                )
                
                # In the case that multiple models have the same metadata, we only want to include the model with the earliest block when the metadata was uploaded to the chain
                hash_metadata_filtered_new_competition_miner_models, same_metadata_blacklist = Benchmarking.filter_models_with_same_metadata(
                    new_competition_miner_models=hash_filtered_new_competition_miner_models
                )
                
                self.blacklisted_miner_models[f"{task}_{sample_rate}HZ"].extend(same_hash_blacklist)
                self.blacklisted_miner_models[f"{task}_{sample_rate}HZ"].extend(same_metadata_blacklist)
                self.miner_models[f"{task}_{sample_rate}HZ"] = hash_metadata_filtered_new_competition_miner_models

                competition = f"{task}_{sample_rate}HZ"
                self.neuron_logger(
                    severity="DEBUG",
                    message=f"Models for competition: {competition}: {self.miner_models[competition]}"
                )
                    
                    
    def run(self) -> None:
        """
        Main validator loop. 
        """ 
        self.neuron_logger(
            severity="INFO", 
            message=f"Starting validator loop with version: {self.version}"
        )
        self.healthcheck_api.append_metric(metric_name="neuron_running", value=True)

        while True: 
            try: 
                # Update knowledge of metagraph and save state before going onto a new competition
                # First, sync metagraph
                self.handle_metagraph_sync()

                # Then, check that hotkey knowledge matches
                self.check_hotkeys()

                # Check to see if validator is still registered on metagraph
                if self.wallet.hotkey.ss58_address not in self.metagraph.hotkeys:
                    self.neuron_logger(
                        severity="ERROR",
                        message=f"Hotkey is not registered on metagraph: {self.wallet.hotkey.ss58_address}."
                    )

                # Save validator state
                self.save_state()
                
                # Query miners 
                self.query_competitions(sample_rates=self.sample_rates, tasks=self.tasks)
                
                # Benchmark models
                self.run_competitions(sample_rates=self.sample_rates, tasks=self.tasks)

                # Check if it's time for a new competition 
                if int(time.time()) >= self.next_competition_timestamp or self.debug_mode:

                    self.neuron_logger(
                        severity="INFO",
                        message="Starting new competition."
                    )
                    
                    # First, sync metagraph
                    self.handle_metagraph_sync()

                    # Then, check that hotkey knowledge matches
                    self.check_hotkeys()

                    # First reset competition scores and overall scores so that we can re-calculate them from validator model data
                    self.init_default_scores()

                    # Calculate scores for each competition
                    self.best_miner_models, self.competition_scores = Benchmarking.determine_competition_scores(
                        competition_scores = self.competition_scores,
                        competition_max_scores = self.competition_max_scores,
                        metric_proportions = self.metric_proportions,
                        best_miner_models = self.best_miner_models,
                        miner_models = self.miner_models,
                        metagraph = self.metagraph,
                        log_level = self.log_level,
                    )

                    # Update validator.scores 
                    self.scores = Benchmarking.calculate_overall_scores(
                        competition_scores = self.competition_scores,
                        scores = self.scores,
                        log_level = self.log_level,
                    )
                    
                    self.neuron_logger(
                        severity="INFO",
                        message=f"Overall miner scores: {self.scores}"
                    )
                    
                    # Update HealthCheck API
                    self.healthcheck_api.update_competition_scores(self.competition_scores)
                    self.healthcheck_api.update_scores(self.scores)
                    
                    # Reset evaluated model cache
                    for comp in self.models_evaluated_today.keys():
                        self.models_evaluated_today[comp] = []

                    # Update timestamp to next day's 9AM (GMT)
                    self.update_next_competition_timestamp()

                    # Update dataset for next day's competition
                    self.generate_new_dataset()
                    
                    # Benchmark SGMSE+ for new dataset as a comparison for miner models
                    self.benchmark_sgmse_for_all_competitions()

                # Handle setting of weights
                self.handle_weight_setting()
                
                # Handle remote logging 
                self.handle_remote_logging()

                self.neuron_logger(
                    severity="TRACE",
                    message=f"Updating HealthCheck API."
                )

                # Update metrics in healthcheck API at end of each iteration
                self.healthcheck_api.update_current_models(self.miner_models)
                self.healthcheck_api.update_best_models(self.best_miner_models)
                self.healthcheck_api.append_metric(metric_name='iterations', value=1)
                self.healthcheck_api.update_rates()
                
                self.neuron_logger(
                    severity="DEBUG",
                    message=f"Competition scores: {self.competition_scores}. Scores: {self.scores}"
                )

                # Sleep for a duration equivalent to 1/3 of the block time (i.e., time between successive blocks).
                self.neuron_logger(
                    severity="DEBUG", 
                    message=f"Sleeping for: {bt.BLOCKTIME/3} seconds"
                )
                time.sleep(bt.BLOCKTIME / 3)

            # If we encounter an unexpected error, log it for debugging.
            except RuntimeError as e:
                self.neuron_logger(
                    severity="ERROR",
                    message=e
                )
                traceback.print_exc()

            # If the user interrupts the program, gracefully exit.
            except KeyboardInterrupt:
                self.neuron_logger(
                    severity="SUCCESS", 
                    message="Keyboard interrupt detected. Exiting validator.")
                sys.exit()

            # If we encounter a general unexpected error, log it for debugging.
            except Exception as e:
                self.neuron_logger(
                    severity="ERROR",
                    message=e
                )
                traceback.print_exc()