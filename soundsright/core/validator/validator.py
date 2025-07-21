import copy
import argparse
from datetime import datetime, timedelta, timezone
from typing import List
import os
import traceback
import secrets
import time
import bittensor as bt
from async_substrate_interface import AsyncSubstrateInterface
import hashlib
import numpy as np
import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor
import sys
import logging
import pickle
from math import sqrt

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
        
        # Bittensor Objects
        self.wallet = None
        self.subtensor = None
        self.dendrite = None
        self.metagraph: bt.metagraph | None = None
        self.async_substrate = None

        # Validator Params
        self.version = Utils.config["module_version"]
        self.neuron_config = None
        self.hotkeys = None
        self.load_validator_state = None
        self.query = None
        self.debug_mode = False
        self.skip_sgmse = False
        self.dataset_size = 25
        self.log_level="INFO" # Init log level
        self.cuda_directory = ""
        self.avg_model_eval_time = 2200
        self.avg_model_size_gb = 20
        self.images_per_cpu = 3
        self.first_run_through_of_the_day = True
        self.tried_accessing_old_cache = False
        self.seed = 10
        self.seed_interval = 100
        self.last_updated_block = 0
        self.last_metagraph_sync_timestamp = 0
        self.seed_reference_block = float("inf")

        # WC Prevention
        self.algorithm = 1
        self.interval = 5400
        self.max_uids = 256 
        self.miner_nonces = {}
        self.cycle = (t := int(time.time())) - (t % self.interval)
        self.trusted_uids = []
        self.trusted_validators_filepath = os.path.join(self.base_path, "trusted_validators.txt")
        self.default_trusted_validators = []
        
        # Benchmarking / Scoring Object Init
        self.scores = None
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
            'DENOISING_16000HZ':40,
            'DEREVERBERATION_16000HZ':40,
            'DENOISING_16000HZ_remainder':10,
            'DEREVERBERATION_16000HZ_remainder':10,
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

        # Remote Logging
        self.remote_logging_interval = 3600
        self.last_remote_logging_timestamp = 0
        self.remote_logging_daily_tries=0

        # Init Functions
        self.cpu_count = Utils.get_cpu_core_count()
        self.gpu_count = Utils.get_gpu_count()
        
        self.apply_config(bt_classes=[bt.subtensor, bt.logging, bt.wallet])
        self.initialize_neuron()

        if self.wc_prevention_protcool:        
            self.init_default_trusted_validators()
            self.update_trusted_uids()

        # Helper Objects
        self.TTSHandler = Data.TTSHandler(
            tts_base_path=self.tts_path, 
            sample_rates=self.sample_rates
        )
        self.metadata_handler = Models.ModelMetadataHandler(
            subtensor=self.subtensor,
            subnet_netuid=self.neuron_config.netuid,
            log_level=self.log_level,
            wallet=self.wallet,
        )

        # Dataset download and initial benchmarking
        dataset_download_outcome = Data.dataset_download(
            wham_path = self.noise_data_path,
            arni_path = self.rir_data_path,
            log_level = self.log_level
        )
        if not dataset_download_outcome: 
            sys.exit()
        
        self.generate_new_dataset()

        self.benchmark_sgmse_for_all_competitions()

    def generate_new_dataset(self, override=True) -> None:

        # Check to see if we need to generate a new dataset
        if override:

            self.neuron_logger(
                severity="INFO",
                message=f"Generating new dataset."
            )

            # Clear existing datasets
            Data.reset_all_data_directories(
                tts_base_path=self.tts_path,
                reverb_base_path=self.reverb_path,
                noise_base_path=self.noise_path,
                log_level=self.log_level
            )

            # Generate new TTS data
            self.TTSHandler.create_openai_tts_dataset_for_all_sample_rates(n=self.dataset_size, seed=self.seed)
            
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
                seed=self.seed
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
        Returns the Unix timestamp for the next competition at 9:00 AM GMT
        that is a multiple of `period_days` after `self.start_date`.

        Args:
            start_date (datetime): The fixed base date to start counting from (should be 09:00 GMT).
            period_days (int): The interval of competition recurrence in days.
        """
        now = datetime.now(timezone.utc)

        # Compute number of full days since start_date
        delta_days = (now - self.start_date).days
        # Compute how many full periods have passed
        periods_passed = delta_days // self.period_days
        # Get the next period start
        next_competition = self.start_date + timedelta(days=(periods_passed + 1) * self.period_days)

        self.neuron_logger(
            severity="INFO",
            message=f"Next competition will be at {next_competition}. In Unix time: {int(next_competition.timestamp())}"
        )
        return int(next_competition.timestamp())

    def update_next_competition_timestamp(self) -> None:
        """
        Updates `next_competition_timestamp` to the next competition time
        that is a multiple of `period_days` after `self.start_date`.
        """
        current_time = datetime.fromtimestamp(self.next_competition_timestamp, tz=timezone.utc)
        delta_days = (current_time - self.start_date).days
        periods_passed = delta_days // self.period_days
        next_time = self.start_date + timedelta(days=(periods_passed + 1) * self.period_days)

        self.next_competition_timestamp = int(next_time.timestamp())

        self.neuron_logger(
            severity="INFO",
            message=f"Next competition will be at {next_time}"
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
        self.cuda_directory = args.cuda_directory

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

        # Init async substrate interface
        self.async_substrate = AsyncSubstrateInterface(
            url=self.neuron_config.subtensor.chain_endpoint
        )
        self.backup_async_substrate = AsyncSubstrateInterface(
            url="wss://entrypoint-finney.opentensor.ai:443"
        )

        # Setup the bittensor objects
        self.setup_bittensor_objects(self.neuron_config)

        self.neuron_logger(
            severity="INFO",
            message=f"Bittensor objects initialized:\nMetagraph: {self.metagraph}\nSubtensor: {self.subtensor}\nWallet: {self.wallet}"
        )

        self.debug_mode = args.debug_mode
        if not self.debug_mode:
            # Validate that the validator has registered to the metagraph correctly
            if not self.validator_validation(self.metagraph, self.wallet, self.subtensor):
                raise IndexError("Unable to find validator key from metagraph")

            # Get the unique identity (UID) from the network
            validator_uid = self.metagraph.hotkeys.index(self.wallet.hotkey.ss58_address)
            self.neuron_logger(
                severity="INFO",
                message=f"Validator is running with UID: {validator_uid}"
            )

        if self.wc_prevention_protcool:
            self.trusted_uids = self.resolve_trusted_uids()
            
        self.skip_sgmse = args.skip_sgmse
            
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
            host=args.healthcheck_host, port=args.healthcheck_port, is_validator = True, seed=self.seed, current_models=self.miner_models, best_models=self.best_miner_models
        )

        # Run healthcheck API
        self.healthcheck_api.run()

        self.neuron_logger(
            severity="INFO",
            message=f"HealthCheck API running at: http://{args.healthcheck_host}:{args.healthcheck_port}"
        )

        # Reset model and model output directories  
        self.neuron_logger(
            severity="TRACE",
            message=f"Resetting directory: {self.model_path}"
        )
        Utils.reset_dir(directory=self.model_path)
        self.neuron_logger(
            severity="TRACE",
            message=f"Directory reset: {self.model_path}"
        )

        self.neuron_logger(
            severity="TRACE",
            message=f"Resetting directory: {self.model_output_path}"
        )
        Utils.reset_dir(directory=self.model_output_path)  
        self.neuron_logger(
            severity="TRACE",
            message=f"Directory reset: {self.model_output_path}"
        )  

        return True
    
    def init_default_trusted_validators(self):
        if not os.path.exists(self.trusted_validators_filepath):
            with open(self.trusted_validators_filepath, 'w') as file:
                for validator in self.default_trusted_validators:
                    file.write(validator + '\n')
            self.neuron_logger(
                severity="DEBUG",
                message=f"File created and validators saved to {self.trusted_validators_filepath}"
            )

    def _validate_value(self, value) -> bool:
        # Must be uint16
        return isinstance(value, int) and 0 < value <= 100000
    
    def _validate_commit_data(self, trusted_uids: list) -> bool:
        # Must be a list with less than max_uids entries
        try:
            if isinstance(trusted_uids, list) and not isinstance(trusted_uids, bool) and len(trusted_uids) <= self.max_uids:
                return True
            else:
                return False
        except Exception:
            return False
    
    def update_cycle(self) -> bool:
        if (t := int(time.time())) - self.interval < self.cycle:
            return False
        self.cycle = t - (t % self.interval)
        return True
    
    def add_miner_nonce(self, hotkey: str, value: int) -> bool:
        if self._validate_value(value=value) and self._validate_hotkey(hotkey=hotkey):
            if hotkey in self.values.keys():
                self.values[hotkey] = value
            else:
                self.values[hotkey] = [value]

            self.neuron_logger(
                severity="TRACE",
                message=f"Added nonce: {value} to hotkey: {hotkey}. New miner values: {self.values}"
            )
            
            return True

        return False
    
    def _clear_miner_nonces(self):
        self.miner_nonces = {}

    def is_valid_ss58_address(self, hotkey):
        return True
    
    def resolve_trusted_uids(self) -> list:
        # Reads distrusted validators from file.
        # File must contain one hotkey per line

        # Read neurons in subnet to memory
        neurons = self.metagraph.neurons
        trusted_uids = []

        with open(self.trusted_validators_filepath, 'r') as f:
            for line in f:
                hotkey = line.strip()
                if self.is_valid_ss58_address(hotkey):
                    for neuron in neurons:
                        if neuron.hotkey == hotkey:
                            self.neuron_logger(
                                severity="TRACE",
                                message=f"Trusted validator hotkey: {hotkey} has uid: {neuron.uid}"
                            )
                            self.trusted_uids.append(neuron.uid)
                else:
                    raise ValueError(f'Invalid hotkey in distrusted validators file: {hotkey} ')
                
        self.trusted_uids = trusted_uids
        self.neuron_logger(
            severity="TRACE",
            message=f"Trusted UIDS: {self.trusted_uids}"
        )

    def commit_trusted_validators(self) -> tuple:
        if not self._validate_commit_data(trusted_uids=self.trusted_uids):
            return False, 0, "Failed to validate list of trusted uids"

        # Create 256 bit byte array (256 bits = 32 bytes)
        trust_state = bytearray(32)

        # Set UIDs as trusted in the bytearray
        for _,uid in enumerate(self.trusted_uids):
            trust_state[uid // 8] |= (1 << (uid % 8))

        metadata = bytes(trust_state)

        upload_outcome = asyncio.run(self.metadata_handler.upload_model_metadata_to_chain(metadata=metadata))

        if upload_outcome:
            self.neuron_logger(
                severity="DEBUG",
                message=f"Successfully committed trusted validator metadata to chain: {metadata}"
            )
        else:
            self.neuron_logger(
                severity="ERROR",
                message=f"Failed to commit trusted validator metadata to chain: {metadata}"
            )

    def update_trusted_uids(self):
        try:
            self.resolve_trusted_uids()
            self.commit_trusted_validators()
        except Exception as e:
            self.neuron_logger(
                severity="ERROR",
                message=f"Error while committing trusted uid metadata to chain: {e}"
            )

    def handle_trusted_validators(self):
        if self.update_cycle:
            self.update_trusted_uids()

    def determine_lower_and_upper_bounds(self, n):

        lower = 50000 - (0.4307 * sqrt((8.33e8 / n)))
        upper = 50000 + (0.4307 * sqrt((8.33e8 / n)))

        return lower, upper

    def determine_weight_mutation_algorithm(self):

        n = len(self.miner_nonces.keys())
        if n == 0:
            return
        
        lower_bound, upper_bound = self.determine_lower_and_upper_bounds(n=n)
        total = 0
        for value in self.miner_nonces.values():
            total += value 
        avg = total / n

        if avg <= lower_bound:
            self.algorithm = 1
        elif avg >= upper_bound:
            self.algorithm = 3
        else:
            self.algorithm = 2

        self.neuron_logger(
            severity="DEBUG",
            message=f"New weight adjustment algorithm determined: {self.algorithm}"
        )

    def _parse_args(self, parser) -> argparse.Namespace:
        return parser.parse_args()
    
    async def get_seed(self) -> int:
        """
        Obtains a seed based on a hash of the extrinsics the most 
        recent block with a block number divisible by 50.
        """

        current_block = self.subtensor.get_current_block()
        remainder = current_block % self.seed_interval
        query_block = current_block - remainder

        self.neuron_logger(
            severity="TRACE",
            message=f"Determining seed based on block with seed interval: {self.seed_interval}. Current block: {current_block}. Remainider: {remainder}. Block to query for extrinsics: {query_block}"
        )

        async with self.async_substrate:
            block_data = await self.async_substrate.get_block(block_number=query_block)
            block_extrinsics = block_data["extrinsics"]
            extrinsics_string = "".join([str(extrinsic) for extrinsic in block_extrinsics])
            hash_obj = hashlib.sha256(extrinsics_string.encode("utf-8"))
            seed = int(hash_obj.hexdigest()[:8], 16)
            self.neuron_logger(
                severity="TRACE",
                message=f"Obtained new seed: {seed} for block: {query_block}"
            )
            return seed, query_block
        
    async def get_seed_with_backup_method(self) -> int:
        """
        Obtains a seed based on a hash of the extrinsics the most 
        recent block with a block number divisible by 50. This is a backup
        that uses the default subtensor endpoint in case the first operation
        fails.
        """

        temp_subtensor = bt.subtensor(network="finney")
        current_block = temp_subtensor.get_current_block()
        remainder = current_block % self.seed_interval
        query_block = current_block - remainder

        self.neuron_logger(
            severity="TRACE",
            message=f"Determining seed with backup method based on block with seed interval: {self.seed_interval}. Current block: {current_block}. Remainider: {remainder}. Block to query for extrinsics: {query_block}"
        )

        async with self.backup_async_substrate:
            block_data = await self.backup_async_substrate.get_block(block_number=query_block)
            block_extrinsics = block_data["extrinsics"]
            extrinsics_string = "".join([str(extrinsic) for extrinsic in block_extrinsics])
            hash_obj = hashlib.sha256(extrinsics_string.encode("utf-8"))
            seed = int(hash_obj.hexdigest()[:8], 16)
            self.neuron_logger(
                severity="TRACE",
                message=f"Obtained new seed: {seed} for block: {query_block}"
            )
            return seed, query_block
        
    async def handle_update_seed_async(self):
        use_backup = False
        try:
            self.seed, self.seed_reference_block = await self.get_seed()
        except Exception as e:
            self.neuron_logger(
                severity="INFO",
                message=f"Default endpoint failed to obtain seed based on block extrinsic because: {e} Resorting to backup endpoint."
            )
            use_backup=True
        
        if use_backup:
            try:
                self.seed, self.seed_reference_block = await self.get_seed_with_backup_method()
            except Exception as e:
                self.neuron_logger(
                    severity="INFO",
                    message=f"Backup endpoint failed to obtain seed based on block extrinsic because: {e} Resorting to default seed."
                )
                self.seed = 10
                self.seed_reference_block = float("inf")

        self.healthcheck_api.update_seed(self.seed)

    def handle_update_seed(self):
        """
        Synchronous wrapper that executes handle_update_seed_async immediately,
        whether or not there's a current event loop.
        """
        try:
            # Try to get the current event loop
            loop = asyncio.get_running_loop()
            
            # If we're already in an event loop, we need to run in a separate thread
            # to avoid "RuntimeError: cannot be called from a running event loop"
            with ThreadPoolExecutor() as executor:
                future = executor.submit(asyncio.run, self.handle_update_seed_async())
                future.result()  # Wait for completion and get any exceptions
                
        except RuntimeError:
            # No event loop is running, so we can safely use asyncio.run()
            asyncio.run(self.handle_update_seed_async())
        except Exception as e:
            # Handle any other exceptions that might occur
            self.neuron_logger(
                severity="ERROR",
                message=f"Failed to execute handle_update_seed_async: {e}"
            )
            # Set fallback values
            self.seed = 10
            self.seed_reference_block = float("inf")
            self.healthcheck_api.update_seed(self.seed)

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
                        self.remove_uid_from_cache(i)
                        
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
        )
        
        self.hotkeys = copy.deepcopy(self.metagraph.hotkeys)
        
        for competition in self.miner_models:
            self.miner_models[competition] = Benchmarking.filter_models_for_deregistered_miners(
                miner_models=self.miner_models[competition],
                hotkeys=self.hotkeys
            )

    def remove_uid_from_cache(self, uid):

        new_model_cache = {}
        for comp in self.model_cache:
            new_model_cache[comp] = []
            for model_data in self.model_cache[comp]:
                if model_data["uid"] != uid:
                    new_model_cache[comp].append(model_data)
        self.model_cache = new_model_cache

    def reset_hotkey_scores(self, hotkey_index) -> None:
        self.scores[hotkey_index] = 0.0
        for competition in self.competition_scores: 
            self.competition_scores[competition][hotkey_index] = 0.0

    def adjust_scores_length(self, metagraph_len) -> None:

        self.neuron_logger(
            severity="TRACE",
            message=f"Checking if score length: {len(self.scores)} needs to be adjusted to fit metagraph length: {metagraph_len}"
        )
        
        if metagraph_len > len(self.scores):
            additional_zeros = np.zeros(
                    (metagraph_len-len(self.scores)),
                    dtype=np.float32,
                )

            self.scores = np.concatenate((self.scores, additional_zeros))
            for competition in self.competition_scores: 
                self.competition_scores[competition] = np.concatenate((self.competition_scores[competition], additional_zeros))

            self.neuron_logger(
                severity="TRACE",
                message=f"Score length: {len(self.scores)} adjusted to fit metagraph length: {metagraph_len}"
            )

    async def send_competition_synapse(self, uid_to_query: int, sample_rate: int, task: str, timeout: int = 5) -> List[bt.synapse]:
        """
        Sends synapses to obtain model metadata for DENOSIING_16000HZ and DEREVERBERATION_16000HZ competitions.
        """
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
        
    async def send_feedback_synapse(self, uid_to_query: int, competition: str, data: dict, timeout: int = 5) -> List[bt.Synapse]:
        """
        Sends FeedbackSynapse to miners 
        """
        self.neuron_logger(
            severity="DEBUG",
            message=f"Sent feedback synapse for competition: {competition} to UID: {uid_to_query} with data: {data}"
        )
        
        axon_to_query = self.metagraph.axons[uid_to_query]
        
        return await self.dendrite.forward(
            axon_to_query,
            Base.FeedbackProtocol(
                competition=competition,
                data=data,
                best_models=self.best_miner_models,
                subnet_version=self.subnet_version
            ),
            timeout=timeout,
            deserialize=True,
        )
    
    def obtain_model_feedback(self):
        """
        Organizes self.miner_models to be sent as FeedbackSynapse objects to miners
        """
        # Init model feedback as empty list
        aggregate_model_feedback = []

        # Obtain uids to query
        uids_to_query = self.get_uids_to_query()

        self.neuron_logger(
            severity="TRACE",
            message="Obtaining model feedback:"
        )

        # Iterate through each uid and find the associated model data 
        for uid in uids_to_query:

            try: 

                # Initialize empty dict for model_feedback
                model_feedback = {}

                # Find the hotkey 
                hotkey = self.hotkeys[uid]
                self.neuron_logger(
                    severity="TRACE",
                    message=f"Hotkey for uid: {uid} is: {hotkey}"
                )

                model_feedback["uid"] = uid
                
                # Iterate through competitions
                for competition in self.miner_models.keys():

                    # Iterate through models in the competition
                    for model_data in self.miner_models[competition]:

                        # If the hotkey matches
                        if model_data["hotkey"] == hotkey:

                            model_feedback["data"] = model_data
                            model_feedback["competition"] = competition

                            self.neuron_logger(
                                severity="TRACE",
                                message=f"Data for hotkey: {hotkey} for competition: {competition} is: {model_data}"
                            )
                
                # Append data to aggregate
                aggregate_model_feedback.append(model_feedback)
                            
            except Exception as e:
                self.neuron_logger(
                    severity="ERROR",
                    message=f"Error obtaining model feedback for uid: {uid}: {e}"
                )
                continue

        return aggregate_model_feedback
    
    def send_feedback_synapses(self):
        """
        Sends all feedback synapses to miners
        """
        # Obtain all model feedback organized by uid 
        model_feedback = self.obtain_model_feedback()

        self.neuron_logger(
            severity="TRACE",
            message=f"Model feedback aggregate to send to miners via FeedbackSynapse: {model_feedback}"
        )

        # Initialize asyncio loop 
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:
            # Iterate through item in model feedback
            for item in model_feedback:
                if Utils.validate_model_feedback(item):
                    
                    uid=item["uid"]
                    competition=item["competition"]
                    data=item["data"]

                    try:
                        response = loop.run_until_complete(self.send_feedback_synapse(
                            uid_to_query=uid,
                            competition=competition,
                            data=data,
                        ))
                        self.neuron_logger(
                            severity="DEBUG",
                            message=f"Recieved response for feedback synapse to uid: {uid}: {response}"
                        )
                    
                    except Exception as e:
                        self.neuron_logger(
                            severity="ERROR",
                            message=f"Error sending feedback synapse to UID: {uid} for competition: {competition} with data: {data}"
                        )

        finally:
            self.dendrite.close_session(using_new_loop=True)

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
        self.neuron_logger(
            severity="TRACE",
            message=f"Cache path: {self.cache_path}. State path: {state_path}"
        )
        old_score_version = str(int(self.score_version) - 1)
        parts = self.cache_path.split(os.sep)
        parts[-1] = old_score_version
        old_cache_path = os.sep.join(parts)
        possible_old_state_path = os.path.join(old_cache_path, "state.npz")
        self.neuron_logger(
            severity="TRACE",
            message=f"Possible old state path: {possible_old_state_path}"
        )

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
                
                self.tried_accessing_old_cache = True
                
            except Exception as e:
                self.neuron_logger(
                    severity="ERROR",
                    message=f"Validator state reset because an exception occurred: {e}"
                )
                self.reset_validator_state(state_path=state_path)
        
        elif not os.path.exists(state_path) and os.path.exists(possible_old_state_path) and not self.tried_accessing_old_cache:
            try:

                self.init_default_scores()

                self.neuron_logger(
                    severity="INFO",
                    message="Attempting to load old state in case of cache reset."
                )
                state = np.load(state_path, allow_pickle=True)
                self.neuron_logger(
                    severity="DEBUG",
                    message=f"Loaded the following old state from file: {state}"
                )
                self.scores = state["scores"]

            except Exception as e:

                self.neuron_logger(
                    severity="DEBUG",
                    message="Old cache could not be accessed. Initializing validator with defaults."
                )
                self.init_default_scores()
                self.step = 0
                self.last_updated_block = 0
                self.hotkeys = None
                self.next_competition_timestamp = self.get_next_competition_timestamp()

            self.tried_accessing_old_cache = True            
        
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

    def handle_metagraph_sync(self, override=False) -> None:
        if override or time.time() - 500 > self.last_metagraph_sync_timestamp:
            self.neuron_logger(
                severity="TRACE",
                message="Metagraph has not been synced in over 500 seconds, or the override was triggered (this happens during weight set)"
            )
            tries=0
            while tries < 5:
                try:
                    asyncio.run(self.sync_metagraph())
                    self.neuron_logger(
                        severity="INFOX",
                        message=f"Metagraph synced: {self.metagraph}"
                    )
                    self.last_metagraph_sync_timestamp = time.time()
                    return
                except TimeoutError as e:
                    self.neuron_logger(
                        severity="ERROR",
                        message=f"Metagraph sync timed out: {e}"
                    )   
                except Exception as e:
                    self.neuron_logger(
                        severity="ERROR",
                        message=f"An error occured while syncing metagraph: {e}"
                    )
                tries+=1

        else:
            current_time = time.time()
            next_update_time = current_time + 500
            self.neuron_logger(
                severity="TRACE",
                message=f"Not enough time in between intervals to sync metagraph. Current time: {current_time}. Next update time: {next_update_time} Last updated timestamp: {self.last_metagraph_sync_timestamp}."
            )

    def handle_weight_setting(self) -> None:
        """
        Checks if setting/committing/revealing weights is appropriate, triggers the process if so.
        """
        # Check if it's time to set/commit new weights
        if self.subtensor.get_current_block() >= self.last_updated_block + 350 and not self.debug_mode: 

            self.handle_metagraph_sync(override=True)
            self.check_hotkeys()

            # Try set/commit weights
            try:
                asyncio.run(self.commit_weights())
                self.last_updated_block = self.subtensor.get_current_block()

            except TimeoutError as e:
                self.neuron_logger(
                    severity="ERROR", 
                    message=f"Set weights timed out: {e}"
                )
            
            except Exception as e:
                self.neuron_logger(
                    severity="ERROR",
                    message=f"Set weights failed due to error: {e}"
                )

    def weights_mutation_alg1(self, weights: list, max_value: int) -> list:
        """Softmax scaling"""
        scaled_weights = [(w * 2.5) for w in weights]
        softmax = np.exp(scaled_weights) / np.sum(np.exp(scaled_weights))
        return ((softmax / np.max(softmax)) * max_value).tolist()
    
    def weights_mutation_alg2(self, weights: list, max_value: int) -> list:
        """Power Scaling"""
        if not weights:
            return []
        
        power = 0.63
        min_w = min(weights)
        max_w = max(weights)
        if max_w == min_w:
            return [max_value] * len(weights)
        norm_weights = [(w - min_w) / (max_w - min_w) for w in weights]
        transformed = [w ** power for w in norm_weights]
        return [w * max_value for w in transformed]

    def weights_mutation_alg3(self, weights: list, max_value: int) -> list:
        """Return weights as is"""
        return [(w * max_value) for w in weights]

    @Utils.timeout_decorator(timeout=30)
    async def commit_weights(self) -> None:
        """Sets the weights for the subnet"""

        def filter_negative_weights(weights):
            return [max(0, w) for w in weights]

        def normalize_weights_list(weights, max_value:int):
            if all(x==1 for x in weights):
                return [(x/max_value) for x in weights]
            elif all(x==0 for x in weights):
                return [0.01 for x in weights]
            else:
                return [(x/max(weights)) for x in weights]
            
        self.healthcheck_api.update_metric(metric_name='weights.targets', value=np.count_nonzero(self.scores))

        weights = self.scores
        max_value = self.subtensor.get_subnet_hyperparameters(netuid=self.neuron_config.netuid).max_weight_limit
        
        self.neuron_logger(
            severity="INFO",
            message=f"Committing weights: {weights}"
        )
        if not self.debug_mode:
        
            self.neuron_logger(
                severity="DEBUGX",
                message=f"Setting weights with the following parameters: netuid={self.neuron_config.netuid}, wallet={self.wallet}, uids={self.metagraph.uids}, weights={weights}, version_key={self.subnet_version}"
            )

            weights = normalize_weights_list(
                weights=weights,
                max_value=max_value,
            )

            weights = filter_negative_weights(weights = weights)

            if self.wc_prevention_protcool:
                # Modify according to miner nonce avg
                if self.algorithm == 1:
                    weights = self.weights_mutation_alg1(
                        weights=weights,
                        max_value=max_value,
                    )
                elif self.algorithm == 2:
                    weights = self.weights_mutation_alg2(
                        weights=weights,
                        max_value=max_value,
                    )
                else:
                    weights = self.weights_mutation_alg3(
                        weights=weights,
                        max_value=max_value,
                    )

            # This is a crucial step that updates the incentive mechanism on the Bittensor blockchain.
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

    def handle_remote_logging(self) -> None:
        """
        References last updated timestamp and specified interval
        to see if remote logging needs to be done for best models 
        from previous competition and current best models. If 
        logging is successful it updates the timestamp         
        """
        current_timestamp = int(time.time())
        
        if (self.last_remote_logging_timestamp + self.remote_logging_interval) <= current_timestamp and not self.debug_mode and self.remote_logging_daily_tries<10:
           
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
            else:
                self.remote_logging_daily_tries+=1

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
        # Check if we need to set weights during this process
        self.handle_weight_setting()

        if self.skip_sgmse:
            return
        
        competition = f"{task}_{sample_rate}HZ"
        task_path = os.path.join(self.noise_path, str(sample_rate)) if task == "DENOISING" else os.path.join(self.reverb_path, str(sample_rate))
        
        sgmse_handler = Models.SGMSEHandler(
            task = task,
            sample_rate = sample_rate,
            task_path = task_path,
            sgmse_path = self.sgmse_path,
            sgmse_output_path = self.sgmse_output_path,
            log_level=self.log_level,
            cuda_directory=self.cuda_directory,
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

    def check_if_time_to_benchmark(self) -> bool:
        """
        Checks if there is time to evaluate a new model in the current competition.
        """
        current_time = int(time.time())
        cache_length = 0
        cache_eval_time = 0
        if Utils.validate_model_cache(model_cache=self.model_cache):
            for comp_models in self.model_cache.values():
                cache_eval_time += len(comp_models) * self.avg_model_eval_time
                cache_length += len(comp_models)

        expected_eval_time = current_time + cache_eval_time

        Utils.subnet_logger(
            severity="TRACE",
            message=f"Checking if there is enough time to benchmark. Current time: {current_time}. Cache eval time is: {cache_eval_time} for cache length: {cache_length}. Expected timestamp of evaluation end is: {expected_eval_time}. Next competition is at: {self.next_competition_timestamp}"
        )

        if expected_eval_time >= self.next_competition_timestamp:
            return False 
        return True
    
    def benchmark_model(self, model_metadata: dict, sample_rate: int, task: str, hotkey: str, block: int) -> dict:
        """Runs benchmarking for miner-submitted model using Models.ModelEvaluationHandler 

        Args:
            :param model_metadata: (dict): Model metadata submitted by miner via synapse
            :param sample_rate: (int): Sample rate
            :param task: (str): DENOISING/DEREVERBERATIOn
            :param hotkey: (str): ss58_address

        Returns:
            dict: model benchmarking results. If model benchmarking could not be performed, returns an empty (no-response) dict
        """
        # Check if we need to set weights during this process
        self.handle_weight_setting()

        try:

            if not self.check_if_time_to_benchmark() and not self.first_run_through_of_the_day:
                self.neuron_logger(
                    severity="DEBUG",
                    message=f"Not enough time in current competition to benchmark model for hotkey: {hotkey}."
                )
                return False

            # Validate that miner data is formatted correctly
            if not Utils.validate_miner_response(model_metadata):
                
                self.neuron_logger(
                    severity="INFO",
                    message=f"Miner with hotkey: {hotkey} has response that was not properly formatted, cannot benchmark: {model_metadata}"
                )
                
                return None
            
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
                miner_models=self.miner_models[f'{task}_{sample_rate}HZ'],
                cuda_directory=self.cuda_directory,
                historical_block=block,
                seed_reference_block=self.seed_reference_block,
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
                
                return None
            
            self.neuron_logger(
                severity="INFO",
                message=f"Model benchmark for task: {task} and sample rate: {sample_rate}: {model_benchmark}"
            )
            
            return model_benchmark
        
        except:
            return None
    
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
                            try:
                                response = loop.run_until_complete(self.get_miner_response(
                                    uid_to_query=uid_to_query,
                                    sample_rate=sample_rate,
                                    task=task,
                                ))
                            except Exception as e:
                                self.neuron_logger(
                                    severity="ERROR",
                                    message=f"Error obtaining response from miner: {uid_to_query}: {e}"
                                )
                                continue

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
                                
                                # In case that synapse response is not formatted correctly and no known historical data:
                                if not Utils.validate_miner_response(response.data):
                                    self.neuron_logger(
                                        severity="DEBUG",
                                        message=f"Miner response is invalid: {response.data}"
                                    )
                                    continue

                                # Check if model has been evaluated today
                                competition_models_evaluated_today=Utils.extract_metadata(self.models_evaluated_today[f"{task}_{sample_rate}HZ"])
                                self.neuron_logger(
                                    severity="TRACE",
                                    message=f"Competition models evaluated today for competition: {task}_{sample_rate}HZ: {competition_models_evaluated_today}",
                                )

                                model_evaluated_today=Utils.dict_in_list(target_dict=response.data, list_of_dicts=competition_models_evaluated_today)
                                # Check if model is blacklisted
                                model_in_blacklist=Utils.dict_in_list(target_dict=response.data, list_of_dicts=blacklisted_miner_models)    

                                # If the model in the synapse is validly formatted, has not been evaluated today and is not blacklisted:
                                if not model_in_blacklist and not model_evaluated_today:

                                    # Check to see if there is historical record of this model
                                    miner_model_all_data = self.find_dict_by_hotkey(competition_miner_models, self.hotkeys[uid_to_query])

                                    # If historical records do not exist
                                    if not miner_model_all_data or Utils.check_if_historical_model_matches_current_model(current_model=response.data, historical_model=miner_model_all_data):

                                        # Append it to cache of models to evaluate
                                        self.model_cache[f"{task}_{sample_rate}HZ"].append(
                                            {
                                                "uid":uid_to_query,
                                                "response_data":response.data,
                                                "block": None,
                                            }
                                        )
                                    
                                    # If historical records do exist and are of the same model currently being submitted by the miner
                                    else:

                                        # Find block 
                                        block = miner_model_all_data.get("block", None)

                                        # Append it to cache of models to evaluate
                                        self.model_cache[f"{task}_{sample_rate}HZ"].append(
                                            {
                                                "uid":uid_to_query,
                                                "response_data":response.data,
                                                "block": block,
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

                                # Construct dict with just the namespace, name and revision
                                miner_model_data = {} 
                                if miner_model_all_data and 'hf_model_namespace' in miner_model_all_data.keys() and 'hf_model_name' in miner_model_all_data.keys() and 'hf_model_revision' in miner_model_all_data.keys():
                                    for k in ['hf_model_namespace','hf_model_name','hf_model_revision']:
                                        miner_model_data[k] = miner_model_all_data[k]

                                # Check if model has been evaluated today
                                competition_models_evaluated_today=Utils.extract_metadata(self.models_evaluated_today[f"{task}_{sample_rate}HZ"])
                                self.neuron_logger(
                                    severity="TRACE",
                                    message=f"Competition models evaluated today for competition: {task}_{sample_rate}HZ: {competition_models_evaluated_today}",
                                )
                                
                                model_evaluated_today=Utils.dict_in_list(target_dict=miner_model_data, list_of_dicts=competition_models_evaluated_today)
                                # Check if model is blacklisted
                                model_in_blacklist=Utils.dict_in_list(target_dict=miner_model_data, list_of_dicts=blacklisted_miner_models)
                                    
                                # If the recored model data is validly formatted and not blacklisted:
                                if not model_in_blacklist and not model_evaluated_today:

                                    # Obtain the block the metadata was submitted at
                                    block = miner_model_all_data.get("block", None)

                                    # Append it to cache of models to evaluate
                                    self.model_cache[f"{task}_{sample_rate}HZ"].append(
                                        {
                                            "uid":uid_to_query,
                                            "response_data":miner_model_data,
                                            "block":block,
                                        }
                                    )
            
        finally:
            self.dendrite.close_session(using_new_loop=True)

        self.neuron_logger(
            severity="TRACE",
            message=f"Pre-filter model cache: {self.model_cache}"
        )
        if not self.debug_mode:

            self.neuron_logger(
                severity="TRACE",
                message="Filtering model cache by validity.",
            )

            self.filter_cache_by_validity()

            self.neuron_logger(
                severity="TRACE",
                message=f"Filtered model cache by validity: {self.model_cache}",
            )

            self.neuron_logger(
                severity="TRACE",
                message="Filtering model cache by coldkey.",
            )

            self.filter_cache_by_ck()

            self.neuron_logger(
                severity="TRACE",
                message=f"Filtered model cache by coldkey: {self.model_cache}",
            )

    def filter_cache_by_validity(self):
        """Makes sure repo and revision exists, and revision is a commit hash."""
        new_model_cache = {}

        for competition in self.model_cache.keys():
            
            filtered_models = []

            for model in self.model_cache[competition]:

                response_data = model.get("response_data", None)

                if response_data:

                    namespace = response_data.get("hf_model_namespace", None)
                    name = response_data.get("hf_model_name", None)
                    revision = response_data.get("hf_model_revision", None)
                    
                    if namespace and name and revision and Models.validate_repo_and_revision(
                        namespace=namespace,
                        name=name,
                        revision=revision,
                        log_level=self.log_level,
                    ):

                        filtered_models.append(model)
            
            new_model_cache[competition] = filtered_models    

        self.model_cache = new_model_cache     

    def filter_cache_by_ck(self):
        """One model per competition per coldkey"""
        new_model_cache = {}

        for competition in self.model_cache.keys():
            
            filtered_models = []
            unique_models = {}

            for model in self.model_cache[competition]:

                uid = model.get("uid", None)
                if isinstance(uid, int) and 0 <= uid < len(self.metagraph.coldkeys):

                    ck = self.metagraph.coldkeys[uid]
                    
                    if ck in unique_models.keys():
                        if uid < unique_models[ck]["uid"]:
                            unique_models[ck] = model
                    
                    else:
                        unique_models[ck] = model 

            filtered_models = list(unique_models.values())
            new_model_cache[competition] = filtered_models    

        self.model_cache = new_model_cache

    def calculate_remaining_cache_length(self):

        length = 0
        for models in self.model_cache.values():
            if isinstance(models, list):
                length += len(models)

        self.neuron_logger(
            severity="TRACE",
            message=f"Length of remaining model evaluation cache: {length}. Cache: {self.model_cache}",
        )
        return length
    
    async def run_eval_loop(self, evaluator: Models.ModelEvaluationHandler, new_competition_miner_models: dict):

        self.neuron_logger(
            severity="TRACE",
            message=f"Starting eval loop."
        )

        while len(evaluator.image_hotkey_list) > 0:

            completion_ref = time.time() + (self.avg_model_eval_time * len(evaluator.image_hotkey_list))
            self.neuron_logger(
                severity="TRACE",
                message=f"Evaluation expected to end at: {completion_ref}. Next competition timestamp: {self.next_competition_timestamp}"
            )

            if completion_ref >= self.next_competition_timestamp:

                self.neuron_logger(
                    severity="TRACE",
                    message=f"Not enough time to do evaluation. Ending loop."
                )

                return new_competition_miner_models
            
            hotkeys, competitions, ports = evaluator.get_next_eval_round()

            self.neuron_logger(
                severity="TRACE",
                message=f"Next evaluation round: hotkeys: {hotkeys} competitions: {competitions} ports: {ports}"
            )

            await evaluator.get_tasks(
                hotkeys=hotkeys,
                competitions=competitions,
                ports=ports,
            )

            benchmarks, competitions = await evaluator.run_eval_group(
                hotkeys=hotkeys,
                competitions=competitions,
            )

            for benchmark, competition in zip(benchmarks, competitions):

                new_competition_miner_models[competition].append(benchmark)
                self.models_evaluated_today[competition].append(benchmark)

        return new_competition_miner_models

    def run_competitions_async(self):

        new_competition_miner_models = copy.deepcopy(self.models_evaluated_today)

        Utils.delete_container(log_level=self.log_level)

        while self.calculate_remaining_cache_length() > 0:

            self.neuron_logger(
                severity="TRACE",
                message="Running next round of async model building and evaluation."
            )

            # Initialize image builder object
            builder = Models.ModelBuilder(
                model_cache=self.model_cache,
                cuda_directory=self.cuda_directory,
                seed_reference_block=self.seed_reference_block,
                cpu_count=self.cpu_count,
                avg_model_size_gb=self.avg_model_size_gb,
                images_per_cpu=self.images_per_cpu,
                model_path=self.model_path,
                subtensor=self.subtensor,
                subnet_netuid=self.neuron_config.netuid,
                hotkeys=self.hotkeys,
                miner_models=self.miner_models,
                first_run_through_of_the_day=self.first_run_through_of_the_day,
                next_competition_timestamp=self.next_competition_timestamp,
                avg_model_eval_time=self.avg_model_eval_time,
                log_level=self.log_level,
            )

            if builder.max_image_count == 0:
                break

            # Filter out cache
            new_model_cache = builder.get_eval_round_from_model_cache()

            # Build model images async
            image_hotkey_list, competitions_list, ports_list = builder.build_images()

            # Verify outputs, if it didn't work:
            if not image_hotkey_list or not competitions_list or not ports_list:

                # End loop if validator is out of time and needs to start next competition
                if builder.time_limit:
                    break

                # Retry otherwise
                continue

            # Query eval cache from builder
            eval_cache = builder.eval_cache

            evaluator = Models.ModelEvaluationHandler(
                eval_cache=eval_cache,
                image_hotkey_list=image_hotkey_list,
                hotkeys=self.hotkeys,
                competitions_list=competitions_list,
                ports_list=ports_list,
                model_path=self.model_path,
                reverb_path=self.reverb_path,
                noise_path=self.noise_path,
                tts_path=self.tts_path,
                model_output_path=self.model_output_path,
                cuda_directory=self.cuda_directory,
                log_level=self.log_level,
            )
            
            outcome = asyncio.run(self.run_eval_loop(evaluator, new_competition_miner_models))

            if builder.time_limit or not outcome:
                break

            self.model_cache = new_model_cache

            self.handle_weight_setting()      

            Utils.delete_container(log_level=self.log_level)
        
        filtered_models = {}
        for comp in new_competition_miner_models.keys():

             # In the case that multiple models have the same hash, we only want to include the model with the earliest block when the metadata was uploaded to the chain
            hash_filtered_models = Benchmarking.filter_models_with_same_hash(
                new_competition_miner_models=new_competition_miner_models[comp],
                hotkeys=self.hotkeys
            )
            
            # In the case that multiple models have the same metadata, we only want to include the model with the earliest block when the metadata was uploaded to the chain
            hash_metadata_filtered_models = Benchmarking.filter_models_with_same_metadata(
                new_competition_miner_models=hash_filtered_models,
                hotkeys=self.hotkeys
            )

            filtered_models[comp] = hash_metadata_filtered_models

        self.miner_models = filtered_models

        self.neuron_logger(
            severity="TRACE",
            message=f"Miner models: {self.miner_models}"
        )

        self.neuron_logger(
            severity="TRACE",
            message=f"Best miner models: {self.best_miner_models}"
        )

        # Extend blacklist and remove duplicate entries
        for comp in self.blacklisted_miner_models.keys():
            self.blacklisted_miner_models[comp] = Benchmarking.remove_blacklist_duplicates(self.blacklisted_miner_models[comp])

        if self.first_run_through_of_the_day:
            self.first_run_through_of_the_day = False     

        # Reset model and model output directories  
        self.neuron_logger(
            severity="TRACE",
            message=f"Resetting directory: {self.model_path}"
        )
        Utils.reset_dir(directory=self.model_path)
        self.neuron_logger(
            severity="TRACE",
            message=f"Directory reset: {self.model_path}"
        )

        self.neuron_logger(
            severity="TRACE",
            message=f"Resetting directory: {self.model_output_path}"
        )
        Utils.reset_dir(directory=self.model_output_path)  
        self.neuron_logger(
            severity="TRACE",
            message=f"Directory reset: {self.model_output_path}"
        )  

    def reset_for_new_competition(self) -> None:
        """
        Aggregate of all the things to reset each competition
        """
        # Update timestamp to next day's 9AM (GMT)
        self.update_next_competition_timestamp()

        # Reset evaluated model cache
        for comp in self.models_evaluated_today.keys():
            self.models_evaluated_today[comp] = []

        # Reset remote logging
        self.remote_logging_daily_tries = 0

        # Reset to first run through of the day
        self.first_run_through_of_the_day = True
                    
    def run(self) -> None:
        """
        Main validator loop. 
        """ 
        self.neuron_logger(
            severity="INFO", 
            message=f"Starting validator loop with version: {self.version}"
        )
        self.healthcheck_api.append_metric(metric_name="neuron_running", value=True)

        # Update knowledge of metagraph and save state before going onto operations
        # First, sync metagraph
        self.handle_metagraph_sync()

        # Then, check that hotkey knowledge matches
        self.check_hotkeys()

        while True: 
            try: 

                timestamp1 = int(time.time()) + 600
                self.neuron_logger(
                    severity="TRACE",
                    message=f"Checking if current time: {int(time.time())} plus 600 seconds: {timestamp1} is less than next competition timestamp: {self.next_competition_timestamp} for main validator loop."
                )

                if timestamp1 < self.next_competition_timestamp:

                    self.neuron_logger(
                        severity="TRACE",
                        message="Executing main validator loop."
                    )

                    # Check to see if validator is still registered on metagraph
                    if self.wallet.hotkey.ss58_address not in self.metagraph.hotkeys:
                        self.neuron_logger(
                            severity="ERROR",
                            message=f"Hotkey is not registered on metagraph: {self.wallet.hotkey.ss58_address}."
                        )

                    # Update metadata about trusted validators if need be
                    if self.wc_prevention_protcool:
                        self.handle_trusted_validators()

                    # Sync metagraph
                    self.handle_metagraph_sync()

                    # Then, check that hotkey knowledge matches
                    self.check_hotkeys()

                    # Query for competition 
                    self.query_competitions(sample_rates=self.sample_rates, tasks=self.tasks)

                    # Run competition
                    self.run_competitions_async()

                    # Handle remote logging 
                    self.handle_remote_logging()

                    # Save validator state
                    self.save_state()

                    self.neuron_logger(
                        severity="TRACE",
                        message=f"Updating HealthCheck API."
                    )

                    # Update metrics in healthcheck API at end of each iteration
                    self.healthcheck_api.update_current_models(self.miner_models)
                    self.healthcheck_api.update_best_models(self.best_miner_models)
                    self.healthcheck_api.append_metric(metric_name='iterations', value=1)
                    self.healthcheck_api.update_rates()

                # Check if it's time for a new competition 
                timestamp2 = int(time.time())
                self.neuron_logger(
                    severity="TRACE",
                    message=f"Checking if current time: {timestamp2} is greater than next competition timestamp: {self.next_competition_timestamp} for starting next competition."
                )

                if timestamp2 >= self.next_competition_timestamp or self.debug_mode:

                    self.neuron_logger(
                        severity="INFO",
                        message="Starting new competition."
                    )

                    # Determine new seed
                    self.handle_update_seed()

                    # First reset competition scores and overall scores so that we can re-calculate them from validator model data
                    self.init_default_scores()

                    # Calculate scores for each competition
                    self.best_miner_models, self.competition_scores = Benchmarking.determine_competition_scores(
                        competition_scores = self.competition_scores,
                        competition_max_scores = self.competition_max_scores,
                        metric_proportions = self.metric_proportions,
                        sgmse_benchmarks=self.sgmse_benchmarks,
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

                    # Sync metagraph
                    self.handle_metagraph_sync()

                    # Then, check that hotkey knowledge matches
                    self.check_hotkeys()

                    self.neuron_logger(
                        severity="DEBUG",
                        message=f"Competition scores: {self.competition_scores}. Scores: {self.scores}"
                    )

                    self.neuron_logger(
                        severity="TRACE",
                        message=f"Best miner models: {self.best_miner_models}"
                    )
                    
                    self.neuron_logger(
                        severity="INFO",
                        message=f"Overall miner scores: {self.scores}"
                    )

                    # Reset validator values for new competition
                    self.reset_for_new_competition()
                    
                    # Send feedback synapses to miners
                    self.send_feedback_synapses()

                    # Update HealthCheck API
                    self.healthcheck_api.update_competition_scores(self.competition_scores)
                    self.healthcheck_api.update_scores(self.scores)
                    self.healthcheck_api.update_next_competition_timestamp(self.next_competition_timestamp)

                    # Update dataset for next day's competition
                    self.generate_new_dataset()
                    
                    # Benchmark SGMSE+ for new dataset as a comparison for miner models
                    self.benchmark_sgmse_for_all_competitions()

                    # Save validator state
                    self.save_state()

                # Handle setting of weights
                self.handle_weight_setting()

                # Sleep for a duration equivalent to 1/3 of the block time (i.e., time between successive blocks).
                self.neuron_logger(
                    severity="DEBUG", 
                    message=f"Sleeping for: {0.1} second"
                )
                time.sleep(0.1)
                
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