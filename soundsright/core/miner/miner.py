from argparse import ArgumentParser
from typing import Tuple
import sys
import bittensor as bt
import hashlib
import json 
import asyncio
import os
import traceback
import time
from dotenv import load_dotenv 
load_dotenv()

# Import custom modules
import soundsright.base as Base
import soundsright.base.utils as Utils
import soundsright.base.models as Models

class SubnetMiner(Base.BaseNeuron):
    """SubnetMiner class for SoundsRight Subnet"""

    def __init__(self, parser: ArgumentParser):
        """
        Initializes the SubnetMiner class with attributes
        neuron_config, model, tokenizer, wallet, subtensor, metagraph,
        miner_uid

        Arguments:
            parser:
                An ArgumentParser instance.

        Returns:
            None
        """
        super().__init__(parser=parser, profile="miner")

        self.neuron_config = self.config(
            bt_classes=[bt.subtensor, bt.logging, bt.wallet, bt.axon]
        )

        # Read command line arguments and perform actions based on them
        args = parser.parse_args()
        self.log_level = args.log_level
        
        # Setup logging
        bt.logging(config=self.neuron_config, logging_dir=self.neuron_config.full_path)
        if args.log_level in ("DEBUG", "DEBUGX"):
            bt.logging.enable_debug()
        elif args.log_level in ("TRACE", "TRACEX"):
            bt.logging.enable_trace()
        else:
            bt.logging.enable_default()
            
        #  Healthcheck API
        self.healthcheck_api = Utils.HealthCheckAPI(
            host=args.healthcheck_host, port=args.healthcheck_port, is_validator = False
        )

        # Run healthcheck API
        self.healthcheck_api.run()

        self.validator_min_stake = args.validator_min_stake

        self.wallet, self.subtensor, self.metagraph, self.miner_uid = self.setup()
        
        self.hotkey = self.wallet.hotkey.ss58_address
        
        self.metadata_handler = Models.ModelMetadataHandler(
            subtensor=self.subtensor,
            subnet_netuid=self.neuron_config.netuid,
            log_level=self.log_level,
            wallet=self.wallet,
        )

        self.validator_stats = {}

        self.miner_model_data = None
        
    def save_state(self):
        """Save miner state to models.json file
        """
        self.neuron_logger(
            severity="INFO",
            message="Saving miner state."
        )

        filename = os.path.join(self.cache_path, "models.json")
        
        with open(filename,"w") as json_file:
            json.dump(self.miner_model_data, json_file)
        
        self.neuron_logger(
            severity="INFOX",
            message=f"Saved the following state to file: {filename} models: {self.miner_model_data}"
        )
        
    def load_state(self):
        """Load miner state from models.json file if it exists
        """
        filename = os.path.join(self.cache_path, "models.json")
        
        # If save file exists:
        if os.path.exists(filename):
            # Load save file data for miner models
            with open(filename, "r") as json_file:
                self.miner_model_data = json.load(json_file)
            
            self.neuron_logger(
                severity="INFOX",
                message=f"Loaded the following state from file: {filename} models: {self.miner_model_data}"
            )
                
        # Otherwise start with a blank canvas and load from .env
        else: 
            self.miner_model_data = {
                "DENOISING_16000HZ":None,
                "DEREVERBERATION_16000HZ":None,
            }

    def update_miner_model_data(self):
        """Updates miner's models with new model data"""
        # Model counter (this cannot be more than 1 or it will cause an error)
        model_counter = 0
        
        # New miner model data, used as reference with existing model data to see if chain needs to be updated
        new_miner_model_data = {}
        
        # Iterate through competitions
        for sample_rate in ["16000HZ"]:
            for task in ["DENOISING","DEREVERBERATION"]:
                # Get .env params
                namespace = os.getenv(f"{task}_{sample_rate}_HF_MODEL_NAMESPACE")
                name = os.getenv(f"{task}_{sample_rate}_HF_MODEL_NAME")
                revision = os.getenv(f"{task}_{sample_rate}_HF_MODEL_REVISION")
                
                # If model is specified for this competition
                if namespace and name and revision: 
                    
                    self.neuron_logger(
                        severity="INFO",
                        message=f"Found specified model for competition: huggingface.co/{namespace}/{name}/{revision}"
                    )
                    
                    # Update new model data dict with information
                    new_miner_model_data[f'{task}_{sample_rate}'] = {
                        'hf_model_namespace':namespace,
                        'hf_model_name':name,
                        'hf_model_revision':revision,
                    }
                    # Add 1 to model counter
                    model_counter+=1
                
                # Set to None if no model data providd
                else:
                    new_miner_model_data[f'{task}_{sample_rate}'] = None 
        
        # Exit miner if model data for more than one competition detected
        if model_counter > 1:
            self.neuron_logger(
                severity="ERROR",
                message="Model data for multiple tasks and/or sample rates detected. Please register a new miner for each new task or sample rate you want to partake in. Exiting miner."
            )
            sys.exit()
            
        upload_outcome = False
            
        # Iterate through competitions to see if metadata has to be updated
        for competition in new_miner_model_data.keys():
            
            # Check that there is new model data and that it differs from old model data loaded from state
            if new_miner_model_data[competition] and competition in self.miner_model_data.keys() and new_miner_model_data[competition] != self.miner_model_data[competition]: 
                
                self.neuron_logger(
                    severity="INFO",
                    message=f"Uploading model metadata to chain: {new_miner_model_data[competition]}"
                )
                
                # Obtain competition id
                competition_id = self.metadata_handler.get_competition_id_from_competition_name(competition)
                
                # Get string of un-hashed metadata
                unhashed_metadata = f"{new_miner_model_data[competition]['hf_model_namespace']}:{new_miner_model_data[competition]['hf_model_name']}:{new_miner_model_data[competition]['hf_model_revision']}:{self.hotkey}:{competition_id}"
                
                # Hash it 
                metadata = hashlib.sha256(unhashed_metadata.encode()).hexdigest()
                
                # Upload to chain
                upload_outcome = asyncio.run(self.metadata_handler.upload_model_metadata_to_chain(metadata=metadata))
        
        # Case that we had to upload metadata to chain for new model data and it was successful
        if upload_outcome: 
            # Update miner model data so it can be saved
            self.miner_model_data = new_miner_model_data
            
            self.neuron_logger(
                severity="INFO",
                message=f"New model data has been uploaded to chain: {self.miner_model_data}. Sleeping for 60 seconds before starting miner operations."
            )
            
            # Sleep for a minute to guarantee that model metadata is uploaded to chain before the miner responds to validators
            time.sleep(60)
        
        # Case that we did not have to upload metadata to chain, or upload was not successful. In either case we default to model information saved to state
        else:
            
            self.neuron_logger(
                severity="INFO",
                message=f"Loaded miner model data from state: {self.miner_model_data} with no upload to chain necessary."
            )

    def _update_validator_stats(self, hotkey, stat_type):
        """Helper function to update the validator stats"""
        if hotkey in self.validator_stats:
            if stat_type in self.validator_stats[hotkey]:
                self.validator_stats[hotkey][stat_type] += 1
            else:
                self.validator_stats[hotkey][stat_type] = 1
        else:
            self.validator_stats[hotkey] = {}
            self.validator_stats[hotkey][stat_type] = 1

    def setup(self) -> Tuple[bt.wallet, bt.subtensor, bt.metagraph, str]:
        """This function setups the neuron.

        The setup function initializes the neuron by registering the
        configuration.

        Arguments:
            None

        Returns:
            wallet:
                An instance of bittensor.wallet containing information about
                the wallet
            subtensor:
                An instance of bittensor.subtensor
            metagraph:
                An instance of bittensor.metagraph
            miner_uid:
                An instance of int consisting of the miner UID

        Raises:
            AttributeError:
                The AttributeError is raised if wallet, subtensor & metagraph cannot be logged.
        """
        bt.logging(config=self.neuron_config, logging_dir=self.neuron_config.full_path)
        self.neuron_logger(
            severity="INFO",
            message=f"Initializing miner for subnet: {self.neuron_config.netuid} on network: {self.neuron_config.subtensor.chain_endpoint} with config:\n {self.neuron_config}"
        )

        # Setup the bittensor objects
        try:
            wallet = bt.wallet(config=self.neuron_config)
            subtensor = bt.subtensor(config=self.neuron_config)
            metagraph = subtensor.metagraph(self.neuron_config.netuid)
        except AttributeError as e:
            self.neuron_logger(
                severity="ERROR",
                message=f"Unable to setup bittensor objects: {e}"
            )
            sys.exit()

        self.neuron_logger(
            severity="INFO",
            message=f"Bittensor objects initialized:\nMetagraph: {metagraph}\
            \nSubtensor: {subtensor}\nWallet: {wallet}"
        )

        # Validate that our hotkey can be found from metagraph
        if wallet.hotkey.ss58_address not in metagraph.hotkeys:
            self.neuron_logger(
                severity="ERROR",
                message=f"Your miner: {wallet} is not registered to chain connection: {subtensor}. Run btcli register and try again"
            )
            sys.exit()

        # Get the unique identity (UID) from the network
        miner_uid = metagraph.hotkeys.index(wallet.hotkey.ss58_address)
        self.neuron_logger(
            severity="INFO",
            message=f"Miner is running with UID: {miner_uid}"
        )

        return wallet, subtensor, metagraph, miner_uid

    def check_whitelist(self, hotkey):
        """
        Checks if a given validator hotkey has been whitelisted.

        Arguments:
            hotkey:
                A str instance depicting a hotkey.

        Returns:
            True:
                True is returned if the hotkey is whitelisted.
            False:
                False is returned if the hotkey is not whitelisted.
        """

        if isinstance(hotkey, bool) or not isinstance(hotkey, str):
            return False

        whitelisted_hotkeys = [
            "5G4gJgvAJCRS6ReaH9QxTCvXAuc4ho5fuobR7CMcHs4PRbbX",  # sn14 dev team test validator
        ]

        if hotkey in whitelisted_hotkeys:
            return True

        return False
    
    def blacklist_fn(self, synapse: Base.Denoising_16kHz_Protocol | Base.Dereverberation_16kHz_Protocol) -> Tuple[bool, str]:
        """
        This function is executed before the synapse data has been
        deserialized.

        On a practical level this means that whatever blacklisting
        operations we want to perform, it must be done based on the
        request headers or other data that can be retrieved outside of
        the request data.

        As it currently stands, we want to blacklist requests that are
        not originating from valid validators. This includes:
        - unregistered hotkeys
        - entities which are not validators
        - entities with insufficient stake

        Returns:
            [True, ""] for blacklisted requests where the reason for
            blacklisting is contained in the quotes.
            [False, ""] for non-blacklisted requests, where the quotes
            contain a formatted string (f"Hotkey {synapse.dendrite.hotkey}
            has insufficient stake: {stake}",)
        """

        # Check whitelisted hotkeys (queries should always be allowed)
        if self.check_whitelist(hotkey=synapse.dendrite.hotkey):
            self.neuron_logger(
                severity="INFO",
                message=f"Accepted whitelisted hotkey: {synapse.dendrite.hotkey})"
            )
            return (False, f"Accepted whitelisted hotkey: {synapse.dendrite.hotkey}")

        # Blacklist entities that have not registered their hotkey
        if synapse.dendrite.hotkey not in self.metagraph.hotkeys:
            self.neuron_logger(
                severity="INFO",
                message=f"Blacklisted unknown hotkey: {synapse.dendrite.hotkey}"
            )
            return (
                True,
                f"Hotkey {synapse.dendrite.hotkey} was not found from metagraph.hotkeys",
            )

        # Blacklist entities that are not validators
        uid = self.metagraph.hotkeys.index(synapse.dendrite.hotkey)
        if not self.metagraph.validator_permit[uid]:
            self.neuron_logger(
                severity="INFO",
                message=f"Blacklisted non-validator: {synapse.dendrite.hotkey}"
            )
            return (True, f"Hotkey {synapse.dendrite.hotkey} is not a validator")

        # Allow all other entities
        self.neuron_logger(
            severity="INFO",
            message=f"Accepted hotkey: {synapse.dendrite.hotkey} (UID: {uid}"
        )
        return (False, f"Accepted hotkey: {synapse.dendrite.hotkey}")

    def blacklist_16kHz_denoising(self, synapse: Base.Denoising_16kHz_Protocol) -> Tuple[bool, str]:
        """Wrapper for the blacklist function to avoid repetition in code"""
        return self.blacklist_fn(synapse=synapse)

    def blacklist_16kHz_dereverberation(self, synapse: Base.Dereverberation_16kHz_Protocol) -> Tuple[bool, str]:
        """Wrapper for the blacklist function to avoid repetition in code"""
        return self.blacklist_fn(synapse=synapse)
    
    def blacklist_feedback(self, synapse: Base.FeedbackProtocol) -> Tuple[bool, str]:
        """Wrapper for the blacklist function to avoid repetition in code"""
        return self.blacklist_fn(synapse=synapse)

    def priority_fn(self, synapse: Base.Denoising_16kHz_Protocol | Base.Dereverberation_16kHz_Protocol) -> float:
        """
        This function defines the priority based on which the validators
        are selected. Higher priority value means the input from the
        validator is processed faster.
        """

        # Prioritize whitelisted validators
        if self.check_whitelist(hotkey=synapse.dendrite.hotkey):
            return 10000000.0

        # Otherwise prioritize validators based on their stake
        uid = self.metagraph.hotkeys.index(synapse.dendrite.hotkey)
        stake = float(self.metagraph.S[uid])

        self.neuron_logger(
            severity="DEBUG",
            message=f"Prioritized: {synapse.dendrite.hotkey} (UID: {uid} - Stake: {stake})"
        )

        return stake    

    def priority_16kHz_denoising(self, synapse: Base.Denoising_16kHz_Protocol) -> float:
        """Wrapper for the priority function to avoid repetition in code"""
        return self.priority_fn(synapse=synapse)

    def priority_16kHz_dereverberation(self, synapse: Base.Dereverberation_16kHz_Protocol) -> float:
        """Wrapper for the priority function to avoid repetition in code"""
        return self.priority_fn(synapse=synapse)
    
    def priority_feedback(self, synapse:Base.FeedbackProtocol) -> float:
        """Wrapper for the priority function to avoid repetition in code"""
        return self.priority_fn(synapse=synapse)

    def forward(self, synapse: Base.Denoising_16kHz_Protocol | Base.Dereverberation_16kHz_Protocol, competition: str) -> Base.Denoising_16kHz_Protocol | Base.Dereverberation_16kHz_Protocol:
        """This function responds to validators with the miner's model data"""

        hotkey = synapse.dendrite.hotkey

        self._update_validator_stats(hotkey, f"received_{competition}_competition_synapse_count")

        # Print version information and perform version checks
        self.neuron_logger(
            severity="DEBUG",
            message=f"Synapse version: {synapse.subnet_version}, our version: {self.subnet_version}"
        )

        # Set data output (None is returned if no model data is provided since it is a default in the init)
        synapse.data = self.miner_model_data[competition]

        self.neuron_logger(
            severity="INFO",
            message=f"Processed synapse from validator: {hotkey} for competition: {competition}"
        )
        
        self._update_validator_stats(hotkey, f"processed_{competition}_competition_synapse_count")

        return synapse

    def forward_16kHz_denoising(self, synapse: Base.Denoising_16kHz_Protocol) -> Base.Denoising_16kHz_Protocol:
        """Wrapper for the forward function to avoid repetition in code"""
        return self.forward(synapse=synapse, competition='DENOISING_16000HZ')

    def forward_16kHz_dereverberation(self, synapse: Base.Dereverberation_16kHz_Protocol) -> Base.Dereverberation_16kHz_Protocol:
        """Wrapper for the forward function to avoid repetition in code"""
        return self.forward(synapse=synapse, competition='DEREVERBERATION_16000HZ')
    
    def receive_feedback(self, synapse: Base.FeedbackProtocol) -> Base.FeedbackProtocol:
        """
        Function to recieve the FeedbackSynapse. Miners should modify this function if
        they wish to store the results of the validator benchmarking.
        """
        validator_hotkey = synapse.dendrite.hotkey
        competition = synapse.data.get("competition", None)
        benchmarking_data = synapse.data.get("benchmarking_data", None)

        if competition and benchmarking_data:
            self.neuron_logger(
                severity="INFO",
                message=f"Recieved feedback synapse from validator: {validator_hotkey} for competition: {competition}. Data: {benchmarking_data}"
            )
        else:
            self.neuron_logger(
                severity="WARNING",
                message=f"Recieved empty feedback synapse from validator: {validator_hotkey}. Please make sure your model config is correct with the verfication script. More information is available in the docs:\nhttps://docs.soundsright.ai/mining/model_formatting.html"
            )

        ### ADD CODE HERE IF YOU WANT TO LOG THE BENCHMARKING DATA
    
    def run(self):
        
        # Load existing model data or start with a blank slate 
        self.load_state() 
        
        # Update known miner model data with .env params
        self.update_miner_model_data()
        
        # Save this updated state
        self.save_state()
        
        # Link the miner to the Axon
        axon = bt.axon(wallet=self.wallet, config=self.neuron_config)

        self.neuron_logger(
            severity="INFO",
            message=f"Linked miner to Axon: {axon}"
        )

        # Attach the miner functions to the Axon
        axon.attach( # DENOISING 16000 HZ
            forward_fn=self.forward_16kHz_denoising,
            blacklist_fn=self.blacklist_16kHz_denoising,
            priority_fn=self.priority_16kHz_denoising,
        ).attach( # DEREVERBERATION 16000 HZ
            forward_fn=self.forward_16kHz_dereverberation,
            blacklist_fn=self.blacklist_16kHz_dereverberation,
            priority_fn=self.priority_16kHz_dereverberation,
        ).attach( # FEEDBACK SYNAPSE
            forward_fn=self.receive_feedback,
            blacklist_fn=self.blacklist_feedback,
            priority_fn=self.priority_feedback,
        )

        self.neuron_logger(
            severity="INFO",
            message=f"Attached functions to Axon: {axon}"
        )

        # Pass the Axon information to the network
        axon.serve(netuid=self.neuron_config.netuid, subtensor=self.subtensor)

        self.neuron_logger(
            severity="INFO",
            message=f"Axon served on network: {self.neuron_config.subtensor.chain_endpoint} with netuid: {self.neuron_config.netuid}"
        )
        # Activate the Miner on the network
        axon.start()
        self.neuron_logger(
            severity="INFO",
            message=f"Axon started on port: {self.neuron_config.axon.port}"
        )

        # This loop maintains the miner's operations until intentionally stopped.
        self.neuron_logger(
            severity="INFO",
            message="Miner has been initialized and we are connected to the network. Start main loop."
        )

        # Get module version
        version = Utils.config["module_version"]

        # When we init, set last_updated_block to current_block
        self.last_updated_block = self.subtensor.get_current_block()
        
        self.healthcheck_api.append_metric(metric_name="neuron_running", value=True)
        
        while True:
            try:
                # Below: Periodically update our knowledge of the network graph.
                if self.step % 600 == 0:
                    self.neuron_logger(
                        severity="DEBUG",
                        message=f"Syncing metagraph: {self.metagraph} with subtensor: {self.subtensor}"
                    )

                    self.metagraph.sync(subtensor=self.subtensor)

                    # Check registration status
                    if self.wallet.hotkey.ss58_address not in self.metagraph.hotkeys:
                        self.neuron_logger(
                            severity="SUCCESS",
                            message=f"Hotkey is not registered on metagraph: {self.wallet.hotkey.ss58_address}."
                        )

                if self.step % 60 == 0:
                    self.metagraph = self.subtensor.metagraph(self.neuron_config.netuid)
                    log = (
                        f"Version:{version} | "
                        f"Step:{self.step} | "
                        f"Block:{self.metagraph.block.item()} | "
                        f"Stake:{self.metagraph.S[self.miner_uid]} | "
                        f"Rank:{self.metagraph.R[self.miner_uid]} | "
                        f"Trust:{self.metagraph.T[self.miner_uid]} | "
                        f"Consensus:{self.metagraph.C[self.miner_uid] } | "
                        f"Incentive:{self.metagraph.I[self.miner_uid]} | "
                        f"Emission:{self.metagraph.E[self.miner_uid]}"
                    )

                    self.neuron_logger(
                        severity="INFO",
                        message=log
                    )

                    # Print validator stats
                    self.neuron_logger(
                        severity="DEBUG",
                        message=f"Validator stats: {self.validator_stats}"
                    )
                
                self.step += 1
                time.sleep(1)

            # If someone intentionally stops the miner, it'll safely terminate operations.
            except KeyboardInterrupt:
                axon.stop()
                self.neuron_logger(
                    severity="SUCCESS",
                    message="Miner killed by keyboard interrupt."
                )
                self.healthcheck_api.append_metric(metric_name="neuron_running", value=False)
                break
            # In case of unforeseen errors, the miner will log the error and continue operations.
            except Exception:
                self.neuron_logger(
                    severity="SUCCESS",
                    message=traceback.format_exc()
                )
                continue