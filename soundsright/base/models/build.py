import bittensor as bt
import asyncio
import hashlib
import os 
import shutil
import time
from math import floor

import soundsright.base.utils as Utils
import soundsright.base.models as Models

class ModelBuilder:
    """
    Class responsible for verifying models and building model images async

    Workflow:
    1. Initialize
    - This process determines the maximum number of images we can build 
    2. Determine part of the model cache we want to build with get_eval_round_from_model_cache. 
    - This downloads the models in a folder named with the miner hotkey as does all the validation as well
    3. Build images with build_images
    - This returns a list of hotkeys whose model images were attempted to be built and a list of booleans 
      whether or not the operation was successful.
    - Each build is tagged as modelapi_{hotkey}
    """

    def __init__(
        self,
        model_cache: dict,
        cuda_directory: str,
        seed_reference_block: int | float,
        cpu_count: int,
        avg_model_size_gb: int,
        images_per_cpu: int,
        model_path: str,
        subtensor: bt.subtensor,
        subnet_netuid: int,
        hotkeys: list,
        miner_models: dict,
        first_run_through_of_the_day: bool,
        next_competition_timestamp: int,
        avg_model_eval_time: int,
        log_level: str,
    ):

        # Machine
        self.cuda_directory = cuda_directory
        self.free_storage_gb = Utils.get_free_space_gb()
        self.avg_model_size_gb = avg_model_size_gb
        self.cpu_count = cpu_count
        self.images_per_cpu = images_per_cpu
        self.first_run_through_of_the_day = first_run_through_of_the_day
        self.next_competition_timestamp = next_competition_timestamp
        self.avg_model_eval_time = avg_model_eval_time

        # Calculations
        self.max_image_count = 0
        self.time_limit = False
        
        cpu_max_count = self.cpu_count * self.images_per_cpu
        storage_max_count = floor(self.free_storage_gb / self.avg_model_size_gb)
        time_limit_max_count = floor((self.next_competition_timestamp - int(time.time()) - 600) / self.avg_model_eval_time)

        if not self.first_run_through_of_the_day:
            self.max_image_count = min([cpu_max_count, storage_max_count, time_limit_max_count])
            if self.max_image_count == time_limit_max_count:
                self.time_limit = True
        else:
            min([cpu_max_count, storage_max_count])

        # Bittensor
        self.hotkeys = hotkeys
        self.seed_reference_block = seed_reference_block
        self.metadata_handler = Models.ModelMetadataHandler(
            subtensor=subtensor, 
            subnet_netuid=subnet_netuid,
            log_level=self.log_level
        )
    
        # Model
        self.model_base_path = model_path
        self.forbidden_model_hashes = [
            "ENZIdw0H8Vbb79lXDQKBqqReXIj2ycgOX1Ob0QoexAU=",
            "Mbx0++bk5q6n+rdVlUblElnturj/zRobTk61WFVHmgg=",
        ]
        self.miner_models = miner_models
        self.model_cache = model_cache
        self.eval_cache = {}

        # Misc
        self.log_level = log_level

    def prepare_directory(self, dir_path):
        """
        Creates directory if it does not exist, removes contents if it does
        """
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
            Utils.subnet_logger(
                severity="TRACE",
                message=f"Created directory: {dir_path}",
                log_level=self.log_level
            )

        else:
            for item in os.listdir(dir_path):
                item_path = os.path.join(dir_path, item)
                try:
                    if os.path.isfile(item_path) or os.path.islink(item_path):
                        os.unlink(item_path)
                    elif os.path.isdir(item_path):
                        shutil.rmtree(item_path)

                except Exception as e:
                    Utils.subnet_logger(
                        severity="ERROR",
                        message=f"Failed to delete item at path: {item_path} because: {e}",
                        log_level=self.log_level
                    )

    def _reset_dir(self, directory: str) -> None:
        """Removes all files and sub-directories in an inputted directory

        Args:
            directory (str): Directory to reset.
        """
        # Check if the directory exists
        if not os.path.exists(directory):
            return

        # Loop through all the files and subdirectories in the directory
        for filename in os.listdir(directory):
            file_path = os.path.join(directory, filename)
            
            # Check if it's a file or directory and remove accordingly
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)  # Remove the file or link
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)  # Remove the directory and its contents
            except Exception as e:
                Utils.subnet_logger(
                    severity="ERROR",
                    message=f"Failed to delete {file_path}. Reason: {e}",
                    log_level=self.log_level
                )

    def get_eval_round_from_model_cache(self):

        remainder_cache = {}
        counter = 0
        
        for competition in self.model_cache:

            remainder_cache[competition] = []
            self.eval_cache[competition] = []

            for model_data in self.model_cache[competition]:

                if counter < self.max_image_count:
                
                    if self.validate_model(
                        model_data=model_data,
                        competition=competition,
                    ):

                        self.eval_cache[competition].append(model_data)

                        counter += 1
                    
                else:

                    remainder_cache[competition].append(model_data)

        return remainder_cache

    def validate_model(self, model_data: dict, competition: str):
        """
        Model validation includes:
        
        1. Verifying model data structure 
        2. Verifying on-chain model metadata
        3. Verifying upload block is before seed determination
        4. Downloading and verifying model repository content
        
        """
        try:
            # 1. Verifying model data structure
            if not isinstance(model_data, dict):
                return False 

            uid = model_data.get("uid", None)
            model_metadata = model_data.get("response_data", None)

            if not uid or not model_metadata or not isinstance(uid, int) or not isinstance(model_metadata, dict):
                return False 
            
            hotkey = self.hotkeys[uid]
            namespace = model_metadata.get("hf_model_namespace", None)
            name = model_metadata.get("hf_model_name", None)
            revision = model_metadata.get("hf_model_revision", None)
            historical_block = model_data.get("block", None)

            if not namespace or not name or not revision:
                return False
            
            if not isinstance(namespace, str) or not isinstance(name, str) or not isinstance(revision, str):
                return False 
            
            model_id = f"{namespace}/{revision}"
            
            Utils.subnet_logger(
                severity="TRACE",
                message=f"Model data has valid formatting: {model_data}",
                log_level=self.log_level
            )
                    
            # 2. Verify on-chain metadata
            model_metadata, model_block = asyncio.run(self.metadata_handler.directly_obtain_model_metadata_from_chain(hotkey=hotkey))

            if not model_metadata or not model_block or model_block == 0:
                return False 
            
            # Update model upload block if necessary
            if isinstance(historical_block, int) and historical_block < model_block:
                models_block = historical_block

            # Obtain competition id from model and miner data
            competition_id = self.metadata_handler.get_competition_id_from_competition_name(competition)
            
            # Determine miner metadata
            metadata_str = f"{namespace}:{name}:{revision}:{hotkey}:{competition_id}"

            if hashlib.sha256(metadata_str.encode()).hexdigest() != model_metadata:
                Utils.subnet_logger(
                    severity="INFO",
                    message=f"Model: {namespace}/{name} metadata could not be validated with on-chain metadata. Exiting model evaluation.",
                    log_level=self.log_level
                )
                return False
            
            # 3. Verify upload block is before seed determination
            if model_block >= self.seed_reference_block:
                Utils.subnet_logger(
                    severity="INFO",
                    message=f"Model: {namespace}/{name} was submitted on block: {model_block} which is greater than the seed reference block: {self.seed_reference_block}. Exiting model evaluation.",
                    log_level=self.log_level
                )
                return False
            else:
                Utils.subnet_logger(
                    severity="TRACE",
                    message=f"Model: {namespace}/{name} was submitted on block: {model_block} which is smaller than the seed reference block: {self.seed_reference_block}.",
                    log_level=self.log_level
                )

            # Check to make sure that namespace, name and revision are unique among submitted models and if not, that it was submitted first
            for model_dict in self.miner_models[competition]:
                if (
                    model_dict['hf_model_namespace'] == namespace
                ) and (
                    model_dict['hf_model_name'] == name
                ) and (
                    model_dict['hf_model_revision'] == revision
                ) and (
                    model_dict['block'] < model_block
                ):
                    return False
                
            # 4. Download repository and verify content
            model_dir = os.path.join(self.model_base_path, hotkey)
            self.prepare_directory(model_dir)

            # Download model to path and obtain model hash
            model_hash, _ = Models.get_model_content_hash(
                model_id=model_id,
                revision=revision,
                local_dir=model_dir,
                log_level=self.log_level
            )

            if not model_hash or model_hash in self.forbidden_model_hashes:
                Utils.subnet_logger(
                    severity="DEBUG",
                    message=f"Model hash for model: {model_id} with revision: {revision} could not be calculated or is invalid.",
                    log_level=self.log_level
                )
                self._reset_dir(directory=model_dir)
                return False 
            
            if not Models.verify_directory_files(directory=model_dir):
                Utils.subnet_logger(
                    severity="DEBUG",
                    message=f"Model: {model_id} with revision: {revision} contains a forbidden file.",
                    log_level=self.log_level
                )
                self._reset_dir(directory=model_dir)
                return False

            # Make sure model hash is unique 
            if model_hash in [model_data['model_hash'] for model_data in self.miner_models]:
                
                # Find block that metadata was uploaded to chain for all models with identical directory hash
                model_blocks_with_same_hash = []
                for model_data in self.miner_models:
                    if model_data['model_hash'] == self.model_hash:
                        model_blocks_with_same_hash.append(model_data['block'])
                
                # Append current model block for comparison
                model_blocks_with_same_hash.append(model_block)
                
                # If it's not unique, don't return False only if this model is the earliest one uploaded to chain
                if min(model_blocks_with_same_hash) != model_block:
                    Utils.subnet_logger(
                        severity="INFO",
                        message=f"Current model: {self.hf_model_id} has identical hash with another model and was not uploaded first. Exiting model evaluation.",
                        log_level=self.log_level
                    )   
                    self._reset_dir(directory=model_dir)
                    return False 

        except Exception as e:

            Utils.subnet_logger(
                severity="TRACE",
                message=f"Error evaluating model: {model_data}: {e}", 
                log_level=self.log_level
            )

            return False

    async def build_images_async(self):

        return await Utils.build_containers_async(
            model_base_path=self.model_base_path,
            eval_cache=self.eval_cache,
            hotkeys=self.hotkeys,
            log_level=self.log_level,
        )
    
    def build_images(self):

        hotkey_list, outcomes = asyncio.run(self.build_images_async())

        hk_len = len(hotkey_list)
        outcomes_len = len(outcomes)
        if len(hotkey_list) != len(outcomes):
            Utils.subnet_logger(
                severity="ERROR",
                message=f"Mismatch in hotkey_list length: {hk_len} and image building outcomes length: {outcomes_len} for asynchronous image building.",
                log_level=self.log_level
            )
            return None
        
        successful_hotkeys = []

        for hk, outcome in zip(hotkey_list, outcomes):
            if outcome:
                successful_hotkeys.append(hk)

        return successful_hotkeys