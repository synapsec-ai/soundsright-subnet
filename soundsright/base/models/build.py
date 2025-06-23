import bittensor as bt
import asyncio
import hashlib
from math import floor

import soundsright.base.utils as Utils
import soundsright.base.models as Models

class ModelBuilder:
    """
    Class responsible for verifying models and building model images async
    """

    def __init__(
        self,
        model_cache: dict,
        cuda_directory: str,
        seed_reference_block: int | float,
        cpu_count: int,
        avg_model_size_gb: int,
        images_per_cpu: int,
        subtensor: bt.subtensor,
        subnet_netuid: int,
        hotkeys: list,
        miner_models: dict,
        log_level: str,
    ):

        # Machine
        self.cuda_directory = cuda_directory
        self.free_storage_gb = Utils.get_free_space_gb()
        self.avg_model_size_gb = avg_model_size_gb
        self.cpu_count = cpu_count
        self.images_per_cpu = images_per_cpu
        cpu_max_count = self.cpu_count * self.images_per_cpu
        storage_max_count = floor(self.free_storage_gb / self.avg_model_size_gb)
        self.max_image_count = min([cpu_max_count, storage_max_count])

        # Bittensor
        self.hotkeys = hotkeys
        self.seed_reference_block = seed_reference_block
        self.metadata_handler = Models.ModelMetadataHandler(
            subtensor=subtensor, 
            subnet_netuid=subnet_netuid,
            log_level=self.log_level
        )
    
        # Model
        self.forbidden_model_hashes = [
            "ENZIdw0H8Vbb79lXDQKBqqReXIj2ycgOX1Ob0QoexAU=",
            "Mbx0++bk5q6n+rdVlUblElnturj/zRobTk61WFVHmgg=",
        ]
        self.miner_models = miner_models
        self.model_cache = model_cache
        self.eval_cache = {}

        # Misc
        self.log_level = log_level

    def filter_model_cache(self):

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
            
            # Check to make sure that the submitted block is not larger than the seed reference block
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
                
            
            


            

            





        except Exception as e:

            Utils.subnet_logger(
                severity="TRACE",
                message=f"Error evaluating model: {model_data}: {e}", 
                log_level=self.log_level
            )

            return False





