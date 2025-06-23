import bittensor as bt
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
                
                    if self.validate_model(model_data=model_data):

                        self.eval_cache[competition].append(model_data)

                        counter += 1
                    
                else:

                    remainder_cache[competition].append(model_data)

        return remainder_cache

    def validate_model(self, model_data):




