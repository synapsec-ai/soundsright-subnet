import bittensor as bt
import soundsright.base.utils as Utils

class ModelMetadataHandler:
    
    def __init__(
        self,
        subtensor: bt.subtensor,
        subnet_netuid: int,
        log_level: str,
        wallet: bt.wallet | None = None,
    ):
    
        self.subtensor = subtensor 
        self.subnet_netuid = subnet_netuid 
        self.wallet = wallet   
        self.log_level = log_level
        self.metadata = ''
        self.metadata_block = 0
        
    Utils.timeout_decorator(timeout=60)
    async def upload_model_metadata_to_chain(self, metadata: str):
        """_summary_

        Args:
            metadata (str): Hash of metadata string

        Returns:
            bool: True if metadata could be uploaded to chain, False otherwise
        """
        outcome = bt.core.extrinsics.serving.publish_metadata(
            self.subtensor,
            wallet=self.wallet,
            netuid=self.subnet_netuid,
            data_type=f"Raw{len(metadata)}",
            data=metadata.encode(),
            wait_for_inclusion=True,
            wait_for_finalization=True,
        )
        
        return outcome
    
    Utils.timeout_decorator(timeout=60)    
    async def obtain_model_metadata_from_chain(self, hotkey: str):
        """_summary_

        Args:
            hotkey (str): ss58_address of miner hotkey

        Returns:
            bool: True if model metadata could be obtained from chain, False otherwise
        """
        try:
            
            metadata = bt.core.extrinsics.serving.get_metadata(
                self=self.subtensor,
                netuid=self.subnet_netuid,
                hotkey=hotkey
            )
            
            commitment = metadata["info"]["fields"][0]
            hex_data = commitment[list(commitment.keys())[0]][2:]
            self.metadata = bytes.fromhex(hex_data).decode()
            
            self.metadata_block = metadata['block']
            
            return True
        
        except Exception as e:
            raise e
            return False
        
    def get_competition_id_from_competition_name(self, competition_name):
        """Obtains competition id from competition name for metadata purposes

        Args:
            :param competition_name: (str): Name of competition

        Returns:
            int | None: int if competition_name is valid, None otherwise
        """
        conversion_dict={
            "DENOISING_16000HZ":1,
            "DEREVERBERATION_16000HZ":2,
        }
        
        if competition_name in conversion_dict.keys():
            return conversion_dict[competition_name]
        
        return None
        
    def get_competition_name_from_competition_id(self, competition_id):
        """Get competition string from numerical id
        
        Args:
            :param competition_id: (int | str): id of competition as used in metadata string
            
        Returns:
            :param competition_name" (str): name of competition as used in dict keys in the rest of the repo
        """
        conversion_dict = {
            "1":"DENOISING_16000HZ",
            "2":"DEREVERBERATION_16000HZ",
        }
        
        if str(competition_id) in conversion_dict.keys():
            return conversion_dict[str(competition_id)]
        
        return None