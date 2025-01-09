import bittensor as bt 
import pydantic 

class Denoising_16kHz_Protocol(bt.Synapse):
    """
    This class is used for miners to report to validators 
    their model for the 16kHz denoising task competition.
    """
    data: dict | None = pydantic.Field(
        default=None,
        description = "HuggingFace model identfication",
    )
    
    subnet_version: int = pydantic.Field(
        ...,
        description="Subnet version provides information about the subnet version the Synapse creator is running at",
        allow_mutation=False,
    )

    def deserialize(self) -> bt.Synapse:
        """Deserialize the instance of the protocol"""
        return self
    
class Dereverberation_16kHz_Protocol(bt.Synapse):
    """
    This class is used for miners to report to validators 
    their model for the 16kHz denoising task competition.
    """
    data: dict | None = pydantic.Field(
        default=None,
        description = "HuggingFace model identfication",
    )
    
    subnet_version: int = pydantic.Field(
        ...,
        description="Subnet version provides information about the subnet version the Synapse creator is running at",
        allow_mutation=False,
    )

    def deserialize(self) -> bt.Synapse:
        """Deserialize the instance of the protocol"""
        return self