"""
Main script for running SoundsRight validator
"""
# Import standard modules
from argparse import ArgumentParser
import os
from dotenv import load_dotenv 
load_dotenv()

# Import subnet modules
import soundsright.core as SoundsRightCore

# The main function parses the configuration and runs the validator.
if __name__ == "__main__":
    # Parse command line arguments
    parser = ArgumentParser()
    
    parser.add_argument(
        "--netuid", 
        type=int, 
        default=105, 
        help="The chain subnet uid."
    )

    parser.add_argument(
        "--cuda_directory",
        type=str,
        help="Path to CUDA directory.",
        default="/usr/local/cuda-12.6"
    )

    parser.add_argument(
        "--load_state",
        type=str,
        default="True",
        help="WARNING: Setting this value to False clears the old state.",
    )

    parser.add_argument(
        "--debug_mode",
        action="store_true",
        default=False,
        help="Running the validator in debug mode ignores selected validity checks. Not to be used in production.",
    )
    
    parser.add_argument(
        "--skip_sgmse",
        action="store_true",
        default=False,
        help="If passed, enables skipping of SGMSE+ benchmarking. Not to be used in production.",
    )
    
    parser.add_argument(
        "--dataset_size",
        default=250,
        type=int,
        help="Size of evaluation dataset."
    )

    parser.add_argument(
        "--log_level",
        type=str,
        default="INFO",
        choices=["INFO", "INFOX", "DEBUG", "DEBUGX", "TRACE", "TRACEX"],
        help="Determine the logging level used by the subnet modules",
    )

    parser.add_argument(
        "--healthcheck_host",
        type=str,
        default="0.0.0.0",
        help="Set the healthcheck API host. Defaults to 0.0.0.0 to expose it outside of the container.",
    )

    parser.add_argument(
        "--healthcheck_port",
        type=int,
        default=6000,
        help="Determine the port used by the healthcheck API.",
    )
    
    # Create a validator based on the Class definitions and initialize it
    subnet_validator = SoundsRightCore.SubnetValidator(parser=parser)
    subnet_validator.run()