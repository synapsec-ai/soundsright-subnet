"""
This miner script executes the main loop for the miner and keeps the
miner active in the Bittensor network.
"""

from argparse import ArgumentParser
import soundsright.core as SoundsRightCore

# This is the main function, which runs the miner.
if __name__ == "__main__":
    # Parse command line arguments
    parser = ArgumentParser()
    
    parser.add_argument(
        "--netuid", 
        type=int,
        default=105, 
        help="The chain subnet uid"
    )
    
    parser.add_argument(
        "--logging.logging_dir",
        type=str,
        default="/var/log/bittensor",
        help="Provide the log directory",
    )

    parser.add_argument(
        "--validator_min_stake",
        type=float,
        default=1000.0,
        help="Determine the minimum stake the validator should have to accept requests",
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
    
    parser.add_argument(
        "--axon.port",
        type=int,
        default=6001,
        help="Axon port, default is 6001. If you want to alter this value you will also need to adjust the exposed ports in the docker-compose.yml file."
    )

    # Create a miner based on the Class definitions
    subnet_miner = SoundsRightCore.SubnetMiner(parser=parser)
    subnet_miner.run()