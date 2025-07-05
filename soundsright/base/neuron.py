"""
Module for SoundsRight subnet neurons.

Neurons are the backbone of the subnet and are providing the subnet
users tools to interact with the subnet and participate in the
value-creation chain. There are two primary neuron classes: validator and miner.
"""

from argparse import ArgumentParser
import os
from datetime import datetime, timezone
import bittensor as bt
import numpy as np
import pickle

# Import custom modules
import soundsright.base.utils as Utils
import soundsright.base.data as Data

def convert_data(data):
    if isinstance(data, dict):
        return {key: convert_data(value) for key, value in data.items()}
    elif isinstance(data, list):
        return [convert_data(item) for item in data]
    elif isinstance(data, np.ndarray):
        return data.item() if data.size == 1 else data.tolist()
    elif isinstance(data, np.float32):
        return float(data.item()) if data.size == 1 else data.tolist()
    else:
        return data

class BaseNeuron:
    """Base neuron class for the SoundsRight Subnet. 
    
    This class handles base operations for both the miner and validator.

    Attributes:
        parser:
            Instance of ArgumentParser with the arguments given as
            command-line arguments in the execution script
        profile:
            Instance of str depicting the profile for the neuron
    """

    def __init__(self, parser: ArgumentParser, profile: str) -> None:
        self.parser = parser
        self.path_hotkey = None
        self.profile = profile
        self.step = 0
        self.last_updated_block = 0
        self.subnet_version = Utils.config["module_version"]
        self.score_version = Utils.config["score_version"]
        self.base_path = os.path.join(os.path.expanduser('~'), ".SoundsRight")
        self.cache_path = None
        self.log_path = None
        self.tts_path = None # Where clean TTS datasets are stored
        self.noise_data_path = None # Where the noise dataset is stored
        self.rir_data_path = None # Where the RIR dataset is stored
        self.reverb_path = None # Where the TTS with reverb added is stored
        self.noise_path = None # Where the TTS with noise added is stored
        self.model_output_path = None # Where the model's outputs are stored
        self.model_path = None # Where the model is stored
        self.sgmse_path = None # Where the SGMSE+ model and its checkpoints will be stored
        self.sgmse_output_path = None # Where the SGMSE+ model outputs will be stored
        self.healthcheck_api = None
        self.log_level = "INFO"
        self.start_date = datetime(2025, 6, 22, 9, 0, tzinfo=timezone.utc) # Reference for when to start competitions (June 19, 2025 @ 9:00 AM GMT)
        self.period_days = 2 # Competition length
        self.wc_prevention_protcool = False # Switch to toggle whether or not to use the WC Prevention Protocol

    def config(self, bt_classes: list) -> bt.config:
        """Applies neuron configuration.

        This function attaches the configuration parameters to the
        necessary bittensor classes and initializes the logging for the
        neuron.

        Args:
            bt_classes:
                A list of Bittensor classes the apply the configuration
                to

        Returns:
            config:
                An instance of Bittensor config class containing the
                neuron configuration

        Raises:
            AttributeError:
                An error occurred during the configuration process
            OSError:
                Unable to create a log path.

        """
        try:
            for bt_class in bt_classes:
                bt_class.add_args(self.parser)
        except AttributeError as e:
            self.neuron_logger(
                severity="ERROR",
                message=f"Unable to attach ArgumentParsers to Bittensor classes: {e}"
            )
            raise AttributeError from e

        config = bt.config(self.parser)

        # Construct log path
        self.path_hotkey = config.wallet.hotkey
        self.log_path = os.path.join(self.base_path, "logs", config.wallet.name, config.wallet.hotkey, str(config.netuid), self.profile)

        # Construct cache path
        self.cache_path = os.path.join(self.base_path, "cache", config.wallet.name, config.wallet.hotkey, str(config.netuid), self.profile, self.score_version)

        # Construct data paths 
        self.noise_path = os.path.join(self.base_path, "data", "noise")
        self.reverb_path = os.path.join(self.base_path, "data", "reverb")
        self.rir_data_path = os.path.join(self.base_path, "data", "rir_data")
        self.noise_data_path = os.path.join(self.base_path, "data", "noise_data")
        self.tts_path = os.path.join(self.base_path, "data", "tts")
        self.model_output_path = os.path.join(self.base_path, "models", "model_output")
        self.model_path = os.path.join(self.base_path, "models", "model")
        self.sgmse_path = os.path.join(self.base_path, "models", "sgmse")
        self.sgmse_output_path = os.path.join(self.base_path, "models", "sgmse_output")
        self.sgmse_ckpt_files = {
            "DENOISING_16000HZ":"train_wsj0_2cta4cov_epoch=159.ckpt",
            "DEREVERBERATION_16000HZ":"epoch=326-step=408750.ckpt",
        }

        # Create the OS paths if they do not exists
        try:
            for os_path in [self.log_path, self.cache_path, self.noise_path, self.reverb_path, self.rir_data_path, self.noise_data_path, self.tts_path, self.model_output_path, self.model_path, self.sgmse_path, self.sgmse_output_path]:
                full_path = os.path.expanduser(os_path)
                if not os.path.exists(full_path):
                    os.makedirs(full_path, exist_ok=True)

                if os_path == self.log_path:
                    config.full_path = os.path.expanduser(os_path)
        except OSError as e:
            self.neuron_logger(
                severity="ERROR",
                message=f"Unable to create log path: {e}"
            )
            raise OSError from e

        return config

    def neuron_logger(self, severity: str, message: str):
        """This method is a wrapper for the bt.logging function to add extra
        functionality around the native logging capabilities"""

        Utils.subnet_logger(severity=severity, message=message, log_level=self.log_level)

        # Append extra information to to the logs if healthcheck API is enabled
        if self.healthcheck_api and severity.upper() in ("SUCCESS", "ERROR", "WARNING"):

            event_severity = severity.lower()

            # Metric
            self.healthcheck_api.append_metric(
                metric_name=f"log_entries.{event_severity}", value=1
            )

            # Store event
            self.healthcheck_api.add_event(
                event_name=f"{event_severity}", event_data=message
            )