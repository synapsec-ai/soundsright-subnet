import os
import time
import glob
import asyncio
import shutil
import hashlib
import bittensor as bt
from typing import List

# Import custom modules
import soundsright.base.benchmarking as Benchmarking
import soundsright.base.models as Models
import soundsright.base.utils as Utils

class ModelEvaluationHandler:
    
    def __init__(
        self, 
        tts_base_path: str, 
        noise_base_path: str,
        reverb_base_path: str,
        model_output_path: str,
        model_path: str,
        sample_rate: int,
        task: str,
        hf_model_namespace: str,
        hf_model_name: str,
        hf_model_revision: str,
        log_level: str,
        subtensor: bt.subtensor,
        subnet_netuid: int,
        miner_hotkey: str,
        miner_models: List[dict],
        cuda_directory: str,
        historical_block: int | None,
        seed_reference_block: int | float,
    ):
        """Initializes ModelEvaluationHandler

        Args:
            :param tts_base_path: (str): Base directory for TTS dataset
            :param noise_base_path: (str): Base directory for denoising dataset
            :param reverb_base_path: (str): Base directory for dereverberation dataset
            :param model_output_path: (str): Directory for model outputs to be temporarily stored for benchmarking
            :param model_path: (str): Directory for model to be temporarily stored for benchmarking
            :param sample_rate: (int): Sample rate 
            :param task: (str): DENOISING or DEREVERBERATION
            :param hf_model_namespace: (str): Namespace from synapse
            :param hf_model_name: (str): Name from synapse
            :param hf_model_revision: (str): Revision from synapse
            :param log_level: (str): Log level from .env
            :param subtensor: (bt.subtensor): Subtensor from validator
            :param subnet_netuid: (int): Netuid from .env
            :param miner_hotkey: (str): ss58 address
            :param miner_models: (List[dict]): Most recent benchmarked model/empty response for each miner for the competition
        """
        # Paths 
        self.tts_path = os.path.join(tts_base_path, str(sample_rate))
        self.noise_path = os.path.join(noise_base_path, str(sample_rate))
        self.reverb_path = os.path.join(reverb_base_path, str(sample_rate))
        self.model_output_path = model_output_path
        self.model_path = model_path 
        self.cuda_directory = cuda_directory
        # Competition
        self.sample_rate = sample_rate
        self.task = task 
        self.task_path = self.noise_path if self.task == "DENOISING" else self.reverb_path
        self.competition = f"{task}_{sample_rate}HZ"
        # Model
        self.hf_model_namespace = hf_model_namespace
        self.hf_model_name = hf_model_name
        self.hf_model_id = f"{hf_model_namespace}/{hf_model_name}"
        self.hf_model_revision = hf_model_revision
        self.historical_block = historical_block # This is used for logging the historical value of the block 
        self.seed_reference_block = seed_reference_block
        self.hf_model_block = None
        self.model_hash = ''
        self.forbidden_model_hashes = [
            "ENZIdw0H8Vbb79lXDQKBqqReXIj2ycgOX1Ob0QoexAU=",
            "Mbx0++bk5q6n+rdVlUblElnturj/zRobTk61WFVHmgg=",
        ]
        # Misc
        self.log_level = log_level
        self.miner_hotkey = miner_hotkey
        self.miner_models = miner_models
        self.metadata_handler = Models.ModelMetadataHandler(
            subtensor=subtensor, 
            subnet_netuid=subnet_netuid,
            log_level=self.log_level
        )

    def obtain_model_metadata(self):
        """
        Validates that the model provided by the miner matches the metadata uploaded to the chain.
        
        Updates ModelEvaluationHandler.model_metadata and ModelEvaluationHandler.hf_model_block with
        on-chain data.
    
        Returns:
            bool: True if model metadata could be obtained, False otherwise
        """
        try:
            outcome = asyncio.run(self.metadata_handler.obtain_model_metadata_from_chain(
                hotkey=self.miner_hotkey,
            ))
            
            if not outcome or not self.metadata_handler.metadata or not self.metadata_handler.metadata_block or self.metadata_handler.metadata_block == 0: 
                    
                Utils.subnet_logger(
                    severity="ERROR",
                    message=f"Could not obtain model metadata from chain for hotkey: {self.miner_hotkey}",
                    log_level=self.log_level
                )
                
                return False
                
            else: 
                
                self.model_metadata = self.metadata_handler.metadata
                self.hf_model_block = self.metadata_handler.metadata_block

                if not isinstance(self.hf_model_block, int):
                    return False

                if self.historical_block and isinstance(self.historical_block, int) and self.historical_block < self.hf_model_block:
                    
                    Utils.subnet_logger(
                        severity="TRACE",
                        message=f"Detected that miner has uploaded metadata for the same model: {self.hf_model_id} more than once. Old block: {self.historical_block}. New block: {self.hf_model_block}. Reverting back to original block of metadata upload.",
                        log_level=self.log_level,
                    )
                    self.hf_model_block = self.historical_block
                
                Utils.subnet_logger(
                    severity="DEBUG",
                    message=f"Recieved model metadata from chain: {self.model_metadata} on block: {self.hf_model_block} for hotkey: {self.miner_hotkey}",
                    log_level=self.log_level
                )
                
                return True       
            
        except Exception as e:   
            
            Utils.subnet_logger(
                severity="ERROR",
                message=f"Could not obtain model metadata from chain for hotkey: {self.miner_hotkey} because: {e}",
                log_level=self.log_level
            )
                
            return False
            
    def validate_model_metadata(self):
        """Validates that the model metadata is for a model belonging to the miner with the following steps:
        
        1. Re-create model metadata string and confirm that its hash matches metadata uploaded to chain
        2. Make sure that model name is unique among models submitted. If it is not, it checks the block that 
        metadata was uploaded to the chain. If the metadata was uploaded first, we assume that this is the 
        miner that originally uploaded the model to Huggingface
        3. Download model and calculate hash of model directory
        4. Make sure that model hash is unique among models submitted. If it is not, it checks the block that 
        metadata was uploaded to the chain. If the metadata was uploaded first, we assume that this is the 
        miner that originally uploaded the model to Huggingface

        Returns:
            bool: True if model metadata checks out, False if otherwise
        """
        # Obtain competition id from model and miner data
        competition_id = self.metadata_handler.get_competition_id_from_competition_name(self.competition)
        
        # Determine miner metadata
        metadata_str = f"{self.hf_model_namespace}:{self.hf_model_name}:{self.hf_model_revision}:{self.miner_hotkey}:{competition_id}"
        
        # Hash it and compare to hash uploaded to chain 
        if hashlib.sha256(metadata_str.encode()).hexdigest() != self.model_metadata:
            Utils.subnet_logger(
                severity="INFO",
                message=f"Model: {self.hf_model_id} metadata could not be validated with on-chain metadata. Exiting model evaluation.",
                log_level=self.log_level
            )
            return False
        
        # Check to make sure that the submitted block is not larger than the seed reference block
        if self.hf_model_block >= self.seed_reference_block:
            return False
        
        # Check to make sure that namespace, name and revision are unique among submitted models and if not, that it was submitted first
        for model_dict in self.miner_models:
            if (
                model_dict['hf_model_namespace'] == self.hf_model_namespace
            ) and (
                model_dict['hf_model_name'] == self.hf_model_name
            ) and (
                model_dict['hf_model_revision'] == self.hf_model_revision
            ) and (
                model_dict['block'] < self.hf_model_block
            ):
                return False
        
        # Download model to path and obtain model hash
        self.model_hash, _ = Models.get_model_content_hash(
            model_id=self.hf_model_id,
            revision=self.hf_model_revision,
            local_dir=self.model_path,
            log_level=self.log_level
        )

        if not self.model_hash or self.model_hash in self.forbidden_model_hashes:
            Utils.subnet_logger(
                severity="DEBUG",
                message=f"Model hash for model: {self.hf_model_id} with revision: {self.hf_model_revision} could not be calculated or is invalid.",
                log_level=self.log_level
            )
            return False 
        
        if not Models.verify_directory_files(directory=self.model_path):
            Utils.subnet_logger(
                severity="DEBUG",
                message=f"Model: {self.hf_model_id} with revision: {self.hf_model_revision} contains a forbidden file.",
                log_level=self.log_level
            )
            return False

        # Make sure model hash is unique 
        if self.model_hash in [model_data['model_hash'] for model_data in self.miner_models]:
            
            # Find block that metadata was uploaded to chain for all models with identical directory hash
            model_blocks_with_same_hash = []
            for model_data in self.miner_models:
                if model_data['model_hash'] == self.model_hash:
                    model_blocks_with_same_hash.append(model_data['block'])
            
            # Append current model block for comparison
            model_blocks_with_same_hash.append(self.hf_model_block)
            
            # If it's not unique, don't return False only if this model is the earliest one uploaded to chain
            if min(model_blocks_with_same_hash) != self.hf_model_block:
                Utils.subnet_logger(
                    severity="INFO",
                    message=f"Current model: {self.hf_model_id} has identical hash with another model and was not uploaded first. Exiting model evaluation.",
                    log_level=self.log_level
                )   
                return False 
        
        return True 
    
    def validate_all_noisy_files_are_enhanced(self):
        noisy_files = sorted([os.path.basename(f) for f in glob.glob(os.path.join(self.task_path, '*.wav'))])
        enhanced_files = sorted([os.path.basename(f) for f in glob.glob(os.path.join(self.model_output_path, '*.wav'))])
        return noisy_files == enhanced_files
    
    def initialize_and_run_model(self):
        """_summary_

        Returns:
            bool: 
        """

        if not Utils.update_dockerfile_cuda_home(directory=self.model_path, cuda_directory=self.cuda_directory, log_level=self.log_level):
            Utils.subnet_logger(
                severity="TRACE",
                message="Dockerfile CUDA_HOME could not be updated successfully.",
                log_level=self.log_level
            )
            
            return False
        
        Utils.subnet_logger(
            severity="TRACE",
            message="Dockerfile CUDA_HOME updated successfully.",
            log_level=self.log_level
        )
        
        Utils.subnet_logger(
            severity="TRACE",
            message=f"Validating container configuration for model: {self.hf_model_namespace}/{self.hf_model_name}.",
            log_level=self.log_level
        )
        
        # Validate container 
        if not Utils.validate_container_config(self.model_path):
            
            Utils.subnet_logger(
                severity="TRACE",
                message=f"Validating container configuration for model failed: {self.hf_model_namespace}/{self.hf_model_name}.",
                log_level=self.log_level,
            )
            
            return False 
        
        Utils.subnet_logger(
            severity="TRACE",
            message=f"Validating container configuration for model succeeded: {self.hf_model_namespace}/{self.hf_model_name}. Now starting container",
            log_level=self.log_level
        )
        
        # Delete any existing containers before starting new one
        Utils.delete_container(log_level=self.log_level)
        
        # Start container
        if not Utils.start_container(directory=self.model_path, log_level=self.log_level, cuda_directory=self.cuda_directory):
            Utils.subnet_logger(
                severity="TRACE",
                message="Container could not be started",
                log_level=self.log_level
            )
            Utils.delete_container(log_level=self.log_level)
            return False
        
        Utils.subnet_logger(
            severity="TRACE",
            message="Container started.",
            log_level=self.log_level
        )
        
        time.sleep(10)
        
        if not Utils.check_container_status(log_level=self.log_level):
            Utils.subnet_logger(
                severity="TRACE",
                message=f"Could not establish connection with API.",
                log_level=self.log_level
            )
            Utils.delete_container(log_level=self.log_level)
            return False
        
        Utils.subnet_logger(
            severity="TRACE",
            message="Connection established with API and status was verified. Commencing model preparation.",
            log_level=self.log_level
        )
        
        time.sleep(1)
        
        if not Utils.prepare(log_level=self.log_level):
            Utils.subnet_logger(
                severity="ERROR",
                message=f"Could not prepare the model.",
                log_level=self.log_level
            )
            Utils.delete_container(log_level=self.log_level)
            return False
        
        Utils.subnet_logger(
            severity="TRACE",
            message="Model preparation successful. Commencing transfer of noisy files.",
            log_level=self.log_level
        )
        
        time.sleep(10)
        
        if not Utils.upload_audio(noisy_dir=self.task_path, log_level=self.log_level):
            
            Utils.subnet_logger(
                severity="TRACE",
                message="Noisy files could not be uploaded to model container. Ending benchmarking of model.",
                log_level=self.log_level
            )
            
            Utils.delete_container(log_level=self.log_level)
            return False
            
        Utils.subnet_logger(
            severity="TRACE",
            message="Noisy files were transferred to model container. Commencing enhancement.",
            log_level=self.log_level
        )    
        
        time.sleep(5)
            
        if not Utils.enhance_audio(log_level=self.log_level):
            
            Utils.subnet_logger(
                severity="TRACE",
                message="Noisy files could not be enhanced by model. Ending benchmarking of model.",
                log_level=self.log_level
            )
            
            Utils.delete_container(log_level=self.log_level)
            return False
            
        Utils.subnet_logger(
            severity="TRACE",
            message="Enhancement complete. Downloading enhanced files from model container.",
            log_level=self.log_level
        )    
            
        time.sleep(5)
            
        if not Utils.download_enhanced(enhanced_dir=self.model_output_path, log_level=self.log_level):
            
            Utils.subnet_logger(
                severity="TRACE",
                message="Enhanced files could not be downloaded. Ending benchmarking of model.",
                log_level=self.log_level
            )
            
            Utils.delete_container(log_level=self.log_level)
            return False
        
        Utils.subnet_logger(
            severity="TRACE",
            message="Downloaded and unzipped files. Validating that all noisy files have been enhanced.",
            log_level=self.log_level
        )
        
        if not self.validate_all_noisy_files_are_enhanced():
            
            Utils.subnet_logger(
                severity="TRACE",
                message="Mismatch detected between noisy and enhanced files. Ending benchmarking of model.",
                log_level=self.log_level
            )
            
            Utils.delete_container(log_level=self.log_level)
            return False 
        
        Utils.subnet_logger(
            severity="TRACE",
            message=f"Validation successful. Deleting containers.",
            log_level=self.log_level
        )
        
        Utils.delete_container(log_level=self.log_level)
        
        return True
        
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
                
    def reset_model_dirs(self):
        """
        Removes files and sub-directories in ModelEvaluationHandler.model_path and 
        ModelEvaluationHandler.model_output_path to make sure all is clear for 
        next model to be benchmarked.
        """
        self._reset_dir(directory=self.model_path)
        self._reset_dir(directory=self.model_output_path)
        
    def download_run_and_evaluate(self):
        """
        Overarching function to verify, download, execute and evaluate model performance.
        
        Returns:
            :param metric_average: (float): Average value for the evaluation metric for the model
            :param confidence_interval: (List[float]) 95% CI for metric score 
            :param metric_values: (List[float]) List of all metric scores for the model evaluation
            :param metric: (str): Name of the metric being used, determined by competition.
            :param model_hash:
            :param hf_model_block:  
        """
        
        Utils.subnet_logger(
            severity="TRACE",
            message=f"Checking if model: {self.hf_model_id} from miner: {self.miner_hotkey} has metadata that can be obtained from chain.",
            log_level=self.log_level
        )
        
        # Attempt to obtain the model metadata stored on-chain
        if not self.obtain_model_metadata():
            # Remove all files from model-based directories (model files and model outcome files)
            self.reset_model_dirs()
            # Return zero for the output metric if the model could not be obtained
            return {}, self.model_hash, self.hf_model_block
        
        Utils.subnet_logger(
            severity="TRACE",
            message=f"Checking if model: {self.hf_model_id} from miner: {self.miner_hotkey} has metadata that can be validated and a unique hash for the model.",
            log_level=self.log_level
        )
        
        # Attempt to validate the model with the data stored on-chain. This step also downloads the model to self.model_path
        if not self.validate_model_metadata():
            # Remove all files from model-based directories (model files and model outcome files)
            self.reset_model_dirs()
            # Return zero for the output metric if the model metadata could not be validated
            return {}, self.model_hash, self.hf_model_block
        
        Utils.subnet_logger(
            severity="TRACE",
            message=f"Running model: {self.hf_model_id} from miner: {self.miner_hotkey} on validator benchmarking dataset.",
            log_level=self.log_level
        )
        
        # Initialize and run the model on the dataset 
        if not self.initialize_and_run_model(): 
            # Remove all files from model-based directories (model files and model outcome files)
            self.reset_model_dirs()
            # Return zero for the output metric if the model could not be initialized or run
            return {}, self.model_hash, self.hf_model_block
        
        Utils.subnet_logger(
            severity="TRACE",
            message=f"Calculating metrics for benchmarking dataset for model: {self.hf_model_id} from miner: {self.miner_hotkey}.",
            log_level=self.log_level
        )
        
        # Calculate metrics (metrics vary depending on sample rate)
        metrics_dict = Benchmarking.calculate_metrics_dict(
            clean_directory=self.tts_path,
            enhanced_directory=self.model_output_path,
            noisy_directory=self.task_path,
            sample_rate=self.sample_rate,
            log_level=self.log_level,
        )
        
        # Remove all files from model-based directories (model files and model outcome files)
        self.reset_model_dirs()
        
        return metrics_dict, self.model_hash, self.hf_model_block