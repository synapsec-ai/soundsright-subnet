import os 
import yaml
import glob
import shutil
from typing import List
import subprocess
from git import Repo
from huggingface_hub import snapshot_download

import soundsright.base.utils as Utils


class SGMSEHandler:
    
    def __init__(self, task: str, sample_rate: int, task_path: str, sgmse_path: str, sgmse_output_path: str, log_level: str, cuda_directory: str) -> None:
        
        self.hf_model_url = "https://huggingface.co/synapsecai/SoundsRightModelTemplate"
        self.task = task
        self.sample_rate = sample_rate
        self.competition = f"{task}_{sample_rate}HZ"
        self.task_path = task_path
        self.sgmse_path = sgmse_path 
        self.sgmse_output_path = sgmse_output_path
        self.cuda_directory = cuda_directory
        self.log_level = log_level
        
    def download_model_container(self) -> bool:
        try:
            snapshot_download(repo_id="synapsecai/SoundsRightModelTemplate", local_dir=self.sgmse_path, revision=self.competition)
            return True
        except Exception as e:
            Utils.subnet_logger(
                severity="ERROR",
                message=f"Could not download SGMSE because: {e}",
                log_level=self.log_level
            )
            return False
    
    def _reset_dir(self, directory):
        """Removes all files and sub-directories in an inputted directory

        Args:
            directory (_type_): _description_
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
        Removes files and sub-directories in ModelEvaluationHandler.sgmse_path and 
        ModelEvaluationHandler.sgmse_output_path to make sure all is clear for 
        next model to be benchmarked.
        """
        self._reset_dir(directory=self.sgmse_path)
        self._reset_dir(directory=self.sgmse_output_path)
    
    def initialize_and_run_model(self):
        """Initializes model and runs the container to enhance audio

        Returns:
            bool: True if operations were successful, False otherwise
        """ 
        # Delete everything before starting container
        Utils.delete_container(log_level=self.log_level)
        
        # Start container
        if not Utils.start_container(directory=self.sgmse_path, log_level=self.log_level, cuda_directory=self.cuda_directory):
            Utils.subnet_logger(
                severity="ERROR",
                message="SGMSE+ container could not be started. Please contact subnet owners if issue persists.",
                log_level=self.log_level
            )
            Utils.delete_container(log_level=self.log_level)
            return False
        
        Utils.subnet_logger(
            severity="TRACE",
            message="SGMSE+ Container started.",
            log_level=self.log_level,
        )
        
        if not Utils.check_container_status(port=6500, log_level=self.log_level):
            Utils.subnet_logger(
                severity="ERROR",
                message=f"Could not establish connection with SGMSE+ API. Please contact subnet owners if issue persists.",
                log_level=self.log_level
            )
            Utils.delete_container(log_level=self.log_level)
            return False
        
        Utils.subnet_logger(
            severity="TRACE",
            message="Connection established with SGMSE+ API and status was verified. Commencing model preparation.",
            log_level=self.log_level
        )
        
        if not Utils.prepare(port=6500, log_level=self.log_level):
            Utils.subnet_logger(
                severity="ERROR",
                message=f"Could not prepare the SGMSE+ model. Please contact subnet owners if issue persists.",
                log_level=self.log_level
            )
            Utils.delete_container(log_level=self.log_level)
            return False
        
        Utils.subnet_logger(
            severity="TRACE",
            message="SGMSE+ model preparation successful. Commencing transfer of noisy files.",
            log_level=self.log_level
        )
        
        if not Utils.upload_audio(port=6500, noisy_dir=self.task_path, log_level=self.log_level):
            
            Utils.subnet_logger(
                severity="ERROR",
                message="Noisy files could not be uploaded to SGMSE+ model container. Ending benchmarking of SGMSE+ model. Please contact subnet owners if issue persists.",
                log_level=self.log_level
            )
            
            Utils.delete_container(log_level=self.log_level)
            return False
            
        Utils.subnet_logger(
            severity="TRACE",
            message="Noisy files were transferred to SGMSE+ model container. Commencing enhancement.",
            log_level=self.log_level,
        )    
            
        if not Utils.enhance_audio(port=6500, log_level=self.log_level):
            
            Utils.subnet_logger(
                severity="ERROR",
                message="Noisy files could not be enhanced by SGMSE+ model. Ending benchmarking of model. Please contact subnet owners if issue persists.",
                log_level=self.log_level
            )
            
            Utils.delete_container(port=6500, log_level=self.log_level)
            return False
            
        Utils.subnet_logger(
            severity="TRACE",
            message="Enhancement complete. Downloading enhanced files from SGMSE+ model container.",
            log_level=self.log_level,
        )    
            
        if not Utils.download_enhanced(port=6500, enhanced_dir=self.sgmse_output_path, log_level=self.log_level):
            
            Utils.subnet_logger(
                severity="ERROR",
                message="Enhanced files could not be downloaded. Ending benchmarking of SGMSE+ model. Please contact subnet owners if issue persists.",
                log_level=self.log_level
            )
            
            Utils.delete_container(log_level=self.log_level)
            return False
        
        Utils.subnet_logger(
            severity="TRACE",
            message="Downloaded and unzipped files from SGMSE+ model API. Validating that all noisy files have been enhanced.",
            log_level=self.log_level,
        )
        
        if not self.validate_all_noisy_files_are_enhanced():
            
            Utils.subnet_logger(
                severity="ERROR",
                message="Mismatch detected between noisy and enhanced files. Ending benchmarking of SGMSE+ model. Please contact subnet owners if issue persists.",
                log_level=self.log_level
            )
            
            Utils.delete_container(log_level=self.log_level)
            return False 
        
        Utils.subnet_logger(
            severity="TRACE",
            message=f"Validation successful. Deleting SGMSE+ container.",
            log_level=self.log_level
        )
        
        Utils.delete_container(log_level=self.log_level)
        
        return True
        
    def validate_all_noisy_files_are_enhanced(self):
        noisy_files = sorted([os.path.basename(f) for f in glob.glob(os.path.join(self.task_path, '*.wav'))])
        enhanced_files = sorted([os.path.basename(f) for f in glob.glob(os.path.join(self.sgmse_output_path, '*.wav'))])
        return noisy_files == enhanced_files
        
    def download_start_and_enhance(self) -> bool:
        # Initial cleaning of model dirs
        self.reset_model_dirs()
        
        # Download model
        if not self.download_model_container():
            Utils.subnet_logger(
                severity="ERROR",
                message=f"Could not download SGMSE container (branch: {self.competition}. Please contact subnet owners if issue persists.)",
                log_level=self.log_level
            )
            self.reset_model_dirs()
            return False
        
        # Replace CUDA_HOME in Dockerfile 
        if not Utils.update_dockerfile_cuda_home(directory=self.sgmse_path, cuda_directory=self.cuda_directory, log_level=self.log_level):
            Utils.subnet_logger(
                severity="TRACE",
                message="Dockerfile CUDA_HOME could not be updated successfully.",
                log_level=self.log_level
            )
            self.reset_model_dirs()
            return False
        
        # Initialize and run model
        if not self.initialize_and_run_model():
            Utils.subnet_logger(
                severity="ERROR",
                message=f"Could not enhance files using SGMSE container (branch: {self.competition}. Please contact subnet owners if issue persists.)",
                log_level=self.log_level
            )
            self.reset_model_dirs()
            return False
        
        return True