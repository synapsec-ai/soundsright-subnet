import hashlib
import os
import base64
import shutil
import subprocess
from git import Repo

import soundsright.base.utils as Utils

def get_directory_content_hash(directory: str):
    """
    Computes a single hash of the combined contents of all files in a directory,
    excluding certain unnecessary files and directories.
    
    Args:
        :param directory: (str): Path to the directory.
    
    Returns:
        str: A base64-encoded hash representing the combined contents of all files.
    """
    hash_obj = hashlib.sha256()
    excluded_dirs = {'.git'}
    excluded_files = {'.lock', '.metadata'}

    # Traverse the directory in a consistent order
    for root, dirs, files in os.walk(directory):
        # Exclude specified directories
        dirs[:] = [d for d in dirs if d not in excluded_dirs]
        sorted_files = sorted(files)
        
        for file_name in sorted_files:  # Sort files to ensure consistent order
            if file_name in excluded_files:
                continue  # Skip excluded files
            
            file_path = os.path.join(root, file_name)
            
            # Update the hash with the relative file path to capture structure
            rel_path = os.path.relpath(file_path, directory).replace(os.sep, '/')
            hash_obj.update(rel_path.encode())
            
            try:
                # Read the entire content to confirm there’s no issue
                with open(file_path, "rb") as f:
                    file_contents = f.read()
                
                # If reading was successful, update the hash with file contents
                hash_obj.update(file_contents)

            except Exception as e:
                continue

    # Encode the final hash in base64 and return it
    return base64.b64encode(hash_obj.digest()).decode(), sorted_files

def get_model_content_hash(
    model_id: str, 
    revision: str, 
    local_dir: str, 
    log_level: str,
):
    """
    Downloads the model and computes the hash of its entire contents.

    Args:
        :param model_id: (str): The repository ID of the Hugging Face model (e.g., 'bert-base-uncased').
        :param revision: (str): The specific branch, tag, or commit hash (default is 'main').
        :param local_dir: (str): Local directory to download the model to.
        :param log_level: (str): One of: INFO, INFOX, DEBUG, DEBUGX, TRACE, TRACEX.

    Returns:
        str: The combined hash of the model's contents.
    """
    # Remove all directory contents if it doesn't exist
    try:
        shutil.rmtree(local_dir)
    except:
        Utils.subnet_logger(
            severity="TRACE",
            message=f"Model directory already deleted: {local_dir}",
            log_level=log_level
        )
    
    repo_url = f"https://huggingface.co/{model_id}"
    
    # Download the model files for the specified revision
    Repo.clone_from(repo_url, local_dir, branch=revision)

    # Compute the hash of the model's contents
    return get_directory_content_hash(directory=local_dir)