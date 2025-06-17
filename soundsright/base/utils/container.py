import os
import yaml 
import subprocess 
import requests 
import zipfile
import time 
import glob
import sys
import re

import soundsright.base.utils as Utils

def check_dockerfile_for_root_user(dockerfile_path):
    """
    Checks if a Dockerfile configures the container to run as a root user,
    considering ARG definitions for the user ID.

    Args:
        directory (str): The directory to search for a Dockerfile.
        dockerfile_path (str): The specific path to the Dockerfile.

    Returns:
        bool: True if the Dockerfile configures the container to run as root, False otherwise.

    Raises:
        FileNotFoundError: If no Dockerfile is found at the specified path.
    """
    try:
        user_line_exists = False
        arg_definitions = {}
        env_definitions = {}
        
        with open(dockerfile_path, "r") as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip()
                
                # Parse ARG directives
                if line.startswith("ARG"):
                    parts = line.split("=")
                    if len(parts) == 2:
                        arg_name = parts[0].split()[1].strip()
                        arg_value = parts[1].strip()
                        arg_definitions[arg_name] = arg_value
                        
                if line.startswith("ENV"):
                    parts = line.split("=")
                    if len(parts) == 2:
                        env_name = parts[0].split()[1].strip()
                        env_value = parts[1].strip()
                        env_definitions[env_name] = env_value
                
                # Parse USER directive
                if line.startswith("USER"):
                    user_line_exists = True
                    user = line.split()[1]
                    
                    # Resolve ARG references in the USER directive
                    if user.startswith("$"):
                        user = arg_definitions.get(user[1:], None)
                        if user:
                            if user == "root" or str(user) == "0" or user.startswith("$"):
                                return True
                        user = env_definitions.get(user[1:], None)
                        if user:
                            if user == "root" or str(user) == "0" or user.startswith("$"):
                                return True
                    
                    # Check if the resolved user is root
                    if user == "root" or str(user) == "0":
                        return True
                        
        # If no USER directive is found, the default is root
        if not user_line_exists:
            return True

    except Exception as e:
        return True  # Default to True if an error occurs to err on the side of caution
    
    return False  # Returns False if no root configuration is detected

def check_dockerfile_for_sensitive_config(dockerfile_path):
    """
    Finds a Dockerfile in the specified directory or its subdirectories and checks
    if the `.bittensor` directory is mounted as a volume.

    Args:
        directory (str): The directory to search for a Dockerfile. Defaults to the current directory.

    Returns:
        bool: True if the `.bittensor` directory is mounted, False otherwise.

    Raises:
        FileNotFoundError: If no Dockerfile is found in the specified directory or its subdirectories.
    """
    sensitive_directories = [
        ".bittensor",
        "bittensor"
    ]

    try:
        with open(dockerfile_path, "r") as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip()
                if any(sensitive in line for sensitive in sensitive_directories):
                    return True
                    
    except Exception as e:
        return True

    # If no VOLUME directive mentions .bittensor, return False
    return False

def update_dockerfile_cuda_home(directory, cuda_directory, log_level):
    pattern = re.compile(r'^(ENV\s+CUDA_HOME=).*$', re.MULTILINE)
    from_pattern = re.compile(r'^(FROM\s+.+)$', re.MULTILINE)
    replacement_line = f'ENV CUDA_HOME={cuda_directory}'

    # Find dockerfile path
    dockerfile_path = None

    # Search for the Dockerfile in the directory and subdirectories
    for root, _, files in os.walk(directory):
        for file in files:
            if file == "Dockerfile":
                dockerfile_path = os.path.join(root, file)
                break
        if dockerfile_path:
            break

    if not dockerfile_path:
        return False 

    try:
        with open(dockerfile_path, 'r') as file:
            content = file.read()
            Utils.subnet_logger(
                severity="TRACE",
                message=f"Old Dockerfile: {content}",
                log_level=log_level
            )

        if pattern.search(content):
            content = pattern.sub(replacement_line, content)
            Utils.subnet_logger(
                severity="TRACE",
                message=f"Updated existing CUDA_HOME line to: {replacement_line}",
                log_level=log_level
            )
        else:
            # Insert CUDA_HOME after first FROM
            match = from_pattern.search(content)
            if match:
                insert_pos = match.end()
                # Insert with newline handling
                content = content[:insert_pos] + f'\n{replacement_line}' + content[insert_pos:]
                Utils.subnet_logger(
                    severity="TRACE",
                    message=f"Inserted CUDA_HOME line after FROM: {replacement_line}",
                    log_level=log_level
                )
            else:
                Utils.subnet_logger(
                    severity="ERROR",
                    message="No FROM line found. Cannot insert CUDA_HOME.",
                    log_level=log_level
                )
                return False

        with open(dockerfile_path, 'w') as file:
            Utils.subnet_logger(
                severity="TRACE",
                message=f"New Dockerfile: {content}",
                log_level=log_level
            )
            file.write(content)
            Utils.subnet_logger(
                severity="TRACE",
                message=f"Successfully updated CUDA_HOME in {dockerfile_path}",
                log_level=log_level
            )
            return True

    except FileNotFoundError:
        Utils.subnet_logger(
            severity="ERROR",
            message=f"Error: File '{dockerfile_path}' not found.",
            log_level=log_level
        )
        return False
    except Exception as e:
        Utils.subnet_logger(
            severity="ERROR",
            message=f"An error occurred while updating the Dockerfile CUDA_HOME: {e}",
            log_level=log_level
        )
        print(f"An error occurred: {e}")
        return False

def validate_container_config(directory) -> bool:
    """
    Makes sure that both the Dockerfile and docker-compose.yml files
    do not run the container as root, 
    Args:
        directory (str): Repository of docker container
        
    Returns: 
    """
    # Find dockerfile path
    dockerfile_path = None

    # Search for the Dockerfile in the directory and subdirectories
    for root, _, files in os.walk(directory):
        for file in files:
            if file == "Dockerfile":
                dockerfile_path = os.path.join(root, file)
                break
        if dockerfile_path:
            break

    if not dockerfile_path:
        return False 
    
    if check_dockerfile_for_sensitive_config(dockerfile_path):
        return False 
        
    return True    
        
def start_container(directory, log_level, cuda_directory) -> bool:
    """Runs the container with docker compose

    Args:
        directory (str): Directory containing the container
    """
    dockerfile_path = None

    # Search for docker-compose.yml in the directory and its subdirectories
    for root, _, files in os.walk(directory):
        for file in files:
            if file == "Dockerfile":
                dockerfile_path = os.path.join(root, file)
                break
        if dockerfile_path:
            break

    if not dockerfile_path:
        return False
    
    if not os.path.isfile(dockerfile_path):
        Utils.subnet_logger(
            severity="ERROR",
            message=f"No `Dockerfile` file found in the specified directory: {directory}",
            log_level=log_level,
        )
        return False

    try:
        result0 = subprocess.run(
            ["podman", "build", "-t", "modelapi", "--file", dockerfile_path], 
            check=True,
            timeout=600,
        )
        if result0.returncode != 0:
            return False
        cuda_insert = f"{cuda_directory}:{cuda_directory}"
        result1 = subprocess.run(
            ["podman", "run", "-d", "--device", "nvidia.com/gpu=all", "--volume", cuda_insert, "--network", "none", "--user", "10002:10002", "--name", "modelapi", "-p", "6500:6500", "modelapi"], 
            check=True,
            timeout=30
        )
        if result1.returncode != 0:
            return False
        return True
        
    except subprocess.TimeoutExpired as e:
        Utils.subnet_logger(
            severity="ERROR",
            message=f"Container operation timed out: {e}",
            log_level=log_level,
        )
        return False
    except subprocess.CalledProcessError as e:
        Utils.subnet_logger(
            severity="ERROR",
            message=f"Container could not be started due to error: {e}",
            log_level=log_level,
        )
        return False
    except Exception as e:
        Utils.subnet_logger(
            severity="ERROR",
            message=f"Container could not be started due to error: {e}",
            log_level=log_level,
        )
        return False
    
def check_container_status(log_level, timeout=5) -> bool:
    
    url = f"http://127.0.0.1:6500/status/"
    try:
        start_time = int(time.time())
        current_time = start_time
        while start_time + 100 >= current_time:
            try:
                res = requests.get(url, timeout=timeout)
                if res.status_code == 200:
                    data=res.json()
                    if "container_running" in data.keys() and data['container_running']:
                        return True
                current_time = int(time.time())
            except:
                current_time = int(time.time())
            
        return False
    
    except Exception as e:
        Utils.subnet_logger(
            severity="ERROR",
            message=f"Container status could not be determiend due to error: {e}",
            log_level=log_level,
        )
        return False

def upload_audio(noisy_dir, log_level, timeout=10,) -> bool:
    """
    Upload audio files to the API.

    Returns:
        bool: True if operation was successful, False otherwise
    """
    url = f"http://127.0.0.1:6500/upload-audio/"
    
    files = sorted(glob.glob(os.path.join(noisy_dir, "*.wav")))

    try:
        with requests.Session() as session:
            file_payload = [
                ("files", (os.path.basename(file), open(file, "rb"), "audio/wav"))
                for file in files
            ]

            Utils.subnet_logger(
                severity="TRACE",
                message=f"Uploading the following files to the model: {file_payload}",
                log_level=log_level
            )

            response = session.post(url, files=file_payload, timeout=timeout)

            for _, file in file_payload:
                file[1].close()  # Ensure all files are closed after the request

            response.raise_for_status()
            data = response.json()

            sorted_files = sorted([file[1][0] for file in file_payload])
            sorted_response = sorted(data["uploaded_files"])
            return sorted_files == sorted_response and data["status"]

    except requests.RequestException as e:
        Utils.subnet_logger(
            severity="ERROR",
            message=f"Uploading audio to model failed because: {e}",
            log_level=log_level
        )
        return False
    except Exception as e:
        Utils.subnet_logger(
            severity="ERROR",
            message=f"Uploading audio to model failed because: {e}",
            log_level=log_level
        )
        return False
    
def prepare(log_level, timeout=10) -> bool:
    
    url = f"http://127.0.0.1:6500/prepare/"
    try:
        res = requests.post(url, timeout=timeout)
        if res.status_code==200:
            data = res.json()
            return data['preparations']
        return False
    except Exception as e:
        Utils.subnet_logger(
            severity="ERROR",
            message=f"Container model could not be prepared due to error: {e}",
            log_level=log_level,
        )
        return False

def enhance_audio(log_level, timeout=600) -> bool:
    """
    Trigger audio enhancement on the API.

    Returns:
        bool: True if enhancement was successful, False otherwise
    """
    url = f"http://127.0.0.1:6500/enhance/"

    try:
        response = requests.post(url, timeout=timeout)
        response.raise_for_status()
        data = response.json()
        return data['status']
    except requests.RequestException as e:
        Utils.subnet_logger(
            severity="ERROR",
            message=f"Audio could not be enhanced due to error: {e}",
            log_level=log_level,
        )
        return False
    except Exception as e:
        Utils.subnet_logger(
            severity="ERROR",
            message=f"Audio could not be enhanced due to error: {e}",
            log_level=log_level,
        )
        return False

def download_enhanced(enhanced_dir, log_level, timeout=10) -> bool:
    """
    Download the zip file containing enhanced audio files, extract its contents, 
    and remove the zip file.

    Args:
        enhanced_dir (str): Directory to save and extract the downloaded zip file.

    Returns:
        bool: True if successful, False otherwise.
    """
    url = "http://127.0.0.1:6500/download-enhanced/"
    zip_file_path = os.path.join(enhanced_dir, "enhanced_audio_files.zip")

    try:
        # Download the ZIP file
        response = requests.get(url, stream=True, timeout=timeout)
        response.raise_for_status()

        # Save the ZIP file to enhanced_dir
        with open(zip_file_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        # Extract the ZIP file contents to enhanced_dir
        with zipfile.ZipFile(zip_file_path, "r") as zip_file:
            zip_file.extractall(enhanced_dir)

        # Delete the ZIP file after extraction
        os.remove(zip_file_path)

        return True
    except requests.RequestException as e:
        Utils.subnet_logger(
            severity="ERROR",
            message=f"Enhanced audio could not be downloaded due to error: {e}",
            log_level=log_level,
        )
        return False
    except Exception as e:
        Utils.subnet_logger(
            severity="ERROR",
            message=f"Enhanced audio could not be downloaded due to error: {e}",
            log_level=log_level,
        )
        return False

def delete_container(log_level) -> bool:
    """Deletes a specified Docker container by name or ID.

    Returns:
        bool: True if the container was successfully deleted, False otherwise.
    """
    try:
        # Delete container
        subprocess.run(
            ["podman", "rm", "-f", "modelapi"],
            check=True
        )
        # Remove all images
        subprocess.run(
            ["podman", "rmi", "-a", "-f"],
            check=True,
        )
        # System prune
        subprocess.run(
            ["podman", "system", "prune", "-a", "-f"],
            check=True,
        )
        return True

    except Exception as e:
        Utils.subnet_logger(
            severity="ERROR",
            message=f"Container deletion failed due to error: {e}",
            log_level=log_level,
        )
        return False