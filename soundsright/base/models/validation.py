import hashlib
import os
import base64
import shutil
import subprocess
import tempfile
import re
import asyncio
from git import Repo, Git, GitCommandError
from huggingface_hub import snapshot_download, HfApi

import soundsright.base.utils as Utils

def _hf_repo_url(namespace: str, name: str) -> str:
    return f"https://huggingface.co/{namespace}/{name}"

def get_file_content_hash(filepath, chunk_size=8192):
    """
    Computes SHA-256 hash of a single file
    """
    sha256_hash = hashlib.sha256()
    
    try:
        with open(filepath, "rb") as f:
            for chunk in iter(lambda: f.read(chunk_size), b""):
                sha256_hash.update(chunk)
    except Exception as e:
        return ""
    
    return sha256_hash.hexdigest()

def get_checkpoint_hash(directory):

    checkpoint_filetypes = [".pt", ".pth", ".ckpt", ".h5", ".bin", ".safetensors", ".msgpack", ".npz", ".onnx", ".engine"]
    output = []

    for root, _, files in os.walk(directory):
        for file in files:
            if any(file.endswith(ext) for ext in checkpoint_filetypes):
                filepath = os.path.join(root, file)
                file_hash = get_file_content_hash(filepath)
                if file_hash:
                    output.append((file_hash))

    return output

def verify_directory_files(directory):
    
    forbidden_hashes = [
        "e3875747b5646092d5c556bae68e5af639e2c1f45f009c669f379cd4d415cbd8",
        "2ec94cf546ef0a9d66f90364bd735820c78d9a214133588e90ce9ce01cd8a73b",
        "b770d098538dec1c06c6917bf327a7922ef326321aa23678f071d86c5f39716f",
    ]
    for root, _, files in os.walk(directory):
        for file_name in files:
            file_path = os.path.join(root, file_name)
            file_hash = get_file_content_hash(file_path)
            
            if file_hash in forbidden_hashes:
                return False

    return True

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

@Utils.timeout_decorator(timeout=60)
async def get_model_content_hash(
    namespace: str,
    name: str,
    revision: str, 
    local_dir: str, 
    log_level: str,
):
    """
    Downloads the model and computes the hash of its entire contents.

    Args:
        :param namespace: (str): The user/org who uploaded the model.
        :param name: (str): The name of the model.
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
    
    try:
        
        url = _hf_repo_url(namespace=namespace,name=name)

        # Clone without checkout, then fetch the revision shallowly and checkout
        Utils.subnet_logger(
            severity="TRACE",
            message=f"Cloning {url} to {local_dir}.",
            log_level=log_level
        )
        repo = Repo.clone_from(url, local_dir, no_checkout=True)

        # Fetch the exact revision (works for branch, tag, or commit SHA)
        repo.git.fetch("origin", revision)
        repo.git.checkout(revision, force=True)
        Utils.subnet_logger(
            severity="TRACE",
            message=f"Fetched revision: {revision}",
            log_level=log_level
        )

        # Compute the hash of the model's contents
        return get_directory_content_hash(directory=local_dir)
    
    except GitCommandError as e:
        Utils.subnet_logger(
            severity="ERROR",
            message=f"Git error cloning/checking out {namespace}/{name}/{revision}: {e}",
            log_level=log_level
        )
    except Exception as e:
        Utils.subnet_logger(
            severity="ERROR",
            message=f"Model {namespace}/{name}/{revision} could not be downloaded because : {e}",
            log_level=log_level
        )
        return None, None

@Utils.timeout_decorator(timeout=20)
async def check_repo_exists(namespace: str, name: str, revision: str) -> bool:
    """
    Check if a Hugging Face repository exists with the specified revision.
    
    Args:
        namespace (str): The namespace/username of the repository
        name (str): The name of the repository
        revision (str): The revision (commit hash, branch, or tag)
    
    Returns:
        bool: True if the repository and revision exist, False otherwise
    """
    try:
        url = _hf_repo_url(namespace, name)
        with tempfile.TemporaryDirectory() as tmpdir:
            repo=Repo.clone_from(
                url,
                tmpdir,
                no_checkout=True
            )
            repo.git.fetch("origin", revision)
            repo.git.checkout(revision, force=True)
        return True
    
    except Exception as e:
        Utils.subnet_logger(
            severity="INFO",
            message=f"Error when checking if repo exists for namespace: {namespace}, name: {name}, revision: {revision}: {e}"
        )
        
        return False

@Utils.timeout_decorator(timeout=20)
async def is_commit_hash(namespace: str, name: str, revision: str) -> bool:
    """
    Check if a revision is actually a commit hash by verifying it with the repository.
    
    This function first checks if the revision looks like a commit hash (40 hex chars),
    then verifies it's actually a commit by checking that it exists in the repo and
    that it's not a branch or tag name.
    
    Args:
        namespace (str): The namespace/username of the repository
        name (str): The name of the repository
        revision (str): The revision string to check
    
    Returns:
        bool: True if the revision is a genuine commit hash, False otherwise
    """
    try:
        url = _hf_repo_url(namespace, name)
        with tempfile.TemporaryDirectory() as tmpdir:
            repo=Repo.clone_from(
                url,
                tmpdir,
                no_checkout=True
            )
            repo.git.fetch("origin", revision)
            repo.git.checkout(revision, force=True)
            all_commits = list(repo.iter_commits(revision))
            return revision in [commit.hexsha for commit in all_commits]
        
    except Exception as e:
        Utils.subnet_logger(
            severity="INFO",
            message=f"Error when checking if commit hash exists for namespace: {namespace}, name: {name}, revision: {revision}: {e}"
        )
        
        return False


def is_valid_commit_hash_format(revision: str) -> bool:
    """
    Check if a revision string has the format of a commit hash without repo verification.
    Use this for initial validation before making API calls.
    
    Args:
        revision (str): The revision string to check
    
    Returns:
        bool: True if the revision has commit hash format, False otherwise
    """
    commit_hash_pattern = r'^[a-f0-9]{40}$'
    return bool(re.match(commit_hash_pattern, revision))

def validate_repo_and_revision(namespace: str, name: str, revision: str, log_level: str) -> tuple[bool, bool]:
    """
    Validate both repository existence and revision format.
    
    Args:
        namespace (str): The namespace/username of the repository
        name (str): The name of the repository
        revision (str): The revision to validate
    
    Returns:
        tuple[bool, bool]: (repo_exists, is_commit_hash)
    """

    if not is_valid_commit_hash_format(revision):

        Utils.subnet_logger(
            severity="TRACE",
            message=f"Revision: {revision} did not pass simple commit hash regex check.",
            log_level=log_level
        )

        return False
    
    Utils.subnet_logger(
        severity="TRACE",
        message=f"Revision: {revision} passed simple commit hash regex check.",
        log_level=log_level
    )

    if not asyncio.run(check_repo_exists(namespace, name, revision)):
        
        Utils.subnet_logger(
            severity="TRACE",
            message=f"Repo: {namespace}/{name} with revision: {revision} does not exist.",
            log_level=log_level
        )

        return False
    
    Utils.subnet_logger(
        severity="TRACE",
        message=f"Repo: {namespace}/{name} with revision: {revision} exists.",
        log_level=log_level
    )

    if not asyncio.run(is_commit_hash(namespace, name, revision)):

        Utils.subnet_logger(
            severity="TRACE",
            message=f"Revision: {revision} is not in valid commit hash format.",
            log_level=log_level
        )

        return False
    
    Utils.subnet_logger(
        severity="TRACE",
        message=f"Revision: {revision} is valid commit hash.",
        log_level=log_level
    )
    
    return True