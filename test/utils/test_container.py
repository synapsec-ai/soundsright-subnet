from git import Repo
import time
import os 
import shutil
import yaml
import glob
import pytest  

import soundsright.base.utils as Utils

def remove_all_in_path(path):
    """
    Removes all files and directories located at the specified path.
    
    Args:
        path (str): The path to the directory to clear.
    """
    if not os.path.isdir(path):
        raise ValueError(f"The specified path '{path}' is not a directory.")

    for filename in os.listdir(path):
        file_path = os.path.join(path, filename)
        
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.remove(file_path)  # Remove file or symlink
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)  # Remove directory and its contents
        except Exception as e:
            print(f"Failed to delete {file_path}. Reason: {e}")
    

def validate_all_noisy_files_are_enhanced(noisy_dir, enhanced_dir):
    noisy_files = sorted([os.path.basename(f) for f in glob.glob(os.path.join(noisy_dir, '*.wav'))])
    enhanced_files = sorted([os.path.basename(f) for f in glob.glob(os.path.join(enhanced_dir, '*.wav'))])
    return noisy_files == enhanced_files
            
@pytest.mark.parametrize("model_id, revision",[
    ("synapsecai/SoundsRightModelTemplate", "DENOISING_16000HZ"),
    ("synapsecai/SoundsRightModelTemplate", "DEREVERBERATION_16000HZ")
])
def test_container(model_id, revision):
    
    model_dir=f"{os.path.expanduser('~')}/.SoundsRight/data/model"
    noisy_dir = f"{os.path.expanduser('~')}/.SoundsRight/data/noise/16000"
    reverb_dir = f"{os.path.expanduser('~')}/.SoundsRight/data/reverb/16000"
    enhanced_noise_dir = f"{os.path.expanduser('~')}/.SoundsRight/test_data/enhanced_noise/16000"
    enhanced_reverb_dir = f"{os.path.expanduser('~')}/.SoundsRight/test_data/enhanced_reverb/16000"
    
    for directory in [model_dir, noisy_dir, reverb_dir, enhanced_noise_dir, enhanced_reverb_dir]:
        if not os.path.exists(directory):
            os.makedirs(directory)    
        
    repo_url = f"https://huggingface.co/{model_id}"
    
    shutil.rmtree(model_dir)
    
    # Download the model files for the specified revision
    Repo.clone_from(repo_url, model_dir, branch=revision)
    
    assert Utils.validate_container_config(model_dir) == True, "Container config invalid"
    
    assert Utils.start_container(directory=model_dir, log_level="INFO"), "Container could not be started"
    time.sleep(5)
    assert Utils.check_container_status(log_level="INFO", timeout=None), "Container status invalid"
    assert Utils.prepare(log_level="INFO",timeout=None), "Model prepration failed"
    time.sleep(5)
    if "DENOISING" in revision:
        assert Utils.upload_audio(noisy_dir, log_level="INFO", timeout=None), "Audio could not be uploaded"
    else: 
        assert Utils.upload_audio(reverb_dir, log_level="INFO", timeout=None), "Audio could not be uploaded"
    time.sleep(5)
    assert Utils.enhance_audio(log_level="INFO", timeout=None), "Files could not be enhanced"
    if "DENOISING" in revision:
        assert Utils.download_enhanced(enhanced_noise_dir, log_level="INFO", timeout=None), "Enhanced files could not be downloaded"
        assert validate_all_noisy_files_are_enhanced(noisy_dir, enhanced_noise_dir), "Mismatch between noisy files and enhanced files"
    else:
        assert Utils.download_enhanced(enhanced_reverb_dir, log_level="INFO", timeout=None), "Enhanced files could not be downloaded"
        assert validate_all_noisy_files_are_enhanced(reverb_dir, enhanced_reverb_dir), "Mismatch between noisy files and enhanced files"
    time.sleep(5)
    assert Utils.delete_container(log_level="INFO"), "Container could not be deleted"
    
    remove_all_in_path(model_dir)
    shutil.rmtree(model_dir)