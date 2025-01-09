import soundsright.base.models as Models
import os
import shutil
import pytest

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
            
@pytest.mark.parametrize("model_id", [
    ("huseinzol05/speech-enhancement-mask-unet"),
    ("sp-uhh/speech-enhancement-sgmse"), 
    ("rollingkevin/speech-enhancement-unet"),
])
def test_get_model_content_hash(model_id):
    
    model_path=f"{os.path.expanduser('~')}/.soundsright/model_test"
    model_dir = os.path.join(os.getcwd(), model_path)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    model_hash_1, sorted_files_1 = Models.get_model_content_hash(
        model_id=model_id,
        revision="main",
        local_dir=model_dir,
        log_level="INFO"
    )
    
    remove_all_in_path(model_dir)
    
    model_hash_2, sorted_files_2 = Models.get_model_content_hash(
        model_id=model_id,
        revision="main",
        local_dir=model_dir,
        log_level="INFO"
    )
    
    remove_all_in_path(model_dir)
    shutil.rmtree(model_dir)
    
    assert len(sorted_files_1) == len(sorted_files_2), "File lengths different"
    assert sorted_files_1 == sorted_files_2, "File order different"
    assert model_hash_1 == model_hash_2, "Hashes different"
