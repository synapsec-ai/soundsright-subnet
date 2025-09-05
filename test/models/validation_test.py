import soundsright.base.models as Models
import os
import shutil
import pytest
import asyncio

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
            
@pytest.mark.parametrize("namespace, name, revision", [
    ("synapsecai", "SoundsRightModelTemplate", "191137582e8f5b67a0d5fee2f60a08a8a46f3d98"),
    ("synapsecai", "SoundsRightModelTemplate", "763a9b1c598bd16648ffccc9b9c67c67b6aa3805"),
])
def test_get_model_content_hash(namespace, name, revision):
    
    model_path=f"{os.path.expanduser('~')}/.soundsright/model_test"
    model_dir = os.path.join(os.getcwd(), model_path)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    model_hash_1, sorted_files_1 = asyncio.run(Models.get_model_content_hash(
        namespace=namespace,
        name=name,
        revision=revision,
        local_dir=model_dir,
        log_level="INFO"
    ))
    
    assert model_hash_1 != None, "Model hash is None"
    assert sorted_files_1 != None, "Sorted files is None"
    assert isinstance(sorted_files_1, list), "Sorted files is not a list"
    assert len(sorted_files_1) != 0, "Sorted files length is zero"

    remove_all_in_path(model_dir)
    
    model_hash_2, sorted_files_2 = asyncio.run(Models.get_model_content_hash(
        namespace=namespace,
        name=name,
        revision=revision,
        local_dir=model_dir,
        log_level="INFO"
    ))
    
    remove_all_in_path(model_dir)
    shutil.rmtree(model_dir)
    
    assert len(sorted_files_1) == len(sorted_files_2), "File lengths different"
    assert sorted_files_1 == sorted_files_2, "File order different"
    assert model_hash_1 == model_hash_2, "Hashes different"

    print(f"Model hash for model: {namespace}/{name} with branch: {revision}: {model_hash_1}")

@pytest.mark.parametrize("namespace, name, revision", [
    ("synapsecai", "SoundsRightModelTemplate", "191137582e8f5b67a0d5fee2f60a08a8a46f3d98"),
    ("synapsecai", "SoundsRightModelTemplate", "763a9b1c598bd16648ffccc9b9c67c67b6aa3805"),
])
def test_get_model_content_hash_bulk(namespace, name, revision):
    
    model_path=f"{os.path.expanduser('~')}/.soundsright/model_test"
    model_dir = os.path.join(os.getcwd(), model_path)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    count = 0
    while count < 2:
    
        model_hash_1, sorted_files_1 = asyncio.run(Models.get_model_content_hash(
            namespace=namespace,
            name=name,
            revision=revision,
            local_dir=model_dir,
            log_level="INFO"
        ))
        
        remove_all_in_path(model_dir)
        assert model_hash_1 != None, "Model download failed in bulk process"
        count += 1
        
@pytest.mark.parametrize("namespace, name, revision, outcome", [
    ("synapsecai", "SoundsRightModelTemplate", "191137582e8f5b67a0d5fee2f60a08a8a46f3d98", True),
    ("synapsecai", "SoundsRightModelTemplate", "763a9b1c598bd16648ffccc9b9c67c67b6aa3805", True),
    ("synapsecai", "SoundsRightModelTemplate", "191137582e8f5b67a0d5fee2f60a08a8a46f3d9", False),
    ("synapsecai", "SoundsRightModelTemplate", "763a9b1c59d53916648ffccc9b9c67c67b6aa3805", False),
])
def test_check_repo_exists(namespace, name, revision, outcome):
    assert asyncio.run(Models.check_repo_exists(namespace=namespace,name=name,revision=revision)) == outcome, f"Model existence check failed for {namespace}/{name}/{revision}"

@pytest.mark.parametrize("namespace, name, revision", [
    ("synapsecai", "SoundsRightModelTemplate", "191137582e8f5b67a0d5fee2f60a08a8a46f3d98"),
    ("synapsecai", "SoundsRightModelTemplate", "763a9b1c598bd16648ffccc9b9c67c67b6aa3805"),
])
def test_check_repo_exists_bulk(namespace, name, revision):
    count = 0
    while count < 10:
        if not asyncio.run(Models.check_repo_exists(namespace=namespace,name=name,revision=revision)):
            raise AssertionError(f"Bulk model existence check failed for {namespace}/{name}/{revision}")
            break
        count += 1
    
@pytest.mark.parametrize("namespace, name, revision, outcome", [
    ("synapsecai", "SoundsRightModelTemplate", "191137582e8f5b67a0d5fee2f60a08a8a46f3d98", True),
    ("synapsecai", "SoundsRightModelTemplate", "763a9b1c598bd16648ffccc9b9c67c67b6aa3805", True),
    ("synapsecai", "SoundsRightModelTemplate", "191137582e8f5b67a0d5fee2f60a08a8a46f3d9", False),
    ("synapsecai", "SoundsRightModelTemplate", "DEREVERBERATION_16000HZ", False),
])
def test_is_commit_hash(namespace, name, revision, outcome):
    assert asyncio.run(Models.is_commit_hash(namespace=namespace, name=name, revision=revision)) == outcome, f"is_commit_hash check failed for {revision}"

@pytest.mark.parametrize("namespace, name, revision", [
    ("synapsecai", "SoundsRightModelTemplate", "191137582e8f5b67a0d5fee2f60a08a8a46f3d98"),
    ("synapsecai", "SoundsRightModelTemplate", "763a9b1c598bd16648ffccc9b9c67c67b6aa3805"),
])
def test_is_commit_hash_bulk(namespace, name, revision):
    count = 0
    while count < 1000:
        if not asyncio.run(Models.is_commit_hash(namespace=namespace,name=name,revision=revision)):
            raise AssertionError(f"Bulk model existence check failed for {namespace}/{name}/{revision}")
            break
        count += 1