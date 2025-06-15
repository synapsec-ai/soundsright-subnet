import soundsright.base.models as Models
import pytest

@pytest.mark.parametrize("namespace, name, revision, outcome",[
    ("synapsecai", "NotARealModel", "DENOISING_16000HZ", False),
    ("synapsecai", "SoundsRightModelTemplate", "DENOISING_16000HZ", False),
    ("synapsecai", "SoundsRightModelTemplate", "main", False),
    ("synapsecai", "SoundsRightModelTemplate", "191137582e8f5b67a0d5fee2f60a08a8a46f3d98", True),
    ("synapsecai", "SoundsRightModelTemplate", "70a8146e69bc18074f6b73f9ea4bf068aaa3ec8b", True),
    ("synapsecai", "SoundsRightModelTemplate", "70a8146e69bc18074f6b73f9ea4bf068aaa3ec8c", False),
])
def test_repo_verification(namespace, name, revision, outcome):

    assert Models.validate_repo_and_revision(namespace=namespace, name=name, revision=revision) == outcome