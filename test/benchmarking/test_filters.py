import pytest
from unittest.mock import patch
import soundsright
import soundsright.base.benchmarking as Benchmarking

@pytest.fixture
def mock_validate_true():
    """Mock Utils.validate_model_benchmark to always return True."""
    with patch.object(soundsright.base.utils, "validate_model_benchmark", return_value=True):
        yield


@pytest.fixture
def hotkeys():
    return ["hk1", "hk2", "hk3", "hk4"]


def test_filter_models_with_same_hash_different_blocks(mock_validate_true, hotkeys):
    models = [
        {"model_hash": "hashA", "block": 10, "hotkey": "hk1"},
        {"model_hash": "hashA", "block": 20, "hotkey": "hk2"},
        {"model_hash": "hashB", "block": 15, "hotkey": "hk3"},
    ]

    result = Benchmarking.filter_models_with_same_hash(models, hotkeys)

    # Keep only the earliest for hashA (block=10) and hashB
    assert len(result) == 2
    assert any(m["block"] == 10 for m in result)
    assert any(m["model_hash"] == "hashB" for m in result)


def test_filter_models_with_same_hash_same_block_uid_order(mock_validate_true, hotkeys):
    models = [
        {"model_hash": "hashA", "block": 10, "hotkey": "hk2"},  # uid=1
        {"model_hash": "hashA", "block": 10, "hotkey": "hk1"},  # uid=0, should be kept
    ]

    result = Benchmarking.filter_models_with_same_hash(models, hotkeys)

    assert len(result) == 1
    assert result[0]["hotkey"] == "hk1"


def test_filter_models_with_same_ckpt_hash_basic(mock_validate_true, hotkeys):
    models = [
        {"ckpt_hash": ["a", "b"], "block": 5, "hotkey": "hk1"},
        {"ckpt_hash": ["b", "c"], "block": 10, "hotkey": "hk2"},  # overlaps "b"
        {"ckpt_hash": ["d"], "block": 7, "hotkey": "hk3"},
    ]

    result = Benchmarking.filter_models_with_same_ckpt_hash(models, hotkeys)

    # Should keep first and third (since second overlaps)
    hashes_kept = {h for m in result for h in m["ckpt_hash"]}
    assert hashes_kept == {"a", "b", "d"}
    assert all(isinstance(r, dict) for r in result)


def test_filter_models_with_same_ckpt_hash_same_block_uid_order(mock_validate_true, hotkeys):
    models = [
        {"ckpt_hash": ["x"], "block": 10, "hotkey": "hk2"},
        {"ckpt_hash": ["x"], "block": 10, "hotkey": "hk1"},
    ]

    result = Benchmarking.filter_models_with_same_ckpt_hash(models, hotkeys)

    # The one with hk1 (lower uid) should appear first and claim the hash
    assert len(result) == 1
    assert result[0]["hotkey"] == "hk1"


def test_filter_models_with_same_metadata_different_blocks(mock_validate_true, hotkeys):
    models = [
        {
            "hf_model_namespace": "nsp1",
            "hf_model_name": "modelA",
            "hf_model_revision": "v1",
            "block": 20,
            "hotkey": "hk1",
        },
        {
            "hf_model_namespace": "nsp1",
            "hf_model_name": "modelA",
            "hf_model_revision": "v1",
            "block": 10,
            "hotkey": "hk2",
        },
        {
            "hf_model_namespace": "nsp2",
            "hf_model_name": "modelB",
            "hf_model_revision": "v1",
            "block": 5,
            "hotkey": "hk3",
        },
    ]

    result = Benchmarking.filter_models_with_same_metadata(models, hotkeys)

    # Expect modelA earliest block (10) and modelB (5)
    assert len(result) == 2
    assert any(m["block"] == 10 for m in result)
    assert any(m["hf_model_name"] == "modelB" for m in result)


def test_filter_models_with_same_metadata_same_block_uid_order(mock_validate_true, hotkeys):
    models = [
        {
            "hf_model_namespace": "nsp",
            "hf_model_name": "modelX",
            "hf_model_revision": "v1",
            "block": 5,
            "hotkey": "hk3",  # uid=2
        },
        {
            "hf_model_namespace": "nsp",
            "hf_model_name": "modelX",
            "hf_model_revision": "v1",
            "block": 5,
            "hotkey": "hk1",  # uid=0
        },
    ]

    result = Benchmarking.filter_models_with_same_metadata(models, hotkeys)

    assert len(result) == 1
    assert result[0]["hotkey"] == "hk1"


def test_filter_ignores_unregistered_hotkeys(mock_validate_true, hotkeys):
    models = [
        {"model_hash": "hashA", "block": 5, "hotkey": "hk_unknown"},
        {"model_hash": "hashB", "block": 5, "hotkey": "hk1"},
    ]

    result = Benchmarking.filter_models_with_same_hash(models, hotkeys)
    assert len(result) == 1
    assert result[0]["model_hash"] == "hashB"
