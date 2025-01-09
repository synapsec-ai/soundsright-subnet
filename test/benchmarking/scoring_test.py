import pytest
from unittest.mock import Mock, patch
import numpy as np
import soundsright.base.benchmarking as Benchmarking 

@pytest.mark.parametrize("new_model_block, old_model_block, start_improvement, end_improvement, decay_block, expected", [
    (0, 0, 0.0035, 0.0015, 50400, 0.0035),  # No difference
    (50400, 0, 0.0035, 0.0015, 50400, 0.0015),  # Full decay, should reach end_improvement
    (25200, 0, 0.0035, 0.0015, 50400, 0.0025),  # Halfway through decay
    (50400, 50400, 0.0035, 0.0015, 50400, 0.0035),  # Same blocks, no improvement adjustment
    (100000, 0, 0.0035, 0.0015, 50400, 0.0015),  # Beyond decay, should cap at end_improvement
    (30000, 10000, 0.0035, 0.0015, 50400, 0.002706),  # Partial decay with a difference of 20,000 blocks
])

def test_calculate_improvement_factor(new_model_block, old_model_block, start_improvement, end_improvement, decay_block, expected):
    result = Benchmarking.calculate_improvement_factor(
        new_model_block, 
        old_model_block, 
        start_improvement=start_improvement, 
        end_improvement=end_improvement, 
        decay_block=decay_block
    )
    assert result == pytest.approx(expected, rel=1e-2), f"Expected {expected}, got {result}"

    
@pytest.mark.parametrize("new_model_metric, new_model_block, old_model_metric, old_model_block, expected", [
    (1.01, 0, 1.0, 0, True),  # New model surpasses old with improvement factor at start value
    (1.007, 25200, 1.0, 0, True),  # Halfway decay improvement factor
    (1.0015, 50400, 1.0, 0, True),  # Full decay improvement factor met
    (1.0014, 50400, 1.0, 0, False),  # Full decay but not enough improvement
    (1.0, 0, 1.0, 0, False),  # No improvement
    (1.02, 0, 1.01, 10000, True),  # New model surpasses, partial decay
    (1.008, 100000, 1.0, 0, True),  # New model exceeds best model with block difference beyond decay
    (1.005, 100000, 1.0, 0, True),  # Exactly at end improvement after max decay
    (1.0002, 50400, 1.0, 0, False),  # New model slightly better but not enough
    (0.99, 0, 1.0, 0, False),  # New model worse than best model
])

def test_new_model_surpasses_historical_model(new_model_metric, new_model_block, old_model_metric, old_model_block, expected):
    result = Benchmarking.new_model_surpasses_historical_model(new_model_metric, new_model_block, old_model_metric, old_model_block)
    assert result == expected, f"Expected {expected} but got {result}"

    
def test_get_best_model_from_list():
    models_data = [
        {
            "name": "Model A",
            "metrics": {
                "PESQ": {"average": 1.1},
                "ESTOI": {"average": 1.2},
            },
        },
        {
            "name": "Model B",
            "metrics": {
                "PESQ": {"average": 1.3},
                "ESTOI": {"average": 1.0},
            },
        },
        {
            "name": "Model C",
            "metrics": {
                "PESQ": {"average": 0.9},
                "ESTOI": {"average": 1.4},
            },
        },
    ]

    # Test for PESQ metric
    best_model_pesq = Benchmarking.get_best_model_from_list(models_data, "PESQ")
    assert best_model_pesq["name"] == "Model B"

    # Test for ESTOI metric
    best_model_estoi = Benchmarking.get_best_model_from_list(models_data, "ESTOI")
    assert best_model_estoi["name"] == "Model C"

    # Test for non-existent metric
    best_model_nonexistent = Benchmarking.get_best_model_from_list(models_data, "SI_SDR")
    assert best_model_nonexistent is None

    # Test with empty models_data
    best_model_empty = Benchmarking.get_best_model_from_list([], "PESQ")
    assert best_model_empty is None

    # Test with invalid metric data
    models_data_invalid = [
        {
            "name": "Model D",
            "metrics": {
                "PESQ": {"average": "invalid"},
            },
        },
        {
            "name": "Model E",
            "metrics": {
                "PESQ": {"average": None},
            },
        },
    ]
    best_model_invalid = Benchmarking.get_best_model_from_list(models_data_invalid, "PESQ")
    assert best_model_invalid is None
            
@pytest.mark.parametrize(
    "competition_scores, competition_max_scores, metric_proportions, best_miner_models, miner_models, expected_scores, expected_best_models",
    [
        # Edge case: No models submitted for a competition
        (
            {"competition1": np.array([0, 0])},
            {"competition1": 100},
            {"competition1": {"PESQ": 1.0}},
            {"competition1": []},
            {"competition1": []},
            {"competition1": np.array([0, 0])},
            {"competition1": []},
        ),
        # Edge case: Single model submitted
        (
            {"competition1": np.array([0, 0])},
            {"competition1": 100},
            {"competition1": {"PESQ": 1.0}},
            {"competition1": []},
            {"competition1": [
                {"hotkey": "miner_hotkey_ss58adr", "metrics": {"PESQ": {"average": 1.5}}, "block": 10},
            ]},
            {"competition1": np.array([100, 0])},
            {"competition1": [
                {"hotkey": "miner_hotkey_ss58adr", "metrics": {"PESQ": {"average": 1.5}}, "block": 10},
            ]},
        ),
        # Edge case: Multiple models submitted, same metric values
        (
            {"competition1": np.array([0, 0])},
            {"competition1": 100},
            {"competition1": {"PESQ": 1.0}},
            {"competition1": []},
            {"competition1": [
                {"hotkey": "miner_hotkey_ss58adr", "metrics": {"PESQ": {"average": 1.1}}, "block": 10},
                {"hotkey": "miner_hotkey_ss58adr2", "metrics": {"PESQ": {"average": 1.1}}, "block": 15},
            ]},
            {"competition1": np.array([100, 0])},
            {"competition1": [
                {"hotkey": "miner_hotkey_ss58adr", "metrics": {"PESQ": {"average": 1.1}}, "block": 10},
            ]},
        ),
        # Edge case: Historical model outperforms current model
        (
            {"competition1": np.array([0, 0])},
            {"competition1": 100},
            {"competition1": {"PESQ": 1.0}},
            {"competition1": [
                {"hotkey": "miner_hotkey_ss58adr", "metrics": {"PESQ": {"average": 1.5}}, "block": 5},
            ]},
            {"competition1": [
                {"hotkey": "miner_hotkey_ss58adr2", "metrics": {"PESQ": {"average": 1.2}}, "block": 10},
            ]},
            {"competition1": np.array([100, 0])},
            {"competition1": [
                {"hotkey": "miner_hotkey_ss58adr", "metrics": {"PESQ": {"average": 1.5}}, "block": 5},
            ]},
        ),
        # Edge case: Historical model outperforms current model in one metric and current model outperforms historical model in the other metric
        (
            {"competition1": np.array([0, 0])},
            {"competition1": 100},
            {"competition1": {"PESQ": 0.3, "ESTOI": 0.7}},
            {"competition1": [
                {"hotkey": "miner_hotkey_ss58adr", "metrics": {"PESQ": {"average": 1.5}, "ESTOI": {"average": 0.5}}, "block": 5},
            ]},
            {"competition1": [
                {"hotkey": "miner_hotkey_ss58adr2", "metrics": {"PESQ": {"average": 1.2}, "ESTOI": {"average": 0.9}}, "block": 10},
            ]},
            {"competition1": np.array([30, 70])},
            {"competition1": [
                {"hotkey": "miner_hotkey_ss58adr", "metrics": {"PESQ": {"average": 1.5}, "ESTOI": {"average": 0.5}}, "block": 5},
                {"hotkey": "miner_hotkey_ss58adr2", "metrics": {"PESQ": {"average": 1.2}, "ESTOI": {"average": 0.9}}, "block": 10},
            ]},
        ),
    ],
)
def test_determine_competition_scores(
    competition_scores,
    competition_max_scores,
    metric_proportions,
    best_miner_models,
    miner_models,
    expected_scores,
    expected_best_models,
):
    metagraph = Mock()
    metagraph.hotkeys = ["miner_hotkey_ss58adr", "miner_hotkey_ss58adr2"]

    new_best_miner_models, updated_competition_scores = Benchmarking.determine_competition_scores(
        competition_scores,
        competition_max_scores,
        metric_proportions,
        best_miner_models,
        miner_models,
        metagraph,
        log_level="INFO",
    )

    for competition, scores in updated_competition_scores.items():
        assert np.array_equal(scores, expected_scores[competition]), (
            f"updated_competition_scores[{competition}]: {scores} "
            f"is not equal to expected_scores[{competition}]: {expected_scores[competition]}"
        )
    
    # Comparison for best-performing models
    assert new_best_miner_models == expected_best_models, (
        f"new_best_miner_models: {new_best_miner_models} "
        f"is not equal to expected_best_models: {expected_best_models}"
    )

    
        
@pytest.mark.parametrize("competition_scores, initial_scores, expected", [
    # Basic case
    ({"comp1": np.array([0,0,15,0]), "comp2": np.array([0,0,0,10]), "comp3": np.array([0,5,0,0])}, np.zeros(4), np.array([0, 5, 15, 10])),

    # No updates (empty competition_scores)
    ({"comp1":np.zeros(3), "comp2":np.zeros(3)}, np.zeros(3), np.array([0, 0, 0])),
])

def test_calculate_overall_scores_varied(competition_scores, initial_scores, expected):
    log_level = "INFO"

    # Call the function
    updated_scores = Benchmarking.calculate_overall_scores(competition_scores, initial_scores, log_level)
    
    # Assertions
    np.testing.assert_array_equal(updated_scores, expected, "The overall scores were not calculated as expected.")
    