from copy import deepcopy
import pytest 

def filter_cache(model_cache):

    for competition in model_cache.keys():

        filtered_models = []
        unique_models = {}

        for model in model_cache[competition]:
            rs = model.get("response_data", None)

            # Verify response data exists
            if rs:

                namespace = rs.get("hf_model_namespace", None)
                name = rs.get("hf_model_name", None)
                block = model.get("block", None)
                uid = model.get("uid", None)

                # Verify namespace, name and uid exist
                if namespace and name and uid:

                    # Create unique key
                    unique_key = f"{namespace}_{name}"
                    uid = int(uid)

                    # In the case this already exists
                    if unique_key in unique_models.keys():

                        # Determine existing block and uid
                        existing_block = unique_models[unique_key].get("block", None)
                        existing_uid = int(unique_models[unique_key].get("uid", 257))

                        # All the ways the new entry can replace the old one:
                        # If both entries do not have a block specified from historical data
                        if not block and not existing_block:
                            # Prioritize the lower uid
                            if uid and existing_uid and uid < existing_uid:
                                unique_models[unique_key] = model

                        # If only the model in the loop has a registered block 
                        elif block and not existing_block:
                            unique_models[unique_key] = model
                        
                        # If blocks are equal values
                        elif block and existing_block and block == existing_block:
                            # Prioritize the lower uid
                            if uid and existing_uid and uid < existing_uid:
                                unique_models[unique_key] = model

                        # If both blocks are specified and the new model has a lower block value
                        elif block and existing_block and block < existing_block:
                            unique_models[unique_key] = model

                    else: 
                        unique_models[unique_key] = model

        filtered_models = list(unique_models.values())
        model_cache[competition] = filtered_models

    return model_cache


def build_model(namespace, name, uid, block=None):
    return {
        "response_data": {
            "hf_model_namespace": namespace,
            "hf_model_name": name,
        },
        "uid": uid,
        "block": block,
    }


def test_filter_different_models():
    cache = {
        "comp1": [
            build_model("org1", "modelA", 100),
            build_model("org2", "modelB", 101),
        ]
    }
    result = filter_cache(deepcopy(cache))
    assert len(result["comp1"]) == 2


def test_same_model_different_uid():
    cache = {
        "comp1": [
            build_model("org1", "modelA", 100),
            build_model("org1", "modelA", 99),
        ]
    }
    result = filter_cache(deepcopy(cache))
    assert len(result["comp1"]) == 1
    assert result["comp1"][0]["uid"] == 99


def test_prioritize_model_with_block():
    cache = {
        "comp1": [
            build_model("org1", "modelA", 100),
            build_model("org1", "modelA", 101, block=5),
        ]
    }
    result = filter_cache(deepcopy(cache))
    assert result["comp1"][0]["block"] == 5


def test_block_comparison_lower_wins():
    cache = {
        "comp1": [
            build_model("org1", "modelA", 100, block=10),
            build_model("org1", "modelA", 101, block=5),
        ]
    }
    result = filter_cache(deepcopy(cache))
    assert result["comp1"][0]["block"] == 5


def test_equal_block_lower_uid_wins():
    cache = {
        "comp1": [
            build_model("org1", "modelA", 102, block=7),
            build_model("org1", "modelA", 101, block=7),
        ]
    }
    result = filter_cache(deepcopy(cache))
    assert result["comp1"][0]["uid"] == 101


def test_missing_response_data():
    cache = {
        "comp1": [
            {"uid": 123, "block": 1},
            build_model("org1", "modelA", 100)
        ]
    }
    result = filter_cache(deepcopy(cache))
    assert len(result["comp1"]) == 1


def test_missing_namespace_or_name_or_uid():
    cache = {
        "comp1": [
            {"response_data": {"hf_model_namespace": "org1"}, "uid": 1},  # Missing name
            {"response_data": {"hf_model_name": "modelA"}, "uid": 1},    # Missing namespace
            {"response_data": {"hf_model_namespace": "org1", "hf_model_name": "modelA"}},  # Missing uid
            build_model("org1", "modelA", 99),
        ]
    }
    result = filter_cache(deepcopy(cache))
    assert len(result["comp1"]) == 1
    assert result["comp1"][0]["uid"] == 99


def test_multiple_competitions():
    cache = {
        "comp1": [build_model("org1", "modelA", 100)],
        "comp2": [build_model("org2", "modelB", 101, block=2), build_model("org2", "modelB", 100, block=1)]
    }
    result = filter_cache(deepcopy(cache))
    assert len(result["comp1"]) == 1
    assert len(result["comp2"]) == 1
    assert result["comp2"][0]["block"] == 1


def test_uid_comparison_with_defaults():
    cache = {
        "comp1": [
            build_model("org1", "modelA", 200),
            {"response_data": {"hf_model_namespace": "org1", "hf_model_name": "modelA"}, "uid": 100}
        ]
    }
    result = filter_cache(deepcopy(cache))
    assert result["comp1"][0]["uid"] == 100