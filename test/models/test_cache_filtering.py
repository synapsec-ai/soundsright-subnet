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

                            # If only the model in storage has a registered block 
                            elif existing_block and not block:
                                unique_models[unique_key] = model
                            
                            # If blocks are equal values
                            elif block and existing_block and block == existing_block:
                                # Prioritize the lower uid
                                if uid and existing_uid and uid < existing_uid:
                                    unique_models[unique_key] = model

                            # If both blocks are specified and the new model has a higher block value (it was submitted later, i.e. this is the most recent version)
                            elif block and existing_block and block > existing_block:
                                unique_models[unique_key] = model

                        else: 
                            unique_models[unique_key] = model

            filtered_models = list(unique_models.values())
            model_cache[competition] = filtered_models
        
        return model_cache



# Define test cases
test_cases = [
    {
        "input": {
            "comp1": [
                {"response_data": {"hf_model_namespace": "user", "hf_model_name": "modelA"}, "block": 10, "uid": 100},
                {"response_data": {"hf_model_namespace": "user", "hf_model_name": "modelA"}, "block": 12, "uid": 90},
                {"response_data": {"hf_model_namespace": "user", "hf_model_name": "modelA"}, "block": 12, "uid": 80},
            ]
        },
        "description": "Chooses most recent block, then lower uid if same block"
    },
    {
        "input": {
            "comp2": [
                {"response_data": {"hf_model_namespace": "user", "hf_model_name": "modelB"}, "uid": 50},
                {"response_data": {"hf_model_namespace": "user", "hf_model_name": "modelB"}, "uid": 40},
            ]
        },
        "description": "No block, picks lower uid"
    },
    {
        "input": {
            "comp3": [
                {"response_data": {"hf_model_namespace": "user", "hf_model_name": "modelC"}, "block": 5, "uid": 20},
                {"response_data": {"hf_model_namespace": "user", "hf_model_name": "modelC"}, "uid": 10},
            ]
        },
        "description": "Existing has block, new has none → replaces"
    },
    {
        "input": {
            "comp4": [
                {"response_data": {"hf_model_namespace": "user", "hf_model_name": "modelD"}, "block": 7, "uid": 9},
                {"response_data": {"hf_model_namespace": "user", "hf_model_name": "modelD"}, "block": 7, "uid": 8},
            ]
        },
        "description": "Same block, picks lower uid"
    },
]

import copy
test_outputs = []
for test in test_cases:
    input_data = copy.deepcopy(test["input"])
    print(f"{test["description"]}")
    print(f"Inputs: {input_data}")
    output = filter_cache(input_data)
    print(f"Outputs: {output}")
    