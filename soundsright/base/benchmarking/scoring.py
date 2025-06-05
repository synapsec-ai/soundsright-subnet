import bittensor as bt
import numpy as np 
from typing import List

import soundsright.base.utils as Utils 

# This function is adapted from the LinearDecay.compute_epsilon function in the taoverse repository
# by macrocosm-os, available at https://github.com/macrocosm-os/taoverse/
def calculate_improvement_factor(new_model_block, old_model_block, start_improvement = 0.0035, end_improvement = 0.0015, decay_block = 50400) -> float:
    block_difference = max(new_model_block - old_model_block, 0)
    block_adjustment = min(block_difference/decay_block, 1)
    improvement_adjustment = block_adjustment * (start_improvement-end_improvement)
    return start_improvement - improvement_adjustment

def new_model_surpasses_historical_model(new_model_metric, new_model_block, old_model_metric, old_model_block) -> bool:
    """
    It is assumed that the higher the metric value, the better the performance.

    A new model must have a performance metric that is higher than the current
    best performing model by an improvement factor.
    """
    # Return False if new model underperforms old model
    if new_model_metric <= old_model_metric:
        return False 
    
    # Otherwise, we want to calculate the improvement factor based on block differential
    improvement_factor = calculate_improvement_factor(new_model_block, old_model_block)

    # Determine the target value for the new model 
    target_value = old_model_metric + (abs(old_model_metric) * improvement_factor)

    # If the new model has performance better or equal to the improvement factor return True
    if new_model_metric >= target_value:
        return True 
    
    # Othewrwise, return False
    return False
    
def get_best_current_model_from_list(models_data: List[dict], metric_name: str, sgmse_value:float) -> dict:
    """Gets the best model submitted during today's competition for a specific metric

    Args:
        current_models_data (List[dict]): List of model performance logs 
        metric_name (str): The metric we want to find the best model for
           
    Returns:
        dict: The dictionary representing the model with the highest average value for the specified metric.
    """
    best_model = None
    highest_average = float('-inf')

    for model in models_data:
        metrics = model.get('metrics', {})
        metric_data = metrics.get(metric_name, {})

        # Ensure the metric_data contains 'average' and it is a number and that it is not identical to the SGMSE value
        if 'average' in metric_data and isinstance(metric_data['average'], (int, float, np.float64)) and metric_data['average'] != sgmse_value:
            avg = metric_data.get("average", 0)
            if avg  > highest_average:
                highest_average = avg
                best_model = model

    return best_model

def get_best_model_from_list(models_data: List[dict], metric_name: str) -> dict:
    """Gets the best model submitted during today's competition for a specific metric

    Args:
        current_models_data (List[dict]): List of model performance logs 
        metric_name (str): The metric we want to find the best model for
           
    Returns:
        dict: The dictionary representing the model with the highest average value for the specified metric.
    """
    best_model = None
    highest_average = float('-inf')

    for model in models_data:
        metrics = model.get('metrics', {})
        metric_data = metrics.get(metric_name, {})

        # Ensure the metric_data contains 'average' and it is a number
        if 'average' in metric_data and isinstance(metric_data['average'], (int, float)):
            avg = metric_data.get("average", 0)
            if avg > highest_average:
                highest_average = avg
                best_model = model

    return best_model

def find_best_model_current_benchmark(best_historical_model, current_models) -> dict:
    """
    Finds the data for the best historical model benchmarked on the current dataset.
    """
    try:
        hotkey = best_historical_model.get("hotkey", None)
        if not hotkey: 
            return None
        for model in current_models:
            model_hotkey = model.get("hotkey", None)
            if model_hotkey and model_hotkey == hotkey:
                return model 
    except:
        return None
    return None

def validate_benchmark(benchmark, metric_name, metagraph):
    """
    Validates model benchmark structure. Outputs True if formatting is correct, False if not.
    """
    # Is the benchmark a dict?
    if not isinstance(benchmark, dict):
        return False 
    
    # Are the required keys in the benchmark dict?
    required_benchmark_keys = ["metrics", "block", "hotkey"]
    for required_key in required_benchmark_keys:
        if required_key not in benchmark.keys():
            return False 

    # Is the hotkey value a string?
    hotkey = benchmark.get("hotkey", None)
    if not isinstance(hotkey, str):
        return False
    
    # Is the hotkey in the metagraph?
    if hotkey not in metagraph.hotkeys:
        return False
        
    # Is the block value an int?
    block = benchmark.get("block", None)
    if not isinstance(block, int):
        return False 
        
    # Does the metrics dict contain the metric_name?
    metrics = benchmark.get("metrics", {})
    if metric_name not in metrics.keys():
        return False 
    
    # Is there an average metric value?
    metric_values = metrics.get(metric_name, {})
    if "average" not in metric_values.keys():
        return False
    
    # Is this average value numeric?
    avg = metric_values.get("average", None)
    if not isinstance(avg, (int, float, np.float64)):
        return False
    
    return True

def validate_historical_benchmark(benchmark, metagraph):
    """
    Validates best model benchmark for a previous competition.
    """
    # Is the benchmark a dict?
    if not isinstance(benchmark, dict):
        return False 
    
    # Is the hotkey key in the dict?
    if "hotkey" not in benchmark.keys():
        return False
    
    # Is the hotkey a string?
    hotkey = benchmark.get("hotkey", None)
    if not isinstance(hotkey, str):
        return False 
    
    # Is the hotkey in the metagraph?
    if hotkey not in metagraph.hotkeys:
        return False
    
    return True

def assign_remainder_scores(
    competition_scores: np.ndarray,
    competition_max_scores: dict,
    competition: str,
    metric_proportions: dict,
    miner_models: list,
    best_model_benchmark: dict,
    metric: str, 
    metagraph: bt.metagraph,
    log_level: str,
):
    best_model_hotkey = best_model_benchmark["hotkey"]
    best_model_avg = best_model_benchmark["metrics"][metric]["average"]
    model_tracker_list = []

    for model_benchmark in miner_models:

        try:

            if validate_benchmark(
                benchmark=model_benchmark,
                metric_name=metric,
                metagraph=metagraph
            ):

                model_tracker = {}
                model_hotkey = model_benchmark.get("hotkey", None)
                model_tracker["hotkey"] = model_hotkey
                model_uid = metagraph.hotkeys.index(model_hotkey)
                model_tracker["uid"] = model_uid
                model_metrics = model_benchmark.get("metrics", None)
                if model_metrics and isinstance(model_metrics, dict):
                    model_avg = model_benchmark["metrics"][metric].get("average", None)

                    if model_hotkey and model_hotkey != best_model_hotkey and model_avg:
                        
                        performance_ratio = model_avg / best_model_avg
                        model_tracker["performance_ratio"] = performance_ratio
                        model_tracker_list.append(model_tracker)

                        Utils.subnet_logger(
                            severity="TRACE",
                            message=f"New entry in model_tracker_list for competition: {competition} and metric: {metric}: {model_tracker}",
                            log_level=log_level
                        )

                    else:
                        continue

                else:
                    continue

            else:
                continue

        except:
            continue

    try:
        if model_tracker_list:
            ratio_sum = sum([m["performance_ratio"] for m in model_tracker_list])
            remainder_key = f"{competition}_remainder"
            remainder_score = competition_max_scores[remainder_key] * metric_proportions[competition][metric]
            for m in model_tracker_list:
                score_ratio = (m["performance_ratio"] / ratio_sum) 
                score = score_ratio * remainder_score
                uid = m["uid"]
                competition_scores[uid] += score
                Utils.subnet_logger(
                    severity="TRACE",
                    message=f"Assigning remainder score of {score} to uid: {uid} for competition: {competition} and metric: {metric}",
                    log_level=log_level
                )

        Utils.subnet_logger(
            severity="TRACE",
            message=f"Scores for competition: {competition} for metric: {metric} after assigning remainder: {competition_scores}",
            log_level=log_level
        )

        return competition_scores
        
    except Exception as e:
        Utils.subnet_logger(
            severity="ERROR",
            message=f"Error when calculating remainder scores for competition: {competition} for metric: {metric}: {e}"
        )
        
    return competition_scores

def determine_competition_scores(
    competition_scores: dict, 
    competition_max_scores: dict,
    metric_proportions: dict,
    sgmse_benchmarks:dict,
    best_miner_models: dict,
    miner_models: dict,
    metagraph: bt.metagraph,
    log_level: str,
):
    """
    Determine current competition winners. For each competition, this function will:

    - Find the best-performing model in the current competition.
    - Find the best-performing model from the previous competition.
    - Find the current competition benchmark for the previous competition's best-performing model 
    - Validate the formatting of each of these benchmarks
    - Assign scores based on the following:
        1. If the best current model doesn't exist but the best historical model from the last competition does,
        assign the score to the best historical model 
        2. If neither the best current model nor the best historical model's current benchmark exist, assign no score. 
        3. If no best current model benchmark exist but there exists a benchmark for the previous competition's best model 
        benchmarked on the current dataset, assign score to best historical model
        4. If no best historical model's current benchmark exists, assign score to the best current model.
        5. If the best current model was also the best model in the last competition, assign the score to the best current model.
        6. If the best current model is different from the best historical model's current benchmark, determine if the 
        current model outperforms the historical model by a significant margin.
    """
    # Construct new log of best performing models to update as we iterate
    new_best_miner_models = {}
    for competition in competition_scores.keys():
        new_best_miner_models[competition] = []

    # Iterate through competitions
    for competition in competition_scores.keys():
        
        # Iterate through metrics in each competition
        for metric_name in metric_proportions[competition].keys():

            try:

                # Determine SGMSE benchmark value
                try:
                    sgmse_value = sgmse_benchmarks[competition][metric_name]["average"]
                except:
                    sgmse_value=0
                
                # Determine the score to assign to the best miner
                competition_metric_score = competition_max_scores[competition] * metric_proportions[competition][metric_name]
                
                # Find best current model 
                current_models = miner_models[competition]
                best_current_model = get_best_current_model_from_list(models_data=current_models, metric_name=metric_name, sgmse_value=sgmse_value)
                best_current_model_is_valid = False
                if best_current_model and validate_benchmark(benchmark=best_current_model, metric_name=metric_name, metagraph=metagraph):
                    best_current_model_is_valid = True

                Utils.subnet_logger(
                    severity="TRACE",
                    message=f"Best model for metric: {metric_name} in current competition: {competition} is: {best_current_model}",
                    log_level=log_level,
                )
                
                # Obtain best historical model 
                best_models = best_miner_models[competition]
                best_historical_model_on_previous_benchmark = get_best_model_from_list(models_data=best_models, metric_name=metric_name)
                best_historical_model_on_previous_benchmark_is_valid = False
                if best_historical_model_on_previous_benchmark and validate_historical_benchmark(benchmark=best_historical_model_on_previous_benchmark, metagraph=metagraph):
                    best_historical_model_on_previous_benchmark_is_valid = True 
                
                # Obtain best historical model on current benchmark
                best_historical_model = find_best_model_current_benchmark(best_historical_model=best_historical_model_on_previous_benchmark, current_models=current_models)
                best_historical_model_is_valid = False 
                if best_historical_model and validate_benchmark(benchmark=best_historical_model, metric_name=metric_name, metagraph=metagraph):
                    best_historical_model_is_valid = True

                # Assign score to best historical model if best current model doesn't exist
                if not best_current_model_is_valid and best_historical_model_on_previous_benchmark_is_valid:
                    best_historical_model_on_previous_benchmark_hotkey = best_historical_model_on_previous_benchmark.get("hotkey", None)
                    if not best_historical_model_on_previous_benchmark_hotkey:
                        continue
                    uid = metagraph.hotkeys.index(best_historical_model_on_previous_benchmark_hotkey)
                    competition_scores[competition][uid] += competition_metric_score
                    
                    # Append to new best performing model knowledge
                    new_best_miner_models[competition].append(best_historical_model_on_previous_benchmark)

                    Utils.subnet_logger(
                        severity="TRACE",
                        message=f"Only best historical model exists for competition: {competition}: {best_historical_model}",
                        log_level=log_level
                    )
                    continue

                # Assign no score if neither best current model or best historical model exist 
                if not best_current_model_is_valid and not best_historical_model_is_valid:
                    Utils.subnet_logger(
                        severity="TRACE",
                        message=f"No current or historical models for competition: {competition}",
                        log_level=log_level
                    )
                    continue

                # Assign score to the best historical model if the best current model does not exist
                if not best_current_model_is_valid and best_historical_model_is_valid:
                    best_historical_model_hotkey = best_historical_model.get("hotkey", None)
                    if not best_historical_model_hotkey:
                        continue 
                    uid = metagraph.hotkeys.index(best_historical_model_hotkey)
                    competition_scores[competition][uid] += competition_metric_score
                    new_best_miner_models[competition].append(best_historical_model)
                    
                    Utils.subnet_logger(
                        severity="TRACE",
                        message=f"Competition winner for metric: {metric_name} in current competition: {competition} is: {best_historical_model}. Assigning score: {competition_metric_score}",
                        log_level=log_level,
                    )

                    competition_scores[competition] = assign_remainder_scores(
                        competition_scores=competition_scores[competition],
                        competition_max_scores=competition_max_scores,
                        competition=competition,
                        metric_proportions=metric_proportions,
                        miner_models=current_models,
                        best_model_benchmark=best_historical_model,
                        metric=metric_name,
                        metagraph=metagraph,
                        log_level=log_level
                    )
                    continue
                
                # Assign score to the best current model if best historical model does not exist
                if not best_historical_model_is_valid and best_current_model_is_valid:
                    best_current_model_hotkey = best_current_model.get("hotkey", None)
                    if not best_current_model_hotkey:
                        continue
                    uid = metagraph.hotkeys.index(best_current_model_hotkey)
                    competition_scores[competition][uid] += competition_metric_score
                    new_best_miner_models[competition].append(best_current_model)
                    
                    Utils.subnet_logger(
                        severity="TRACE",
                        message=f"Competition winner for metric: {metric_name} in current competition: {competition} is: {best_current_model}. Assigning score: {competition_metric_score}",
                        log_level=log_level,
                    )

                    competition_scores[competition] = assign_remainder_scores(
                        competition_scores=competition_scores[competition],
                        competition_max_scores=competition_max_scores,
                        competition=competition,
                        metric_proportions=metric_proportions,
                        miner_models=current_models,
                        best_model_benchmark=best_current_model,
                        metric=metric_name,
                        metagraph=metagraph,
                        log_level=log_level
                    )
                    continue
                
                Utils.subnet_logger(
                    severity="TRACE",
                    message=f"Best historical model for metric: {metric_name} in current competition: {competition} is: {best_historical_model}",
                    log_level=log_level,
                )

                best_current_model_hotkey = best_current_model.get("hotkey", None)
                best_historical_model_hotkey = best_historical_model.get("hotkey", None)

                # If the best current model is the best historical model 
                if best_current_model_is_valid and best_historical_model_is_valid and best_current_model_hotkey and best_historical_model_hotkey and best_current_model_hotkey == best_historical_model_hotkey:
                    uid = metagraph.hotkeys.index(best_current_model_hotkey)
                    competition_scores[competition][uid] += competition_metric_score
                    new_best_miner_models[competition].append(best_current_model)
                    
                    Utils.subnet_logger(
                        severity="TRACE",
                        message=f"Competition winner for metric: {metric_name} in current competition: {competition} is: {best_current_model}. Assigning score: {competition_metric_score}",
                        log_level=log_level,
                    )

                    competition_scores[competition] = assign_remainder_scores(
                        competition_scores=competition_scores[competition],
                        competition_max_scores=competition_max_scores,
                        competition=competition,
                        metric_proportions=metric_proportions,
                        miner_models=current_models,
                        best_model_benchmark=best_current_model,
                        metric=metric_name,
                        metagraph=metagraph,
                        log_level=log_level
                    )
                    continue

                best_historical_model_hotkey = best_historical_model.get("hotkey", None)
                
                # Determine actual metric average values
                best_current_model_metric_value = best_current_model['metrics'][metric_name]['average']
                best_historical_model_metric_value = best_historical_model['metrics'][metric_name]['average']
                
                # Determine metadata upload block
                best_current_model_block = best_current_model['block']
                best_historical_model_block = best_historical_model['block']
            
                # Determine if new model beats historical model performance by signficiant margin
                if new_model_surpasses_historical_model(
                    new_model_metric = best_current_model_metric_value,
                    new_model_block = best_current_model_block,
                    old_model_metric = best_historical_model_metric_value,
                    old_model_block = best_historical_model_block,
                ):
                    
                    # If so, assign score to new model
                    uid = metagraph.hotkeys.index(best_current_model_hotkey)
                    competition_scores[competition][uid] += competition_metric_score
                    
                    # Append to new best performing model knowledge
                    new_best_miner_models[competition].append(best_current_model)
                    
                    Utils.subnet_logger(
                        severity="TRACE",
                        message=f"Competition winner for metric: {metric_name} in current competition: {competition} is: {best_current_model}. Assigning score: {competition_metric_score}",
                        log_level=log_level,
                    )

                    competition_scores[competition] = assign_remainder_scores(
                        competition_scores=competition_scores[competition],
                        competition_max_scores=competition_max_scores,
                        competition=competition,
                        metric_proportions=metric_proportions,
                        miner_models=current_models,
                        best_model_benchmark=best_current_model,
                        metric=metric_name,
                        metagraph=metagraph,
                        log_level=log_level
                    )
            
                # Otherwise, assign score to old model
                else: 
                    
                    uid = metagraph.hotkeys.index(best_historical_model_hotkey)
                    competition_scores[competition][uid] += competition_metric_score
                    
                    # Append to new best performing model knowledge
                    new_best_miner_models[competition].append(best_historical_model)
                    
                    Utils.subnet_logger(
                        severity="TRACE",
                        message=f"Competition winner for metric: {metric_name} in current competition: {competition} is: {best_historical_model}. Assigning score: {competition_metric_score}",
                        log_level=log_level,
                    )

                    competition_scores[competition] = assign_remainder_scores(
                        competition_scores=competition_scores[competition],
                        competition_max_scores=competition_max_scores,
                        competition=competition,
                        metric_proportions=metric_proportions,
                        miner_models=current_models,
                        best_model_benchmark=best_historical_model,
                        metric=metric_name,
                        metagraph=metagraph,
                        log_level=log_level
                    )
            
            except Exception as e:

                Utils.subnet_logger(
                    severity="ERROR",
                    message=f"Error calculating scores for metric: {metric_name} in competition: {competition}: {e}",
                    log_level=log_level
                )
                
    Utils.subnet_logger(
        severity="INFO",
        message=f"New best performing models: {new_best_miner_models}.",
        log_level=log_level,
    )
    
    Utils.subnet_logger(
        severity="INFO",
        message=f"New competition scores: {competition_scores}",
        log_level=log_level,
    )

    return new_best_miner_models, competition_scores

def calculate_overall_scores(
    competition_scores: dict, 
    scores: np.ndarray, 
    log_level: str
):
    for competition in competition_scores: 
        for i, _ in enumerate(competition_scores[competition]):
            scores[i] += competition_scores[competition][i]
    
    return scores

def filter_models_with_same_hash(new_competition_miner_models: list, hotkeys: list) -> list:
    """
    Filter out model results if there are two models with the same directory hash.
    
    We keep the model whose metadata was uploaded to the chain first.
    
    Args:
        :param new_competition_miner_models: (List[dict]): List of benchmarking results for models in current competition 
        
    Returns:
        List[dict]: Filtered list of benchmarking results for models in current competition 
    """
    # Dictionary to store the minimum 'block' for each unique 'model_hash'
    unique_models = {}

    for item in new_competition_miner_models:
        if Utils.validate_model_benchmark(item):
       
            model_hash = item['model_hash']
            block = item['block']
            hotkey = item["hotkey"]
            # Only include benchmark results where the hotkey is still registered
            if hotkey in hotkeys: 
                uid = hotkeys.index(hotkey)
            
                # Check if this model_hash is already in the unique_models dictionary
                if model_hash in unique_models:
                    # Keep the entry with the lowest 'block' value
                    if block < unique_models[model_hash]['block']:
                        unique_models[model_hash] = item
                    # If two models have the same hash and were submitted on the same block:
                    elif block == unique_models[model_hash]['block']:
                        # If the hotkey can be found 
                        if isinstance(unique_models[model_hash], dict) and "hotkey" in unique_models[model_hash].keys() and unique_models[model_hash]["hotkey"] in hotkeys:
                            # Determine UID 
                            ref_uid = hotkeys.index(unique_models[model_hash]["hotkey"])
                            # If the uid of the current model being referenced in the for loop is less than the uid of the one stored in unique_models
                            if uid < ref_uid:
                                # Update unique_models
                                unique_models[model_hash] = item
                            
                else:
                    # If model_hash not seen before, add it to unique_models
                    unique_models[model_hash] = item

    # Return a list of unique items with the lowest 'block' value for each 'model_hash'
    return list(unique_models.values())

def filter_models_with_same_metadata(new_competition_miner_models: list, hotkeys: list) -> list:
    """Filter out model results if there are two models with the same model (namspace, name, revision and class).
    
    We keep the model whose metadata was uploaded to the chain first.

    Args:
        :param new_competition_miner_models: (List[dict]): List of benchmarking results for models in current competition 
        
    Returns:
        List[dict]: Filtered list of benchmarking results for models in current competition 
    """
    unique_models = {}
    
    for item in new_competition_miner_models:
        # Verify item structure
        if Utils.validate_model_benchmark(item):
        
            model_id = f"{item['hf_model_namespace']}{item['hf_model_name']}{item['hf_model_revision']}"
            block = item['block']
            hotkey = item["hotkey"]
            # If the hotkey can be found we consider it for the filtering
            if hotkey in hotkeys:
                uid = hotkeys.index(hotkey)
            
                # Check if this model_hash is already in the unique_models dictionary
                if model_id in unique_models:
                    # Keep the entry with the lowest 'block' value
                    if block < unique_models[model_id]['block']:
                        unique_models[model_id] = item
                    # In the case that both models were submitted at the exact same block
                    elif block == unique_models[model_id]['block']:
                        # If the hotkey can be found 
                        if isinstance(unique_models[model_id], dict) and "hotkey" in unique_models[model_id].keys() and unique_models[model_id]["hotkey"] in hotkeys:
                            # Determine UID 
                            ref_uid = hotkeys.index(unique_models[model_id]["hotkey"])
                            # If the uid of the current model being referenced in the for loop is less than the uid of the one stored in unique_models
                            if uid < ref_uid:
                                unique_models[model_id] = item

                        # If not, assume the miner is deregistered and update accordingly
                        else:
                            unique_models[model_id] = item

                else:
                    # If model_hash not seen before, add it to unique_models
                    unique_models[model_id] = item
            

    # Return a list of unique items with the lowest 'block' value for each 'model_hash'
    return list(unique_models.values())

def filter_models_for_deregistered_miners(miner_models, hotkeys):
    """Removes models from list if the miner who submitted it has deregistered.

    Args:
        :param new_competition_miner_models: (List[dict]): List of new models 
        hotkeys (List[str]): List of currently registered miner hotkeys

    Returns:
        List[dict]: List of models submitted by miners with registered hotkeys
    """
    registered_models = []
    
    for model in miner_models:
        if model and isinstance(model, dict) and 'hotkey' in model.keys():
            if 'hotkey' in model.keys() and model['hotkey'] in hotkeys:
                registered_models.append(model)
            
    return registered_models

def remove_blacklist_duplicates(blacklist):
    unique_dicts=[]
    for d in blacklist:
        if isinstance(d,tuple):
            try:
                d=dict(d)
            except Exception as e:
                continue
        try:
            if d not in unique_dicts:
                unique_dicts.append(d)
        except:
            continue
    return unique_dicts