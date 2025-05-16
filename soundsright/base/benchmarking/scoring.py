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
    # Return False is new model underperforms old model
    if new_model_metric <= old_model_metric:
        return False 
    # Otherwise, we want to calculate the improvement factor based on block differential
    improvement_factor = calculate_improvement_factor(new_model_block, old_model_block)
    # If the new model has performance better or equal to the improvement factor return True
    if (new_model_metric / old_model_metric) >= (improvement_factor + 1):
        return True 
    # Othewrwise, return False
    return False
    
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
            if metric_data['average'] > highest_average:
                highest_average = metric_data['average']
                best_model = model

    return best_model

def determine_competition_scores(
    competition_scores: dict, 
    competition_max_scores: dict,
    metric_proportions: dict,
    best_miner_models: dict,
    miner_models: dict,
    metagraph: bt.metagraph,
    log_level: str,
):
    
    # Construct new log of best performing models to update as we iterate
    new_best_miner_models = {}
    for competition in competition_scores.keys():
        new_best_miner_models[competition] = []

    # Iterate through competitions
    for competition in competition_scores.keys():
        
        # Iterate through metrics in each competition
        for metric_name in metric_proportions[competition].keys():
            
            # Determine the score to assign to the best miner
            competition_metric_score = competition_max_scores[competition] * metric_proportions[competition][metric_name]
            
            # Find best current model 
            current_models = miner_models[competition]
            best_current_model = get_best_model_from_list(models_data=current_models, metric_name=metric_name)
            
            # Continue to next iteration in loop in the case that no miner models have been submitted
            if not best_current_model:
                continue
            
            Utils.subnet_logger(
                severity="TRACE",
                message=f"Best model for metric: {metric_name} in current competition: {competition} is: {best_current_model}",
                log_level=log_level,
            )
            
            # Obtain best historical model 
            best_models = best_miner_models[competition]
            best_historical_model = get_best_model_from_list(models_data=best_models, metric_name=metric_name)
            
            # Assign score to the best current model if best historical model does not exist
            if not best_historical_model:
                
                uid = metagraph.hotkeys.index(best_current_model['hotkey'])
                competition_scores[competition][uid] += competition_metric_score
                new_best_miner_models[competition].append(best_current_model)
                
                Utils.subnet_logger(
                    severity="TRACE",
                    message=f"Competition winner for metric: {metric_name} in current competition: {competition} is: {best_current_model}. Assigning score: {competition_metric_score}",
                    log_level=log_level,
                )

                continue
            
            Utils.subnet_logger(
                severity="TRACE",
                message=f"Best historical model for metric: {metric_name} in current competition: {competition} is: {best_historical_model}",
                log_level=log_level,
            )
            
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
                uid = metagraph.hotkeys.index(best_current_model['hotkey'])
                competition_scores[competition][uid] += competition_metric_score
                
                # Append to new best performing model knowledge
                new_best_miner_models[competition].append(best_current_model)
                
                Utils.subnet_logger(
                    severity="TRACE",
                    message=f"Competition winner for metric: {metric_name} in current competition: {competition} is: {best_current_model}. Assigning score: {competition_metric_score}",
                    log_level=log_level,
                )
        
            # Otherwise, assign score to old model
            else: 
                
                uid = metagraph.hotkeys.index(best_historical_model['hotkey'])
                competition_scores[competition][uid] += competition_metric_score
                
                # Append to new best performing model knowledge
                new_best_miner_models[competition].append(best_historical_model)
                
                Utils.subnet_logger(
                    severity="TRACE",
                    message=f"Competition winner for metric: {metric_name} in current competition: {competition} is: {best_historical_model}. Assigning score: {competition_metric_score}",
                    log_level=log_level,
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

def filter_models_with_same_hash(new_competition_miner_models: list) -> list:
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
    blacklisted_models = []

    for item in new_competition_miner_models:
        if item and isinstance(item, dict) and 'model_hash' in item.keys() and 'block' in item.keys() and 'hf_model_namespace' in item.keys() and 'hf_model_name' in item.keys() and 'hf_model_revision' in item.keys():
            model_hash = item['model_hash']
            block = item['block']
            
            # Check if this model_hash is already in the unique_models dictionary
            if model_hash in unique_models:
                # Keep the entry with the lowest 'block' value
                if block < unique_models[model_hash]['block']:
                    blacklist_model = unique_models[model_hash]
                    filtered_blacklist_model = {
                        'hf_model_namespace':blacklist_model['hf_model_namespace'],
                        'hf_model_name':blacklist_model['hf_model_name'],
                        'hf_model_revision':blacklist_model['hf_model_revision'],
                    }
                    blacklisted_models.append(filtered_blacklist_model)
                    unique_models[model_hash] = item
            else:
                # If model_hash not seen before, add it to unique_models
                unique_models[model_hash] = item

    # Return a list of unique items with the lowest 'block' value for each 'model_hash'
    return list(unique_models.values()), blacklisted_models

def filter_models_with_same_metadata(new_competition_miner_models: list) -> list:
    """Filter out model results if there are two models with the same model (namspace, name, revision and class).
    
    We keep the model whose metadata was uploaded to the chain first.

    Args:
        :param new_competition_miner_models: (List[dict]): List of benchmarking results for models in current competition 
        
    Returns:
        List[dict]: Filtered list of benchmarking results for models in current competition 
    """
    unique_models = {}
    blacklisted_models = []
    
    for item in new_competition_miner_models:
        if item and isinstance(item, dict) and 'block' in item.keys() and 'hf_model_namespace' in item.keys() and 'hf_model_name' in item.keys() and 'hf_model_revision' in item.keys():
        
            model_id = f"{item['hf_model_namespace']}{item['hf_model_name']}{item['hf_model_revision']}"
            block = item['block']
            
            # Check if this model_hash is already in the unique_models dictionary
            if model_id in unique_models:
                # Keep the entry with the lowest 'block' value
                if block < unique_models[model_id]['block']:
                    blacklist_model = unique_models[model_id]
                    filtered_blacklist_model = {
                        'hf_model_namespace':blacklist_model['hf_model_namespace'],
                        'hf_model_name':blacklist_model['hf_model_name'],
                        'hf_model_revision':blacklist_model['hf_model_revision'],
                    }
                    blacklisted_models.append(filtered_blacklist_model)
                    unique_models[model_id] = item
            else:
                # If model_hash not seen before, add it to unique_models
                unique_models[model_id] = item

    # Return a list of unique items with the lowest 'block' value for each 'model_hash'
    return list(unique_models.values()), blacklisted_models

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
        try:
            if d not in unique_dicts:
                unique_dicts.append(d)
        except:
            continue
    return unique_dicts