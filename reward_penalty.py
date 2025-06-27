# reward_penalty.py

from config import (
    sarima_param_history,
    prophet_param_history
)
import joblib

def compute_reward(mae, rmse):
    """
    Compute a reward based on MAE and RMSE. Higher rewards for lower errors.
    """
    alpha = 0.7  # Weight for MAE
    beta = 0.3   # Weight for RMSE
    
    # Compute a "badness" measure where lower is better
    badness = alpha * mae + beta * rmse
    
    # Convert badness to reward (higher is better)
    reward = 1.0 / (1.0 + badness)  # Ensures reward is between 0 and 1
    return reward

def update_param_history(history_dict, asin, param_tuple, rmse, mae):
    """
    Update the parameter history with new RMSE and MAE values.
    """
    reward = compute_reward(mae, rmse)
    key = (asin, param_tuple)
    if key not in history_dict:
        history_dict[key] = {
            'score': reward,
            'count': 1,
            'avg_rmse': rmse,
            'avg_mae': mae
        }
    else:
        # Update weighted averages
        prev = history_dict[key]
        new_count = prev['count'] + 1
        prev['avg_rmse'] = (prev['avg_rmse'] * prev['count'] + rmse) / new_count
        prev['avg_mae'] = (prev['avg_mae'] * prev['count'] + mae) / new_count
        prev['score'] = (prev['score'] * prev['count'] + reward) / new_count
        prev['count'] = new_count

def save_param_histories():
    """
    Save the parameter histories to disk using joblib.
    """
    from config import sarima_param_history, prophet_param_history
    joblib.dump(sarima_param_history, "sarima_param_history.pkl")
    joblib.dump(prophet_param_history, "prophet_param_history.pkl")

def load_param_histories():
    """
    Load the parameter histories from disk using joblib.
    """
    from config import sarima_param_history, prophet_param_history
    try:
        sarima_param_history = joblib.load("sarima_param_history.pkl")
    except FileNotFoundError:
        sarima_param_history = {}
    try:
        prophet_param_history = joblib.load("prophet_param_history.pkl")
    except FileNotFoundError:
        prophet_param_history = {}
    return sarima_param_history, prophet_param_history
