# reward_penalty.py

from config import sarima_param_history, prophet_param_history
from config import forecast_errors, sarima_feedback, prophet_feedback, xgboost_feedback
from config import out_of_range_counter, out_of_range_stats
from config import FALLBACK_THRESHOLD, SARIMA_WEIGHT
from config import POOR_PARAM_FOUND, EARLY_STOP_THRESHOLD
import joblib

def compute_reward(mae, rmse):
    """
    Example reward function that returns higher reward for lower RMSE/MAE.
    Adjust the weighting as needed.
    """
    alpha = 0.7  # Weight for MAE
    beta = 0.3   # Weight for RMSE

    # Compute a "badness" measure
    badness = alpha * mae + beta * rmse  # Lower is better

    # Convert badness to reward (higher is better)
    reward = 1.0 / (1.0 + badness)  # Ensures reward is between 0 and 1
    return reward

def update_param_history(history_dict, asin, param_tuple, rmse, mae):
    """
    Updates the global parameter-history dictionary with a new RMSE/MAE.
    - Computes a new reward.
    - Accumulates it into 'score'.
    - Tracks how many times this param set was tried ('count').
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
        # Weighted average for the RMSE, MAE, and score
        prev = history_dict[key]
        new_count = prev['count'] + 1
        prev['avg_rmse'] = (prev['avg_rmse'] * prev['count'] + rmse) / new_count
        prev['avg_mae'] = (prev['avg_mae'] * prev['count'] + mae) / new_count
        prev['score'] = (prev['score'] * prev['count'] + reward) / new_count
        prev['count'] = new_count

def save_param_histories():
    """
    Saves the SARIMA and Prophet parameter histories to disk using joblib.
    """
    joblib.dump(sarima_param_history, "sarima_param_history.pkl")
    joblib.dump(prophet_param_history, "prophet_param_history.pkl")

def load_param_histories():
    """
    Loads the SARIMA and Prophet parameter histories from disk using joblib.
    If the files do not exist, initializes empty dictionaries.
    """
    global sarima_param_history, prophet_param_history
    try:
        sarima_param_history = joblib.load("sarima_param_history.pkl")
    except FileNotFoundError:
        sarima_param_history = {}
    try:
        prophet_param_history = joblib.load("prophet_param_history.pkl")
    except FileNotFoundError:
        prophet_param_history = {}
