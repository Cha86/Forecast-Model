# config.py

import joblib
from collections import Counter

# Global counters and feedback dictionaries
PARAM_COUNTER = 0
POOR_PARAM_FOUND = False
EARLY_STOP_THRESHOLD = 10000  # Threshold for early stopping in Prophet
FALLBACK_THRESHOLD = 20        # Threshold for deciding between SARIMA and Prophet
SARIMA_WEIGHT = 0.4            # Weight for SARIMA in blending (example value)

forecast_params_used = {}
changepoint_counter = Counter()
seasonality_counter = Counter()
holiday_counter = Counter()
out_of_range_counter = Counter()
out_of_range_stats = {}

prophet_feedback = {}
sarima_feedback = {}
xgboost_feedback = {}
forecast_errors = {}

# Parameter histories for reward/penalty mechanisms
sarima_param_history = {}   # Key: (asin, (p,d,q,P,D,Q,m)), Value: dict with 'score', 'count', etc.
prophet_param_history = {}  # Key: (asin, (changepoint_prior_scale, seasonality_prior_scale, holidays_prior_scale)), Value: dict with 'score', 'count', etc.
