# config.py

import os
from collections import Counter

# Configuration Parameters
FALLBACK_THRESHOLD = 20  # Threshold for model selection
SARIMA_WEIGHT = 0.4      # Weight for SARIMA in ensemble

# Global Dictionaries for Feedback
forecast_params_used = {}

changepoint_counter = Counter()
seasonality_counter = Counter()
holiday_counter = Counter()
out_of_range_counter = Counter()
out_of_range_stats = {}

# Feedback Dictionaries
prophet_feedback = {}
sarima_feedback = {}
xgboost_feedback = {}
forecast_errors = {}

# Parameter History Dictionaries
sarima_param_history = {}  # key = (asin, (p,d,q,P,D,Q,m)), value = dict with 'score', 'count', etc.
prophet_param_history = {} # key = (asin, (changepoint, seasonality, holiday)), value = dict with 'score', 'count', etc.

# Early Stopping Flags
PARAM_COUNTER = 0
POOR_PARAM_FOUND = False
EARLY_STOP_THRESHOLD = 10_000
