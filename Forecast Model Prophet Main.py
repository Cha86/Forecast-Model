import pandas as pd
import numpy as np
import shap
import xgboost as xgb
from collections import Counter

from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
import matplotlib.pyplot as plt
import warnings
import os
from openpyxl import Workbook
from openpyxl.utils.dataframe import dataframe_to_rows
from statsmodels.tsa.statespace.sarimax import SARIMAX
import joblib
from sklearn.metrics import (
    mean_absolute_error,
    median_absolute_error,
    mean_absolute_error,
    mean_squared_error,
    mean_absolute_percentage_error
)
from math import sqrt
from pykalman import KalmanFilter
from statsmodels.tsa.stattools import adfuller
import time
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from scipy.optimize import minimize

warnings.filterwarnings("ignore")

forecast_params_used = {}

changepoint_counter = Counter()
seasonality_counter = Counter()
holiday_counter = Counter()
out_of_range_counter = Counter()
out_of_range_stats = {}

# Global dictionaries to store feedback for each model type
prophet_feedback = {}
sarima_feedback = {}
xgboost_feedback = {}
forecast_errors = {}

# Global dictionaries to store the accumulated performance history
sarima_param_history = {}  # key = (asin, (p,d,q,P,D,Q,m)), value = dict with 'score', 'count', etc.
prophet_param_history = {} # key = (asin, (changepoint, seasonality, holiday)), value = dict with 'score', 'count', etc.

##############################
# Reward & Penalty Functions
##############################

def compute_reward(mae, rmse):
    """
    Example reward function that returns higher reward for lower RMSE/MAE.
    Feel free to adjust or add more sophisticated weighting.
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
    - We compute a new reward
    - We accumulate it into 'score'
    - We track how many times this param set was tried ('count')
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
    joblib.dump(sarima_param_history, "sarima_param_history.pkl")
    joblib.dump(prophet_param_history, "prophet_param_history.pkl")

def load_param_histories():
    global sarima_param_history, prophet_param_history
    try:
        sarima_param_history = joblib.load("sarima_param_history.pkl")
    except:
        sarima_param_history = {}
    try:
        prophet_param_history = joblib.load("prophet_param_history.pkl")
    except:
        prophet_param_history = {}

    # PATCH any missing keys (avg_mae, avg_rmse) in old records
    for d in [sarima_param_history, prophet_param_history]:
        for (asin, param_tuple), entry in d.items():
            if 'avg_mae' not in entry:
                entry['avg_mae'] = None
            if 'avg_rmse' not in entry:
                entry['avg_rmse'] = None
    


##############################
# Configuration
##############################
# ↓↓↓ For more aggressive fallback to SARIMA, reduce threshold to 20 or another smaller number
FALLBACK_THRESHOLD = 20  # e.g., using 20% or smaller data sets for SARIMA
SARIMA_WEIGHT = 0.4      # Weight for SARIMA forecast if combining with Amazon or other model

##############################
# Counter
##############################

def record_forecast_params(product_id, changepoint_prior_scale, seasonality_prior_scale, holidays_prior_scale):
    """Record the parameters used for a specific product."""
    forecast_params_used[product_id] = {
        'changepoint_prior_scale': changepoint_prior_scale,
        'seasonality_prior_scale': seasonality_prior_scale,
        'holidays_prior_scale': holidays_prior_scale,
    }
    # Update global counters
    changepoint_counter[changepoint_prior_scale] += 1
    seasonality_counter[seasonality_prior_scale] += 1
    holiday_counter[holidays_prior_scale] += 1

def summarize_usage():
    """Summarize parameter usage for all products and the most common settings."""
    print("\n=== Forecast Parameter Usage by Product ===")
    for product_id, params in forecast_params_used.items():
        print(f"Product {product_id}:")
        print(f"  Changepoint Prior Scale: {params['changepoint_prior_scale']}")
        print(f"  Seasonality Prior Scale: {params['seasonality_prior_scale']}")
        print(f"  Holidays Prior Scale: {params['holidays_prior_scale']}")
    
    print("\n=== Most Common Parameters Across All Products ===")
    print("Most Common Changepoint Settings:")
    for value, count in changepoint_counter.most_common():
        print(f"  {value}: {count} times")
    
    print("\nMost Common Seasonality Settings:")
    for value, count in seasonality_counter.most_common():
        print(f"  {value}: {count} times")
    
    print("\nMost Common Holiday Settings:")
    for value, count in holiday_counter.most_common():
        print(f"  {value}: {count} times")
    print("================================")


##############################
# Added Functions for Caching
##############################

def save_model(model, model_type, asin, ts_data):
    model_cache_folder = "model_cache"
    os.makedirs(model_cache_folder, exist_ok=True)
    model_path = os.path.join(model_cache_folder, f"{asin}_{model_type}.pkl")
    
    # Safely set the attribute if the model object supports it
    try:
        model.last_train_date = ts_data['ds'].max()
    except AttributeError:
        print(f"Warning: Model object does not support 'last_train_date' attribute.")
    
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")

def load_model(model_type, asin):
    model_cache_folder = "model_cache"
    model_path = os.path.join(model_cache_folder, f"{asin}_{model_type}.pkl")
    exog_path = os.path.join(model_cache_folder, f"{asin}_{model_type}_exog.pkl")

    if os.path.exists(model_path):
        model = joblib.load(model_path)
        if os.path.exists(exog_path):
            exog = joblib.load(exog_path)
            return model, exog
        return model, None
    return None, None

def is_model_up_to_date(cached_model_path, ts_data):
    """Check if a cached model was trained up to the most recent data in ts_data."""
    if not os.path.exists(cached_model_path):
        return False
    model = joblib.load(cached_model_path)
    if hasattr(model, 'last_train_date'):
        last_train_date = model.last_train_date
        latest_data_date = ts_data['ds'].max()
        return last_train_date >= latest_data_date
    return False


##############################
# Kalman filter-based Missing Data Handling
##############################

def kalman_smooth(series):
    """Use a Kalman Filter to fill missing values in the series."""
    values = series.values.astype(float)
    initial_state_mean = np.nanmean(values)
    if np.isnan(initial_state_mean):
        initial_state_mean = 0.0

    kf = KalmanFilter(
        initial_state_mean=initial_state_mean,
        n_dim_obs=1,
        n_dim_state=1,
        initial_state_covariance=1,
        transition_matrices=[1],
        observation_matrices=[1],
        transition_covariance=0.1,
        observation_covariance=1.0
    )

    masked_values = np.ma.array(values, mask=np.isnan(values))
    kf = kf.em(masked_values, n_iter=5)
    smoothed_state_means, _ = kf.smooth(masked_values)

    smoothed_state_means_1d = smoothed_state_means[:, 0]
    filled_values = values.copy()
    nan_idx = np.isnan(values)
    filled_values[nan_idx] = smoothed_state_means_1d[nan_idx]

    filled_values = np.round(filled_values).astype(int)
    return pd.Series(filled_values, index=series.index)

def handle_missing_data(data):
    """Handle missing data in 'y' using Kalman smoothing if enough data points exist."""
    if len(data['y']) < 2:
        print("Not enough data points for Kalman smoothing. Using fallback method for missing data.")
        data['y'] = data['y'].fillna(method='bfill').fillna(method='ffill').astype(int)
    else:
        data['y'] = kalman_smooth(data['y'])
    return data

def handle_outliers(data):
    """Clip outliers in 'y' based on IQR."""
    Q1 = data['y'].quantile(0.25)
    Q3 = data['y'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    clipped = data['y'].clip(lower=lower_bound, upper=upper_bound).round().astype(int)
    data['y'] = clipped
    return data

def min_max_scale_data(df, feature_cols):
    """
    Apply MinMax scaling to specified feature columns.
    Returns the scaled DataFrame and a dictionary of scalers.
    """
    scalers = {}
    for col in feature_cols:
        scaler = MinMaxScaler()
        df[col] = scaler.fit_transform(df[[col]])
        scalers[col] = scaler
    return df, scalers

def preprocess_data(data):
    """
    Preprocess data by handling missing values, outliers, and (optionally) scaling if needed.
    """
    data = handle_missing_data(data)
    data = handle_outliers(data)
    # Example: scale 'y' using MinMax (optional, comment out if not desired)
    # data, scalers = min_max_scale_data(data, ['y'])
    return data


##############################
# Differencing and Stationarity
##############################

def differencing(timeseries, m):
    """Create differenced series for potential stationarity checks."""
    info = []
    tcopy = timeseries.copy()
    original = tcopy.copy()
    for i in range(3):
        original.name = f"d{i}_D0_m0"
        info.append(original)
        original = original.diff().dropna()
    for period in m:
        for j in range(3):
            d_series = info[j].diff(periods=period).dropna()
            d_series.name = f"d{j}_D1_m{period}"
            info.append(d_series)
    for period in m:
        for j in range(3):
            d_series = info[j+3].diff(periods=period).dropna()
            d_series.name = f"d{j}_D2_m{period}"
            info.append(d_series)
    df_info = pd.concat(info, axis=1)
    return df_info

def adf_summary(diff_series):
    """Perform ADF tests on multiple differenced series and return a summary DataFrame."""
    summary = []
    for col in diff_series.columns:
        series = diff_series[col].dropna()
        if len(series) < 3:
            summary.append([np.nan]*7)
            continue
        a, b, c, d, e, f = adfuller(series)
        g, h, i = e.values()
        results = [a, b, c, d, g, h, i]
        summary.append(results)
    columns = ["Test Statistic", "p-value", "#Lags Used", "No. of Obs. Used",
               "Critical Value (1%)", "Critical Value (5%)", "Critical Value (10%)"]
    index = diff_series.columns
    summary = pd.DataFrame(summary, index=index, columns=columns)
    return summary

def create_holiday_regressors(ts_data, holidays):
    """Create exogenous holiday regressors for SARIMA."""
    holiday_names = holidays['holiday'].unique()
    exog = pd.DataFrame(index=ts_data.index)
    exog['ds'] = ts_data['ds']
    for h in holiday_names:
        holiday_dates = holidays[holidays['holiday'] == h]['ds']
        exog[h] = exog['ds'].isin(holiday_dates).astype(int)
    exog.drop(columns=['ds'], inplace=True)
    return exog


##############################
# Custom SARIMA fitting
##############################

def calculate_forecast_metrics(actual, predicted):
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    mae = mean_absolute_error(actual, predicted)
    return rmse, mae

def fit_sarima_model(data, holidays, seasonal_period=52, asin=None):
    """
    Automatically fit a SARIMA model by iterating over a set of p, d, q, P, D, Q, m values.
    We store a reward for each param set in sarima_param_history, so next time 
    we can skip or penalize poor-performing sets for the same ASIN.
    
    Returns:
        (best_model, best_params)
    """
    exog = create_holiday_regressors(data, holidays)
    sample_size = len(data)
    if sample_size < 52:
        if sample_size >= 12:
            seasonal_period = 12
        elif sample_size >= 4:
            seasonal_period = None
        else:
            seasonal_period = 1

    p_values = [0, 1, 2]
    d_values = [0, 1]
    q_values = [0, 1, 2]
    P_values = [0, 1]
    D_values = [0, 1]
    Q_values = [0, 1]

    if seasonal_period is None:
        m_values = [1]
        seasonal = False
    else:
        m_values = [seasonal_period]
        seasonal = True

    best_rmse = float('inf')
    best_model = None
    best_metrics = None

    skip_threshold_score = 0.001  # skip param sets with < 0.001 historical score

    for p in p_values:
        for d in d_values:
            for q in q_values:
                if seasonal:
                    for P in P_values:
                        for D in D_values:
                            for Q in Q_values:
                                for m in m_values:
                                    param_tuple = (p, d, q, P, D, Q, m)
                                    if asin is not None:
                                        key = (asin, param_tuple)
                                        if key in sarima_param_history:
                                            if sarima_param_history[key]['score'] < skip_threshold_score:
                                                continue
                                    try:
                                        model = SARIMAX(
                                            data['y'],
                                            order=(p, d, q),
                                            seasonal_order=(P, D, Q, m),
                                            exog=exog if not exog.empty else None,
                                            enforce_stationarity=False,
                                            enforce_invertibility=False
                                        )
                                        model_fit = model.fit(disp=False)
                                        forecast = model_fit.fittedvalues
                                        actual = data['y']
                                        rmse, mae = calculate_forecast_metrics(actual, forecast)

                                        if asin is not None:
                                            update_param_history(
                                                sarima_param_history, 
                                                asin, 
                                                param_tuple, 
                                                rmse, 
                                                mae
                                            )
                                        if rmse < best_rmse:
                                            best_rmse = rmse
                                            best_model = model_fit
                                            best_metrics = {
                                                'RMSE': rmse,
                                                'MAE': mae,
                                                'p': p, 'd': d, 'q': q,
                                                'P': P, 'D': D, 'Q': Q, 'm': m
                                            }
                                    except:
                                        continue
                else:
                    try:
                        param_tuple = (p, d, q, 0, 0, 0, 1)
                        if asin is not None:
                            key = (asin, param_tuple)
                            if key in sarima_param_history:
                                if sarima_param_history[key]['score'] < skip_threshold_score:
                                    print(f"Skipping SARIMA params {param_tuple} for ASIN {asin} due to low historical score.")
                                    continue
                        model = SARIMAX(
                            data['y'],
                            order=(p, d, q),
                            seasonal_order=(0, 0, 0, 1),
                            exog=exog if not exog.empty else None,
                            enforce_stationarity=False,
                            enforce_invertibility=False
                        )
                        model_fit = model.fit(disp=False)
                        forecast = model_fit.fittedvalues
                        actual = data['y']
                        rmse, mae = calculate_forecast_metrics(actual, forecast)

                        if asin is not None:
                            update_param_history(
                                sarima_param_history, 
                                asin, 
                                param_tuple, 
                                rmse, 
                                mae
                            )
                        if rmse < best_rmse:
                            best_rmse = rmse
                            best_model = model_fit
                            best_metrics = {
                                'RMSE': rmse,
                                'MAE': mae,
                                'p': p, 'd': d, 'q': q,
                                'P': 0, 'D': 0, 'Q': 0, 'm': 1
                            }
                    except:
                        continue

    if best_model is None:
        print("No suitable SARIMA model found.")
        return None, None
    else:
        if best_metrics is not None:
            print(
                f"Best SARIMA Model for {asin}: "
                f"(p,d,q)=({best_metrics['p']},{best_metrics['d']},{best_metrics['q']}), "
                f"(P,D,Q,m)=({best_metrics['P']},{best_metrics['D']},{best_metrics['Q']},{best_metrics['m']}), "
                f"RMSE={best_metrics['RMSE']:.2f}, MAE={best_metrics['MAE']:.2f}"
            )
        return best_model, (best_metrics['p'], 
                            best_metrics['d'], 
                            best_metrics['q'], 
                            best_metrics['P'], 
                            best_metrics['D'], 
                            best_metrics['Q'], 
                            best_metrics['m'])



def sarima_forecast(model_fit, steps, last_date, exog=None):
    """Generate SARIMA forecasts for the specified number of steps ahead."""
    k_exog = model_fit.model.k_exog
    if exog is not None and k_exog > 0:
        # Align exog columns if needed
        exog = exog.iloc[:, :k_exog]

    forecast_values = model_fit.forecast(steps=steps, exog=exog)
    forecast_values = forecast_values.round().astype(int)
    future_dates = pd.date_range(start=last_date + pd.Timedelta(weeks=1), periods=steps, freq='W')
    # Instead of "SARIMA Forecast", unify to "MyForecast":
    return pd.DataFrame({'ds': future_dates, 'MyForecast': forecast_values})

def generate_future_exog(holidays, steps, last_date):
    """Generate exogenous holiday regressors for future forecasts."""
    future_dates = pd.date_range(start=last_date + pd.Timedelta(weeks=1), periods=steps, freq='W')
    exog_future = pd.DataFrame(index=future_dates)
    holiday_names = holidays['holiday'].unique()
    for h in holiday_names:
        exog_future[h] = exog_future.index.isin(holidays[holidays['holiday'] == h]['ds'])
    exog_future = exog_future.astype(int)
    return exog_future

def choose_forecast_model(ts_data, threshold=FALLBACK_THRESHOLD, holidays=None):
    """
    Basic decision for model selection:
    - If dataset size <= threshold, use SARIMA.
    - Else, default to Prophet.
    """
    # Extract ASIN before fitting the model
    asin = ts_data['asin'].iloc[0]

    if len(ts_data) <= threshold:
        print(f"Dataset size ({len(ts_data)}) is <= threshold ({threshold}). Using SARIMA.")
        
        # Unpack the tuple returned by fit_sarima_model
        fitted_model, best_params = fit_sarima_model(ts_data, holidays, seasonal_period=52, asin=asin)

        if fitted_model is not None:
            save_model(fitted_model, "SARIMA", asin, ts_data)
            return fitted_model, "SARIMA"
        else:
            print(f"SARIMA model fitting failed for {asin}, skipping.")
            return None, "SARIMA_Failed"
    else:
        print(f"Dataset size ({len(ts_data)}) is > threshold ({threshold}). Using Prophet.")
        return None, "Prophet"

def generate_date_from_week(row):
    """Convert year-week format into a datetime object for the beginning of that week."""
    week_str = row['week']
    year = row['year']
    week_number = int(week_str[1:])
    return pd.to_datetime(f'{year}-W{week_number - 1}-0', format='%Y-W%U-%w')

def clean_weekly_sales_data(data):
    """Placeholder for additional cleaning steps if needed."""
    return data

def load_weekly_sales_data(file_path):
    """Load weekly sales data from Excel, ensuring required columns are present."""
    data = pd.read_excel(file_path)
    data.columns = data.columns.str.strip().str.lower()

    required_columns = ['product title', 'week', 'year', 'units_sold', 'asin']
    missing_required = [col for col in required_columns if col not in data.columns]
    if missing_required:
        raise ValueError(f"Missing required columns: {missing_required}")

    data['ds'] = data.apply(generate_date_from_week, axis=1)
    data = data.rename(columns={'units_sold': 'y'})
    data['y'] = data['y'].astype(int)
    data = clean_weekly_sales_data(data)
    return data

def load_asins_to_forecast(file_path):
    """Load a list of ASINs from either a text or Excel file."""
    if file_path.endswith('.txt'):
        with open(file_path, 'r') as file:
            asins = [line.strip() for line in file if line.strip()]
    elif file_path.endswith('.xlsx') or file_path.endswith('.xls'):
        df = pd.read_excel(file_path)
        asins = df.iloc[:, 0].dropna().astype(str).tolist()
    else:
        raise ValueError("Unsupported file format for ASINs to Forecast file.")
    return asins

def load_amazon_forecasts_from_folder(folder_path, asin):
    """
    Load Amazon forecast data from multiple Excel files in a folder.
    Each file should correspond to a specific forecast type (Mean, P70, etc.).
    """
    forecast_data = {}
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.xlsx'):
            # Extract forecast type from filename, e.g., 'Mean Forecast.xlsx' -> 'Mean'
            forecast_type = os.path.splitext(file_name)[0].replace('_', ' ').title()
            if forecast_type.endswith(' Forecast'):
                forecast_type = forecast_type.replace(' Forecast', '')
            file_path = os.path.join(folder_path, file_name)
            df = pd.read_excel(file_path)
    
            df.columns = df.columns.str.strip().str.upper()
            if 'ASIN' not in df.columns:
                print(f"Error: Column 'ASIN' not found in {file_name}")
                continue
    
            asin_row = df[df['ASIN'].str.upper() == asin.upper()]
            if asin_row.empty:
                print(f"Warning: ASIN {asin} not found in {file_name}")
                continue
    
            week_columns = [col for col in df.columns if 'W' in col.upper()]
            if not week_columns:
                print(f"Warning: No week columns found in {file_name}")
                continue
    
            forecast_values = (
                asin_row.iloc[0][week_columns]
                .astype(str)
                .str.replace(',', '')
                .astype(float)
                .round()
                .astype(int)
                .values
            )
            forecast_data[forecast_type] = forecast_values
    return forecast_data

def add_lag_features(ts_data, lag_weeks=1):
    """Add lag features to the time series DataFrame."""
    ts_data = ts_data.copy()
    ts_data = ts_data.sort_values('ds')
    ts_data[f'lag_{lag_weeks}_week'] = ts_data['y'].shift(lag_weeks)
    ts_data[f'lag_{lag_weeks}_week'].fillna(0, inplace=True)
    return ts_data

def prepare_time_series_with_lags(data, asin, lag_weeks=1):
    """
    Prepare a single ASIN's data for modeling:
    - rename columns
    - sort by date
    - preprocess (missing/outliers)
    - add lag features
    """
    ts_data = data[data['asin'] == asin].copy()
    ts_data = ts_data.sort_values('ds')
    ts_data = preprocess_data(ts_data)
    ts_data = add_lag_features(ts_data, lag_weeks)
    return ts_data

def get_shifted_holidays():
    """
    Example holiday DataFrame with 'ds' as date and 'holiday' as holiday name.
    Adjust to match your actual holiday data.
    """
    holidays_list = [
        ('Prime Day', '2023-06-27'),
        ('Prime Day', '2023-06-28'),
        ('Black Friday', '2023-11-10'),
        ('Cyber Monday', '2023-11-13'),
        ('Christmas', '2023-12-11'),
        ('Prime Day', '2024-07-02'),
        ('Prime Day', '2024-07-03'),
        ('Black Friday', '2024-11-15'),
        ('Cyber Monday', '2024-11-18'),
        ('Christmas', '2024-12-09'),
    ]
    holidays = pd.DataFrame(holidays_list, columns=['holiday', 'ds'])
    holidays['ds'] = pd.to_datetime(holidays['ds'])
    return holidays


##############################
# XGBoost Training and SHAP Feature Importance
##############################

def train_xgboost(ts_data, target='y', features=None):
    """
    Train an XGBoost model on the provided ts_data, using specified features for regression.
    Returns the trained model, feature names, and SHAP values for importance.
    """
    if features is None:
        # Default to just the lag feature if not specified
        features = [col for col in ts_data.columns if col.startswith('lag_')]

    # Drop rows where target or features are NaN
    valid_data = ts_data.dropna(subset=[target] + features)
    if valid_data.empty:
        print("No valid data available for XGBoost training.")
        return None, None, None

    X = valid_data[features]
    y = valid_data[target]

    # Basic train/test split (can be replaced with walk-forward if needed)
    split_idx = int(len(valid_data) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    model = xgb.XGBRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )
    model.fit(X_train, y_train, 
              eval_set=[(X_test, y_test)], 
              verbose=False)

    explainer = shap.Explainer(model)
    shap_values = explainer(X)

    # Suppress the SHAP plot in code (no display in many environments)
    shap.summary_plot(shap_values, X, show=False)

    return model, features, shap_values

def xgboost_forecast(model, ts_data, forecast_steps=16, target='y', features=None):
    """
    Generate XGBoost forecasts for the specified future steps using last known features as the basis.
    We unify the forecast column to "MyForecast".
    """
    if features is None:
        features = [col for col in ts_data.columns if col.startswith('lag_')]

    ts_data = ts_data.sort_values('ds').copy()
    last_date = ts_data['ds'].max()

    future_dates = pd.date_range(start=last_date + pd.Timedelta(weeks=1), periods=forecast_steps, freq='W')
    forecasts = []

    current_data = ts_data.copy()

    for i in range(forecast_steps):
        X_pred = current_data.iloc[[-1]][features]
        y_pred = model.predict(X_pred)[0]
        y_pred = max(int(round(y_pred)), 0)

        forecast_date = future_dates[i]
        forecasts.append((forecast_date, y_pred))

        for feat in features:
            if 'lag_' in feat:
                lag_num = int(feat.split('_')[1])
                if lag_num == 1:
                    current_data.loc[current_data.index[-1], feat] = y_pred
                else:
                    prev_feat = f'lag_{lag_num-1}_week'
                    current_data.loc[current_data.index[-1], feat] = current_data.loc[current_data.index[-1], prev_feat]

        new_row = current_data.iloc[[-1]].copy()
        new_row['ds'] = forecast_date
        new_row[target] = y_pred
        current_data = pd.concat([current_data, new_row], ignore_index=True)

    df_forecasts = pd.DataFrame(forecasts, columns=['ds', 'MyForecast'])
    return df_forecasts


##############################
# Ensemble Approach (SARIMA, Prophet, XGBoost)
##############################

def ensemble_forecast(sarima_preds, prophet_preds, xgb_preds, amazon_mean_preds, weights):
    """
    Weighted ensemble of forecasts from SARIMA, Prophet, XGBoost, and Amazon Mean.
    weights: [w_sarima, w_prophet, w_xgb, w_mean]
    """
    return (
        weights[0] * sarima_preds
        + weights[1] * prophet_preds
        + weights[2] * xgb_preds
        + weights[3] * amazon_mean_preds
    )

def evaluate_forecast(actual, forecast):
    """Compute RMSE for forecast evaluation."""
    rmse = sqrt(mean_squared_error(actual, forecast))
    return rmse

def walk_forward_validation_ensemble(ts_data, n_test, model_sarima, model_prophet, model_xgb, weights):
    """
    Example walk-forward validation for an ensemble approach.
    """
    predictions = []
    actuals = []
    train = ts_data.iloc[:-n_test].copy()
    test = ts_data.iloc[-n_test:].copy()

    for i in range(n_test):
        test_point = ts_data.iloc[len(train) + i]

        # SARIMA forecast for next step
        if model_sarima is not None:
            exog_test = None
            sarima_value = model_sarima.forecast(steps=1, exog=exog_test)
            sarima_value = int(round(sarima_value.iloc[0]))
        else:
            sarima_value = 0

        # Prophet forecast for next step
        if model_prophet is not None:
            future = pd.DataFrame({'ds': [test_point['ds']]})
            prophet_predict = model_prophet.predict(future)['yhat'].values[0]
            prophet_value = int(round(max(prophet_predict, 0)))
        else:
            prophet_value = 0

        # XGBoost forecast for next step
        if model_xgb is not None:
            x_test = test_point[[c for c in ts_data.columns if c.startswith('lag_')]].values.reshape(1, -1)
            xgb_value = model_xgb.predict(x_test)[0]
            xgb_value = int(round(max(xgb_value, 0)))
        else:
            xgb_value = 0

        # Simple approach: ignoring Amazon in this snippet
        ensemble_pred = sarima_value
        ensemble_pred = int(round(ensemble_pred))
        predictions.append(ensemble_pred)
        actuals.append(test_point['y'])

    rmse = evaluate_forecast(actuals, predictions)
    return rmse, predictions, actuals

def print_ensemble_summary(rmse, predictions, actuals):
    """Display a user-friendly summary of the ensemble results."""
    print("\n=== Ensemble Forecast Summary ===")
    print(f"Number of Observations: {len(actuals)}")
    print(f"RMSE: {rmse:.2f}")
    mae_val = mean_absolute_error(actuals, predictions)
    mape_val = mean_absolute_percentage_error(actuals, predictions)*100
    print(f"MAE: {mae_val:.2f}")
    print(f"MAPE: {mape_val:.2f}%")
    print("=================================\n")

def create_decision_matrix(sarima_rmse, prophet_rmse, xgb_rmse, ensemble_rmse):
    """
    Create a simple decision matrix showing which model performed best.
    """
    data = {
        'Model': ['SARIMA', 'Prophet', 'XGBoost', 'Ensemble'],
        'RMSE': [sarima_rmse, prophet_rmse, xgb_rmse, ensemble_rmse]
    }
    df = pd.DataFrame(data)
    df.sort_values(by='RMSE', inplace=True)
    return df


##############################
# Prophet Customization
##############################

def forecast_with_custom_params(ts_data, forecast_data,
                                changepoint_prior_scale,
                                seasonality_prior_scale,
                                holidays_prior_scale,
                                horizon=16,
                                weights=None):
    """
    Train and forecast with Prophet using user-defined hyperparams and custom weighting for regressors.
    We'll unify the final forecast column to "MyForecast".
    """
    # Default weights if not provided
    weights = weights or {
        'Amazon Mean': 0.7,
        'Amazon P70': 0.2,
        'Amazon P80': 0.1
    }

    future_dates = pd.date_range(start=ts_data['ds'].max() + pd.Timedelta(days=7), periods=horizon, freq='W')
    future = pd.DataFrame({'ds': future_dates})
    combined_df = pd.concat([ts_data, future], ignore_index=True)

    # Attach forecast_data (Amazon forecasts) as regressors
    for forecast_type, values in forecast_data.items():
        values_to_use = values[:horizon] if len(values) > horizon else values
        extended_values = np.concatenate(
            [
                np.full(len(ts_data), np.nan),
                values_to_use,
                np.full(max(horizon - len(values_to_use), 0),
                        values_to_use[-1] if len(values_to_use) > 0 else 0)
            ]
        )
        extended_values = extended_values[:len(combined_df)]
        combined_df[f'Amazon_{forecast_type} Forecast'] = extended_values

    regressor_cols = [col for col in combined_df.columns if col.startswith('Amazon_')]
    combined_df[regressor_cols] = combined_df[regressor_cols].fillna(0)

    holidays = get_shifted_holidays()
    combined_df['prime_day'] = combined_df['ds'].apply(
        lambda x: 0.2 if x in holidays[holidays['holiday'] == 'Prime Day']['ds'].values else 0
    )

    n = len(ts_data)
    split = int(n * 0.8)
    train_df = combined_df.iloc[:split].dropna(subset=['y']).copy()
    test_df = combined_df.iloc[split:].copy()

    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        seasonality_mode='multiplicative',
        holidays=holidays,
        changepoint_prior_scale=changepoint_prior_scale,
        seasonality_prior_scale=seasonality_prior_scale,
        holidays_prior_scale=holidays_prior_scale,
        growth='linear'
    )

    # Add regressors dynamically
    for regressor in regressor_cols + ['prime_day']:
        model.add_regressor(regressor, mode='multiplicative')

    model.fit(train_df[['ds', 'y'] + regressor_cols + ['prime_day']])

    # Evaluate on test split
    test_forecast = model.predict(test_df.drop(columns='y').copy())
    test_actual = combined_df.iloc[split:].dropna(subset=['y'])
    if not test_actual.empty:
        test_eval = test_forecast[test_forecast['ds'].isin(test_actual['ds'])]
        # We'll do a rough integer rounding for this test evaluation:
        test_eval['MyForecast'] = test_eval['yhat'].round().astype(int)
        mae = mean_absolute_error(test_actual['y'], test_eval['MyForecast'])
        rmse_val = sqrt(mean_squared_error(test_actual['y'], test_eval['MyForecast']))
        print(f"Prophet Test MAE: {mae:.4f}, RMSE: {rmse_val:.4f}")

    future_df = combined_df[combined_df['y'].isna()].drop(columns='y').copy()
    forecast = model.predict(future_df)

    # Apply custom weighting mechanism to adjust forecasts
    for amazon_col, weight in weights.items():
        if f'Amazon_{amazon_col} Forecast' in future_df.columns:
            forecast['yhat'] += weight * future_df[f'Amazon_{amazon_col} Forecast']

    forecast['MyForecast'] = forecast['yhat'].round().astype(int).clip(lower=0)

    return forecast[['ds', 'MyForecast', 'yhat', 'yhat_upper']], model


##############################
# Prophet Parameter Optimization
##############################

PARAM_COUNTER = 0
POOR_PARAM_FOUND = False
EARLY_STOP_THRESHOLD = 10_000

def optimize_prophet_params(ts_data, forecast_data, param_grid, horizon=16, asin=None):
    global PARAM_COUNTER, POOR_PARAM_FOUND, prophet_feedback

    best_rmse = float('inf')
    best_params = None
    best_rmse_values = None

    skip_threshold_score = 0.001  # if average reward < this, skip trying it again

    for changepoint_prior_scale in param_grid['changepoint_prior_scale']:
        for seasonality_prior_scale in param_grid['seasonality_prior_scale']:
            for holidays_prior_scale in param_grid['holidays_prior_scale']:
                PARAM_COUNTER += 1
                param_tuple = (changepoint_prior_scale, seasonality_prior_scale, holidays_prior_scale)

                # If we have a bad param in prophet_param_history, skip it
                if asin is not None:
                    key = (asin, param_tuple)
                    if key in prophet_param_history:
                        if prophet_param_history[key]['score'] < skip_threshold_score:
                            print(f"Skipping prophet params {param_tuple} for ASIN {asin} due to low historical score.")
                            continue

                print(f"Testing Params #{PARAM_COUNTER} for {asin}: changepoint={changepoint_prior_scale}, "
                      f"seasonality={seasonality_prior_scale}, holidays={holidays_prior_scale}")
                try:
                    forecast, _ = forecast_with_custom_params(
                        ts_data, forecast_data,
                        changepoint_prior_scale,
                        seasonality_prior_scale,
                        holidays_prior_scale,
                        horizon=horizon
                    )
                    if forecast.empty or 'MyForecast' not in forecast.columns:
                        raise ValueError("Forecast failed to generate 'MyForecast' column.")

                    # Calculate average RMSE across the Amazon forecast streams
                    rmse_values = calculate_rmse(forecast, forecast_data, horizon)
                    avg_rmse = np.mean(list(rmse_values.values()))

                    mae_val = avg_rmse

                    # Update the param history with reward
                    if asin is not None:
                        update_param_history(
                            prophet_param_history,
                            asin,
                            param_tuple,
                            avg_rmse,
                            mae_val
                        )

                    if avg_rmse < best_rmse:
                        best_rmse = avg_rmse
                        best_params = {
                            'changepoint_prior_scale': changepoint_prior_scale,
                            'seasonality_prior_scale': seasonality_prior_scale,
                            'holidays_prior_scale': holidays_prior_scale
                        }
                        best_rmse_values = rmse_values

                    if avg_rmse > EARLY_STOP_THRESHOLD:
                        print("Early stopping triggered due to poor parameter performance.")
                        POOR_PARAM_FOUND = True
                        return best_params, best_rmse_values

                except Exception as e:
                    print(f"Error during optimization: {e}")
                    continue

    if best_params is None:
        print(f"Prophet optimization failed for {asin}. Using default parameters.")
        best_params = {
            'changepoint_prior_scale': 0.1,
            'seasonality_prior_scale': 1,
            'holidays_prior_scale': 10
        }
        best_rmse_values = {}

    # Record feedback for Prophet
    asin_label = asin if asin is not None else 'unknown'
    prophet_feedback[asin_label] = {
        'best_params': best_params,
        'rmse_values': best_rmse_values,
        'total_tests': PARAM_COUNTER
    }
    return best_params, best_rmse_values



def calculate_rmse(forecast, forecast_data, horizon):
    """Calculate RMSE comparing our 'MyForecast' vs. Amazon forecast streams."""
    rmse_values = {}
    for forecast_type, values in forecast_data.items():
        values = values[:horizon]
        if len(values) < horizon:
            if len(values) > 0:
                values = np.pad(values, (0, horizon - len(values)), 'constant', constant_values=values[-1])
            else:
                values = np.zeros(horizon, dtype=int)
        rmse = np.sqrt(((forecast['MyForecast'] - values) ** 2).mean())
        rmse_values[forecast_type] = rmse
    return rmse_values


##############################
# Forecast Formatting & Adjustments
##############################

def format_output_with_forecasts(my_forecast_df, forecast_data, horizon=16):
    """
    Combine our 'MyForecast' with Amazon forecast data for comparison.
    Also compute the difference and percentage difference columns.
    """
    comparison = my_forecast_df.copy()
    for forecast_type, values in forecast_data.items():
        values = values[:horizon]
        values = np.array(values, dtype=int)
        if 'Forecast' in forecast_type:
            amazon_col_name = f'Amazon {forecast_type}'
        else:
            amazon_col_name = f'Amazon {forecast_type} Forecast'
        forecast_df = pd.DataFrame({
            'ds': my_forecast_df['ds'],
            amazon_col_name: values
        })
        comparison = comparison.merge(forecast_df, on='ds', how='left')
    comparison.fillna(0, inplace=True)

    for col in comparison.columns:
        if col.startswith('Amazon ') and col.endswith('Forecast'):
            base_name = col.replace('Amazon ', '').replace(' Forecast', '')
            diff_col = f"Diff_{base_name}"
            pct_col = f"Pct_{base_name}"
            comparison[diff_col] = (comparison['MyForecast'] - comparison[col]).round().astype(int)
            comparison[pct_col] = np.where(
                comparison[col] != 0,
                (comparison[diff_col] / comparison[col]) * 100,
                0
            )

    comparison['MyForecast'] = comparison['MyForecast'].round().astype(int)

    for col in comparison.columns:
        if col.startswith("Amazon ") or col.startswith("Diff_") or col.startswith("Pct_"):
            comparison[col] = comparison[col].astype(int, errors='ignore')
    return comparison

def adjust_forecast_weights(forecast, yhat_weight, yhat_upper_weight):
    """Adjust final MyForecast by combining yhat and yhat_upper with specified weights."""
    if 'yhat' not in forecast or 'yhat_upper' not in forecast:
        raise KeyError("'yhat' or 'yhat_upper' not found in forecast DataFrame.")
    adj_forecast = (yhat_weight * forecast['yhat'] + yhat_upper_weight * forecast['yhat_upper']).clip(lower=0)
    adj_forecast = adj_forecast.round().astype(int)
    forecast['MyForecast'] = adj_forecast
    return forecast

def find_best_forecast_weights(forecast, comparison, weights):
    """
    Brute force search over (yhat_weight, yhat_upper_weight) pairs to minimize average RMSE vs. Amazon columns.
    """
    best_rmse = float('inf')
    best_weights = None
    rmse_results = {}
    for yhat_weight, yhat_upper_weight in weights:
        adjusted_forecast = adjust_forecast_weights(forecast.copy(), yhat_weight, yhat_upper_weight)
        rmse_values = {}
        for amazon_col in comparison.columns:
            if amazon_col.startswith('Amazon '):
                rmse = np.sqrt(((comparison[amazon_col] - adjusted_forecast['MyForecast']) ** 2).mean())
                rmse_values[amazon_col] = rmse
        avg_rmse = np.mean(list(rmse_values.values()))
        rmse_results[(yhat_weight, yhat_upper_weight)] = avg_rmse
        if avg_rmse < best_rmse:
            best_rmse = avg_rmse
            best_weights = (yhat_weight, yhat_upper_weight)
    return best_weights, rmse_results

def auto_find_best_weights(forecast, comparison, step=0.05):
    """
    Auto-scan [0.5, 1.0] to find best combination of yhat/yhat_upper weights for final "MyForecast".
    """
    best_rmse = float('inf')
    best_weights = None
    candidates = np.arange(0.5, 1.0 + step, step)
    for w in candidates:
        yhat_weight = w
        yhat_upper_weight = 1 - w
        adjusted_forecast = adjust_forecast_weights(forecast.copy(), yhat_weight, yhat_upper_weight)
        rmse_values = {}
        for amazon_col in comparison.columns:
            if amazon_col.startswith('Amazon '):
                rmse = np.sqrt(((comparison[amazon_col] - adjusted_forecast['MyForecast']) ** 2).mean())
                rmse_values[amazon_col] = rmse
        avg_rmse = np.mean(list(rmse_values.values()))
        if avg_rmse < best_rmse:
            best_rmse = avg_rmse
            best_weights = (yhat_weight, yhat_upper_weight)
    return best_weights, best_rmse


##############################
# Cross Validation
##############################

def try_cross_validation_with_fallback(model, ts_data, horizons, initial='365 days', period='180 days'):
    """
    Attempt multiple horizon values for cross-validation until one works or all fail.
    """
    for horizon in horizons:
        try:
            df_cv = cross_validation(model, initial=initial, period=period, horizon=horizon)
            df_performance = performance_metrics(df_cv)
            print(f"Cross-validation successful with horizon={horizon}")
            return df_cv, df_performance
        except ValueError as e:
            if "Less data than horizon" in str(e):
                print(f"Not enough data for horizon={horizon}, trying smaller horizon.")
            else:
                print(f"Error with horizon={horizon}: {e}")
    print("The ASIN has too little data for cross-validation with any tested horizons.")
    return None, None

def validate_best_params(ts_data, best_params, initial='365 days', period='180 days'):
    """
    Validate the chosen best_params via cross-validation, if possible.
    """
    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        seasonality_mode='multiplicative',
        changepoint_prior_scale=best_params['changepoint_prior_scale'],
        seasonality_prior_scale=best_params['seasonality_prior_scale'],
        holidays_prior_scale=best_params['holidays_prior_scale']
    )
    model.fit(ts_data[['ds','y']])
    candidate_horizons = ['180 days', '90 days', '60 days', '30 days']
    df_cv, df_performance = try_cross_validation_with_fallback(model, ts_data, candidate_horizons, initial=initial, period=period)
    if df_cv is not None and df_performance is not None:
        print("Validation Metrics with chosen best_params:")
        print(df_performance)
        return True
    else:
        print("No valid cross-validation results due to insufficient data.")
        return False


##############################
# Summary Statistics & Visualization
##############################

def calculate_summary_statistics(ts_data, forecast_df, horizon):
    """
    Calculate basic summary stats for historical data and partial sums for 'MyForecast'.
    """
    summary_stats = {
        "min": ts_data["y"].min(),
        "max": ts_data["y"].max(),
        "mean": ts_data["y"].mean(),
        "median": ts_data["y"].median(),
        "std_dev": ts_data["y"].std(),
        "total_sales": ts_data["y"].sum(),
        "data_range": (ts_data["ds"].min(), ts_data["ds"].max())
    }

    total_forecast_16 = forecast_df['MyForecast'][:16].sum() if len(forecast_df) >= 16 else forecast_df['MyForecast'].sum()
    total_forecast_8 = forecast_df['MyForecast'][:8].sum() if len(forecast_df) >= 8 else forecast_df['MyForecast'].sum()
    total_forecast_4 = forecast_df['MyForecast'][:4].sum() if len(forecast_df) >= 4 else forecast_df['MyForecast'].sum()

    max_forecast = forecast_df['MyForecast'].max()
    min_forecast = forecast_df['MyForecast'].min()
    max_week = forecast_df.loc[forecast_df['MyForecast'].idxmax(), 'ds'] if 'ds' in forecast_df.columns else None
    min_week = forecast_df.loc[forecast_df['MyForecast'].idxmin(), 'ds'] if 'ds' in forecast_df.columns else None
    return summary_stats, total_forecast_16, total_forecast_8, total_forecast_4, max_forecast, min_forecast, max_week, min_week

def visualize_forecast_with_comparison(ts_data, comparison, summary_stats,
                                       total_forecast_16, total_forecast_8, total_forecast_4,
                                       max_forecast, min_forecast, max_week, min_week,
                                       asin, product_title, folder_name):
    """
    Visualize historical data vs. forecast (MyForecast + Amazon) and annotate with summary stats.
    """
    fig, ax = plt.subplots(figsize=(16, 10))
    ax.plot(ts_data['ds'], ts_data['y'], label='Historical Sales', marker='o', color='black')
    ax.plot(comparison['ds'], comparison['MyForecast'], label='MyForecast', marker='o', linestyle='--', color='blue')

    for column in comparison.columns:
        if column.startswith('Amazon ') and column.endswith('Forecast'):
            ax.plot(comparison['ds'], comparison[column], label=f'{column}', linestyle=':')

    ax.set_title(f'Sales Forecast Comparison for {product_title}')
    ax.set_xlabel('Date')
    ax.set_ylabel('Units Sold')
    ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1))
    ax.grid()

    summary_text = (
        f"Historical Summary:\n"
        f"Range: {summary_stats['data_range'][0].strftime('%Y-%m-%d')} to {summary_stats['data_range'][1].strftime('%Y-%m-%d')}\n"
        f"Min: {summary_stats['min']:.0f}\n"
        f"Max: {summary_stats['max']:.0f}\n"
        f"Mean: {summary_stats['mean']:.0f}\n"
        f"Median: {summary_stats['median']:.0f}\n"
        f"Std Dev: {summary_stats['std_dev']:.0f}\n"
        f"Total Historical Sales: {summary_stats['total_sales']:.0f} units\n\n"
        f"Forecast Summary:\n"
        f"Total Forecast (16 Weeks): {total_forecast_16:.0f}\n"
        f"Total Forecast (8 Weeks): {total_forecast_8:.0f}\n"
        f"Total Forecast (4 Weeks): {total_forecast_4:.0f}\n"
        f"Max Forecast: {max_forecast:.0f} (Week of {max_week.strftime('%Y-%m-%d') if max_week else 'N/A'})\n"
        f"Min Forecast: {min_forecast:.0f} (Week of {min_week.strftime('%Y-%m-%d') if min_week else 'N/A'})"
    )

    plt.gcf().text(0.78, 0.5, summary_text, fontsize=10, bbox=dict(facecolor='white', alpha=0.8), va='center')
    plt.subplots_adjust(right=0.75)

    os.makedirs(folder_name, exist_ok=True)
    graph_file_path = os.path.join(folder_name, f"{product_title.replace('/', '_')}_{asin}.png")
    plt.savefig(graph_file_path)
    plt.close()
    print(f"Graph saved to {graph_file_path}")


##############################
# Excel Output
##############################

def save_summary_to_excel(comparison,
                          summary_stats,
                          total_forecast_16,
                          total_forecast_8,
                          total_forecast_4,
                          max_forecast,
                          min_forecast,
                          max_week,
                          min_week,
                          output_file_path,
                          metrics=None,
                          base_year=2025):

    def week_label_to_date(week_label, base_year):
        # Convert week label like 'W01' to a start date
        week_num = int(week_label[1:])
        # Assume first week of base_year starts on January 1st
        start_date = pd.Timestamp(f'{base_year}-01-01') + pd.Timedelta(weeks=week_num-1)
        return start_date.strftime('%Y-%m-%d')

    # Create Week_Start_Date based on available information
    if 'ds' in comparison.columns:
        comparison['ds'] = pd.to_datetime(comparison['ds'], errors='coerce')
        comparison['Week_Start_Date'] = comparison['ds'].dt.strftime('%Y-%m-%d')
    elif 'Week' in comparison.columns:
        comparison['Week_Start_Date'] = comparison['Week'].apply(lambda w: week_label_to_date(w, base_year))

    # Create Week label column if not present
    if 'Week' not in comparison.columns:
        # Sort by Week_Start_Date to ensure correct order
        comparison = comparison.sort_values('Week_Start_Date').reset_index(drop=True)
        comparison['Week'] = ['W' + str(i+1) for i in range(len(comparison))]

    # Round forecast values to integers for specified columns
    for col in ['MyForecast', 'Amazon Mean Forecast', 'Amazon P70 Forecast',
                'Amazon P80 Forecast', 'Amazon P90 Forecast']:
        if col in comparison.columns:
            comparison[col] = comparison[col].round().astype(int)

    # Drop 'ds' column if it exists since we now use Week and Week_Start_Date
    if 'ds' in comparison.columns:
        comparison.drop(columns=['ds'], inplace=True)

    # Define the desired column order, including the new columns
    columns_to_include = [
        'Week', 'Week_Start_Date', 'ASIN', 'MyForecast', 'Amazon Mean Forecast',
        'Amazon P70 Forecast', 'Amazon P80 Forecast', 'Amazon P90 Forecast',
        'Product Title', 'is_holiday_week'
    ]

    # Ensure all desired columns are present in the DataFrame
    for col in columns_to_include:
        if col not in comparison.columns:
            comparison[col] = np.nan

    comparison = comparison[columns_to_include]

    wb = Workbook()
    ws1 = wb.active
    ws1.title = "Forecast Comparison"

    from openpyxl.utils.dataframe import dataframe_to_rows
    for r in dataframe_to_rows(comparison, index=False, header=True):
        ws1.append(r)

    # Prepare summary sheet as before
    ws2 = wb.create_sheet(title="Summary")
    summary_data = {
        "Metric": [
            "Historical Range",
            "Min Sales",
            "Max Sales",
            "Mean Sales",
            "Median Sales",
            "Std Dev Sales",
            "Total Historical Sales",
            "Total Forecast (16 Weeks)",
            "Total Forecast (8 Weeks)",
            "Total Forecast (4 Weeks)",
            "Max Forecast",
            "Max Forecast Week",
            "Min Forecast",
            "Min Forecast Week"
        ],
        "Value": [
            f"{summary_stats['data_range'][0].strftime('%Y-%m-%d')} to {summary_stats['data_range'][1].strftime('%Y-%m-%d')}",
            f"{summary_stats['min']:.0f}",
            f"{summary_stats['max']:.0f}",
            f"{summary_stats['mean']:.0f}",
            f"{summary_stats['median']:.0f}",
            f"{summary_stats['std_dev']:.0f}",
            f"{summary_stats['total_sales']:.0f} units",
            f"{total_forecast_16:.0f}",
            f"{total_forecast_8:.0f}",
            f"{total_forecast_4:.0f}",
            f"{max_forecast:.0f}",
            f"{max_week.strftime('%Y-%m-%d') if max_week else 'N/A'}",
            f"{min_forecast:.0f}",
            f"{min_week.strftime('%Y-%m-%d') if min_week else 'N/A'}"
        ]
    }
    if metrics is not None:
        for k, v in metrics.items():
            summary_data["Metric"].append(k)
            summary_data["Value"].append(str(v))

    summary_df = pd.DataFrame(summary_data)
    for r in dataframe_to_rows(summary_df, index=False, header=True):
        ws2.append(r)

    wb.save(output_file_path)
    print(f"Comparison and summary saved to '{output_file_path}'")


def save_4_8_16_forecast_summary(consolidated_data, output_folder):
    """
    Save the cumulative 4th, 8th, and 16th-week forecasts for each ASIN.
    """
    summary_data = []

    for asin, df in consolidated_data.items():
        if df.empty:
            summary_data.append([asin, "No data", None, None, None])
            continue

        product_title = df['Product Title'].iloc[0] if 'Product Title' in df.columns else ''
        
        df['Cumulative Forecast'] = df['MyForecast'].cumsum()

        four_wk_val = df['Cumulative Forecast'].iloc[3] if len(df) > 3 else None
        eight_wk_val = df['Cumulative Forecast'].iloc[7] if len(df) > 7 else None
        sixteen_wk_val = df['Cumulative Forecast'].iloc[15] if len(df) > 15 else None

        summary_data.append([asin, product_title, four_wk_val, eight_wk_val, sixteen_wk_val])

    summary_df = pd.DataFrame(
        summary_data,
        columns=["ASIN", "Product Title", "4th Week Cumulative Forecast", "8th Week Cumulative Forecast", "16th Week Cumulative Forecast"]
    )

    output_file_path = os.path.join(output_folder, "4-8-16_Weeks_Forecast_Summary.xlsx")
    summary_df.to_excel(output_file_path, index=False)
    print(f"4-8-16 Weeks Forecast Summary saved to {output_file_path}")

def save_forecast_to_excel(output_path, consolidated_data, missing_asin_data, base_year=2025):
    """
    Save multiple ASIN forecasts (with "MyForecast") and any missing ASIN data into one Excel file,
    each ASIN in a separate sheet.
    """

    desired_columns = [
        'Week', 'Week_Start_Date', 'ASIN', 'MyForecast', 'Amazon Mean Forecast',
        'Amazon P70 Forecast', 'Amazon P80 Forecast', 'Amazon P90 Forecast',
        'Product Title', 'is_holiday_week'
    ]

    def week_label_to_date(week_label, base_year):
        week_num = int(week_label[1:])
        start_date = pd.Timestamp(f'{base_year}-01-01') + pd.Timedelta(weeks=week_num-1)
        return start_date.strftime('%Y-%m-%d')

    wb = Workbook()

    for asin, forecast_df in consolidated_data.items():
        df_for_excel = forecast_df.copy()

        # Calculate Week_Start_Date
        if 'ds' in df_for_excel.columns:
            df_for_excel['ds'] = pd.to_datetime(df_for_excel['ds'], errors='coerce')
            df_for_excel['Week_Start_Date'] = df_for_excel['ds'].dt.strftime('%Y-%m-%d')
            # Create Week labels if missing
            if 'Week' not in df_for_excel.columns:
                df_for_excel = df_for_excel.sort_values('Week_Start_Date').reset_index(drop=True)
                df_for_excel['Week'] = ['W' + str(i+1).zfill(2) for i in range(len(df_for_excel))]
        elif 'Week' in df_for_excel.columns:
            df_for_excel['Week_Start_Date'] = df_for_excel['Week'].apply(lambda w: week_label_to_date(w, base_year))

        # Round forecast columns to integers
        forecast_cols = ['MyForecast', 'Amazon Mean Forecast', 'Amazon P70 Forecast', 
                         'Amazon P80 Forecast', 'Amazon P90 Forecast']
        for col in forecast_cols:
            if col in df_for_excel.columns:
                df_for_excel[col] = df_for_excel[col].round().astype(int)

        # Ensure all desired columns exist
        for col in desired_columns:
            if col not in df_for_excel.columns:
                df_for_excel[col] = np.nan

        # Fill 'ASIN' and 'Product Title' if missing
        df_for_excel['ASIN'] = asin
        df_for_excel['Product Title'] = forecast_df['Product Title'].iloc[0] if 'Product Title' in forecast_df.columns else ''

        # Select desired columns only
        df_for_excel = df_for_excel[desired_columns]

        ws = wb.create_sheet(title=str(asin)[:31])  # Excel sheet names limited to 31 characters

        for r in dataframe_to_rows(df_for_excel, index=False, header=True):
            ws.append(r)

    # Handle missing ASIN data
    if not missing_asin_data.empty:
        ws_missing = wb.create_sheet(title="No ASIN")
        for r in dataframe_to_rows(missing_asin_data, index=False, header=True):
            ws_missing.append(r)

    # Remove default 'Sheet' if it exists
    if 'Sheet' in wb.sheetnames:
        del wb['Sheet']

    # Add a Summary Sheet
    ws_summary = wb.create_sheet(title="Summary")
    ws_summary.append(["ASIN", "Product Title", "4 Week Forecast", "8 Week Forecast", "16 Week Forecast"])

    for asin, df in consolidated_data.items():
        if df.empty:
            ws_summary.append([asin, "No data", None, None, None])
            continue

        product_title = df['Product Title'].iloc[0] if 'Product Title' in df.columns else ''
        four_wk_val = df['MyForecast'].iloc[:4].sum() if len(df) >= 4 else df['MyForecast'].sum()
        eight_wk_val = df['MyForecast'].iloc[:8].sum() if len(df) >= 8 else df['MyForecast'].sum()
        sixteen_wk_val = df['MyForecast'].iloc[:16].sum() if len(df) >= 16 else df['MyForecast'].sum()

        ws_summary.append([asin, product_title, four_wk_val, eight_wk_val, sixteen_wk_val])

    wb.save(output_path)
    print(f"All forecasts saved to '{output_path}'")


def save_feedback_to_excel(feedback_dict, filename):
    """
    Save feedback information from models into an Excel file.
    """
    records = []
    for asin, info in feedback_dict.items():
        record = {'ASIN': asin}
        best_params = info.get('best_params', {})
        for param, value in best_params.items():
            record[param] = value
        if 'rmse_values' in info and info['rmse_values']:
            for k, v in info['rmse_values'].items():
                record[f'RMSE_{k}'] = v
        record['Total Tests'] = info.get('total_tests', None)
        records.append(record)
    
    df_feedback = pd.DataFrame(records)
    with pd.ExcelWriter(filename, engine='openpyxl') as writer:
        df_feedback.to_excel(writer, index=False, sheet_name='Model Feedback')
    print(f"Feedback saved to {filename}")

def generate_4_week_report(consolidated_forecasts):
    """
    Generate an Excel report for the first 4 weeks of "MyForecast" for each ASIN,
    plus Amazon columns if present.
    """
    report_rows = []
    for asin, comp_df in consolidated_forecasts.items():
        # Determine the model used based on available info
        # (We only unify the final forecast column as "MyForecast")
        product_title = ''
        if 'Product Title' in comp_df.columns and not comp_df.empty:
            product_title = comp_df['Product Title'].iloc[0]

        if 'MyForecast' in comp_df.columns:
            model_used = 'Unified'
            my_forecast_column = 'MyForecast'
        else:
            model_used = 'Unknown'
            my_forecast_column = None

        if my_forecast_column and my_forecast_column in comp_df.columns:
            my_4w_forecast = comp_df[my_forecast_column].iloc[:4].sum()
        else:
            my_4w_forecast = np.nan

        amz_mean = comp_df['Amazon Mean Forecast'].iloc[:4].sum() if 'Amazon Mean Forecast' in comp_df.columns else np.nan
        amz_p70  = comp_df['Amazon P70 Forecast'].iloc[:4].sum() if 'Amazon P70 Forecast' in comp_df.columns else np.nan
        amz_p80  = comp_df['Amazon P80 Forecast'].iloc[:4].sum() if 'Amazon P80 Forecast' in comp_df.columns else np.nan
        amz_p90  = comp_df['Amazon P90 Forecast'].iloc[:4].sum() if 'Amazon P90 Forecast' in comp_df.columns else np.nan

        report_rows.append({
            'ASIN': asin,
            'Product Title': product_title,
            'My 4 Weeks Forecast': my_4w_forecast,
            'AMZ Forecast Mean': amz_mean,
            'AMZ Forecast P70': amz_p70,
            'AMZ Forecast P80': amz_p80,
            'AMZ Forecast P90': amz_p90
        })

    report_df = pd.DataFrame(report_rows)
    report_filename = '4_week_report.xlsx'
    report_df.to_excel(report_filename, index=False)
    print(f"4-week report saved to {report_filename}")

def generate_combined_weekly_report(consolidated_forecasts):
    """
    Generate an Excel report summarizing 4-, 8-, and 16-week forecasts for both
    your model.
    
    Output Columns:
    ASIN, Product Name, 4 Weeks Forecast, 8 Weeks Forecast, 16 Weeks Forecast,
    """
    report_rows = []
    for asin, comp_df in consolidated_forecasts.items():
        # Extract product name if available
        product_name = ""
        if 'Product Title' in comp_df.columns and not comp_df.empty:
            product_name = comp_df['Product Title'].iloc[0]
        
        # Use 'MyForecast' column; if missing, default to zeros
        my_forecast = comp_df['MyForecast'] if 'MyForecast' in comp_df.columns else pd.Series([0]*len(comp_df))
        # Use 'Amazon Mean Forecast' column; if missing, default to zeros
        #amz_forecast = comp_df['Amazon Mean Forecast'] if 'Amazon Mean Forecast' in comp_df.columns else pd.Series([0]*len(comp_df))

        # Calculate cumulative sums for specified weeks
        forecast_4 = int(round(my_forecast.iloc[:4].sum())) if len(my_forecast) >= 4 else int(round(my_forecast.sum()))
        forecast_8 = int(round(my_forecast.iloc[:8].sum())) if len(my_forecast) >= 8 else int(round(my_forecast.sum()))
        forecast_16 = int(round(my_forecast.iloc[:16].sum())) if len(my_forecast) >= 16 else int(round(my_forecast.sum()))

        report_rows.append({
            'ASIN': asin,
            'Product Name': product_name,
            '4 Weeks Forecast': forecast_4,
            '8 Weeks Forecast': forecast_8,
            '16 Weeks Forecast': forecast_16
        })

    report_df = pd.DataFrame(report_rows)
    report_filename = 'combined_4_8_16_week_report.xlsx'
    report_df.to_excel(report_filename, index=False)
    print(f"Combined 4-8-16 week report saved to {report_filename}")

def save_consolidated_forecasts(output_path, consolidated_data, base_year=2025):
    import numpy as np
    import pandas as pd
    from openpyxl import Workbook
    from openpyxl.utils.dataframe import dataframe_to_rows

    desired_columns = [
        'Week', 'Week_Start_Date', 'ASIN', 'MyForecast', 'Amazon Mean Forecast',
        'Amazon P70 Forecast', 'Amazon P80 Forecast', 'Amazon P90 Forecast',
        'Product Title', 'is_holiday_week'
    ]

    def week_label_to_date(week_label, base_year):
        week_num = int(week_label[1:])
        start_date = pd.Timestamp(f'{base_year}-01-01') + pd.Timedelta(weeks=week_num-1)
        return start_date.strftime('%Y-%m-%d')

    consolidated_list = []
    for asin, df in consolidated_data.items():
        df_copy = df.copy()

        # Calculate Week_Start_Date
        if 'ds' in df_copy.columns:
            df_copy['ds'] = pd.to_datetime(df_copy['ds'], errors='coerce')
            df_copy['Week_Start_Date'] = df_copy['ds'].dt.strftime('%Y-%m-%d')
        elif 'Week' in df_copy.columns:
            df_copy['Week_Start_Date'] = df_copy['Week'].apply(lambda w: week_label_to_date(w, base_year))

        # Round forecast columns to integers
        forecast_cols = ['MyForecast', 'Amazon Mean Forecast', 'Amazon P70 Forecast', 
                         'Amazon P80 Forecast', 'Amazon P90 Forecast']
        for col in forecast_cols:
            if col in df_copy.columns:
                df_copy[col] = df_copy[col].round().astype(int)

        # Create Week labels if missing
        if 'Week' not in df_copy.columns:
            df_copy = df_copy.sort_values('Week_Start_Date').reset_index(drop=True)
            df_copy['Week'] = ['W' + str(i+1).zfill(2) for i in range(len(df_copy))]

        # Ensure all desired columns exist
        for col in desired_columns:
            if col not in df_copy.columns:
                df_copy[col] = np.nan

        # Select desired columns only
        df_copy = df_copy[desired_columns]

        consolidated_list.append(df_copy)

    # Combine all ASIN dataframes
    if consolidated_list:
        consolidated_df = pd.concat(consolidated_list, ignore_index=True)
        consolidated_df = consolidated_df.sort_values(['Week_Start_Date', 'ASIN']).reset_index(drop=True)
    else:
        consolidated_df = pd.DataFrame(columns=desired_columns)

    # Save the consolidated DataFrame to Excel
    with pd.ExcelWriter(output_path) as writer:
        consolidated_df.to_excel(writer, sheet_name='Consolidated Forecast', index=False)
    print(f"All forecasts saved to '{output_path}'")


##############################
# Additional Prophet Cross-Validation
##############################

def cross_validate_prophet_model(ts_data, initial='180 days', period='90 days', horizon='90 days'):
    """Run Prophet's built-in cross-validation and performance metrics."""
    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        seasonality_mode='multiplicative'
    )
    model.fit(ts_data[['ds', 'y']])
    df_cv = cross_validation(model, initial=initial, period=period, horizon=horizon)
    df_performance = performance_metrics(df_cv)
    print("Prophet Model Cross-Validation Results:")
    print(df_performance)
    return df_cv, df_performance


##############################
# Analysis of Amazon vs. Prophet
##############################

def analyze_amazon_buying_habits(comparison, holidays):
    """
    Analyze Amazon forecast columns vs. 'MyForecast', highlighting holiday weeks, etc.
    """
    if 'y' not in comparison.columns:
        print("Warning: 'y' column missing in the comparison DataFrame. Skipping analysis.")
        return

    amazon_cols = [
        col for col in comparison.columns
        if col.startswith('Amazon ') and col.endswith('Forecast') and not col.endswith('Forecast Forecast')
    ]
    if not amazon_cols:
        print("No Amazon forecasts found for analysis.")
        return

    myforecast_series = comparison.get('MyForecast', pd.Series(index=comparison.index))
    ds_dates = comparison.get('ds', pd.Series(index=comparison.index))
    holiday_dates = holidays['ds'].values if holidays is not None else []
    comparison['is_holiday_week'] = ds_dates.isin(holiday_dates) if 'ds' in comparison.columns else False

    amazon_types = ['Mean', 'P70', 'P80', 'P90']
    errors = {}
    for forecast_type in amazon_types:
        forecast_col = f"Amazon {forecast_type} Forecast"
        if forecast_col in comparison.columns:
            valid_data = comparison[['y', forecast_col]].dropna()
            if len(valid_data) == 0:
                continue
            current_rmse = np.sqrt(mean_squared_error(comparison['y'], comparison[forecast_col]))
            current_mae = mean_absolute_error(comparison['y'], comparison[forecast_col]) * 100
            errors[forecast_type] = {
                'RMSE': current_rmse,
                'MAE': current_mae
            }

    if errors:
        best_forecast_type = min(errors, key=lambda x: errors[x]['RMSE'])
        print(f"\nBest Amazon forecast type: {best_forecast_type} with RMSE={errors[best_forecast_type]['RMSE']:.4f}")
    
    for forecast_type in amazon_types:
        forecast_col = f"Amazon {forecast_type} Forecast"
        if forecast_col not in comparison.columns:
            continue
        amazon_vals = comparison[forecast_col].values
        safe_myforecast = np.where(myforecast_series == 0, 1e-9, myforecast_series)
        ratio = amazon_vals / safe_myforecast
        diff = amazon_vals - myforecast_series
        avg_ratio = np.mean(ratio)
        avg_diff = np.mean(diff)
        print(f"\nFor {forecast_type}:")
        print(f"  Average Amazon/MyForecast Ratio: {avg_ratio:.2f}")
        print(f"  Average Difference (Amazon - MyForecast): {avg_diff:.2f}")

        if avg_diff > 0:
            print("  Amazon tends to forecast more than MyForecast on average.")
        elif avg_diff < 0:
            print("  Amazon tends to forecast less than MyForecast on average.")
        else:
            print("  Amazon forecasts similarly to MyForecast on average.")

        holiday_mask = comparison['is_holiday_week']
        if holiday_mask.any():
            holiday_ratio = ratio[holiday_mask]
            holiday_diff = diff[holiday_mask]
            if len(holiday_diff) > 0:
                print("  During holiday weeks:")
                print(f"    Avg Ratio (Amazon/MyForecast): {np.mean(holiday_ratio):.2f}")
                print(f"    Avg Diff (Amazon-MyForecast): {np.mean(holiday_diff):.2f}")

        weeks = np.arange(1, len(ratio) + 1)
        segments = {
            'Short-term (Weeks 1-4)': (weeks <= 4),
            'Mid-term (Weeks 5-12)': (weeks >= 5) & (weeks <= 12),
            'Long-term (Weeks 13+)': (weeks > 12)
        }

        for segment_name, mask in segments.items():
            if mask.any():
                seg_ratio = ratio[mask]
                seg_diff = diff[mask]
                print(f"  {segment_name}:")
                print(f"    Avg Ratio (Amazon/MyForecast): {np.mean(seg_ratio):.2f}")
                print(f"    Avg Diff (Amazon-MyForecast): {np.mean(seg_diff):.2f}")


##############################
# Additional/Custom Backtesting Function
##############################

def calculate_extended_metrics(actual, predicted):
    """
    Calculate extended metrics (RMSE, wQL, MASE, MAE, WAPE, etc.) for demonstration.
    """
    actual = np.array(actual)
    predicted = np.array(predicted)
    
    # 1. Root Mean Squared Error (RMSE)
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    
    # 2. Mean Absolute Error (MAE)
    mae = mean_absolute_error(actual, predicted)
    
    # 3. Weighted Quantile Loss (wQL) and Average wQL
    alpha = 0.5  # For simplistic quantile approach; adjust as needed
    residuals = actual - predicted
    quantile_loss = np.where(residuals >= 0, alpha * residuals, (1 - alpha) * -residuals)
    wql = np.sum(np.abs(quantile_loss))
    avg_wql = np.mean(np.abs(quantile_loss)) if len(quantile_loss) > 0 else 0
    
    # 4. Weighted Absolute Percentage Error (WAPE)
    total_actual = np.sum(np.abs(actual))
    if total_actual != 0:
        wape = np.sum(np.abs(actual - predicted)) / total_actual * 100
    else:
        wape = 0
    
    # 5. Mean Absolute Scaled Error (MASE)
    # Naive forecast: using the previous period's actual value
    # Shift actual by one period to get naive forecast
    if len(actual) > 1:
        naive_forecast = actual[:-1]
        actual_shifted = actual[1:]
        naive_error = np.abs(actual_shifted - naive_forecast)
        mae_naive = np.mean(naive_error) if len(naive_error) > 0 else 1
    else:
        mae_naive = 1  # Avoid division by zero
    
    mase = mae / mae_naive if mae_naive != 0 else np.nan
    
    return {
        "RMSE": rmse,
        "MAE": mae,
        "wQL": wql,
        "Average wQL": avg_wql,
        "MASE": mase,
        "WAPE": wape
    }

def backtest_forecast(ts_data, model_function, horizon=16, folder_name="Training and Testing data"):
    """
    Example backtesting function for demonstration.
    """
    os.makedirs(folder_name, exist_ok=True)
    ts_data = ts_data.sort_values('ds').copy()
    n = len(ts_data)
    split_idx = int(n * 0.8)
    train = ts_data.iloc[:split_idx].copy()
    test = ts_data.iloc[split_idx:].copy()

    fitted_model = model_function(train)

    try:
        last_date = train['ds'].iloc[-1]
        steps = len(test)
        test_forecast_df = pd.DataFrame({
            'ds': test['ds'],
            'MyForecast': [train['y'].iloc[-1]] * steps
        })
    except Exception as e:
        print(f"Error in backtest forecasting step: {e}")
        return

    backtest_comparison = test_forecast_df.merge(test[['ds','y']], on='ds', how='left')

    metrics = calculate_extended_metrics(backtest_comparison['y'], backtest_comparison['MyForecast'])
    print("Backtest Metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    train_file = os.path.join(folder_name, "training_data.csv")
    test_file = os.path.join(folder_name, "testing_data.csv")
    train.to_csv(train_file, index=False)
    test.to_csv(test_file, index=False)

    plt.figure(figsize=(10,6))
    plt.plot(train['ds'], train['y'], label='Training', color='blue')
    plt.plot(test['ds'], test['y'], label='Testing Actual', color='orange')
    plt.plot(backtest_comparison['ds'], backtest_comparison['MyForecast'], label='Testing Forecast', color='green')
    plt.legend()
    plt.title("Backtest: Training vs Testing Forecast")
    plt.xlabel("Date")
    plt.ylabel("Values")
    backtest_plot_path = os.path.join(folder_name, "training_testing_graph.png")
    plt.savefig(backtest_plot_path)
    plt.close()
    print(f"Training/Testing data and graph saved to '{folder_name}'.")


##############################
# Function to Optimize Ensemble Weights
##############################

def optimize_ensemble_weights(actual, sarima_preds, prophet_preds, xgb_preds):
    """
    Example of optimizing ensemble weights for three model forecasts.
    """
    initial_weights = [1/3, 1/3, 1/3]
    constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
    bounds = [(0, 1), (0, 1), (0, 1)]

    def objective(weights):
        ensemble_preds = (weights[0] * sarima_preds +
                          weights[1] * prophet_preds +
                          weights[2] * xgb_preds)
        rmse = np.sqrt(mean_squared_error(actual, ensemble_preds))
        mape = mean_absolute_error(actual, ensemble_preds)
        return rmse + mape

    result = minimize(objective, initial_weights, constraints=constraints, bounds=bounds, method='SLSQP')
    if result.success:
        optimized_weights = result.x
        ensemble_preds = (optimized_weights[0] * sarima_preds +
                          optimized_weights[1] * prophet_preds +
                          optimized_weights[2] * xgb_preds)
        rmse = np.sqrt(mean_squared_error(actual, ensemble_preds))
        mae = mean_absolute_error(actual, ensemble_preds)
        return {
            'weights': {
                'SARIMA': optimized_weights[0],
                'Prophet': optimized_weights[1],
                'XGBoost': optimized_weights[2],
            },
            'metrics': {
                'RMSE': rmse,
                'MAE': mae,
            },
            'ensemble_preds': ensemble_preds
        }
    else:
        raise ValueError("Optimization failed: " + result.message)


##############################
# Adjust Forecast If Out of Range
##############################

def adjust_forecast_if_out_of_range(
    comparison,
    asin,
    forecast_col_name='MyForecast',
    adjustment_threshold=0.3
):
    """
    Adjust 'MyForecast' if it is too far from Amazon Mean Forecast.
    Uses a higher threshold (0.6) ONLY if the last 5 sales data points
    average is lower than the overall average.
    """
    global out_of_range_counter
    global out_of_range_stats

    # 1) Debugging: Print a small sample of columns
    print("\nAmazon Forecast Statistics for Debugging:")
    required_amazon_cols = ['Amazon Mean Forecast', 'Amazon P70 Forecast', 'Amazon P80 Forecast', 'Amazon P90 Forecast']
    existing_cols = [col for col in required_amazon_cols if col in comparison.columns]
    print(comparison[existing_cols].head())

    # 2) Compute average sales of the last 5 data points
    #    Make sure we have 'y' in the DataFrame
    if 'y' in comparison.columns:
        recent_5 = comparison.dropna(subset=['y']).sort_values('ds').tail(5)
        last_5_avg = recent_5['y'].mean() if len(recent_5) > 0 else 0
        overall_avg = comparison['y'].mean()
    else:
        # Fallback if there's no 'y' column at all
        last_5_avg = 0
        overall_avg = 0

    # 3) Decide local threshold
    if last_5_avg < overall_avg:
        # Use a higher threshold if recent sales are below overall average
        local_threshold = 0.6
        print(f"Recent 5-wk avg {last_5_avg:.2f} is below overall avg {overall_avg:.2f}. "
              f"Using threshold={int(local_threshold*100)}%.")
    else:
        local_threshold = adjustment_threshold

    # 4) Identify rows "out of range" based on local_threshold
    #    (±30% or ±60%, depending on local_threshold)
    comparison['is_out_of_range'] = (
        (comparison[forecast_col_name] < comparison['Amazon Mean Forecast'] * (1 - local_threshold)) |
        (comparison[forecast_col_name] > comparison['Amazon Mean Forecast'] * (1 + local_threshold))
    )

    adjustment_mask = comparison['is_out_of_range']
    total_forecasts = len(comparison)

    # 5) If any rows are out of range, adjust the forecast
    if adjustment_mask.any():
        num_adjustments = adjustment_mask.sum()
        print(f"\nAdjusting {num_adjustments} out-of-range forecasts for ASIN {asin} using Amazon Mean Forecast.")
        out_of_range_counter[asin] += num_adjustments

        if asin not in out_of_range_stats:
            out_of_range_stats[asin] = {'total': 0, 'adjusted': 0}
        out_of_range_stats[asin]['total'] += total_forecasts
        out_of_range_stats[asin]['adjusted'] += num_adjustments

        # Adjust MyForecast to be within ±local_threshold of Amazon Mean
        comparison.loc[adjustment_mask, forecast_col_name] = (
            comparison.loc[adjustment_mask, 'Amazon Mean Forecast'] *
            comparison.loc[adjustment_mask, forecast_col_name] /
            comparison.loc[adjustment_mask, 'Amazon Mean Forecast']
        ).clip(lower=(1 - local_threshold), upper=(1 + local_threshold)) * \
          comparison.loc[adjustment_mask, 'Amazon Mean Forecast']

        # Ensure non-negative after adjustment
        comparison[forecast_col_name] = comparison[forecast_col_name].clip(lower=0)

        # Debug: see if anything is still out of range
        comparison['is_still_out_of_range'] = (
            (comparison[forecast_col_name] < comparison['Amazon Mean Forecast'] * (1 - local_threshold)) |
            (comparison[forecast_col_name] > comparison['Amazon Mean Forecast'] * (1 + local_threshold))
        )
        if comparison['is_still_out_of_range'].any():
            print("\nRows still out of range after primary adjustment:")
            print(
                comparison.loc[comparison['is_still_out_of_range'],
                               ['ds', forecast_col_name, 'Amazon Mean Forecast',
                                'Amazon P70 Forecast', 'Amazon P80 Forecast', 'Amazon P90 Forecast']]
            )

    # Debug: show adjusted rows
    if adjustment_mask.any():
        print("\nAdjusted forecasts for out-of-range rows:")
        print(
            comparison.loc[adjustment_mask,
                           ['ds', forecast_col_name, 'Amazon Mean Forecast', 'Amazon P70 Forecast', 'Amazon P80 Forecast']]
        )

    # Final check or summary
    if existing_cols:
        print("\nFinal adjusted forecast data (head):")
        print(comparison[['ds', forecast_col_name] + existing_cols].head())
    else:
        print("Warning: No Amazon forecast columns were found. No adjustments applied.")

    # Clean up temporary columns, if desired
    if 'is_still_out_of_range' in comparison.columns:
        comparison.drop(columns=['is_still_out_of_range'], inplace=True)
    comparison.drop(columns=['is_out_of_range'], inplace=True)

    return comparison

def log_fallback_triggers(comparison, asin, product_title, fallback_file="fallback_triggers.csv"):
    """
    Logs products where the fallback mechanism was triggered to a separate file.
    """
    if 'is_still_out_of_range' not in comparison.columns:
        print(f"No 'is_still_out_of_range' column present in DataFrame for ASIN: {asin}. No fallback to log.")
        return

    fallback_rows = comparison[comparison['is_still_out_of_range']]
    if not fallback_rows.empty:
        print(f"Outlier detected for ASIN: {asin}, Product: {product_title}")
        fallback_info = {
            "ASIN": [asin],
            "Product Title": [product_title],
            "Outlier Weeks": [fallback_rows['ds'].tolist()],
            "Total Adjustments": [fallback_rows.shape[0]]
        }

        fallback_df = pd.DataFrame(fallback_info)

        if os.path.exists(fallback_file):
            existing_data = pd.read_csv(fallback_file)
            fallback_df = pd.concat([existing_data, fallback_df], ignore_index=True)

        fallback_df.to_csv(fallback_file, index=False)
        print(f"Fallback log updated: {fallback_file}")
    else:
        print(f"No outliers detected for ASIN: {asin}")


##############################
# Feedback Loop Functions
##############################

def record_forecast_errors(asin, forecast_df, actual_df):
    """
    Compare 'MyForecast' vs. actual data for a given ASIN and record errors.
    """
    merged = forecast_df.merge(actual_df[['ds', 'y']], on='ds', how='left')
    merged = merged.dropna(subset=['y'])
    if merged.empty:
        print(f"No actual data available yet to compare for ASIN {asin}.")
        return
    mae = mean_absolute_error(merged['y'], merged['MyForecast'])
    mape = mean_absolute_error(merged['y'], merged['MyForecast']) * 100
    forecast_errors.setdefault(asin, []).append({'mae': mae, 'mape': mape})
    print(f"Recorded forecast errors for ASIN {asin}: MAE={mae}, MAPE={mape}%")


def update_prophet_model_with_feedback(asin, ts_data, forecast_data, param_grid, horizon, current_model=None):
    """
    Update the Prophet model for a given ASIN using new actual data.
    """
    print(f"Updating Prophet model for ASIN {asin} with new data...")
    best_params, _ = optimize_prophet_params(ts_data, forecast_data, param_grid, horizon=horizon)
    forecast, updated_model = forecast_with_custom_params(
        ts_data, forecast_data,
        best_params['changepoint_prior_scale'],
        best_params['seasonality_prior_scale'],
        best_params['holidays_prior_scale'],
        horizon=horizon
    )
    save_model(updated_model, "Prophet", asin, ts_data)
    print(f"Prophet model for ASIN {asin} updated.")
    return forecast, updated_model

##############################
# Analyze PO Order
##############################

def analyze_po_data_by_asin(po_file="po database.xlsx", product_filter=None, output_folder="po_analysis_by_asin"):
    """
    Analyze PO orders by each ASIN to understand buying habits based on requested quantities.
    Generates weekly and monthly requested quantity trend graphs and creates an Excel file 
    for each ASIN with separate sheets for weekly and monthly data.

    Parameters:
      po_file: Path to the PO Excel file.
      product_filter: Optional ASIN or product filter to limit analysis.
      output_folder: Folder to save individual ASIN reports and graphs.

    Returns:
      A dictionary with aggregated data and paths to generated graphs for each ASIN.
    """
    # Load PO data
    po_df = pd.read_excel(po_file)

    # Standardize column names (strip spaces)
    po_df.columns = po_df.columns.str.strip()

    # Convert date columns to datetime
    for date_col in ['Order date', 'Expected date']:
        if date_col in po_df.columns:
            po_df[date_col] = pd.to_datetime(po_df[date_col], errors='coerce')

    # Filter by product if specified
    if product_filter:
        po_df = po_df[po_df['ASIN'] == product_filter]

    # Ensure numeric conversion for Requested quantity
    po_df['Requested quantity'] = pd.to_numeric(po_df['Requested quantity'], errors='coerce').fillna(0)

    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    results = {}
    unique_asins = po_df['ASIN'].dropna().unique()
    for asin in unique_asins:
        # Filter data for the current ASIN
        asin_df = po_df[po_df['ASIN'] == asin].copy()
        if asin_df.empty:
            continue

        asin_df['Order_Week_Period'] = asin_df['Order date'].dt.to_period('W-SUN')
        asin_df['Order Week'] = asin_df['Order_Week_Period'].apply(lambda p: p.end_time)

        weekly_agg = (
            asin_df
            .groupby('Order Week', as_index=False)
            .agg({'Requested quantity': 'sum'})
        )

        asin_df['Order_Month_Period'] = asin_df['Order date'].dt.to_period('M')
        asin_df['Order Month'] = asin_df['Order_Month_Period'].apply(lambda p: p.end_time)
        monthly_trend = (
            asin_df
            .groupby('Order Month', as_index=False)
            .agg({'Requested quantity': 'sum'})
        )

        # Plot weekly trend for this ASIN
        plt.figure(figsize=(10, 6))
        plt.plot(weekly_agg['Order Week'], weekly_agg['Requested quantity'], marker='o')
        plt.title(f'Weekly Requested Quantities for ASIN {asin}')
        plt.xlabel('Week')
        plt.ylabel('Total Requested Quantity')
        plt.grid()
        plt.xticks(rotation=45)
        plt.tight_layout()
        weekly_chart = os.path.join(output_folder, f"{asin}_weekly_requested_quantity.png")
        plt.savefig(weekly_chart)
        plt.close()

        # Plot monthly trend for this ASIN
        plt.figure(figsize=(10, 6))
        plt.plot(monthly_trend['Order Month'], monthly_trend['Requested quantity'], marker='o', color='green')
        plt.title(f'Monthly Requested Quantities for ASIN {asin}')
        plt.xlabel('Month')
        plt.ylabel('Total Requested Quantity')
        plt.grid()
        plt.xticks(rotation=45)
        plt.tight_layout()
        monthly_chart = os.path.join(output_folder, f"{asin}_monthly_requested_quantity.png")
        plt.savefig(monthly_chart)
        plt.close()

        # Save weekly and monthly data into an Excel file for this ASIN
        excel_path = os.path.join(output_folder, f"{asin}_po_data.xlsx")
        with pd.ExcelWriter(excel_path) as writer:
            weekly_agg.to_excel(writer, sheet_name='Weekly Quantity', index=False)
            monthly_trend.to_excel(writer, sheet_name='Monthly Trend', index=False)

        # Store results for this ASIN
        results[asin] = {
            'weekly_agg': weekly_agg,
            'monthly_trend': monthly_trend,
            'weekly_chart': weekly_chart,
            'monthly_chart': monthly_chart,
            'excel_report': excel_path
        }

    print(f"Generated graphs and Excel reports for {len(unique_asins)} ASINs in folder '{output_folder}'.")
    return results

##############################
# Compare PO Orders with Forecasts
##############################

def compare_historical_sales_po(asin, sales_df, po_df, output_folder="po_forecast_comparison"):
    """
    Compare historical sales with PO requested quantities for a given ASIN.
    Overlays historical sales and PO trends, computes correlation, 
    calculates growth percentages and volume insights,
    and provides a basic prediction for future PO quantities.
    Saves overlay graphs and merged data.
    
    Parameters:
      asin: The ASIN to analyze.
      sales_df: DataFrame containing weekly aggregated historical sales with columns ['ds', 'y'].
                - 'ds' is assumed to be a Sunday date (week-end).
      po_df: DataFrame containing historical PO data with columns ['ASIN', 'Order date', 'Requested quantity'].
      output_folder: Directory where results will be saved.
    
    Returns:
      merged_df: DataFrame merging historical sales and PO data.
      correlation: Correlation coefficient between sales and PO requested quantities.
      growth_info: Dictionary containing growth percentages, volume insights, and predicted next week PO quantity.
    """
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Ensure numeric conversion for Requested quantity; handle commas and spaces
    po_df = po_df.copy()
    po_df['Requested quantity'] = (
         po_df['Requested quantity']
         .astype(str)
         .str.replace(',', '')
         .str.strip()
         .replace('', '0')
    )
    po_df['Requested quantity'] = pd.to_numeric(po_df['Requested quantity'], errors='coerce').fillna(0)
    
    asin_po = po_df[po_df['ASIN'] == asin].copy()
    asin_po['Order date'] = pd.to_datetime(asin_po['Order date'], errors='coerce')
    asin_po.sort_values('Order date', inplace=True)

    # 3) Create a daily PO DataFrame (raw, no summation)
    # Rename columns to be consistent
    asin_po.rename(columns={'Order date': 'Date', 'Requested quantity': 'Daily_PO_Qty'}, inplace=True)

    # 4) Basic volume stats on the raw/daily POs
    total_po_qty = asin_po['Daily_PO_Qty'].sum()
    avg_po_qty   = asin_po['Daily_PO_Qty'].mean() if len(asin_po) else 0
    max_po_qty   = asin_po['Daily_PO_Qty'].max() if len(asin_po) else 0
    min_po_qty   = asin_po['Daily_PO_Qty'].min() if len(asin_po) else 0

    volume_insights = {
        'Total_PO_Quantity': total_po_qty,
        'Average_PO_Quantity': avg_po_qty,
        'Max_PO_Quantity': max_po_qty,
        'Min_PO_Quantity': min_po_qty
    }

    # 5) Basic linear model (optional) to predict next daily PO
    predicted_next_po_qty = 0
    if len(asin_po) > 1:
        asin_po = asin_po.reset_index(drop=True)
        asin_po['Index'] = np.arange(1, len(asin_po) + 1)
        X = asin_po[['Index']]
        y = asin_po['Daily_PO_Qty']
        model = LinearRegression()
        model.fit(X, y)
        next_index = asin_po['Index'].iloc[-1] + 1
        predicted_next_po_qty = max(model.predict([[next_index]])[0], 0)

    prediction_info = {
        'Predicted_Next_Daily_PO_Quantity': predicted_next_po_qty
    }

    # 6) Merge weekly sales with daily PO if you want a combined DataFrame
    #    This is purely optional. The DS column is Sunday-based in sales_df, daily in PO.
    #    We'll do a left join on date. Many rows won't match exactly unless a daily date
    #    equals a Sunday from your sales.
    merged_df = pd.merge(
        sales_df.rename(columns={'ds': 'Date'}), 
        asin_po[['Date','Daily_PO_Qty']], 
        on='Date', 
        how='outer'
    ).sort_values('Date')
    merged_df['Daily_PO_Qty'] = merged_df['Daily_PO_Qty'].fillna(0)
    merged_df['y'] = merged_df['y'].fillna(0)

    # 7) Plot overlay: we can try to plot daily PO vs. weekly sales on the same axis
    plt.figure(figsize=(12,6))
    # Weekly sales
    plt.plot(
        merged_df['Date'], 
        merged_df['y'], 
        marker='o', 
        label='Weekly Sales (y)'
    )
    # Daily PO
    plt.plot(
        merged_df['Date'], 
        merged_df['Daily_PO_Qty'], 
        marker='x',
        label='Daily PO Qty'
    )
    plt.title(f"Daily POs vs Weekly Sales for ASIN {asin}")
    plt.xlabel("Date")
    plt.ylabel("Quantity")
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()

    chart_path = os.path.join(output_folder, f"{asin}_dailyPO_vs_weeklySales.png")
    plt.savefig(chart_path)
    plt.close()
    print(f"Chart saved to {chart_path}")

    # 8) Correlation
    # We'll only check correlation on days that have data in both columns:
    valid_idx = (merged_df['y'] > 0) & (merged_df['Daily_PO_Qty'] > 0)
    if valid_idx.sum() > 1:
        correlation = merged_df.loc[valid_idx, ['y','Daily_PO_Qty']].corr().iloc[0, 1]
        print(f"Correlation between weekly sales and daily PO for ASIN {asin}: {correlation:.2f}")
    else:
        correlation = None
        print(f"Not enough overlapping data for correlation on ASIN {asin}.")

    # 9) Write to Excel: two sheets - "Weekly Sales" and "Daily PO"
    # Also put merged in a third sheet if desired.
    excel_filename = os.path.join(output_folder, f"{asin}_sales_po_comparison.xlsx")
    with pd.ExcelWriter(excel_filename) as writer:
        # Sheet 1: Weekly Sales
        sales_df.to_excel(writer, sheet_name='Weekly Sales', index=False)

        # Sheet 2: Daily PO
        asin_po.to_excel(writer, sheet_name='Daily PO', index=False)

        # Optionally a third sheet with merged data
        merged_df.to_excel(writer, sheet_name='Merged (Optional)', index=False)

        # Another sheet with basic stats or growth info
        stats_df = pd.DataFrame([volume_insights])
        stats_df.to_excel(writer, sheet_name='PO Volume Insights', index=False)

        pred_df = pd.DataFrame([prediction_info])
        pred_df.to_excel(writer, sheet_name='PO Prediction', index=False)

    print(f"Excel saved to {excel_filename}")

    return merged_df, correlation, {
        'volume_insights': volume_insights,
        'prediction_info': prediction_info
    }


##############################
# Main
##############################

def main():
    # Load existing parameter histories
    load_param_histories()

    sales_file = 'weekly_sales_data.xlsx'
    forecasts_folder = 'forecasts_folder'
    asins_to_forecast_file = 'ASINs to Forecast.xlsx'
    output_file = 'consolidated_forecast.xlsx'
    horizon = 16

    data = load_weekly_sales_data(sales_file)
    valid_data = data[data['asin'].notna() & (data['asin'] != '#N/A')]
    missing_asin_data = data[data['asin'].isna() | (data['asin'] == '#N/A')]

    if not missing_asin_data.empty:
        print("The following entries have no ASIN and will be noted in the forecast file:")
        print(missing_asin_data[['product title', 'week', 'year', 'y']].to_string())

    asins_to_forecast = load_asins_to_forecast(asins_to_forecast_file)
    print(f"ASINs to forecast: {asins_to_forecast}")

    asin_list = valid_data['asin'].unique()
    # Filter to ensure only valid ASINs in the list to forecast
    asin_list = [asin for asin in asin_list if asin in asins_to_forecast]

    consolidated_forecasts = {}
    param_grid = {
        'changepoint_prior_scale': [0.05, 0.1, 0.2, 0.3, 0.4, 0.5],
        'seasonality_prior_scale': [0.05, 0.1, 1, 2, 3, 4, 5],
        'holidays_prior_scale': [5, 10, 15]
    }

    holidays = get_shifted_holidays()

    # Optional cross-validation test on the first ASIN in the list
    if len(asin_list) > 0:
        test_asin = asin_list[0]
        test_ts_data = prepare_time_series_with_lags(valid_data, test_asin, lag_weeks=1)
        if not test_ts_data.empty and len(test_ts_data.dropna()) >= 2:
            print(f"Performing cross-validation on ASIN {test_asin} Prophet model...")
            cross_validate_prophet_model(test_ts_data, initial='180 days', period='90 days', horizon='90 days')
        else:
            print(f"Not enough data for {test_asin} to perform cross-validation test.")

    insufficient_data_folder = "Insufficient data"
    sufficient_data_folder = "Sufficient data"
    os.makedirs(insufficient_data_folder, exist_ok=True)
    os.makedirs(sufficient_data_folder, exist_ok=True)

    global PARAM_COUNTER

    for asin in asin_list:
        if pd.isna(asin) or asin == '#N/A':
            print(f"Skipping invalid ASIN: {asin}")
            continue

        product_title = valid_data[valid_data['asin'] == asin]['product title'].iloc[0]
        print(f"\nProcessing ASIN: {asin} - {product_title}")
        
        # Load Amazon forecasts
        forecast_data = load_amazon_forecasts_from_folder(forecasts_folder, asin)
        if not forecast_data:
            print(f"No forecast data found for ASIN {asin}, skipping.")
            continue

        # Prepare data
        ts_data = prepare_time_series_with_lags(valid_data, asin, lag_weeks=1)
        print(f"Time series data for ASIN {asin} prepared. Dataset size: {len(ts_data)}")

        non_nan_count = len(ts_data.dropna())
        if non_nan_count < 10:
            print(f"ASIN {asin} has only {non_nan_count} data points. "
                f"Using fallback: 90% Amazon Mean + 10% 'naive SARIMA'.")

            # Step A: Create a naive SARIMA fallback forecast
            if not ts_data.empty:
                last_y_value = ts_data['y'].iloc[-1]
                last_date = ts_data['ds'].iloc[-1]
            else:
                last_y_value = 0
                last_date = pd.to_datetime("today")

            horizon = 16  # or whatever your default horizon is
            future_dates = pd.date_range(start=last_date + pd.Timedelta(weeks=1),
                                        periods=horizon, freq='W')
            fallback_sarima_vals = [last_y_value]*horizon

            # Step B: Put these into a temporary DataFrame
            fallback_df = pd.DataFrame({
                'ds': future_dates,
                'Fallback_SARIMA': fallback_sarima_vals
            })

            # Step C: Retrieve (or pad) Amazon Mean Forecast from `forecast_data`
            #         Typically your code merges Amazon forecasts later,
            #         but we can do it inline here since we’re skipping main flow.

            # 1) Identify which key in forecast_data holds the "Mean" forecast
            amz_mean_key = None
            for ftype in forecast_data.keys():
                if 'mean' in ftype.lower():
                    amz_mean_key = ftype
                    break

            if amz_mean_key is None:
                print("Warning: No 'Mean' forecast found in Amazon data. Using 0.")
                amazon_mean = [0]*horizon
            else:
                raw_arr = forecast_data[amz_mean_key]
                # If raw_arr is shorter than horizon, pad; if longer, truncate
                if len(raw_arr) >= horizon:
                    amazon_mean = raw_arr[:horizon]
                else:
                    pad_needed = horizon - len(raw_arr)
                    amazon_mean = list(raw_arr) + [raw_arr[-1]]*pad_needed

            fallback_df['Amazon Mean Forecast'] = amazon_mean

            # Step D: Compute final weighted forecast (90% Amazon Mean, 10% fallback)
            weight_amz = 0.9
            weight_sarima = 0.1
            final_vals = []
            for i in range(horizon):
                mean_val = fallback_df['Amazon Mean Forecast'].iloc[i]
                sarima_val = fallback_df['Fallback_SARIMA'].iloc[i]
                combined = (weight_amz * mean_val) + (weight_sarima * sarima_val)
                final_vals.append(int(round(max(combined, 0))))

            fallback_df['MyForecast'] = final_vals
            fallback_df['ASIN'] = asin
            fallback_df['Product Title'] = product_title

            # Step E: Store in consolidated_forecasts & skip the usual pipeline
            consolidated_forecasts[asin] = fallback_df
            continue  # Important: skip the Prophet/SARIMA logic below

        # Decide model type (SARIMA or Prophet)
        model, model_type = choose_forecast_model(ts_data, threshold=FALLBACK_THRESHOLD, holidays=holidays)

        # Train XGBoost, but do NOT blend with MyForecast
        xgb_model, xgb_features, xgb_shap_values = train_xgboost(ts_data, target='y')

        # =========================
        # SARIMA Block
        # =========================
        if model_type == "SARIMA":
            n = len(ts_data)
            split = int(n * 0.8)
            train_sarima = ts_data.iloc[:split]
            test_sarima = ts_data.iloc[split:]

            # Create exogenous variables for the test set
            exog_test = create_holiday_regressors(test_sarima, holidays)

            if model is None:
                print(f"SARIMA model fitting failed for {asin}, skipping.")
                no_data_output = os.path.join(insufficient_data_folder, f"{asin}_no_data.txt")
                with open(no_data_output, 'w') as f:
                    f.write("Insufficient data for training/forecasting.\n")
                continue

            try:
                # =========================
                # 1. Fit SARIMA Model (R/P system inside)
                # =========================
                best_sarima_model, best_params = fit_sarima_model(
                    data=ts_data,
                    holidays=holidays,
                    seasonal_period=52,
                    asin=asin  # For R/P tracking
                )

                if best_sarima_model is None:
                    print(f"SARIMA model fitting failed for {asin}, skipping.")
                    no_data_output = os.path.join(insufficient_data_folder, f"{asin}_model_failed.txt")
                    with open(no_data_output, 'w') as f:
                        f.write("Model fitting failed.\n")
                    continue

                # Extract SARIMA parameters for history
                param_tuple = best_params  # Assuming best_params is a tuple like (p,d,q,P,D,Q,m)

                # =========================
                # 2. Forecast on Test Set (Evaluation)
                # =========================
                steps = len(test_sarima)
                sarima_test_forecast_df = sarima_forecast(
                    model_fit=best_sarima_model,
                    steps=steps,
                    last_date=train_sarima['ds'].iloc[-1],
                    exog=exog_test
                )

                # Evaluate forecast on the test portion
                sarima_preds = sarima_test_forecast_df['MyForecast'].values
                sarima_mae = mean_absolute_error(test_sarima['y'], sarima_preds)
                sarima_rmse = sqrt(mean_squared_error(test_sarima['y'], sarima_preds))
                print(f"SARIMA Test MAE: {sarima_mae:.4f}, RMSE: {sarima_rmse:.4f}")

                # =========================
                # 3. Update SARIMA Parameter History
                # =========================
                update_param_history(
                    history_dict=sarima_param_history,
                    asin=asin,
                    param_tuple=param_tuple,
                    rmse=sarima_rmse,
                    mae=sarima_mae
                )

                # =========================
                # 4. Generate Final Future Forecast
                # =========================
                last_date_full = ts_data['ds'].iloc[-1]
                exog_future = generate_future_exog(holidays, steps=horizon, last_date=last_date_full)
                final_forecast_df = sarima_forecast(
                    model_fit=best_sarima_model,
                    steps=horizon,
                    last_date=train_sarima['ds'].iloc[-1],
                    exog=exog_future
                )

                if final_forecast_df.empty:
                    print(f"Forecasting failed for ASIN {asin}, skipping.")
                    no_data_output = os.path.join(insufficient_data_folder, f'{asin}_forecast_failed.txt')
                    with open(no_data_output, 'w') as f:
                        f.write('Failed to forecast due to insufficient data.\n')
                    continue

                # =========================
                # 5. Create Comparison DataFrame
                # =========================
                comparison = final_forecast_df.copy()
                comparison['ASIN'] = asin
                comparison['Product Title'] = product_title

                # Merge historical 'y' for fallback detection
                comparison = comparison.merge(ts_data[['ds', 'y']], on='ds', how='left')

                # =========================
                # 6. Blend with Amazon Forecasts Only
                # =========================
                if forecast_data:
                    for ftype, values in forecast_data.items():
                        # Load each forecast type
                        horizon_values = values[:horizon] if len(values) >= horizon else values
                        if len(horizon_values) < horizon and len(horizon_values) > 0:
                            horizon_values = np.pad(
                                horizon_values, (0, horizon - len(horizon_values)),
                                'constant', constant_values=horizon_values[-1]
                            )
                        elif len(horizon_values) == 0:
                            horizon_values = np.zeros(horizon, dtype=int)

                        # Assign to DataFrame columns
                        ftype_lower = ftype.lower()
                        if 'mean' in ftype_lower:
                            comparison['Amazon Mean Forecast'] = horizon_values
                        elif 'p70' in ftype_lower:
                            comparison['Amazon P70 Forecast'] = horizon_values
                        elif 'p80' in ftype_lower:
                            comparison['Amazon P80 Forecast'] = horizon_values
                        elif 'p90' in ftype_lower:
                            comparison['Amazon P90 Forecast'] = horizon_values
                        else:
                            print(f"Warning: Unrecognized forecast type '{ftype}'. Skipping.")

                    # Weighted blend with Amazon
                    MEAN_WEIGHT = 0.7
                    P70_WEIGHT  = 0.2
                    P80_WEIGHT  = 0.1

                    blended_amz = (
                        MEAN_WEIGHT * comparison['Amazon Mean Forecast']
                        + P70_WEIGHT * comparison.get('Amazon P70 Forecast', 0)
                        + P80_WEIGHT * comparison.get('Amazon P80 Forecast', 0)
                    ).clip(lower=0)

                    FALLBACK_RATIO = 0.3  # 30% Amazon, 70% SARIMA
                    comparison['MyForecast'] = (
                        (1 - FALLBACK_RATIO) * comparison['MyForecast']
                        + FALLBACK_RATIO * blended_amz
                    ).clip(lower=0)

                # =========================
                # 7. Generate XGBoost Forecast Separately (No Blending)
                # =========================
                if xgb_model is not None:
                    xgb_future_df = xgboost_forecast(
                        xgb_model, ts_data,
                        forecast_steps=horizon, target='y',
                        features=xgb_features
                    )
                    comparison = comparison.merge(xgb_future_df, on='ds', how='left', suffixes=('', '_XGB'))
                    comparison['MyForecast_XGB'] = comparison['MyForecast_XGB'].fillna(0)
                    print(f"XGBoost forecasts generated for ASIN {asin} and saved separately (not blended).")

                # =========================
                # 8. Adjust Forecasts if Out of Range
                # =========================
                comparison = adjust_forecast_if_out_of_range(
                    comparison, asin, forecast_col_name='MyForecast', adjustment_threshold=0.3
                )

                # =========================
                # 9. Adjust Forecast Based on Past 8-Week Sales
                # =========================
                past_8_weeks = ts_data.sort_values('ds').tail(8)
                if not past_8_weeks.empty and 'MyForecast' in comparison.columns:
                    past8_avg = past_8_weeks['y'].mean()
                    forecast_mean = comparison['MyForecast'].mean()
                    if forecast_mean > 1.5 * past8_avg:
                        print(f"Adjusting SARIMA forecast: past 8-week avg={past8_avg:.2f}, forecast mean={forecast_mean:.2f}")
                        comparison['MyForecast'] = (
                            0.8 * past8_avg + 0.2 * comparison['MyForecast']
                        ).clip(lower=0)

                # =========================
                # 10. Summaries and Visualization
                # =========================
                summary_stats, total_forecast_16, total_forecast_8, total_forecast_4, \
                max_forecast, min_forecast, max_week, min_week = calculate_summary_statistics(
                    ts_data, comparison, horizon=horizon
                )
                visualize_forecast_with_comparison(
                    ts_data, comparison, summary_stats,
                    total_forecast_16, total_forecast_8, total_forecast_4,
                    max_forecast, min_forecast, max_week, min_week,
                    asin, product_title, sufficient_data_folder
                )

                # =========================
                # 11. Log Fallback Triggers
                # =========================
                log_fallback_triggers(comparison, asin, product_title)

                # =========================
                # 12. Save Final Forecast and Summary
                # =========================
                output_file_name = f'forecast_summary_{asin}.xlsx'
                output_file_path = os.path.join(sufficient_data_folder, output_file_name)
                with pd.ExcelWriter(output_file_path, mode='w') as writer:
                    comparison.to_excel(writer, index=False)

                save_summary_to_excel(
                    comparison,
                    summary_stats,
                    total_forecast_16,
                    total_forecast_8,
                    total_forecast_4,
                    max_forecast,
                    min_forecast,
                    max_week,
                    min_week,
                    output_file_path,
                    metrics=None
                )
                consolidated_forecasts[asin] = comparison

            except ValueError as e:
                print(f"Error during SARIMA prediction for ASIN {asin}: {e}")
                continue

        # =========================
        # Prophet Block
        # =========================
        else:
            # Check for a cached model
            cached_model_path = os.path.join("model_cache", f"{asin}_Prophet.pkl")
            if os.path.exists(cached_model_path) and is_model_up_to_date(cached_model_path, ts_data):
                print(f"Using up-to-date cached Prophet model for ASIN {asin}.")
                cached_prophet_model = joblib.load(cached_model_path)
                last_train_date = ts_data['ds'].max()
                future_dates = pd.date_range(start=last_train_date + pd.Timedelta(days=7),
                                             periods=horizon, freq='W-SUN')
                future = pd.DataFrame({'ds': future_dates})
                # Add zero columns for Amazon placeholders
                for forecast_type in forecast_data.keys():
                    future[f"Amazon_{forecast_type} Forecast"] = 0
                future['prime_day'] = 0
                forecast = cached_prophet_model.predict(future)
                forecast['MyForecast'] = forecast['yhat'].round().astype(int).clip(lower=0)
            else:
                print(f"No valid cached model or outdated cache for ASIN {asin}. Training a new Prophet model...")
                best_params, _ = optimize_prophet_params(
                    ts_data=ts_data,
                    forecast_data=forecast_data,
                    param_grid=param_grid,
                    horizon=horizon,
                    asin=asin
                )
                forecast, trained_prophet_model = forecast_with_custom_params(
                    ts_data=ts_data,
                    forecast_data=forecast_data,
                    changepoint_prior_scale=best_params['changepoint_prior_scale'],
                    seasonality_prior_scale=best_params['seasonality_prior_scale'],
                    holidays_prior_scale=best_params['holidays_prior_scale'],
                    horizon=horizon
                )
                if trained_prophet_model is not None:
                    save_model(trained_prophet_model, "Prophet", asin, ts_data)
                else:
                    print("Failed to train the Prophet model.")
                    no_data_output = os.path.join(insufficient_data_folder, f"{asin}_final_forecast_failed.txt")
                    with open(no_data_output, 'w') as f:
                        f.write("Final forecasting failed.\n")
                    continue

            # If forecast is empty, skip
            if 'forecast' not in locals() or forecast.empty:
                print(f"Forecasting failed for ASIN {asin}, skipping.")
                no_data_output = os.path.join(insufficient_data_folder, f"{asin}_final_forecast_failed.txt")
                with open(no_data_output, 'w') as f:
                    f.write("Final forecasting failed.\n")
                continue

            # Format final Prophet forecast
            comparison = format_output_with_forecasts(forecast, forecast_data, horizon=horizon)
            best_weights, best_rmse = auto_find_best_weights(forecast, comparison, step=0.05)
            print(f"Auto best weights for ASIN {asin}: {best_weights} with RMSE={best_rmse}")
    
            forecast = adjust_forecast_weights(forecast.copy(), *best_weights)
            comparison = format_output_with_forecasts(forecast, forecast_data, horizon=horizon)

            print("\n--- Forecast Before Out-of-Range Adjustment ---")
            print(comparison[['MyForecast']].head(10))
            print("-----------------------------------------\n")

            # Adjust out-of-range
            comparison = adjust_forecast_if_out_of_range(comparison, asin, adjustment_threshold=0.3)

            # Log fallback triggers
            log_fallback_triggers(comparison, asin, product_title)

            # Adjust if past 8-week sales are much lower
            past_8_weeks = ts_data.sort_values('ds').tail(8)
            if not past_8_weeks.empty and 'MyForecast' in comparison.columns:
                past8_avg = past_8_weeks['y'].mean()
                forecast_mean = comparison['MyForecast'].mean()
                if forecast_mean > 1.5 * past8_avg:
                    print(f"Adjusting Prophet forecast: past 8-week avg={past8_avg:.2f}, forecast mean={forecast_mean:.2f}")
                    comparison['MyForecast'] = (
                        0.8 * past8_avg + 0.2 * comparison['MyForecast']
                    ).clip(lower=0)

            # =========================
            # Summaries and Visualization
            # =========================
            summary_stats, total_forecast_16, total_forecast_8, total_forecast_4, \
            max_forecast, min_forecast, max_week, min_week = calculate_summary_statistics(
                ts_data, comparison, horizon=horizon
            )
            visualize_forecast_with_comparison(
                ts_data, comparison, summary_stats, total_forecast_16,
                total_forecast_8, total_forecast_4, max_forecast,
                min_forecast, max_week, min_week,
                asin, product_title, sufficient_data_folder
            )

            # =========================
            # Log Fallback Triggers Again
            # =========================
            log_fallback_triggers(comparison, asin, product_title)

            # =========================
            # Ensure 'y' is in comparison
            # =========================
            if 'ds' in comparison.columns and 'y' not in comparison.columns:
                comparison = comparison.merge(ts_data[['ds', 'y']], on='ds', how='left')

            analyze_amazon_buying_habits(comparison, holidays)

            if 'y' not in comparison.columns:
                comparison = comparison.merge(ts_data[['ds', 'y']], on='ds', how='left')

            if 'y' in comparison.columns:
                comparison_historical = comparison.dropna(subset=['y'])
            else:
                comparison_historical = pd.DataFrame()

            if comparison_historical.empty:
                print("No overlapping historical data to calculate metrics. Skipping metrics.")
                metrics = {}
            else:
                MAE = mean_absolute_error(comparison_historical['y'], comparison_historical['MyForecast'])
                MEDAE = median_absolute_error(comparison_historical['y'], comparison_historical['MyForecast'])
                MSE = mean_squared_error(comparison_historical['y'], comparison_historical['MyForecast'])
                RMSE = sqrt(MSE)

                print('Mean Absolute Error (MAE): ' + str(np.round(MAE, 2)))
                print('Median Absolute Error (MedAE): ' + str(np.round(MEDAE, 2)))
                print('Mean Squared Error (MSE): ' + str(np.round(MSE, 2)))
                print('Root Mean Squared Error (RMSE): ' + str(np.round(RMSE, 2)))

                metrics = {
                    "Mean Absolute Error (MAE)": np.round(MAE, 2),
                    "Median Absolute Error (MedAE)": np.round(MEDAE, 2),
                    "Mean Squared Error (MSE)": np.round(MSE, 2),
                    "Root Mean Squared Error (RMSE)": np.round(RMSE, 2)
                }

            # =========================
            # Generate XGBoost Forecasts Separately (No Blending)
            # =========================
            if xgb_model is not None:
                xgb_future_df = xgboost_forecast(
                    xgb_model, ts_data,
                    forecast_steps=horizon, target='y',
                    features=xgb_features
                )
                comparison = comparison.merge(xgb_future_df, on='ds', how='left', suffixes=('', '_XGB'))
                comparison['MyForecast_XGB'] = comparison['MyForecast_XGB'].fillna(0)
                print(f"XGBoost forecasts generated for ASIN {asin} and saved separately (not blended).")

            # =========================
            # Final Summary & Chart
            # =========================
            summary_stats, total_forecast_16, total_forecast_8, total_forecast_4, \
            max_forecast, min_forecast, max_week, min_week = calculate_summary_statistics(
                ts_data, comparison, horizon=horizon
            )
            visualize_forecast_with_comparison(
                ts_data, comparison, summary_stats, total_forecast_16,
                total_forecast_8, total_forecast_4, max_forecast,
                min_forecast, max_week, min_week,
                asin, product_title, sufficient_data_folder
            )

            comparison['ASIN'] = asin
            comparison['Product Title'] = product_title

            # =========================
            # Save Final Forecast and Summary
            # =========================
            output_file_name = f'forecast_summary_{asin}.xlsx'
            output_file_path = os.path.join(sufficient_data_folder, output_file_name)
            with pd.ExcelWriter(output_file_path, mode='w') as writer:
                comparison.to_excel(writer, index=False)

            save_summary_to_excel(
                comparison,
                summary_stats,
                total_forecast_16,
                total_forecast_8,
                total_forecast_4,
                max_forecast,
                min_forecast,
                max_week,
                min_week,
                output_file_path,
                metrics=metrics 
            )
            consolidated_forecasts[asin] = comparison

    # =========================
    # After Processing All ASINs
    # =========================
    final_output_path = output_file
    save_forecast_to_excel(final_output_path, consolidated_forecasts, missing_asin_data, base_year=2025)

    save_feedback_to_excel(prophet_feedback, "prophet_feedback.xlsx")
    generate_4_week_report(consolidated_forecasts)
    generate_combined_weekly_report(consolidated_forecasts)

    print("\n--- Analyzing Purchase Order Data ---")
    analyze_po_data_by_asin(po_file="po database.xlsx", output_folder="po_analysis_by_asin")

    # Load PO data for comparison
    po_data = pd.read_excel("po database.xlsx")
    po_data.columns = po_data.columns.str.strip()  # Standardize column names

    print("\n--- Comparing Historical Sales with PO Data ---")
    comparison_output_folder = "po forecast comparison"
    os.makedirs(comparison_output_folder, exist_ok=True)

    # Loop through each ASIN for which forecasts were generated
    for asin, comp_df in consolidated_forecasts.items():
        print(f"\nComparing sales and PO for ASIN {asin}")
        # Filter historical sales data for the current ASIN and aggregate weekly
        sales_subset = valid_data[valid_data['asin'] == asin][['ds', 'y']].copy()
        sales_subset['Week_Period'] = sales_subset['ds'].dt.to_period('W-SUN')
        sales_subset['ds'] = sales_subset['Week_Period'].apply(lambda r: r.end_time)
        weekly_sales = sales_subset.groupby('ds', as_index=False)['y'].sum()

        # Compare historical sales with purchase order data
        merged_result, correlation, growth_info = compare_historical_sales_po(
            asin=asin,
            sales_df=weekly_sales,
            po_df=po_data,
            output_folder=comparison_output_folder
        )

        print(f"Growth Info for ASIN {asin}: {growth_info}")
        print(f"Correlation for ASIN {asin}: {correlation}")

    # Save the parameter histories at the end
    save_param_histories()

    print(f"Total number of parameter sets tested: {PARAM_COUNTER}")        
    if POOR_PARAM_FOUND:
        print("Note: Early stopping occurred for some ASINs due to poor parameter performance.")


##############################
# Run the Main Script
##############################

if __name__ == '__main__':
    main()
    summarize_usage()