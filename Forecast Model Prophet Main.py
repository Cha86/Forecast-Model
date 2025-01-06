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
    mean_absolute_percentage_error,
    mean_squared_error
)
from math import sqrt
from pykalman import KalmanFilter
from statsmodels.tsa.stattools import adfuller
import time
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from scipy.optimize import minimize
import logging

warnings.filterwarnings("ignore")

# Initialize global variables for tracking
forecast_params_used = {}
changepoint_counter = Counter()
seasonality_counter = Counter()
holiday_counter = Counter()
out_of_range_counter = Counter()
out_of_range_stats = {}

##############################
# Configuration
##############################
FALLBACK_THRESHOLD = 20  # Threshold for using SARIMA instead of Prophet
SARIMA_WEIGHT = 0.4      # Weight for SARIMA forecast if combining with other models

##############################
# Data Loading and Cleaning
##############################
def load_and_clean_data(file_path):
    """
    Load weekly sales data from an Excel file, convert Week and Year to datetime,
    and handle duplicate entries by retaining the maximum 'units_sold' per ASIN and week.

    Parameters:
        file_path (str): Path to the Excel file containing sales data.

    Returns:
        pd.DataFrame: Cleaned and aggregated sales data.
    """
    df = pd.read_excel(file_path)
    df.columns = df.columns.str.strip().str.lower()

    # Convert Week and Year to datetime (start of the week)
    def get_week_start(row):
        year = int(row['year'])
        week_str = row['week'].strip().upper()
        if week_str.startswith('W'):
            week_number = int(week_str[1:])
        else:
            week_number = int(week_str)

        # ISO weeks can go up to 53
        try:
            return pd.to_datetime(f"{year}-W{week_number:02}-1", format="%G-W%V-%u")
        except:
            # Fallback in case of invalid week numbers
            return pd.to_datetime(f"{year}-W01-1", format="%G-W%V-%u")

    df['ds'] = df.apply(get_week_start, axis=1)
    df = df.rename(columns={'units_sold': 'y'})

    # Aggregate duplicate weeks by taking the maximum 'y' per ASIN and 'ds'
    df_aggregated = df.groupby(['asin', 'ds', 'product title'], as_index=False).agg({
        'y': 'max'
    })

    # Optional: Handle outliers by capping at the 99th percentile
    upper_cap = df_aggregated['y'].quantile(0.99)
    df_aggregated['y'] = np.where(df_aggregated['y'] > upper_cap, upper_cap, df_aggregated['y'])

    # Sort data chronologically
    df_aggregated.sort_values(['asin', 'ds'], inplace=True)

    return df_aggregated

##############################
# Parameter Recording and Summarization
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
# Model Caching
##############################
def save_model(model, model_type, asin, ts_data):
    """
    Save the trained model to disk with a specific naming convention.

    Parameters:
        model: Trained model object.
        model_type (str): Type of the model (e.g., 'SARIMA', 'Prophet').
        asin (str): ASIN of the product.
        ts_data (pd.DataFrame): Time series data used for training.
    """
    model_cache_folder = "model_cache"
    os.makedirs(model_cache_folder, exist_ok=True)
    model_path = os.path.join(model_cache_folder, f"{asin}_{model_type}.pkl")
    # Store the last training date in the model object
    model.last_train_date = ts_data['ds'].max()
    joblib.dump(model, model_path)

def load_model(model_type, asin):
    """
    Load a cached model from disk if available.

    Parameters:
        model_type (str): Type of the model.
        asin (str): ASIN of the product.

    Returns:
        tuple: (model, exog) if available, else (None, None)
    """
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
    """
    Check if a cached model was trained up to the most recent data in ts_data.

    Parameters:
        cached_model_path (str): Path to the cached model.
        ts_data (pd.DataFrame): Current time series data.

    Returns:
        bool: True if up-to-date, False otherwise.
    """
    if not os.path.exists(cached_model_path):
        return False
    model = joblib.load(cached_model_path)
    if hasattr(model, 'last_train_date'):
        last_train_date = model.last_train_date
        latest_data_date = ts_data['ds'].max()
        return last_train_date >= latest_data_date
    return False

##############################
# Handling Missing Data with Kalman Filter
##############################
def kalman_smooth(series):
    """
    Use a Kalman Filter to smooth and fill missing values in the series.

    Parameters:
        series (pd.Series): Time series data.

    Returns:
        pd.Series: Smoothed and filled series.
    """
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
    """
    Handle missing data in 'y' using Kalman smoothing if enough data points exist.

    Parameters:
        data (pd.DataFrame): DataFrame containing the 'y' column.

    Returns:
        pd.DataFrame: DataFrame with filled 'y' values.
    """
    if len(data['y']) < 2:
        print("Not enough data points for Kalman smoothing. Using fallback method for missing data.")
        data['y'] = data['y'].fillna(method='bfill').fillna(method='ffill').astype(int)
    else:
        data['y'] = kalman_smooth(data['y'])
    return data

def handle_outliers(data):
    """
    Clip outliers in 'y' based on the Interquartile Range (IQR).

    Parameters:
        data (pd.DataFrame): DataFrame containing the 'y' column.

    Returns:
        pd.DataFrame: DataFrame with outliers clipped.
    """
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

    Parameters:
        df (pd.DataFrame): DataFrame containing the features.
        feature_cols (list): List of feature column names to scale.

    Returns:
        tuple: (scaled DataFrame, dictionary of scalers)
    """
    scalers = {}
    for col in feature_cols:
        scaler = MinMaxScaler()
        df[col] = scaler.fit_transform(df[[col]])
        scalers[col] = scaler
    return df, scalers

def preprocess_data(data):
    """
    Preprocess data by handling missing values and outliers.

    Parameters:
        data (pd.DataFrame): Raw data.

    Returns:
        pd.DataFrame: Preprocessed data.
    """
    data = handle_missing_data(data)
    data = handle_outliers(data)
    # Uncomment the following line if scaling is desired
    # data, scalers = min_max_scale_data(data, ['y'])
    return data

##############################
# Differencing and Stationarity
##############################
def differencing(timeseries, m):
    """
    Create differenced series for potential stationarity checks.

    Parameters:
        timeseries (pd.Series): Original time series.
        m (list): List of seasonal periods.

    Returns:
        pd.DataFrame: DataFrame containing differenced series.
    """
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
    """
    Perform Augmented Dickey-Fuller tests on multiple differenced series.

    Parameters:
        diff_series (pd.DataFrame): Differenced series.

    Returns:
        pd.DataFrame: Summary of ADF test results.
    """
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
    """
    Create exogenous holiday regressors for SARIMA.

    Parameters:
        ts_data (pd.DataFrame): Time series data.
        holidays (pd.DataFrame): Holiday data.

    Returns:
        pd.DataFrame: Exogenous variables DataFrame.
    """
    holiday_names = holidays['holiday'].unique()
    exog = pd.DataFrame(index=ts_data.index)
    exog['ds'] = ts_data['ds']
    for h in holiday_names:
        exog[h] = exog['ds'].isin(holidays[holidays['holiday'] == h]['ds']).astype(int)
    exog.drop(columns=['ds'], inplace=True)
    return exog

##############################
# SARIMA Model Fitting and Forecasting
##############################
def calculate_forecast_metrics(actual, predicted):
    """
    Calculate RMSE and MAPE between actual and predicted values.

    Parameters:
        actual (pd.Series or np.array): Actual values.
        predicted (pd.Series or np.array): Predicted values.

    Returns:
        tuple: (RMSE, MAPE)
    """
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    
    # Handle zero actuals by excluding them from MAPE calculation
    mask = actual != 0
    if np.any(mask):
        mape = mean_absolute_percentage_error(actual[mask], predicted[mask]) * 100
    else:
        mape = np.nan  # or set to a default value
    
    return rmse, mape

def fit_sarima_model(data, holidays, seasonal_period=52):
    """
    Automatically fit a SARIMA model by iterating over a set of p, d, q, P, D, Q, m values.

    Parameters:
        data (pd.DataFrame): Time series data.
        holidays (pd.DataFrame): Holiday data.
        seasonal_period (int): Seasonal period (default is 52 for weekly data).

    Returns:
        SARIMAXResults or None: Fitted SARIMA model or None if fitting fails.
    """
    exog = create_holiday_regressors(data, holidays)
    sample_size = len(data)
    if sample_size < 52:
        if sample_size >= 12:
            seasonal_period = 12
            print(f"Dataset too small for m=52. Adjusting seasonal_period to m={seasonal_period} based on data size.")
        elif sample_size >= 4:
            seasonal_period = None
            print("Dataset too small for robust seasonal modeling. Using non-seasonal SARIMA.")
        else:
            seasonal_period = 1
            print(f"Dataset too small. Adjusting seasonal_period to m={seasonal_period}.")
    
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

    for p in p_values:
        for d in d_values:
            for q in q_values:
                if seasonal:
                    for P in P_values:
                        for D in D_values:
                            for Q in Q_values:
                                for m in m_values:
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
                                        rmse, mape = calculate_forecast_metrics(actual, forecast)
                                        if rmse < best_rmse:
                                            best_rmse = rmse
                                            best_model = model_fit
                                            best_metrics = {'RMSE': rmse, 'MAPE': mape,
                                                            'p': p, 'd': d, 'q': q,
                                                            'P': P, 'D': D, 'Q': Q, 'm': m}
                                    except:
                                        continue
                else:
                    try:
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
                        rmse, mape = calculate_forecast_metrics(actual, forecast)
                        if rmse < best_rmse:
                            best_rmse = rmse
                            best_model = model_fit
                            best_metrics = {'RMSE': rmse, 'MAPE': mape,
                                            'p': p, 'd': d, 'q': q,
                                            'P': 0, 'D': 0, 'Q': 0, 'm': 1}
                    except:
                        continue

    if best_model is None:
        print("No suitable SARIMA model found.")
        return None
    else:
        print(f"Best SARIMA Model: (p,d,q)=({best_metrics['p']},{best_metrics['d']},{best_metrics['q']}), "
              f"(P,D,Q,m)=({best_metrics['P']},{best_metrics['D']},{best_metrics['Q']},{best_metrics['m']}), "
              f"RMSE={best_metrics['RMSE']:.2f}, MAPE={best_metrics['MAPE']:.2f}%")
        return best_model

def sarima_forecast(model_fit, steps, last_date, exog=None):
    """
    Generate SARIMA forecasts for the specified number of steps ahead.

    Parameters:
        model_fit (SARIMAXResults): Fitted SARIMA model.
        steps (int): Number of steps to forecast.
        last_date (pd.Timestamp): Last date in the training data.
        exog (pd.DataFrame or None): Exogenous variables for forecasting.

    Returns:
        pd.DataFrame: DataFrame containing forecast dates and values.
    """
    if exog is not None and not exog.empty:
        forecast_values = model_fit.forecast(steps=steps, exog=exog)
    else:
        forecast_values = model_fit.forecast(steps=steps)
    forecast_values = forecast_values.round().astype(int)
    future_dates = pd.date_range(start=last_date + pd.Timedelta(weeks=1), periods=steps, freq='W')
    return pd.DataFrame({'ds': future_dates, 'SARIMA Forecast': forecast_values})

def generate_future_exog(holidays, steps, last_date):
    """
    Generate exogenous holiday regressors for future forecasts.

    Parameters:
        holidays (pd.DataFrame): Holiday data.
        steps (int): Number of future steps.
        last_date (pd.Timestamp): Last date in the training data.

    Returns:
        pd.DataFrame: Exogenous variables for future dates.
    """
    future_dates = pd.date_range(start=last_date + pd.Timedelta(weeks=1), periods=steps, freq='W')
    exog_future = pd.DataFrame(index=future_dates)
    holiday_names = holidays['holiday'].unique()
    for h in holiday_names:
        exog_future[h] = exog_future.index.isin(holidays[holidays['holiday'] == h]['ds']).astype(int)
    return exog_future

##############################
# ASIN and Forecast Data Loading
##############################
def generate_date_from_week(row):
    """
    Convert year-week format into a datetime object for the beginning of that week.

    Parameters:
        row (pd.Series): Row containing 'week' and 'year'.

    Returns:
        pd.Timestamp: Start date of the week.
    """
    week_str = row['week']
    year = row['year']
    week_number = int(week_str[1:])  # Assuming week format like 'W01'
    return pd.to_datetime(f'{year}-W{week_number - 1}-0', format='%Y-W%U-%w')

def clean_weekly_sales_data(data):
    """Placeholder for additional cleaning steps if needed."""
    return data

def load_weekly_sales_data(file_path):
    """
    Load weekly sales data from Excel, ensuring required columns are present.

    Parameters:
        file_path (str): Path to the sales data Excel file.

    Returns:
        pd.DataFrame: Loaded and partially cleaned data.
    """
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
    """
    Load a list of ASINs from either a text or Excel file.

    Parameters:
        file_path (str): Path to the ASINs file.

    Returns:
        list: List of ASINs to forecast.
    """
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
    Each file should contain columns with weeks (WXX) and an 'ASIN' column.

    Parameters:
        folder_path (str): Path to the folder containing forecast Excel files.
        asin (str): ASIN to load forecasts for.

    Returns:
        dict: Dictionary with forecast types as keys and forecast values as numpy arrays.
    """
    forecast_data = {}
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.xlsx'):
            # Extract forecast type without ' Forecast' suffix
            forecast_type = os.path.splitext(file_name)[0].replace('_', ' ').title()
            if forecast_type.endswith(' Forecast'):
                forecast_type = forecast_type.replace(' Forecast', '')  # Remove if present
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

##############################
# Feature Engineering for XGBoost
##############################
def add_lag_features(ts_data, lag_weeks=1):
    """
    Add lag features to the time series DataFrame.

    Parameters:
        ts_data (pd.DataFrame): Time series data.
        lag_weeks (int): Number of weeks to lag.

    Returns:
        pd.DataFrame: DataFrame with added lag features.
    """
    ts_data = ts_data.copy()
    ts_data = ts_data.sort_values('ds')
    ts_data[f'lag_{lag_weeks}_week'] = ts_data['y'].shift(lag_weeks)
    ts_data[f'lag_{lag_weeks}_week'].fillna(0, inplace=True)
    return ts_data

def prepare_time_series_with_lags(data, asin, lag_weeks=1, holidays=None):
    """
    Prepare a single ASIN's data for modeling:
    - Rename columns
    - Sort by date
    - Preprocess (missing/outliers)
    - Add lag features

    Parameters:
        data (pd.DataFrame): Cleaned and aggregated sales data.
        asin (str): ASIN to prepare data for.
        lag_weeks (int): Number of weeks to lag.
        holidays (pd.DataFrame): Holiday data.

    Returns:
        pd.DataFrame: Prepared time series data with lag features.
    """
    ts_data = data[data['asin'] == asin].copy()
    ts_data = ts_data.sort_values('ds')
    ts_data = preprocess_data(ts_data)
    ts_data = add_lag_features(ts_data, lag_weeks)
    return ts_data

##############################
# Holiday Data
##############################
def get_shifted_holidays():
    """
    Example holiday DataFrame with 'ds' as date and 'holiday' as holiday name.
    Adjust to match your actual holiday data.

    Returns:
        pd.DataFrame: DataFrame containing holiday information.
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
# XGBoost Training and Forecasting
##############################
def train_xgboost(ts_data, target='y', features=None):
    """
    Train an XGBoost model on the provided ts_data, using specified features for regression.

    Parameters:
        ts_data (pd.DataFrame): Time series data with features.
        target (str): Target column name.
        features (list or None): List of feature column names. If None, default to lag features.

    Returns:
        tuple: (trained model, feature names, SHAP values)
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

    # Initialize XGBoost regressor with hyperparameters
    model = xgb.XGBRegressor(
        n_estimators=200,       # Increased from 100
        learning_rate=0.05,     # Decreased from 0.1
        max_depth=4,            # Slightly deeper tree
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )
    model.fit(X_train, y_train, 
              eval_set=[(X_test, y_test)], 
              verbose=False)

    # Calculate SHAP values to evaluate feature importance
    explainer = shap.Explainer(model)
    shap_values = explainer(X)

    # Summarize feature importance (plot suppressed in script)
    shap.summary_plot(shap_values, X, show=False)

    return model, features, shap_values

def xgboost_forecast(model, ts_data, forecast_steps=16, target='y', features=None):
    """
    Generate XGBoost forecasts for the specified future steps using last known features as the basis.
    Because XGBoost is not a time-series model by default, we do a naive approach:
      1. We'll shift the lag features step-by-step and predict each week in sequence.

    Parameters:
        model (XGBRegressor): Trained XGBoost model.
        ts_data (pd.DataFrame): Time series data with features.
        forecast_steps (int): Number of future steps to forecast.
        target (str): Target column name.
        features (list or None): List of feature column names.

    Returns:
        pd.DataFrame: DataFrame containing forecast dates and values.
    """
    if features is None:
        features = [col for col in ts_data.columns if col.startswith('lag_')]

    # Sort data by date
    ts_data = ts_data.sort_values('ds').copy()
    last_date = ts_data['ds'].max()

    future_dates = pd.date_range(start=last_date + pd.Timedelta(weeks=1), periods=forecast_steps, freq='W')
    forecasts = []

    current_data = ts_data.copy()

    for i in range(forecast_steps):
        # Prepare row for prediction
        X_pred = current_data.iloc[[-1]][features]
        # Predict next step
        y_pred = model.predict(X_pred)[0]
        # Round & clip negative
        y_pred = max(int(round(y_pred)), 0)

        # Store forecast
        forecast_date = future_dates[i]
        forecasts.append((forecast_date, y_pred))

        # Update dataset for next iteration
        for feat in features:
            if 'lag_' in feat:
                lag_num = int(feat.split('_')[1])  # e.g. 'lag_1_week' -> 1
                if lag_num == 1:
                    # This feature takes the newly predicted value
                    current_data.loc[current_data.index[-1], feat] = y_pred
                else:
                    # Shift from a previous lag
                    prev_feat = f'lag_{lag_num-1}_week'
                    current_data.loc[current_data.index[-1], feat] = current_data.loc[current_data.index[-1], prev_feat]

        # Append a new row with the predicted y
        new_row = current_data.iloc[[-1]].copy()
        new_row['ds'] = forecast_date
        new_row[target] = y_pred
        current_data = pd.concat([current_data, new_row], ignore_index=True)

    df_forecasts = pd.DataFrame(forecasts, columns=['ds', 'XGBoost Forecast'])
    return df_forecasts

##############################
# Ensemble Approach (SARIMA, Prophet, XGBoost)
##############################
def ensemble_forecast(sarima_preds, prophet_preds, xgb_preds, amazon_mean_preds, weights):
    """
    Weighted ensemble of forecasts from SARIMA, Prophet, XGBoost, and Amazon Mean.

    Parameters:
        sarima_preds (pd.Series or np.array): SARIMA forecast values.
        prophet_preds (pd.Series or np.array): Prophet forecast values.
        xgb_preds (pd.Series or np.array): XGBoost forecast values.
        amazon_mean_preds (pd.Series or np.array): Amazon Mean forecast values.
        weights (list): Weights for [SARIMA, Prophet, XGBoost, Amazon Mean].

    Returns:
        pd.Series or np.array: Ensemble forecast values.
    """
    return (
        weights[0] * sarima_preds
        + weights[1] * prophet_preds
        + weights[2] * xgb_preds
        + weights[3] * amazon_mean_preds
    )

def evaluate_forecast(actual, forecast):
    """
    Compute RMSE for forecast evaluation.

    Parameters:
        actual (array-like): Actual values.
        forecast (array-like): Forecasted values.

    Returns:
        float: RMSE value.
    """
    rmse = sqrt(mean_squared_error(actual, forecast))
    return rmse

def walk_forward_validation_ensemble(ts_data, n_test, model_sarima, model_prophet, model_xgb, weights):
    """
    Example walk-forward validation for an ensemble approach.

    Parameters:
        ts_data (pd.DataFrame): Time series data.
        n_test (int): Number of test samples.
        model_sarima: Trained SARIMA model.
        model_prophet: Trained Prophet model.
        model_xgb: Trained XGBoost model.
        weights (list): Weights for [SARIMA, Prophet, XGBoost].

    Returns:
        tuple: (RMSE, predictions, actuals)
    """
    predictions = []
    actuals = []
    train = ts_data.iloc[:-n_test].copy()
    test = ts_data.iloc[-n_test:].copy()

    for i in range(n_test):
        test_point = test.iloc[i]

        # SARIMA forecast for next step
        if model_sarima is not None:
            try:
                sarima_forecast = model_sarima.forecast(steps=1)
                sarima_value = int(round(sarima_forecast.iloc[0]))
            except:
                sarima_value = 0
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

        # Amazon Mean Forecast (assuming it's available)
        amazon_mean = test_point.get('Amazon Mean Forecast', 0)
        amazon_mean = int(round(max(amazon_mean, 0)))

        # Ensemble prediction
        ensemble_pred = ensemble_forecast(sarima_value, prophet_value, xgb_value, amazon_mean, weights)
        ensemble_pred = int(round(ensemble_pred))

        predictions.append(ensemble_pred)
        actuals.append(test_point['y'])

    rmse = evaluate_forecast(actuals, predictions)
    return rmse, predictions, actuals

def print_ensemble_summary(rmse, predictions, actuals):
    """
    Display a user-friendly summary of the ensemble results.

    Parameters:
        rmse (float): RMSE value.
        predictions (list): List of forecasted values.
        actuals (list): List of actual values.
    """
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

    Parameters:
        sarima_rmse (float): RMSE for SARIMA.
        prophet_rmse (float): RMSE for Prophet.
        xgb_rmse (float): RMSE for XGBoost.
        ensemble_rmse (float): RMSE for Ensemble.

    Returns:
        pd.DataFrame: Decision matrix DataFrame.
    """
    data = {
        'Model': ['SARIMA', 'Prophet', 'XGBoost', 'Ensemble'],
        'RMSE': [sarima_rmse, prophet_rmse, xgb_rmse, ensemble_rmse]
    }
    df = pd.DataFrame(data)
    df.sort_values(by='RMSE', inplace=True)
    return df

##############################
# Prophet Customization and Forecasting
##############################
def forecast_with_custom_params(ts_data, forecast_data,
                                changepoint_prior_scale,
                                seasonality_prior_scale,
                                holidays_prior_scale,
                                horizon=16,
                                weights=None):
    """
    Train and forecast with Prophet using user-defined hyperparams and custom weighting for regressors.

    Parameters:
        ts_data (pd.DataFrame): Training time series data.
        forecast_data (dict): Amazon forecast data.
        changepoint_prior_scale (float): Prophet changepoint prior scale.
        seasonality_prior_scale (float): Prophet seasonality prior scale.
        holidays_prior_scale (float): Prophet holidays prior scale.
        horizon (int): Number of weeks to forecast.
        weights (dict or None): Weights for Amazon forecast types.

    Returns:
        tuple: (forecast DataFrame, trained Prophet model)
    """
    # Default weights if not provided
    weights = weights or {
        'Mean': 0.7,
        'P70': 0.2,
        'P80': 0.1
    }

    future_dates = pd.date_range(start=ts_data['ds'].max() + pd.Timedelta(days=7), periods=horizon, freq='W')
    future = pd.DataFrame({'ds': future_dates})
    combined_df = pd.concat([ts_data, future], ignore_index=True)

    # Attach forecast_data (Amazon forecasts) as regressors
    for forecast_type, values in forecast_data.items():
        values = values[:horizon]
        if len(values) < horizon:
            # Pad with the last value to match horizon
            if len(values) > 0:
                values = np.pad(values, (0, horizon - len(values)), 'constant', constant_values=values[-1])
            else:
                values = np.zeros(horizon, dtype=int)
        
        # **Identify the forecasted rows only**
        # Get the indices of future dates
        future_mask = combined_df['ds'].isin(future_dates)
        # Assign Amazon forecast values only to the future rows
        combined_df.loc[future_mask, f'Amazon_{forecast_type} Forecast'] = values

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
        test_eval['Prophet Forecast'] = test_eval['yhat'].round().astype(int)
        # Ensure no non-finite values
        test_eval['Prophet Forecast'].replace([np.inf, -np.inf], np.nan, inplace=True)
        test_eval['Prophet Forecast'].fillna(0, inplace=True)
        test_eval['Prophet Forecast'] = test_eval['Prophet Forecast'].astype(int)
        mape = mean_absolute_percentage_error(test_actual['y'], test_eval['Prophet Forecast'])
        rmse_val = sqrt(mean_squared_error(test_actual['y'], test_eval['Prophet Forecast']))
        print(f"Prophet Test MAPE: {mape:.4f}, RMSE: {rmse_val:.4f}")

    future_df = combined_df[combined_df['y'].isna()].drop(columns='y').copy()
    forecast = model.predict(future_df)
    # Replace non-finite values in 'yhat' and 'yhat_upper' before conversion
    forecast['yhat'].replace([np.inf, -np.inf], np.nan, inplace=True)
    forecast['yhat'].fillna(0, inplace=True)
    forecast['yhat_upper'].replace([np.inf, -np.inf], np.nan, inplace=True)
    forecast['yhat_upper'].fillna(forecast['yhat'], inplace=True)  # Use 'yhat' if 'yhat_upper' is NaN

    # Ensure no non-finite values before conversion
    if not np.all(np.isfinite(forecast['yhat'])):
        print("Warning: Non-finite values detected in 'yhat'. Replacing with 0.")
        forecast['yhat'] = forecast['yhat'].fillna(0).clip(lower=0)
    if not np.all(np.isfinite(forecast['yhat_upper'])):
        print("Warning: Non-finite values detected in 'yhat_upper'. Replacing with 'yhat'.")
        forecast['yhat_upper'] = forecast['yhat'].fillna(0)

    forecast['Prophet Forecast'] = forecast['yhat'].round().astype(int).clip(lower=0)

    # Apply custom weighting mechanism to adjust forecasts
    try:
        for amazon_col, weight in weights.items():
            if f'Amazon_{amazon_col} Forecast' in future_df.columns:
                # Ensure the 'Amazon_* Forecast' values align with forecast rows
                # Here, 'future_df' corresponds to 'forecast', so indexing aligns
                forecast.loc[:, 'yhat'] += weight * future_df[f'Amazon_{amazon_col} Forecast']
    except KeyError as e:
        print(f"Warning: Missing Amazon forecast column during weighting adjustment: {e}")

    # Replace non-finite values after weighting
    forecast['yhat'].replace([np.inf, -np.inf], np.nan, inplace=True)
    forecast['yhat'].fillna(0, inplace=True)

    forecast['Prophet Forecast'] = forecast['yhat'].round().astype(int).clip(lower=0)

    return forecast[['ds', 'Prophet Forecast', 'yhat', 'yhat_upper']], model


##############################
# Prophet Parameter Optimization
##############################
PARAM_COUNTER = 0
POOR_PARAM_FOUND = False
EARLY_STOP_THRESHOLD = 10_000

def optimize_prophet_params(ts_data, forecast_data, param_grid, horizon=16):
    """
    Optimize Prophet hyperparameters by grid search to minimize RMSE across Amazon forecast types.

    Parameters:
        ts_data (pd.DataFrame): Training time series data.
        forecast_data (dict): Amazon forecast data.
        param_grid (dict): Dictionary containing lists of hyperparameters to try.
        horizon (int): Number of weeks to forecast.

    Returns:
        tuple: (best_params dict, best_rmse_values dict)
    """
    global changepoint_counter, seasonality_counter, holiday_counter
    global PARAM_COUNTER, POOR_PARAM_FOUND
    best_rmse = float('inf')
    best_params = None
    best_rmse_values = None
    for changepoint_prior_scale in param_grid['changepoint_prior_scale']:
        for seasonality_prior_scale in param_grid['seasonality_prior_scale']:
            for holidays_prior_scale in param_grid['holidays_prior_scale']:
                PARAM_COUNTER += 1
                print(f"Testing Params #{PARAM_COUNTER}: changepoint={changepoint_prior_scale}, "
                      f"seasonality={seasonality_prior_scale}, holidays={holidays_prior_scale}")
                try:
                    forecast, _ = forecast_with_custom_params(
                        ts_data, forecast_data,
                        changepoint_prior_scale,
                        seasonality_prior_scale,
                        holidays_prior_scale,
                        horizon=horizon,
                        weights={
                            'Mean': 0.7,
                            'P70': 0.2,
                            'P80': 0.1
                        }
                    )
                    if forecast.empty or 'Prophet Forecast' not in forecast.columns:
                        raise ValueError("Forecast failed to generate 'Prophet Forecast'.")
                    rmse_values = calculate_rmse(forecast, forecast_data, horizon)
                    avg_rmse = np.mean(list(rmse_values.values()))
                    if avg_rmse > EARLY_STOP_THRESHOLD:
                        print("Early stopping triggered due to poor parameter performance.")
                        POOR_PARAM_FOUND = True
                        return best_params, best_rmse_values
                    if avg_rmse < best_rmse:
                        best_rmse = avg_rmse
                        best_params = {
                            'changepoint_prior_scale': changepoint_prior_scale,
                            'seasonality_prior_scale': seasonality_prior_scale,
                            'holidays_prior_scale': holidays_prior_scale
                        }
                        best_rmse_values = rmse_values
                except Exception as e:
                    print(f"Error during optimization: {e}")
                    continue
    if best_params is None:
        print("Optimization failed. Using default parameters.")
        best_params = {'changepoint_prior_scale': 0.1, 'seasonality_prior_scale': 1, 'holidays_prior_scale': 10}
        best_rmse_values = {}
    return best_params, best_rmse_values

def calculate_rmse(forecast, forecast_data, horizon):
    """
    Calculate RMSE comparing Prophet forecast with various Amazon forecast streams.

    Parameters:
        forecast (pd.DataFrame): Forecast DataFrame containing 'Prophet Forecast'.
        forecast_data (dict): Amazon forecast data.
        horizon (int): Number of weeks to forecast.

    Returns:
        dict: RMSE values for each Amazon forecast type.
    """
    rmse_values = {}
    for forecast_type, values in forecast_data.items():
        values = values[:horizon]
        if len(values) < horizon:
            # Pad with the last value to match horizon
            if len(values) > 0:
                values = np.pad(values, (0, horizon - len(values)), 'constant', constant_values=values[-1])
            else:
                values = np.zeros(horizon, dtype=int)
        rmse = np.sqrt(((forecast['Prophet Forecast'] - values) ** 2).mean())
        rmse_values[forecast_type] = rmse
    return rmse_values

##############################
# Forecast Formatting and Adjustments
##############################
def format_output_with_forecasts(prophet_forecast, forecast_data, horizon=16):
    """
    Combine Prophet forecast with Amazon forecast data for easy comparison.
    Also compute the difference and percentage difference columns.

    Parameters:
        prophet_forecast (pd.DataFrame): DataFrame containing Prophet forecasts.
        forecast_data (dict): Amazon forecast data.
        horizon (int): Number of weeks to forecast.

    Returns:
        pd.DataFrame: Combined forecast comparison DataFrame.
    """
    comparison = prophet_forecast.copy()
    for forecast_type, values in forecast_data.items():
        values = values[:horizon]
        values = np.array(values, dtype=int)
        # Check if 'Forecast' is already in forecast_type
        if 'Forecast' in forecast_type:
            amazon_col_name = f'Amazon {forecast_type}'
        else:
            amazon_col_name = f'Amazon {forecast_type} Forecast'
        forecast_df = pd.DataFrame({
            'ds': prophet_forecast['ds'],
            amazon_col_name: values
        })
        comparison = comparison.merge(forecast_df, on='ds', how='left')
    comparison.fillna(0, inplace=True)
    for col in comparison.columns:
        if col.startswith('Amazon ') and col.endswith('Forecast'):
            base_name = col.replace('Amazon ', '').replace(' Forecast', '')
            diff_col = f"Diff_{base_name}"
            pct_col = f"Pct_{base_name}"
            comparison[diff_col] = (comparison['Prophet Forecast'] - comparison[col]).astype(int)
            comparison[pct_col] = np.where(
                comparison[col] != 0,
                (comparison[diff_col] / comparison[col]) * 100,
                0
            )
    comparison['Prophet Forecast'] = comparison['Prophet Forecast'].astype(int)
    for col in comparison.columns:
        if col.startswith("Amazon ") or col.startswith("Diff_") or col.startswith("Pct_"):
            comparison[col] = comparison[col].astype(int, errors='ignore')
    return comparison

def adjust_forecast_weights(forecast, yhat_weight, yhat_upper_weight):
    """
    Adjust final Prophet forecast by combining yhat and yhat_upper with specified weights.

    Parameters:
        forecast (pd.DataFrame): Forecast DataFrame containing 'yhat' and 'yhat_upper'.
        yhat_weight (float): Weight for 'yhat'.
        yhat_upper_weight (float): Weight for 'yhat_upper'.

    Returns:
        pd.DataFrame: Forecast DataFrame with adjusted 'Prophet Forecast'.
    """
    if 'yhat' not in forecast or 'yhat_upper' not in forecast:
        raise KeyError("'yhat' or 'yhat_upper' not found in forecast DataFrame.")
    adj_forecast = (
        yhat_weight * forecast['yhat'] + yhat_upper_weight * forecast['yhat_upper']
    ).clip(lower=0)
    adj_forecast = adj_forecast.round().astype(int)
    forecast['Prophet Forecast'] = adj_forecast
    return forecast

def find_best_forecast_weights(forecast, comparison, weights):
    """
    Brute force search over possible (yhat_weight, yhat_upper_weight) pairs to minimize average RMSE vs Amazon columns.

    Parameters:
        forecast (pd.DataFrame): Forecast DataFrame.
        comparison (pd.DataFrame): Comparison DataFrame containing Amazon forecasts.
        weights (list): List of tuples representing (yhat_weight, yhat_upper_weight).

    Returns:
        tuple: (best_weights, rmse_results)
    """
    best_rmse = float('inf')
    best_weights = None
    rmse_results = {}
    for w in weights:
        yhat_weight, yhat_upper_weight = w
        adjusted_forecast = adjust_forecast_weights(forecast.copy(), yhat_weight, yhat_upper_weight)
        rmse_values = {}
        for amazon_col in comparison.columns:
            if amazon_col.startswith('Amazon '):
                rmse = np.sqrt(((comparison[amazon_col] - adjusted_forecast['Prophet Forecast']) ** 2).mean())
                rmse_values[amazon_col] = rmse
        avg_rmse = np.mean(list(rmse_values.values()))
        rmse_results[w] = avg_rmse
        if avg_rmse < best_rmse:
            best_rmse = avg_rmse
            best_weights = w
    return best_weights, rmse_results

def auto_find_best_weights(forecast, comparison, step=0.05):
    """
    Automatically scan the range [0.5, 1.0] in increments of 'step' to find the best combination of yhat/yhat_upper weights.

    Parameters:
        forecast (pd.DataFrame): Forecast DataFrame.
        comparison (pd.DataFrame): Comparison DataFrame containing Amazon forecasts.
        step (float): Increment step for scanning weights.

    Returns:
        tuple: (best_weights, best_rmse)
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
                rmse = np.sqrt(((comparison[amazon_col] - adjusted_forecast['Prophet Forecast']) ** 2).mean())
                rmse_values[amazon_col] = rmse
        avg_rmse = np.mean(list(rmse_values.values()))
        if avg_rmse < best_rmse:
            best_rmse = avg_rmse
            best_weights = (yhat_weight, yhat_upper_weight)
    return best_weights, best_rmse

##############################
# Cross Validation for Prophet
##############################
def cross_validate_prophet_model(ts_data, initial='365 days', period='180 days', horizon='180 days'):
    """
    Run Prophet's built-in cross-validation and performance metrics.

    Parameters:
        ts_data (pd.DataFrame): Time series data.
        initial (str): Initial training period.
        period (str): Frequency of cross-validation.
        horizon (str): Forecast horizon.

    Returns:
        tuple: (cross-validation DataFrame, performance metrics DataFrame)
    """
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
# Amazon vs. Prophet Analysis
##############################
def analyze_amazon_buying_habits(comparison, holidays):
    """
    Analyze Amazon forecast columns compared to Prophet's forecast,
    highlighting holiday weeks and segmenting short-, mid-, and long-term.

    Parameters:
        comparison (pd.DataFrame): Forecast comparison DataFrame.
        holidays (pd.DataFrame): Holiday data.
    """
    # Check if 'y' column exists
    if 'y' not in comparison.columns:
        print("Warning: 'y' column missing in the comparison DataFrame. Skipping analysis.")
        return

    # Check for Amazon forecast columns
    amazon_cols = [
        col for col in comparison.columns
        if col.startswith('Amazon ') and col.endswith('Forecast') and not col.endswith('Forecast Forecast')
    ]
    if not amazon_cols:
        print("No Amazon forecasts found for analysis.")
        return

    # Prepare data for analysis
    prophet_forecast = comparison.get('Prophet Forecast', pd.Series(index=comparison.index))
    ds_dates = comparison.get('ds', pd.Series(index=comparison.index))
    holiday_dates = holidays['ds'].values if holidays is not None else []
    comparison['is_holiday_week'] = ds_dates.isin(holiday_dates) if 'ds' in comparison.columns else False

    # Improve error analysis for Amazon forecast types
    amazon_types = ['Mean', 'P70', 'P80', 'P90']
    errors = {}
    for forecast_type in amazon_types:
        forecast_col = f"Amazon {forecast_type} Forecast"
        if forecast_col in comparison.columns:
            current_rmse = mean_squared_error(comparison['y'], comparison[forecast_col], squared=False)
            current_mape = mean_absolute_percentage_error(comparison['y'], comparison[forecast_col]) * 100
            errors[forecast_type] = {
                'RMSE': current_rmse,
                'MAPE': current_mape
            }

    # Output best forecast type
    if errors:
        best_forecast_type = min(errors, key=lambda x: errors[x]['RMSE'])
        print(f"\nBest Amazon forecast type: {best_forecast_type} with RMSE={errors[best_forecast_type]['RMSE']:.4f}")
    
    # Analyze forecasts
    for forecast_type in amazon_types:
        forecast_col = f"Amazon {forecast_type} Forecast"
        if forecast_col not in comparison.columns:
            continue
        amazon_forecast = comparison[forecast_col].values
        safe_prophet = np.where(prophet_forecast == 0, 1e-9, prophet_forecast)
        ratio = amazon_forecast / safe_prophet
        diff = amazon_forecast - prophet_forecast
        avg_ratio = np.mean(ratio)
        avg_diff = np.mean(diff)
        print(f"\nFor {forecast_type}:")
        print(f"  Average Amazon/Prophet Ratio: {avg_ratio:.2f}")
        print(f"  Average Difference (Amazon - Prophet): {avg_diff:.2f}")

        if avg_diff > 0:
            print("  Amazon tends to forecast more than Prophet on average.")
        elif avg_diff < 0:
            print("  Amazon tends to forecast less than Prophet on average.")
        else:
            print("  Amazon forecasts similarly to Prophet on average.")

        # Analyze holiday weeks
        holiday_mask = comparison['is_holiday_week']
        if holiday_mask.any():
            holiday_ratio = ratio[holiday_mask]
            holiday_diff = diff[holiday_mask]
            if len(holiday_diff) > 0:
                print("  During holiday weeks:")
                print(f"    Avg Ratio (Amazon/Prophet): {np.mean(holiday_ratio):.2f}")
                print(f"    Avg Diff (Amazon-Prophet): {np.mean(holiday_diff):.2f}")

        # Segment analysis
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
                print(f"    Avg Ratio (Amazon/Prophet): {np.mean(seg_ratio):.2f}")
                print(f"    Avg Diff (Amazon-Prophet): {np.mean(seg_diff):.2f}")

##############################
# Forecast Summary Statistics and Visualization
##############################
def calculate_summary_statistics(ts_data, forecast_df, horizon):
    """
    Calculate basic summary statistics for historical data and partial sums for forecast.

    Parameters:
        ts_data (pd.DataFrame): Historical time series data.
        forecast_df (pd.DataFrame): Forecast comparison DataFrame.
        horizon (int): Number of weeks forecasted.

    Returns:
        tuple: (summary_stats dict, total_forecast_16, total_forecast_8, total_forecast_4,
                max_forecast, min_forecast, max_week, min_week)
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

    total_forecast_16 = forecast_df['Prophet Forecast'][:16].sum() if len(forecast_df) >= 16 else forecast_df['Prophet Forecast'].sum()
    total_forecast_8 = forecast_df['Prophet Forecast'][:8].sum() if len(forecast_df) >= 8 else forecast_df['Prophet Forecast'].sum()
    total_forecast_4 = forecast_df['Prophet Forecast'][:4].sum() if len(forecast_df) >= 4 else forecast_df['Prophet Forecast'].sum()

    max_forecast = forecast_df['Prophet Forecast'].max()
    min_forecast = forecast_df['Prophet Forecast'].min()
    max_week = forecast_df.loc[forecast_df['Prophet Forecast'].idxmax(), 'ds'] if 'ds' in forecast_df.columns else None
    min_week = forecast_df.loc[forecast_df['Prophet Forecast'].idxmin(), 'ds'] if 'ds' in forecast_df.columns else None
    return summary_stats, total_forecast_16, total_forecast_8, total_forecast_4, max_forecast, min_forecast, max_week, min_week

def visualize_forecast_with_comparison(ts_data, comparison, summary_stats,
                                       total_forecast_16, total_forecast_8, total_forecast_4,
                                       max_forecast, min_forecast, max_week, min_week,
                                       asin, product_title, folder_name):
    """
    Visualize historical data vs. forecast (Prophet and Amazon) and annotate with summary statistics.

    Parameters:
        ts_data (pd.DataFrame): Historical time series data.
        comparison (pd.DataFrame): Forecast comparison DataFrame.
        summary_stats (dict): Summary statistics dictionary.
        total_forecast_16 (int): Total forecast for 16 weeks.
        total_forecast_8 (int): Total forecast for 8 weeks.
        total_forecast_4 (int): Total forecast for 4 weeks.
        max_forecast (int): Maximum forecasted value.
        min_forecast (int): Minimum forecasted value.
        max_week (pd.Timestamp or None): Week of maximum forecast.
        min_week (pd.Timestamp or None): Week of minimum forecast.
        asin (str): ASIN of the product.
        product_title (str): Title of the product.
        folder_name (str): Folder to save the visualization.
    """
    fig, ax = plt.subplots(figsize=(16, 10))
    ax.plot(ts_data['ds'], ts_data['y'], label='Historical Sales', marker='o', color='black')
    ax.plot(comparison['ds'], comparison['Prophet Forecast'], label='Prophet Forecast', marker='o', linestyle='--', color='blue')

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
# Excel Output Functions
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
                          metrics=None):
    """
    Save forecast comparison and summary statistics to an Excel file.

    Parameters:
        comparison (pd.DataFrame): Forecast comparison DataFrame.
        summary_stats (dict): Summary statistics dictionary.
        total_forecast_16 (int): Total forecast for 16 weeks.
        total_forecast_8 (int): Total forecast for 8 weeks.
        total_forecast_4 (int): Total forecast for 4 weeks.
        max_forecast (int): Maximum forecasted value.
        min_forecast (int): Minimum forecasted value.
        max_week (pd.Timestamp or None): Week of maximum forecast.
        min_week (pd.Timestamp or None): Week of minimum forecast.
        output_file_path (str): Path to save the Excel file.
        metrics (dict or None): Dictionary of performance metrics.
    """
    desired_columns = [
        'Week', 'ASIN', 'Prophet Forecast', 'Amazon Mean Forecast',
        'Amazon P70 Forecast', 'Amazon P80 Forecast', 'Amazon P90 Forecast',
        'Product Title', 'is_holiday_week'
    ]

    comparison_for_excel = comparison.copy()
    if 'ds' in comparison_for_excel.columns:
        for i in range(len(comparison_for_excel)):
            comparison_for_excel.loc[i, 'Week'] = f"W{str(i+1).zfill(2)}"
        comparison_for_excel.drop(columns=['ds'], inplace=True, errors='ignore')

    for col in desired_columns:
        if col not in comparison_for_excel.columns:
            comparison_for_excel[col] = np.nan

    comparison_for_excel = comparison_for_excel[desired_columns]

    wb = Workbook()
    ws1 = wb.active
    ws1.title = "Forecast Comparison"
    for r in dataframe_to_rows(comparison_for_excel, index=False, header=True):
        ws1.append(r)

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

def save_forecast_to_excel(output_path, consolidated_data, missing_asin_data):
    """
    Save multiple ASIN forecasts and any missing ASIN data into one Excel file,
    each ASIN in a separate sheet.
    Also create a "4-8-16 Weeks Summary" sheet with each ASIN's forecast at weeks 4, 8, and 16.

    Parameters:
        output_path (str): Path to save the consolidated Excel file.
        consolidated_data (dict): Dictionary containing ASIN as keys and DataFrames as values.
        missing_asin_data (pd.DataFrame): DataFrame containing entries with missing ASINs.
    """
    desired_columns = [
        'Week', 'ASIN', 'Prophet Forecast', 'Amazon Mean Forecast',
        'Amazon P70 Forecast', 'Amazon P80 Forecast', 'Amazon P90 Forecast',
        'Product Title', 'is_holiday_week'
    ]

    wb = Workbook()

    # Store consolidated_data in separate sheets
    for asin, forecast_df in consolidated_data.items():
        df_for_excel = forecast_df.copy()
        if 'ds' in df_for_excel.columns:
            for i in range(len(df_for_excel)):
                df_for_excel.loc[i, 'Week'] = f"W{str(i+1).zfill(2)}"
            df_for_excel.drop(columns=['ds'], inplace=True, errors='ignore')

        for col in desired_columns:
            if col not in df_for_excel.columns:
                df_for_excel[col] = np.nan

        df_for_excel = df_for_excel[desired_columns]

        ws = wb.create_sheet(title=str(asin)[:31])  # Excel sheet names limited to 31 chars
        for r in dataframe_to_rows(df_for_excel, index=False, header=True):
            ws.append(r)

    # If there are missing ASIN data, add a separate sheet
    if not missing_asin_data.empty:
        ws_missing = wb.create_sheet(title="No ASIN")
        for r in dataframe_to_rows(missing_asin_data, index=False, header=True):
            ws_missing.append(r)

    # Remove default 'Sheet' if it still exists
    if 'Sheet' in wb.sheetnames:
        del wb['Sheet']

    # Create a special sheet with 4, 8, 16 week summary
    ws_summary = wb.create_sheet(title="4-16 Weeks Summary")
    ws_summary.append(["ASIN", "Product Title", "4 Week Forecast", "8 Week Forecast", "16 Week Forecast"])

    for asin, df in consolidated_data.items():
        if df.empty:
            ws_summary.append([asin, "No data", None, None, None])
            continue

        product_title = df['Product Title'].iloc[0] if 'Product Title' in df.columns else ''
        
        # Calculate cumulative sum for forecasts
        df['Cumulative Forecast'] = df['Prophet Forecast'].cumsum()

        # Retrieve cumulative values for 4 weeks, 8 weeks, and 16 weeks
        four_wk_val = df['Cumulative Forecast'].iloc[3] if len(df) >= 4 else df['Prophet Forecast'].sum()
        eight_wk_val = df['Cumulative Forecast'].iloc[7] if len(df) >= 8 else df['Prophet Forecast'].sum()
        sixteen_wk_val = df['Cumulative Forecast'].iloc[15] if len(df) >= 16 else df['Prophet Forecast'].sum()

        ws_summary.append([asin, product_title, four_wk_val, eight_wk_val, sixteen_wk_val])

    wb.save(output_path)
    print(f"All forecasts saved to {output_path}")

##############################
# Forecast Adjustment and Logging
##############################
def adjust_forecast_if_out_of_range(comparison, asin, adjustment_threshold=0.3):
    """
    Adjust the Prophet forecast if it deviates significantly from Amazon forecasts.

    Parameters:
        comparison (pd.DataFrame): Forecast comparison DataFrame.
        asin (str): ASIN of the product.
        adjustment_threshold (float): The ratio threshold to trigger adjustments.

    Returns:
        pd.DataFrame: Updated comparison with adjusted forecasts if needed.
    """
    global out_of_range_counter
    global out_of_range_stats

    # Ensure Amazon forecast columns exist
    for col in ['Amazon Mean Forecast', 'Amazon P70 Forecast', 'Amazon P80 Forecast', 'Amazon P90 Forecast']:
        if col not in comparison.columns:
            print(f"Warning: Missing column {col}. Initializing it with zeros.")
            comparison[col] = 0

    # Debug: Verify Amazon forecast data
    print("\nAmazon Forecast Statistics for Debugging:")
    print(comparison[['Amazon Mean Forecast', 'Amazon P70 Forecast', 'Amazon P80 Forecast', 'Amazon P90 Forecast']].head())

    # Calculate out-of-range rows
    comparison['is_out_of_range'] = (
        (comparison['Prophet Forecast'] < comparison['Amazon Mean Forecast'] * (1 - adjustment_threshold)) |
        (comparison['Prophet Forecast'] > comparison['Amazon Mean Forecast'] * (1 + adjustment_threshold))
    )

    adjustment_mask = comparison['is_out_of_range']

    # Total forecasts for ASIN
    total_forecasts = len(comparison)

    # Apply adjustment to out-of-range rows
    if adjustment_mask.any():
        num_adjustments = adjustment_mask.sum()
        print(f"\nAdjusting {num_adjustments} out-of-range forecasts for ASIN {asin} using Amazon data.")

        # Update the global counter
        out_of_range_counter[asin] += num_adjustments

        # Update percentage stats
        if asin not in out_of_range_stats:
            out_of_range_stats[asin] = {'total': 0, 'adjusted': 0}
        out_of_range_stats[asin]['total'] += total_forecasts
        out_of_range_stats[asin]['adjusted'] += num_adjustments

        comparison.loc[adjustment_mask, 'Prophet Forecast'] = (
            0.7 * comparison.loc[adjustment_mask, 'Amazon Mean Forecast'] +
            0.2 * comparison.loc[adjustment_mask, 'Amazon P70 Forecast'] +
            0.1 * comparison.loc[adjustment_mask, 'Amazon P80 Forecast']
        ).clip(lower=0)

        # Identify rows still out of range after adjustment
        comparison['is_still_out_of_range'] = (
            (comparison['Prophet Forecast'] < comparison['Amazon Mean Forecast'] * (1 - adjustment_threshold)) |
            (comparison['Prophet Forecast'] > comparison['Amazon Mean Forecast'] * (1 + adjustment_threshold))
        )

        # Debug: Rows still out of range
        if comparison['is_still_out_of_range'].any():
            print("\nRows still out of range after primary adjustment:")
            print(comparison.loc[comparison['is_still_out_of_range'], [
                'ds', 'Prophet Forecast', 'Amazon Mean Forecast', 'Amazon P70 Forecast', 'Amazon P80 Forecast', 'Amazon P90 Forecast'
            ]])

    # Debug: Adjusted rows
    print("\nAdjusted forecasts for out-of-range rows:")
    print(comparison.loc[adjustment_mask, ['ds', 'Prophet Forecast', 'Amazon Mean Forecast', 'Amazon P70 Forecast', 'Amazon P80 Forecast']])

    # Final debug for the adjusted data
    available_columns = [col for col in ['Week', 'ASIN', 'Prophet Forecast', 'Amazon Mean Forecast', 'Amazon P70 Forecast', 'Amazon P80 Forecast', 'Amazon P90 Forecast'] if col in comparison.columns]
    if available_columns:
        print("\nFinal adjusted forecast data:")
        print(comparison[available_columns].head())
    else:
        print("Warning: None of the specified columns ('Week', 'ASIN') are in the DataFrame.")

    return comparison

def log_fallback_triggers(comparison, asin, product_title, fallback_file="fallback_triggers.csv"):
    """
    Logs products where the fallback mechanism was triggered to a separate file.

    Parameters:
        comparison (pd.DataFrame): The DataFrame containing forecast comparison.
        asin (str): The ASIN of the product.
        product_title (str): The title of the product.
        fallback_file (str): The file to store fallback trigger logs.

    Returns:
        None
    """
    # Detect rows where the fallback mechanism was triggered
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

        # Check if the fallback file exists
        if os.path.exists(fallback_file):
            # Append to existing file
            existing_data = pd.read_csv(fallback_file)
            fallback_df = pd.concat([existing_data, fallback_df], ignore_index=True)

        # Save to file
        fallback_df.to_csv(fallback_file, index=False)
        print(f"Fallback log updated: {fallback_file}")
    else:
        print(f"No outliers detected for ASIN: {asin}")

##############################
# Forecast Model Selection
##############################
def choose_forecast_model(ts_data, threshold=20, holidays=None):
    """
    Decide whether to use SARIMA or Prophet based on the size and characteristics of the time series data.

    Parameters:
        ts_data (pd.DataFrame): Time series data containing at least 'ds' and 'y' columns.
        threshold (int): Minimum number of data points required to use Prophet. If data points are fewer than this,
                         SARIMA is chosen.
        holidays (pd.DataFrame or None): DataFrame containing holiday information for Prophet.

    Returns:
        tuple:
            - model: Trained SARIMA or Prophet model.
            - model_type (str): Type of the model used ('SARIMA' or 'Prophet').
    """
    global PARAM_COUNTER, POOR_PARAM_FOUND
    
    PARAM_COUNTER += 1
    print(f"Choosing forecast model for current ASIN. Attempt #{PARAM_COUNTER}")
    
    # Decide based on the number of data points
    if len(ts_data) < threshold:
        print(f"Data points ({len(ts_data)}) less than threshold ({threshold}). Using SARIMA model.")
        model_fit = fit_sarima_model(ts_data, holidays=holidays)
        if model_fit is None:
            print("Failed to fit SARIMA model. Falling back to Prophet.")
            model_fit, model_type = fallback_to_prophet(ts_data, holidays=holidays)
        else:
            model_type = "SARIMA"
    else:
        print(f"Data points ({len(ts_data)}) meet or exceed threshold ({threshold}). Using Prophet model.")
        model_fit, model_type = fallback_to_prophet(ts_data, holidays=holidays)
    
    # Early stopping if poor parameters are found
    if POOR_PARAM_FOUND:
        print("Early stopping due to poor parameter performance.")
        return None, None
    
    return model_fit, model_type

def fallback_to_prophet(ts_data, holidays=None):
    """
    Train a Prophet model as a fallback option.

    Parameters:
        ts_data (pd.DataFrame): Time series data containing at least 'ds' and 'y' columns.
        holidays (pd.DataFrame or None): DataFrame containing holiday information for Prophet.

    Returns:
        tuple:
            - model: Trained Prophet model.
            - model_type (str): Type of the model used ('Prophet').
    """
    try:
        model = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=True,
            seasonality_mode='multiplicative',
            holidays=holidays
        )
        
        # Add any additional regressors if present
        regressor_cols = [col for col in ts_data.columns if col.startswith('lag_') or col in ['Amazon Mean Forecast', 'Amazon P70 Forecast', 'Amazon P80 Forecast', 'Amazon P90 Forecast']]
        for regressor in regressor_cols:
            model.add_regressor(regressor)
        
        model.fit(ts_data[['ds', 'y'] + regressor_cols])
        print("Successfully trained Prophet model as a fallback.")
        return model, "Prophet"
    except Exception as e:
        print(f"Failed to train Prophet model as a fallback: {e}")
        return None, None

##############################
# Main Workflow
##############################
def main():
    # File paths
    sales_file = 'weekly_sales_data.xlsx'             # Path to your sales data Excel file
    forecasts_folder = 'forecasts_folder'             # Folder containing Amazon forecast Excel files
    asins_to_forecast_file = 'ASINs to Forecast.xlsx'  # File containing ASINs to forecast
    output_file = 'consolidated_forecast.xlsx'        # Output Excel file for consolidated forecasts
    horizon = 16                                       # Forecast horizon in weeks

    # Load and clean data with duplicate handling
    data = load_weekly_sales_data(sales_file)
    valid_data = data[data['asin'].notna() & (data['asin'] != '#N/A')]
    missing_asin_data = data[data['asin'].isna() | (data['asin'] == '#N/A')]
    if not missing_asin_data.empty:
        print("The following entries have no ASIN and will be noted in the forecast file:")
        print(missing_asin_data[['product title', 'week', 'year', 'y']].to_string())

    # Load requested ASINs
    asins_to_forecast = load_asins_to_forecast(asins_to_forecast_file)
    print(f"ASINs to forecast: {asins_to_forecast}")

    # Filter valid_data to only include the ASINs we care about
    asin_list = valid_data['asin'].unique()
    asin_list = [asin for asin in asin_list if asin in asins_to_forecast]

    # Initialize containers for forecasts
    consolidated_forecasts = {}
    param_grid = {
        'changepoint_prior_scale': [0.05, 0.1, 0.2, 0.3, 0.4, 0.5],
        'seasonality_prior_scale': [0.5, 0.1, 1, 2, 3, 4, 5],
        'holidays_prior_scale': [5, 10, 15]
    }

    holidays = get_shifted_holidays()

    # Optional example cross-validation on a sample ASIN
    if len(asin_list) > 0:
        test_asin = asin_list[0]
        test_ts_data = prepare_time_series_with_lags(valid_data, test_asin, lag_weeks=1, holidays=holidays)
        if not test_ts_data.empty and len(test_ts_data.dropna()) >= 2:
            print(f"Performing cross-validation on ASIN {test_asin} Prophet model...")
            cross_validate_prophet_model(test_ts_data, initial='365 days', period='180 days', horizon='90 days')
        else:
            print(f"Not enough data for {test_asin} to perform cross-validation test.")

    # Create necessary folders
    insufficient_data_folder = "Insufficient data"
    sufficient_data_folder = "Sufficient data"
    os.makedirs(insufficient_data_folder, exist_ok=True)
    os.makedirs(sufficient_data_folder, exist_ok=True)

    global PARAM_COUNTER

    for asin in asin_list:
        product_title = valid_data[valid_data['asin'] == asin]['product title'].iloc[0]
        print(f"\nProcessing ASIN: {asin} - {product_title}")
        forecast_data = load_amazon_forecasts_from_folder(forecasts_folder, asin)
        if not forecast_data:
            print(f"No forecast data found for ASIN {asin}, skipping.")
            continue

        # Prepare data for all models
        ts_data = prepare_time_series_with_lags(valid_data, asin, lag_weeks=1, holidays=holidays)
        non_nan_count = len(ts_data.dropna())
        if non_nan_count < 2:
            print(f"Not enough data for ASIN {asin} (only {non_nan_count} data points), skipping.")
            no_data_output = os.path.join(insufficient_data_folder, f"{asin}_no_data.txt")
            with open(no_data_output, 'w') as f:
                f.write("Insufficient data for training/forecasting.\n")
            continue

        # 1) Decide model type (SARIMA or Prophet)
        model, model_type = choose_forecast_model(ts_data, threshold=FALLBACK_THRESHOLD, holidays=holidays)

        if model is None:
            print(f"Failed to train both SARIMA and Prophet models for ASIN {asin}, skipping.")
            no_data_output = os.path.join(insufficient_data_folder, f"{asin}_model_training_failed.txt")
            with open(no_data_output, 'w') as f:
                f.write("Failed to train both SARIMA and Prophet models.\n")
            continue

        # 2) Train XGBoost
        xgb_model, xgb_features, xgb_shap_values = train_xgboost(ts_data, target='y')
        if xgb_model is not None:
            # Optionally perform additional checks or validations for XGBoost here
            pass

        # =========================
        # If model_type == SARIMA
        # =========================
        if model_type == "SARIMA":
            n = len(ts_data)
            split = int(n * 0.8)
            train_sarima = ts_data.iloc[:split]
            test_sarima = ts_data.iloc[split:]

            exog_test = create_holiday_regressors(test_sarima, holidays)

            if model is not None:
                try:
                    steps = len(test_sarima)
                    preds_df = sarima_forecast(
                        model, steps=steps,
                        last_date=train_sarima['ds'].iloc[-1],
                        exog=exog_test
                    )
                    preds = preds_df['SARIMA Forecast'].values

                    sarima_mape = mean_absolute_percentage_error(test_sarima['y'], preds)
                    sarima_rmse = sqrt(mean_squared_error(test_sarima['y'], preds))
                    print(f"SARIMA Test MAPE: {sarima_mape:.4f}, RMSE: {sarima_rmse:.4f}")

                    # Generate final SARIMA forecast for horizon
                    last_date_full = ts_data['ds'].iloc[-1]
                    exog_future = generate_future_exog(holidays, steps=horizon, last_date=last_date_full)
                    final_forecast_df = sarima_forecast(model, steps=horizon, last_date=last_date_full, exog=exog_future)
                    if final_forecast_df.empty:
                        print(f"Forecasting failed for ASIN {asin}, skipping.")
                        no_data_output = os.path.join(insufficient_data_folder, f"{asin}_forecast_failed.txt")
                        with open(no_data_output, 'w') as f:
                            f.write("Failed to forecast due to insufficient data.\n")
                        continue

                    comparison = final_forecast_df.copy()
                    comparison['ASIN'] = asin
                    comparison['Product Title'] = product_title
                    comparison = comparison.merge(ts_data[['ds','y']], on='ds', how='left')
                    
                    comparison_historical = comparison.dropna(subset=['y'])

                    if comparison_historical.empty:
                        print("No overlapping historical data to calculate metrics. Skipping metrics.")
                        metrics = {}
                    else:
                        MAE = mean_absolute_error(comparison_historical['y'], comparison_historical['SARIMA Forecast'])
                        MEDAE = median_absolute_error(comparison_historical['y'], comparison_historical['SARIMA Forecast'])
                        MSE = mean_squared_error(comparison_historical['y'], comparison_historical['SARIMA Forecast'])
                        RMSE = sqrt(MSE)
                        MAPE = mean_absolute_percentage_error(comparison_historical['y'], comparison_historical['SARIMA Forecast'])

                        print('Mean Absolute Error (MAE): ' + str(np.round(MAE, 2)))
                        print('Median Absolute Error (MedAE): ' + str(np.round(MEDAE, 2)))
                        print('Mean Squared Error (MSE): ' + str(np.round(MSE, 2)))
                        print('Root Mean Squared Error (RMSE): ' + str(np.round(RMSE, 2)))
                        print('Mean Absolute Percentage Error (MAPE): ' + str(np.round(MAPE, 2)) + ' %')

                        metrics = {
                            "Mean Absolute Error (MAE)": np.round(MAE, 2),
                            "Median Absolute Error (MedAE)": np.round(MEDAE, 2),
                            "Mean Squared Error (MSE)": np.round(MSE, 2),
                            "Root Mean Squared Error (RMSE)": np.round(RMSE, 2),
                            "Mean Absolute Percentage Error (MAPE)": str(np.round(MAPE, 2)) + " %"
                        }

                    # Combine SARIMA forecast with Amazon data (optional)
                    amazon_cols = [col for col in forecast_data.keys()]
                    if amazon_cols:
                        for ftype, values in forecast_data.items():
                            horizon_values = values[:horizon] if len(values) >= horizon else values
                            if len(horizon_values) < horizon and len(horizon_values) > 0:
                                horizon_values = np.pad(horizon_values, (0, horizon - len(horizon_values)),
                                                        'constant', constant_values=horizon_values[-1])
                            elif len(horizon_values) == 0:
                                horizon_values = np.zeros(horizon, dtype=int)
                            comparison[f'Amazon {ftype} Forecast'] = horizon_values

                        # Compute mean Amazon forecast
                        a_cols = [c for c in comparison.columns if c.startswith('Amazon ')]
                        if a_cols:
                            comparison['Amazon Mean Forecast'] = comparison[a_cols].mean(axis=1)

                            MEAN_WEIGHT = 0.7
                            P70_WEIGHT = 0.2
                            P80_WEIGHT = 0.1

                            comparison['Prophet Forecast'] = (
                                MEAN_WEIGHT * comparison['Amazon Mean Forecast'] +
                                P70_WEIGHT * comparison.get('Amazon P70 Forecast', 0) +
                                P80_WEIGHT * comparison.get('Amazon P80 Forecast', 0)
                            ).clip(lower=0)

                            # Optionally blend this with SARIMA
                            comparison['Prophet Forecast'] = (
                                SARIMA_WEIGHT * comparison['Prophet Forecast']
                                + (1 - SARIMA_WEIGHT) * comparison['Amazon Mean Forecast']
                            ).clip(lower=0)

                    # Optionally incorporate XGBoost into an ensemble
                    if xgb_model is not None:
                        xgb_future_df = xgboost_forecast(xgb_model, ts_data,
                                                         forecast_steps=horizon,
                                                         target='y',
                                                         features=xgb_features)
                        comparison = comparison.merge(xgb_future_df, on='ds', how='left')
                        print("Merged comparison DataFrame columns:")
                        print(comparison.columns)
                        comparison['XGBoost Forecast'] = comparison['XGBoost Forecast'].fillna(0)

                        comparison['Prophet Forecast'] = (
                            0.7 * comparison['XGBoost Forecast'] +
                            0.3 * comparison['Amazon Mean Forecast']
                        ).clip(lower=0)

                    # Summaries and Save
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

                    # Adjust forecasts if they are out of range
                    comparison = adjust_forecast_if_out_of_range(comparison, asin, adjustment_threshold=0.3)
                    log_fallback_triggers(comparison, asin, product_title)

                    # Save to Excel
                    output_file_name = f'forecast_summary_{asin}.xlsx'
                    output_file_path = os.path.join(sufficient_data_folder, output_file_name)
                    with pd.ExcelWriter(output_file_path, mode='w') as writer:
                        comparison.to_excel(writer, index=False)
                    save_summary_to_excel(
                        comparison, summary_stats,
                        total_forecast_16, total_forecast_8, total_forecast_4,
                        max_forecast, min_forecast, max_week, min_week,
                        output_file_path, metrics=metrics
                    )
                    consolidated_forecasts[asin] = comparison

                except ValueError as e:
                    print(f"Error during SARIMA prediction for ASIN {asin}: {e}")
                    continue
            else:
                print(f"SARIMA model fitting failed for {asin}, skipping.")
                continue

        # =========================
        # If model_type == Prophet
        # =========================
        else:
            cached_model_path = os.path.join("model_cache", f"{asin}_Prophet.pkl")
            if os.path.exists(cached_model_path):
                if is_model_up_to_date(cached_model_path, ts_data):
                    print(f"Using up-to-date cached Prophet model for ASIN {asin}.")
                    cached_prophet_model = joblib.load(cached_model_path)
                    last_train_date = ts_data['ds'].max()
                    future_dates = pd.date_range(start=last_train_date + pd.Timedelta(days=7),
                                                 periods=horizon, freq='W')
                    future = pd.DataFrame({'ds': future_dates})
                    # Add zero columns for regressor placeholders
                    for forecast_type in forecast_data.keys():
                        future[f"Amazon_{forecast_type} Forecast"] = 0
                    future['prime_day'] = 0
                    forecast = cached_prophet_model.predict(future)
                    forecast['Prophet Forecast'] = forecast['yhat'].round().astype(int)
                    # Handle non-finite values
                    forecast['Prophet Forecast'].replace([np.inf, -np.inf], 0, inplace=True)
                    forecast['Prophet Forecast'].fillna(0, inplace=True)
                    forecast['Prophet Forecast'] = forecast['Prophet Forecast'].astype(int)
                else:
                    print(f"Cached Prophet model for ASIN {asin} is outdated. Retraining with updated data...")
                    best_params, _ = optimize_prophet_params(ts_data, forecast_data, param_grid, horizon=horizon)
                    forecast, trained_prophet_model = forecast_with_custom_params(
                        ts_data, forecast_data,
                        best_params['changepoint_prior_scale'],
                        best_params['seasonality_prior_scale'],
                        best_params['holidays_prior_scale'],
                        horizon=horizon,
                        weights={
                            'Mean': 0.7,
                            'P70': 0.2,
                            'P80': 0.1
                        }
                    )

                    if trained_prophet_model is not None:
                        save_model(trained_prophet_model, "Prophet", asin, ts_data)
                    else:
                        print("Failed to retrain the Prophet model.")
            else:
                print(f"No cached Prophet model found for ASIN {asin}. Training a new model...")
                best_params, _ = optimize_prophet_params(ts_data, forecast_data, param_grid, horizon=horizon)
                forecast, trained_prophet_model = forecast_with_custom_params(
                    ts_data, forecast_data,
                    best_params['changepoint_prior_scale'],
                    best_params['seasonality_prior_scale'],
                    best_params['holidays_prior_scale'],
                    horizon=horizon
                )
                if trained_prophet_model is not None:
                    save_model(trained_prophet_model, "Prophet", asin, ts_data)
                else:
                    print("Failed to train the Prophet model.")

            # If we failed to produce 'forecast' above, skip
            if 'forecast' not in locals() or forecast.empty:
                print(f"Forecasting failed for ASIN {asin}, skipping.")
                no_data_output = os.path.join(insufficient_data_folder, f"{asin}_final_forecast_failed.txt")
                with open(no_data_output, 'w') as f:
                    f.write("Final forecasting failed.\n")
                continue

            # Combine forecasts with Amazon data
            comparison = format_output_with_forecasts(forecast, forecast_data, horizon=horizon)
            best_weights, best_rmse = auto_find_best_weights(forecast, comparison, step=0.05)
            print(f"Auto best weights for ASIN {asin}: {best_weights} with RMSE={best_rmse}")

            # Apply the best weights to adjust the forecast
            forecast = adjust_forecast_weights(forecast.copy(), *best_weights)

            # Update the comparison DataFrame after adjustment
            comparison = format_output_with_forecasts(forecast, forecast_data, horizon=horizon)
            print("\n--- Prophet Forecast Before Adjustment ---")
            print(comparison[['Prophet Forecast']].head(10))  # Adjust the number as needed
            print("-----------------------------------------\n")

            # Adjust forecasts if they are out of range
            comparison = adjust_forecast_if_out_of_range(comparison, asin, adjustment_threshold=0.3)

            # Log fallback triggers for out-of-range forecasts
            log_fallback_triggers(comparison, asin, product_title)

            # Analyze Amazon vs. Prophet
            analyze_amazon_buying_habits(comparison, holidays)

            # Merge historical data to compute error metrics
            comparison = comparison.merge(ts_data[['ds','y']], on='ds', how='left')
            comparison_historical = comparison.dropna(subset=['y'])

            if comparison_historical.empty:
                print("No overlapping historical data to calculate metrics. Skipping metrics.")
                metrics = {}
            else:
                MAE = mean_absolute_error(comparison_historical['y'], comparison_historical['Prophet Forecast'])
                MEDAE = median_absolute_error(comparison_historical['y'], comparison_historical['Prophet Forecast'])
                MSE = mean_squared_error(comparison_historical['y'], comparison_historical['Prophet Forecast'])
                RMSE = sqrt(MSE)
                MAPE = mean_absolute_percentage_error(comparison_historical['y'], comparison_historical['Prophet Forecast'])

                print('Mean Absolute Error (MAE): ' + str(np.round(MAE, 2)))
                print('Median Absolute Error (MedAE): ' + str(np.round(MEDAE, 2)))
                print('Mean Squared Error (MSE): ' + str(np.round(MSE, 2)))
                print('Root Mean Squared Error (RMSE): ' + str(np.round(RMSE, 2)))
                print('Mean Absolute Percentage Error (MAPE): ' + str(np.round(MAPE, 2)) + ' %')

                metrics = {
                    "Mean Absolute Error (MAE)": np.round(MAE, 2),
                    "Median Absolute Error (MedAE)": np.round(MEDAE, 2),
                    "Mean Squared Error (MSE)": np.round(MSE, 2),
                    "Root Mean Squared Error (RMSE)": np.round(RMSE, 2),
                    "Mean Absolute Percentage Error (MAPE)": str(np.round(MAPE, 2)) + " %"
                }

            # Optionally incorporate XGBoost into the final ensemble:
            if xgb_model is not None:
                xgb_future_df = xgboost_forecast(xgb_model, ts_data, forecast_steps=horizon, target='y', features=xgb_features)
                comparison = comparison.merge(xgb_future_df, on='ds', how='left')
                comparison['XGBoost Forecast'] = comparison['XGBoost Forecast'].fillna(0)
                # Combine Prophet forecast + XGBoost + Amazon Mean
                # Example weights: [0.5, 1.0, 0.2, 0.3] or adjust as needed
                if 'Amazon Mean Forecast' not in comparison.columns:
                    comparison['Amazon Mean Forecast'] = 0
                comparison['Prophet Forecast'] = ensemble_forecast(
                    comparison['Prophet Forecast'],
                    0,  # ignoring a separate Prophet column
                    comparison['XGBoost Forecast'],
                    comparison['Amazon Mean Forecast'],
                    [0.5, 0, 0.2, 0.3]
                ).clip(lower=0)

            # Summaries
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

            # Save to Excel
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

    # Save all forecasts to Excel
    final_output_path = output_file
    save_forecast_to_excel(final_output_path, consolidated_forecasts, missing_asin_data)

    print(f"Total number of parameter sets tested: {PARAM_COUNTER}")
    if POOR_PARAM_FOUND:
        print("Note: Early stopping occurred for some ASINs due to poor parameter performance.")

##############################
# Run the Main Script
##############################
if __name__ == '__main__':
    main()
    summarize_usage()
