# preprocessing.py

import pandas as pd
import numpy as np
from pykalman import KalmanFilter
from sklearn.preprocessing import MinMaxScaler

def kalman_smooth(series):
    """
    Use a Kalman Filter to fill missing values in the series.
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
    Apply Min-Max scaling to specified feature columns.
    """
    scalers = {}
    for col in feature_cols:
        scaler = MinMaxScaler()
        df[col] = scaler.fit_transform(df[[col]])
        scalers[col] = scaler
    return df, scalers

def preprocess_data(data):
    """
    Complete preprocessing pipeline: handle missing data, outliers, and scaling.
    """
    data = handle_missing_data(data)
    data = handle_outliers(data)
    # Example: scale 'y' using MinMax (optional)
    # data, scalers = min_max_scale_data(data, ['y'])
    return data

def prepare_time_series_with_lags(data, asin, lag_weeks=1):
    """
    Prepare time series data for forecasting by adding lag features.
    """
    ts_data = data[data['asin'] == asin].sort_values('ds').copy()
    for lag in range(1, lag_weeks + 1):
        ts_data[f'lag_{lag}_week'] = ts_data['y'].shift(lag)
    ts_data = ts_data.dropna().reset_index(drop=True)
    return ts_data
