import pandas as pd
import numpy as np
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
    mean_squared_error,
)
from math import sqrt
from pykalman import KalmanFilter
from statsmodels.tsa.stattools import adfuller
import time

warnings.filterwarnings("ignore")

##############################
# Configuration
##############################
SARIMA_WEIGHT = 0.6  # Weight for SARIMA forecast in final combination
CAP_VALUE = 2000     # Example cap for Prophet if using logistic growth
PERCENT_TOLERANCE = 0.30  # 30% difference tolerance for fallback
EARLY_STOP_THRESHOLD = 10_000

##############################
# Added Functions for Caching
##############################

def save_model(model, model_type, asin, ts_data):
    model_cache_folder = "model_cache"
    os.makedirs(model_cache_folder, exist_ok=True)
    model_path = os.path.join(model_cache_folder, f"{asin}_{model_type}.pkl")
    model.last_train_date = ts_data['ds'].max()
    joblib.dump(model, model_path)

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
    if len(data['y']) < 2:
        print("Not enough data points for Kalman smoothing. Using fallback method for missing data.")
        data['y'] = data['y'].fillna(method='bfill').fillna(method='ffill').astype(int)
    else:
        data['y'] = kalman_smooth(data['y'])
    return data

def handle_outliers(data):
    Q1 = data['y'].quantile(0.25)
    Q3 = data['y'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    clipped = data['y'].clip(lower=lower_bound, upper=upper_bound).round().astype(int)
    data['y'] = clipped
    return data

def preprocess_data(data):
    data = handle_missing_data(data)
    data = handle_outliers(data)
    return data

##############################
# Differencing and Stationarity
##############################

def differencing(timeseries, m):
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
    columns = [
        "Test Statistic", "p-value", "#Lags Used", "No. of Obs. Used",
        "Critical Value (1%)", "Critical Value (5%)", "Critical Value (10%)"
    ]
    index = diff_series.columns
    summary = pd.DataFrame(summary, index=index, columns=columns)
    return summary

def create_holiday_regressors(ts_data, holidays):
    holiday_names = holidays['holiday'].unique()
    exog = pd.DataFrame(index=ts_data.index)
    exog['ds'] = ts_data['ds']
    for h in holiday_names:
        holiday_dates = holidays[holidays['holiday'] == h]['ds']
        exog[h] = exog['ds'].isin(holiday_dates).astype(int)
    exog.drop(columns=['ds'], inplace=True)
    return exog

##############################
# SARIMA Fitting
##############################

def calculate_forecast_metrics(actual, predicted):
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    mape = mean_absolute_percentage_error(actual, predicted)*100
    return rmse, mape

def fit_sarima_model(data, holidays, seasonal_period=52):
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
                                        rmse_val, mape_val = calculate_forecast_metrics(actual, forecast)
                                        if rmse_val < best_rmse:
                                            best_rmse = rmse_val
                                            best_model = model_fit
                                            best_metrics = {
                                                'RMSE': rmse_val,
                                                'MAPE': mape_val,
                                                'p': p, 'd': d, 'q': q,
                                                'P': P, 'D': D, 'Q': Q, 'm': m
                                            }
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
                        rmse_val, mape_val = calculate_forecast_metrics(actual, forecast)
                        if rmse_val < best_rmse:
                            best_rmse = rmse_val
                            best_model = model_fit
                            best_metrics = {
                                'RMSE': rmse_val,
                                'MAPE': mape_val,
                                'p': p, 'd': d, 'q': q,
                                'P': 0, 'D': 0, 'Q': 0, 'm': 1
                            }
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
    Return a DataFrame with columns ['ds', 'Forecast'].
    We'll unify the column name as 'Forecast' for consistency.
    """
    k_exog = model_fit.model.k_exog
    if exog is not None and k_exog > 0:
        exog = exog.iloc[:, :k_exog]

    forecast_values = model_fit.forecast(steps=steps, exog=exog)
    forecast_values = forecast_values.clip(lower=0).round().astype(int)

    future_dates = pd.date_range(start=last_date + pd.Timedelta(weeks=1), periods=steps, freq='W')
    return pd.DataFrame({'ds': future_dates, 'Forecast': forecast_values})

def generate_future_exog(holidays, steps, last_date):
    future_dates = pd.date_range(start=last_date + pd.Timedelta(weeks=1), periods=steps, freq='W')
    exog_future = pd.DataFrame(index=future_dates)
    holiday_names = holidays['holiday'].unique()
    for h in holiday_names:
        exog_future[h] = exog_future.index.isin(holidays[holidays['holiday'] == h]['ds'])
    exog_future = exog_future.astype(int)
    return exog_future

def choose_forecast_model(ts_data, threshold=50, holidays=None):
    if len(ts_data) <= threshold:
        print("Dataset size is small. Using SARIMA for forecasting.")
        sarima_model = fit_sarima_model(ts_data, holidays, seasonal_period=52)
        if sarima_model is not None:
            asin = ts_data['asin'].iloc[0]
            save_model(sarima_model, "SARIMA", asin, ts_data)
        return sarima_model, "SARIMA"
    else:
        print("Dataset size is sufficient. Using Prophet for forecasting.")
        return None, "Prophet"

def generate_date_from_week(row):
    week_str = row['week']
    year = row['year']
    week_number = int(week_str[1:])
    return pd.to_datetime(f'{year}-W{week_number - 1}-0', format='%Y-W%U-%w')

def clean_weekly_sales_data(data):
    # Optionally remove duplicates or do other cleaning
    return data

def load_weekly_sales_data(file_path):
    data = pd.read_excel(file_path)
    data.columns = data.columns.str.strip().str.lower()

    required_columns = ['product title', 'week', 'year', 'units_sold', 'asin']
    missing_required = [col for col in required_columns if col not in data.columns]
    if missing_required:
        raise ValueError(f"Missing required columns: {missing_required}")

    data['date'] = data.apply(generate_date_from_week, axis=1)
    data = data.rename(columns={'units_sold': 'y'})
    data['y'] = data['y'].astype(int)
    data = clean_weekly_sales_data(data)
    return data

def load_asins_to_forecast(file_path):
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
    forecast_data = {}
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.xlsx'):
            forecast_type = os.path.splitext(file_name)[0].replace('_', ' ').title()
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
    ts_data = ts_data.copy()
    ts_data = ts_data.sort_values('ds')
    ts_data[f'lag_{lag_weeks}_week'] = ts_data['y'].shift(lag_weeks)
    ts_data[f'lag_{lag_weeks}_week'].fillna(0, inplace=True)
    return ts_data

def prepare_time_series_with_lags(data, asin, lag_weeks=1):
    ts_data = data[data['asin'] == asin].rename(columns={'date': 'ds', 'y': 'y'})
    ts_data = ts_data.sort_values('ds')
    ts_data = preprocess_data(ts_data)
    return add_lag_features(ts_data, lag_weeks)

def get_shifted_holidays():
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
# Prophet Forecast
##############################

def forecast_with_custom_params(ts_data, forecast_data,
                                changepoint_prior_scale,
                                seasonality_prior_scale,
                                holidays_prior_scale,
                                horizon=16):
    from prophet import Prophet

    future_dates = pd.date_range(start=ts_data['ds'].max() + pd.Timedelta(days=7),
                                 periods=horizon, freq='W')
    future = pd.DataFrame({'ds': future_dates})
    combined_df = pd.concat([ts_data, future], ignore_index=True)

    # Insert Amazon forecast data as regressors
    for forecast_type, values in forecast_data.items():
        values_to_use = values[:horizon] if len(values) > horizon else values
        extended_values = np.concatenate([
            np.full(len(ts_data), np.nan),
            values_to_use,
            np.full(max(horizon - len(values_to_use), 0),
                    values_to_use[-1] if len(values_to_use) > 0 else 0)
        ])
        extended_values = extended_values[:len(combined_df)]
        combined_df[f'Amazon_{forecast_type}'] = extended_values

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
        growth='linear',
        yearly_seasonality=True,
        weekly_seasonality=True,
        seasonality_mode='multiplicative',
        holidays=holidays,
        changepoint_prior_scale=changepoint_prior_scale,
        seasonality_prior_scale=seasonality_prior_scale,
        holidays_prior_scale=holidays_prior_scale
    )

    for regressor in regressor_cols + ['prime_day']:
        model.add_regressor(regressor, mode='multiplicative')

    model.fit(train_df[['ds', 'y'] + regressor_cols + ['prime_day']])

    test_future = test_df.drop(columns='y', errors='ignore').copy()
    test_forecast = model.predict(test_future)
    test_actual = combined_df.iloc[split:].dropna(subset=['y'])
    if not test_actual.empty:
        test_eval = test_forecast[test_forecast['ds'].isin(test_actual['ds'])]
        test_eval['Forecast'] = test_eval['yhat'].clip(lower=0).round().astype(int)
        mape_val = mean_absolute_percentage_error(test_actual['y'], test_eval['Forecast'])
        rmse_val = sqrt(mean_squared_error(test_actual['y'], test_eval['Forecast']))
        print(f"Prophet Test MAPE: {mape_val:.4f}, RMSE: {rmse_val:.4f}")

    future_df = combined_df[combined_df['y'].isna()].drop(columns='y', errors='ignore').copy()
    forecast = model.predict(future_df)
    # We'll unify the column name 'Forecast' for Prophet as well
    forecast['Forecast'] = forecast['yhat'].clip(lower=0).round().astype(int)
    return forecast[['ds', 'Forecast', 'yhat', 'yhat_upper']], model

##############################
# Parameter Optimization for Prophet
##############################

PARAM_COUNTER = 0
POOR_PARAM_FOUND = False

def optimize_prophet_params(ts_data, forecast_data, param_grid, horizon=16):
    global PARAM_COUNTER, POOR_PARAM_FOUND
    best_rmse = float('inf')
    best_params = None
    best_rmse_values = None
    for changepoint_prior_scale in param_grid['changepoint_prior_scale']:
        for seasonality_prior_scale in param_grid['seasonality_prior_scale']:
            for holidays_prior_scale in param_grid['holidays_prior_scale']:
                PARAM_COUNTER += 1
                print(f"Testing Params #{PARAM_COUNTER}: "
                      f"changepoint={changepoint_prior_scale}, "
                      f"seasonality={seasonality_prior_scale}, "
                      f"holidays={holidays_prior_scale}")
                try:
                    forecast, _ = forecast_with_custom_params(
                        ts_data, forecast_data,
                        changepoint_prior_scale,
                        seasonality_prior_scale,
                        holidays_prior_scale,
                        horizon
                    )
                    if forecast.empty or 'Forecast' not in forecast.columns:
                        raise ValueError("Forecast failed to generate 'Forecast' column for Prophet.")

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
        best_params = {
            'changepoint_prior_scale': 0.1,
            'seasonality_prior_scale': 1,
            'holidays_prior_scale': 10
        }
        best_rmse_values = {}
    return best_params, best_rmse_values

def calculate_rmse(forecast, forecast_data, horizon):
    rmse_values = {}
    for forecast_type, values in forecast_data.items():
        val_slice = values[:horizon]
        err = (forecast['Forecast'] - val_slice[:len(forecast)])**2
        rmse_values[forecast_type] = np.sqrt(err.mean())
    return rmse_values

##############################
# Formatting and Visualization
##############################

def format_output_with_forecasts(forecast_df, forecast_data, horizon=16):
    """
    We'll unify the final 'Forecast' column, merging any Amazon forecast columns.
    Then we do a fallback if it's out of a 30% difference range compared to 'Amazon Mean'.
    """
    comparison = forecast_df.copy()

    # Merge in the Amazon columns
    for forecast_type, values in forecast_data.items():
        val_slice = values[:horizon]
        val_slice = np.array(val_slice, dtype=int)
        temp_df = pd.DataFrame({
            'ds': comparison['ds'],
            f'Amazon {forecast_type}': val_slice
        })
        comparison = comparison.merge(temp_df, on='ds', how='left')

    comparison.fillna(0, inplace=True)

    # If we do not see 'Forecast' in columns, check if 'SARIMA Forecast' or 'Forecast' from Prophet is present
    # But we already unify 'Forecast' for Prophet, 'Forecast' for SARIMA, so we should have 'Forecast' by default.
    if 'Forecast' not in comparison.columns:
        raise KeyError("No 'Forecast' column found. Make sure SARIMA or Prophet returns 'Forecast'.")

    # 30% difference tolerance check if 'Amazon Mean' is present
    if 'Amazon Mean' in comparison.columns:
        for i in range(len(comparison)):
            model_val = comparison.loc[i, 'Forecast']
            amz_mean = comparison.loc[i, 'Amazon Mean']
            if amz_mean == 0:
                continue
            abs_percent_diff = abs(model_val - amz_mean) / float(amz_mean)
            if abs_percent_diff > PERCENT_TOLERANCE:
                print(f"Row {i} => ModelVal={model_val} is outside 30% range of AmazonMean={amz_mean}. Fallback used.")
                fallback_val = 0.5 * model_val + 0.5 * amz_mean
                fallback_val = max(0, round(fallback_val))
                comparison.loc[i, 'Forecast'] = fallback_val

    # Clip negative
    comparison['Forecast'] = comparison['Forecast'].clip(lower=0).round().astype(int)

    # Create "Diff_*" and "Pct_*" columns
    for col in comparison.columns:
        if col.startswith('Amazon '):
            diff_col = f"Diff_{col.split('Amazon ')[1]}"
            pct_col = f"Pct_{col.split('Amazon ')[1]}"
            comparison[diff_col] = comparison['Forecast'] - comparison[col]
            comparison[pct_col] = np.where(
                comparison[col] != 0,
                (comparison[diff_col] / comparison[col]) * 100,
                0
            )

    return comparison

def adjust_forecast_weights(forecast, yhat_weight, yhat_upper_weight):
    if 'yhat' not in forecast or 'yhat_upper' not in forecast:
        raise KeyError("'yhat' or 'yhat_upper' not found in forecast DataFrame.")
    adj_forecast = (yhat_weight * forecast['yhat'] +
                    yhat_upper_weight * forecast['yhat_upper']).clip(lower=0)
    adj_forecast = adj_forecast.round().astype(int)
    forecast['Forecast'] = adj_forecast
    return forecast

def auto_find_best_weights(forecast, comparison, horizon=16, step=0.05):
    best_rmse = float('inf')
    best_weights = None
    candidates = np.arange(0.5, 1.0 + step, step)
    for w in candidates:
        yhat_weight = w
        yhat_upper_weight = 1 - w
        
        temp = forecast.copy()
        # Weighted forecast
        temp['Forecast'] = (
            yhat_weight * temp['yhat'] + yhat_upper_weight * temp['yhat_upper']
        ).clip(lower=0).round().astype(int)
        
        # Now call format_output_with_forecasts with horizon=16 (or your actual horizon)
        comp = format_output_with_forecasts(temp, comparison, horizon)
        
        # Then compute RMSE vs. Amazon columns, etc.
        rmse_vals = {}
        for col in comp.columns:
            if col.startswith('Amazon '):
                rmse = np.sqrt(((comp['Forecast'] - comp[col])**2).mean())
                rmse_vals[col] = rmse

        if rmse_vals:
            avg_rmse = np.mean(list(rmse_vals.values()))
            if avg_rmse < best_rmse:
                best_rmse = avg_rmse
                best_weights = (yhat_weight, yhat_upper_weight)

    return best_weights, best_rmse


##############################
# Cross-Validation for Prophet
##############################

def cross_validate_prophet_model(ts_data,
                                 initial='365 days',
                                 period='180 days',
                                 horizon='180 days'):
    from prophet import Prophet
    model = Prophet(
        growth='linear',
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

def try_cross_validation_with_fallback(model, ts_data, horizons, initial='365 days', period='180 days'):
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
    from prophet import Prophet
    model = Prophet(
        growth='linear',
        yearly_seasonality=True,
        weekly_seasonality=True,
        seasonality_mode='multiplicative',
        changepoint_prior_scale=best_params['changepoint_prior_scale'],
        seasonality_prior_scale=best_params['seasonality_prior_scale'],
        holidays_prior_scale=best_params['holidays_prior_scale']
    )
    model.fit(ts_data[['ds', 'y']])
    candidate_horizons = ['180 days', '90 days', '60 days', '30 days']
    df_cv, df_performance = try_cross_validation_with_fallback(model, ts_data, candidate_horizons,
                                                               initial=initial, period=period)
    if df_cv is not None and df_performance is not None:
        print("Validation Metrics with chosen best_params:")
        print(df_performance)
        return True
    else:
        print("No valid cross-validation results due to insufficient data.")
        return False

##############################
# Summaries and Graphing
##############################

def calculate_summary_statistics(ts_data, forecast_df, horizon):
    # We'll look for 'Forecast' in forecast_df
    forecast_col = 'Forecast'
    if forecast_col not in forecast_df.columns:
        raise KeyError("'Forecast' column not found in forecast DataFrame. Ensure unification was done properly.")

    summary_stats = {
        "min": ts_data["y"].min(),
        "max": ts_data["y"].max(),
        "mean": ts_data["y"].mean(),
        "median": ts_data["y"].median(),
        "std_dev": ts_data["y"].std(),
        "total_sales": ts_data["y"].sum(),
        "data_range": (ts_data["ds"].min(), ts_data["ds"].max())
    }

    total_forecast_16 = forecast_df[forecast_col][:16].sum() if len(forecast_df) >= 16 else forecast_df[forecast_col].sum()
    total_forecast_8 = forecast_df[forecast_col][:8].sum() if len(forecast_df) >= 8 else forecast_df[forecast_col].sum()
    total_forecast_4 = forecast_df[forecast_col][:4].sum() if len(forecast_df) >= 4 else forecast_df[forecast_col].sum()

    max_forecast = forecast_df[forecast_col].max()
    min_forecast = forecast_df[forecast_col].min()
    max_week = forecast_df.loc[forecast_df[forecast_col].idxmax(), 'ds'] if 'ds' in forecast_df.columns else None
    min_week = forecast_df.loc[forecast_df[forecast_col].idxmin(), 'ds'] if 'ds' in forecast_df.columns else None
    return (summary_stats,
            total_forecast_16,
            total_forecast_8,
            total_forecast_4,
            max_forecast,
            min_forecast,
            max_week,
            min_week)

def visualize_forecast_with_comparison(ts_data, comparison, summary_stats,
                                       total_forecast_16, total_forecast_8, total_forecast_4,
                                       max_forecast, min_forecast, max_week, min_week,
                                       asin, product_title, folder_name):
    # We need a 'Forecast' column in 'comparison'
    if 'Forecast' not in comparison.columns:
        # If needed, unify again, or raise error
        raise KeyError("'Forecast' column is missing in the comparison DataFrame. Cannot plot.")

    # Ensure we have 'ds'
    if 'ds' not in comparison.columns:
        # Attempt to unify with original ts_data
        comparison = pd.concat([ts_data[['ds']], comparison], axis=1).drop_duplicates(subset=['ds'])
        if 'ds' not in comparison.columns:
            raise KeyError("'ds' column is missing in the comparison DataFrame even after merging.")

    fig, ax = plt.subplots(figsize=(16, 10))
    ax.plot(ts_data['ds'], ts_data['y'], label='Historical Sales', marker='o', color='black')
    ax.plot(comparison['ds'], comparison['Forecast'], label='Forecast', marker='o', linestyle='--', color='blue')

    for column in comparison.columns:
        if column.startswith('Amazon '):
            ax.plot(comparison['ds'], comparison[column], label=column, linestyle=':')

    ax.set_title(f'Sales Forecast Comparison for {product_title}')
    ax.set_xlabel('Date')
    ax.set_ylabel('Units Sold')
    ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1))
    ax.grid()

    # Summary Text
    summary_text = (
        f"Historical Summary:\n"
        f"Range: {summary_stats['data_range'][0].strftime('%Y-%m-%d')} to "
        f"{summary_stats['data_range'][1].strftime('%Y-%m-%d')}\n"
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
        f"Max Forecast: {max_forecast:.0f} "
        f"(Week of {max_week.strftime('%Y-%m-%d') if max_week else 'N/A'})\n"
        f"Min Forecast: {min_forecast:.0f} "
        f"(Week of {min_week.strftime('%Y-%m-%d') if min_week else 'N/A'})"
    )

    plt.gcf().text(0.78, 0.5, summary_text, fontsize=10,
                   bbox=dict(facecolor='white', alpha=0.8), va='center')
    plt.subplots_adjust(right=0.75)

    os.makedirs(folder_name, exist_ok=True)
    graph_file_path = os.path.join(folder_name, f"{product_title.replace('/', '_')}_{asin}.png")
    plt.savefig(graph_file_path)
    plt.close()
    print(f"Graph saved to {graph_file_path}")

##############################
# Excel Saving
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
    from openpyxl import Workbook
    from openpyxl.utils.dataframe import dataframe_to_rows

    desired_columns = [
        'Week', 'ASIN', 'Forecast', 'Amazon Mean Forecast',
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
            f"{summary_stats['data_range'][0].strftime('%Y-%m-%d')} "
            f"to {summary_stats['data_range'][1].strftime('%Y-%m-%d')}",
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
    from openpyxl import Workbook
    from openpyxl.utils.dataframe import dataframe_to_rows

    wb = Workbook()
    for asin, forecast_df in consolidated_data.items():
        df_for_excel = forecast_df.copy()
        if 'ds' in df_for_excel.columns:
            for i in range(len(df_for_excel)):
                df_for_excel.loc[i, 'Week'] = f"W{str(i+1).zfill(2)}"
            df_for_excel.drop(columns=['ds'], inplace=True, errors='ignore')

        desired_columns = [
            'Week', 'ASIN', 'Forecast', 'Amazon Mean Forecast',
            'Amazon P70 Forecast', 'Amazon P80 Forecast', 'Amazon P90 Forecast',
            'Product Title', 'is_holiday_week'
        ]
        for col in desired_columns:
            if col not in df_for_excel.columns:
                df_for_excel[col] = np.nan

        df_for_excel = df_for_excel[desired_columns]
        ws = wb.create_sheet(title=str(asin)[:31])
        for r in dataframe_to_rows(df_for_excel, index=False, header=True):
            ws.append(r)

    if not missing_asin_data.empty:
        ws_missing = wb.create_sheet(title="No ASIN")
        for r in dataframe_to_rows(missing_asin_data, index=False, header=True):
            ws_missing.append(r)

    if 'Sheet' in wb.sheetnames:
        del wb['Sheet']

    wb.save(output_path)
    print(f"All forecasts saved to {output_path}")

def save_4_16weeks_forecast(consolidated_forecasts, output_file_path):
    """
    Create a new Excel file that contains columns:
        ASIN,
        Total Forecast for first 4 weeks,
        Total Forecast for first 8 weeks,
        Total Forecast for first 16 weeks.
    """
    from openpyxl import Workbook
    from openpyxl.utils.dataframe import dataframe_to_rows

    data_list = []
    for asin, df in consolidated_forecasts.items():
        # Decide which forecast column to use
        if "Forecast" in df.columns:
            forecast_col = "Forecast"
        else:
            raise KeyError(f"No 'Forecast' column found for ASIN {asin} in consolidated data.")

        total_4 = df[forecast_col].iloc[:4].sum() if len(df) >= 4 else df[forecast_col].sum()
        total_8 = df[forecast_col].iloc[:8].sum() if len(df) >= 8 else df[forecast_col].sum()
        total_16 = df[forecast_col].iloc[:16].sum() if len(df) >= 16 else df[forecast_col].sum()

        data_list.append({
            'ASIN': asin,
            '4 Weeks Forecast': total_4,
            '8 Weeks Forecast': total_8,
            '16 Weeks Forecast': total_16
        })

    result_df = pd.DataFrame(data_list)

    wb = Workbook()
    ws = wb.active
    ws.title = "4-16 Weeks Forecast"

    for row in dataframe_to_rows(result_df, index=False, header=True):
        ws.append(row)

    wb.save(output_file_path)
    print(f"Saved 4/8/16-weeks forecast summary to '{output_file_path}'")

##############################
# Analyze Amazon Buying Habits
##############################

def analyze_amazon_buying_habits(comparison, holidays):
    amazon_cols = [col for col in comparison.columns if col.startswith('Amazon ')]
    if not amazon_cols:
        print("No Amazon forecasts found for analysis.")
        return

    if 'Forecast' in comparison.columns:
        model_col = 'Forecast'
    else:
        raise KeyError("No 'Forecast' column to compare against Amazon forecasts in analyze_amazon_buying_habits.")

    print("Columns in `comparison` df:", comparison.columns.tolist())
    print("Unique columns in `comparison` df:", set(comparison.columns))
    print("Shape of comparison[model_col]:", comparison[model_col].shape)

    model_forecast_vals = comparison[model_col].values

    # If we want a holiday mask
    if 'is_holiday_week' in comparison.columns:
        holiday_mask = comparison['is_holiday_week']
    else:
        # or create one from the DS
        holiday_mask = comparison['ds'].isin(holidays['ds'].values) if 'ds' in comparison.columns else np.array([False]*len(comparison))

    for col in amazon_cols:
        amazon_forecast = comparison[col].values
        safe_model_forecast = np.where(model_forecast_vals == 0, 1e-9, model_forecast_vals)

        ratio = amazon_forecast / safe_model_forecast
        diff = amazon_forecast - model_forecast_vals
        avg_ratio = np.mean(ratio)
        avg_diff = np.mean(diff)
        print(f"\nFor {col}:")
        print(f"  Average Amazon/Model Ratio: {avg_ratio:.2f}")
        print(f"  Average Difference (Amazon - Model): {avg_diff:.2f}")

        if avg_diff > 0:
            print("  Amazon tends to forecast more than our model on average.")
        elif avg_diff < 0:
            print("  Amazon tends to forecast less than our model on average.")
        else:
            print("  Amazon forecasts similarly to our model on average.")

        if holiday_mask.any():
            holiday_ratio = ratio[holiday_mask]
            holiday_diff = diff[holiday_mask]
            if len(holiday_diff) > 0:
                print("  During holiday weeks:")
                print(f"    Avg Ratio (Amazon/Model): {np.mean(holiday_ratio):.2f}")
                print(f"    Avg Diff (Amazon-Model): {np.mean(holiday_diff):.2f}")

        weeks = np.arange(1, len(ratio)+1)
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
                print(f"    Avg Ratio (Amazon/Model): {np.mean(seg_ratio):.2f}")
                print(f"    Avg Diff (Amazon-Model): {np.mean(seg_diff):.2f}")

##############################
# Main Execution
##############################

def main():
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
    asin_list = [a for a in asin_list if a in asins_to_forecast]

    consolidated_forecasts = {}
    param_grid = {
        'changepoint_prior_scale': [0.05, 0.1, 0.2],
        'seasonality_prior_scale': [0.5, 1, 2],
        'holidays_prior_scale': [5, 10]
    }

    holidays = get_shifted_holidays()

    # Optional cross-validation on the first ASIN
    if len(asin_list) > 0:
        test_asin = asin_list[0]
        test_ts_data = prepare_time_series_with_lags(valid_data, test_asin, lag_weeks=1)
        if not test_ts_data.empty and len(test_ts_data.dropna()) >= 2:
            print(f"Performing cross-validation on ASIN {test_asin} Prophet model...")
            cross_validate_prophet_model(test_ts_data, initial='365 days', period='180 days', horizon='90 days')
        else:
            print(f"Not enough data for {test_asin} to perform cross-validation test.")

    insufficient_data_folder = "Insufficient data"
    sufficient_data_folder = "Sufficient data"
    os.makedirs(insufficient_data_folder, exist_ok=True)
    os.makedirs(sufficient_data_folder, exist_ok=True)

    global PARAM_COUNTER, POOR_PARAM_FOUND

    for asin in asin_list:
        product_title = valid_data[valid_data['asin'] == asin]['product title'].iloc[0]
        print(f"\nProcessing ASIN: {asin} - {product_title}")
        forecast_data = load_amazon_forecasts_from_folder(forecasts_folder, asin)
        if not forecast_data:
            print(f"No forecast data found for ASIN {asin}, skipping.")
            continue

        ts_data = prepare_time_series_with_lags(valid_data, asin, lag_weeks=1)
        non_nan_count = len(ts_data.dropna(subset=['y']))
        if non_nan_count < 2:
            print(f"Not enough data for ASIN {asin} (only {non_nan_count} data points), skipping.")
            no_data_output = os.path.join(insufficient_data_folder, f"{asin}_no_data.txt")
            with open(no_data_output, 'w') as f:
                f.write("Insufficient data for training/forecasting.\n")
            continue

        model, model_type = choose_forecast_model(ts_data, threshold=50, holidays=holidays)

        if model_type == "SARIMA":
            # SARIMA logic
            n = len(ts_data)
            split = int(n * 0.8)
            train_sarima = ts_data.iloc[:split]
            test_sarima = ts_data.iloc[split:]

            exog_train = create_holiday_regressors(train_sarima, holidays)
            exog_test = create_holiday_regressors(test_sarima, holidays)

            if model is not None:
                try:
                    steps = len(test_sarima)
                    preds_df = sarima_forecast(
                        model, steps=steps,
                        last_date=train_sarima['ds'].iloc[-1],
                        exog=exog_test
                    )
                    preds = preds_df['Forecast'].values

                    sarima_mape = mean_absolute_percentage_error(test_sarima['y'], preds)
                    sarima_rmse = sqrt(mean_squared_error(test_sarima['y'], preds))
                    print(f"SARIMA Test MAPE: {sarima_mape:.4f}, RMSE: {sarima_rmse:.4f}")

                    last_date_full = ts_data['ds'].iloc[-1]
                    exog_future = generate_future_exog(holidays, steps=horizon, last_date=last_date_full)
                    final_forecast_df = sarima_forecast(
                        model, steps=horizon, last_date=last_date_full, exog=exog_future
                    )
                    if final_forecast_df.empty:
                        print(f"Forecasting failed for ASIN {asin}, skipping.")
                        no_data_output = os.path.join(insufficient_data_folder, f"{asin}_forecast_failed.txt")
                        with open(no_data_output, 'w') as f:
                            f.write("Failed to forecast due to insufficient data.\n")
                        continue

                    # final_forecast_df => columns: ds, Forecast
                    comparison = format_output_with_forecasts(final_forecast_df, forecast_data, horizon=horizon)
                    # Now we keep the column name as 'Forecast' (no rename)

                    comparison['ASIN'] = asin
                    comparison['Product Title'] = product_title

                    comparison = comparison.merge(ts_data[['ds', 'y']], on='ds', how='left')
                    comparison_historical = comparison.dropna(subset=['y'])

                    if comparison_historical.empty:
                        print("No overlapping historical data to calculate metrics. Skipping metrics.")
                        metrics = {}
                    else:
                        mae_val = mean_absolute_error(comparison_historical['y'], comparison_historical['Forecast'])
                        medae_val = median_absolute_error(comparison_historical['y'], comparison_historical['Forecast'])
                        mse_val = mean_squared_error(comparison_historical['y'], comparison_historical['Forecast'])
                        rmse_val = sqrt(mse_val)
                        mape_val = mean_absolute_percentage_error(comparison_historical['y'], comparison_historical['Forecast'])

                        print('Mean Absolute Error (MAE): ' + str(np.round(mae_val, 2)))
                        print('Median Absolute Error (MedAE): ' + str(np.round(medae_val, 2)))
                        print('Mean Squared Error (MSE): ' + str(np.round(mse_val, 2)))
                        print('Root Mean Squared Error (RMSE): ' + str(np.round(rmse_val, 2)))
                        print('Mean Absolute Percentage Error (MAPE): ' + str(np.round(mape_val, 2)) + ' %')

                        metrics = {
                            "Mean Absolute Error (MAE)": np.round(mae_val, 2),
                            "Median Absolute Error (MedAE)": np.round(medae_val, 2),
                            "Mean Squared Error (MSE)": np.round(mse_val, 2),
                            "Root Mean Squared Error (RMSE)": np.round(rmse_val, 2),
                            "Mean Absolute Percentage Error (MAPE)": str(np.round(mape_val, 2)) + " %"
                        }

                    # optional blend with Amazon Mean
                    if 'Amazon Mean' in comparison.columns:
                        comparison['Forecast'] = (
                            SARIMA_WEIGHT * comparison['Forecast']
                            + (1 - SARIMA_WEIGHT) * comparison['Amazon Mean']
                        ).clip(lower=0).round().astype(int)

                    (summary_stats,
                     total_forecast_16,
                     total_forecast_8,
                     total_forecast_4,
                     max_forecast,
                     min_forecast,
                     max_week,
                     min_week) = calculate_summary_statistics(ts_data, comparison, horizon=horizon)

                    visualize_forecast_with_comparison(
                        ts_data, comparison, summary_stats,
                        total_forecast_16, total_forecast_8, total_forecast_4,
                        max_forecast, min_forecast, max_week, min_week,
                        asin, product_title, sufficient_data_folder
                    )

                    output_file_name = f'forecast_summary_{asin}.xlsx'
                    output_file_path = os.path.join(sufficient_data_folder, output_file_name)
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

                except ValueError as e:
                    print(f"Error during SARIMA prediction for ASIN {asin}: {e}")
                    continue
            else:
                print(f"SARIMA model fitting failed for {asin}, skipping.")
                continue

        else:
            # Prophet logic
            cached_model_path = os.path.join("model_cache", f"{asin}_Prophet.pkl")
            if os.path.exists(cached_model_path):
                if is_model_up_to_date(cached_model_path, ts_data):
                    print(f"Using up-to-date cached Prophet model for ASIN {asin}.")
                    cached_prophet_model = joblib.load(cached_model_path)
                    last_train_date = ts_data['ds'].max()
                    future_dates = pd.date_range(
                        start=last_train_date + pd.Timedelta(days=7),
                        periods=horizon,
                        freq='W'
                    )
                    future = pd.DataFrame({'ds': future_dates})
                    for forecast_type in forecast_data.keys():
                        future[f"Amazon_{forecast_type}"] = 0
                    future['prime_day'] = 0
                    prophet_output = cached_prophet_model.predict(future)
                    # unify the forecast column
                    prophet_output['Forecast'] = prophet_output['yhat'].clip(lower=0).round().astype(int)
                    forecast = prophet_output[['ds', 'Forecast', 'yhat', 'yhat_upper']]
                else:
                    print(f"Cached Prophet model for ASIN {asin} is outdated. Retraining with updated data...")
                    best_params, _ = optimize_prophet_params(ts_data, forecast_data, param_grid, horizon=horizon)
                    forecast, trained_prophet_model = forecast_with_custom_params(
                        ts_data,
                        forecast_data,
                        best_params['changepoint_prior_scale'],
                        best_params['seasonality_prior_scale'],
                        best_params['holidays_prior_scale'],
                        horizon=horizon
                    )
                    if trained_prophet_model is not None:
                        save_model(trained_prophet_model, "Prophet", asin, ts_data)
                    else:
                        print("Failed to retrain the Prophet model.")
            else:
                print(f"No cached Prophet model found for ASIN {asin}. Training a new model...")
                best_params, _ = optimize_prophet_params(
                    ts_data, forecast_data, param_grid, horizon=horizon
                )
                forecast, trained_prophet_model = forecast_with_custom_params(
                    ts_data,
                    forecast_data,
                    best_params['changepoint_prior_scale'],
                    best_params['seasonality_prior_scale'],
                    best_params['holidays_prior_scale'],
                    horizon=horizon
                )
                if trained_prophet_model is not None:
                    save_model(trained_prophet_model, "Prophet", asin, ts_data)
                else:
                    print("Failed to train the Prophet model.")

            if 'forecast' not in locals() or forecast.empty:
                print(f"Forecasting failed for ASIN {asin}, skipping.")
                no_data_output = os.path.join(insufficient_data_folder, f"{asin}_final_forecast_failed.txt")
                with open(no_data_output, 'w') as f:
                    f.write("Final forecasting failed.\n")
                continue

            comparison = format_output_with_forecasts(forecast, forecast_data, horizon=horizon)

            # auto find best yhat/yhat_upper weights
            best_weights, best_rmse = auto_find_best_weights(forecast, forecast_data)
            # but to keep code simpler, let's replicate your approach
            # (or do: best_weights, best_rmse = auto_find_best_weights(forecast, comparison, step=0.05))
            print(f"Auto best weights for ASIN {asin}: {best_weights} with RMSE={best_rmse}")

            forecast = adjust_forecast_weights(forecast, *best_weights)
            comparison = format_output_with_forecasts(forecast, forecast_data, horizon=horizon)

            comparison['ASIN'] = asin
            comparison['Product Title'] = product_title

            # Insert is_holiday_week if missing
            if 'is_holiday_week' not in comparison.columns:
                comparison['is_holiday_week'] = comparison['ds'].apply(
                    lambda x: x in holidays['ds'].values
                )

            analyze_amazon_buying_habits(comparison, holidays)

            comparison = comparison.merge(ts_data[['ds', 'y']], on='ds', how='left')
            comparison_historical = comparison.dropna(subset=['y'])

            if comparison_historical.empty:
                print("No overlapping historical data to calculate metrics. Skipping metrics.")
                metrics = {}
            else:
                mae_val = mean_absolute_error(
                    comparison_historical['y'],
                    comparison_historical['Forecast']
                )
                medae_val = median_absolute_error(
                    comparison_historical['y'],
                    comparison_historical['Forecast']
                )
                mse_val = mean_squared_error(
                    comparison_historical['y'],
                    comparison_historical['Forecast']
                )
                rmse_val = sqrt(mse_val)
                mape_val = mean_absolute_percentage_error(
                    comparison_historical['y'],
                    comparison_historical['Forecast']
                )

                print('Mean Absolute Error (MAE): ' + str(np.round(mae_val, 2)))
                print('Median Absolute Error (MedAE): ' + str(np.round(medae_val, 2)))
                print('Mean Squared Error (MSE): ' + str(np.round(mse_val, 2)))
                print('Root Mean Squared Error (RMSE): ' + str(np.round(rmse_val, 2)))
                print('Mean Absolute Percentage Error (MAPE): ' + str(np.round(mape_val, 2)) + ' %')

                metrics = {
                    "Mean Absolute Error (MAE)": np.round(mae_val, 2),
                    "Median Absolute Error (MedAE)": np.round(medae_val, 2),
                    "Mean Squared Error (MSE)": np.round(mse_val, 2),
                    "Root Mean Squared Error (RMSE)": np.round(rmse_val, 2),
                    "Mean Absolute Percentage Error (MAPE)": str(np.round(mape_val, 2)) + " %"
                }

            # Create Week label
            for i in range(len(comparison)):
                comparison.loc[i, 'Week'] = f"W{str(i+1).zfill(2)}"

            desired_final_cols = [
                'ds',
                'Week',
                'ASIN',
                'Forecast',
                'Amazon Mean Forecast',
                'Amazon P70 Forecast',
                'Amazon P80 Forecast',
                'Amazon P90 Forecast',
                'Product Title',
                'is_holiday_week'
            ]
            comparison = comparison[[col for col in desired_final_cols if col in comparison.columns]]

            (summary_stats,
             total_forecast_16,
             total_forecast_8,
             total_forecast_4,
             max_forecast,
             min_forecast,
             max_week,
             min_week) = calculate_summary_statistics(ts_data, comparison, horizon=horizon)

            visualize_forecast_with_comparison(
                ts_data, comparison, summary_stats,
                total_forecast_16, total_forecast_8, total_forecast_4,
                max_forecast, min_forecast, max_week, min_week,
                asin, product_title, sufficient_data_folder
            )

            output_file_name = f'forecast_summary_{asin}.xlsx'
            output_file_path = os.path.join(sufficient_data_folder, output_file_name)
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

    final_output_path = output_file
    save_forecast_to_excel(final_output_path, consolidated_forecasts, missing_asin_data)

    forecast_4_16_file = "4-16weeks forecast.xlsx"
    save_4_16weeks_forecast(consolidated_forecasts, forecast_4_16_file)

    print(f"Total number of parameter sets tested: {PARAM_COUNTER}")
    if POOR_PARAM_FOUND:
        print("Note: Early stopping occurred for some ASINs due to poor parameter performance.")


if __name__ == '__main__':
    main()
