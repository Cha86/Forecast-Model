# models/prophet_helpers.py

import numpy as np
import pandas as pd
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
from sklearn.metrics import mean_absolute_error, mean_squared_error
from math import sqrt
from reward_penalty import update_param_history
from config import prophet_param_history, EARLY_STOP_THRESHOLD, POOR_PARAM_FOUND, PARAM_COUNTER, prophet_feedback

def forecast_with_custom_params(ts_data, forecast_data,
                                changepoint_prior_scale,
                                seasonality_prior_scale,
                                holidays_prior_scale,
                                horizon=16,
                                weights=None,
                                get_shifted_holidays=None):
    """
    Train Prophet with custom parameters and generate forecasts.
    """
    if get_shifted_holidays is None:
        raise ValueError("get_shifted_holidays function not provided!")

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
        values_to_use = values[:horizon] if len(values) > horizon else values
        extended_values = np.concatenate([
            np.full(len(ts_data), np.nan),
            values_to_use,
            np.full(max(horizon - len(values_to_use), 0),
                    values_to_use[-1] if len(values_to_use) > 0 else 0)
        ])
        extended_values = extended_values[:len(combined_df)]
        combined_df[f'Amazon_{forecast_type} Forecast'] = extended_values

    regressor_cols = [col for col in combined_df.columns if col.startswith('Amazon_')]
    combined_df[regressor_cols] = combined_df[regressor_cols].fillna(0)

    holidays_df = get_shifted_holidays()
    combined_df['prime_day'] = combined_df['ds'].apply(
        lambda x: 0.2 if x in holidays_df[holidays_df['holiday'] == 'Prime Day']['ds'].values else 0
    )

    n = len(ts_data)
    split = int(n * 0.8)
    train_df = combined_df.iloc[:split].dropna(subset=['y']).copy()
    test_df = combined_df.iloc[split:].copy()

    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        seasonality_mode='multiplicative',
        holidays=holidays_df,
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

def optimize_prophet_params(ts_data, forecast_data, param_grid, horizon=16, asin=None, get_shifted_holidays=None):
    """
    Optimize Prophet model parameters using grid search and parameter history.
    """
    from config import PARAM_COUNTER, POOR_PARAM_FOUND
    from config import prophet_feedback, EARLY_STOP_THRESHOLD, prophet_param_history

    best_rmse = float('inf')
    best_params = None
    best_rmse_values = None
    skip_threshold_score = 0.001

    for changepoint_prior_scale in param_grid['changepoint_prior_scale']:
        for seasonality_prior_scale in param_grid['seasonality_prior_scale']:
            for holidays_prior_scale in param_grid['holidays_prior_scale']:
                PARAM_COUNTER += 1
                param_tuple = (changepoint_prior_scale, seasonality_prior_scale, holidays_prior_scale)

                if asin is not None and param_tuple in prophet_param_history:
                    if prophet_param_history[(asin, param_tuple)]['score'] < skip_threshold_score:
                        continue

                print(f"Testing Params #{PARAM_COUNTER} for {asin}: changepoint={changepoint_prior_scale}, "
                      f"seasonality={seasonality_prior_scale}, holidays={holidays_prior_scale}")

                try:
                    forecast, _ = forecast_with_custom_params(
                        ts_data, forecast_data,
                        changepoint_prior_scale,
                        seasonality_prior_scale,
                        holidays_prior_scale,
                        horizon=horizon,
                        get_shifted_holidays=get_shifted_holidays
                    )
                    if forecast.empty or 'MyForecast' not in forecast.columns:
                        raise ValueError("Forecast failed to generate 'MyForecast' column.")

                    # Calculate average RMSE across the Amazon forecast streams
                    rmse_values = calculate_rmse(forecast, forecast_data, horizon)
                    avg_rmse = np.mean(list(rmse_values.values()))
                    mae_val = avg_rmse

                    # Update param history with reward
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

    asin_label = asin if asin is not None else 'unknown'
    prophet_feedback[asin_label] = {
        'best_params': best_params,
        'rmse_values': best_rmse_values,
        'total_tests': PARAM_COUNTER
    }
    return best_params, best_rmse_values

def calculate_rmse(forecast, forecast_data, horizon):
    """
    Calculate RMSE comparing 'MyForecast' vs. Amazon forecast streams.
    """
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
