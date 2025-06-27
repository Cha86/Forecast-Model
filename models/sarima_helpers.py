# models/sarima_helpers.py

import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from reward_penalty import update_param_history
from config import sarima_param_history
from sklearn.metrics import mean_squared_error, mean_absolute_error
from math import sqrt

def calculate_forecast_metrics(actual, predicted):
    """
    Calculate RMSE and MAE between actual and predicted values.
    """
    rmse = sqrt(mean_squared_error(actual, predicted))
    mae = mean_absolute_error(actual, predicted)
    return rmse, mae

def create_holiday_regressors(ts_data, holidays):
    """
    Create exogenous regressors based on holidays.
    """
    holiday_names = holidays['holiday'].unique()
    exog = pd.DataFrame(index=ts_data.index)
    exog['ds'] = ts_data['ds']
    for h in holiday_names:
        holiday_dates = holidays[holidays['holiday'] == h]['ds']
        exog[h] = exog['ds'].isin(holiday_dates).astype(int)
    exog.drop(columns=['ds'], inplace=True)
    return exog

def fit_sarima_model(data, holidays, seasonal_period=52, asin=None):
    """
    Fit a SARIMA model with grid search over specified parameters.
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
    skip_threshold_score = 0.001

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
                                            update_param_history(sarima_param_history, asin, param_tuple, rmse, mae)

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
                    param_tuple = (p, d, q, 0, 0, 0, 1)
                    if asin is not None:
                        key = (asin, param_tuple)
                        if key in sarima_param_history:
                            if sarima_param_history[key]['score'] < skip_threshold_score:
                                print(f"Skipping SARIMA params {param_tuple} for ASIN {asin} due to low historical score.")
                                continue
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
                        rmse, mae = calculate_forecast_metrics(actual, forecast)

                        if asin is not None:
                            update_param_history(sarima_param_history, asin, param_tuple, rmse, mae)

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
        return best_model, (
            best_metrics['p'], best_metrics['d'], best_metrics['q'],
            best_metrics['P'], best_metrics['D'], best_metrics['Q'], best_metrics['m']
        )

def sarima_forecast(model_fit, steps, last_date, exog=None):
    """
    Generate SARIMA forecasts for the specified number of steps ahead.
    """
    k_exog = model_fit.model.k_exog
    if exog is not None and k_exog > 0:
        exog = exog.iloc[:, :k_exog]
    forecast_values = model_fit.forecast(steps=steps, exog=exog)
    forecast_values = forecast_values.round().astype(int)
    future_dates = pd.date_range(start=last_date + pd.Timedelta(weeks=1), periods=steps, freq='W')
    return pd.DataFrame({'ds': future_dates, 'MyForecast': forecast_values})

