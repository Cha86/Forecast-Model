# models/xgboost_helpers.py

import pandas as pd
import numpy as np
import shap
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error
from math import sqrt

def train_xgboost(ts_data, target='y', features=None):
    """
    Train an XGBoost regressor on the time series data with specified features.
    """
    if features is None:
        features = [col for col in ts_data.columns if col.startswith('lag_')]
    valid_data = ts_data.dropna(subset=[target] + features)
    if valid_data.empty:
        print("No valid data available for XGBoost training.")
        return None, None, None

    X = valid_data[features]
    y = valid_data[target]

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
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

    explainer = shap.Explainer(model)
    shap_values = explainer(X)

    shap.summary_plot(shap_values, X, show=False)

    return model, features, shap_values

def xgboost_forecast(model, ts_data, forecast_steps=16, target='y', features=None):
    """
    Generate XGBoost forecasts for the specified number of steps ahead.
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
