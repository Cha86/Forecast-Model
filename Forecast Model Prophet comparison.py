import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt


def load_data(file_path):
    """Load and preprocess sales data from a CSV file."""
    expected_columns = ['date', 'ASIN', 'Model', 'Brand', 'units_sold']
    data = pd.read_csv(file_path)

    if len(data.columns) > len(expected_columns):
        data = data.iloc[:, :len(expected_columns)]

    data.columns = expected_columns
    data['date'] = pd.to_datetime(data['date'], errors='coerce')
    data = data.sort_values(by='date')
    return data


def load_amazon_forecasts(file_path):
    """Load and clean Amazon forecast data."""
    raw_forecasts = pd.read_excel(file_path, sheet_name=None)  # Read all sheets
    cleaned_forecasts = {}

    for sheet_name, df in raw_forecasts.items():
        df_cleaned = df.dropna(how='all')  # Drop empty rows
        df_cleaned.iloc[:, 1:] = df_cleaned.iloc[:, 1:].apply(pd.to_numeric, errors='coerce').fillna(0)
        cleaned_forecasts[sheet_name] = df_cleaned

    return cleaned_forecasts


def prepare_time_series(data, asin):
    """Prepare time series data for Prophet."""
    asin_data = data[data['ASIN'] == asin]
    if asin_data.empty:
        raise ValueError(f"No data found for ASIN: {asin}")

    ts_data = asin_data.rename(columns={'date': 'ds', 'units_sold': 'y'})
    ts_data['y'] = ts_data['y'].interpolate().bfill().clip(lower=0)
    return ts_data


def forecast_demand(ts_data, horizon=16):
    """Forecast demand using Prophet."""
    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        seasonality_mode='multiplicative',
        changepoint_prior_scale=0.1,
        seasonality_prior_scale=5
    )
    model.fit(ts_data)

    future = model.make_future_dataframe(periods=horizon, freq='W')
    forecast = model.predict(future)
    forecast['yhat'] = forecast['yhat'].clip(lower=0)
    return model, forecast


def visualize_forecast_with_comparison(ts_data, prophet_forecast, amazon_forecasts, horizon=16):
    """Visualize historical data, Prophet forecast, and Amazon forecasts."""
    plt.figure(figsize=(16, 10))

    # Plot historical data
    plt.plot(ts_data['ds'], ts_data['y'], label='Historical Sales', marker='o', color='black')

    # Plot Prophet forecast
    prophet_forecast_horizon = prophet_forecast.iloc[-horizon:]
    plt.plot(
        prophet_forecast_horizon['ds'], 
        prophet_forecast_horizon['yhat'], 
        label='Prophet Forecast', 
        marker='o', 
        linestyle='--', 
        color='blue'
    )

    # Plot Amazon forecasts
    for sheet_name, df in amazon_forecasts.items():
        forecast_values = df.iloc[:, -horizon:].mean(axis=0).values
        plt.plot(
            prophet_forecast_horizon['ds'], 
            forecast_values, 
            label=f'{sheet_name} (Amazon Forecast)', 
            linestyle='--'
        )

    plt.title('Sales Forecast Comparison')
    plt.xlabel('Date')
    plt.ylabel('Units Sold')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()


def main():
    file_path = 'weekly_sales_data.csv'
    amazon_forecast_path = 'amazon_forecasts.xlsx'
    asin = 'B0BH7GTY9C'

    # Load datasets
    data = load_data(file_path)
    amazon_forecasts = load_amazon_forecasts(amazon_forecast_path)

    # Prepare data and forecast
    ts_data = prepare_time_series(data, asin)
    model, forecast = forecast_demand(ts_data, horizon=16)

    # Visualize results
    visualize_forecast_with_comparison(ts_data, forecast, amazon_forecasts, horizon=16)


if __name__ == '__main__':
    main()
