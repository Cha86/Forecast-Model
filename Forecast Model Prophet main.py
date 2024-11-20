import pandas as pd
from prophet import Prophet
from prophet.plot import add_changepoints_to_plot
import matplotlib.pyplot as plt


def load_data(file_path):
    """Load and preprocess sales data."""
    data = pd.read_csv(file_path)
    data.columns = ['date', 'ASIN', 'Model', 'Brand', 'units_sold']
    data['date'] = pd.to_datetime(data['date'], errors='coerce')
    data = data.sort_values(by='date')
    return data


def create_holidays_dataframe():
    """Create holidays dataframe."""
    holidays = pd.DataFrame({
        'holiday': [
            'Prime Day', 'Black Friday', 'Cyber Monday', 
            'Christmas', 'New Year'
        ],
        'ds': [
            '2023-07-16', '2023-11-24', '2023-12-02',
            '2023-12-25', '2024-01-01'
        ],
        'lower_window': [-1, -1, 0, -1, -1],
        'upper_window': [1, 0, 1, 0, 1]
    })
    holidays['ds'] = pd.to_datetime(holidays['ds'])
    return holidays


def prepare_time_series(data, asin):
    """Prepare time series data."""
    asin_data = data[data['ASIN'] == asin]
    if asin_data.empty:
        raise ValueError(f"No data found for ASIN: {asin}")
    
    ts_data = asin_data.rename(columns={'date': 'ds', 'units_sold': 'y'})
    ts_data['y'] = ts_data['y'].interpolate().bfill().clip(lower=0)  # Fill missing and handle negatives
    ts_data['cap'] = ts_data['y'].max() * 1.5  # Logistic growth capacity
    ts_data['floor'] = 0
    return ts_data


def forecast_demand_with_events(ts_data, holidays, horizon=52):
    """Forecast demand using Prophet."""
    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        seasonality_mode='multiplicative',  # Better for growing seasonality
        holidays=holidays,
        changepoint_prior_scale=0.2,  # Adjust for flexibility
        seasonality_prior_scale=10,
        holidays_prior_scale=10
    )
    model.fit(ts_data)

    future = model.make_future_dataframe(periods=horizon, freq='W')
    future['cap'] = ts_data['cap'].max()
    future['floor'] = 0
    forecast = model.predict(future)

    # Smooth forecast to avoid large spikes
    forecast['yhat'] = forecast['yhat'].clip(lower=0)
    forecast['yhat'] = forecast['yhat'].rolling(window=3, min_periods=1).mean()
    return forecast, model


def plot_forecast(ts_data, forecast, model):
    """Plot forecast vs actuals."""
    plt.figure(figsize=(14, 8))
    plt.plot(ts_data['ds'], ts_data['y'], label='Historical Sales', marker='o', color='blue')
    plt.plot(forecast['ds'], forecast['yhat'], label='Forecast', linestyle='--', color='green')
    plt.title('Sales Forecast with Prophet')
    plt.xlabel('Date')
    plt.ylabel('Units Sold')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()

    # Add changepoints
    fig = model.plot(forecast)
    add_changepoints_to_plot(fig.gca(), model, forecast)


def format_forecast_output(forecast, asin, model, brand):
    """Format the forecast output for Excel."""
    forecast_weeks = [
        f"Week {i + 1} ({row['ds'].strftime('%d %b')} - {(row['ds'] + pd.Timedelta(days=6)).strftime('%d %b')})"
        for i, row in forecast.iterrows()
    ]
    forecast_values = forecast['yhat'].round().astype(int).tolist()
    output = pd.DataFrame({
        "ASIN": [asin],
        "Model": [model],
        "Brand": [brand],
        **{week: [value] for week, value in zip(forecast_weeks, forecast_values)}
    })
    return output


def main():
    file_path = 'weekly_sales_data.csv'  # Replace with your file
    asin = 'B0BH7GTY9C'  # Replace with your ASIN

    data = load_data(file_path)
    holidays = create_holidays_dataframe()
    ts_data = prepare_time_series(data, asin)

    forecast_horizon = 52
    forecast, model = forecast_demand_with_events(ts_data, holidays, horizon=forecast_horizon)

    product_title = data[data['ASIN'] == asin]['Model'].iloc[0]
    brand = data[data['ASIN'] == asin]['Brand'].iloc[0]

    output = format_forecast_output(forecast, asin, product_title, brand)
    output.to_excel('sales_forecast.xlsx', index=False)
    print("Forecast saved to 'sales_forecast.xlsx'")

    plot_forecast(ts_data, forecast, model)


if __name__ == '__main__':
    main()

