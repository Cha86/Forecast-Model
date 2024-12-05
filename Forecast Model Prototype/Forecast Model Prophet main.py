import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt


def load_data(file_path):
    """Load and preprocess sales data from a CSV file."""
    # Define expected columns
    expected_columns = ['date', 'ASIN', 'Model', 'Brand', 'units_sold']
    
    # Load the CSV file
    data = pd.read_csv(file_path)
    
    # Select only the first 5 columns if the CSV contains extra columns
    if len(data.columns) > len(expected_columns):
        data = data.iloc[:, :len(expected_columns)]
    
    # Rename columns to expected names
    data.columns = expected_columns
    
    # Convert 'date' column to datetime
    data['date'] = pd.to_datetime(data['date'], errors='coerce')
    data = data.sort_values(by='date')
    return data


def create_holidays_dataframe():
    """Create a holidays dataframe for events."""
    holidays = pd.DataFrame({
        'holiday': [
            'Prime Day', 'Black Friday', 'Cyber Monday', 
            'Christmas', 'New Year'
        ],
        'ds': [
            '2024-07-16', '2024-11-24', '2024-12-02',
            '2024-12-25', '2024-01-01'
        ],
        'lower_window': [0, 0, 0, -1, 0],  # Extend for surrounding days if needed
        'upper_window': [1, 0, 0, 0, 0]
    })
    holidays['ds'] = pd.to_datetime(holidays['ds'])
    return holidays


def prepare_time_series(data, asin):
    """Prepare time series data for Prophet."""
    asin_data = data[data['ASIN'] == asin]
    if asin_data.empty:
        raise ValueError(f"No data found for ASIN: {asin}")

    ts_data = asin_data.rename(columns={'date': 'ds', 'units_sold': 'y'})
    ts_data['y'] = ts_data['y'].interpolate().bfill().clip(lower=0)  # Avoid missing or negative sales
    return ts_data


def forecast_demand_with_events(ts_data, holidays, horizon=52):
    """Forecast demand using Prophet with event-based regressors."""
    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        seasonality_mode='multiplicative',
        holidays=holidays,
        changepoint_prior_scale=0.1,  # Increased flexibility for trends
        seasonality_prior_scale=5,   # Control seasonality fluctuation
        holidays_prior_scale=2,      # Reduce over-reliance on holiday peaks
    )
    model.fit(ts_data)

    future = model.make_future_dataframe(periods=horizon, freq='W')
    forecast = model.predict(future)

    # Smooth and clip forecasts
    forecast['yhat'] = forecast['yhat'].clip(lower=0)
    forecast['yhat'] = forecast['yhat'].rolling(window=2, min_periods=1).mean()
    return model, forecast


def plot_forecast(ts_data, forecast):
    """Plot historical sales and forecast."""
    plt.figure(figsize=(14, 8))
    plt.plot(ts_data['ds'], ts_data['y'], label='Historical Sales', marker='o', color='blue')
    plt.plot(forecast['ds'], forecast['yhat'], label='Forecast', linestyle='--', color='green')
    plt.title('Sales Forecast Validation')
    plt.xlabel('Date')
    plt.ylabel('Units Sold')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()


def format_forecast_output(forecast, asin, model, brand):
    """Format the forecast output to match required Excel structure."""
    week_labels = [
        f"Week {i + 1} ({row['ds'].strftime('%d %b')} - {(row['ds'] + pd.Timedelta(days=6)).strftime('%d %b')})"
        for i, row in forecast.iterrows()
    ]
    forecast_values = forecast['yhat'].round().astype(int).tolist()
    output = pd.DataFrame({
        "ASIN": [asin],
        "Model": [model],
        "Brand": [brand],
        **{week: [value] for week, value in zip(week_labels, forecast_values)}
    })
    return output


def main():
    file_path = 'weekly_sales_data.csv'  # Replace with your input file
    asin = 'B0BH7GTY9C'  # Replace with your ASIN

    data = load_data(file_path)
    holidays = create_holidays_dataframe()
    ts_data = prepare_time_series(data, asin)

    forecast_horizon = 52
    model, forecast = forecast_demand_with_events(ts_data, holidays, horizon=forecast_horizon)

    product_title = data[data['ASIN'] == asin]['Model'].iloc[0]
    brand = data[data['ASIN'] == asin]['Brand'].iloc[0]

    output = format_forecast_output(forecast, asin, product_title, brand)
    output.to_excel('sales_forecast.xlsx', index=False)
    print("Forecast saved to 'sales_forecast.xlsx'")

    plot_forecast(ts_data, forecast)

    # Plot using the Prophet model
    fig = model.plot(forecast)
    plt.show()


if __name__ == '__main__':
    main()


