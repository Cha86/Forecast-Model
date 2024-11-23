import pandas as pd
import numpy as np
from prophet import Prophet
import matplotlib.pyplot as plt
import warnings
import os

warnings.filterwarnings("ignore")  # Ignore warnings for cleaner output

def load_data(file_path):
    """Load and preprocess sales data from an Excel file."""
    required_columns = ['date', 'ASIN', 'units_sold']
    optional_columns = ['Model', 'Brand', 'price', 'promotion']
    data = pd.read_excel(file_path)

    missing_required = [col for col in required_columns if col not in data.columns]
    if missing_required:
        raise ValueError(f"Missing required columns: {missing_required}")

    present_optional_columns = [col for col in optional_columns if col in data.columns]
    data = data[required_columns + present_optional_columns]
    data['date'] = pd.to_datetime(data['date'], errors='coerce')
    data = data.sort_values(by='date')
    return data

def load_amazon_forecasts_from_folder(folder_path, asin):
    """Load Amazon forecast data from multiple Excel files in a folder and convert forecasts to integers."""
    forecast_data = {}
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.xlsx'):
            # Extract the forecast type from the file name
            forecast_type = os.path.splitext(file_name)[0]
            forecast_type = forecast_type.replace('_', ' ').title()

            file_path = os.path.join(folder_path, file_name)
            # Read the Excel file
            df = pd.read_excel(file_path)

            # Normalize column names
            df.columns = df.columns.str.strip().str.upper()

            if 'ASIN' not in df.columns:
                print(f"Error: Column 'ASIN' not found in {file_name}")
                continue

            asin_row = df[df['ASIN'].str.upper() == asin.upper()]
            if asin_row.empty:
                print(f"Warning: ASIN {asin} not found in {file_name}")
                continue

            week_columns = [col for col in df.columns if 'WEEK' in col.upper()]
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

def prepare_time_series(data, asin):
    """Prepare time series data for Prophet with additional regressors."""
    asin_data = data[data['ASIN'] == asin]
    if asin_data.empty:
        raise ValueError(f"No data found for ASIN: {asin}")

    ts_data = asin_data.rename(columns={'date': 'ds', 'units_sold': 'y'})
    ts_data['y'] = ts_data['y'].interpolate().bfill().clip(lower=0)
    ts_data['month'] = ts_data['ds'].dt.month
    ts_data['is_high_season'] = ts_data['month'].apply(lambda x: 1 if x in [6, 7, 8] else 0)
    regressors = ['is_high_season']
    return ts_data, regressors

def get_holidays():
    """Create a DataFrame of holidays."""
    holidays_list = [
        ('New Year\'s Day', '2023-01-01'),
        ('Prime Day', '2023-07-16'),
        ('Prime Day', '2023-07-17'),
        ('Thanksgiving', '2023-11-23'),
        ('Black Friday', '2023-11-24'),
        ('Cyber Monday', '2023-11-27'),
        ('Christmas', '2023-12-25'),
        ('New Year\'s Day', '2024-01-01'),
        ('Prime Day', '2024-07-16'),
        ('Prime Day', '2024-07-17'),
        ('Thanksgiving', '2024-11-28'),
        ('Black Friday', '2024-11-29'),
        ('Cyber Monday', '2024-12-02'),
        ('Christmas', '2024-12-25')
    ]
    holidays = pd.DataFrame(holidays_list, columns=['holiday', 'ds'])
    holidays['ds'] = pd.to_datetime(holidays['ds'])
    holidays['lower_window'] = 0
    holidays['upper_window'] = 1
    return holidays

def forecast_demand(ts_data, regressors, horizon=52):
    """Forecast demand using Prophet with adjusted parameters."""
    holidays = get_holidays()

    model = Prophet(
        yearly_seasonality=False,
        weekly_seasonality=True,
        daily_seasonality=False,
        seasonality_mode='multiplicative',
        changepoint_prior_scale=0.1,
        seasonality_prior_scale=10,
        holidays=holidays
    )
    model.add_seasonality(name='monthly', period=30.5, fourier_order=5)
    model.add_seasonality(name='yearly', period=365.25, fourier_order=10)

    for regressor in regressors:
        model.add_regressor(regressor)

    model.fit(ts_data)
    future = model.make_future_dataframe(periods=horizon, freq='W')
    future['month'] = future['ds'].dt.month
    future['is_high_season'] = future['month'].apply(lambda x: 1 if x in [6, 7, 8] else 0)

    forecast = model.predict(future)
    forecast['yhat'] = forecast['yhat'].clip(lower=0)
    forecast['Prophet Forecast'] = forecast['yhat'].round().astype(int)
    return forecast[['ds', 'Prophet Forecast']]

def format_output_with_folder(mean_forecast, forecast_data, horizon=52):
    """Format output for comparison using dynamically loaded forecasts."""
    prophet_forecast_horizon = mean_forecast.iloc[-horizon:].copy()
    prophet_forecast_horizon = prophet_forecast_horizon.reset_index(drop=True)
    prophet_forecast_horizon['Week'] = 'Week ' + prophet_forecast_horizon.index.astype(str)
    comparison = prophet_forecast_horizon[['Week', 'ds', 'Prophet Forecast']]

    for forecast_type, values in forecast_data.items():
        values = values[:horizon]
        forecast_df = pd.DataFrame({
            'Week': ['Week ' + str(i) for i in range(len(values))],
            f'Amazon {forecast_type}': values
        })
        comparison = comparison.merge(forecast_df, on='Week', how='left')

    comparison.fillna(0, inplace=True)
    comparison = comparison.round(1)
    return comparison

def visualize_forecast_with_comparison(ts_data, comparison):
    """Visualize historical data, Prophet forecast, and Amazon forecasts."""
    fig, ax = plt.subplots(figsize=(16, 12))

    ax.plot(ts_data['ds'], ts_data['y'], label='Historical Sales', marker='o', color='black')

    ax.plot(
        comparison['ds'],
        comparison['Prophet Forecast'],
        label='Prophet Forecast',
        linestyle='--',
        color='blue',
        marker='o'
    )

    for col in comparison.columns:
        if col.startswith('Amazon '):
            ax.plot(
                comparison['ds'],
                comparison[col],
                label=col,
                linestyle=':',
                marker='x'
            )

    ax.set_title('Sales Forecast Comparison')
    ax.set_xlabel('Date')
    ax.set_ylabel('Units Sold')
    ax.legend()
    ax.grid()

    plt.tight_layout()
    plt.show()

def main():
    folder_path = 'forecasts_folder'
    file_path = 'weekly_sales_data.xlsx'
    asin = 'B091HTG6DQ'

    data = load_data(file_path)
    forecast_data = load_amazon_forecasts_from_folder(folder_path, asin)
    ts_data, regressors = prepare_time_series(data, asin)
    mean_forecast = forecast_demand(ts_data, regressors, horizon=52)

    comparison = format_output_with_folder(mean_forecast, forecast_data, horizon=52)

    print("Comparison DataFrame:")
    print(comparison)

    output_file_path = os.path.abspath('forecast_comparison_summary.xlsx')
    comparison.to_excel(output_file_path, index=False)
    print(f"Comparison and summary saved to '{output_file_path}'")

    visualize_forecast_with_comparison(ts_data, comparison)

if __name__ == '__main__':
    main()


