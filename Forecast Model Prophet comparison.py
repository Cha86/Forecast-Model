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
    """Load Amazon forecast data from multiple Excel files in a folder."""
    forecast_data = {}
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.xlsx'):
            forecast_type = os.path.splitext(file_name)[0]  # File name without extension
            forecast_type = forecast_type.replace('_', ' ').title()

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

            week_columns = [col for col in df.columns if 'WEEK' in col.upper()]
            if not week_columns:
                print(f"Warning: No week columns found in {file_name}")
                continue

            forecast_values = asin_row.iloc[0][week_columns].astype(str).str.replace(',', '').astype(float).values
            forecast_data[forecast_type] = forecast_values

    return forecast_data


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


def prepare_time_series(data, asin):
    """Prepare time series data for Prophet."""
    asin_data = data[data['ASIN'] == asin]
    if asin_data.empty:
        raise ValueError(f"No data found for ASIN: {asin}")

    ts_data = asin_data.rename(columns={'date': 'ds', 'units_sold': 'y'})
    ts_data['y'] = ts_data['y'].interpolate().bfill().clip(lower=0)
    return ts_data


def forecast_demand(ts_data, horizon=16):
    """Forecast demand using Prophet with holiday effects."""
    holidays = get_holidays()

    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        seasonality_mode='multiplicative',
        changepoint_prior_scale=0.2,
        seasonality_prior_scale=10,
        holidays=holidays
    )

    model.add_seasonality(name='monthly', period=30.5, fourier_order=5)
    model.add_seasonality(name='yearly', period=365.25, fourier_order=10)

    model.fit(ts_data)

    future = model.make_future_dataframe(periods=horizon, freq='W')

    forecast = model.predict(future)
    forecast['yhat'] = forecast['yhat'].clip(lower=0)
    forecast['Prophet Forecast'] = forecast['yhat'].round().astype(int)
    mean_forecast = forecast[['ds', 'Prophet Forecast']]

    return mean_forecast


def summarize_historical_data(ts_data):
    """Summarize historical sales data."""
    summary_stats = ts_data['y'].describe().to_dict()
    return summary_stats


def format_output_for_display(mean_forecast, forecast_data, horizon=16):
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


def visualize_forecast_with_comparison(ts_data, comparison, summary_stats, total_forecast_16, total_forecast_8):
    """Visualize historical data, Prophet forecast, and Amazon forecasts with summary."""
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

    summary_text = (
        f"Historical Summary:\n"
        f"Min: {summary_stats['min']:.0f}\n"
        f"Max: {summary_stats['max']:.0f}\n"
        f"Mean: {summary_stats['mean']:.0f}\n\n"
        f"Total Forecast (16 Weeks): {total_forecast_16:.0f}\n"
        f"Total Forecast (8 Weeks): {total_forecast_8:.0f}"
    )
    plt.gcf().text(0.02, 0.95, summary_text, fontsize=10, verticalalignment='top',
                   bbox=dict(facecolor='white', alpha=0.8))

    plt.tight_layout()
    plt.show()


def main():
    folder_path = 'forecasts_folder'
    file_path = 'weekly_sales_data.xlsx'
    asin = 'B091HTG6DQ'

    data = load_data(file_path)
    forecast_data = load_amazon_forecasts_from_folder(folder_path, asin)
    ts_data = prepare_time_series(data, asin)
    mean_forecast = forecast_demand(ts_data, horizon=16)

    summary_stats = summarize_historical_data(ts_data)

    comparison = format_output_for_display(mean_forecast, forecast_data, horizon=16)

    total_forecast_16 = comparison['Prophet Forecast'].sum()
    total_forecast_8 = comparison['Prophet Forecast'].iloc[:8].sum()

    print("Comparison DataFrame:")
    print(comparison.head(16))

    output_file_path = os.path.abspath('forecast_comparison_summary.xlsx')
    comparison.to_excel(output_file_path, index=False)
    print(f"Comparison and summary saved to '{output_file_path}'")

    visualize_forecast_with_comparison(ts_data, comparison, summary_stats, total_forecast_16, total_forecast_8)


if __name__ == '__main__':
    main()



