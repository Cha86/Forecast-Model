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
    data = pd.read_excel(file_path)

    missing_required = [col for col in required_columns if col not in data.columns]
    if missing_required:
        raise ValueError(f"Missing required columns: {missing_required}")

    data = data[required_columns]
    data['date'] = pd.to_datetime(data['date'], errors='coerce')
    data = data.sort_values(by='date')
    return data


def load_amazon_forecasts_from_folder(folder_path, asin):
    """Load Amazon forecast data from multiple Excel files in a folder."""
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


def add_lag_features(ts_data, lag_weeks=1):
    """Add lag-based regressor features."""
    ts_data = ts_data.copy()
    ts_data[f'lag_{lag_weeks}_week'] = ts_data['y'].shift(lag_weeks)
    ts_data[f'lag_{lag_weeks}_week'].fillna(0, inplace=True)  # Replace NaN with 0
    return ts_data


def prepare_time_series_with_lags(data, asin, lag_weeks=1):
    """Prepare time series data for Prophet with lag features."""
    ts_data = data[data['ASIN'] == asin].rename(columns={'date': 'ds', 'units_sold': 'y'})
    ts_data['y'] = ts_data['y'].interpolate().bfill().clip(lower=0)
    ts_data = add_lag_features(ts_data, lag_weeks)
    return ts_data


def get_shifted_holidays():
    """Create a DataFrame of holidays shifted two weeks earlier."""
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


def forecast_demand_with_custom_increase(ts_data, forecast_data, horizon=20):
    """Forecast demand using Prophet and align with P70."""
    future_dates = pd.date_range(start=ts_data['ds'].max() + pd.Timedelta(days=7), periods=horizon, freq='W')
    future = pd.DataFrame({'ds': future_dates})

    combined_df = pd.concat([ts_data, future], ignore_index=True)

    for forecast_type, values in forecast_data.items():
        values_to_use = values[:horizon] if len(values) > horizon else values
        extended_values = np.concatenate(
            [
                np.full(len(ts_data), np.nan),
                values_to_use,
                np.full(max(horizon - len(values_to_use), 0), values_to_use[-1] if len(values_to_use) > 0 else 0)
            ]
        )
        extended_values = extended_values[:len(combined_df)]
        combined_df[f'Amazon_{forecast_type}'] = extended_values

    regressor_cols = [col for col in combined_df.columns if col.startswith('Amazon_')]
    combined_df[regressor_cols] = combined_df[regressor_cols].fillna(0)

    holidays = get_shifted_holidays()
    combined_df['prime_day'] = combined_df['ds'].apply(
        lambda x: 0.2 if x in holidays[holidays['holiday'] == 'Prime Day']['ds'].values else 0
    )

    last_lag_value = ts_data['y'].iloc[-1] if not ts_data.empty else 0
    combined_df['lag_1_week'] = combined_df['lag_1_week'].fillna(last_lag_value)

    train_df = combined_df[~combined_df['y'].isna()].copy()
    future_df = combined_df[combined_df['y'].isna()].drop(columns='y').copy()

    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        seasonality_mode='multiplicative',
        holidays=holidays,
        changepoint_prior_scale=0.17,
        seasonality_prior_scale=4.5,
        holidays_prior_scale=15
    )

    for regressor in regressor_cols + ['prime_day', 'lag_1_week']:
        model.add_regressor(regressor, mode='multiplicative')

    model.fit(train_df)

    forecast = model.predict(future_df)

    # Adjust forecast to P70
    forecast['Prophet Forecast'] = (
        0.9 * forecast['yhat_upper'] + 0.1 * forecast['yhat']
    ).clip(lower=0).round().astype(int)

    mean_forecast = forecast[['ds', 'Prophet Forecast']][:horizon]

    return mean_forecast


def format_output_with_folder(mean_forecast, forecast_data, horizon=20):
    """Format output for comparison using dynamically loaded forecasts."""
    prophet_forecast_horizon = mean_forecast.iloc[:horizon].copy()

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
    comparison.iloc[:, 3:] = comparison.iloc[:, 3:].astype(int)
    return comparison.iloc[:horizon]




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

    for col in comparison.columns[3:]:
        ax.plot(comparison['ds'], comparison[col], label=col, linestyle=':', marker='x')

    ax.set_title('Sales Forecast Comparison')
    ax.set_xlabel('Date')
    ax.set_ylabel('Units Sold')
    ax.legend()
    ax.grid()
    plt.tight_layout()
    plt.show()


def main():
    folder_path = 'forecasts_folder2'
    file_path = 'weekly_sales_data.xlsx'
    asin = 'B08KGVH7YC'

    data = load_data(file_path)
    forecast_data = load_amazon_forecasts_from_folder(folder_path, asin)
    ts_data = prepare_time_series_with_lags(data, asin, lag_weeks=1)
    mean_forecast = forecast_demand_with_custom_increase(ts_data, forecast_data, horizon=20)

    comparison = format_output_with_folder(mean_forecast, forecast_data, horizon=20)
    print("Comparison DataFrame:")
    print(comparison)

    output_file_path = os.path.abspath('forecast_comparison_summary.xlsx')
    comparison.to_excel(output_file_path, index=False)
    print(f"Comparison and summary saved to '{output_file_path}'")

    visualize_forecast_with_comparison(ts_data, comparison)


if __name__ == '__main__':
    main()
