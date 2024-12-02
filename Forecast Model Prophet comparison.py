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


def forecast_with_custom_params(ts_data, forecast_data, changepoint_prior_scale, seasonality_prior_scale, holidays_prior_scale, horizon=20):
    """Forecast demand using Prophet with custom parameters."""
    print(f"Testing: changepoint_prior_scale={changepoint_prior_scale}, "
          f"seasonality_prior_scale={seasonality_prior_scale}, holidays_prior_scale={holidays_prior_scale}")

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

    train_df = combined_df[~combined_df['y'].isna()].copy()
    future_df = combined_df[combined_df['y'].isna()].drop(columns='y').copy()

    model = Prophet(
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

    try:
        model.fit(train_df)
        forecast = model.predict(future_df)
        forecast['Prophet Forecast'] = forecast['yhat']
        return forecast[['ds', 'Prophet Forecast']]
    except Exception as e:
        print(f"Error during forecasting: {e}")
        return pd.DataFrame(columns=['ds', 'Prophet Forecast'])


def optimize_prophet_params(ts_data, forecast_data, param_grid, horizon=20):
    """Optimize Prophet parameters to minimize RMSE against Amazon forecasts."""
    print("Starting optimization...")
    best_rmse = float('inf')
    best_params = None

    for changepoint_prior_scale in param_grid['changepoint_prior_scale']:
        for seasonality_prior_scale in param_grid['seasonality_prior_scale']:
            for holidays_prior_scale in param_grid['holidays_prior_scale']:
                try:
                    forecast = forecast_with_custom_params(
                        ts_data, forecast_data,
                        changepoint_prior_scale,
                        seasonality_prior_scale,
                        holidays_prior_scale,
                        horizon
                    )
                    if forecast.empty or 'Prophet Forecast' not in forecast.columns:
                        raise ValueError("Forecast failed to generate 'Prophet Forecast'.")

                    rmse_values = calculate_rmse(forecast, forecast_data, horizon)
                    avg_rmse = np.mean(list(rmse_values.values()))

                    if avg_rmse < best_rmse:
                        best_rmse = avg_rmse
                        best_params = {
                            'changepoint_prior_scale': changepoint_prior_scale,
                            'seasonality_prior_scale': seasonality_prior_scale,
                            'holidays_prior_scale': holidays_prior_scale
                        }
                except Exception as e:
                    print(f"Error during optimization: {e}")
                    continue

    if best_params is None:
        print("Optimization failed. No valid parameters found.")
        return {'changepoint_prior_scale': 0.1, 'seasonality_prior_scale': 1, 'holidays_prior_scale': 10}

    print(f"Optimization complete. Best Parameters Found: {best_params}")
    return best_params


def calculate_rmse(forecast, forecast_data, horizon):
    """Calculate RMSE between Prophet forecast and Amazon forecasts."""
    comparison = forecast.copy()
    rmse_values = {}
    for forecast_type, values in forecast_data.items():
        values = values[:horizon]
        rmse_values[forecast_type] = np.sqrt(((comparison['Prophet Forecast'] - values) ** 2).mean())
    return rmse_values


def format_output_with_forecasts(prophet_forecast, forecast_data, horizon=20):
    """Format output for comparison using Prophet and Amazon forecasts."""
    comparison = prophet_forecast.copy()

    for forecast_type, values in forecast_data.items():
        values = values[:horizon]
        forecast_df = pd.DataFrame({
            'ds': prophet_forecast['ds'],  # Match the dates
            f'Amazon {forecast_type}': values
        })
        comparison = comparison.merge(forecast_df, on='ds', how='left')

    comparison.fillna(0, inplace=True)
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

    for col in comparison.columns[2:]:
        ax.plot(comparison['ds'], comparison[col], label=col, linestyle=':', marker='x')

    ax.set_title('Sales Forecast Comparison')
    ax.set_xlabel('Date')
    ax.set_ylabel('Units Sold')
    ax.legend()
    ax.grid()
    plt.tight_layout()
    plt.show()


def adjust_forecast_weights(forecast, yhat_weight, yhat_upper_weight):
    """
    Adjust forecast weights dynamically for yhat and yhat_upper.
    """
    if 'yhat' not in forecast or 'yhat_upper' not in forecast:
        raise KeyError("'yhat' or 'yhat_upper' not found in forecast DataFrame.")

    forecast['Prophet Forecast'] = (
        yhat_weight * forecast['yhat'] + yhat_upper_weight * forecast['yhat_upper']
    ).clip(lower=0).round().astype(int)
    return forecast



def find_best_forecast_weights(forecast, comparison, weights):
    """
    Find the best weight combination for yhat and yhat_upper by comparing to Amazon's forecasts.
    """
    best_rmse = float('inf')
    best_weights = None
    rmse_results = {}

    for yhat_weight, yhat_upper_weight in weights:
        # Adjust the forecast with current weights
        adjusted_forecast = adjust_forecast_weights(forecast.copy(), yhat_weight, yhat_upper_weight)

        # Compute RMSE for each Amazon forecast
        rmse_values = {}
        for amazon_col in comparison.columns[3:]:  # Amazon forecasts start from column index 3
            rmse = np.sqrt(((comparison[amazon_col] - adjusted_forecast['Prophet Forecast']) ** 2).mean())
            rmse_values[amazon_col] = rmse
        
        avg_rmse = np.mean(list(rmse_values.values()))
        rmse_results[(yhat_weight, yhat_upper_weight)] = avg_rmse

        if avg_rmse < best_rmse:
            best_rmse = avg_rmse
            best_weights = (yhat_weight, yhat_upper_weight)
    
    return best_weights, rmse_results


def main():
    folder_path = 'forecasts_folder2'
    file_path = 'weekly_sales_data.xlsx'
    asin = 'B08KGVH7YC'

    # Load and prepare data
    data = load_data(file_path)
    forecast_data = load_amazon_forecasts_from_folder(folder_path, asin)
    ts_data = prepare_time_series_with_lags(data, asin, lag_weeks=1)

    # Define the parameter grid for optimization
    param_grid = {
        'changepoint_prior_scale': [0.1, 0.2, 0.3],
        'seasonality_prior_scale': [1, 2, 3],
        'holidays_prior_scale': [10, 15, 20]
    }

    # Perform optimization
    best_params = optimize_prophet_params(ts_data, forecast_data, param_grid, horizon=20)

    if not best_params:
        print("No valid parameters found. Exiting.")
        return

    # Generate forecast using the best parameters
    forecast = forecast_with_custom_params(
        ts_data, forecast_data,
        best_params['changepoint_prior_scale'],
        best_params['seasonality_prior_scale'],
        best_params['holidays_prior_scale'],
        horizon=20
    )

    if forecast.empty or 'Prophet Forecast' not in forecast.columns:
        print("Failed to generate forecast. Exiting.")
        return

    # Format the comparison dataframe
    comparison = format_output_with_forecasts(forecast, forecast_data, horizon=20)

    # Save and visualize the results
    output_file_path = os.path.abspath('forecast_comparison_summary.xlsx')
    comparison.to_excel(output_file_path, index=False)
    print(f"Comparison and summary saved to '{output_file_path}'")

    visualize_forecast_with_comparison(ts_data, comparison)


if __name__ == '__main__':
    main()

