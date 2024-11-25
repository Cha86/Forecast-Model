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

def prepare_time_series(data, asin):
    """Prepare time series data for Prophet."""
    ts_data = data[data['ASIN'] == asin].rename(columns={'date': 'ds', 'units_sold': 'y'})
    ts_data['y'] = ts_data['y'].interpolate().bfill().clip(lower=0)
    return ts_data

def forecast_demand_with_amazon(ts_data, forecast_data, horizon=52):
    """Forecast demand using Prophet with Amazon forecasts as regressors."""
    # Create the future dataframe
    future = pd.date_range(start=ts_data['ds'].max() + pd.Timedelta(days=7), periods=horizon, freq='W')
    future = pd.DataFrame({'ds': future})

    # Combine historical and future data
    combined_df = pd.concat([ts_data[['ds', 'y']], future], ignore_index=True)

    # Add Amazon forecasts as regressors
    for forecast_type, values in forecast_data.items():
        # Ensure the values array matches the combined_df length
        full_values = np.concatenate([np.full(len(ts_data), np.nan), values])
        if len(full_values) < len(combined_df):
            full_values = np.append(full_values, [np.nan] * (len(combined_df) - len(full_values)))
        elif len(full_values) > len(combined_df):
            full_values = full_values[:len(combined_df)]

        combined_df[f'Amazon_{forecast_type}'] = full_values

    # Fill missing regressor values in historical data
    regressor_cols = [col for col in combined_df.columns if col.startswith('Amazon_')]
    combined_df[regressor_cols] = combined_df[regressor_cols].fillna(method='ffill').fillna(0)

    # Split back into train and future data
    train_df = combined_df[~combined_df['y'].isna()].copy()
    future_df = combined_df[combined_df['y'].isna()].drop(columns='y').copy()

    # Initialize Prophet model
    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        seasonality_mode='multiplicative',
        changepoint_prior_scale=0.2,
        seasonality_prior_scale=10
    )

    # Add regressors to the model
    for regressor in regressor_cols:
        model.add_regressor(regressor)

    # Fit the model
    model.fit(train_df)

    # Predict future
    forecast = model.predict(future_df)

    # Prepare the forecast dataframe
    forecast['Prophet Forecast'] = forecast['yhat'].clip(lower=0).round().astype(int)
    mean_forecast = forecast[['ds', 'Prophet Forecast']]

    return mean_forecast


def format_output_with_folder(mean_forecast, forecast_data, horizon=52):
    """Format output for comparison using dynamically loaded forecasts."""
    prophet_forecast_horizon = mean_forecast.iloc[:horizon].copy()
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
    ts_data = prepare_time_series(data, asin)
    mean_forecast = forecast_demand_with_amazon(ts_data, forecast_data, horizon=52)

    comparison = format_output_with_folder(mean_forecast, forecast_data, horizon=52)
    print("Comparison DataFrame:")
    print(comparison)

    output_file_path = os.path.abspath('forecast_comparison_summary.xlsx')
    comparison.to_excel(output_file_path, index=False)
    print(f"Comparison and summary saved to '{output_file_path}'")

    visualize_forecast_with_comparison(ts_data, comparison)

if __name__ == '__main__':
    main()




