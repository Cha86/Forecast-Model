import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt


def load_data(file_path):
    """Load and preprocess sales data from an Excel file."""
    expected_columns = ['date', 'ASIN', 'Model', 'Brand', 'units_sold']
    data = pd.read_excel(file_path)

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
        if 'ASIN' in df.columns:
            # Transpose and format horizontally oriented data
            df_cleaned = df.set_index(['ASIN', 'Model', 'Brand']).T.reset_index()
            df_cleaned.rename(columns={'index': 'Week'}, inplace=True)
            df_cleaned = df_cleaned.reset_index(drop=True)  # Ensure proper index for slicing
            cleaned_forecasts[sheet_name] = df_cleaned
        else:
            print(f"Unexpected format in sheet: {sheet_name}")
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


def format_output_for_display(prophet_forecast, amazon_forecasts, horizon=16):
    """Format output for comparison with clean week labels and forecast values."""
    # Prepare Prophet forecast
    prophet_forecast_horizon = prophet_forecast.iloc[-horizon:].copy()
    prophet_forecast_horizon['yhat'] = prophet_forecast_horizon['yhat'].round().astype(int)
    prophet_forecast_horizon = prophet_forecast_horizon[['ds', 'yhat']].rename(columns={'yhat': 'Prophet Forecast'})

    # Add week labels
    prophet_forecast_horizon['Week'] = [f"Week {i}" for i in range(len(prophet_forecast_horizon))]

    # Prepare Amazon forecasts
    comparison = prophet_forecast_horizon[['Week', 'ds', 'Prophet Forecast']].copy()
    for sheet_name, df in amazon_forecasts.items():
        amazon_values = df.iloc[:horizon, 1:].mean(axis=1).round().astype(int).values
        comparison[sheet_name] = amazon_values

    return comparison


def summarize_historical_data(ts_data):
    """Summarize historical sales data."""
    summary_stats = ts_data['y'].describe().to_dict()
    return summary_stats


def visualize_forecast_with_comparison(ts_data, comparison, summary_stats, total_forecast_16, total_forecast_8):
    """Visualize historical data, Prophet forecast, and Amazon forecasts with summary."""
    fig, ax = plt.subplots(figsize=(16, 12))

    # Plot historical data
    ax.plot(ts_data['ds'], ts_data['y'], label='Historical Sales', marker='o', color='black')

    # Plot Prophet forecast
    ax.plot(
        comparison['ds'], 
        comparison['Prophet Forecast'], 
        label='Prophet Forecast', 
        marker='o', 
        linestyle='--', 
        color='blue'
    )

    # Plot Amazon forecasts for all datasets
    for column in comparison.columns[3:]:
        ax.plot(
            comparison['ds'], 
            comparison[column], 
            label=f'{column}', 
            linestyle='--'
        )

    ax.set_title('Sales Forecast Comparison')
    ax.set_xlabel('Date')
    ax.set_ylabel('Units Sold')
    ax.legend()
    ax.grid()

    # Add summary on the upper left
    summary_text = (
        f"Historical Summary:\n"
        f"Min: {summary_stats['min']:.0f}\n"
        f"Max: {summary_stats['max']:.0f}\n"
        f"Mean: {summary_stats['mean']:.0f}\n\n"
        f"Total Forecast (16 Weeks): {total_forecast_16:.0f}\n"
        f"Total Forecast (8 Weeks): {total_forecast_8:.0f}"
    )
    plt.gcf().text(0.1, 0.7, summary_text, fontsize=10, bbox=dict(facecolor='white', alpha=0.8))

    plt.tight_layout()
    plt.show()


def main():
    file_path = 'weekly_sales_data.xlsx'
    amazon_forecast_path = 'amazon_forecasts.xlsx'
    asin = 'B091HTG6DQ'

    # Load datasets
    data = load_data(file_path)
    amazon_forecasts = load_amazon_forecasts(amazon_forecast_path)
    
    # Prepare data and forecast
    ts_data = prepare_time_series(data, asin)
    model, forecast = forecast_demand(ts_data, horizon=16)

    # Summarize forecast and historical data
    summary_stats = summarize_historical_data(ts_data)
    comparison = format_output_for_display(forecast, amazon_forecasts, horizon=16)

    # Calculate forecast totals
    total_forecast_16 = comparison['Prophet Forecast'].iloc[:16].sum()
    total_forecast_8 = comparison['Prophet Forecast'].iloc[:8].sum()

    # Save results
    comparison.to_excel('forecast_comparison_summary.xlsx', index=False)
    print("Comparison and summary saved to 'forecast_comparison_summary.xlsx'")

    # Visualize results
    visualize_forecast_with_comparison(ts_data, comparison, summary_stats, total_forecast_16, total_forecast_8)


if __name__ == '__main__':
    main()






