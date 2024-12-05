import pandas as pd
import numpy as np
from statsforecast import StatsForecast
from statsforecast.models import AutoARIMA
import matplotlib.pyplot as plt


def load_data(file_path):
    data = pd.read_csv(file_path)
    data['date'] = pd.to_datetime(data['date'])
    data = data.sort_values(by='date')
    print(f"Loaded Data:\n{data.head()}")
    return data


def prepare_time_series(data, asin):
    asin_data = data[data['ASIN'] == asin]
    ts_data = asin_data[['date', 'units_sold']].rename(columns={'date': 'ds', 'units_sold': 'y'})
    ts_data['unique_id'] = asin  
    ts_data = ts_data.set_index('ds').asfreq('W-SUN')
    ts_data['y'] = ts_data['y'].interpolate()
    ts_data['y'] = ts_data['y'].fillna(method='bfill')
    print(f"Time Series Data:\n{ts_data.head()}")
    return ts_data.reset_index()


def forecast_demand(ts_data, horizon=8):
    ts_data = ts_data.rename(columns={'ds': 'ds', 'y': 'y', 'unique_id': 'unique_id'})
    models = [AutoARIMA()]
    sf = StatsForecast(models=models, freq='W')
    try:
        forecast = sf.forecast(df=ts_data, h=horizon)
        forecast_values = forecast['AutoARIMA'].values
    except Exception as e:
        print(f"Error during forecasting: {e}")
        raise
    return np.round(forecast_values).astype(int) 


def format_output(asin, product_title, brand, forecast_values, start_date):
    # Calculate week ranges
    future_weeks = pd.date_range(start=start_date + pd.Timedelta(days=7), periods=len(forecast_values), freq='W')
    week_labels = [f"Week {i} ({week.strftime('%d %b')} - {(week + pd.Timedelta(days=6)).strftime('%d %b')})"
                   for i, week in enumerate(future_weeks)]

    output = pd.DataFrame({
        "ASIN": [asin],
        "Model": [product_title],
        "Brand": [brand],
        **{label: [forecast_values[i]] for i, label in enumerate(week_labels)}
    })
    return output


def save_output_to_excel(output, file_name='sales_forecast.xlsx'):
    output.to_excel(file_name, index=False)
    print(f"Forecast saved to {file_name}")

def plot_input_data(ts_data):
    plt.figure(figsize=(12, 6))
    plt.plot(ts_data['ds'], ts_data['y'], label='Historical Sales', color='blue', marker='o')
    plt.title('Historical Sales Data', fontsize=16)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Units Sold', fontsize=12)
    plt.xticks(rotation=45)
    plt.legend()
    plt.grid()
    plt.show()
    


def main():

    file_path = 'weekly_sales_data.csv'
    weekly_sales = load_data(file_path)

    asin = 'B08KGVH7YC'
    ts_data = prepare_time_series(weekly_sales, asin)

    forecast_horizon = 8  
    forecast_values = forecast_demand(ts_data, horizon=forecast_horizon)

    product_title = weekly_sales[weekly_sales['ASIN'] == asin]['Product Title'].iloc[0]
    brand = weekly_sales[weekly_sales['ASIN'] == asin]['Brand'].iloc[0]
    start_date = ts_data['ds'].iloc[-1]  # Start from the last known date

    output = format_output(asin, product_title, brand, forecast_values, start_date)

    save_output_to_excel(output)

    print(output)



if __name__ == '__main__':
    main()
