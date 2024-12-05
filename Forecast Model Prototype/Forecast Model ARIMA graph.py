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
    ts_data['unique_id'] = asin  # Add a unique identifier column
    ts_data = ts_data.set_index('ds').asfreq('W-SUN')
    ts_data['y'] = ts_data['y'].interpolate()
    ts_data['y'] = ts_data['y'].fillna(method='bfill')
    print(f"Time Series Data:\n{ts_data.head()}")
    return ts_data.reset_index()


# Forecast Using AutoARIMA
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
    return forecast_values



def format_output(asin, product_title, brand, forecast_values):
    output = pd.DataFrame({
        'ASIN': [asin],
        'Product Title': [product_title],
        'Brand': [brand],
        'Week 0': [forecast_values[0]],
        'Week 1': [forecast_values[1]],
        'Week 2': [forecast_values[2]],
        'Week 3': [forecast_values[3]],
        'Week 4': [forecast_values[4]],
        'Week 5': [forecast_values[5]],
        'Week 6': [forecast_values[6]],
        'Week 7': [forecast_values[7]],
    })
    return output



def display_output(output):
    metadata = {
        'Program': 'Retail',
        'Distributor View': 'Manufacturing',
        'View By': 'ASIN',
        'Countries': 'US',
        'Businesses': 'Gigabyte',
        'Locale': 'en_US',
        'Search for ASINs': output['ASIN'][0],
        'Forecasting Statistic': 'Sales Forecast',
        'Report Updated': pd.Timestamp.now().strftime('%m/%d/%y')
    }

    for key, value in metadata.items():
        print(f"{key}=[{value}]")
    print(output)



def plot_forecast(ts_data, forecast_values):
    plt.figure(figsize=(14, 8))
    
    # Plot actual sales
    plt.plot(ts_data['ds'], ts_data['y'], label='Actual Sales', marker='o', linestyle='-', color='blue')
    
    # Annotate actual sales
    for i, (x, y) in enumerate(zip(ts_data['ds'], ts_data['y'])):
        plt.text(x, y + 20, f'{int(y)}', ha='center', fontsize=9, color='darkblue')
    
    # Forecast weeks
    future_weeks = pd.date_range(ts_data['ds'].iloc[-1] + pd.Timedelta(days=7), periods=len(forecast_values), freq='W')
    
    # Plot forecast
    plt.plot(future_weeks, forecast_values, label='Sales Forecast', marker='o', linestyle='-', color='green')
    
    # Annotate forecasted sales
    for i, (x, y) in enumerate(zip(future_weeks, forecast_values)):
        plt.text(x, y + 20, f'{int(y)}', ha='center', fontsize=9, color='green')
    
    # Extend x-axis to include future weeks
    plt.xlim(ts_data['ds'].iloc[0], future_weeks[-1] + pd.Timedelta(days=7))
    
    # titles and labels
    plt.title('Sales Forecast with Weekly Sales Data', fontsize=16)
    plt.xlabel('Week', fontsize=12)
    plt.ylabel('Units Sold', fontsize=12)
    plt.xticks(rotation=45)
    plt.legend()
    plt.grid()
    plt.tight_layout() 
    plt.show()
    plt.show()


def main():
    file_path = 'weekly_sales_data.csv'
    weekly_sales = load_data(file_path)
    asin = 'B08KGVH7YC'
    ts_data = prepare_time_series(weekly_sales, asin)
    forecast_values = forecast_demand(ts_data, horizon=8)
    product_title = weekly_sales[weekly_sales['ASIN'] == asin]['Product Title'].iloc[0]
    brand = weekly_sales[weekly_sales['ASIN'] == asin]['Brand'].iloc[0]
    output = format_output(asin, product_title, brand, forecast_values)
    display_output(output)
    plot_forecast(ts_data, forecast_values)


if __name__ == '__main__':
    main()