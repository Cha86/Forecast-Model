# data_loader.py

import pandas as pd
import os

def generate_date_from_week(row):
    """
    Convert year-week format into a datetime object representing the end of the week (Sunday).
    Assumes weeks start on Monday and end on Sunday.
    """
    week_str = row['week']
    year = row['year']
    week_number = int(week_str[1:])  # Assuming format 'W01', 'W02', etc.
    return pd.to_datetime(f'{year}-W{week_number}-7', format='%G-W%V-%u')  # %G: ISO year, %V: ISO week, %u: weekday (7 for Sunday)

def clean_weekly_sales_data(data):
    """
    Placeholder for additional cleaning steps if needed.
    """
    # Example: Remove duplicates
    data = data.drop_duplicates(subset=['product title', 'week', 'year', 'asin'])
    return data

def load_weekly_sales_data(file_path):
    """
    Load weekly sales data from an Excel file, ensuring required columns are present.
    """
    data = pd.read_excel(file_path)
    data.columns = data.columns.str.strip().str.lower()

    required_columns = ['product title', 'week', 'year', 'units_sold', 'asin']
    missing_required = [col for col in required_columns if col not in data.columns]
    if missing_required:
        raise ValueError(f"Missing required columns: {missing_required}")

    data['ds'] = data.apply(generate_date_from_week, axis=1)
    data = data.rename(columns={'units_sold': 'y'})
    data['y'] = data['y'].astype(int)
    data = clean_weekly_sales_data(data)
    return data

def load_asins_to_forecast(file_path):
    """
    Load a list of ASINs from either a text or Excel file.
    """
    if file_path.endswith('.txt'):
        with open(file_path, 'r') as file:
            asins = [line.strip() for line in file if line.strip()]
    elif file_path.endswith('.xlsx') or file_path.endswith('.xls'):
        df = pd.read_excel(file_path)
        asins = df.iloc[:, 0].dropna().astype(str).tolist()
    else:
        raise ValueError("Unsupported file format for ASINs to Forecast file.")
    return asins

def load_amazon_forecasts_from_folder(folder_path, asin):
    """
    Load Amazon forecast data from multiple Excel files in a folder.
    Each file should correspond to a specific forecast type (Mean, P70, etc.).
    """
    forecast_data = {}
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.xlsx') or file_name.endswith('.xls'):
            # Extract forecast type from filename, e.g., 'Mean Forecast.xlsx' -> 'Mean'
            forecast_type = os.path.splitext(file_name)[0].replace('_', ' ').title()
            if forecast_type.endswith(' Forecast'):
                forecast_type = forecast_type.replace(' Forecast', '')
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

            week_columns = [col for col in df.columns if 'W' in col.upper()]
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
