# analysis.py

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

def analyze_po_data_by_asin(po_file="po database.xlsx", product_filter=None, output_folder="po_analysis_by_asin"):
    """
    Analyze PO orders by each ASIN to understand buying habits based on requested quantities.
    Generates weekly and monthly requested quantity trend graphs and creates an Excel file 
    for each ASIN with separate sheets for weekly and monthly data.

    Parameters:
      po_file: Path to the PO Excel file.
      product_filter: Optional ASIN or product filter to limit analysis.
      output_folder: Folder to save individual ASIN reports and graphs.

    Returns:
      A dictionary with aggregated data and paths to generated graphs for each ASIN.
    """
    # Load PO data
    po_df = pd.read_excel(po_file)

    # Standardize column names (strip spaces)
    po_df.columns = po_df.columns.str.strip()

    # Convert date columns to datetime
    for date_col in ['Order date', 'Expected date']:
        if date_col in po_df.columns:
            po_df[date_col] = pd.to_datetime(po_df[date_col], errors='coerce')

    # Filter by product if specified
    if product_filter:
        po_df = po_df[po_df['ASIN'] == product_filter]

    # Ensure numeric conversion for Requested quantity
    po_df['Requested quantity'] = pd.to_numeric(po_df['Requested quantity'], errors='coerce').fillna(0)

    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    results = {}
    unique_asins = po_df['ASIN'].dropna().unique()
    for asin in unique_asins:
        # Filter data for the current ASIN
        asin_df = po_df[po_df['ASIN'] == asin].copy()
        if asin_df.empty:
            continue

        asin_df['Order_Week_Period'] = asin_df['Order date'].dt.to_period('W-SUN')
        asin_df['Order Week'] = asin_df['Order_Week_Period'].apply(lambda p: p.end_time)

        weekly_agg = (
            asin_df
            .groupby('Order Week', as_index=False)
            .agg({'Requested quantity': 'sum'})
        )

        asin_df['Order_Month_Period'] = asin_df['Order date'].dt.to_period('M')
        asin_df['Order Month'] = asin_df['Order_Month_Period'].apply(lambda p: p.end_time)
        monthly_trend = (
            asin_df
            .groupby('Order Month', as_index=False)
            .agg({'Requested quantity': 'sum'})
        )

        # Plot weekly trend for this ASIN
        plt.figure(figsize=(10, 6))
        plt.plot(weekly_agg['Order Week'], weekly_agg['Requested quantity'], marker='o')
        plt.title(f'Weekly Requested Quantities for ASIN {asin}')
        plt.xlabel('Week')
        plt.ylabel('Total Requested Quantity')
        plt.grid()
        plt.xticks(rotation=45)
        plt.tight_layout()
        weekly_chart = os.path.join(output_folder, f"{asin}_weekly_requested_quantity.png")
        plt.savefig(weekly_chart)
        plt.close()

        # Plot monthly trend for this ASIN
        plt.figure(figsize=(10, 6))
        plt.plot(monthly_trend['Order Month'], monthly_trend['Requested quantity'], marker='o', color='green')
        plt.title(f'Monthly Requested Quantities for ASIN {asin}')
        plt.xlabel('Month')
        plt.ylabel('Total Requested Quantity')
        plt.grid()
        plt.xticks(rotation=45)
        plt.tight_layout()
        monthly_chart = os.path.join(output_folder, f"{asin}_monthly_requested_quantity.png")
        plt.savefig(monthly_chart)
        plt.close()

        # Save weekly and monthly data into an Excel file for this ASIN
        excel_path = os.path.join(output_folder, f"{asin}_po_data.xlsx")
        with pd.ExcelWriter(excel_path) as writer:
            weekly_agg.to_excel(writer, sheet_name='Weekly Quantity', index=False)
            monthly_trend.to_excel(writer, sheet_name='Monthly Trend', index=False)

        # Store results for this ASIN
        results[asin] = {
            'weekly_agg': weekly_agg,
            'monthly_trend': monthly_trend,
            'weekly_chart': weekly_chart,
            'monthly_chart': monthly_chart,
            'excel_report': excel_path
        }

    print(f"Generated graphs and Excel reports for {len(unique_asins)} ASINs in folder '{output_folder}'.")
    return results

def compare_historical_sales_po(asin, sales_df, po_df, output_folder="po_forecast_comparison"):
    """
    Compare historical sales with PO requested quantities for a given ASIN.
    Overlays historical sales and PO trends, computes correlation, 
    calculates growth percentages and volume insights,
    and provides a basic prediction for future PO quantities.
    Saves overlay graphs and merged data.

    Parameters:
      asin: The ASIN to analyze.
      sales_df: DataFrame containing weekly aggregated historical sales with columns ['ds', 'y'].
                - 'ds' is assumed to be a Sunday date (week-end).
      po_df: DataFrame containing historical PO data with columns ['ASIN', 'Order date', 'Requested quantity'].
      output_folder: Directory where results will be saved.

    Returns:
      merged_df: DataFrame merging historical sales and PO data.
      correlation: Correlation coefficient between sales and PO requested quantities.
      growth_info: Dictionary containing growth percentages, volume insights, and predicted next week PO quantity.
    """
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Ensure numeric conversion for Requested quantity; handle commas and spaces
    po_df = po_df.copy()
    po_df['Requested quantity'] = (
         po_df['Requested quantity']
         .astype(str)
         .str.replace(',', '')
         .str.strip()
         .replace('', '0')
    )
    po_df['Requested quantity'] = pd.to_numeric(po_df['Requested quantity'], errors='coerce').fillna(0)
    
    asin_po = po_df[po_df['ASIN'] == asin].copy()
    asin_po['Order date'] = pd.to_datetime(asin_po['Order date'], errors='coerce')
    asin_po.sort_values('Order date', inplace=True)

    # 3) Create a daily PO DataFrame (raw, no summation)
    # Rename columns to be consistent
    asin_po.rename(columns={'Order date': 'Date', 'Requested quantity': 'Daily_PO_Qty'}, inplace=True)

    # 4) Basic volume stats on the raw/daily POs
    total_po_qty = asin_po['Daily_PO_Qty'].sum()
    avg_po_qty   = asin_po['Daily_PO_Qty'].mean() if len(asin_po) else 0
    max_po_qty   = asin_po['Daily_PO_Qty'].max() if len(asin_po) else 0
    min_po_qty   = asin_po['Daily_PO_Qty'].min() if len(asin_po) else 0

    volume_insights = {
        'Total_PO_Quantity': total_po_qty,
        'Average_PO_Quantity': avg_po_qty,
        'Max_PO_Quantity': max_po_qty,
        'Min_PO_Quantity': min_po_qty
    }

    # 5) Basic linear model (optional) to predict next daily PO
    predicted_next_po_qty = 0
    if len(asin_po) > 1:
        asin_po = asin_po.reset_index(drop=True)
        asin_po['Index'] = np.arange(1, len(asin_po) + 1)
        X = asin_po[['Index']]
        y = asin_po['Daily_PO_Qty']
        from sklearn.linear_model import LinearRegression
        model = LinearRegression()
        model.fit(X, y)
        next_index = asin_po['Index'].iloc[-1] + 1
        predicted_next_po_qty = max(model.predict([[next_index]])[0], 0)

    prediction_info = {
        'Predicted_Next_Daily_PO_Quantity': predicted_next_po_qty
    }

    # 6) Merge weekly sales with daily PO if you want a combined DataFrame
    #    This is purely optional. The DS column is Sunday-based in sales_df, daily in PO.
    #    We'll do a left join on date. Many rows won't match exactly unless a daily date
    #    equals a Sunday from your sales.
    merged_df = pd.merge(
        sales_df.rename(columns={'ds': 'Date'}), 
        asin_po[['Date','Daily_PO_Qty']], 
        on='Date', 
        how='outer'
    ).sort_values('Date')
    merged_df['Daily_PO_Qty'] = merged_df['Daily_PO_Qty'].fillna(0)
    merged_df['y'] = merged_df['y'].fillna(0)

    # 7) Plot overlay: we can try to plot daily PO vs. weekly sales on the same axis
    plt.figure(figsize=(12,6))
    # Weekly sales
    plt.plot(
        merged_df['Date'], 
        merged_df['y'], 
        marker='o', 
        label='Weekly Sales (y)'
    )
    # Daily PO
    plt.plot(
        merged_df['Date'], 
        merged_df['Daily_PO_Qty'], 
        marker='x',
        label='Daily PO Qty'
    )
    plt.title(f"Daily POs vs Weekly Sales for ASIN {asin}")
    plt.xlabel("Date")
    plt.ylabel("Quantity")
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()

    chart_path = os.path.join(output_folder, f"{asin}_dailyPO_vs_weeklySales.png")
    plt.savefig(chart_path)
    plt.close()
    print(f"Chart saved to {chart_path}")

    # 8) Correlation
    # We'll only check correlation on days that have data in both columns:
    valid_idx = (merged_df['y'] > 0) & (merged_df['Daily_PO_Qty'] > 0)
    if valid_idx.sum() > 1:
        correlation = merged_df.loc[valid_idx, ['y','Daily_PO_Qty']].corr().iloc[0, 1]
        print(f"Correlation between weekly sales and daily PO for ASIN {asin}: {correlation:.2f}")
    else:
        correlation = None
        print(f"Not enough overlapping data for correlation on ASIN {asin}.")

    # 9) Write to Excel: two sheets - "Weekly Sales" and "Daily PO"
    # Also put merged in a third sheet if desired.
    excel_filename = os.path.join(output_folder, f"{asin}_sales_po_comparison.xlsx")
    with pd.ExcelWriter(excel_filename) as writer:
        # Sheet 1: Weekly Sales
        sales_df.to_excel(writer, sheet_name='Weekly Sales', index=False)

        # Sheet 2: Daily PO
        asin_po.to_excel(writer, sheet_name='Daily PO', index=False)

        # Optionally a third sheet with merged data
        merged_df.to_excel(writer, sheet_name='Merged (Optional)', index=False)

        # Another sheet with basic stats or growth info
        stats_df = pd.DataFrame([volume_insights])
        stats_df.to_excel(writer, sheet_name='PO Volume Insights', index=False)

        pred_df = pd.DataFrame([prediction_info])
        pred_df.to_excel(writer, sheet_name='PO Prediction', index=False)

    print(f"Excel saved to {excel_filename}")

    return merged_df, correlation, {
        'volume_insights': volume_insights,
        'prediction_info': prediction_info
    }
