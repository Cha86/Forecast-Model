# main.py

import os
import pandas as pd
import numpy as np
from data_loader import (
    load_weekly_sales_data,
    load_asins_to_forecast,
    load_amazon_forecasts_from_folder
)
from preprocessing import preprocess_data, prepare_time_series_with_lags
from models.sarima_helpers import fit_sarima_model, sarima_forecast, create_holiday_regressors
from models.xgboost_helpers import train_xgboost, xgboost_forecast
from models.prophet_helpers import (
    forecast_with_custom_params,
    optimize_prophet_params,
    calculate_rmse
)
from excel_output import (
    save_summary_to_excel,
    save_forecast_to_excel,
    save_feedback_to_excel,
    generate_4_week_report,
    generate_combined_weekly_report,
    save_4_8_16_forecast_summary
)
from reward_penalty import load_param_histories, save_param_histories, update_param_history
from analysis import analyze_po_data_by_asin, compare_historical_sales_po
from forecasting_utils import adjust_forecast_if_out_of_range, summarize_usage
from config import (
    PARAM_COUNTER,
    POOR_PARAM_FOUND,
    FALLBACK_THRESHOLD,
    forecast_errors,
    prophet_feedback,
    sarima_feedback,
    xgboost_feedback
)
from sklearn.metrics import mean_absolute_error, mean_squared_error, median_absolute_error
from math import sqrt
from prophet.diagnostics import cross_validation, performance_metrics
from prophet import Prophet

def get_shifted_holidays():
    """
    Example holiday DataFrame with 'ds' as date and 'holiday' as holiday name.
    Adjust to match your actual holiday data.
    """
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

def cross_validate_prophet_model(ts_data, initial='180 days', period='90 days', horizon='90 days'):
    """
    Run Prophet's built-in cross-validation and performance metrics.
    """
    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        seasonality_mode='multiplicative'
    )
    model.fit(ts_data[['ds', 'y']])
    df_cv = cross_validation(model, initial=initial, period=period, horizon=horizon)
    df_performance = performance_metrics(df_cv)
    print("Prophet Model Cross-Validation Results:")
    print(df_performance)
    return df_cv, df_performance

def choose_forecast_model(ts_data, threshold=FALLBACK_THRESHOLD, holidays=None):
    """
    Decide whether to use SARIMA or Prophet based on data sufficiency.
    Placeholder function: implement your logic here.
    """
    # Example logic: use SARIMA if data length is sufficient, else use Prophet
    if len(ts_data) >= 52:  # Example threshold
        model_fit, params = fit_sarima_model(ts_data, holidays, seasonal_period=52, asin=None)
        if model_fit:
            return model_fit, "SARIMA"
    # Fallback to Prophet
    return None, "Prophet"

def main():
    """
    Main function to orchestrate the forecasting process.
    """
    # Load existing parameter histories
    sarima_param_history, prophet_param_history = load_param_histories()

    sales_file = 'weekly_sales_data.xlsx'
    forecasts_folder = 'forecasts_folder'
    asins_to_forecast_file = 'ASINs to Forecast.xlsx'
    output_file = 'consolidated_forecast.xlsx'
    horizon = 16

    # Load and preprocess data
    data = load_weekly_sales_data(sales_file)
    valid_data = data[data['asin'].notna() & (data['asin'] != '#N/A')]
    missing_asin_data = data[data['asin'].isna() | (data['asin'] == '#N/A')]

    if not missing_asin_data.empty:
        print("The following entries have no ASIN and will be noted in the forecast file:")
        print(missing_asin_data[['product title', 'week', 'year', 'y']].to_string())

    asins_to_forecast = load_asins_to_forecast(asins_to_forecast_file)
    print(f"ASINs to forecast: {asins_to_forecast}")

    asin_list = valid_data['asin'].unique()
    asin_list = [asin for asin in asin_list if asin in asins_to_forecast]

    consolidated_forecasts = {}
    param_grid = {
        'changepoint_prior_scale': [0.05, 0.1, 0.2, 0.3, 0.4, 0.5],
        'seasonality_prior_scale': [0.05, 0.1, 1, 2, 3, 4, 5],
        'holidays_prior_scale': [5, 10, 15]
    }

    holidays = get_shifted_holidays()

    # Optional cross-validation test on the first ASIN in the list
    if len(asin_list) > 0:
        test_asin = asin_list[0]
        test_ts_data = prepare_time_series_with_lags(valid_data, test_asin, lag_weeks=1)
        if not test_ts_data.empty and len(test_ts_data.dropna()) >= 2:
            print(f"Performing cross-validation on ASIN {test_asin} Prophet model...")
            cross_validate_prophet_model(test_ts_data, initial='180 days', period='90 days', horizon='90 days')
        else:
            print(f"Not enough data for {test_asin} to perform cross-validation test.")

    insufficient_data_folder = "Insufficient data"
    sufficient_data_folder = "Sufficient data"
    os.makedirs(insufficient_data_folder, exist_ok=True)
    os.makedirs(sufficient_data_folder, exist_ok=True)

    global PARAM_COUNTER

    for asin in asin_list:
        if pd.isna(asin) or asin == '#N/A':
            print(f"Skipping invalid ASIN: {asin}")
            continue

        product_title = valid_data[valid_data['asin'] == asin]['product title'].iloc[0]
        print(f"\nProcessing ASIN: {asin} - {product_title}")
        
        # Load Amazon forecasts
        forecast_data = load_amazon_forecasts_from_folder(forecasts_folder, asin)
        if not forecast_data:
            print(f"No forecast data found for ASIN {asin}, skipping.")
            continue

        # Prepare data
        ts_data = prepare_time_series_with_lags(valid_data, asin, lag_weeks=1)
        print(f"Time series data for ASIN {asin} prepared. Dataset size: {len(ts_data)}")
        non_nan_count = len(ts_data.dropna())
        if non_nan_count < 2:
            print(f"Not enough data for ASIN {asin} (only {non_nan_count} data points), skipping.")
            no_data_output = os.path.join(insufficient_data_folder, f"{asin}_no_data.txt")
            with open(no_data_output, 'w') as f:
                f.write("Insufficient data for training/forecasting.\n")
            continue

        # Decide model type (SARIMA or Prophet)
        model, model_type = choose_forecast_model(ts_data, threshold=FALLBACK_THRESHOLD, holidays=holidays)

        # Train XGBoost, but do NOT blend with MyForecast
        xgb_model, xgb_features, xgb_shap_values = train_xgboost(ts_data, target='y')

        if model_type == "SARIMA":
            n = len(ts_data)
            split = int(n * 0.8)
            train_sarima = ts_data.iloc[:split]
            test_sarima = ts_data.iloc[split:]

            # Create exogenous variables for the test set
            exog_test = create_holiday_regressors(test_sarima, holidays)

            if model is None:
                print(f"SARIMA model fitting failed for {asin}, skipping.")
                no_data_output = os.path.join(insufficient_data_folder, f"{asin}_no_data.txt")
                with open(no_data_output, 'w') as f:
                    f.write("Insufficient data for training/forecasting.\n")
                continue

            try:
                best_sarima_model, best_params = fit_sarima_model(
                    data=ts_data,
                    holidays=holidays,
                    seasonal_period=52,
                    asin=asin  # For R/P tracking
                )

                if best_sarima_model is None:
                    print(f"SARIMA model fitting failed for {asin}, skipping.")
                    no_data_output = os.path.join(insufficient_data_folder, f"{asin}_model_failed.txt")
                    with open(no_data_output, 'w') as f:
                        f.write("Model fitting failed.\n")
                    continue

                # Extract SARIMA parameters for history
                param_tuple = best_params  # Assuming best_params is a tuple like (p,d,q,P,D,Q,m)

                steps = len(test_sarima)
                sarima_test_forecast_df = sarima_forecast(
                    model_fit=best_sarima_model,
                    steps=steps,
                    last_date=train_sarima['ds'].iloc[-1],
                    exog=exog_test
                )

                # Evaluate forecast on the test portion
                sarima_preds = sarima_test_forecast_df['MyForecast'].values
                sarima_mae = mean_absolute_error(test_sarima['y'], sarima_preds)
                sarima_rmse = sqrt(mean_squared_error(test_sarima['y'], sarima_preds))
                print(f"SARIMA Test MAE: {sarima_mae:.4f}, RMSE: {sarima_rmse:.4f}")

                update_param_history(
                    history_dict=sarima_param_history,
                    asin=asin,
                    param_tuple=param_tuple,
                    rmse=sarima_rmse,
                    mae=sarima_mae
                )

                last_date_full = ts_data['ds'].iloc[-1]
                exog_future = create_holiday_regressors(ts_data.tail(horizon), holidays)  # Adjust as needed
                final_forecast_df = sarima_forecast(
                    model_fit=best_sarima_model,
                    steps=horizon,
                    last_date=train_sarima['ds'].iloc[-1],
                    exog=exog_future
                )

                if final_forecast_df.empty:
                    print(f"Forecasting failed for ASIN {asin}, skipping.")
                    no_data_output = os.path.join(insufficient_data_folder, f'{asin}_forecast_failed.txt')
                    with open(no_data_output, 'w') as f:
                        f.write('Failed to forecast due to insufficient data.\n')
                    continue

                comparison = final_forecast_df.copy()
                comparison['ASIN'] = asin
                comparison['Product Title'] = product_title

                # Merge historical 'y' for fallback detection
                comparison = comparison.merge(ts_data[['ds', 'y']], on='ds', how='left')

                if forecast_data:
                    for ftype, values in forecast_data.items():
                        # Load each forecast type
                        horizon_values = values[:horizon] if len(values) >= horizon else values
                        if len(horizon_values) < horizon and len(horizon_values) > 0:
                            horizon_values = np.pad(
                                horizon_values, (0, horizon - len(horizon_values)),
                                'constant', constant_values=horizon_values[-1]
                            )
                        elif len(horizon_values) == 0:
                            horizon_values = np.zeros(horizon, dtype=int)

                        # Assign to DataFrame columns
                        ftype_lower = ftype.lower()
                        if 'mean' in ftype_lower:
                            comparison['Amazon Mean Forecast'] = horizon_values
                        elif 'p70' in ftype_lower:
                            comparison['Amazon P70 Forecast'] = horizon_values
                        elif 'p80' in ftype_lower:
                            comparison['Amazon P80 Forecast'] = horizon_values
                        elif 'p90' in ftype_lower:
                            comparison['Amazon P90 Forecast'] = horizon_values
                        else:
                            print(f"Warning: Unrecognized forecast type '{ftype}'. Skipping.")

                    # Weighted blend with Amazon
                    MEAN_WEIGHT = 0.7
                    P70_WEIGHT  = 0.2
                    P80_WEIGHT  = 0.1

                    blended_amz = (
                        MEAN_WEIGHT * comparison['Amazon Mean Forecast']
                        + P70_WEIGHT * comparison.get('Amazon P70 Forecast', 0)
                        + P80_WEIGHT * comparison.get('Amazon P80 Forecast', 0)
                    ).clip(lower=0)

                    FALLBACK_RATIO = 0.3  # 30% Amazon, 70% SARIMA
                    comparison['MyForecast'] = (
                        (1 - FALLBACK_RATIO) * comparison['MyForecast']
                        + FALLBACK_RATIO * blended_amz
                    ).clip(lower=0)

                if xgb_model is not None:
                    xgb_future_df = xgboost_forecast(
                        xgb_model, ts_data,
                        forecast_steps=horizon, target='y',
                        features=xgb_features
                    )
                    comparison = comparison.merge(xgb_future_df, on='ds', how='left', suffixes=('', '_XGB'))
                    comparison['MyForecast_XGB'] = comparison['MyForecast_XGB'].fillna(0)
                    print(f"XGBoost forecasts generated for ASIN {asin} and saved separately (not blended).")

                comparison = adjust_forecast_if_out_of_range(
                    comparison, asin, forecast_col_name='MyForecast', adjustment_threshold=0.3
                )

                past_8_weeks = ts_data.sort_values('ds').tail(8)
                if not past_8_weeks.empty and 'MyForecast' in comparison.columns:
                    past8_avg = past_8_weeks['y'].mean()
                    forecast_mean = comparison['MyForecast'].mean()
                    if forecast_mean > 1.5 * past8_avg:
                        print(f"Adjusting SARIMA forecast: past 8-week avg={past8_avg:.2f}, forecast mean={forecast_mean:.2f}")
                        comparison['MyForecast'] = (
                            0.8 * past8_avg + 0.2 * comparison['MyForecast']
                        ).clip(lower=0)

                # Summaries and Visualization
                summary_stats, total_forecast_16, total_forecast_8, total_forecast_4, \
                max_forecast, min_forecast, max_week, min_week = calculate_summary_statistics(
                    ts_data, comparison, horizon=horizon
                )
                visualize_forecast_with_comparison(
                    ts_data, comparison, summary_stats,
                    total_forecast_16, total_forecast_8, total_forecast_4,
                    max_forecast, min_forecast, max_week, min_week,
                    asin, product_title, sufficient_data_folder
                )

                # Log fallback triggers
                from forecasting_utils import log_fallback_triggers
                log_fallback_triggers(comparison, asin, product_title)

                # Save Final Forecast and Summary
                output_file_name = f'forecast_summary_{asin}.xlsx'
                output_file_path = os.path.join(sufficient_data_folder, output_file_name)
                with pd.ExcelWriter(output_file_path, mode='w') as writer:
                    comparison.to_excel(writer, index=False)

                save_summary_to_excel(
                    comparison,
                    summary_stats,
                    total_forecast_16,
                    total_forecast_8,
                    total_forecast_4,
                    max_forecast,
                    min_forecast,
                    max_week,
                    min_week,
                    output_file_path,
                    metrics=None  # Replace with actual metrics if available
                )
                consolidated_forecasts[asin] = comparison

            except ValueError as e:
                print(f"Error during SARIMA prediction for ASIN {asin}: {e}")
                continue

        else:
            # Prophet model block (similar to SARIMA but using Prophet)
            # This block should include training Prophet, optimizing parameters,
            # generating forecasts, adjusting forecasts, logging, saving to Excel, etc.
            # Due to brevity, this implementation is omitted but should follow similar steps.
            pass  # Replace with actual Prophet model handling

    # After processing all ASINs
    final_output_path = output_file
    save_forecast_to_excel(final_output_path, consolidated_forecasts, missing_asin_data, base_year=2025)
    save_feedback_to_excel(prophet_feedback, "prophet_feedback.xlsx")
    generate_4_week_report(consolidated_forecasts)
    generate_combined_weekly_report(consolidated_forecasts)

    print("\n--- Analyzing Purchase Order Data ---")
    analyze_po_data_by_asin(po_file="po database.xlsx", output_folder="po_analysis_by_asin")

    # Load PO data for comparison
    po_data = pd.read_excel("po database.xlsx")
    po_data.columns = po_data.columns.str.strip()

    print("\n--- Comparing Historical Sales with PO Data ---")
    comparison_output_folder = "po_forecast_comparison"
    os.makedirs(comparison_output_folder, exist_ok=True)

    for asin, comp_df in consolidated_forecasts.items():
        print(f"\nComparing sales and PO for ASIN {asin}")
        # Filter historical sales data for the current ASIN and aggregate weekly
        sales_subset = valid_data[valid_data['asin'] == asin][['ds', 'y']].copy()
        sales_subset['Week_Period'] = sales_subset['ds'].dt.to_period('W-SUN')
        sales_subset['ds'] = sales_subset['Week_Period'].apply(lambda r: r.end_time)
        weekly_sales = sales_subset.groupby('ds', as_index=False)['y'].sum()

        # Compare historical sales with purchase order data
        merged_result, correlation, growth_info = compare_historical_sales_po(
            asin=asin,
            sales_df=weekly_sales,
            po_df=po_data,
            output_folder=comparison_output_folder
        )

        print(f"Growth Info for ASIN {asin}: {growth_info}")
        print(f"Correlation for ASIN {asin}: {correlation}")

    # Save the parameter histories at the end
    save_param_histories()

    print(f"Total number of parameter sets tested: {PARAM_COUNTER}")        
    if POOR_PARAM_FOUND:
        print("Note: Early stopping occurred for some ASINs due to poor parameter performance.")

    summarize_usage()

if __name__ == '__main__':
    main()
