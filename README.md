📈 Forecast Model

Welcome to the Forecast-Model project!
This repo is home to a robust time series forecasting tool, built primarily using Python's heavy-hitters: Prophet, SARIMA, and a sprinkle of custom optimization magic. It’s designed to predict sales trends, inventory needs, and purchase order (PO) requirements — all while keeping future operations running smoother than a freshly waxed surfboard. 🏄

🚀 Project Purpose

The goal of this project is to forecast future sales and inventory levels for a set of Amazon products (ASINs). This helps businesses:

Plan purchase orders (POs) more efficiently

Optimize inventory to avoid stockouts or overstock

Make data-driven business decisions with confidence

This system is particularly geared towards Amazon sales data but can be easily adapted for broader e-commerce or retail datasets.


🛠️ Key Features

Multi-model forecasting: Uses Prophet and SARIMA models depending on data availability and performance.

PO Coverage Analysis: Evaluates how upcoming POs impact future inventory.

Restock Suggestions: Auto-generates restock recommendations based on forecasted sales and current inventory levels.

Flexible Input: Works with Excel/CSV files for seamless integration into existing workflows.

Optimization Caching: Saves model parameters for faster re-training and better forecasting accuracy.

🧩 Project Structure
| Stage | What happens | Key files / functions |
|-------|--------------|-----------------------|
| **Data ingestion** | Loads weekly sales, current inventory (“Runrate”), Amazon’s own forecast Excel exports, and historical purchase-order (“PO”) data. | `load_weekly_sales_data`, `load_runrate_inventory`, `load_amazon_forecasts_from_folder`, `merge_historical_data` |
| **Data prep** | - Cleans dates & column names<br>- Fills gaps with a Kalman filter, caps outliers with IQR<br>- Adds handy features (lagged demand, holiday lead-time, seasonality clustering, PO quantities). | `preprocess_data`, `add_lag_features`, `detect_seasonal_periods`, `add_holiday_lead_time_features` |
| **Model triage** | For each ASIN, decides **Prophet vs SARIMA**: <br>• Fewer than *FALLBACK_THRESHOLD* (20) observations → SARIMA <br>• Otherwise start with Prophet. | `choose_forecast_model` |
| **Model training** | *Prophet route*<br>  • Bayesian hyper-param search with Optuna<br>  • Regressors = Amazon forecasts, PO qty, holiday dummies<br><br>*SARIMA route*<br>  • Grid search over (p,d,q)(P,D,Q,m)<br>  • Rewards good combos; caches them for re-use. | `optimize_prophet_params_with_optuna`, `fit_sarima_model` |
| **Forecast generation** | Rolls out 16-week horizon, then: <br>• Optionally blends Prophet/SARIMA/Amazon with RMSE-weighted ensemble<br>• Seasonality overrides (STL-based “high/low/regular” weeks)<br>• Safety valve: if forecast drifts >30 % from Amazon mean it’s pulled back toward sanity. | `ensemble_forecast`, `apply_seasonal_adjustment`, `adjust_forecast_if_out_of_range` |
| **Inventory & PO intelligence** | Simulates inventory burn-down, flags stock-outs, suggests reorder qty, and even predicts Amazon’s future PO behaviour with a logistic + linear stack. | `compute_inventory_coverage`, `generate_restock_suggestions`, `train_amazon_po_model`, `forecast_amazon_po_orders` |
| **Reporting** | Spits out: <br>• Per-ASIN Excel workbooks (forecast, summary, charts)<br>• Combined 4/8/16-week roll-ups<br>• Low-coverage urgent list<br>• PNG graphs for every SKU<br>• Parameter history & model-feedback spreadsheets | `save_forecast_to_excel`, `generate_combined_weekly_report`, `generate_low_coverage_report`, `visualize_forecast_with_comparison` |
| **Caching & feedback loop** | All trained models (.pkl) and their scores persist in `/model_cache/`; future runs warm-start and learn which hyper-params are winning. | `save_model`, `load_model`, `update_param_history` |
| **Orchestrator** | `main()` drives the Prophet/SARIMA path;<br>`main_po_forecast_pipeline()` repeats the journey with PO regressors baked in. | bottom-of-file `main()` and `main_po_forecast_pipeline()` |

# 🛒 Demand–Inventory Forecast Suite

This repo houses a full-stack, opinionated forecasting pipeline that:

* **Predicts weekly sales** for every ASIN you track  
* **Blends** Prophet, SARIMA, and Amazon’s own forecasts with an ensemble that auto-tunes its weights  
* **Simulates inventory** depletion, flags stock-out risk, and spits out reorder quantities  
* **Learns** from its mistakes (parameter caching + Optuna) so it gets sharper every run  
* **Explains itself** with Excel reports and pretty PNG charts

---

## 🚀 Quick start

```bash
# 1.  Clone and install deps
git clone https://github.com/Cha86/Forecast-Model.git
cd Forecast-Model
pip install -r requirements.txt     # generated below

# 2.  Drop fresh data files into the repo root:
#     - weekly_sales_data.xlsx         (sales history)
#     - Runrate.xlsx                   (current on-hand inventory)
#     - inventory_data.xlsx            (historical inventory snapshots)
#     - po database.xlsx               (purchase-order history)
#     - forecasts_folder/*.xlsx        (Amazon “Mean / P70 / …” forecasts)
#     - ASINs to Forecast.xlsx         (1st column = ASIN list)

# 3.  Fire it up
python Forecast\ Model\ Prophet\ Main.py   # or just `python main.py` if you rename

```

This is an example of what the output looks like for forecasting sales for one of the Motherboard models:
![image](https://github.com/user-attachments/assets/c72c0ba4-9f6f-4071-81e6-55470c523ad9)

It would also generate an xlsx file with the forecast:
![image](https://github.com/user-attachments/assets/fa5e900c-0550-49a7-8238-a530b2d48ae2)



