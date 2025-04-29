📈 Forecast Model


Welcome to the Forecast-Model project!
This repo is home to a robust time series forecasting tool, built primarily using Python's heavy-hitters: Prophet, SARIMA, and a sprinkle of custom optimization magic. It’s designed to predict sales trends, inventory needs, and purchase order (PO) requirements — all while keeping future operations running smoother than a freshly waxed surfboard. 🏄

🚀 Project Purpose
The goal of this project is to forecast future sales and inventory levels for a set of products (ASINs). This helps businesses:

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
Here's a quick tour:

bash
Copy
Edit
.
├── Forecast Model Prophet Main.py   # Main script to run forecasts
├── models/                          # Saved Prophet/SARIMA models
├── model_cache/                     # Cached model parameters (pickled)
├── forecasts_folder/                # Generated forecast outputs
├── po_forecast_output/              # PO-based forecast adjustments
├── po_coverage_analysis/            # Analysis of PO impact on inventory
├── spreadsheets/                    # All input/output Excel files
└── requirements.txt                 # (You should create this if not yet)

📊 How to Use It
Prepare your data:

weekly_sales_data.xlsx

inventory_data.xlsx

po database.xlsx

Forecasts_folder (Amazon Forecast values)

Run Forecast Model Prophet Main.py

Review outputs in forecasts_folder/, po_coverage_analysis/, and others.


This is an example of what the output looks like for forecasting sales for one of the Motherboard models:
![image](https://github.com/user-attachments/assets/c72c0ba4-9f6f-4071-81e6-55470c523ad9)

It would also generate an xlsx file with the forecast:
![image](https://github.com/user-attachments/assets/fa5e900c-0550-49a7-8238-a530b2d48ae2)



