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
Week	Week_Start_Date	ASIN	MyForecast	Product Title	is_holiday_week	Trend	Inventory Coverage	Stockout Risk	Reorder Urgency	Sales Trend	Seasonality Index	Lifecycle Stage
W14	2025-03-30	B089FWWN62	46	B550I AORUS PRO AX		Stable	12.35	Low	Normal	Decreasing (▼)	0.91	Decline
W15	2025-04-06	B089FWWN62	38	B550I AORUS PRO AX		Stable	13.74	Low	Normal	Decreasing (▼)	0.83	Decline
W16	2025-04-13	B089FWWN62	40	B550I AORUS PRO AX		Stable	12.1	Low	Normal	Decreasing (▼)	0.95	Decline
W17	2025-04-20	B089FWWN62	64	B550I AORUS PRO AX		Stable	6.94	Low	Normal	Decreasing (▼)	1.15	Decline
W18	2025-04-27	B089FWWN62	68	B550I AORUS PRO AX		Stable	5.52	Low	Normal	Decreasing (▼)	1.17	Decline
W19	2025-05-04	B089FWWN62	61	B550I AORUS PRO AX		Stable	5.09	Low	Normal	Decreasing (▼)	1.16	Decline
W20	2025-05-11	B089FWWN62	63	B550I AORUS PRO AX		Stable	3.92	Low	Normal	Decreasing (▼)	1.19	Decline
W21	2025-05-18	B089FWWN62	53	B550I AORUS PRO AX		Stable	3.52	Low	Normal	Decreasing (▼)	0.87	Decline
W22	2025-05-25	B089FWWN62	45	B550I AORUS PRO AX		Stable	2.96	Low	Normal	Decreasing (▼)	0.95	Decline
W23	2025-06-01	B089FWWN62	61	B550I AORUS PRO AX		Stable	1.45	Low	Normal	Decreasing (▼)	0.86	Decline
W24	2025-06-08	B089FWWN62	66	B550I AORUS PRO AX		Stable	0.41	High	Urgent	Decreasing (▼)	0.88	Decline
W25	2025-06-15	B089FWWN62	66	B550I AORUS PRO AX		Stable	0	High	Urgent	Decreasing (▼)	1.16	Decline
W26	2025-06-22	B089FWWN62	67	B550I AORUS PRO AX		Stable	0	High	Urgent	Decreasing (▼)	0.92	Decline
W27	2025-06-29	B089FWWN62	63	B550I AORUS PRO AX		Stable	0	High	Urgent	Decreasing (▼)	1.12	Decline
W28	2025-07-06	B089FWWN62	65	B550I AORUS PRO AX		Stable	0	High	Urgent	Decreasing (▼)	0.96	Decline
W29	2025-07-13	B089FWWN62	77	B550I AORUS PRO AX		Stable	0	High	Urgent	Decreasing (▼)	1.05	Decline
![image](https://github.com/user-attachments/assets/de8a8253-e508-4a31-8f84-970741aa3aa8)


