Next-Day Sales Prediction for Grocery Store
Overview

This project builds a machine learning model to predict the next day’s sales for each product in a grocery store using historical daily sales data. The goal is to help a small retailer reduce stock issues by forecasting demand per product.

Dataset:
Grocery Store Sales Dataset in 2025 (1,900 records)

Key columns used:

transaction_date – date of transaction

product_name – name of the product

quantity – units sold

Other columns like unit_price, customer_id, or store_name are ignored for simplicity.

Notebook Steps
1. Load Libraries & Dataset

Import pandas, numpy, matplotlib, seaborn, and scikit-learn.

Load the CSV dataset and inspect the first rows.

2. Preprocess & Aggregate

Convert transaction_date to datetime.

Aggregate daily sales per product.

Pivot the table to wide format: rows = dates, columns = products. Missing sales are filled with 0.

3. Feature Engineering

Create lag features for previous 3 days’ sales (lag1, lag2, lag3) for each product.

The target is the next day’s sales for each product.

Drop rows with missing values due to lags or the last day without a “next day”.

4. Train/Test Split & Model

Split data chronologically (80% train, 20% test).

Train a Random Forest Regressor using MultiOutputRegressor to predict sales for all products simultaneously.

5. Model Evaluation

Compute RMSE, MAE, and R² per product.

Visualize actual vs predicted sales for one or more products.

Optional: time-series cross-validation for more robust performance.

6. Optional Extensions

Predict tomorrow’s sales using the most recent 3 days of data.

Use gradient boosting models (XGBoost, LightGBM) for better performance.

Experiment with more lag features, rolling averages, or other temporal features.

Metrics Example
RMSE per product:
Apples            1.60
Bananas           1.24
Bread             1.60
...

How to Use

Open the notebook on Kaggle or locally.

Make sure the CSV dataset is in the same directory or in Kaggle input path.

Run cells sequentially from Load Libraries → Preprocess → Lag Features → Train/Test → Evaluation → Visualization.

For forecasting tomorrow’s sales, use the last few rows of the lagged features as input.

Requirements

Python 3.x

pandas, numpy, matplotlib, seaborn, scikit-learn
