import polars as pl
import joblib
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import numpy as np
import os
import matplotlib.pyplot as plt
import random

def evaluate_demand_models(data_path: str, lgbm_model_path: str, lr_model_path: str, plot: bool = False):
    """
    Evaluates the performance of the LightGBM and Linear Regression demand models.
    """
    print(f"Loading test data from {data_path}...")
    test_df = pl.read_parquet(data_path)

    # Define target and features
    target_col = "total_units"
    feature_cols = [
        col for col in test_df.columns 
        if col not in [
            "SHOP_DATE", "PROD_CODE", "total_units", "total_sales", 
            "day_of_week", "month", "year", "day", "is_weekend"
        ]
    ]
    
    X_test = test_df[feature_cols]
    y_test = test_df[target_col]

    print("Loading models...")
    lgbm_model = joblib.load(lgbm_model_path)
    lr_model = joblib.load(lr_model_path)

    print("Making predictions...")
    lgbm_preds = lgbm_model.predict(X_test.to_numpy())
    lr_preds = lr_model.predict(X_test.to_numpy())

    print("Evaluating LightGBM model...")
    lgbm_r2 = r2_score(y_test.to_numpy(), lgbm_preds)
    lgbm_mae = mean_absolute_error(y_test.to_numpy(), lgbm_preds)
    lgbm_rmse = np.sqrt(mean_squared_error(y_test.to_numpy(), lgbm_preds))

    print("Evaluating Linear Regression model...")
    lr_r2 = r2_score(y_test.to_numpy(), lr_preds)
    lr_mae = mean_absolute_error(y_test.to_numpy(), lr_preds)
    lr_rmse = np.sqrt(mean_squared_error(y_test.to_numpy(), lr_preds))

    print("\n--- Model Performance Comparison ---")
    print(f"| Metric                | LightGBM         | Linear Regression |")
    print(f"|-----------------------|------------------|-------------------|")
    print(f"| R-squared             | {lgbm_r2:<16.4f} | {lr_r2:<17.4f} |")
    print(f"| Mean Absolute Error   | {lgbm_mae:<16.4f} | {lr_mae:<17.4f} |")
    print(f"| Root Mean Squared Error| {lgbm_rmse:<16.4f} | {lr_rmse:<17.4f} |")
    print("------------------------------------\n")

    if plot:
        print("Generating plot...")
        
        # Add predictions to the dataframe
        test_df = test_df.with_columns([
            pl.Series("lgbm_preds", lgbm_preds),
            pl.Series("lr_preds", lr_preds)
        ])

        # Select a random SKU
        random_sku = random.choice(test_df["PROD_CODE"].unique().to_list())
        sku_df = test_df.filter(pl.col("PROD_CODE") == random_sku).sort("SHOP_DATE")

        plt.figure(figsize=(15, 7))
        plt.plot(sku_df["SHOP_DATE"], sku_df[target_col], label="Actual Demand", color='black', linestyle='--')
        plt.plot(sku_df["SHOP_DATE"], sku_df["lgbm_preds"], label="LightGBM Predictions", color='blue')
        plt.plot(sku_df["SHOP_DATE"], sku_df["lr_preds"], label="Linear Regression Predictions", color='red')
        
        plt.title(f"Demand Prediction vs. Actual for SKU: {random_sku}")
        plt.xlabel("Date")
        plt.ylabel("Total Units (Scaled)")
        plt.legend()
        plt.grid(True)
        
        # Save the plot
        output_dir = "reports/figures"
        os.makedirs(output_dir, exist_ok=True)
        plot_path = os.path.join(output_dir, f"demand_prediction_{random_sku}.png")
        plt.savefig(plot_path)
        print(f"Plot saved to {plot_path}")


if __name__ == "__main__":
    TEST_DATA_PATH = "data/processed/test_scaled.parquet"
    LGBM_MODEL_PATH = "models/demand_model/lgbm_demand_model.joblib"
    LR_MODEL_PATH = "models/demand_model/linear_regression_demand_model.joblib"

    # Add a command line argument to enable plotting
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--plot", action="store_true", help="Generate and save a plot of predictions vs actuals for a random SKU.")
    args = parser.parse_args()

    if not all(os.path.exists(p) for p in [TEST_DATA_PATH, LGBM_MODEL_PATH, LR_MODEL_PATH]):
        print("Error: One or more required files (test data, models) not found.")
    else:
        evaluate_demand_models(TEST_DATA_PATH, LGBM_MODEL_PATH, LR_MODEL_PATH, plot=args.plot)