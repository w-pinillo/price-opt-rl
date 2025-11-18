import polars as pl
import joblib
import numpy as np
import os

def calibrate_demand_noise(data_path: str, model_path: str):
    """
    Loads a trained demand model, makes predictions on the validation set,
    calculates residuals, and determines the standard deviation of these residuals
    to be used as noise_std for the MLDemandSimulator.
    """
    print(f"Loading validation data from {data_path}...")
    val_df = pl.read_parquet(data_path)

    print(f"Loading demand model from {model_path}...")
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        return None
    lgbm = joblib.load(model_path)

    # Define target and features - MUST MATCH THE FEATURES USED DURING TRAINING
    target_col = "total_units"
    feature_cols = ['avg_price', 'SHOP_WEEK', 'PROD_CATEGORY_DEP00001', 'PROD_CATEGORY_DEP00002', 'PROD_CATEGORY_DEP00004', 'PROD_CATEGORY_DEP00008', 'PROD_CATEGORY_DEP00009', 'PROD_CATEGORY_DEP00010', 'PROD_CATEGORY_DEP00011', 'PROD_CATEGORY_DEP00013', 'PROD_CATEGORY_DEP00016', 'PROD_CATEGORY_DEP00019', 'PROD_CATEGORY_DEP00021', 'PROD_CATEGORY_DEP00022', 'PROD_CATEGORY_DEP00023', 'PROD_CATEGORY_DEP00036', 'PROD_CATEGORY_DEP00037', 'PROD_CATEGORY_DEP00041', 'PROD_CATEGORY_DEP00048', 'PROD_CATEGORY_DEP00049', 'PROD_CATEGORY_DEP00051', 'PROD_CATEGORY_DEP00052', 'PROD_CATEGORY_DEP00054', 'PROD_CATEGORY_DEP00062', 'PROD_CATEGORY_DEP00067', 'PROD_CATEGORY_DEP00077', 'PROD_CATEGORY_DEP00081', 'PROD_CATEGORY_DEP00083', 'PROD_CATEGORY_DEP00084', 'PROD_CATEGORY_DEP00085', 'PROD_CATEGORY_DEP00086', 'PROD_CATEGORY_DEP00088', 'day_of_month', 'week_of_year', 'days_since_price_change', 'day_of_week_sin', 'day_of_week_cos', 'month_sin', 'month_cos', 'lag_1_units', 'lag_7_units', 'lag_14_units', 'lag_28_units', 'rolling_mean_7_units', 'rolling_mean_28_units', 'rolling_std_7_units', 'rolling_std_28_units', 'price_change_pct', 'price_position']
    
    print("Preparing features and target for prediction...")
    X_val = val_df[feature_cols]
    y_val = val_df[target_col]

    print("Making predictions on the validation set...")
    y_pred = lgbm.predict(X_val.to_numpy())

    # Calculate residuals
    residuals = y_val.to_numpy() - y_pred

    # Calculate the standard deviation of the residuals
    noise_std = np.std(residuals)

    print(f"\n--- Demand Noise Calibration Results ---")
    print(f"Standard deviation of residuals (noise_std): {noise_std:.4f}")
    print("--------------------------------------\n")

    return noise_std

if __name__ == "__main__":
    VAL_DATA_PATH = "data/processed/val_scaled.parquet"
    MODEL_PATH = "models/demand_model/lgbm_demand_model.joblib"
    
    if not os.path.exists(VAL_DATA_PATH):
        print(f"Error: Validation data not found at {VAL_DATA_PATH}")
    else:
        calibrate_demand_noise(VAL_DATA_PATH, MODEL_PATH)
