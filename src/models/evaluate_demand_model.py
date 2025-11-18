import joblib
import polars as pl
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os
import json
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

def evaluate_demand_model(model_path: str, data_path: str, scalers_dir: str, output_path: str):
    """
    Loads a trained LightGBM model, evaluates its performance on validation data,
    and saves a performance plot.
    """
    print(f"Loading model from {model_path}...")
    model = joblib.load(model_path)

    print(f"Loading validation data from {data_path}...")
    val_df = pl.read_parquet(data_path)

    # --- Load Feature List from file ---
    feature_names_path = os.path.join(os.path.dirname(model_path), "feature_names.json")
    if not os.path.exists(feature_names_path):
        raise FileNotFoundError(f"Feature names file not found at {feature_names_path}")
    with open(feature_names_path, 'r') as f:
        feature_names = json.load(f)
    print("Loaded feature names from file.")

    # --- Prepare Data for Evaluation ---
    if not all(col in val_df.columns for col in feature_names):
        raise ValueError("Not all feature names from training are present in the validation data.")
    
    X_val = val_df[feature_names]
    y_val_scaled = val_df["total_units"]

    # --- Make Predictions ---
    print("Making predictions on validation data...")
    predictions_scaled = model.predict(X_val.to_pandas())

    # --- Inverse Transform for Interpretable Metrics ---
    print("Loading scaler for 'total_units' to inverse transform values...")
    units_scaler_path = os.path.join(scalers_dir, "total_units_standardscaler.joblib")
    if not os.path.exists(units_scaler_path):
        raise FileNotFoundError(f"Scaler for 'total_units' not found at {units_scaler_path}")
    units_scaler = joblib.load(units_scaler_path)

    y_val_unscaled = units_scaler.inverse_transform(y_val_scaled.to_numpy().reshape(-1, 1)).flatten()
    predictions_unscaled = units_scaler.inverse_transform(predictions_scaled.reshape(-1, 1)).flatten()

    # --- Calculate Metrics ---
    r2 = r2_score(y_val_unscaled, predictions_unscaled)
    mae = mean_absolute_error(y_val_unscaled, predictions_unscaled)
    rmse = np.sqrt(mean_squared_error(y_val_unscaled, predictions_unscaled))

    print("\n--- Demand Model Performance Metrics ---")
    print(f"R-squared (R²): {r2:.4f}")
    print(f"Mean Absolute Error (MAE): {mae:.4f} units")
    print(f"Root Mean Squared Error (RMSE): {rmse:.4f} units")
    print("----------------------------------------\n")

    # --- Create and Save Plot ---
    print("Generating 'Actual vs. Predicted' plot...")
    plt.figure(figsize=(10, 10))
    sns.scatterplot(x=y_val_unscaled, y=predictions_unscaled, alpha=0.5)
    max_val = max(y_val_unscaled.max(), predictions_unscaled.max())
    min_val = min(y_val_unscaled.min(), predictions_unscaled.min())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
    plt.xlabel("Actual Units Sold")
    plt.ylabel("Predicted Units Sold")
    plt.title("Demand Model Performance: Actual vs. Predicted")
    plt.grid(True)
    plt.tight_layout()

    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)

    plt.savefig(output_path)
    print(f"Performance plot saved to {output_path}")
    plt.close()

if __name__ == "__main__":
    MODEL_PATH = "models/demand_model/lgbm_demand_model.joblib"
    DATA_PATH = "data/processed/val_scaled.parquet"
    SCALERS_DIR = "models/scalers"
    OUTPUT_PATH = "reports/figures/demand_model_performance.png"
    
    evaluate_demand_model(MODEL_PATH, DATA_PATH, SCALERS_DIR, OUTPUT_PATH)
