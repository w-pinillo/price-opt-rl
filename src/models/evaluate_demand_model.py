import polars as pl
import lightgbm as lgb
import joblib
import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

def evaluate_demand_model(data_path: str, model_path: str, figures_output_dir: str):
    """
    Loads a trained demand model and evaluates its performance on a given dataset.
    """
    print(f"Loading data from {data_path}...")
    val_df = pl.read_parquet(data_path)

    print(f"Loading model from {model_path}...")
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        return
    lgbm = joblib.load(model_path)

    # Define target and features (must match training)
    target_col = "total_units"
    feature_cols = [
        col for col in val_df.columns 
        if col not in [
            "SHOP_DATE", "PROD_CODE", "total_units", "total_sales", 
            "day_of_week", "month", "year", "day", "is_weekend"
        ]
    ]
    
    print("Defining features and target for validation set...")
    X_val = val_df[feature_cols]
    y_val = val_df[target_col]

    print("Making predictions on the validation set...")
    y_pred = lgbm.predict(X_val.to_numpy())

    # Calculate metrics
    r2 = r2_score(y_val.to_numpy(), y_pred)
    mae = mean_absolute_error(y_val.to_numpy(), y_pred)
    rmse = np.sqrt(mean_squared_error(y_val.to_numpy(), y_pred))

    print("\n--- Demand Model Evaluation Results ---")
    print(f"R-squared (R²): {r2:.4f}")
    print(f"Mean Absolute Error (MAE): {mae:.4f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
    print("-------------------------------------\n")

    # Create and save the Prediction vs. Actual plot
    print("Generating and saving Prediction vs. Actual plot...")
    os.makedirs(figures_output_dir, exist_ok=True)
    
    plt.figure(figsize=(10, 6))
    plt.scatter(y_val, y_pred, alpha=0.3)
    plt.plot([y_val.min(), y_val.max()], [y_val.min(), y_val.max()], '--', color='red', linewidth=2)
    plt.title('Demand Model: Actual vs. Predicted')
    plt.xlabel('Actual Units Sold')
    plt.ylabel('Predicted Units Sold')
    plt.grid(True)
    
    plot_path = os.path.join(figures_output_dir, 'demand_model_actual_vs_predicted.png')
    plt.savefig(plot_path)
    print(f"Plot saved to {plot_path}")

if __name__ == "__main__":
    DATA_PATH = "data/processed/val_scaled.parquet"
    MODEL_PATH = "models/demand_model/lgbm_demand_model.joblib"
    FIGURES_DIR = "reports/figures"
    
    if not os.path.exists(DATA_PATH):
        print(f"Error: Validation data not found at {DATA_PATH}")
    else:
        evaluate_demand_model(DATA_PATH, MODEL_PATH, FIGURES_DIR)
