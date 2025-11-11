import polars as pl
import os
from src.etl import load_raw_data, save_interim
from src.features import select_top_products, aggregate_daily, generate_time_series_features, temporal_split
from src.utils import fit_save_scalers, apply_scalers

def run_data_pipeline(raw_data_path: str, processed_data_dir: str, scalers_dir: str):
    """
    Runs the complete data preparation pipeline.
    """
    print("Starting data preparation pipeline...")

    # 1. Load raw data
    print(f"Loading raw data from {raw_data_path}...")
    lazy_df = load_raw_data(raw_data_path)
    print("Raw data loaded.")

    # 2. Select top 100 products and filter the DataFrame
    print("Selecting top 100 products...")
    top_product_ids = select_top_products(lazy_df, n=100, output_path=processed_data_dir)
    print(f"Selected {len(top_product_ids)} top products.")

    # 3. Aggregate daily data for selected products
    print("Aggregating daily data...")
    product_daily_df = aggregate_daily(lazy_df, top_product_ids)
    print(f"Daily aggregated data shape: {product_daily_df.shape}")

    # 4. Generate time series features
    print("Generating time series features...")
    df_with_features = generate_time_series_features(product_daily_df)
    print(f"Data with features shape: {df_with_features.shape}")

    # Handle NaNs introduced by lag features (e.g., for the first few rows)
    # For simplicity, we'll drop rows with NaNs. A more sophisticated approach might impute.
    df_with_features = df_with_features.drop_nulls()
    print(f"Data after dropping NaNs: {df_with_features.shape}")

    # One-hot encode PROD_CATEGORY
    print("One-hot encoding PROD_CATEGORY...")
    df_with_features = df_with_features.to_dummies(columns=["PROD_CATEGORY"])
    print(f"Data shape after one-hot encoding: {df_with_features.shape}")

    # Define feature columns for scaling
    feature_cols = [
        "avg_price", "total_units", "total_sales",
        "day_of_week_sin", "day_of_week_cos", "month_sin", "month_cos",
        "lag_1_units", "lag_7_units", "lag_14_units", "lag_28_units",
        "rolling_mean_7_units", "rolling_mean_28_units",
        "rolling_std_7_units", "rolling_std_28_units",
        "price_change_pct",
        "day_of_month", "week_of_year", "is_weekend", # New temporal features
        "days_since_price_change", "price_position", # New product features
        "SHOP_WEEK"
    ]

    # Dynamically add one-hot encoded PROD_CATEGORY columns
    prod_category_cols = [col for col in df_with_features.columns if col.startswith("PROD_CATEGORY_")]
    feature_cols.extend(prod_category_cols)

    # Define temporal split dates
    train_end_date = "2008-01-10"
    val_end_date = "2008-04-10"

    # 5. Temporal splitting
    print(f"Splitting data into train (up to {train_end_date}), val (up to {val_end_date}), and test sets...")
    train_df, val_df, test_df = temporal_split(df_with_features, train_end_date, val_end_date)
    print(f"Train shape: {train_df.shape}, Val shape: {val_df.shape}, Test shape: {test_df.shape}")

    # 6. Scale numeric features
    print("Fitting and saving scalers...")
    fit_save_scalers(train_df, feature_cols, scalers_dir)
    
    print("Applying scalers to train, validation, and test sets...")
    # Load scalers to ensure consistency, though they are already in memory from fit_save_scalers
    # This step is more critical when applying scalers in a separate inference pipeline
    loaded_scalers = fit_save_scalers(train_df, feature_cols, scalers_dir) # Re-using fit_save_scalers to get the dict of scalers

    train_df_scaled = apply_scalers(train_df, loaded_scalers, feature_cols)
    val_df_scaled = apply_scalers(val_df, loaded_scalers, feature_cols)
    test_df_scaled = apply_scalers(test_df, loaded_scalers, feature_cols)
    print("Features scaled.")

    # 7. Save the split and scaled DataFrames
    print("Saving split and scaled dataframes...")
    save_interim(train_df_scaled, processed_data_dir, "train_scaled.parquet")
    save_interim(val_df_scaled, processed_data_dir, "val_scaled.parquet")
    save_interim(test_df_scaled, processed_data_dir, "test_scaled.parquet")
    print("Split and scaled dataframes saved.")

    print("Data preparation pipeline completed.")

if __name__ == "__main__":
    RAW_DATA_PATH = "data/transactions.parquet"
    PROCESSED_DATA_DIR = "data/processed"
    SCALERS_DIR = "models/scalers"

    # Ensure the raw data file exists
    if not os.path.exists(RAW_DATA_PATH):
        print(f"Error: Raw data file not found at {RAW_DATA_PATH}")
        print("Please ensure 'transactions.parquet' is in the 'data/' directory.")
    else:
        # Ensure scalers directory exists
        os.makedirs(SCALERS_DIR, exist_ok=True)
        run_data_pipeline(RAW_DATA_PATH, PROCESSED_DATA_DIR, SCALERS_DIR)
