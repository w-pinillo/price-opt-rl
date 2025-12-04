import polars as pl
import os
import gc
from src.etl import load_raw_data, save_interim
from src.features import select_top_products, aggregate_daily_lazy, generate_time_series_features_lazy
from src.utils import fit_save_scalers, apply_scalers, load_scalers

def run_data_pipeline(raw_data_path: str, processed_data_dir: str, scalers_dir: str):
    """
    Runs the complete data preparation pipeline in a memory-efficient way
    by processing data splits sequentially.
    """
    print("Starting data preparation pipeline...")

    # 1. Load raw data lazily
    print(f"Planning to load raw data from {raw_data_path}...")
    lazy_df = load_raw_data(raw_data_path)
    print("Raw data loading planned.")

    # 2. Select top 100 products and get their IDs
    print("Selecting top 100 products...")
    top_product_ids = select_top_products(lazy_df, n=100, output_path=processed_data_dir)
    print(f"Selected {len(top_product_ids)} top products.")

    # 3. Lazily aggregate daily data and generate features
    print("Planning data aggregation and feature generation...")
    lazy_aggregated = aggregate_daily_lazy(lazy_df, top_product_ids)
    lazy_with_features = generate_time_series_features_lazy(lazy_aggregated)
    print("Aggregation and feature generation planned.")

    # Define feature columns for scaling
    feature_cols = [
        "avg_price", "total_units", "total_sales",
        "day_of_week_sin", "day_of_week_cos", "month_sin", "month_cos",
        "lag_1_units", "lag_7_units", "lag_14_units", "lag_28_units",
        "rolling_mean_7_units", "rolling_mean_28_units",
        "rolling_std_7_units", "rolling_std_28_units",
        "price_change_pct",
        "day_of_month", "week_of_year", "is_weekend",
        "days_since_price_change", "price_position",
        "SHOP_WEEK"
    ]

    # Define temporal split dates
    train_end_date = pl.lit("2008-01-10").str.strptime(pl.Date, "%Y-%m-%d")
    val_end_date = pl.lit("2008-04-10").str.strptime(pl.Date, "%Y-%m-%d")

    # --- Process Train Split ---
    print("\nProcessing training set...")
    train_df_lazy = lazy_with_features.filter(pl.col("SHOP_DATE") <= train_end_date)
    train_df = train_df_lazy.collect(streaming=True)
    train_df = train_df.drop_nulls()
    print(f"Train shape after dropping NaNs: {train_df.shape}")
    
    print("Fitting and saving scalers...")
    fit_save_scalers(train_df, feature_cols, scalers_dir)
    
    print("Applying scalers to training set...")
    loaded_scalers = load_scalers(scalers_dir, feature_cols)
    train_df_scaled = apply_scalers(train_df, loaded_scalers, feature_cols)
    
    print("Saving scaled training set...")
    save_interim(train_df_scaled, processed_data_dir, "train_scaled.parquet")
    
    del train_df, train_df_scaled, loaded_scalers
    gc.collect()
    print("Training set processed and memory freed.")

    # --- Process Validation Split ---
    print("\nProcessing validation set...")
    val_df_lazy = lazy_with_features.filter(
        (pl.col("SHOP_DATE") > train_end_date) & (pl.col("SHOP_DATE") <= val_end_date)
    )
    val_df = val_df_lazy.collect(streaming=True)
    val_df = val_df.drop_nulls()
    print(f"Validation shape after dropping NaNs: {val_df.shape}")

    print("Loading scalers and applying to validation set...")
    loaded_scalers = load_scalers(scalers_dir, feature_cols)
    val_df_scaled = apply_scalers(val_df, loaded_scalers, feature_cols)

    print("Saving scaled validation set...")
    save_interim(val_df_scaled, processed_data_dir, "val_scaled.parquet")
    
    del val_df, val_df_scaled, loaded_scalers
    gc.collect()
    print("Validation set processed and memory freed.")

    # --- Process Test Split ---
    print("\nProcessing test set...")
    test_df_lazy = lazy_with_features.filter(pl.col("SHOP_DATE") > val_end_date)
    test_df = test_df_lazy.collect(streaming=True)
    test_df = test_df.drop_nulls()
    print(f"Test shape after dropping NaNs: {test_df.shape}")

    print("Loading scalers and applying to test set...")
    loaded_scalers = load_scalers(scalers_dir, feature_cols)
    test_df_scaled = apply_scalers(test_df, loaded_scalers, feature_cols)

    print("Saving scaled test set...")
    save_interim(test_df_scaled, processed_data_dir, "test_scaled.parquet")
    
    del test_df, test_df_scaled, loaded_scalers
    gc.collect()
    print("Test set processed and memory freed.")

    print("\nData preparation pipeline completed.")

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

