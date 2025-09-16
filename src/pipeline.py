import polars as pl
import os
from src.etl import load_raw_data, save_interim
from src.features import select_top_products, aggregate_daily, generate_time_series_features

def run_data_pipeline(raw_data_path: str, processed_data_dir: str):
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

    # 5. Save the final processed DataFrame
    output_filename = "top100_daily.parquet"
    save_interim(df_with_features, processed_data_dir, output_filename)
    print("Data preparation pipeline completed.")

if __name__ == "__main__":
    RAW_DATA_PATH = "data/transactions.parquet"
    PROCESSED_DATA_DIR = "data/processed"

    # Ensure the raw data file exists
    if not os.path.exists(RAW_DATA_PATH):
        print(f"Error: Raw data file not found at {RAW_DATA_PATH}")
        print("Please ensure 'transactions.parquet' is in the 'data/' directory.")
    else:
        run_data_pipeline(RAW_DATA_PATH, PROCESSED_DATA_DIR)
