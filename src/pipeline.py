import polars as pl
import os
import gc
from src.etl import load_raw_data, save_product_ids
from src.features import select_top_products, aggregate_daily_lazy, generate_time_series_features_lazy
from src.utils import fit_save_scalers, load_scalers, apply_scalers # Import scaling utilities
from omegaconf import OmegaConf # Import OmegaConf

# Helper function for processing and saving a data split
def process_and_save_split(
    lazy_df: pl.LazyFrame,
    output_dir: str,
    output_filename: str,
    feature_cols: list,
    scalers_dir: str, # New argument for scalers directory
    train_scalers: dict = None, # Pass scalers from training for val/test
    id_cols: list = None
) -> dict: # Returns the scalers fit on the training data
    if id_cols is None:
        id_cols = ["SHOP_DATE", "product_id"]

    # 1. OPTIMIZATION: Cast to Float32 immediately.
    cols_to_cast_to_float32 = [col for col in feature_cols if col not in id_cols]
    lazy_df = lazy_df.with_columns(
        [pl.col(c).cast(pl.Float32) for c in cols_to_cast_to_float32]
    )

    # Collect the DataFrame to apply scikit-learn scalers
    df_collect = lazy_df.collect()

    current_scalers = {}
    if train_scalers is None: # This is the training split
        print(f"Fitting and saving scalers to {scalers_dir}...")
        current_scalers = fit_save_scalers(df_collect, feature_cols, scalers_dir)
    else: # This is a validation or test split
        print(f"Loading scalers from {scalers_dir} and applying...")
        current_scalers = load_scalers(scalers_dir, feature_cols)
    
    # Apply scaling
    df_scaled = apply_scalers(df_collect, current_scalers, feature_cols)

    # 4. Handle NaNs and save
    df_scaled = df_scaled.drop_nulls()

    # 5. SINK TO DISK (The OOM Fix)
    output_file_path = os.path.join(output_dir, output_filename)
    print(f"Streaming processed data to {output_file_path}...")
    df_scaled.write_parquet(output_file_path) # Use write_parquet for collected DataFrame

    del lazy_df, df_collect, df_scaled
    gc.collect()

    return current_scalers # Return the fitted scalers for subsequent splits


def run_data_pipeline(config):
    """
    Runs the complete data preparation pipeline in a memory-efficient way
    by processing data splits sequentially, with an option to use pre-aggregated data.
    """
    print("Starting data preparation pipeline...")

    processed_data_dir = config.paths.processed_data_dir
    scalers_dir = config.paths.scalers_dir # Get scalers directory from config
    raw_data_path = config.paths.raw_data
    pre_aggregated_data_path = config.data_config.pre_aggregated_data_path
    use_pre_aggregated_data = config.data_config.use_pre_aggregated_data

    # Ensure scalers directory is created
    os.makedirs(scalers_dir, exist_ok=True)

    lazy_df_aggregated = None
    if use_pre_aggregated_data:
        print(f"Loading pre-aggregated daily data from {pre_aggregated_data_path}...")
        lazy_df_aggregated = pl.scan_parquet(pre_aggregated_data_path)
        # Assuming the pre-aggregated data already has columns named consistently
        # with what aggregate_daily_lazy would produce (SHOP_DATE, PROD_CODE, avg_price, total_units, total_sales, SHOP_WEEK)
        # No need to call load_raw_data, select_top_products, aggregate_daily_lazy
    else:
        # 1. Load raw data lazily
        print(f"Planning to load raw data from {raw_data_path}...")
        lazy_df = load_raw_data(raw_data_path)
        print("Raw data loading planned.")

        # 2. Select top 100 products and get their IDs
        print("Selecting top 100 products...")
        top_product_ids = select_top_products(lazy_df, n=100, output_path=processed_data_dir)
        print(f"Selected {len(top_product_ids)} top products.")
        
        # 3. Lazily aggregate daily data
        print("Planning data aggregation...")
        lazy_df_aggregated = aggregate_daily_lazy(lazy_df, top_product_ids)
        print("Aggregation planned.")

    # 4. Generate features on the aggregated (or pre-aggregated) data
    print("Planning feature generation...")
    lazy_with_features = generate_time_series_features_lazy(lazy_df_aggregated)
    print("Feature generation planned.")

    # 5. Rename PROD_CODE to product_id for consistency with downstream usage (e.g., process_and_save_split)
    # This ensures that whether data comes from raw or pre-aggregated source, the ID column is standardized.
    lazy_with_features = lazy_with_features.rename({"PROD_CODE": "product_id"})

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
        # "SHOP_WEEK" removed from feature_cols as it's an ID/context column, not a feature to be scaled
    ]
    
    # Identify non-feature columns that are needed for context but not scaling
    id_cols = ["SHOP_DATE", "product_id", "SHOP_WEEK"] # SHOP_WEEK is an int, but used for joining later, not a feature to be scaled

    # Define temporal split dates
    train_end_date = pl.lit("2008-01-10").str.strptime(pl.Date, "%Y-%m-%d")
    val_end_date = pl.lit("2008-04-10").str.strptime(pl.Date, "%Y-%m-%d")

    # --- Process Train Split ---
    print("\nProcessing training set...")
    train_df_lazy = lazy_with_features.filter(pl.col("SHOP_DATE") <= train_end_date)
    
    # Process and save train split, fitting and saving scalers
    # The returned 'train_scalers' are the fitted scaler objects
    train_scalers = process_and_save_split(
        lazy_df=train_df_lazy,
        output_dir=processed_data_dir,
        output_filename="train_scaled.parquet",
        feature_cols=feature_cols,
        scalers_dir=scalers_dir, # Pass scalers_dir
        train_scalers=None, # Indicate this is the train split
        id_cols=id_cols
    )
    print("Training set processed and memory freed.")

    # --- Process Validation Split ---
    print("\nProcessing validation set...")
    val_df_lazy = lazy_with_features.filter(
        (pl.col("SHOP_DATE") > train_end_date) & (pl.col("SHOP_DATE") <= val_end_date)
    )
    # Process and save validation split, using scalers from training set
    process_and_save_split(
        lazy_df=val_df_lazy,
        output_dir=processed_data_dir,
        output_filename="val_scaled.parquet",
        feature_cols=feature_cols,
        scalers_dir=scalers_dir, # Pass scalers_dir
        train_scalers=train_scalers, # Pass the fitted scalers from training
        id_cols=id_cols
    )
    print("Validation set processed and memory freed.")

    # --- Process Test Split ---
    print("\nProcessing test set...")
    test_df_lazy = lazy_with_features.filter(pl.col("SHOP_DATE") > val_end_date)
    # Process and save test split, using scalers from training set
    process_and_save_split(
        lazy_df=test_df_lazy,
        output_dir=processed_data_dir,
        output_filename="test_scaled.parquet",
        feature_cols=feature_cols,
        scalers_dir=scalers_dir, # Pass scalers_dir
        train_scalers=train_scalers, # Pass the fitted scalers from training
        id_cols=id_cols
    )
    print("Test set processed and memory freed.")

    print("\nData preparation pipeline completed.")

if __name__ == "__main__":
    # Load the configuration
    config = OmegaConf.load("configs/base_config.yaml")

    # Ensure processed data and scalers directory exists
    os.makedirs(config.paths.processed_data_dir, exist_ok=True)
    
    # Check if raw data path exists only if we are not using pre-aggregated data
    if not config.data_config.use_pre_aggregated_data and not os.path.exists(config.paths.raw_data):
        print(f"Error: Raw data file not found at {config.paths.raw_data}")
        print("Please ensure 'transactions.parquet' is in the 'data/' directory or set 'use_pre_aggregated_data' to true.")
    elif config.data_config.use_pre_aggregated_data and not os.path.exists(config.data_config.pre_aggregated_data_path):
        print(f"Error: Pre-aggregated data file not found at {config.data_config.pre_aggregated_data_path}")
        print("Please ensure the pre-aggregated parquet file exists or set 'use_pre_aggregated_data' to false.")
    else:
        run_data_pipeline(config)


