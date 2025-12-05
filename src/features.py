import polars as pl
import json
import os
import numpy as np
from src.etl import save_product_ids

def select_top_products(lazy_df: pl.LazyFrame, n: int = 100, output_path: str = "data/processed") -> list:
    """
    Selects the top N products based on total sales after filtering out products
    with missing days. Saves the list of top product IDs to a JSON file.
    """
    # Create a LazyFrame of PROD_CODEs that do not have missing days
    products_no_missing_lazy = (
        lazy_df
        .group_by("PROD_CODE")
        .agg([
            pl.col("SHOP_DATE").min().alias("min_date"),
            pl.col("SHOP_DATE").max().alias("max_date"),
            pl.col("SHOP_DATE").n_unique().alias("unique_dates")
        ])
        .with_columns(
            (pl.col("max_date") - pl.col("min_date")).dt.total_days().alias("expected_days")
        )
        .with_columns(
            (pl.col("expected_days") + 1 - pl.col("unique_dates")).alias("missing_days")
        )
        .filter(pl.col("missing_days") == 0)
        .select("PROD_CODE")
    )
    
    # NEW STEP: Cache this intermediate result to disk to reduce memory pressure
    temp_products_no_missing_path = os.path.join(output_path, "temp_products_no_missing.parquet")
    print(f"Caching intermediate products_no_missing_lazy to {temp_products_no_missing_path}...")
    products_no_missing_lazy.sink_parquet(temp_products_no_missing_path)
    # Read it back lazily
    products_no_missing_lazy_from_disk = pl.scan_parquet(temp_products_no_missing_path)

    # Filter the main lazy_df by semi-joining with products_no_missing_lazy_from_disk
    lazy_df_filtered = lazy_df.semi_join(products_no_missing_lazy_from_disk, on="PROD_CODE")

    # Create a list of the top N products by sales from the filtered lazy_df
    # This collect is on a very small DataFrame (N rows)
    top_products_list = (
        lazy_df_filtered
        .group_by("PROD_CODE")
        .agg(pl.sum("SPEND").alias("total_sales"))
        .sort("total_sales", descending=True)
        .limit(n)
        .select("PROD_CODE")
        .collect() # This collection is intentional as the list is small (N=100)
        .to_series()
        .to_list()
    )
    
    save_product_ids(top_products_list, output_path)

    return top_products_list

# Removed: aggregate_daily as it contained a .collect() call.
# The pipeline should use aggregate_daily_lazy directly.

def aggregate_daily_lazy(lazy_df: pl.LazyFrame, product_ids: list) -> pl.LazyFrame:
    """
    Lazily aggregates transaction data to a daily level for selected products.
    """
    return (
        lazy_df
        .filter(pl.col("PROD_CODE").is_in(product_ids))
        .with_columns(
            (pl.col("SPEND") / pl.col("QUANTITY")).alias("PRICE")
        )
        .group_by(["SHOP_DATE", "PROD_CODE"])
        .agg([
            pl.mean("PRICE").alias("avg_price"),
            pl.sum("QUANTITY").alias("total_units"),
            pl.sum("SPEND").alias("total_sales"),
            pl.first("SHOP_WEEK").alias("SHOP_WEEK"),
        ])
        .sort(["SHOP_DATE", "PROD_CODE"])
    )

# Removed: generate_time_series_features as it contained a .collect() call.
# The pipeline should use generate_time_series_features_lazy directly.

def generate_time_series_features_lazy(lazy_df: pl.LazyFrame) -> pl.LazyFrame:
    """
    Lazily generates time-series features for the daily aggregated product data.
    """
    # First, create intermediate time-based features
    lazy_df_with_base_features = lazy_df.with_columns([
        pl.col("SHOP_DATE").dt.weekday().alias("day_of_week"), # Monday=1, Sunday=7
        pl.col("SHOP_DATE").dt.month().alias("month"),
        pl.col("SHOP_DATE").dt.year().alias("year"),
        pl.col("SHOP_DATE").dt.day().alias("day_of_month"),
        pl.col("SHOP_DATE").dt.week().alias("week_of_year"),
        (pl.col("SHOP_DATE").dt.weekday().is_in([6, 7])).alias("is_weekend"), # Saturday=6, Sunday=7
    ])

    # Now, use the intermediate features to create the final feature set
    lazy_df_with_all_features = lazy_df_with_base_features.with_columns([
        # Seasonality features (sin/cos transformations)
        (2 * np.pi * pl.col("day_of_week") / 7).sin().alias("day_of_week_sin"),
        (2 * np.pi * pl.col("day_of_week") / 7).cos().alias("day_of_week_cos"),
        (2 * np.pi * pl.col("month") / 12).sin().alias("month_sin"),
        (2 * np.pi * pl.col("month") / 12).cos().alias("month_cos"),

        # Lag features for total_units
        pl.col("total_units").shift(1).over("PROD_CODE").alias("lag_1_units"),
        pl.col("total_units").shift(7).over("PROD_CODE").alias("lag_7_units"),
        pl.col("total_units").shift(14).over("PROD_CODE").alias("lag_14_units"),
        pl.col("total_units").shift(28).over("PROD_CODE").alias("lag_28_units"),
        
        # Rolling mean features for total_units
        pl.col("total_units").rolling_mean(window_size=7).over("PROD_CODE").alias("rolling_mean_7_units"),
        pl.col("total_units").rolling_mean(window_size=28).over("PROD_CODE").alias("rolling_mean_28_units"),

        # Rolling std dev features for total_units
        pl.col("total_units").rolling_std(window_size=7).over("PROD_CODE").alias("rolling_std_7_units"),
        pl.col("total_units").rolling_std(window_size=28).over("PROD_CODE").alias("rolling_std_28_units"),
        
        # Price change percentage
        ((pl.col("avg_price") / (pl.col("avg_price").shift(1).over("PROD_CODE") + 1e-6)) - 1)
        .fill_nan(0.0)
        .pipe(lambda expr: pl.when(expr.is_infinite()).then(0.0).otherwise(expr)) # Handle infinite values
        .alias("price_change_pct"),
        
        # Days since last price change (simplified for now, assumes daily data)
        (pl.int_range(0, pl.count()).over("PROD_CODE") + 1).alias("days_since_price_change"),

        # Price position
        ((pl.col("avg_price") - pl.min("avg_price").over("PROD_CODE")) / 
         (pl.max("avg_price").over("PROD_CODE") - pl.min("avg_price").over("PROD_CODE") + 1e-6)
        )
        .fill_nan(0.0)
        .pipe(lambda expr: pl.when(expr.is_infinite()).then(0.0).otherwise(expr)) # Handle infinite values
        .alias("price_position"),
    ])
    
    return lazy_df_with_all_features



