import polars as pl
import json
import os
from src.etl import save_product_ids

def select_top_products(lazy_df: pl.LazyFrame, n: int = 100, output_path: str = "data/processed") -> list:
    """
    Selects the top N products based on total sales after filtering out products
    with missing days. Saves the list of top product IDs to a JSON file.
    """
    # Filter products without missing days
    products_no_missing_list = (
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
        .collect()
        .to_series()
        .to_list()
    )

    lazy_df_filtered = lazy_df.filter(pl.col("PROD_CODE").is_in(products_no_missing_list))

    # Create a list of the top N products by sales
    top_products_list = (
        lazy_df_filtered
        .group_by("PROD_CODE")
        .agg(pl.sum("SPEND").alias("total_sales"))
        .sort("total_sales", descending=True)
        .limit(n)
        .select("PROD_CODE")
        .collect()
        .to_series()
        .to_list()
    )
    
    save_product_ids(top_products_list, output_path)

    return top_products_list

def aggregate_daily(lazy_df: pl.LazyFrame, product_ids: list) -> pl.DataFrame:
    """
    Aggregates transaction data to a daily level for selected products.
    """
    product_daily = (
        lazy_df
        .filter(pl.col("PROD_CODE").is_in(product_ids))
        .with_columns(
            (pl.col("SPEND") / pl.col("QUANTITY")).alias("PRICE")
        )
        .group_by(["SHOP_DATE", "PROD_CODE"])
        .agg([
            pl.mean("PRICE").alias("avg_price"),
            pl.sum("QUANTITY").alias("total_units"),
            pl.sum("SPEND").alias("total_sales")
        ])
        .sort(["SHOP_DATE", "PROD_CODE"])
        .collect(streaming=True) # Use streaming for potentially large results
    )
    return product_daily

def generate_time_series_features(df: pl.DataFrame) -> pl.DataFrame:
    """
    Generates time-series features for the daily aggregated product data.
    """
    df_with_features = df.with_columns([
        # Time-based features
        pl.col("SHOP_DATE").dt.weekday().alias("day_of_week"),
        pl.col("SHOP_DATE").dt.month().alias("month"),
        pl.col("SHOP_DATE").dt.year().alias("year"),
        pl.col("SHOP_DATE").dt.day().alias("day"),
        (pl.col("SHOP_DATE").dt.weekday().is_in([6, 7])).alias("is_weekend"),
        
        # Lag features for total_units
        pl.col("total_units").shift(1).over("PROD_CODE").alias("lag_1_units"),
        pl.col("total_units").shift(7).over("PROD_CODE").alias("lag_7_units"),
        pl.col("total_units").shift(30).over("PROD_CODE").alias("lag_30_units"),
        
        # Rolling mean features for total_units
        pl.col("total_units").rolling_mean(window_size=7).over("PROD_CODE").alias("rolling_mean_7_units"),
        pl.col("total_units").rolling_mean(window_size=30).over("PROD_CODE").alias("rolling_mean_30_units"),
        
        # Price change percentage
        (pl.col("avg_price") / pl.col("avg_price").shift(1).over("PROD_CODE") - 1).alias("price_change_pct"),
    ])
    return df_with_features


