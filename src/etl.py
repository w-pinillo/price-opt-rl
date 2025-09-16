import polars as pl
import json
import os

def load_raw_data(path: str) -> pl.LazyFrame:
    """
    Loads the raw transactions parquet file, converts SHOP_DATE to datetime,
    and drops unnecessary columns.
    """
    lazy_df = pl.scan_parquet(path)

    lazy_df = (
        lazy_df
        .with_columns(
            pl.col("SHOP_DATE").cast(pl.Utf8).str.strptime(pl.Date, format="%Y%m%d")
        )
        .drop([
            'CUST_CODE',
            'BASKET_ID',
            'BASKET_SIZE',
            '__null_dask_index__'
        ])
    )
    return lazy_df

def save_interim(df: pl.DataFrame, output_path: str, filename: str):
    """
    Saves the interim DataFrame to a specified path.
    """
    os.makedirs(output_path, exist_ok=True)
    full_path = os.path.join(output_path, filename)
    df.write_parquet(full_path)
    print(f"Interim data saved to {full_path}")

def save_product_ids(product_ids: list, output_path: str, filename: str = "top100_ids.json"):
    """
    Saves a list of product IDs to a JSON file.
    """
    os.makedirs(output_path, exist_ok=True)
    full_path = os.path.join(output_path, filename)
    with open(full_path, 'w') as f:
        json.dump(product_ids, f)
    print(f"Product IDs saved to {full_path}")
