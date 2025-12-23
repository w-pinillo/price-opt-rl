import joblib
import os
import polars as pl
import json

def load_data_registry(data_path: str, output_path: str):
    """
    Loads the main dataset, creates a product ID mapping (raw_id to dense_id),
    saves the mapping to product_registry.json, and then creates a dictionary
    where keys are dense_ids and values are the corresponding product DataFrames.
    Also calculates average daily revenue for each product.

    Returns:
        tuple: A tuple containing:
            - data_registry (dict): Keys are dense_ids, values are product DataFrames.
            - product_mapper (dict): Maps raw PROD_CODEs to dense_ids.
            - avg_daily_revenue_registry (dict): Keys are dense_ids, values are avg daily revenue.
    """
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data not found at {data_path}")

    full_df = pl.read_parquet(data_path)
    full_df = full_df.fill_nan(0)

    # Ensure 'product_id' column exists, renaming from 'PROD_CODE' if necessary.
    if "product_id" not in full_df.columns:
        if "PROD_CODE" in full_df.columns:
            full_df = full_df.rename({"PROD_CODE": "product_id"})
        else:
            raise ValueError("Dataset must contain either 'PROD_CODE' or 'product_id' column.")

    unique_product_codes = sorted(full_df["product_id"].unique().to_list()) # Changed PROD_CODE to product_id
    product_mapper = {raw_id: idx for idx, raw_id in enumerate(unique_product_codes)}

    # Save product_registry.json
    os.makedirs(output_path, exist_ok=True)
    registry_file = os.path.join(output_path, "product_registry.json")
    with open(registry_file, 'w') as f:
        json.dump(product_mapper, f, indent=4)

    data_registry = {}
    avg_daily_revenue_registry = {}

    for raw_id, dense_id in product_mapper.items():
        product_df = full_df.filter(pl.col("product_id") == raw_id) # Changed PROD_CODE to product_id
        data_registry[dense_id] = product_df

        # Calculate average daily revenue for the product
        # Unscale sales before calculating revenue
        # Assuming 'total_sales' is a scaled column, we need the scaler to unscale it.
        # For now, we will assume sales are NOT scaled and proceed. This may need adjustment.
        total_sales = product_df["total_sales"].sum()
        num_unique_days = product_df["SHOP_DATE"].n_unique()
        avg_daily_revenue = total_sales / num_unique_days if num_unique_days > 0 else 0
        avg_daily_revenue_registry[dense_id] = avg_daily_revenue

    return data_registry, product_mapper, avg_daily_revenue_registry

def load_product_registry(registry_path: str) -> dict:
    """
    Loads the product_registry.json file.
    """
    registry_file = os.path.join(registry_path, "product_registry.json")
    if not os.path.exists(registry_file):
        raise FileNotFoundError(f"Product registry not found at {registry_file}")
    with open(registry_file, 'r') as f:
        product_mapper = json.load(f)
    return product_mapper
