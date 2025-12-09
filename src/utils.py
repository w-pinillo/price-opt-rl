import joblib
import os
import polars as pl
from sklearn.preprocessing import StandardScaler, RobustScaler
import random
import numpy as np
import torch
import gymnasium as gym

def fit_save_scalers(df: pl.DataFrame, feature_cols: list, output_path: str, scaler_type: str = "StandardScaler"):
    """
    Fits scalers on the provided DataFrame for the specified feature columns
    and saves them to the output path.
    """
    os.makedirs(output_path, exist_ok=True)
    scalers = {}
    for col in feature_cols:
        if scaler_type == "StandardScaler":
            scaler = StandardScaler()
        elif scaler_type == "RobustScaler":
            scaler = RobustScaler()
        else:
            raise ValueError(f"Unsupported scaler type: {scaler_type}")
        
        scaled_data = scaler.fit_transform(df[col].to_numpy().reshape(-1, 1))
        scalers[col] = scaler
        
        scaler_filename = os.path.join(output_path, f"{col}_{scaler_type.lower()}.joblib")
        joblib.dump(scaler, scaler_filename)
        print(f"Scaler for {col} saved to {scaler_filename}")
    return scalers

def load_scalers(input_path: str, feature_cols: list, scaler_type: str = "StandardScaler") -> dict:
    """
    Loads scalers for the specified feature columns from the input path.
    """
    scalers = {}
    for col in feature_cols:
        scaler_filename = os.path.join(input_path, f"{col}_{scaler_type.lower()}.joblib")
        if os.path.exists(scaler_filename):
            scalers[col] = joblib.load(scaler_filename)
        else:
            print(f"Warning: Scaler file not found for {col} at {scaler_filename}")
    return scalers

def apply_scalers(df: pl.DataFrame, scalers: dict, feature_cols: list) -> pl.DataFrame:
    """
    Applies loaded scalers to the specified feature columns in the DataFrame.
    """
    df_scaled = df.clone()
    for col in feature_cols:
        if col in scalers:
            scaled_data = scalers[col].transform(df_scaled[col].to_numpy().reshape(-1, 1))
            df_scaled = df_scaled.with_columns(pl.Series(name=col, values=scaled_data.flatten()))
        else:
            print(f"Warning: Scaler not found for column {col}. Skipping scaling for this column.")
    return df_scaled

def seed_everything(seed: int):
    """
    Sets the seed for major libraries for reproducibility.
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def make_multi_product_env(data_registry: dict, product_mapper: dict, avg_daily_revenue_registry: dict, config: dict, raw_data_df: pl.DataFrame, historical_avg_prices: dict):
    """
    Creates, configures, and returns the multi-product pricing environment.
    """
    # Import MultiProductPriceEnv locally to avoid circular dependency
    from src.envs.multi_product_price_env import MultiProductPriceEnv

    env = MultiProductPriceEnv(
        data_registry=data_registry,
        raw_data_df=raw_data_df,
        product_mapper=product_mapper,
        config=config,
        historical_avg_prices=historical_avg_prices,
        avg_daily_revenue_registry=avg_daily_revenue_registry
    )
    return env