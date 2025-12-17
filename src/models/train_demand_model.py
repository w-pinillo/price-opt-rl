import polars as pl
import lightgbm as lgb
import joblib
import os
import json # Import json
import yaml # Import yaml

def train_demand_model(data_path: str, model_output_path: str, lgbm_params: dict):
    """
    Trains a LightGBM model to predict demand (total_units) and saves it,
    along with the list of feature names.
    """
    print(f"Loading data from {data_path}...")
    train_df = pl.read_parquet(data_path)

    # Define target and features
    target_col = "total_units"
    
    # Exclude identifiers, the target itself, and other metrics that would leak information.
    # Also excluding raw time features that are now represented by sin/cos transformations.
    feature_cols = [
        col for col in train_df.columns 
        if col not in [
            "SHOP_DATE", "product_id", "PROD_CATEGORY", "total_units", "total_sales", 
            "day_of_week", "month", "year", "day", "is_weekend"
        ]
    ]
    
    print("Defining features and target...")
    print(f"Feature columns used: {feature_cols}")
    X_train = train_df[feature_cols]
    y_train = train_df[target_col]

    # Print the actual feature names that will be used for training
    print(f"Actual features used for training: {X_train.columns}")

    print(f"Feature columns being passed to LightGBM: {X_train.columns}")
    print("Training LightGBM demand model...")
    # Initialize and train the model
    lgbm = lgb.LGBMRegressor(**lgbm_params)
    lgbm.fit(X_train.to_numpy(), y_train.to_numpy())

    # Ensure the output directory exists
    output_dir = os.path.dirname(model_output_path)
    os.makedirs(output_dir, exist_ok=True)

    print(f"Saving trained model to {model_output_path}...")
    joblib.dump(lgbm, model_output_path)
    print("Model saved successfully.")

    # Save the feature names to a json file
    feature_names_path = os.path.join(output_dir, "feature_names.json")
    with open(feature_names_path, 'w') as f:
        json.dump(X_train.columns, f)
    print(f"Feature names saved to {feature_names_path}")


if __name__ == "__main__":
    # Load base config to get paths and model parameters
    with open("configs/base_config.yaml", 'r') as f:
        base_config = yaml.safe_load(f)

    DATA_PATH = base_config['paths']['processed_data_dir'] + '/train_scaled.parquet'
    MODEL_OUTPUT_PATH = base_config['models']['demand_model_path']
    LGBM_PARAMS = base_config['models']['lgbm_demand_model_params']
    
    if not os.path.exists(DATA_PATH):
        print(f"Error: Training data not found at {DATA_PATH}")
    else:
        train_demand_model(DATA_PATH, MODEL_OUTPUT_PATH, LGBM_PARAMS)
