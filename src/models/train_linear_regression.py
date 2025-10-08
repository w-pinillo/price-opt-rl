import polars as pl
from sklearn.linear_model import LinearRegression
import joblib
import os

def train_linear_regression_model(data_path: str, model_output_path: str):
    """
    Trains a Linear Regression model to predict demand (total_units) and saves it.
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
            "SHOP_DATE", "PROD_CODE", "total_units", "total_sales", 
            "day_of_week", "month", "year", "day", "is_weekend"
        ]
    ]
    
    print("Defining features and target...")
    print(f"Feature columns used: {feature_cols}")
    X_train = train_df[feature_cols]
    y_train = train_df[target_col]

    print("Training Linear Regression demand model...")
    # Initialize and train the model
    lr = LinearRegression()
    lr.fit(X_train.to_numpy(), y_train.to_numpy())

    # Ensure the output directory exists
    output_dir = os.path.dirname(model_output_path)
    os.makedirs(output_dir, exist_ok=True)

    print(f"Saving trained model to {model_output_path}...")
    joblib.dump(lr, model_output_path)
    print("Model saved successfully.")

if __name__ == "__main__":
    DATA_PATH = "data/processed/train_scaled.parquet"
    MODEL_OUTPUT_PATH = "models/demand_model/linear_regression_demand_model.joblib"
    
    if not os.path.exists(DATA_PATH):
        print(f"Error: Training data not found at {DATA_PATH}")
    else:
        train_linear_regression_model(DATA_PATH, MODEL_OUTPUT_PATH)
