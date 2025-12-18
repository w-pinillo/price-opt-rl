import numpy as np
import joblib
import os
import pandas as pd # Import pandas

class MLDemandSimulator:
    """
    Simulates demand using a pre-trained machine learning model.
    """
    def __init__(self, model_path: str, noise_std: float = 0.0, feature_names: list = None, random_generator: np.random.Generator = None):
        print(f"Loading demand model from {model_path}...")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Demand model not found at {model_path}")
        self.model = joblib.load(model_path)
        self.noise_std = noise_std
        self.feature_names = self.model.feature_name_ # Store feature names
        self.random_generator = random_generator if random_generator is not None else np.random.default_rng()
        print("Demand model loaded successfully.")

    def simulate_demand(self, features: np.ndarray) -> float:
        """
        Simulates the units sold given a feature vector.
        """
        # The ML model expects a 2D array for prediction
        if features.ndim == 1:
            features = features.reshape(1, -1)
            
        # Convert to DataFrame with feature names if available
        if self.feature_names:
            features_df = pd.DataFrame(features, columns=self.feature_names)
        else:
            features_df = pd.DataFrame(features) # Fallback if no names provided

        # Predict demand
        predicted_demand = self.model.predict(features_df)[0]

        # Add noise if noise_std is greater than 0
        if self.noise_std > 0:
            noise = self.random_generator.normal(0, self.noise_std)
            predicted_demand += noise

        # Ensure demand is non-negative
        return max(0, predicted_demand)
