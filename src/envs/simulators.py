import numpy as np
import joblib
import os
import pandas as pd
import json

class MLDemandSimulator:
    """
    Simulates demand using a pre-trained machine learning model.
    """
    def __init__(self, model_path: str, noise_std: float = 0.0, env_feature_names: list = None, random_generator: np.random.Generator = None):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Demand model not found at {model_path}")
        self.model = joblib.load(model_path)
        self.noise_std = noise_std
        
        # Determine the directory of the model_path
        model_dir = os.path.dirname(model_path)
        feature_names_path = os.path.join(model_dir, "feature_names.json")
        if not os.path.exists(feature_names_path):
            raise FileNotFoundError(f"Feature names file not found at {feature_names_path}. "
                                    f"Ensure it was saved during demand model training.")
        with open(feature_names_path, 'r') as f:
            self.trained_feature_names = json.load(f) # Load feature names from file
        
        self.env_feature_names = env_feature_names # Store environment's feature names
        
        self.random_generator = random_generator if random_generator is not None else np.random.default_rng()

    def simulate_demand(self, features: np.ndarray) -> float:
        """
        Simulates the units sold given a feature vector.
        """
        # The ML model expects a 2D array for prediction
        if features.ndim == 1:
            features = features.reshape(1, -1)
            
        # Convert to DataFrame using the environment's feature names
        # Then reorder to match the trained model's feature names
        if self.env_feature_names:
            obs_df = pd.DataFrame(features, columns=self.env_feature_names)
            features_df = obs_df[self.trained_feature_names]
        else:
            # Fallback if env_feature_names are not provided (shouldn't happen if setup correctly)
            # This might lead to incorrect predictions if feature order doesn't match
            features_df = pd.DataFrame(features)

        # Predict demand
        predicted_demand = self.model.predict(features_df)[0]

        # Add noise if noise_std is greater than 0
        if self.noise_std > 0:
            noise = self.random_generator.normal(0, self.noise_std)
            predicted_demand += noise

        # Ensure demand is non-negative
        return max(0, predicted_demand)