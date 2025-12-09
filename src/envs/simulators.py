import numpy as np
import joblib
import os
import pandas as pd # Import pandas

class ParametricDemandSimulator:
    """
    Simulates demand using a parametric log-linear elasticity model for multiple products.
    demand_t = base_t * exp(beta_price * (price_t - ref_price) + noise)
    """
    def __init__(self, beta_price: dict, noise_std: dict, base_demand: dict, ref_price: dict, random_generator: np.random.Generator):
        self.beta_price_map = beta_price
        self.noise_std_map = noise_std
        self.base_demand_map = base_demand
        self.ref_price_map = ref_price
        self.random_generator = random_generator

    def simulate_demand(self, product_id: str, current_price: float, current_ref_price: float = None) -> float:
        """
        Simulates the units sold given the current price for a specific product.
        current_ref_price can be used to represent the historical price or a baseline.
        """
        # Look up parameters for the given product_id
        beta_price = self.beta_price_map[product_id]
        noise_std = self.noise_std_map[product_id]
        base_demand = self.base_demand_map[product_id]
        ref_price = self.ref_price_map[product_id]

        if current_ref_price is None:
            current_ref_price = ref_price

        # Calculate the price effect
        price_effect = beta_price * (current_price - current_ref_price)

        # Add noise
        noise = self.random_generator.normal(0, noise_std)

        # Calculate demand
        demand = base_demand * np.exp(price_effect + noise)

        # Ensure demand is non-negative
        return max(0, demand)

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
        self.feature_names = feature_names # Store feature names
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
