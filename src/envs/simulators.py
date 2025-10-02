import numpy as np
import joblib
import os

class ParametricDemandSimulator:
    """
    Simulates demand using a parametric log-linear elasticity model.
    demand_t = base_t * exp(beta_price * (price_t - ref_price) + noise)
    """
    def __init__(self, beta_price: float, noise_std: float, base_demand: float, ref_price: float):
        self.beta_price = beta_price
        self.noise_std = noise_std
        self.base_demand = base_demand
        self.ref_price = ref_price

    def simulate_demand(self, current_price: float, current_ref_price: float = None) -> float:
        """
        Simulates the units sold given the current price.
        current_ref_price can be used to represent the historical price or a baseline.
        """
        if current_ref_price is None:
            current_ref_price = self.ref_price

        # Calculate the price effect
        price_effect = self.beta_price * (current_price - current_ref_price)

        # Add noise
        noise = np.random.normal(0, self.noise_std)

        # Calculate demand
        demand = self.base_demand * np.exp(price_effect + noise)

        # Ensure demand is non-negative
        return max(0, demand)

class MLDemandSimulator:
    """
    Simulates demand using a pre-trained machine learning model.
    """
    def __init__(self, model_path: str):
        print(f"Loading demand model from {model_path}...")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Demand model not found at {model_path}")
        self.model = joblib.load(model_path)
        print("Demand model loaded successfully.")

    def simulate_demand(self, features: np.ndarray) -> float:
        """
        Simulates the units sold given a feature vector.
        """
        # The ML model expects a 2D array for prediction
        if features.ndim == 1:
            features = features.reshape(1, -1)
            
        # Predict demand
        predicted_demand = self.model.predict(features)[0]

        # Ensure demand is non-negative
        return max(0, predicted_demand)
