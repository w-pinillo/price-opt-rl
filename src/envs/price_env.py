import gymnasium as gym
from gymnasium import spaces
import numpy as np
import polars as pl
from sklearn.preprocessing import StandardScaler
import os # Added for os.path.join
from typing import Optional
import json # Import json

from src.envs.simulators import ParametricDemandSimulator, MLDemandSimulator
from src.utils import load_scalers, apply_scalers # Import load_scalers and apply_scalers

class PriceEnv(gym.Env):
    """
    OpenAI Gym environment for dynamic pricing.

    It takes a pre-filtered DataFrame for a single product and the configuration.
    The data is expected to be scaled, but it also requires the scaler for the
    price column to de-scale it for the simulator.

    State: Scaled features for a given product on a given day.
    Action: Discrete price multiplier.
    Reward: Revenue generated.
    """
    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self, data_registry: dict, product_mapper: dict, avg_daily_revenue_registry: dict, config: dict, raw_data_df: pl.DataFrame, historical_avg_prices: dict, render_mode=None):
        super().__init__()

        self.data_registry = data_registry
        self.raw_data_df = raw_data_df # Full raw DataFrame for ground truth
        self.product_mapper = product_mapper
        self.avg_daily_revenue_registry = avg_daily_revenue_registry
        self.config = config
        self.historical_avg_prices = historical_avg_prices # For fixed cost calculation
        
        self.action_type = self.config['env']['action_type']
        self.episode_horizon = self.config['env']['episode_horizon']
        
        # Current product details, set during reset
        self.current_product_id = None 
        self.current_product_df = None # Scaled data for observation
        self.current_raw_product_df = None # Raw data for ground truth calculations

        # Load all_scaled_features from the saved JSON file
        feature_cols_path = os.path.join(self.config['paths']['scalers_dir'], "feature_cols_for_env.json")
        if not os.path.exists(feature_cols_path):
            raise FileNotFoundError(f"Feature columns definition not found at {feature_cols_path}. Please run the data pipeline first.")
        with open(feature_cols_path, 'r') as f:
            self.all_scaled_features = json.load(f)

        # Load all scalers
        self.scalers = load_scalers(self.config['paths']['scalers_dir'], self.all_scaled_features)
        self.avg_price_scaler = self.scalers['avg_price'] # Get avg_price_scaler from loaded scalers

        # Features for the observation space (used by the agent)
        # This list should be carefully chosen to represent the agent's state
        self.feature_cols = [
            "avg_price",
            "lag_1_units", "lag_7_units", "lag_14_units", "lag_28_units",
            "rolling_mean_7_units", "rolling_mean_28_units",
            "rolling_std_7_units", "rolling_std_28_units",
            "price_change_pct", "days_since_price_change", "price_position"
        ]
        
        # Dynamically add PROD_CATEGORY OHE columns to feature_cols if they exist in all_scaled_features
        for col in self.all_scaled_features:
            if col.startswith("PROD_CATEGORY_DEP_") and col not in self.feature_cols:
                self.feature_cols.append(col)

        self.time_cols = ["day_of_week", "month", "year", "day_of_month", "week_of_year", "is_weekend"]
        
        # Features for the ML demand model (must match training features exactly)
        # Dynamically construct this list based on all_scaled_features and exclusions
        ml_model_exclusions = [
            "SHOP_DATE", "product_id", "total_units", "total_sales", 
            "day_of_week", "month", "year", "day", "is_weekend", "SHOP_WEEK"
        ]
        self.feature_cols_ml_model = [
            col for col in self.all_scaled_features
            if col not in ml_model_exclusions
        ]
        # Ensure the order of features for the ML model is consistent
        # This is crucial as LightGBM expects features in the same order as training
        # We assume self.all_scaled_features maintains a consistent order.

        # Initialize demand simulator based on config
        simulator_approach = self.config['env']['demand_simulator_approach']
        if simulator_approach == "parametric":
            self.demand_simulator = ParametricDemandSimulator(
                **self.config['env']['parametric_simulator'],
                random_generator=self.np_random
            )
        elif simulator_approach == "ml_model":
            model_path = os.path.join(self.config['paths']['models_dir'], 'demand_model/lgbm_demand_model.joblib')
            # It's crucial that the feature_names passed to MLDemandSimulator are in the exact order
            # the model was trained with. We get this from the feature_names.json saved during training.
            demand_model_feature_names_path = os.path.join(self.config['paths']['models_dir'], 'demand_model/feature_names.json')
            if not os.path.exists(demand_model_feature_names_path):
                raise FileNotFoundError(f"Demand model feature names not found at {demand_model_feature_names_path}. Please train the demand model first.")
            with open(demand_model_feature_names_path, 'r') as f:
                ml_model_expected_feature_names = json.load(f)

            # Reorder self.feature_cols_ml_model to match the order expected by the trained model
            # This ensures consistency even if the order in all_scaled_features changes slightly
            # Filter ml_model_expected_feature_names to only include those that are actually in all_scaled_features
            # This handles cases where some OHE columns might not be present in every product's data
            ordered_ml_features = [col for col in ml_model_expected_feature_names if col in self.all_scaled_features]
            
            # Additional check: ensure all features in ml_model_expected_feature_names are present in ordered_ml_features,
            # or handle missing ones if they can genuinely be absent (e.g., specific OHE categories).
            # For now, we assume they should all be present in all_scaled_features.
            
            self.demand_simulator = MLDemandSimulator(
                model_path=model_path,
                noise_std=self.config['env']['ml_model_simulator']['noise_std'],
                feature_names=ordered_ml_features, # Pass the ordered and filtered feature names
                random_generator=self.np_random
            )
        else:
            raise ValueError(f"Unknown demand_simulator_approach: {simulator_approach}")

        if self.action_type == "discrete":
            self.action_space = spaces.Discrete(len(self.config['env']['discrete_action_map']))
        else:
            self.action_space = spaces.Box(
                low=self.config['env']['action_low'],
                high=self.config['env']['action_high'],
                shape=(1,), dtype=np.float32
            )

        # Adjusted observation_space shape to be a dictionary
        self.observation_space = spaces.Dict({
            "product_id": spaces.Discrete(len(self.product_mapper)), # Number of unique products
            "features": spaces.Box(
                low=-np.inf, high=np.inf, 
                shape=(len(self.feature_cols) + len(self.time_cols),), 
                dtype=np.float32
            )
        })

        self.render_mode = render_mode
        self.current_step = 0
        self.start_step = 0

    def _get_obs(self):
        current_row = self.current_product_df.row(self.current_step, named=True)
        obs_features = [current_row[col] for col in self.feature_cols if col in current_row]
        obs_time_cols = [current_row[col] for col in self.time_cols if col in current_row]
        
        # Ensure that features and time_cols are concatenated correctly.
        # It's important to maintain the order and consistency of the observation space.
        # If some time_cols are not in current_row, they will be skipped, which might alter observation space shape.
        # This logic needs to be robust. For now, assuming they are always present.
        
        return {
            "product_id": self.current_product_id,
            "features": np.array(obs_features + obs_time_cols, dtype=np.float32)
        }

    def _get_info(self):
        current_row = self.current_product_df.row(self.current_step, named=True)
        return {
            "date": current_row.get("SHOP_DATE"),
            "product_code": self.current_raw_product_id_str, # Use the stored raw product ID string
            "product_id": self.current_product_id,
            "scaled_avg_price": current_row.get("avg_price"),
        }

    def reset(self, seed=None, options=None, product_id: Optional[str] = None, sequential: bool = False):
        super().reset(seed=seed)

        if product_id is not None:
            # If a specific product_id is provided, use it
            if product_id not in self.product_mapper:
                raise ValueError(f"Product ID {product_id} not found in product_mapper.")
            self.current_raw_product_id_str = product_id
            self.current_product_id = self.product_mapper[product_id]
        else:
            # Otherwise, sample a random product
            # Get a random raw product ID string, then map to dense ID
            self.current_raw_product_id_str = self.np_random.choice(list(self.product_mapper.keys()))
            self.current_product_id = self.product_mapper[self.current_raw_product_id_str]
        
        self.current_product_df = self.data_registry[self.current_product_id]
        
        # Filter the raw_data_df to get the raw data for the current product
        self.current_raw_product_df = self.raw_data_df.filter(pl.col("PROD_CODE") == self.current_raw_product_id_str)

        if sequential:
            self.start_step = 0
        else:
            # max_start_step should be based on the shorter of the two dataframes to prevent out-of-bounds access
            # Episode horizon should not exceed the available data for a product
            max_start_step = min(len(self.current_product_df), len(self.current_raw_product_df)) - self.episode_horizon - 1
            self.start_step = self.np_random.integers(0, max_start_step + 1) if max_start_step > 0 else 0
        
        self.current_step = self.start_step

        observation = self._get_obs()
        info = self._get_info()
        return observation, info

    def step(self, action):
        current_scaled_row = self.current_product_df.row(self.current_step, named=True)
        current_raw_row = self.current_raw_product_df.row(self.current_step, named=True)
        
        # Get the true historical price directly from the raw data
        true_historical_price = current_raw_row["avg_price"]
        
        if self.action_type == "discrete":
            price_multiplier = self.config['env']['discrete_action_map'][action]
        else:
            price_multiplier = action[0]

        # Calculate the new price based on the true historical price and the agent's action
        new_price = true_historical_price * price_multiplier

        # Simulate demand (units sold) based on the new price
        if self.config['env']['demand_simulator_approach'] == "parametric":
            units_sold = self.demand_simulator.simulate_demand(
                current_price=new_price, 
                current_ref_price=true_historical_price # Reference price for elasticity is the true historical price
            )
        elif self.config['env']['demand_simulator_approach'] == "ml_model":
            # --- Prepare features for ML model ---
            # The ML model expects scaled features.
            # We need to create a temporary row with all features (including new_price scaled)
            # and then apply scalers to get the appropriate input for the ML model.

            # Get the features from the current SCALED row
            # Ensure all features in self.all_scaled_features are included
            row_data_for_ml_input = {}
            for col in self.all_scaled_features:
                # Handle cases where a feature might not be in the current row (e.g., some OHE columns for specific products)
                # For OHE columns, if they are not present, their value should be 0.
                if col.startswith("PROD_CATEGORY_DEP_"):
                    row_data_for_ml_input[col] = [current_scaled_row.get(col, 0.0)]
                else:
                    row_data_for_ml_input[col] = [current_scaled_row[col]]

            # Create a temporary Polars DataFrame from the current scaled row
            current_df_for_ml_input = pl.DataFrame(row_data_for_ml_input)
            
            # Scale the new_price using the avg_price_scaler
            scaled_new_price = self.avg_price_scaler.transform(np.array(new_price).reshape(-1, 1))[0][0]
            
            # Update the 'avg_price' feature in the temporary DataFrame with the scaled new_price
            current_df_for_ml_input = current_df_for_ml_input.with_columns(pl.Series(name="avg_price", values=[scaled_new_price]))

            # Extract features in the order expected by the ML model
            # This order is now provided by self.demand_simulator.feature_names
            ml_features = np.array([current_df_for_ml_input[col].item() for col in self.demand_simulator.feature_names], dtype=np.float32)
            
            # Simulate demand using the ML model. The ML model predicts scaled units.
            scaled_units_sold = self.demand_simulator.simulate_demand(features=ml_features)

            # Inverse scale the predicted units to get actual units sold
            # For this to work, we need a scaler for total_units. Let's assume it's available.
            # If not, this part will require careful adjustment or a new scaler.
            if 'total_units' in self.scalers:
                units_sold = self.scalers['total_units'].inverse_transform(np.array(scaled_units_sold).reshape(-1, 1))[0][0]
                units_sold = max(0, units_sold) # Ensure units sold is non-negative
            else:
                # Fallback: if no scaler for total_units, assume simulator output is already unscaled
                # This needs to be explicitly checked in src/envs/simulators.py and train_demand_model.py
                print("Warning: No scaler found for 'total_units'. Assuming ML model simulator outputs unscaled units.")
                units_sold = scaled_units_sold
            
        else:
            raise ValueError(f"Unknown demand_simulator_approach: {self.config['env']['demand_simulator_approach']}")

        # Ensure units_sold is an integer (cannot sell partial units)
        units_sold = round(units_sold)

        # Calculate reward (revenue)
        revenue = new_price * units_sold
        
        # Calculate gross profit
        fixed_cost_per_unit = self.historical_avg_prices[self.current_raw_product_id_str] * self.config['env'].get('cost_ratio', 0.7) # Assume 30% margin
        gross_profit = (new_price - fixed_cost_per_unit) * units_sold

        self.current_step += 1
        done = self.current_step >= (self.start_step + self.episode_horizon) or self.current_step >= len(self.current_product_df) -1

        observation = self._get_obs() if not done else { "product_id": self.current_product_id, "features": np.zeros(self.observation_space["features"].shape) }
        info = self._get_info() if not done else {}
        if not done:
            info["units_sold"] = units_sold
            info["price"] = new_price # Add new_price to info dictionary
            info["revenue"] = revenue
            info["gross_profit"] = gross_profit
        
        # The gymnasium step function returns 5 values: obs, reward, terminated, truncated, info
        terminated = done 
        truncated = False # Not using time limit wrapper

        # The reward should be the primary metric for the agent, which we defined as profit
        reward = gross_profit # Agent should optimize for profit now, not just revenue

        return observation, reward, terminated, truncated, info

    def render(self):
        if self.render_mode == "human":
            print(f"Step: {self.current_step}, Info: {self._get_info()}")

    def close(self):
        pass