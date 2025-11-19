import gymnasium as gym
from gymnasium import spaces
import numpy as np
import polars as pl
from sklearn.preprocessing import StandardScaler
import os # Added for os.path.join

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

    def __init__(self, data: pl.DataFrame, config: dict, render_mode=None): # Removed prod_category_cols
        super().__init__()

        self.df = data
        self.config = config
        
        self.action_type = self.config['env']['action_type']
        self.episode_horizon = self.config['env']['episode_horizon']

        # Define all features that were scaled during data preparation
        self.all_scaled_features = [
            "avg_price", "total_units", "total_sales",
            "day_of_week_sin", "day_of_week_cos", "month_sin", "month_cos",
            "lag_1_units", "lag_7_units", "lag_14_units", "lag_28_units",
            "rolling_mean_7_units", "rolling_mean_28_units",
            "rolling_std_7_units", "rolling_std_28_units",
            "price_change_pct",
            "day_of_month", "week_of_year", "is_weekend",
            "days_since_price_change", "price_position",
            "SHOP_WEEK"
        ]
        # Removed self.all_scaled_features.extend(prod_category_cols)

        # Load all scalers
        self.scalers = load_scalers(self.config['paths']['scalers_dir'], self.all_scaled_features)
        self.avg_price_scaler = self.scalers['avg_price'] # Get avg_price_scaler from loaded scalers

        # Features for the observation space (used by the agent)
        # This list should be carefully chosen to represent the agent's state
        self.feature_cols = [
            "avg_price", "total_units", "total_sales",
            "lag_1_units", "lag_7_units", "lag_14_units", "lag_28_units",
            "rolling_mean_7_units", "rolling_mean_28_units",
            "rolling_std_7_units", "rolling_std_28_units",
            "price_change_pct", "days_since_price_change", "price_position"
        ]
        # Removed self.feature_cols.extend(prod_category_cols)
        self.time_cols = ["day_of_week", "month", "year", "day_of_month", "week_of_year", "is_weekend"]
        
        # Features for the ML demand model (must match training features exactly)
        # Dynamically construct this list based on all_scaled_features and exclusions
        ml_model_exclusions = [
            "SHOP_DATE", "PROD_CODE", "total_units", "total_sales", 
            "day_of_week", "month", "year", "day", "is_weekend"
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
            self.demand_simulator = MLDemandSimulator(
                model_path=model_path,
                noise_std=self.config['env']['ml_model_simulator']['noise_std'],
                feature_names=self.feature_cols_ml_model,
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

        # Adjusted observation_space shape
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, 
            shape=(len(self.feature_cols) + len(self.time_cols),), 
            dtype=np.float32
        )

        self.render_mode = render_mode
        self.current_step = 0
        self.start_step = 0

    def _get_obs(self):
        current_row = self.df.row(self.current_step, named=True)
        obs_features = [current_row[col] for col in self.feature_cols + self.time_cols]
        return np.array(obs_features, dtype=np.float32)

    def _get_info(self):
        current_row = self.df.row(self.current_step, named=True)
        return {
            "date": current_row.get("SHOP_DATE"),
            "product_id": current_row.get("PROD_CODE"),
            "scaled_avg_price": current_row.get("avg_price"),
        }

    def reset(self, seed=None, options=None, sequential: bool = False):
        super().reset(seed=seed)

        if sequential:
            self.start_step = 0
        else:
            max_start_step = len(self.df) - self.episode_horizon - 1
            self.start_step = self.np_random.integers(0, max_start_step + 1) if max_start_step > 0 else 0
        
        self.current_step = self.start_step

        observation = self._get_obs()
        info = self._get_info()
        return observation, info

    def step(self, action):
        current_row = self.df.row(self.current_step, named=True)
        
        # Inverse transform the current scaled avg_price to get the real price
        scaled_price = np.array(current_row["avg_price"]).reshape(-1, 1)
        unscaled_price = self.avg_price_scaler.inverse_transform(scaled_price)[0][0]

        if self.action_type == "discrete":
            price_multiplier = self.config['env']['discrete_action_map'][action]
        else:
            price_multiplier = action[0]

        new_price = unscaled_price * price_multiplier

        # Simulate demand (units sold) based on the new price
        if self.config['env']['demand_simulator_approach'] == "parametric":
            units_sold = self.demand_simulator.simulate_demand(
                current_price=new_price, 
                current_ref_price=unscaled_price
            )
        elif self.config['env']['demand_simulator_approach'] == "ml_model":
            # Construct feature vector for ML model
            # The ML model expects features in the order defined by self.feature_cols_ml_model
            # All features must be scaled before passing to the ML model.

            # Create a Polars DataFrame from the current row for scaling
            # Ensure all_scaled_features are present in the row for proper scaling
            row_data_for_scaling = {col: [current_row[col]] for col in self.all_scaled_features if col in current_row}
            # Add SHOP_DATE and PROD_CODE if they are needed for any internal logic, but not for ML model input
            if "SHOP_DATE" in current_row:
                row_data_for_scaling["SHOP_DATE"] = [current_row["SHOP_DATE"]]
            if "PROD_CODE" in current_row:
                row_data_for_scaling["PROD_CODE"] = [current_row["PROD_CODE"]]

            current_df_for_scaling = pl.DataFrame(row_data_for_scaling)
            
            # Apply all scalers to the current row's features
            scaled_df_for_ml = apply_scalers(current_df_for_scaling, self.scalers, self.all_scaled_features)

            # Scale the new_price using the avg_price_scaler
            scaled_new_price = self.avg_price_scaler.transform(np.array(new_price).reshape(-1, 1))[0][0]
            
            # Update the scaled avg_price in the DataFrame with the new action's price
            scaled_df_for_ml = scaled_df_for_ml.with_columns(pl.Series(name="avg_price", values=[scaled_new_price]))

            # Extract features in the order expected by the ML model
            # Ensure that all columns in self.feature_cols_ml_model are present in scaled_df_for_ml
            ml_features = np.array([scaled_df_for_ml[col].item() for col in self.feature_cols_ml_model], dtype=np.float32)
            
            units_sold = self.demand_simulator.simulate_demand(features=ml_features)
        else:
            raise ValueError(f"Unknown demand_simulator_approach: {self.config['env']['demand_simulator_approach']}")


        # Calculate reward (revenue)
        reward = new_price * units_sold

        self.current_step += 1
        done = self.current_step >= (self.start_step + self.episode_horizon) or self.current_step >= len(self.df) -1

        observation = self._get_obs() if not done else np.zeros(self.observation_space.shape)
        info = self._get_info() if not done else {}
        if not done:
            info["units_sold"] = units_sold # Add units_sold to info dictionary
        
        # The gymnasium step function returns 5 values: obs, reward, terminated, truncated, info
        terminated = done 
        truncated = False # Not using time limit wrapper

        return observation, reward, terminated, truncated, info

    def render(self):
        if self.render_mode == "human":
            print(f"Step: {self.current_step}, Info: {self._get_info()}")

    def close(self):
        pass