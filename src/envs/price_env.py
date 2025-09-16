import gymnasium as gym
from gymnasium import spaces
import numpy as np
import polars as pl
import yaml
import os

from src.utils import load_scalers, apply_scalers
from src.envs.simulators import ParametricDemandSimulator

class PriceEnv(gym.Env):
    """
    OpenAI Gym environment for dynamic pricing.

    State: Scaled features for a given product on a given day.
    Action: Discrete price multiplier (e.g., -10%, -5%, 0%, +5%, +10%).
    Reward: Revenue generated.
    """
    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self, config_path="config.yaml", product_id=None, start_date=None, end_date=None, render_mode=None):
        super().__init__()

        self.config = self._load_config(config_path)
        self.action_type = self.config['env']['action_type']
        self.reward_formulation = self.config['env']['reward_formulation']
        self.episode_horizon = self.config['env']['episode_horizon']

        # Load processed data
        self.processed_data_dir = self.config['paths']['processed_data_dir']
        self.scalers_dir = self.config['paths']['scalers_dir']
        self.df = self._load_data(product_id, start_date, end_date)

        # Load scalers
        self.feature_cols = [
            "avg_price", "total_units", "total_sales",
            "lag_1_units", "lag_7_units", "lag_30_units",
            "rolling_mean_7_units", "rolling_mean_30_units",
            "price_change_pct"
        ]
        self.scalers = load_scalers(self.scalers_dir, self.feature_cols)

        # Initialize demand simulator
        self.demand_simulator = ParametricDemandSimulator(
            beta_price=self.config['env']['parametric_simulator']['beta_price'],
            noise_std=self.config['env']['parametric_simulator']['noise_std'],
            base_demand=self.config['env']['parametric_simulator']['base_demand'],
            ref_price=self.config['env']['parametric_simulator']['ref_price']
        )

        # Define action space
        if self.action_type == "discrete":
            self.action_space = spaces.Discrete(len(self.config['env']['discrete_action_map']))
        elif self.action_type == "continuous":
            self.action_space = spaces.Box(low=self.config['env']['action_low'],
                                           high=self.config['env']['action_high'],
                                           shape=(1,), dtype=np.float32)
        else:
            raise ValueError(f"Unknown action type: {self.action_type}")

        # Define observation space (example: 16 features from the processed data)
        # The actual shape will depend on the number of features used as state
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(len(self.feature_cols) + 5,), dtype=np.float32) # +5 for time features

        self.render_mode = render_mode
        self.current_step = 0
        self.product_id = product_id
        self.start_date = start_date
        self.end_date = end_date

    def _load_config(self, config_path):
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)

    def _load_data(self, product_id, start_date, end_date):
        # Load the full processed data
        df = pl.read_parquet(os.path.join(self.processed_data_dir, "train_scaled.parquet")) # Assuming training data for env

        if product_id:
            df = df.filter(pl.col("PROD_CODE") == product_id)
        if start_date:
            df = df.filter(pl.col("SHOP_DATE") >= start_date)
        if end_date:
            df = df.filter(pl.col("SHOP_DATE") <= end_date)
        
        # Sort by date to ensure correct time series progression
        df = df.sort("SHOP_DATE")
        return df

    def _get_obs(self):
        # Return the scaled features for the current step
        current_row = self.df.row(self.current_step, named=True)
        # Extract relevant features for the observation space
        # For now, let's use all feature_cols + time features
        obs_features = [
            current_row[col] for col in self.feature_cols
        ] + [
            current_row["day_of_week"],
            current_row["month"],
            current_row["year"],
            current_row["day"],
            current_row["is_weekend"],
        ]
        return np.array(obs_features, dtype=np.float32)

    def _get_info(self):
        current_row = self.df.row(self.current_step, named=True)
        return {
            "date": current_row["SHOP_DATE"],
            "product_id": current_row["PROD_CODE"],
            "avg_price": current_row["avg_price"], # Scaled price
            "total_units": current_row["total_units"], # Scaled units
            "total_sales": current_row["total_sales"], # Scaled sales
        }

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0 # Start from the beginning of the loaded data
        # Optionally, implement random_start=True for better generalization as per objectives.md
        # For now, always start from the beginning

        observation = self._get_obs()
        info = self._get_info()
        return observation, info

    def step(self, action):
        current_row = self.df.row(self.current_step, named=True)
        original_price = current_row["avg_price"] # This is the scaled price

        # Convert scaled price back to original scale for action application
        # This requires inverse transforming avg_price
        # For simplicity, let's assume the original price is needed for the simulator
        # and the action is applied to this original price.
        # This part needs careful implementation based on how the simulator expects price.
        # For now, let's use the scaled price and apply action multiplier.

        if self.action_type == "discrete":
            price_multiplier = self.config['env']['discrete_action_map'][action]
        elif self.action_type == "continuous":
            price_multiplier = action[0] # Action is a 1-element array
        else:
            raise ValueError("Invalid action type")

        # Apply action to the current price (scaled or unscaled, depending on simulator expectation)
        # For now, let's assume action is applied to the *unscaled* price for demand simulation
        # This means we need to inverse transform the current_row["avg_price"] first.
        # Let's get the original avg_price from the unscaled data if available, or inverse transform.
        # For simplicity, let's assume the simulator works with unscaled prices.
        # We need to load the unscaled data or inverse transform the current price.

        # For now, let's assume the simulator takes the scaled price and applies the multiplier
        # This is a simplification and needs to be refined.
        # Let's assume the simulator expects an unscaled price.
        # We need to inverse transform the current avg_price to get the real price.
        # This requires the scaler for avg_price.

        # Inverse transform the current scaled avg_price to get the real price
        avg_price_scaler = self.scalers["avg_price"]
        # The current_row["avg_price"] is a single scalar, but inverse_transform expects 2D array
        current_unscaled_avg_price = avg_price_scaler.inverse_transform(np.array(current_row["avg_price"]).reshape(-1, 1))[0][0]

        # Apply the price multiplier to the unscaled price
        new_price = current_unscaled_avg_price * price_multiplier

        # Simulate demand (units sold) based on the new price and other factors
        # The simulator needs to be designed to take relevant features.
        # For now, let's pass the new_price and some base demand.
        # The simulator should ideally take other state features as well.
        # For simplicity, let's assume base_demand is from config and beta_price is used.
        # The simulator will need to be more sophisticated.
        # For now, a very basic simulation:
        units_sold = self.demand_simulator.simulate_demand(new_price, current_unscaled_avg_price) # Simplified

        # Calculate reward (revenue)
        reward = new_price * units_sold

        self.current_step += 1
        done = self.current_step >= len(self.df) - 1 or self.current_step >= self.episode_horizon

        observation = self._get_obs() if not done else np.zeros(self.observation_space.shape) # Placeholder for terminal state
        info = self._get_info() if not done else {}

        return observation, reward, done, False, info # last False is for truncated

    def render(self):
        if self.render_mode == "human":
            print(f"Step: {self.current_step}, Obs: {self._get_obs()}, Info: {self._get_info()}")

    def close(self):
        pass
