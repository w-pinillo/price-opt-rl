import gymnasium as gym
from gymnasium import spaces
import numpy as np
import polars as pl
import os

from src.envs.simulators import ParametricDemandSimulator
from src.utils import load_scalers

class MultiProductPriceEnv(gym.Env):
    """
    OpenAI Gym environment for dynamic pricing for multiple products.

    It takes a data_registry (dict of product_id -> DataFrame), a product_mapper,
    and the configuration. At the start of each episode, it randomly samples a
    product to train on.

    State: A dictionary containing the product_id and scaled features.
    Action: Discrete price multiplier.
    Reward: Revenue generated.
    """
    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self, data_registry: dict, product_mapper: dict, config: dict, raw_data_df: pl.DataFrame, historical_avg_prices: dict, avg_daily_revenue_registry: dict, render_mode=None):
        super().__init__()

        self.data_registry = data_registry
        self.product_mapper = product_mapper
        self.config = config
        self.raw_data_df = raw_data_df
        self.historical_avg_prices = historical_avg_prices
        self.avg_daily_revenue_registry = avg_daily_revenue_registry

        self.action_type = self.config['env']['action_type']
        self.episode_horizon = self.config['env']['episode_horizon']
        
        self.product_ids = list(self.data_registry.keys())

        # Define all features that were scaled during data preparation
        self.all_scaled_features = [
            "avg_price", "total_units", "total_sales",
            "day_of_week_sin", "day_of_week_cos", "month_sin", "month_cos",
            "lag_1_units", "lag_7_units", "lag_14_units", "lag_28_units",
            "rolling_mean_7_units", "rolling_mean_28_units",
            "rolling_std_7_units", "rolling_std_28_units",
            "price_change_pct", "day_of_month", "week_of_year", "is_weekend",
            "days_since_price_change", "price_position", "SHOP_WEEK"
        ]

        self.scalers = load_scalers(self.config['paths']['scalers_dir'], self.all_scaled_features)
        self.avg_price_scaler = self.scalers['avg_price']

        self.feature_cols = [
            "avg_price", "lag_1_units", "lag_7_units", "lag_14_units", "lag_28_units",
            "rolling_mean_7_units", "rolling_mean_28_units",
            "rolling_std_7_units", "rolling_std_28_units",
            "price_change_pct", "days_since_price_change", "price_position"
        ]
        self.time_cols = ["day_of_week", "month", "year", "day_of_month", "week_of_year", "is_weekend"]

        # --- Refactored Simulator Instantiation ---
        # Create parameter maps for each product. For now, we use the same global
        # parameters for all products, but this architecture supports per-product models.
        sim_params = self.config['env']['parametric_simulator']
        beta_price_map = {pid: sim_params['beta_price'] for pid in self.product_ids}
        noise_std_map = {pid: sim_params['noise_std'] for pid in self.product_ids}
        base_demand_map = {pid: sim_params['base_demand'] for pid in self.product_ids}
        ref_price_map = {pid: sim_params['ref_price'] for pid in self.product_ids}

        self.demand_simulator = ParametricDemandSimulator(
            beta_price=beta_price_map,
            noise_std=noise_std_map,
            base_demand=base_demand_map,
            ref_price=ref_price_map,
            random_generator=self.np_random
        )

        if self.action_type == "discrete":
            self.action_space = spaces.Discrete(len(self.config['env']['discrete_action_map']))
        else:
            self.action_space = spaces.Box(
                low=self.config['env']['action_low'],
                high=self.config['env']['action_high'],
                shape=(1,), dtype=np.float32
            )

        self.observation_space = spaces.Dict({
            "product_id": spaces.Discrete(len(self.product_ids)),
            "features": spaces.Box(
                low=-np.inf, high=np.inf, 
                shape=(len(self.feature_cols) + len(self.time_cols),), 
                dtype=np.float32
            )
        })

        self.render_mode = render_mode
        self.current_step = 0
        self.start_step = 0
        self.current_product_id = None
        self.df = None

    def _get_obs(self):
        current_row = self.df.row(self.current_step, named=True)
        obs_features = [current_row[col] for col in self.feature_cols + self.time_cols]
        
        return {
            "product_id": self.current_product_id,
            "features": np.array(obs_features, dtype=np.float32)
        }

    def _get_info(self):
        current_row = self.df.row(self.current_step, named=True)
        return {
            "date": current_row.get("SHOP_DATE"),
            "product_id": self.current_product_id, # This is the dense id
            "raw_product_id": current_row.get("PROD_CODE"),
            "scaled_avg_price": current_row.get("avg_price"),
        }

    def reset(self, seed=None, options=None, product_id: int = None, sequential: bool = False):
        super().reset(seed=seed)

        if product_id is not None:
            self.current_product_id = product_id
        else:
            self.current_product_id = self.np_random.choice(self.product_ids)
        
        self.df = self.data_registry[self.current_product_id]

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
        
        scaled_price = np.array(current_row["avg_price"]).reshape(-1, 1)
        unscaled_price = self.avg_price_scaler.inverse_transform(scaled_price)[0][0]

        if self.action_type == "discrete":
            # Convert action to scalar integer as it comes as a numpy array from VecEnv
            action_index = int(action)
            price_multiplier = self.config['env']['discrete_action_map'][action_index]
        else:
            price_multiplier = action[0]

        new_price = unscaled_price * price_multiplier

        units_sold = self.demand_simulator.simulate_demand(
            product_id=self.current_product_id,
            current_price=new_price, 
            current_ref_price=unscaled_price
        )

        reward = new_price * units_sold

        self.current_step += 1
        done = self.current_step >= (self.start_step + self.episode_horizon) or self.current_step >= len(self.df) - 1

        if not done:
            observation = self._get_obs()
            info = self._get_info()
            info["units_sold"] = units_sold
            info["price"] = new_price
        else:
            # Create a zeroed-out observation for the terminal state
            obs_features_shape = self.observation_space["features"].shape
            observation = {
                "product_id": self.current_product_id,
                "features": np.zeros(obs_features_shape, dtype=np.float32)
            }
            info = {}

        terminated = done
        truncated = False

        return observation, reward, terminated, truncated, info

    def render(self):
        if self.render_mode == "human":
            print(f"Step: {self.current_step}, Info: {self._get_info()}")

    def close(self):
        pass
