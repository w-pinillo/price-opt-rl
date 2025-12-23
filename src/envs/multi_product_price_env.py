import gymnasium as gym
from gymnasium import spaces
import numpy as np
import polars as pl
import os
import json # Import json

from src.envs.simulators import MLDemandSimulator
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
        self.cost_ratio = self.config['env'].get('cost_ratio', 0.75) # Default to 25% profit margin

        # Pre-compute costs for all products
        self.costs = {
            prod_id: self.historical_avg_prices.get(prod_id, 0) * self.cost_ratio 
            for prod_id in self.product_mapper.keys()
        }
        
        self.product_ids = list(self.data_registry.keys())

        # --- Simulator Instantiation ---
        model_path = self.config['models']['demand_model_path']
        model_dir = os.path.dirname(model_path)
        feature_names_path = os.path.join(model_dir, "feature_names.json")
        if not os.path.exists(feature_names_path):
            raise FileNotFoundError(f"Feature names file not found at {feature_names_path}. "
                                    f"Ensure it was saved during demand model training.")
        with open(feature_names_path, 'r') as f:
            env_feature_names_list = json.load(f)

        self.all_feature_cols = env_feature_names_list # Define environment's feature order

        self.demand_simulator = MLDemandSimulator(
            model_path=model_path,
            noise_std=self.config['env']['ml_model_simulator']['noise_std'],
            env_feature_names=self.all_feature_cols, # Pass environment's feature names to simulator
            random_generator=self.np_random
        )

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

        # Get the index of 'avg_price' for later substitution
        self.price_feature_index = self.all_feature_cols.index("avg_price")

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
                shape=(len(self.all_feature_cols),), 
                dtype=np.float32
            )
        })

        self.render_mode = render_mode
        self.current_step = 0
        self.start_step = 0
        self.current_product_id = None
        self.raw_product_id = None
        self.df = None

    def _get_obs(self):
        current_row = self.df.row(self.current_step, named=True)
        obs_features = [current_row[col] for col in self.all_feature_cols]
        
        return {
            "product_id": self.current_product_id,
            "features": np.array(obs_features, dtype=np.float32)
        }

    def _get_info(self):
        current_row = self.df.row(self.current_step, named=True)
        cost_per_unit = self.costs.get(self.raw_product_id, 0)
        return {
            "date": current_row.get("SHOP_DATE"),
            "product_id": self.current_product_id, # This is the dense id
            "raw_product_id": self.raw_product_id,
            "scaled_avg_price": current_row.get("avg_price"),
            "cost_per_unit": cost_per_unit
        }

    def reset(self, seed=None, options=None, product_id: int = None, sequential: bool = False):
        super().reset(seed=seed)

        if product_id is not None:
            self.current_product_id = product_id
        else:
            self.current_product_id = self.np_random.choice(self.product_ids)
        
        # Get the raw product id from the dense id
        self.raw_product_id = next(key for key, value in self.product_mapper.items() if value == self.current_product_id)

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
            action_index = int(action)
            price_multiplier = self.config['env']['discrete_action_map'][action_index]
        else:
            price_multiplier = float(action[0])

        new_price = unscaled_price * price_multiplier

        # Get the current feature vector from the observation
        obs_features = self._get_obs()['features'].copy()
        
        # Scale the new price
        scaled_new_price = self.avg_price_scaler.transform(np.array([[new_price]]))[0][0]
        
        # Substitute the historical price with the new price in the feature vector
        obs_features[self.price_feature_index] = scaled_new_price
        
        # Simulate demand using the ML model
        units_sold = self.demand_simulator.simulate_demand(obs_features)
        units_sold = round(max(0, units_sold))

        # --- REWARD CALCULATION ---
        cost_per_unit = self.costs.get(self.raw_product_id, 0)
        profit = (new_price - cost_per_unit) * units_sold
        reward = profit

        self.current_step += 1
        done = self.current_step >= (self.start_step + self.episode_horizon) or self.current_step >= len(self.df) - 1

        if not done:
            observation = self._get_obs()
            info = self._get_info()
            info["units_sold"] = units_sold
            info["price"] = new_price
            info["profit"] = profit
        else:
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
