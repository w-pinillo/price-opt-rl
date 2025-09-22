import gymnasium as gym
from gymnasium import spaces
import numpy as np
import polars as pl
from sklearn.preprocessing import StandardScaler

from src.envs.simulators import ParametricDemandSimulator

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

    def __init__(self, data: pl.DataFrame, config: dict, avg_price_scaler: StandardScaler, render_mode=None):
        super().__init__()

        self.df = data
        self.config = config
        self.avg_price_scaler = avg_price_scaler
        
        self.action_type = self.config['env']['action_type']
        self.episode_horizon = self.config['env']['episode_horizon']

        self.feature_cols = [
            "avg_price", "total_units", "total_sales",
            "lag_1_units", "lag_7_units", "lag_30_units",
            "rolling_mean_7_units", "rolling_mean_30_units",
            "price_change_pct"
        ]
        self.time_cols = ["day_of_week", "month", "year", "day", "is_weekend"]
        
        self.demand_simulator = ParametricDemandSimulator(
            **self.config['env']['parametric_simulator']
        )

        if self.action_type == "discrete":
            self.action_space = spaces.Discrete(len(self.config['env']['discrete_action_map']))
        else:
            self.action_space = spaces.Box(
                low=self.config['env']['action_low'],
                high=self.config['env']['action_high'],
                shape=(1,), dtype=np.float32
            )

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
        units_sold = self.demand_simulator.simulate_demand(
            current_price=new_price, 
            current_ref_price=unscaled_price
        )

        # Calculate reward (revenue)
        reward = new_price * units_sold

        self.current_step += 1
        done = self.current_step >= (self.start_step + self.episode_horizon) or self.current_step >= len(self.df) -1

        observation = self._get_obs() if not done else np.zeros(self.observation_space.shape)
        info = self._get_info() if not done else {}
        
        # The gymnasium step function returns 5 values: obs, reward, terminated, truncated, info
        terminated = done 
        truncated = False # Not using time limit wrapper

        return observation, reward, terminated, truncated, info

    def render(self):
        if self.render_mode == "human":
            print(f"Step: {self.current_step}, Info: {self._get_info()}")

    def close(self):
        pass