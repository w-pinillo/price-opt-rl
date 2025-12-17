"""
Baseline policies for the pricing problem.
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List
import os

import numpy as np
import polars as pl
import joblib # For loading scalers
from sklearn.preprocessing import StandardScaler # Specific scaler type
import pandas as pd # Needed for HistoricalBaseline


class BasePolicy(ABC):
    """
    Base class for all policies.
    """

    def __init__(self, **kwargs):
        self.name = "Base"

    @abstractmethod
    def predict(
        self,
        obs: np.ndarray,
        info: Optional[Dict[str, Any]] = None,
        current_raw_row: Optional[Dict[str, Any]] = None,
    ) -> float:
        """
        Predict the price multiplier for the next step.

        Args:
            obs: The current observation from the environment.
            info: A dictionary with auxiliary information from the environment.
            current_raw_row: A dictionary containing unscaled, raw data for the current step.

        Returns:
            The price multiplier.
        """
        raise NotImplementedError

    def __str__(self):
        return self.name


class HistoricalPolicy(BasePolicy):
    """
    A policy that replays the historical prices.
    The environment's reference price is the historical price, so this policy
    simply returns a multiplier of 1.0.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "Historical"

    def predict(self, obs: np.ndarray, info: Optional[Dict[str, Any]] = None, current_raw_row: Optional[Dict[str, Any]] = None) -> float:
        """
        Returns a multiplier of 1.0 to use the historical price.

        Args:
            obs: The current observation from the environment.
            info: A dictionary with auxiliary information from the environment.
            current_raw_row: A dictionary containing unscaled, raw data for the current step.

        Returns:
            A price multiplier of 1.0.
        """
        return 1.0


class RuleBasedPolicy(BasePolicy):
    """
    A rule-based policy that sets the price to the median historical price.
    The policy calculates the price multiplier needed to reach the median price
    from the current unscaled average price in the environment.
    """

    def __init__(self, historical_data_path: str, price_scaler_path: str, **kwargs):
        super().__init__(**kwargs)
        self.name = "RuleBased (Median)"

        # Load historical data and calculate median price
        historical_df = pl.read_parquet(historical_data_path)
        self.median_historical_price = historical_df["avg_price"].median()

        # Load the price scaler
        self.price_scaler: StandardScaler = joblib.load(price_scaler_path)

    def predict(
        self,
        obs: np.ndarray,
        info: Optional[Dict[str, Any]] = None,
        current_raw_row: Optional[Dict[str, Any]] = None,
    ) -> float:
        """
        Predicts the price multiplier to set the price to the median historical price.

        Args:
            obs: The current observation from the environment (not directly used by this policy).
            info: A dictionary with auxiliary information from the environment,
                  expected to contain 'scaled_avg_price'.
            current_raw_row: Not used by this policy.

        Returns:
            The price multiplier to achieve the median historical price.
        """
        if info is None or 'scaled_avg_price' not in info:
            raise ValueError("Info dictionary must contain 'scaled_avg_price' for RuleBasedPolicy.")

        scaled_current_avg_price = np.array(info['scaled_avg_price']).reshape(-1, 1)
        unscaled_current_avg_price = self.price_scaler.inverse_transform(scaled_current_avg_price)[0][0]

        # Calculate the multiplier to reach the median historical price
        if unscaled_current_avg_price == 0:
            return 1.0
        
        price_multiplier = self.median_historical_price / unscaled_current_avg_price
        return float(price_multiplier)


class HistoricalBaseline:
    """
    A baseline policy that replays historical prices as actions.

    This class performs a "Simulated History" evaluation by converting
    historical prices into the normalized action space expected by the
    Reinforcement Learning environment. It is designed for multi-product
    scenarios where the product and time step are specified for each action.
    """

    def __init__(
        self,
        historical_data: pd.DataFrame,
        product_id_col: str,
        time_step_col: str,
        price_col: str,
        env: Any,
    ):
        """
        Initializes the HistoricalBaseline.

        Args:
            historical_data: DataFrame with historical product data.
            product_id_col: Name of the column for product identifiers.
            time_step_col: Name of the column for the date or time step.
            price_col: Name of the column for the historical price.
            env: The environment object, used to access price and action
                 space constraints (e.g., env.min_price, env.max_price,
                 and env.action_space).
        """
        if historical_data.empty:
            raise ValueError("historical_data DataFrame cannot be empty.")

        self.price_col = price_col

        # Set up a multi-index for efficient lookups
        self.data = historical_data.set_index([product_id_col, time_step_col])
        if not self.data.index.is_unique:
            # If duplicates exist for a (product, step) pair, average them.
            self.data = self.data.groupby(level=[0, 1]).mean()
        
        # Get price and action space boundaries from the environment
        try:
            # Assuming a continuous Box action space, common in pricing
            self.min_price = env.min_price
            self.max_price = env.max_price
            self.action_low = env.action_space.low[0]
            self.action_high = env.action_space.high[0]
        except (AttributeError, TypeError, IndexError) as e:
            raise ValueError(
                "The environment object `env` must expose `min_price`, `max_price`, "
                "and a continuous `action_space` with `low` and `high` attributes."
            ) from e

        self.price_range = self.max_price - self.min_price
        if self.price_range <= 0:
            raise ValueError("max_price must be greater than min_price in the environment.")
        
        self.action_range = self.action_high - self.action_low

    def get_action(self, state: Any, current_step: Any, product_id: Any) -> float:
        """
        Calculates the normalized action that corresponds to the historical price.

        1.  Looks up the historical price for the given product and step.
        2.  If missing, returns a neutral 'hold' action (0.0).
        3.  Normalizes the price into the environment's action space.
        4.  Clips the result to ensure it's within the valid action bounds.

        Args:
            state: The current state from the environment (not used by this baseline).
            current_step: The current time step or date to look up.
            product_id: The product ID for which to get the action.

        Returns:
            The calculated and clipped normalized action as a float.
        """
        try:
            # 1. Lookup: Find the historical price for the given product and step
            historical_price = self.data.loc[(product_id, current_step), self.price_col]
        except KeyError:
            # 2. Edge Case: If data is missing, default to a neutral "Hold" action
            # A neutral action is typically 0 in a [-1, 1] space.
            return 0.0

        # 3. Normalization: Reverse-engineer the action from the historical price.
        # This formula first scales the price to the [0, 1] range.
        # It represents how far along the price is from min_price to max_price.
        scaled_price = (historical_price - self.min_price) / self.price_range

        # Then, map this [0, 1] value to the environment's action space (e.g., [-1, 1]).
        action = self.action_low + scaled_price * self.action_range

        # 4. Clipping: Ensure the action is within valid bounds. This handles cases
        # where historical prices were outside the env's current min/max range.
        clipped_action = np.clip(action, self.action_low, self.action_high)

        return float(clipped_action)
