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