"""
This script provides the core function for evaluating a single policy
in the pricing environment for a full episode.
"""
import pandas as pd
import numpy as np

# Import type hints
from src.envs.price_env import PriceEnv
from src.baselines import BasePolicy


def evaluate_policy(policy: BasePolicy, env: PriceEnv) -> pd.DataFrame:
    """
    Runs a given policy in the environment for one full episode using sequential steps.

    Args:
        policy: The policy to evaluate (can be a DRL agent or a baseline).
        env: The Gym environment to run the evaluation in.

    Returns:
        A pandas DataFrame with the results of the evaluation (price, units, revenue).
    """
    obs, info = env.reset(sequential=True)
    done = False
    results = []

    while not done:
        # Get the raw row for model-based policies if needed
        current_raw_row = env.df.row(env.current_step, named=True)
        
        # DRL agents from stable-baselines3 have a different interface than our BasePolicy
        if isinstance(policy, BasePolicy):
            price_multiplier = policy.predict(obs=obs, info=info, current_raw_row=current_raw_row)
            
            # If env is discrete, find the closest action index to match the multiplier
            if env.action_type == "discrete":
                action_map = env.config['env']['discrete_action_map']
                action = (np.abs(np.array(action_map) - price_multiplier)).argmin()
            else:
                # Continuous env expects a numpy array
                action = np.array([price_multiplier])

        else: # Assumes a stable-baselines3 agent
            action, _ = policy.predict(obs, deterministic=True)

        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        if not done:
            # For logging, determine the actual price multiplier used in the step
            if isinstance(policy, BasePolicy):
                final_action_value = price_multiplier
            elif env.action_type == "discrete":
                final_action_value = env.config['env']['discrete_action_map'][action]
            else: # Continuous SB3 agent
                final_action_value = action[0]

            results.append({
                "date": info["date"],
                "product_id": info["product_id"],
                "price_multiplier": final_action_value,
                "price": info.get("price", 0),
                "reward": reward,
                "units_sold": info.get("units_sold", 0)
            })

    return pd.DataFrame(results)