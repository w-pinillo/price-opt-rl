"""
Evaluation script for comparing DRL agents against baselines.
"""
import pandas as pd
import numpy as np
import torch
import yaml
import json
import joblib
import polars as pl
from stable_baselines3 import PPO, DQN
from tqdm import tqdm
import os
import importlib

from src.envs.price_env import PriceEnv
from src.utils import make_env, seed_everything

def get_agent_class(agent_name: str):
    """
    Dynamically imports and returns the agent class from stable-baselines3.
    """
    try:
        module = importlib.import_module("stable_baselines3")
        return getattr(module, agent_name)
    except (ImportError, AttributeError):
        raise ValueError(f"Agent '{agent_name}' not found in stable-baselines3.")

def backtest_policy(model, env, avg_price_scaler):
    """
    Backtests a trained model on a given environment sequentially.
    Returns total reward and a DataFrame of daily results with unscaled prices.
    """
    obs, info = env.reset(sequential=True)
    done = False
    total_reward = 0
    daily_results = []

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += reward
        if not done:
            # Unscale the price for logging
            scaled_price = info['scaled_avg_price']
            unscaled_price = avg_price_scaler.inverse_transform(np.array(scaled_price).reshape(-1, 1))[0][0]
            info['unscaled_price'] = unscaled_price
            info['reward'] = reward
            daily_results.append(info)
    
    return total_reward, pd.DataFrame(daily_results)

def do_nothing_policy(env, avg_price_scaler):
    """
    A baseline policy that always chooses the 'do nothing' action.
    Returns total reward and a DataFrame of daily results with unscaled prices.
    """
    obs, info = env.reset(sequential=True)
    done = False
    total_reward = 0
    daily_results = []
    do_nothing_action = 2 # Index for 1.0 multiplier in discrete_action_map

    while not done:
        obs, reward, terminated, truncated, info = env.step(do_nothing_action)
        done = terminated or truncated
        total_reward += reward
        if not done:
            scaled_price = info['scaled_avg_price']
            unscaled_price = avg_price_scaler.inverse_transform(np.array(scaled_price).reshape(-1, 1))[0][0]
            info['unscaled_price'] = unscaled_price
            info['reward'] = reward
            daily_results.append(info)

    return total_reward, pd.DataFrame(daily_results)

def calculate_metrics(results_df):
    """
    Calculates evaluation metrics from a results dataframe.
    """
    if results_df.empty:
        return {
            "total_revenue": 0,
            "avg_daily_revenue": 0,
            "avg_price": 0,
            "price_volatility": 0
        }

    total_revenue = results_df['reward'].sum()
    avg_daily_revenue = results_df['reward'].mean()
    avg_price = results_df['unscaled_price'].mean()
    price_volatility = results_df['unscaled_price'].std()

    return {
        "total_revenue": total_revenue,
        "avg_daily_revenue": avg_daily_revenue,
        "avg_price": avg_price,
        "price_volatility": price_volatility
    }

def main():
    """
    Main evaluation loop.
    """
    # Load config
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    # Seed for reproducibility
    seed_everything(config["training"]["seed"])

    # Load data and scalers
    test_df = pl.read_parquet(config["paths"]["test_data"])
    avg_price_scaler = joblib.load(config["paths"]["price_scaler"])
    
    product_id = config['training']['product_id']
    agent_name = config['training']['agent_name']
    agent_class = get_agent_class(agent_name)

    all_results = []

    print(f"--- Evaluating Product ID: {product_id} ---")
    env = make_env(data=test_df, config=config, product_id=product_id)
    
    # Baseline
    _, baseline_results_df = do_nothing_policy(env, avg_price_scaler)
    baseline_metrics = calculate_metrics(baseline_results_df)
    baseline_metrics['product_id'] = product_id
    baseline_metrics['agent'] = 'baseline'
    all_results.append(baseline_metrics)

    # Trained Agent
    model_path = os.path.join(config['paths']['models_dir'], agent_name.lower(), 'best_model', 'best_model.zip')
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}. Aborting.")
        return

    model = agent_class.load(model_path, env=env)
    _, agent_results_df = backtest_policy(model, env, avg_price_scaler)
    agent_metrics = calculate_metrics(agent_results_df)
    agent_metrics['product_id'] = product_id
    agent_metrics['agent'] = agent_name.lower()
    all_results.append(agent_metrics)

    # --- Save Results ---
    results_df = pd.DataFrame(all_results)
    output_path = "reports/tables"
    os.makedirs(output_path, exist_ok=True)
    results_df.to_csv(os.path.join(output_path, f"metrics_{product_id}.csv"), index=False)

    print(f"\nResults saved to {os.path.join(output_path, f'metrics_{product_id}.csv')}")

    # --- Report Aggregated Results ---
    print("\n--- Evaluation Results ---")
    for _, row in results_df.iterrows():
        print(f"Agent: {row['agent']:<10} | Total Revenue: ${row['total_revenue']:>12,.2f} | Avg Price: ${row['avg_price']:.2f}")
    print("--------------------------")

if __name__ == "__main__":
    main()
