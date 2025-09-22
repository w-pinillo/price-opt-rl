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

from src.envs.price_env import PriceEnv
from src.utils import make_env, seed_everything

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

    # Load data, scalers, and product IDs
    test_df = pl.read_parquet(config["paths"]["test_data"])
    avg_price_scaler = joblib.load(config["paths"]["price_scaler"])
    with open(config["paths"]["top_100_ids"], 'r') as f:
        top_100_ids = json.load(f)

    all_results = []

    for product_id in tqdm(top_100_ids, desc="Evaluating Products"):
        env = make_env(data=test_df, config=config, product_id=product_id)
        
        # Baseline
        _, baseline_results_df = do_nothing_policy(env, avg_price_scaler)
        baseline_metrics = calculate_metrics(baseline_results_df)
        baseline_metrics['product_id'] = product_id
        baseline_metrics['agent'] = 'baseline'
        all_results.append(baseline_metrics)

        # DQN
        dqn_model = DQN.load(config["paths"]["dqn_model"], env=env)
        _, dqn_results_df = backtest_policy(dqn_model, env, avg_price_scaler)
        dqn_metrics = calculate_metrics(dqn_results_df)
        dqn_metrics['product_id'] = product_id
        dqn_metrics['agent'] = 'dqn'
        all_results.append(dqn_metrics)

        # PPO
        ppo_model = PPO.load(config["paths"]["ppo_model"], env=env)
        _, ppo_results_df = backtest_policy(ppo_model, env, avg_price_scaler)
        ppo_metrics = calculate_metrics(ppo_results_df)
        ppo_metrics['product_id'] = product_id
        ppo_metrics['agent'] = 'ppo'
        all_results.append(ppo_metrics)

    # --- Save Results ---
    results_df = pd.DataFrame(all_results)
    output_path = "reports/tables"
    os.makedirs(output_path, exist_ok=True)
    results_df.to_csv(os.path.join(output_path, "metrics_summary.csv"), index=False)

    print(f"\nResults saved to {os.path.join(output_path, 'metrics_summary.csv')}")

    # --- Report Aggregated Results ---
    agg_df = results_df.groupby('agent')['total_revenue'].sum().reset_index()
    print("\n--- Aggregated Evaluation Results ---")
    for _, row in agg_df.iterrows():
        print(f"{row['agent']:<10} Total Revenue: ${row['total_revenue']:,.2f}")
    print("-------------------------------------")

if __name__ == "__main__":
    main()