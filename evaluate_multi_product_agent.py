import argparse
import yaml
import os
import polars as pl
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logging

from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from src.utils import make_multi_product_env
from src.data_utils import load_data_registry
from src.envs.simulators import MLDemandSimulator

# This is a placeholder for get_agent_class from train_agent.py
# In a real implementation, this might be moved to a shared utils file.
import importlib
def get_agent_class(agent_name: str):
    module = importlib.import_module("stable_baselines3")
    return getattr(module, agent_name.upper())

def calculate_price_volatility(prices: list) -> float:
    """Calculates the volatility of a sequence of prices."""
    if len(prices) < 2:
        return 0.0
    price_changes = np.diff(prices)
    return np.std(price_changes)

def calculate_historical_baseline(
    scaled_product_df: pl.DataFrame,
    raw_product_df: pl.DataFrame,
    episode_horizon: int,
    cost_ratio: float,
    simulator: MLDemandSimulator,
    avg_price_scaler,
    all_feature_cols: list
) -> tuple[float, float]:
    """
    Calculates the 'Historical Price' baseline profit and volatility using an ML demand simulator.
    This baseline uses the actual historical prices to simulate what the profit would have been.
    """
    # Ensure we have enough data
    if len(raw_product_df) < episode_horizon or len(scaled_product_df) < episode_horizon:
        return 0.0, 0.0

    total_profit = 0
    prices = []

    sim_df_scaled = scaled_product_df.head(episode_horizon)
    sim_df_raw = raw_product_df.head(episode_horizon)
    
    price_feature_index = all_feature_cols.index("avg_price")

    for i in range(episode_horizon):
        raw_row = sim_df_raw.row(i, named=True)
        scaled_row = sim_df_scaled.row(i, named=True)

        # Use the actual historical price from the raw data
        historical_price = raw_row['avg_price']
        prices.append(historical_price)

        # --- Feature vector preparation for ML model ---
        # The feature vector from scaled_row contains scaled values.
        # The demand model expects unscaled features, especially price.
        current_features = [scaled_row[col] for col in all_feature_cols]
        feature_vector_for_demand_model = np.array(current_features, dtype=np.float32)

        # Unscale the avg_price feature for the demand model
        # The avg_price is at price_feature_index
        scaled_price = feature_vector_for_demand_model[price_feature_index]
        # Reshape for scaler.inverse_transform, which expects 2D array
        unscaled_price = avg_price_scaler.inverse_transform(np.array(scaled_price).reshape(1, -1))[0][0]
        
        # Replace the scaled price with the unscaled price
        feature_vector_for_demand_model[price_feature_index] = unscaled_price

        # Simulate demand using the ML model
        units_sold = simulator.simulate_demand(feature_vector_for_demand_model)
        
        units_sold = round(max(0, units_sold))

        # Profit calculation
        fixed_cost_per_unit = historical_price * cost_ratio
        profit_margin_per_unit = historical_price - fixed_cost_per_unit
        step_profit = profit_margin_per_unit * units_sold
        total_profit += step_profit
        
    volatility = calculate_price_volatility(prices)

    return total_profit, volatility

def evaluate(agent_path: str, episodes: int, use_pre_aggregated_data: bool, pre_aggregated_data_path: str, log_product_ids: list):
    """
    Evaluates a trained multi-product agent on all products in the test set.
    """
    # --- 1. Load Configurations and Data ---
    run_dir = os.path.dirname(agent_path)
    config_file_in_run_dir = os.path.join(run_dir, "config.yaml")

    if not os.path.exists(config_file_in_run_dir):
        # If config.yaml is not in the same directory, check the parent directory.
        # This handles the case where the model is saved in a 'best_model' subfolder.
        run_dir = os.path.dirname(run_dir)
        config_file_in_run_dir = os.path.join(run_dir, "config.yaml")

    if not os.path.exists(config_file_in_run_dir):
        raise FileNotFoundError(f"Config file not found in agent's run directory or its parent: {config_file_in_run_dir}")

    with open(config_file_in_run_dir, 'r') as f:
        config = yaml.safe_load(f)

    if use_pre_aggregated_data:
        raw_data_df = pl.read_parquet(pre_aggregated_data_path)
    else:
        # Load raw data directly from the configured path
        if 'raw_data' not in config['paths']:
            raise KeyError("config['paths']['raw_data'] is not defined. Cannot load raw data.")
        raw_data_df = pl.read_parquet(config['paths']['raw_data'])

    historical_median_prices = raw_data_df.group_by("PROD_CODE").agg(
        pl.median("avg_price").alias("historical_median_price")
    ).to_pandas().set_index("PROD_CODE")['historical_median_price'].to_dict()
    
    historical_sales_volatility = raw_data_df.group_by("PROD_CODE").agg(
        pl.std("total_units").alias("sales_volatility")
    ).to_pandas().set_index("PROD_CODE")['sales_volatility'].to_dict()

    registry_path = os.path.join(run_dir, 'product_registry')
    data_path = os.path.join(config['paths']['processed_data_dir'], "test_scaled.parquet")

    data_registry, product_mapper, avg_daily_revenue_registry = load_data_registry(
        data_path=data_path,
        output_path=registry_path
    )
    
    # --- 2. Load the trained agent ---
    agent_name = config['training']['agent']
    agent_class = get_agent_class(agent_name)
    # The path passed to the load function should not have the .zip extension.
    if agent_path.endswith('.zip'):
        agent_path = agent_path[:-4]
    # The stable-baselines3 load function adds the .zip extension automatically.
    agent = agent_class.load(agent_path)

    # Create a dummy VecEnv to load the normalizer stats
    env_fn = lambda: make_multi_product_env(
        data_registry=data_registry, 
        product_mapper=product_mapper, 
        avg_daily_revenue_registry=avg_daily_revenue_registry, 
        config=config, 
        raw_data_df=raw_data_df, 
        historical_avg_prices=historical_median_prices
    )
    dummy_env = DummyVecEnv([env_fn])

    # Load the saved VecNormalize stats
    vec_normalize_path = os.path.join(run_dir, "vecnormalize.pkl")
    if not os.path.exists(vec_normalize_path):
        raise FileNotFoundError(f"VecNormalize stats not found at {vec_normalize_path}. Cannot evaluate a normalized model without them.")

    eval_env = VecNormalize.load(vec_normalize_path, dummy_env)
    
    # Set to evaluation mode
    eval_env.training = False
    # Don't normalize rewards during evaluation to get true reward values
    eval_env.norm_reward = False

    # --- 3. Get necessary components from the environment for baseline calculation ---
    demand_simulator = eval_env.get_attr("demand_simulator")[0]
    avg_price_scaler = eval_env.get_attr("avg_price_scaler")[0]
    all_feature_cols = eval_env.get_attr("all_feature_cols")[0]

    all_results = []
    product_list = list(product_mapper.keys())
    
    # If log_product_ids is provided, filter the product_list to only evaluate those products
    if log_product_ids:
        product_list = [pid for pid in product_list if pid in log_product_ids]
        print(f"--- Running evaluation for a subset of {len(product_list)} products: {product_list} ---")
        
    
    print(f"--- Episode Horizon: {config['env']['episode_horizon']} ---")
    print(f"Running evaluation for {len(product_list)} products over {episodes} episodes each...")
    

    
    for i, raw_product_id in enumerate(product_list):
    
        dense_id = product_mapper[raw_product_id]
    
        agent_profits = []
    
        agent_price_sequences = []        
        print(f"\n({i+1}/{len(product_list)}) Evaluating Product: {raw_product_id}")

        for episode in range(episodes):
            unnormalized_obs_dict, _ = eval_env.env_method('reset', product_id=dense_id, sequential=True)[0]
            obs = eval_env.normalize_obs(unnormalized_obs_dict)
            
            done = False
            episode_profit = 0
            episode_prices = []
            
            while not done:
                action, _states = agent.predict(obs, deterministic=True)
                obs, reward, dones, info_list = eval_env.step([action]) 

                done = dones[0]
                episode_profit += reward[0]

                if not done:
                    episode_prices.append(info_list[0]['price'])
            
            agent_profits.append(episode_profit)
            agent_price_sequences.append(episode_prices)
        
        # --- 4. Calculate Baseline Profit ---
        raw_product_df_eval = raw_data_df.filter(pl.col("PROD_CODE") == raw_product_id)
        scaled_product_df_eval = data_registry[dense_id]
        
        if raw_product_id in log_product_ids:
            print(f"\n--- Price Log for Product: {raw_product_id} ---")
            print(f"Agent Prices (Episode 0): {[f'{p:.2f}' for p in agent_price_sequences[0]] if agent_price_sequences else 'N/A'}")
            if not raw_product_df_eval.is_empty():
                print(f"Historical Average Prices: {[f'{p:.2f}' for p in raw_product_df_eval.head(len(agent_price_sequences[0]))['avg_price'].to_list()]}")
            else:
                print(f"Historical Average Prices: N/A (raw_product_df_eval is empty)")
            print(f"--------------------------------------")
            
        # Determine the effective episode horizon based on available data
        effective_episode_horizon = min(len(scaled_product_df_eval), config['env']['episode_horizon'])

        historical_profit, historical_volatility = calculate_historical_baseline(
            scaled_product_df=scaled_product_df_eval,
            raw_product_df=raw_product_df_eval,
            episode_horizon=effective_episode_horizon,
            cost_ratio=config['env'].get('cost_ratio', 0.7),
            simulator=demand_simulator,
            avg_price_scaler=avg_price_scaler,
            all_feature_cols=all_feature_cols
        )

        avg_agent_profit = np.mean(agent_profits)
        agent_profit_std = np.std(agent_profits)
        avg_agent_volatility = np.mean([calculate_price_volatility(p) for p in agent_price_sequences])
        
        improvement_over_historical = ((avg_agent_profit - historical_profit) / abs(historical_profit)) * 100 if historical_profit != 0 else float('inf')

        all_results.append({
            "Product ID": raw_product_id,
            "Agent Profit (Mean)": avg_agent_profit,
            "Agent Profit (Std)": agent_profit_std,
            "Agent Volatility": avg_agent_volatility,
            "Historical Profit": historical_profit,
            "Historical Volatility": historical_volatility,
            "Improvement vs Historical (%)": improvement_over_historical,
            "Sales Volatility": historical_sales_volatility.get(raw_product_id, 0)
        })

    results_df = pd.DataFrame(all_results)
    
    print("\n\n--- Evaluation Summary ---")
    results_df = results_df.rename(columns={"Agent Profit": "Agent Profit (Mean)", "Agent Profit Std": "Agent Profit (Std)"})
    print(results_df.to_string(index=False, float_format="%.2f"))
    print("\n" + "-"*30)

    total_agent_profit = results_df['Agent Profit (Mean)'].sum()
    total_historical_profit = results_df['Historical Profit'].sum()
    portfolio_improvement_percentage = ((total_agent_profit - total_historical_profit) / abs(total_historical_profit)) * 100 if total_historical_profit != 0 else float('inf')

    print("\n--- Aggregate Performance ---")
    print(f"Total Agent Profit: {total_agent_profit:,.2f}")
    print(f"Total Historical Profit: {total_historical_profit:,.2f}")
    print(f"Portfolio-wide Improvement vs. Historical Baseline: {portfolio_improvement_percentage:.2f}%")
    print("--- End of Summary ---")
    
    results_csv_path = os.path.join(run_dir, "evaluation_results.csv")
    results_df.to_csv(results_csv_path, index=False)
    print(f"\nDetailed evaluation results saved to {results_csv_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a multi-product DRL agent across all products.")
    parser.add_argument("--agent-path", type=str, required=True, help="Path to the trained agent's .zip file.")
    parser.add_argument("--episodes", type=int, default=1, help="Number of episodes to run for evaluation.")
    parser.add_argument("--use-pre-aggregated-data", action="store_true", help="Use pre-aggregated data for evaluation.")
    parser.add_argument("--pre-aggregated-data-path", type=str, default="data/processed/top100_daily.parquet", help="Path to the pre-aggregated data file.")
    parser.add_argument("--log-product-ids", type=str, default="", help="Comma-separated list of product IDs to log pricing decisions for.")
    
    args = parser.parse_args()
    
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    if args.log_product_ids:
        log_product_ids = [pid.strip() for pid in args.log_product_ids.split(',')]
    else:
        log_product_ids = []
    
    evaluate(args.agent_path, args.episodes, args.use_pre_aggregated_data, args.pre_aggregated_data_path, log_product_ids)
