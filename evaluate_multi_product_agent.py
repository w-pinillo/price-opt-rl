import argparse
import yaml
import os
import polars as pl
import numpy as np
import pandas as pd # Import pandas for table display
import matplotlib.pyplot as plt

from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from src.utils import make_multi_product_env
from src.data_utils import load_data_registry, load_product_registry
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

def calculate_trend_based_baseline(
    scaled_product_df: pl.DataFrame, 
    raw_product_df: pl.DataFrame, 
    episode_horizon: int, 
    cost_ratio: float, 
    simulator: MLDemandSimulator, 
    avg_price_scaler, 
    all_feature_cols: list
) -> tuple[float, float]:
    """
    Calculates the 'Trend-Based Heuristic' baseline profit and volatility using an ML demand simulator.
    """
    # Calculate moving averages on the raw, unscaled prices
    raw_df_ma = raw_product_df.with_columns([
        pl.col("avg_price").rolling_mean(window_size=7).alias("ma_7"),
        pl.col("avg_price").rolling_mean(window_size=30).alias("ma_30")
    ]).drop_nulls()

    # Ensure we have enough data
    if len(raw_df_ma) < episode_horizon or len(scaled_product_df) < episode_horizon:
        return 0.0, 0.0

    total_profit = 0
    prices = []
    
    sim_df_scaled = scaled_product_df.head(episode_horizon)
    sim_df_raw = raw_df_ma.head(episode_horizon)

    price_feature_index = all_feature_cols.index("avg_price")

    for i in range(episode_horizon):
        raw_row = sim_df_raw.row(i, named=True)
        scaled_row = sim_df_scaled.row(i, named=True)

        ma_7 = raw_row['ma_7']
        ma_30 = raw_row['ma_30']
        
        if ma_7 > ma_30:
            price_multiplier = 1.05
        elif ma_7 < ma_30:
            price_multiplier = 0.95
        else:
            price_multiplier = 1.00
        
        # Heuristic price is based on the unscaled historical price
        heuristic_price = raw_row['avg_price'] * price_multiplier
        prices.append(heuristic_price)

        # --- Feature vector preparation for ML model ---
        # 1. Get the scaled feature vector for the current step
        feature_vector = np.array([scaled_row[col] for col in all_feature_cols], dtype=np.float32)

        # 2. Scale the new heuristic price
        scaled_heuristic_price = avg_price_scaler.transform(np.array([[heuristic_price]]))[0][0]

        # 3. Substitute the scaled price into the feature vector
        feature_vector[price_feature_index] = scaled_heuristic_price
        
        # 4. Simulate demand using the ML model
        units_sold = simulator.simulate_demand(feature_vector)
        units_sold = round(max(0, units_sold))

        # Profit calculation
        fixed_cost_per_unit = raw_row['avg_price'] * cost_ratio
        total_profit += (heuristic_price - fixed_cost_per_unit) * units_sold
        
    volatility = calculate_price_volatility(prices)
    return total_profit, volatility

def generate_scatter_plot(results_df: pd.DataFrame, output_dir: str):
    """Generates and saves a scatter plot of improvement vs. sales volatility."""
    plt.figure(figsize=(10, 6))
    plt.scatter(results_df['Sales Volatility'], results_df['Improvement vs Trend (%)'], alpha=0.6)
    plt.title('Agent Profit Improvement vs. Historical Sales Volatility')
    plt.xlabel('Historical Sales Volatility (StdDev of Units Sold)')
    plt.ylabel('Profit Improvement vs. Trend-Based Baseline (%)')
    plt.grid(True)
    
    plot_path = os.path.join(output_dir, "improvement_vs_volatility.png")
    plt.savefig(plot_path)
    print(f"\nScatter plot saved to {plot_path}")

def evaluate(agent_path: str, episodes: int, use_pre_aggregated_data: bool, pre_aggregated_data_path: str, log_product_ids: list):
    """
    Evaluates a trained multi-product agent on all products in the test set.
    """
    # --- 1. Load Configurations and Data ---
    run_dir = os.path.dirname(agent_path)
    config_file_in_run_dir = os.path.join(run_dir, "config.yaml")

    if not os.path.exists(config_file_in_run_dir):
        raise FileNotFoundError(f"Config file not found in agent's run directory: {config_file_in_run_dir}")

    with open(config_file_in_run_dir, 'r') as f:
        config = yaml.safe_load(f)

    if use_pre_aggregated_data:
        raw_data_df = pl.read_parquet(pre_aggregated_data_path)
    else:
        # Load raw data directly from the configured path
        print("Loading raw data from config['paths']['raw_data']...")
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

    print("Loading data registries...")
    data_registry, product_mapper, avg_daily_revenue_registry = load_data_registry(
        data_path=data_path,
        output_path=registry_path
    )
    
    # --- 2. Load the trained agent ---
    agent_name = config['training']['agent']
    agent_class = get_agent_class(agent_name)
    print(f"Loading agent from {agent_path}...")
    # The path passed to the load function should not have the .zip extension.
    if agent_path.endswith('.zip'):
        agent_path = agent_path[:-4]
    # The stable-baselines3 load function adds the .zip extension automatically.
    agent = agent_class.load(agent_path)

    print("Creating evaluation environment...")
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

    print(f"Loading VecNormalize stats from {vec_normalize_path} and wrapping the environment.")
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
            
        trend_based_profit, trend_based_volatility = calculate_trend_based_baseline(
            scaled_product_df=scaled_product_df_eval,
            raw_product_df=raw_product_df_eval,
            episode_horizon=config['env']['episode_horizon'],
            cost_ratio=config['env'].get('cost_ratio', 0.7),
            simulator=demand_simulator,
            avg_price_scaler=avg_price_scaler,
            all_feature_cols=all_feature_cols
        )

        avg_agent_profit = np.mean(agent_profits)
        agent_profit_std = np.std(agent_profits)
        avg_agent_volatility = np.mean([calculate_price_volatility(p) for p in agent_price_sequences])
        
        improvement_over_trend = ((avg_agent_profit - trend_based_profit) / abs(trend_based_profit)) * 100 if trend_based_profit != 0 else float('inf')

        all_results.append({
            "Product ID": raw_product_id,
            "Agent Profit (Mean)": avg_agent_profit,
            "Agent Profit (Std)": agent_profit_std,
            "Agent Volatility": avg_agent_volatility,
            "Trend-Based Profit": trend_based_profit,
            "Trend-Based Volatility": trend_based_volatility,
            "Improvement vs Trend (%)": improvement_over_trend,
            "Sales Volatility": historical_sales_volatility.get(raw_product_id, 0)
        })

    results_df = pd.DataFrame(all_results)
    
    print("\n\n--- Evaluation Summary ---")
    results_df = results_df.rename(columns={"Agent Profit": "Agent Profit (Mean)", "Agent Profit Std": "Agent Profit (Std)"})
    print(results_df.to_string(index=False, float_format="%.2f"))
    print("\n" + "-"*30)

    total_agent_profit = results_df['Agent Profit (Mean)'].sum()
    total_trend_based_profit = results_df['Trend-Based Profit'].sum()
    portfolio_improvement_percentage = ((total_agent_profit - total_trend_based_profit) / abs(total_trend_based_profit)) * 100

    print("\n--- Aggregate Performance ---")
    print(f"Total Agent Profit: {total_agent_profit:,.2f}")
    print(f"Total Trend-Based Profit: {total_trend_based_profit:,.2f}")
    print(f"Portfolio-wide Improvement vs. Trend-Based Baseline: {portfolio_improvement_percentage:.2f}%")
    print("--- End of Summary ---")
    
    generate_scatter_plot(results_df, run_dir)
    
    results_csv_path = os.path.join(run_dir, "evaluation_results.csv")
    results_df.to_csv(results_csv_path, index=False)
    print(f"\nDetailed evaluation results saved to {results_csv_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a multi-product DRL agent across all products.")
    parser.add_argument("--agent-path", type=str, required=True, help="Path to the trained agent's .zip file.")
    parser.add_argument("--use-pre-aggregated-data", action="store_true", help="Use pre-aggregated data for evaluation.")
    parser.add_argument("--pre-aggregated-data-path", type=str, default="data/processed/top100_daily.parquet", help="Path to the pre-aggregated data file.")
    parser.add_argument("--log-product-ids", type=str, default="", help="Comma-separated list of product IDs to log pricing decisions for.")
    
    args = parser.parse_args()
    
    log_product_ids = [pid.strip() for pid in args.log_product_ids.split(',')] if args.log_product_ids else []
    
    NUM_EVAL_EPISODES = 1
    
    evaluate(args.agent_path, NUM_EVAL_EPISODES, args.use_pre_aggregated_data, args.pre_aggregated_data_path, log_product_ids)