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
from src.envs.simulators import ParametricDemandSimulator

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

def calculate_do_nothing_baseline(raw_product_df: pl.DataFrame, median_price: float, episode_horizon: int, cost_ratio: float) -> tuple[float, float]:
    """
    Calculates the 'Do-Nothing' baseline profit and volatility.
    Volatility is 0 as the price is constant.
    """
    fixed_cost_per_unit = median_price * cost_ratio
    total_units_in_horizon = raw_product_df.head(episode_horizon)['total_units'].sum()
    gross_profit = (median_price - fixed_cost_per_unit) * total_units_in_horizon
    return gross_profit, 0.0 # Profit and Volatility

def calculate_trend_based_baseline(raw_product_df: pl.DataFrame, product_id: int, episode_horizon: int, cost_ratio: float, simulator: ParametricDemandSimulator) -> tuple[float, float]:
    """
    Calculates the 'Trend-Based Heuristic' baseline profit and volatility.
    """
    df = raw_product_df.with_columns([
        pl.col("avg_price").rolling_mean(window_size=7).alias("ma_7"),
        pl.col("avg_price").rolling_mean(window_size=30).alias("ma_30")
    ]).drop_nulls()

    if len(df) < episode_horizon:
        return 0.0, 0.0

    total_profit = 0
    prices = []
    
    sim_df = df.head(episode_horizon)
    
    for row in sim_df.iter_rows(named=True):
        ma_7 = row['ma_7']
        ma_30 = row['ma_30']
        
        if ma_7 > ma_30:
            price_multiplier = 1.05
        else:
            price_multiplier = 0.95
            
        heuristic_price = row['avg_price'] * price_multiplier
        prices.append(heuristic_price)
        
        units_sold = simulator.simulate_demand(
            product_id=product_id,
            current_price=heuristic_price,
            current_ref_price=row['avg_price']
        )
        units_sold = round(max(0, units_sold))

        fixed_cost_per_unit = row['avg_price'] * cost_ratio
        total_profit += (heuristic_price - fixed_cost_per_unit) * units_sold
        
    volatility = calculate_price_volatility(prices)
    return total_profit, volatility

def generate_scatter_plot(results_df: pd.DataFrame, output_dir: str):
    """Generates and saves a scatter plot of improvement vs. sales volatility."""
    plt.figure(figsize=(10, 6))
    plt.scatter(results_df['Sales Volatility'], results_df['Improvement vs Do-Nothing (%)'], alpha=0.6)
    plt.title('Agent Profit Improvement vs. Historical Sales Volatility')
    plt.xlabel('Historical Sales Volatility (StdDev of Units Sold)')
    plt.ylabel('Profit Improvement vs. Do-Nothing (%)')
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
    # Handle the .zip extension automatically added by stable-baselines3
    if agent_path.endswith('.zip'):
        agent_path = agent_path[:-4]
    agent = agent_class.load(f"{agent_path}.zip")

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

    demand_simulator = eval_env.get_attr("demand_simulator")[0]

    all_results = []
    product_list = list(product_mapper.keys())
    
    print(f"Running evaluation for {len(product_list)} products over {episodes} episodes each...")

    for i, raw_product_id in enumerate(product_list):
        dense_id = product_mapper[raw_product_id]
        agent_profits = []
        agent_price_sequences = []
        
        print(f"\n({i+1}/{len(product_list)}) Evaluating Product: {raw_product_id}")

        for episode in range(episodes):
            # Use env_method to call the custom reset method on the underlying environment
            # This returns the un-normalized observation, so we must normalize it
            unnormalized_obs_dict, _ = eval_env.env_method('reset', product_id=dense_id, sequential=True)[0]
            obs = eval_env.normalize_obs(unnormalized_obs_dict)
            
            done = False
            episode_profit = 0
            episode_prices = []
            
            while not done:
                action, _states = agent.predict(obs, deterministic=True)
                # Correctly unpack 4 values from VecEnv.step()
                obs, reward, dones, info_list = eval_env.step([action]) 

                done = dones[0] # Get the done status for the single environment
                
                # Reward is a numpy array (reward for each env). We need the scalar for our single env.
                episode_profit += reward[0]

                if not done: # If the episode is not yet done
                    episode_prices.append(info_list[0]['price']) # info_list is a list of dicts, get price from the first (and only) env
            
            agent_profits.append(episode_profit)
            agent_price_sequences.append(episode_prices)
        
        raw_product_df_eval = raw_data_df.filter(pl.col("PROD_CODE") == raw_product_id)
        
        # Add logging here
        if raw_product_id in log_product_ids:
            print(f"\n--- Price Log for Product: {raw_product_id} ---")
            print(f"Agent Prices (Episode 0): {[f'{p:.2f}' for p in agent_price_sequences[0]] if agent_price_sequences else 'N/A'}")
            # Ensure raw_product_df_eval is not empty before accessing 'avg_price'
            if not raw_product_df_eval.is_empty():
                print(f"Historical Average Prices: {[f'{p:.2f}' for p in raw_product_df_eval.head(len(agent_price_sequences[0]))['avg_price'].to_list()]}")
            else:
                print(f"Historical Average Prices: N/A (raw_product_df_eval is empty)")
            print(f"--------------------------------------")
            
        median_price = historical_median_prices[raw_product_id]
        do_nothing_profit, do_nothing_volatility = calculate_do_nothing_baseline(
            raw_product_df_eval,
            median_price,
            config['env']['episode_horizon'],
            config['env'].get('cost_ratio', 0.7)
        )

        trend_based_profit, trend_based_volatility = calculate_trend_based_baseline(
            raw_product_df_eval,
            dense_id,
            config['env']['episode_horizon'],
            config['env'].get('cost_ratio', 0.7),
            demand_simulator
        )

        avg_agent_profit = np.mean(agent_profits)
        avg_agent_volatility = np.mean([calculate_price_volatility(p) for p in agent_price_sequences])
        
        improvement_over_do_nothing = ((avg_agent_profit - do_nothing_profit) / abs(do_nothing_profit)) * 100 if do_nothing_profit != 0 else float('inf')
        improvement_over_trend = ((avg_agent_profit - trend_based_profit) / abs(trend_based_profit)) * 100 if trend_based_profit != 0 else float('inf')

        all_results.append({
            "Product ID": raw_product_id,
            "Agent Profit": avg_agent_profit,
            "Agent Volatility": avg_agent_volatility,
            "Do-Nothing Profit": do_nothing_profit,
            "Do-Nothing Volatility": do_nothing_volatility,
            "Trend-Based Profit": trend_based_profit,
            "Trend-Based Volatility": trend_based_volatility,
            "Improvement vs Do-Nothing (%)": improvement_over_do_nothing,
            "Improvement vs Trend (%)": improvement_over_trend,
            "Sales Volatility": historical_sales_volatility.get(raw_product_id, 0)
        })

    results_df = pd.DataFrame(all_results)
    
    print("\n\n--- Evaluation Summary ---")
    print(results_df.to_string(index=False, float_format="%.2f"))
    print("\n" + "-"*30)

    avg_improvement_do_nothing = results_df[results_df['Improvement vs Do-Nothing (%)'] != np.inf]['Improvement vs Do-Nothing (%)'].mean()
    avg_improvement_trend = results_df[results_df['Improvement vs Trend (%)'] != np.inf]['Improvement vs Trend (%)'].mean()
    products_improved_do_nothing = (results_df['Improvement vs Do-Nothing (%)'] > 0).sum()
    products_improved_trend = (results_df['Improvement vs Trend (%)'] > 0).sum()
    total_products = len(results_df)

    print("\n--- Aggregate Performance ---")
    print(f"Average Improvement over 'Do-Nothing' Baseline: {avg_improvement_do_nothing:.2f}%")
    print(f"Agent outperformed 'Do-Nothing' for {products_improved_do_nothing} of {total_products} products ({products_improved_do_nothing/total_products:.1%})")
    print("-" * 20)
    print(f"Average Improvement over 'Trend-Based' Baseline: {avg_improvement_trend:.2f}%")
    print(f"Agent outperformed 'Trend-Based' for {products_improved_trend} of {total_products} products ({products_improved_trend/total_products:.1%})")
    print("--- End of Summary ---")
    
    # Generate and save the scatter plot
    generate_scatter_plot(results_df, run_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a multi-product DRL agent across all products.")
    parser.add_argument("--agent-path", type=str, required=True, help="Path to the trained agent's .zip file.")
    parser.add_argument("--episodes", type=int, default=1, help="Number of episodes to run per product for evaluation.")
    parser.add_argument("--use-pre-aggregated-data", action="store_true", help="Use pre-aggregated data for evaluation.")
    parser.add_argument("--pre-aggregated-data-path", type=str, default="data/processed/top100_daily.parquet", help="Path to the pre-aggregated data file.")
    parser.add_argument("--log-product-ids", type=str, default="", help="Comma-separated list of product IDs to log pricing decisions for.")
    
    args = parser.parse_args()
    
    log_product_ids = [pid.strip() for pid in args.log_product_ids.split(',')] if args.log_product_ids else []
    
    evaluate(args.agent_path, args.episodes, args.use_pre_aggregated_data, args.pre_aggregated_data_path, log_product_ids)
