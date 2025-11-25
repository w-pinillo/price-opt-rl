"""
Master script for running the full evaluation suite.

This script orchestrates the evaluation of trained DRL agents and baseline policies
against the test environment. It iterates through all specified products and policies,
runs backtests, aggregates the results, computes final metrics, and saves them to disk.
"""
import argparse
import yaml
import pandas as pd
import polars as pl
from pathlib import Path
from tqdm import tqdm
import os
import glob
import numpy as np
from stable_baselines3 import DQN

from src.evaluation import evaluate_policy
from src.baselines import BasePolicy, HistoricalPolicy, RuleBasedPolicy
from src.utils import make_env, seed_everything

def compute_kpis(results_df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes key performance indicators from the detailed evaluation results.

    Args:
        results_df: A DataFrame with detailed results from all evaluation runs.

    Returns:
        A summary DataFrame with KPIs aggregated by policy.
    """
    # Ensure 'date' column is datetime for unique day counting
    results_df['date'] = pd.to_datetime(results_df['date'])

    summary_kpis = results_df.groupby('policy').agg(
        total_revenue=('reward', 'sum'),
        total_units_sold=('units_sold', 'sum'),
        average_price=('price', 'mean'),
        num_unique_days=('date', lambda x: x.nunique())
    ).reset_index()

    # Calculate Average Daily Revenue
    summary_kpis['average_daily_revenue'] = summary_kpis['total_revenue'] / summary_kpis['num_unique_days']
    summary_kpis.drop(columns=['num_unique_days'], inplace=True) # Drop temporary column

    # Calculate Price Volatility
    price_volatility = results_df.groupby(['policy', 'product_id'])['price'].std().reset_index()
    avg_price_volatility = price_volatility.groupby('policy')['price'].mean().reset_index()
    avg_price_volatility.rename(columns={'price': 'price_volatility'}, inplace=True)

    # Calculate Number of Price Changes
    # Sort first to ensure correct diff calculation within each policy-product group
    results_df_sorted = results_df.sort_values(by=['policy', 'product_id', 'date'])
    
    # Use -1 to not count the first element of each product as a price change
    price_changes = results_df_sorted.groupby(['policy', 'product_id'])['price'].apply(
        lambda x: (x.diff() != 0).sum() - 1
    ).reset_index(name='num_price_changes_per_product')
    
    # Sum up price changes across all products for each policy
    total_price_changes = price_changes.groupby('policy')['num_price_changes_per_product'].sum().reset_index()
    total_price_changes.rename(columns={'num_price_changes_per_product': 'total_price_changes'}, inplace=True)

    # Merge all KPIs
    summary_df = pd.merge(summary_kpis, avg_price_volatility, on='policy')
    summary_df = pd.merge(summary_df, total_price_changes, on='policy')
    
    return summary_df


def calculate_bootstrapped_kpis(full_results_df: pd.DataFrame, n_bootstraps: int = 1000, ci_level: float = 0.95) -> pd.DataFrame:
    """
    Calculates bootstrapped confidence intervals for KPIs.

    Args:
        full_results_df: DataFrame containing results from all policy and product runs.
        n_bootstraps: Number of bootstrap samples to draw.
        ci_level: Confidence interval level (e.g., 0.95 for 95% CI).

    Returns:
        DataFrame with confidence intervals for each KPI per policy.
    """
    bootstrapped_kpis = []
    unique_products = full_results_df['product_id'].unique()
    policies = full_results_df['policy'].unique()

    for _ in tqdm(range(n_bootstraps), desc="Bootstrapping"):
        # Resample products with replacement
        resampled_products = np.random.choice(unique_products, size=len(unique_products), replace=True)
        
        # Filter results for resampled products, maintaining policy grouping
        resampled_df_list = []
        for policy in policies:
            policy_df = full_results_df[full_results_df['policy'] == policy]
            resampled_policy_df = policy_df[policy_df['product_id'].isin(resampled_products)]
            resampled_df_list.append(resampled_policy_df)
        
        resampled_full_df = pd.concat(resampled_df_list)
        
        # Calculate KPIs for the resampled data
        if not resampled_full_df.empty:
            kpis = compute_kpis(resampled_full_df)
            bootstrapped_kpis.append(kpis)

    if not bootstrapped_kpis:
        return pd.DataFrame() # Return empty DataFrame if no bootstrapped samples generated

    bootstrapped_kpis_df = pd.concat(bootstrapped_kpis, ignore_index=True)

    # Calculate CIs
    lower_bound = (1 - ci_level) / 2
    upper_bound = 1 - lower_bound

    ci_df = bootstrapped_kpis_df.groupby('policy').agg(
        total_revenue_lower=(f'total_revenue', lambda x: x.quantile(lower_bound)),
        total_revenue_upper=(f'total_revenue', lambda x: x.quantile(upper_bound)),
        total_units_sold_lower=(f'total_units_sold', lambda x: x.quantile(lower_bound)),
        total_units_sold_upper=(f'total_units_sold', lambda x: x.quantile(upper_bound)),
        average_price_lower=(f'average_price', lambda x: x.quantile(lower_bound)),
        average_price_upper=(f'average_price', lambda x: x.quantile(upper_bound)),
        average_daily_revenue_lower=(f'average_daily_revenue', lambda x: x.quantile(lower_bound)),
        average_daily_revenue_upper=(f'average_daily_revenue', lambda x: x.quantile(upper_bound)),
        price_volatility_lower=(f'price_volatility', lambda x: x.quantile(lower_bound)),
        price_volatility_upper=(f'price_volatility', lambda x: x.quantile(upper_bound)),
        total_price_changes_lower=(f'total_price_changes', lambda x: x.quantile(lower_bound)),
        total_price_changes_upper=(f'total_price_changes', lambda x: x.quantile(upper_bound)),
    ).reset_index()

    return ci_df


def run_full_evaluation(config_path: Path, experiment_name: str = "dqn_baseline", num_products: int | None = None, n_bootstraps: int = 1000):
    """
    Main function to run the full evaluation campaign.

    Args:
        config_path: Path to the base configuration file.
        experiment_name: The prefix for the experiment directory to find the DRL agent.
        num_products: Optional. If provided, limits the evaluation to this many products.
        n_bootstraps: Number of bootstrap samples for confidence interval calculation.
    """
    print("--- Starting Full Evaluation ---")

    # 1. Load configuration and data
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    print(f"Loaded configuration from {config_path}")
    
    seed_everything(config['training']['seed'])
    test_df = pl.read_parquet(config['paths']['test_data'])
    product_ids_to_eval = test_df["PROD_CODE"].unique().to_list()
    
    if num_products is not None:
        product_ids_to_eval = product_ids_to_eval[:num_products]
        print(f"Limiting evaluation to {num_products} products for testing.")

    # 2. Load DRL Agent
    print(f"Searching for DRL agent from experiment: '{experiment_name}_*'...")
    list_of_exp_dirs = glob.glob(f"models/{experiment_name}_*")
    if not list_of_exp_dirs:
        raise FileNotFoundError(f"No '{experiment_name}_*' directories found. Please train a model first.")
    
    latest_exp_dir = max(list_of_exp_dirs, key=os.path.getctime)
    drl_agent_path = Path(latest_exp_dir) / "best_model" / "best_model.zip"
    if not drl_agent_path.exists():
        drl_agent_path = Path(latest_exp_dir) / "final_model.zip"
        if not drl_agent_path.exists():
            raise FileNotFoundError(f"Neither best_model.zip nor final_model.zip found in {latest_exp_dir}")

    print(f"Loading DRL agent from: {drl_agent_path}")
    temp_env = make_env(data=test_df, config=config, product_id=product_ids_to_eval[0])
    drl_agent = DQN.load(drl_agent_path, env=temp_env)

    # 3. Instantiate Baseline Policies
    historical_policy = HistoricalPolicy()
    rule_based_policy = RuleBasedPolicy(
        historical_data_path=Path(config['paths']['processed_data_dir']) / 'train_scaled.parquet',
        price_scaler_path=config['paths']['price_scaler']
    )

    policies_to_evaluate = {
        "DQN": drl_agent,
        "Historical": historical_policy,
        "Rule-Based (Median)": rule_based_policy
    }
    print(f"Policies to evaluate: {list(policies_to_evaluate.keys())}")

    # 4. Run backtests for each product and policy
    all_results = []
    for product_id in tqdm(product_ids_to_eval, desc="Evaluating Products"):
        for policy_name, policy in policies_to_evaluate.items():
            env = make_env(data=test_df, config=config, product_id=product_id)
            
            if not isinstance(policy, BasePolicy):
                policy.set_env(env)
            
            result_df = evaluate_policy(policy, env)
            result_df["policy"] = policy_name
            all_results.append(result_df)
    
    print("All backtests complete.")

    # 5. Aggregate results and compute KPIs
    if not all_results:
        print("Warning: No results were generated. Skipping KPI calculation.")
        return
        
    full_results_df = pd.concat(all_results, ignore_index=True)

    # Save detailed results for plotting in notebook
    output_path_detailed = Path("reports") / "tables" / "detailed_evaluation_results.csv"
    output_path_detailed.parent.mkdir(parents=True, exist_ok=True)
    full_results_df.to_csv(output_path_detailed, index=False)
    print(f"\nDetailed evaluation results saved to {output_path_detailed}")

    kpi_df = compute_kpis(full_results_df)

    # Calculate bootstrapped confidence intervals
    print(f"\nCalculating {n_bootstraps} bootstraps for 95% confidence intervals...")
    ci_df = calculate_bootstrapped_kpis(full_results_df, n_bootstraps=n_bootstraps)
    
    # Merge CIs into the main KPI DataFrame
    final_kpi_df = pd.merge(kpi_df, ci_df, on='policy', how='left')

    print("\n--- Evaluation Summary with 95% Confidence Intervals ---")
    print(final_kpi_df.to_string())

    # 6. Save results
    output_path = Path("reports") / "tables" / "evaluation_summary.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    final_kpi_df.to_csv(output_path, index=False)
    print(f"\nEvaluation summary saved to {output_path}")
    print("--- Full Evaluation Finished Successfully ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run full evaluation of pricing policies.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/base_config.yaml",
        help="Path to the base configuration file."
    )
    parser.add_argument(
        "--experiment_name",
        type=str,
        default="dqn_baseline",
        help="Prefix of the experiment directory to load the DRL agent from (e.g., 'dqn_baseline')."
    )
    parser.add_argument(
        "--num_products",
        type=int,
        default=None,
        help="Optional. Limits the evaluation to this many products for testing purposes."
    )
    parser.add_argument(
        "--n_bootstraps",
        type=int,
        default=1000,
        help="Number of bootstrap samples for confidence interval calculation."
    )
    args = parser.parse_args()

    run_full_evaluation(Path(args.config), args.experiment_name, args.num_products, args.n_bootstraps)
