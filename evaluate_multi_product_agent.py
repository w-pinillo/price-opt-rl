import argparse
import yaml
import os
import polars as pl
import numpy as np

from src.utils import make_multi_product_env
from src.data_utils import load_data_registry, load_product_registry

# This is a placeholder for get_agent_class from train_agent.py
# In a real implementation, this might be moved to a shared utils file.
import importlib
def get_agent_class(agent_name: str):
    module = importlib.import_module("stable_baselines3")
    return getattr(module, agent_name.upper())

def evaluate(agent_path: str, raw_product_id: str, episodes: int, config_path: str):
    """
    Evaluates a trained multi-product agent on a single, specific product.
    """
    # --- 1. Load configurations and data ---
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Use the run directory from the agent path to find the correct product registry
    run_dir = os.path.dirname(os.path.dirname(agent_path)) # e.g., models/dqn_baseline_.../best_model.zip -> models/dqn_baseline_...
    registry_path = os.path.join(run_dir, 'product_registry')

    data_path = os.path.join(config['paths']['processed_data_dir'], "test_scaled.parquet")

    print("Loading data registries...")
    data_registry, product_mapper, avg_daily_revenue_registry = load_data_registry(
        data_path=data_path,
        output_path=registry_path # This will just load if it exists, and re-create if not
    )
    
    # Invert the mapper to get raw_id from dense_id
    # Note: The product_mapper from load_data_registry is raw -> dense. We need to find our raw_id.
    if raw_product_id not in product_mapper:
        raise ValueError(f"Product ID {raw_product_id} not found in the product registry.")
    
    dense_product_id = product_mapper[raw_product_id]

    # --- 2. Load the trained agent ---
    agent_name = config['agent']
    agent_class = get_agent_class(agent_name)
    print(f"Loading agent from {agent_path}...")
    agent = agent_class.load(agent_path)

    # --- 3. Create the evaluation environment ---
    print(f"Creating evaluation environment for product: {raw_product_id} (Dense ID: {dense_product_id})")
    eval_env = make_multi_product_env(data_registry, product_mapper, config)

    # --- 4. Run the evaluation loop ---
    total_revenues = []
    print(f"Running evaluation for {episodes} episodes...")

    for episode in range(episodes):
        obs, info = eval_env.reset(product_id=dense_product_id, sequential=True) # Use sequential=True for consistent evaluation
        done = False
        episode_revenue = 0
        
        while not done:
            action, _states = agent.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = eval_env.step(action)
            done = terminated or truncated
            episode_revenue += reward
        
        total_revenues.append(episode_revenue)
        print(f"Episode {episode + 1}/{episodes} | Revenue: {episode_revenue:.2f}")

    # --- 5. Report results ---
    avg_agent_revenue = np.mean(total_revenues)
    baseline_revenue = avg_daily_revenue_registry[dense_product_id] * config['env']['episode_horizon']

    print("\n--- Evaluation Summary ---")
    print(f"Product ID: {raw_product_id}")
    print(f"Number of Episodes: {episodes}")
    print("-" * 20)
    print(f"Average Agent Revenue per Episode: {avg_agent_revenue:.2f}")
    print(f"Baseline Historical Revenue per Episode: {baseline_revenue:.2f}")
    print("-" * 20)
    
    improvement = ((avg_agent_revenue - baseline_revenue) / baseline_revenue) * 100 if baseline_revenue > 0 else float('inf')
    print(f"Improvement over baseline: {improvement:.2f}%")
    print("--- End of Summary ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a multi-product DRL agent.")
    parser.add_argument("--agent-path", type=str, required=True, help="Path to the trained agent's .zip file.")
    parser.add_argument("--product-id", type=str, required=True, help="The raw PROD_CODE of the product to evaluate.")
    parser.add_argument("--episodes", type=int, default=10, help="Number of episodes to run for evaluation.")
    parser.add_argument("--config-path", type=str, required=True, help="Path to the training configuration file (e.g., base_config.yaml).")
    
    args = parser.parse_args()
    
    evaluate(args.agent_path, args.product_id, args.episodes, args.config_path)
