import argparse
import os
import yaml
from datetime import datetime
import src.models.train_agent as train_agent
import collections.abc

def merge_configs(d, u):
    """
    Recursively merge two dictionaries. 'u' overrides 'd'.
    """
    for k, v in u.items():
        if isinstance(v, collections.abc.Mapping):
            d[k] = merge_configs(d.get(k, {}), v)
        else:
            d[k] = v
    return d

def run_experiment(exp_config_path: str):
    """
    Run a single experiment based on a configuration file.
    """
    # --- 1. Load Configurations ---
    try:
        with open("configs/base_config.yaml", "r") as f:
            base_config = yaml.safe_load(f)
    except FileNotFoundError:
        print("Error: base_config.yaml not found in configs/ directory.")
        return

    try:
        with open(exp_config_path, "r") as f:
            exp_config = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Error: Experiment config file not found at {exp_config_path}")
        return

    # --- 2. Merge Configurations ---
    config = merge_configs(base_config, exp_config)

    # --- 3. Create Unique Output Directory ---
    exp_name = os.path.splitext(os.path.basename(exp_config_path))[0]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(config["paths"]["models_dir"], f"{exp_name}_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    
    # --- 4. Save Final Config for Reproducibility ---
    with open(os.path.join(run_dir, "config.yaml"), "w") as f:
        yaml.dump(config, f, default_flow_style=False)

    print(f"--- Starting Experiment: {exp_name} ---")
    print(f"Output directory: {run_dir}")
    print("\n--- Configuration ---")
    print(yaml.dump(config, default_flow_style=False, indent=2))
    print("---------------------\n")

    # --- 5. Start Training ---
    try:
        train_agent.train(config=config, run_dir=run_dir)
        print(f"\n--- Experiment {exp_name} Finished Successfully ---")
    except Exception as e:
        print(f"\n--- Experiment {exp_name} Failed ---")
        print(f"Error: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a DRL experiment.")
    parser.add_argument(
        "exp_config",
        type=str,
        help="Path to the experiment configuration file (e.g., 'configs/experiments/dqn_baseline.yaml')",
    )
    args = parser.parse_args()
    
    run_experiment(args.exp_config)
