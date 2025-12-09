import yaml
import os
import polars as pl
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_util import make_vec_env, SubprocVecEnv, DummyVecEnv
import importlib

from src.utils import seed_everything, make_multi_product_env
from src.data_utils import load_data_registry
from src.models.custom_feature_extractor import CustomFeatureExtractor

def get_agent_class(agent_name: str):
    """
    Dynamically imports and returns the agent class from stable-baselines3.
    """
    try:
        module = importlib.import_module("stable_baselines3")
        return getattr(module, agent_name.upper())
    except (ImportError, AttributeError):
        raise ValueError(f"Agent '{agent_name}' not found in stable-baselines3.")

def train(config: dict, run_dir: str):
    """
    Main function to train a specified multi-product RL agent.

    Args:
        config (dict): The experiment configuration.
        run_dir (str): The directory for saving all experiment artifacts.
    """
    # --- 1. Set seed ---
    seed_everything(config['training']['seed'])

    # --- 2. Load data registries and mappers ---
    data_path = os.path.join(config['paths']['processed_data_dir'], "train_scaled.parquet")
    registry_output_path = os.path.join(run_dir, 'product_registry')
    
    data_registry, product_mapper, avg_daily_revenue_registry = load_data_registry(
        data_path=data_path,
        output_path=registry_output_path
    )

    # --- 2b. Load Raw Data for Environment ---
    if config['data_config']['use_pre_aggregated_data']:
        raw_data_df = pl.read_parquet(config['data_config']['pre_aggregated_data_path'])
    else:
        # This is a fallback and might not be suitable for large datasets
        # that cause OOM issues. The primary path is the pre-aggregated one.
        raw_data_df = pl.read_parquet(config['paths']['raw_data'])

    historical_median_prices = raw_data_df.group_by("PROD_CODE").agg(
        pl.median("avg_price").alias("historical_median_price")
    ).to_pandas().set_index("PROD_CODE")['historical_median_price'].to_dict()


    # --- 3. Create training and evaluation environments ---
    n_envs = config['training'].get('n_envs', 1)
    print(f"Creating {n_envs} parallel multi-product environment(s).")

    env = make_vec_env(
        env_id=make_multi_product_env,
        n_envs=n_envs,
        env_kwargs={
            'data_registry': data_registry, 
            'product_mapper': product_mapper, 
            'avg_daily_revenue_registry': avg_daily_revenue_registry,
            'config': config,
            'raw_data_df': raw_data_df,
            'historical_avg_prices': historical_median_prices
        },
        monitor_dir=os.path.join(run_dir, "train_monitor"),
        vec_env_cls=SubprocVecEnv if n_envs > 1 else DummyVecEnv
    )

    eval_env = make_multi_product_env(
        data_registry, 
        product_mapper, 
        avg_daily_revenue_registry, 
        config, 
        raw_data_df, 
        historical_median_prices
    )
    eval_env = Monitor(eval_env, os.path.join(run_dir, "eval_monitor.csv"))

    # --- 4. Define Callbacks ---
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(run_dir, 'best_model'),
        log_path=os.path.join(run_dir, 'logs'),
        eval_freq=config['training']['eval_freq'],
        n_eval_episodes=config['training']['n_eval_episodes'],
        deterministic=True,
        render=False
    )

    # --- 5. Instantiate and Train the Model ---
    agent_name = config['agent']
    print(f"Instantiating {agent_name.upper()} model...")
    agent_class = get_agent_class(agent_name)
    
    agent_config = config.get('agent_params', {}).get(agent_name, {})
    
    # Define policy_kwargs for the custom feature extractor
    policy_kwargs = {
        'features_extractor_class': CustomFeatureExtractor
    }
    # Remove embedding_dim from agent_config as it's handled by policy_kwargs
    if 'embedding_dim' in agent_config:
        del agent_config['embedding_dim']
    
    # Remove net_arch from agent_config to let SB3 infer input dimension from feature extractor
    if 'net_arch' in agent_config:
        print(f"WARNING: Removing 'net_arch' from agent_config for {agent_name} to allow SB3 to infer policy network input dimensions from the CustomFeatureExtractor.")
        del agent_config['net_arch']

    model = agent_class(
        "MultiInputPolicy", # Changed from "MlpPolicy"
        env,
        policy_kwargs=policy_kwargs,
        verbose=1,
        tensorboard_log=os.path.join(run_dir, 'tensorboard_log'),
        device=config['training']['device'],
        **agent_config
    )

    print("Starting model training...")
    model.learn(
        total_timesteps=config['training']['total_timesteps'],
        callback=eval_callback
    )

    # --- 6. Save the final model ---
    final_model_path = os.path.join(run_dir, 'final_model.zip')
    model.save(final_model_path)

    print(f"\nTraining complete.")
    print(f"Final model saved to: {final_model_path}")
    print(f"Best performing model saved in: {os.path.join(run_dir, 'best_model')}")

    return eval_callback.best_mean_reward