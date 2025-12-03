import yaml
import os
import polars as pl
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_util import make_vec_env, SubprocVecEnv, DummyVecEnv
import importlib

from src.utils import seed_everything, make_env

def get_agent_class(agent_name: str):
    """
    Dynamically imports and returns the agent class from stable-baselines3.
    """
    try:
        module = importlib.import_module("stable_baselines3")
        # Agent classes in SB3 are uppercase (e.g., DQN, PPO)
        return getattr(module, agent_name.upper())
    except (ImportError, AttributeError):
        raise ValueError(f"Agent '{agent_name}' not found in stable-baselines3.")

def train(config: dict, run_dir: str):
    """
    Main function to train a specified RL agent.

    Args:
        config (dict): The experiment configuration.
        run_dir (str): The directory for saving all experiment artifacts.
    """
    # --- 1. Set seed ---
    seed_everything(config['training']['seed'])

    # --- 2. Load data ---
    data_path = os.path.join(config['paths']['processed_data_dir'], "train_scaled.parquet")
    try:
        data = pl.read_parquet(data_path)
    except FileNotFoundError:
        print(f"Error: Training data not found at {data_path}")
        return

    # --- 3. Create training and evaluation environments ---
    product_id = config['training']['product_id']
    n_envs = config['training'].get('n_envs', 1)
    print(f"Creating environment for product: {product_id} with {n_envs} parallel instance(s).")

    # Use make_vec_env to create vectorized environments for training
    env = make_vec_env(
        env_id=make_env,
        n_envs=n_envs,
        env_kwargs={'data': data, 'config': config, 'product_id': product_id},
        monitor_dir=os.path.join(run_dir, "train_monitor"),
        # Use SubprocVecEnv for multiple processes, DummyVecEnv for a single process
        vec_env_cls=SubprocVecEnv if n_envs > 1 else DummyVecEnv
    )

    # Evaluation environment remains a single, separate instance for consistency
    eval_env = make_env(data, config, product_id)
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
    
    # Select agent-specific hyperparameters from the config
    if agent_name in config.get('agent_params', {}):
        agent_config = config['agent_params'][agent_name]
    else:
        # Fallback to empty config if no specific params are provided
        agent_config = {}
        print(f"Warning: No specific parameters found for agent '{agent_name}'. Using defaults.")


    model = agent_class(
        "MlpPolicy",
        env,
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

    # --- 6. Save the final model and return best reward ---
    final_model_path = os.path.join(run_dir, 'final_model.zip')
    model.save(final_model_path)

    print(f"\nTraining complete.")
    print(f"Final model saved to: {final_model_path}")
    print(f"Best performing model saved in: {os.path.join(run_dir, 'best_model')}")

    # Return the best mean reward for Optuna
    return eval_callback.best_mean_reward