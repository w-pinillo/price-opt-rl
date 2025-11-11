import yaml
import os
import polars as pl
from stable_baselines3 import DQN, PPO, A2C
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
import importlib

from src.utils import seed_everything, make_env

def get_agent_class(agent_name: str):
    """
    Dynamically imports and returns the agent class from stable-baselines3.
    """
    try:
        module = importlib.import_module("stable_baselines3")
        return getattr(module, agent_name)
    except (ImportError, AttributeError):
        raise ValueError(f"Agent '{agent_name}' not found in stable-baselines3.")

def train_agent():
    """
    Main function to train a specified RL agent.
    """
    # --- 1. Load configuration and set seed ---
    with open("config.yaml", 'r') as f:
        config = yaml.safe_load(f)

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
    print(f"Creating environment for product: {product_id}")
    env = make_env(data, config, product_id)
    env = Monitor(env)

    eval_env = make_env(data, config, product_id)
    eval_env = Monitor(eval_env)

    # --- 4. Define Callbacks ---
    agent_name = config['training']['agent_name']
    model_dir = os.path.join(config['paths']['models_dir'], agent_name.lower())
    os.makedirs(model_dir, exist_ok=True)
    
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(model_dir, 'best_model'),
        log_path=os.path.join(model_dir, 'logs'),
        eval_freq=config['training']['eval_freq'],
        n_eval_episodes=config['training']['n_eval_episodes'],
        deterministic=True,
        render=False
    )

    # --- 5. Instantiate and Train the Model ---
    print(f"Instantiating {agent_name} model...")
    agent_class = get_agent_class(agent_name)
    agent_config = config['agents'][agent_name]

    model = agent_class(
        "MlpPolicy",
        env,
        verbose=1,
        tensorboard_log=os.path.join(model_dir, 'tensorboard_log'),
        **agent_config
    )

    print("Starting model training...")
    model.learn(
        total_timesteps=config['training']['total_timesteps'],
        callback=eval_callback
    )

    # --- 6. Save the final model ---
    final_model_path = os.path.join(model_dir, 'final_model.zip')
    model.save(final_model_path)

    print(f"\nTraining complete.")
    print(f"Final model saved to: {final_model_path}")
    print(f"Best performing model saved in: {os.path.join(model_dir, 'best_model')}")

if __name__ == "__main__":
    train_agent()
