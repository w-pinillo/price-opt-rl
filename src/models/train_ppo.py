import yaml
import os
import polars as pl
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor

from src.utils import seed_everything, make_env

def train_ppo():
    """
    Main function to train the PPO agent.
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
    PRODUCT_ID_TO_TRAIN = "PRD0904358" 
    
    print(f"Creating environment for product: {PRODUCT_ID_TO_TRAIN}")
    env = make_env(data, config, PRODUCT_ID_TO_TRAIN)
    env = Monitor(env)

    eval_env = make_env(data, config, PRODUCT_ID_TO_TRAIN)
    eval_env = Monitor(eval_env)

    # --- 4. Define Callbacks ---
    model_dir = os.path.join(config['paths']['models_dir'], 'ppo')
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

    # --- 5. Instantiate and Train the PPO Model ---
    print("Instantiating PPO model...")
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=config['ppo']['learning_rate'],
        n_steps=config['ppo']['n_steps'],
        batch_size=config['ppo']['batch_size'],
        n_epochs=config['ppo']['n_epochs'],
        gamma=config['ppo']['gamma'],
        gae_lambda=config['ppo']['gae_lambda'],
        policy_kwargs={"net_arch": config['ppo']['policy_architecture']},
        verbose=1,
        tensorboard_log=os.path.join(model_dir, 'tensorboard_log')
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
    train_ppo()
