import yaml
import os
import polars as pl
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_util import make_vec_env, SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.vec_env import VecNormalize
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

    # --- 2. Load data registries and mappers for TRAINING ---
    train_data_path = os.path.join(config['paths']['processed_data_dir'], "train_scaled.parquet")
    registry_output_path = os.path.join(run_dir, 'product_registry')
    
    train_data_registry, product_mapper, train_avg_daily_revenue = load_data_registry(
        data_path=train_data_path,
        output_path=registry_output_path
    )

    # --- Load data registries and mappers for EVALUATION ---
    val_data_path = os.path.join(config['paths']['processed_data_dir'], "val_scaled.parquet")
    eval_data_registry, _, eval_avg_daily_revenue = load_data_registry(
        data_path=val_data_path,
        output_path=registry_output_path # Overwrite is fine as mapper is the same
    )

    # --- 2b. Load Raw Data for Environments ---
    if config['data_config']['use_pre_aggregated_data']:
        raw_data_df = pl.read_parquet(config['data_config']['pre_aggregated_data_path'])
    else:
        raw_data_df = pl.read_parquet(config['paths']['raw_data'])

    historical_median_prices = raw_data_df.group_by("PROD_CODE").agg(
        pl.median("avg_price").alias("historical_median_price")
    ).to_pandas().set_index("PROD_CODE")['historical_median_price'].to_dict()

    # --- 3. Create training and evaluation environments ---
    n_envs = config['training'].get('n_envs', 1)

    # Training Environment
    train_env = make_vec_env(
        env_id=make_multi_product_env,
        n_envs=n_envs,
        env_kwargs={
            'data_registry': train_data_registry, 
            'product_mapper': product_mapper, 
            'avg_daily_revenue_registry': train_avg_daily_revenue,
            'config': config,
            'raw_data_df': raw_data_df,
            'historical_avg_prices': historical_median_prices
        },
        monitor_dir=os.path.join(run_dir, "train_monitor"),
        vec_env_cls=SubprocVecEnv if n_envs > 1 else DummyVecEnv
    )

    # Evaluation Environment (using validation data)
    eval_env = make_vec_env(
        env_id=make_multi_product_env,
        n_envs=1, # Evaluation is typically done on a single environment
        env_kwargs={
            'data_registry': eval_data_registry, 
            'product_mapper': product_mapper, 
            'avg_daily_revenue_registry': eval_avg_daily_revenue,
            'config': config,
            'raw_data_df': raw_data_df,
            'historical_avg_prices': historical_median_prices
        }
    )

    # Wrap the training environment with VecNormalize
    train_env = VecNormalize(train_env, norm_obs=True, norm_reward=True, clip_obs=10., clip_reward=10., norm_obs_keys=['features'])
    
    # Wrap the evaluation environment with VecNormalize, using the stats from the training env
    eval_env = VecNormalize(eval_env, training=False, norm_obs=True, norm_reward=False, clip_obs=10., norm_obs_keys=['features'])


    # --- 5. Instantiate and Train the Model ---
    agent_name = config['agent']
    agent_class = get_agent_class(agent_name)
    
    agent_config = config.get('agent_params', {}).get(agent_name, {})
    
    policy_kwargs_for_agent = {
        'features_extractor_class': CustomFeatureExtractor
    }
    
    if 'policy_kwargs' in agent_config:
        policy_kwargs_for_agent.update(agent_config['policy_kwargs'])
        del agent_config['policy_kwargs']
    
    if 'embedding_dim' in agent_config:
        del agent_config['embedding_dim']
    
    if 'net_arch' in agent_config:
        del agent_config['net_arch']

    model = agent_class(
        "MultiInputPolicy",
        train_env,
        policy_kwargs=policy_kwargs_for_agent,
        verbose=0, # Set verbose to 0 to reduce output
        tensorboard_log=os.path.join(run_dir, 'tensorboard_log'),
        device=config['training']['device'],
        **agent_config
    )

    # --- Set up Evaluation Callback ---
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(run_dir, 'best_model'),
        log_path=os.path.join(run_dir, 'eval_logs'),
        eval_freq=config['training']['eval_freq'],
        n_eval_episodes=config['training']['n_eval_episodes'],
        deterministic=True,
        render=False
    )

    model.learn(
        total_timesteps=config['training']['total_timesteps'],
        callback=eval_callback
    )

    # --- 6. Save the final model and VecNormalize stats ---
    final_model_path = os.path.join(run_dir, 'final_model.zip')
    model.save(final_model_path)
    train_env.save(os.path.join(run_dir, "vecnormalize.pkl"))
    
    # --- 7. Return the best mean reward from the callback ---
    best_mean_reward = eval_callback.best_mean_reward
    
    # Optuna works best with float, handle cases where it might be np.float
    return float(best_mean_reward)