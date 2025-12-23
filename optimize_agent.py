import argparse
import os
import yaml
import optuna
from datetime import datetime
import collections.abc
import src.models.train_agent as train_agent

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

def define_search_space(trial: optuna.trial.Trial, agent_name: str) -> dict:
    """
    Defines the hyperparameter search space for a given agent using Optuna trial object.
    Based on the user's proposed configuration.
    """
    if agent_name.lower() == 'dqn':
        # Note: 'network_architecture' is not tunable via this dict. It's part of the policy_kwargs in base_config.
        # 'epsilon_decay' is also not a direct parameter in SB3's DQN, it uses a linear schedule by default.
        return {
            'learning_rate': trial.suggest_categorical('learning_rate', [1e-4, 5e-4, 1e-3]),
            'buffer_size': trial.suggest_categorical('buffer_size', [10000, 25000, 50000]),
            'batch_size': trial.suggest_categorical('batch_size', [32, 64, 128]),
            'gamma': trial.suggest_categorical('gamma', [0.99, 0.95]),
            'target_update_interval': trial.suggest_categorical('target_update_interval', [1000, 5000]),
        }
    elif agent_name.lower() == 'ppo':
        return {
            'learning_rate': trial.suggest_categorical('learning_rate', [1e-4, 3e-4, 1e-3]),
            'n_steps': trial.suggest_categorical('n_steps', [512, 1024, 2048]),
            'batch_size': trial.suggest_categorical('batch_size', [32, 64, 128]),
            'n_epochs': trial.suggest_categorical('n_epochs', [5, 10, 20]),
            'gamma': trial.suggest_categorical('gamma', [0.99, 0.95]),
            'gae_lambda': trial.suggest_categorical('gae_lambda', [0.9, 0.95, 0.98]),
            'clip_range': trial.suggest_categorical('clip_range', [0.1, 0.2, 0.3]),
        }
    elif agent_name.lower() == 'sac':
        return {
            'learning_rate': trial.suggest_categorical('learning_rate', [1e-4, 3e-4, 1e-3]),
            'buffer_size': trial.suggest_categorical('buffer_size', [50000, 100000]),
            'batch_size': trial.suggest_categorical('batch_size', [64, 128, 256]),
            'gamma': trial.suggest_categorical('gamma', [0.99, 0.95]),
            'tau': trial.suggest_categorical('tau', [0.005, 0.01, 0.02]),
        }
    else:
        raise ValueError(f"Unknown agent for optimization: {agent_name}")

def objective(trial: optuna.trial.Trial, agent_name: str, study_dir: str) -> float:
    """
    The objective function for Optuna to optimize.
    """
    # --- 1. Load Base Config ---
    with open("configs/base_config.yaml", "r") as f:
        config = yaml.safe_load(f)

    # --- 2. Define and Merge Hyperparameters ---
    agent_params = define_search_space(trial, agent_name)
    override_config = {
        'agent': agent_name,
        'training': {
            'agent': agent_name
        },
        'agent_params': {
            agent_name: agent_params
        }
    }

    # SAC requires a continuous action space
    if agent_name.lower() == 'sac':
        override_config['env'] = {'action_type': 'continuous'}

    config = merge_configs(config, override_config)

    # --- 3. Create Unique Run Directory for this Trial ---
    run_name = f"trial_{trial.number}"
    run_dir = os.path.join(study_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)

    with open(os.path.join(run_dir, "config.yaml"), "w") as f:
        yaml.dump(config, f, default_flow_style=False)
        
    # --- 4. Train Agent and Get Reward ---
    try:
        # Reduce verbosity for optimization runs
        config['training']['total_timesteps'] = 10000 # Use shorter runs for faster tuning
        config['training']['eval_freq'] = 1000
        
        mean_reward = train_agent.train(config=config, run_dir=run_dir)
        
        # Handle cases where training might fail and return None
        if mean_reward is None:
            raise optuna.exceptions.TrialPruned("Training script returned None.")

    except Exception as e:
        print(f"--- Trial {trial.number} Failed ---")
        print(f"Error: {e}")
        # Prune trial if it fails
        raise optuna.exceptions.TrialPruned()
    
    return mean_reward

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hyperparameter optimization for DRL agents.")
    parser.add_argument(
        "--agent",
        type=str,
        required=True,
        choices=['dqn', 'ppo', 'sac'],
        help="The agent to optimize (e.g., 'dqn')."
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=20,
        help="The number of optimization trials to run."
    )
    parser.add_argument(
        "--study-name",
        type=str,
        default=None,
        help="Name for the Optuna study. If not provided, a timestamped name is generated."
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=2,
        help="The number of parallel jobs to run for optimization."
    )
    args = parser.parse_args()

    # --- Create Study Directory ---
    if args.study_name:
        study_name = args.study_name
    else:
        study_name = f"{args.agent}_optimization_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    study_dir = os.path.join("models", study_name)
    os.makedirs(study_dir, exist_ok=True)

    print(f"--- Starting Optuna Study: {study_name} ---")
    print(f"Agent: {args.agent.upper()}")
    print(f"Number of trials: {args.n_trials}")
    print(f"Study results will be saved in: {study_dir}")

    # The lambda is used to pass extra arguments to the objective function
    objective_func = lambda trial: objective(trial, args.agent, study_dir)
    
    study = optuna.create_study(direction="maximize")
    study.optimize(objective_func, n_trials=args.n_trials, n_jobs=args.n_jobs)

    # --- Print and Save Results ---
    print("\n--- Optimization Finished ---")
    print(f"Number of finished trials: {len(study.trials)}")
    print("Best trial:")
    best_trial = study.best_trial
    print(f"  Value (Best Mean Reward): {best_trial.value}")
    print("  Params: ")
    for key, value in best_trial.params.items():
        print(f"    {key}: {value}")
        
    results_path = os.path.join(study_dir, "study_results.txt")
    with open(results_path, "w") as f:
        f.write(f"Best trial value: {best_trial.value}\n")
        f.write("Best parameters:\n")
        for key, value in best_trial.params.items():
            f.write(f"  {key}: {value}\n")
    print(f"Study results saved to {results_path}")
