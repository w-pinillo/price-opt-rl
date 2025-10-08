import os
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

def plot_learning_curves(log_dirs, output_path):
    """
    Plots the learning curves (mean reward) from TensorBoard log directories.

    Args:
        log_dirs (dict): A dictionary where keys are model names (e.g., 'DQN', 'PPO')
                         and values are paths to their TensorBoard log directories.
        output_path (str): The path to save the generated plot.
    """
    plt.figure(figsize=(12, 7))

    for model_name, log_dir in log_dirs.items():
        if not os.path.exists(log_dir):
            print(f"Warning: Log directory not found for {model_name}: {log_dir}")
            continue

        # Initialize EventAccumulator
        event_acc = EventAccumulator(log_dir)
        event_acc.Reload()

        # Check if 'eval/mean_reward' scalar is available
        if 'eval/mean_reward' in event_acc.Tags()['scalars']:
            rewards = event_acc.Scalars('eval/mean_reward')
            timesteps = [s.step for s in rewards]
            mean_rewards = [s.value for s in rewards]

            plt.plot(timesteps, mean_rewards, label=f'{model_name} Mean Reward')
        else:
            print(f"Warning: 'eval/mean_reward' scalar not found in logs for {model_name} at {log_dir}")
            # Fallback: check for 'rollout/ep_rew_mean' if 'eval/mean_reward' is not found
            if 'rollout/ep_rew_mean' in event_acc.Tags()['scalars']:
                rewards = event_acc.Scalars('rollout/ep_rew_mean')
                timesteps = [s.step for s in rewards]
                mean_rewards = [s.value for s in rewards]
                plt.plot(timesteps, mean_rewards, label=f'{model_name} Episode Reward Mean (Fallback)')
                print(f"Using 'rollout/ep_rew_mean' for {model_name}.")
            else:
                print(f"Error: Neither 'eval/mean_reward' nor 'rollout/ep_rew_mean' found for {model_name}.")


    plt.title('DRL Agent Learning Curves (Mean Reward)')
    plt.xlabel('Timesteps')
    plt.ylabel('Mean Reward')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # Ensure output directory exists
    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(output_path)
    print(f"Learning curves plot saved to {output_path}")

if __name__ == "__main__":
    # Define the assumed log directories
    base_project_path = "/home/stwpinillo/Documents/master/price-opt-rl"
    
    # Corrected log paths based on ls -R output
    dqn_log_dir = os.path.join(base_project_path, "models", "dqn", "tensorboard_log", "DQN_2")
    ppo_log_dir = os.path.join(base_project_path, "models", "ppo", "tensorboard_log", "PPO_1")

    log_directories = {
        "DQN": dqn_log_dir,
        "PPO": ppo_log_dir,
    }

    output_plot_path = os.path.join(base_project_path, "reports", "figures", "drl_learning_curves.png")

    plot_learning_curves(log_directories, output_plot_path)