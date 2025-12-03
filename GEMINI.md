# Gemini Work Log

This file contains a log of significant events, decisions, and outcomes that occur during the project's development.

---

## Memory Optimization for Resource-Constrained Environments (December 2, 2025)

**Context:** During experimentation on a system with 16GB RAM and 16GB VRAM (compared to a 32GB RAM development environment), out-of-memory (OOM) issues were encountered. Analysis revealed that excessive RAM consumption was primarily due to high parallelization settings.

**Root Causes Identified:**
-   **High `n_envs` in Experiment Configurations:** Baseline configurations for DQN (`dqn_baseline.yaml`) and PPO (`ppo_baseline.yaml`) used `n_envs` values of 4 and 16 respectively. When `n_envs > 1`, `stable-baselines3` utilizes `SubprocVecEnv`, spawning a separate Python process for each environment, leading to rapid RAM exhaustion.
-   **Large DQN Replay Buffer:** The `buffer_size` of `1,000,000` for DQN in `dqn_baseline.yaml` demanded significant system RAM.
-   **Parallel Hyperparameter Optimization:** The `optimize_agent.py` script's default `n_jobs=2` for Optuna trials launched multiple `train_agent.py` processes concurrently.

**Solutions Implemented:**
-   **Reduced `n_envs`:**
    -   `configs/experiments/dqn_baseline.yaml`: `n_envs` reduced from `4` to `1`.
    -   `configs/experiments/ppo_baseline.yaml`: `n_envs` reduced from `16` to `1`.
-   **Reduced DQN `buffer_size`:**
    -   `configs/experiments/dqn_baseline.yaml`: `buffer_size` reduced from `1,000,000` to `100,000`.

**Optimality Assessment:**
- These changes addressed the immediate OOM issues by significantly reducing the memory footprint of individual training runs and parallel optimization trials. While some parallel training benefits are reduced, stability on resource-constrained hardware is prioritized, enabling successful completion of experiments. Further tuning can be explored as hardware capabilities improve.