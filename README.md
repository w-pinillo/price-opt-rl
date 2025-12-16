# Master's Thesis: Dynamic Price Optimization using Deep Reinforcement Learning

This repository contains the code and resources for the Master's Thesis on Dynamic Price Optimization using Deep Reinforcement Learning.

## Project Structure

- `data/`: Contains the raw and processed data used in the project.
- `models/`: Contains the trained models and scalers.
- `notebooks/`: Contains the Jupyter notebooks for exploratory data analysis and evaluation.
- `src/`: Contains the source code for the project.
- `reports/`: Contains the figures and tables for the thesis.

## Running the DRL Pipeline

This section outlines how to train and evaluate Deep Reinforcement Learning (DRL) agents (PPO and DQN) using the provided pipeline. The process involves hyperparameter optimization to find the best model configurations and subsequent evaluation of their performance against baselines.

### 1. Hyperparameter Optimization

Use the `optimize_agent.py` script to run an Optuna study, which trains multiple agents with different hyperparameter combinations to identify the best-performing configuration.

**Key Arguments:**
*   `--agent {dqn, ppo}`: Specify the agent type (DQN or PPO).
*   `--n-trials <int>`: Number of optimization trials (e.g., 50 for a thorough search).
*   `--study-name <str>`: A custom name for the Optuna study (e.g., `ppo_optimization_run`).
*   `--n-jobs <int>`: Number of parallel jobs to run (set to 1 to avoid memory issues on resource-constrained systems, as noted in `GEMINI.md`).

**Output:**
*   For each study, a directory will be created under `models/<study_name>/`.
*   Inside this directory, each trial will have its own sub-directory (e.g., `trial_0/`).
*   Each trial directory will contain the trained model (`final_model.zip`), its configuration (`config.yaml`), and `VecNormalize` statistics (`vecnormalize.pkl`).
*   A `study_results.txt` file will summarize the best trial's parameters and performance.

**Example Commands:**

```bash
# For PPO Agent
python optimize_agent.py --agent ppo --n-trials 20 --study-name ppo_optimization_run --n-jobs 1

# For DQN Agent
python optimize_agent.py --agent dqn --n-trials 20 --study-name dqn_optimization_run --n-jobs 1
```

### 2. Evaluation

After optimization, use the `evaluate_multi_product_agent.py` script to assess the performance of the trained models. It compares the agent's policy against a "Trend-Based" heuristic baseline.

**Important Note:** The optimization script (`optimize_agent.py`) saves the final trained model for each trial as `final_model.zip`. Use this file for evaluation.

**Key Arguments:**
*   `--agent-path <str>`: Path to the trained agent's `.zip` file (e.g., `models/ppo_optimization_run/trial_X/final_model.zip`).
*   `--episodes <int>`: Number of evaluation episodes per product (default is 1).
*   `--use-pre-aggregated-data`: Flag to use the pre-aggregated dataset (`data/processed/top100_daily.parquet`) to prevent Out-Of-Memory (OOM) errors, especially on systems with limited RAM.
*   `--log-product-ids <str>`: Comma-separated list of product IDs for which to log detailed pricing decisions during evaluation.

**Output:**
*   A detailed "Evaluation Summary" table printed to the console, showing profit, volatility, and improvement against the baseline for each product.
*   An "Aggregate Performance" summary with overall average improvement and the percentage of products outperformed.
*   A scatter plot (`improvement_vs_volatility.png`) saved in the agent's specific run directory (e.g., `models/ppo_optimization_run/trial_X/`).

**Example Commands:**

```bash
# Identify the best trial from the `study_results.txt` in your optimization run directory
# For example, if trial_0 was the best for PPO:

# For Best PPO Model
python evaluate_multi_product_agent.py \
    --agent-path models/ppo_optimization_run/trial_0/final_model.zip \
    --episodes 1 \
    --use-pre-aggregated-data \
    --log-product-ids "PRD0900008,PRD0900097"

# For Best DQN Model
python evaluate_multi_product_agent.py \
    --agent-path models/dqn_optimization_run/trial_0/final_model.zip \
    --episodes 1 \
    --use-pre-aggregated-data \
    --log-product-ids "PRD0900008,PRD0900097"
```

**Note on VecNormalize:**
The `evaluate_multi_product_agent.py` script automatically loads the `vecnormalize.pkl` file from the agent's training directory. This file contains the normalization statistics for observations and rewards, which are crucial for the correct evaluation of models trained with `VecNormalize`.

