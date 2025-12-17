# Dynamic Price Optimization using Deep Reinforcement Learning

This repository contains the code for a Master's Thesis on using Deep Reinforcement Learning (DRL) for dynamic price optimization.

The project implements a full pipeline to train and evaluate DRL agents (PPO and DQN) on transactional data to learn optimal pricing policies.

## Getting Started

### Prerequisites

- Python 3.8+
- Poetry

### Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/w-pinillo/price-opt-rl.git
    cd price-opt-rl
    ```

2. Install the dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

The DRL pipeline consists of two main stages: hyperparameter optimization and model evaluation.

### 1. Hyperparameter Optimization

Use the `optimize_agent.py` script to run an Optuna study and find the best hyperparameters for your agent.

**Example:**

```bash
# For PPO Agent
python optimize_agent.py --agent ppo --n-trials 20 --study-name ppo_optimization_run --n-jobs 1

# For DQN Agent
python optimize_agent.py --agent dqn --n-trials 20 --study-name dqn_optimization_run --n-jobs 1
```

A `study_results.txt` file will be created in the `models/<study_name>/` directory, summarizing the best trial's parameters and performance.

### 2. Model Evaluation

Evaluate the trained agent against a baseline using the `evaluate_multi_product_agent.py` script.

**Example:**

```bash
# For Best PPO Model
python evaluate_multi_product_agent.py \
    --agent-path models/ppo_optimization_run/trial_0/final_model.zip \
    --episodes 1 \
    --use-pre-aggregated-data

# For Best DQN Model
python evaluate_multi_product_agent.py \
    --agent-path models/dqn_optimization_run/trial_0/final_model.zip \
    --episodes 1 \
    --use-pre-aggregated-data
```

The evaluation script will output a detailed summary of the agent's performance.

## Project Structure

| Path      | Description                                                 |
| :-------- | :---------------------------------------------------------- |
| `data/`   | Raw and processed data used in the project.                 |
| `models/` | Trained models, Optuna studies, and scalers.                |
| `reports/`| Figures and tables for the thesis.                          |
| `src/`    | Source code for the DRL pipeline, including data processing, environments, and models. |

---
*This `README.md` was simplified based on the original content.*