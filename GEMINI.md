# Gemini Project Log

## Objective 1 - Data Preparation

**Status:** Completed

**Details:**
- The data preparation pipeline has been successfully executed.
- The final feature set includes:
    - **Lags:** Demand lags for 1, 7, 14, and 28 days.
    - **Rolling Stats:** 7 and 28-day rolling means and standard deviations for demand.
    - **Seasonality:** Cyclical `sin/cos` features for `day_of_week` and `month`.
- Processed, scaled, and split datasets (`train`, `val`, `test`) have been saved to `data/processed/`.
- The corresponding feature scalers have been saved to `models/scalers/`.

**Optimality Assessment:**
- The data preparation pipeline is modular, reproducible, and now includes a more robust feature set to better capture demand dynamics and seasonality, providing a solid foundation for training the DRL agents.

## Objective 2 — Define the MDP and implement the simulation environment

**Status:** Completed

**Details:**
- The initial `ParametricDemandSimulator` is complete.
- To create a higher-fidelity environment, a new ML-based simulator has been developed:
    - A LightGBM model was trained to predict demand (`src/models/train_demand_model.py`).
    - The model shows excellent performance on the validation set (R² > 0.99).
    - A new `MLDemandSimulator` class was added to `src/envs/simulators.py`.
- The `PriceEnv` has been updated to integrate the `MLDemandSimulator`, including constructing the proper feature vector, ensuring feature scaling consistency, handling out-of-bound predictions, and passing the environment's random generator for reproducible stochasticity.
- The core environment logic has been thoroughly validated, covering the Markov property of state transitions, episode termination and reset mechanisms, and the accuracy of reward calculation.
- The `config.yaml` has been refactored to allow for dynamic selection between `parametric` and `ml` simulators.
- **Feature Selection for Demand Model:** Through an iterative process of training and feature importance analysis, the feature set for the demand model was significantly optimized. It was discovered that the 30 `PROD_CATEGORY` features were redundant and could be removed without any meaningful loss in model performance. The final model uses a lean set of **18 features** while maintaining an R² of ~0.9945, providing a highly accurate simulation and an efficient state space for the DRL agent.

**Optimality Assessment:**
- The simulation environment is now upgraded to a more realistic, data-driven model, which will allow for the training of more robust and effective DRL agents. The state space has been optimized for DRL learning efficiency.

## Objective 3 — Implement and train DRL agents (DQN and PPO)

**Status:** Completed

**Details:**
- A robust experimental workflow has been established to facilitate scientific evaluation of DRL methods.
- **Configuration Management:** A `configs` directory now separates a `base_config.yaml` from experiment-specific configurations (e.g., `dqn_baseline.yaml`), allowing for clear, modular, and reproducible experiment definitions.
- **Experiment Runner:** A new master script, `run_experiment.py`, automates the experimental process. It merges configurations, creates unique timestamped output directories for all artifacts (models, logs, configs), and calls the training logic.
- **Modular Training:** The core logic in `src/models/train_agent.py` has been refactored into a `train` function that accepts a configuration object and an output directory, making it a reusable component of the experimental framework.
- This new structure enables systematic tracking of metrics, comparison of different algorithms, and effortless testing of various hyperparameter setups, fulfilling a key requirement for rigorous scientific investigation.

### Sub-Objective 3.1 — Hyperparameter Optimization

**Status:** Completed (Initial Phase)

**Details:**
- A dedicated script, `optimize_agent.py`, has been created to automate hyperparameter tuning using Optuna.
- Search spaces have been defined for DQN, PPO, and the newly added SAC agent, allowing for comprehensive tuning of all three.
- Initial optimization studies have been successfully executed for DQN (discrete space), PPO (discrete space), and SAC (continuous space).
- The results, including best-performing parameters, per-trial logs, and trained models, are systematically saved into unique study directories within the `models/` folder.

**Optimality Assessment:**
- The project is now fully equipped for large-scale, automated hyperparameter optimization. The framework is robust, reproducible, and provides a clear path for identifying the best possible agent for the final evaluation. The initial runs have validated the process and provided strong baseline parameters.

## Objective 4 — Evaluation and comparison

**Status:** Completed

**Details:**
- The evaluation plan has been streamlined to focus on simple, reliable baselines: `HistoricalPolicy` and `RuleBasedPolicy`, which have been implemented in `src/baselines.py`.
- The `src/evaluation.py` script was refactored into a reusable module for evaluating a single policy on an environment.
- A new orchestrator script, `run_evaluation.py`, has been created to manage the full evaluation campaign. This script handles:
    - Loading trained DRL agents and baseline policies.
    - Iterating through all relevant products (with an option to limit products for testing via `--num_products`).
    - Running each policy on the test environment.
    - Calculating a comprehensive suite of business and operational KPIs, including: **Total Revenue, Total Units Sold, Average Price, Price Volatility, Average Daily Revenue, and Total Number of Price Changes.**
    - **Bootstrapping has been implemented to obtain 95% confidence intervals for all calculated KPIs, providing statistical rigor to the performance comparisons.**
    - Saving the aggregated KPI results to `reports/tables/evaluation_summary.csv` and the detailed daily results to `reports/tables/detailed_evaluation_results.csv`.
- Finally, **code for the `notebooks/02-Evaluation.ipynb` has been provided** to load these saved results, perform further analysis, and generate visualizations (bar charts for KPIs with CIs and time series plots for price/units/revenue for sample products).
- This setup validates that the core components (environment, DRL agent, baselines, and evaluation loop) are all functioning correctly together and generating comprehensive metrics with statistical confidence, which can then be thoroughly analyzed and visualized.

**Optimality Assessment:**
- The project now has a robust, modular, and complete framework for policy evaluation. It is capable of generating a rich set of metrics across multiple products and policies, along with their 95% confidence intervals. The evaluation outputs are structured for comprehensive analysis and visualization in the provided Jupyter notebook, fulfilling all requirements for rigorous scientific investigation and reporting.