# Overview / Purpose

Implement a full pipeline that goes from raw transactional data to a trained DRL agent (DQN and PPO), and a robust evaluation comparing the learned pricing policies against historical and simple rule baselines. The pipeline should be reproducible, modular, and documented.

# Project layout (suggested)

```
data/
├── raw/ (incoming CSVs)
├── interim/
└── processed/
src/
├── etl.py (data loading & consolidation)
├── features.py (feature engineering)
├── envs/price_env.py (Gym environment)
├── models/train_dqn.py
├── models/train_ppo.py
├── evaluation.py
└── utils.py
notebooks/ (EDA and experiments)
reports/
models/ (saved agents and scalers)
config.yaml
requirements.txt
README.md
```

## Objective 1 — Data preparation

### Goal
Build a dataset with 3 years of daily data for the top-100 products and produce cleaned, feature-rich train/val/test splits and saved scalers.

### Steps to implement (in Python)

*   **Consolidate raw CSVs**
    *   Implement a function that reads all CSV files from `data/raw/`, parses dates, normalizes column types and unifies column names.
    *   Save a consolidated interim file in `data/interim/`.
*   **Select top-100 products**
    *   Implement a function to compute total units (or total revenue) per product and return the 100 IDs with highest totals.
    *   Save the list of top IDs to `data/processed/top100_ids.json`.
*   **Aggregate daily per product**
    *   Implement a function that groups transactions to daily level per product and produces columns: `date`, `prod_id`, `price` (average/median), `units`, `revenue`, `category`, `competitor_price` (if available), `promo` flag (if available).
    *   Save `data/processed/top100_daily.csv`.
*   **Generate time series features**
    *   For each product, compute lag features (lag 1, lag 7, lag 30), rolling means (7, 30), price change percentage, day-of-week and month features, weekend flag, and holiday flag if holiday calendar available.
    *   Optionally compute a rolling local elasticity estimate (log-log regression over a sliding window).
    *   Save resulting feature dataset and list of feature column names for the environment.
*   **Scale numeric features**
    *   Fit scalers (`StandardScaler` or `RobustScaler`) on training portion and persist them to `models/scalers/`.
    *   Implement functions to apply scalers to val/test and at inference.
*   **Temporal splitting**
    *   Split data by date into train, validation and test sets. Example: train = first 30 months, val = next 3 months, test = last 3 months (adapt to your exact 3-year span).
    *   Save the splits as CSV files.

### Files / functions to create

*   `src/etl.py`: `load_raw_csvs(path)`, `unify_columns(df)`, `save_interim(df)`
*   `src/features.py`: `select_top_products(df, n)`, `aggregate_daily(df, product_ids)`, `generate_time_features(df)`
*   `src/utils.py`: `fit_save_scalers(df, feature_cols, out_path)`, `load_scalers(path)`

### Tests / checks

*   All top100 products have at least one year of daily observations.
*   No NaNs remain in features intended for the agent (impute or drop as needed).
*   Verify that feature distributions (means/std) in training do not drastically differ from validation unless expected.
*   Quick EDA: plot price and units for 3 sample products.

### Acceptance criteria

*   `data/processed/top100_daily.csv` exists and contains full feature columns.
*   Scalers persisted under `models/scalers/`.
*   Train/Val/Test CSVs written to `data/processed/` and ready for the environment.

## Objective 2 — Define the MDP and implement the simulation environment

### Goal
Formalize state, action, and reward; implement an OpenAI Gym compatible environment that simulates demand given a price action.

### Design decisions to make first

*   **Action type**: discrete (e.g., set of percent changes) or continuous (price multiplier). Decide and document in `config.yaml`.
*   **Reward formulation**: revenue or profit. A reward shaping ablation (e.g., `revenue - λ·volatility`) can be tested. Decide primary business objective.
*   **Demand simulator approach**: parametric log-linear elasticity model (recommended for prototyping) or learned demand model (XGBoost/LightGBM) trained on historical data.

### Steps to implement (in Python)

*   Create a Gym environment class (file: `src/envs/price_env.py`) that implements `reset`, `step`, `render` and `seed`.
*   **Observation (state)**:
    *   A vector including selected features: recent demand lags, rolling means, competitor price, product embedding or one-hot ID (if multi-product), temporal variables (dow, month).
    *   Implement `observation_space` as a `Box` with matching shape.
*   **Action space**:
    *   For discrete actions: define a mapping from discrete index to price multiplier (or percent change).
    *   For continuous actions: define `action_space` `Box` with lower/upper multiplier bounds.
*   **Step logic**:
    *   Given action, compute new price.
    *   Use the demand simulator to compute demand at that price.
    *   Compute `units_sold` (respect inventory if modeled), compute revenue / profit.
    *   Compute `reward` = revenue (or custom), apply optional penalties for volatility.
    *   Update internal state (advance time, update lags and rolling features).
    *   Return `next_state`, `reward`, `done`, `info` (include price, units, revenue in `info`).
*   **Demand simulator options**:
    *   **Parametric**: `demand_t = base_t * exp(beta_price * (price_t - ref_price) + beta_comp * (comp_price_t - ref_comp) + seasonality + noise)`. Make betas configurable.
    *   **ML**: train a supervised model that predicts units given price and features; use it inside the environment as a deterministic/stochastic simulator.
*   **Episode horizon and reset**:
    *   Episode horizon = N days (e.g., one year). Implement `reset(random_start=True)` for better generalization.
*   **Multi-product scaling**:
    *   Prototype with 1 product per env. To scale, add product id/embedding to the observation and use vectorized envs (`VecEnv` / `SubprocVecEnv`) during training.

### Files / functions to create

*   `src/envs/price_env.py`: `PriceEnv` class, configurable via config.
*   `src/envs/simulators.py`: `parametric_demand(base, price, ref_price, betas, noise)` and wrapper to use ML model when requested.

### Tests / checks

*   Running a full episode with random policy should not produce NaNs or negative units.
*   Check that increasing price (other factors equal) tends to reduce predicted demand (sanity check).
*   Render or log a single episode to verify state transitions and reward calculations.

### Acceptance criteria

*   Gym environment is instantiable and supports `step()` and `reset()` without errors.
*   Configurable simulator type and parameters are documented and loadable via config file.

## Objective 3 — Implement and train DRL agents (DQN and PPO)

### Goal
Implement end-to-end training scripts to train DQN (for discrete actions) and PPO (for continuous or discrete), plus utilities for evaluation during training and saving best checkpoints.

### Steps to implement (in Python)

*   Set up a configuration file `config.yaml` that contains all hyperparameters, paths and simulator betas (learning rate, gamma, buffer size, eval frequency, total timesteps, seeds).
*   Create training scripts:
    *   `src/models/train_dqn.py`: constructs environment, sets seeds, instantiates DQN with selected policy architecture, trains, uses an evaluation callback to save best model.
    *   `src/models/train_ppo.py`: same for PPO.
*   **Policy/network architecture**:
    *   Use multi-layer perceptrons with sizes configurable in `config.yaml` (example: `[256, 256]`).
    *   Expose `policy_kwargs` or a custom policy file if a custom network is required.
*   **Logging and checkpointing**:
    *   Log training metrics to TensorBoard; save model checkpoints and the best model to `models/dqn/` and `models/ppo/`.
    *   Implement early stopping if validation reward does not improve for M evaluations (configurable).
*   **Multi-product vs single product**:
    *   Start with single-env prototypes to ensure pipeline works; move to vectorized envs using multiple product instances or product embedding in state.
*   **Experiment tracking**:
    *   Optionally integrate Weights & Biases or log experiment metadata to a JSON file per run.

### Files / functions to create

*   `src/models/train_dqn.py`
*   `src/models/train_ppo.py`
*   `src/policies.py` (optional custom policy wrappers)
*   `src/utils.py`: `seed_everything(seed)`, `make_env(config, split)`

### Tests / checks

*   Do a short smoke run: 10k–50k timesteps with a single product and check the mean reward trend.
*   After training, load the saved model and run it interactively for one episode; ensure `predict(action)` works and the environment responds.

### Acceptance criteria

*   Trained models saved to disk: `models/dqn/best_model` and `models/ppo/best_model`.
*   Training logs available in TensorBoard and model files reproducible given `config` and `seed`.

## Objective 4 — Evaluation and comparison

### Goal
Compare trained agents against a comprehensive suite of baselines (from simple heuristics to strong model-based policies), compute business and operational metrics, run robustness checks, and produce a full analysis report.

### Steps to implement (in Python)

*   **Train demand model for baselines**
    *   Train a demand prediction model (e.g., XGBoost) on the historical training data (`train_scaled.parquet`).
    *   Crucially, this model must not have any privileged access to the simulator's internal parameters to ensure a fair comparison.
*   **Implement evaluation backtest script (`src/evaluation.py`)** that:
    *   Loads the trained DRL agents (DQN, PPO) and the test split.
    *   Runs the agents over the test period and collects daily results.
    *   Runs the full suite of baselines over the same period:
        *   **Historical Policy:** Replay the actual historical prices.
        *   **Rule-Based Policy:** A simple business heuristic (e.g., `price = median_historical_price`).
        *   **Greedy Model-Based Policy:** At each step, use the demand model to choose the price that maximizes immediate expected revenue.
        *   **Model-Based Planning (MPC):** At each step, use the demand model to simulate outcomes over a short future horizon (e.g., 7 days) and select the price that maximizes cumulative revenue over that horizon.

### Compute metrics

*   **Business KPIs:** Total Revenue, Average Daily Revenue, Total Units Sold.
*   **Operational KPIs:** Average Price, Price Volatility (std. dev. of prices), Number of Price Changes.

### Statistical comparison

*   Use bootstrapping to obtain 95% confidence intervals for key performance metrics (e.g., total revenue) and for the performance difference between policies.

### Robustness and Sensitivity Analysis

*   **Simulator Misspecification:** Create alternate versions of the simulation environment where the demand elasticity parameter is perturbed (e.g., ±20%). Evaluate all policies (agents and baselines) in these environments to measure their robustness.
*   **Reward Shaping Ablation:** Compare the performance of agents trained with the pure revenue reward vs. agents trained with a reward function that penalizes price volatility (e.g., `revenue - λ * price_change^2`).

### Visualization outputs

*   Time series plots: price vs units vs revenue for agent vs. best baseline.
*   Bar charts comparing all policies on key metrics (e.g., Total Revenue, Price Volatility) with 95% CIs.
*   Robustness plots showing performance degradation under simulator misspecification.
*   Training learning curves for DRL agents.

### Reporting

*   Create an evaluation notebook that exports:
    *   A summary table of all metrics comparing all policies.
    *   Confidence intervals and interpretation of results.
    *   Figures saved to `reports/figures/`.
*   Write a thesis section summarizing the evaluation methodology and results.

### Files / functions to create

*   `src/evaluation.py`: `evaluate_policy(...)`, `backtest_model(...)`
*   `src/baselines.py`: Implementations for Historical, Rule-based, Greedy, and MPC policies.
*   `reports/evaluation_notebook.ipynb` (deliverable)
*   `reports/tables/metrics_summary.csv`

### Tests / checks

*   Backtest dataset alignment: ensure all policies' predicted daily results align with dates in the test set.
*   Bootstrap script returns stable CIs with sufficient iterations (e.g., 1000).
*   Ensure the demand model for baselines is trained without data leakage from the validation/test sets or the simulator's true parameters.

### Acceptance criteria

*   `reports/tables/metrics_summary.csv` with metrics for all agents and baselines exists.
*   The evaluation notebook contains a comprehensive analysis, including robustness checks.
*   Visual artifacts (plots) are saved under `reports/figures`.

## Cross-cutting tasks (reproducibility, documentation, deliverables)

*   Maintain a single config file (`config.yaml`) that records experiment hyperparameters, simulator betas, file paths and seed values.
*   Always fix seeds for `numpy`, `random`, `torch` and make environments deterministic where possible.
*   Log all experiments to TensorBoard; optionally track runs with Weights & Biases.
*   Version control the repo and tag milestones (e.g., `v0.1_prototype`, `v1.0_results`).
*   Delivery artifacts: code, notebooks, trained models, evaluation plots, thesis document (following university structure).

## Minimal recommended hyperparameters and config keys (examples to store in `config.yaml`)

*   `action_type`: `discrete` | `continuous`
*   `discrete_action_map`: list of multipliers (if discrete)
*   `action_low`, `action_high` (if continuous)
*   `beta_price` (simulator elasticity)
*   `noise_std` (simulator)
*   `learning_rate_dqn`, `buffer_size`, `batch_size`, `gamma`, `target_update`
*   `learning_rate_ppo`, `n_steps`, `batch_size`, `n_epochs`, `gamma`, `gae_lambda`
*   `total_timesteps`
*   `eval_freq`, `n_eval_episodes`
*   `seed`

## Short LLM prompt to generate starter files (one-line ready prompt you can paste)

> "Act as an expert machine learning engineer. Produce a skeleton Python project implementing: (1) ETL to produce daily features for top-100 products; (2) an OpenAI Gym environment 'PriceEnv' that simulates demand with a configurable log-linear elasticity model; (3) training scripts for DQN and PPO using stable-baselines3; (4) an evaluation script that backtests learned policies vs historical prices and computes revenue and volatility metrics. Return only plain markdown describing file names, short function signatures, config keys and a step-by-step runbook for running experiments. Do not output code blocks."

## Suggested timeline (high level)

*   **Weeks 1–2**: Data consolidation, EDA, feature engineering, top-100 selection.
*   **Weeks 3–4**: Implement Gym environment and parametric demand simulator; run sanity checks.
*   **Weeks 5–7**: Implement and train DQN and PPO prototypes; tune hyperparameters.
*   **Week 8**: Full evaluation, sensitivity analysis, visualization and write-up.
