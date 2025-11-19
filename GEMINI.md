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
- The project has been updated with all necessary dependencies for training, including `stable-baselines3`, `torch`, and `tensorboard`.
- The `PriceEnv` in `src/envs/price_env.py` has been refactored to be more modular and accept data, configuration, and scalers directly.
- Utility functions `seed_everything` and `make_env` have been added to `src/utils.py` to support reproducible training.
- Training scripts for both DQN (`src/models/train_dqn.py`) and PPO (`src/models/train_ppo.py`) have been created.
- Both models have been successfully trained as a smoke test, and the resulting model files have been saved to `models/dqn/` and `models/ppo/` respectively.

**Optimality Assessment:**
- The training pipeline is now complete for both DQN and PPO agents. The scripts are configurable and leverage best practices like callbacks for saving the best model. This fulfills the requirements of Objective 3.

## Objective 4 — Evaluation and comparison

**Status:** In Progress

**Details:**
- The initial evaluation plan has been expanded into a more robust and comprehensive framework as defined in `objectives.md`.
- The plan now includes comparing the DRL agents against a wider suite of baselines:
    - **Historical Policy:** Replays historical prices.
    - **Rule-Based Policy:** A median-based pricing strategy.
    - **Greedy Model-Based:** A myopic policy using a trained demand model.
    - **Model-Based Planning (MPC):** A stronger, forward-looking baseline.
- The evaluation will include robustness checks by testing all policies in simulators with perturbed demand elasticity (±20%).
- Fairness will be strictly enforced by ensuring model-based baselines are trained only on historical data, with no leakage from the simulator's ground truth.
- Key performance metrics will include a balanced scorecard of business (Revenue, Units Sold) and operational (Price Volatility) KPIs, with 95% confidence intervals calculated via bootstrapping.

**Optimality Assessment:**
- The evaluation plan has been significantly enhanced to provide a rigorous, thesis-ready comparison of the DRL agents. It is designed to isolate the value added by the agents and test their robustness. Implementation of this new plan is the next step.

**Next Steps:** Implement the enhanced evaluation framework. This involves creating the new baseline policies (`src/baselines.py`) and updating the evaluation script (`src/evaluation.py`) and notebook (`notebooks/02-Evaluation.ipynb`) to reflect the new experiments and metrics.