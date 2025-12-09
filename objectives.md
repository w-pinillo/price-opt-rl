# Overview / Purpose

Implement a full pipeline that goes from raw transactional data to a trained DRL agent (DQN and PPO), and a robust evaluation comparing the learned pricing policies against historical and simple rule baselines. The pipeline should be reproducible, modular, and documented.

## Objective 1 — Data preparation
**Status: Completed**

### Goal
Build a dataset with 3 years of daily data for the top-100 products and produce cleaned, feature-rich train/val/test splits and saved scalers.

## Objective 2 — Define the MDP and implement the simulation environment
**Status: Completed**

### Goal
Formalize state, action, and reward; implement an OpenAI Gym compatible environment that simulates demand given a price action.

## Objective 3 — Implement and train DRL agents (DQN and PPO)
**Status: Completed**

### Goal
Implement end-to-end training scripts to train DQN (for discrete actions) and PPO (for continuous or discrete), plus utilities for evaluation during training and saving best checkpoints.

## Objective 4 — Evaluation and comparison
**Status: Completed**

### Goal
To provide a comprehensive evaluation of the DRL pricing agents by answering three fundamental questions:
1.  **Improvement:** Do they make better decisions than past strategies, leading to increased revenue or efficiency?
2.  **Adaptability:** Do they outperform simple, safe rules that lack dynamic decision-making capabilities?
3.  **Reliability:** Are their pricing strategies stable and consistent over time, and robust to changes in market conditions?

## Objective 5 - Hardware Acceleration and Experimentation
**Status: In Progress**

### Goal
Configure the project and system environment to leverage a powerful desktop with a dedicated NVIDIA GPU (Ryzen 7 9700X, RTX 5060 Ti 16GB) to accelerate training and hyperparameter tuning.

### To-Do List

- [x] **Environment Setup:** Install NVIDIA drivers, CUDA Toolkit, and cuDNN.
- [x] **Python Dependencies:** Create a Python virtual environment and install the GPU-enabled version of PyTorch.
- [x] **Configuration:** Add a `device: cuda` parameter to `configs/base_config.yaml`.
- [ ] **Code Modification:** Modify `src/models/train_agent.py` to use the new `device` configuration.
- [ ] **Performance Tuning:** Increase `n_envs`, `batch_size`, and `buffer_size` in experiment configs to better utilize the hardware.
- [ ] **Parallelization:** Update `optimize_agent.py` to run Optuna trials in parallel by setting the `n_jobs` parameter.
- [ ] **Verification:** Run a short test experiment and verify GPU utilization is active and efficient using a tool like `nvidia-smi`.

### Notes on Memory Optimization (from previous work)

**Context:** During experimentation on a system with 16GB RAM, initial out-of-memory (OOM) issues were encountered due to high parallelization settings and an inefficient data pipeline.

**Solutions Implemented:**
-   **Reduced Parallelism (December 2, 2025):** `n_envs` and `buffer_size` were reduced in baseline experiment configs.
-   **Data Pipeline Refactoring (December 5, 2025):** The pipeline was refactored to optionally bypass raw data processing and load a pre-aggregated daily dataset directly. This was controlled by a `use_pre_aggregated_data` flag in `configs/base_config.yaml`.

**Current Status (December 5, 2025):**
- **Resolved:** The critical Out-Of-Memory (OOM) blocker related to the data pipeline has been resolved. By using the pre-aggregated data option, the pipeline runs efficiently on memory-constrained systems.

**Criticality:**
- With the data pipeline unblocked, there are no longer critical memory-related impediments to proceeding with agent training and evaluation.

**Recommendations for Future Experiments and Hyperparameter Optimization:**
-   **Limit Optuna Parallelization:** When running `optimize_agent.py`, it is still advisable to explicitly set `--n-jobs 1` to prevent excessive memory use from concurrent training processes.
-   **Monitor GPU Usage:** Regularly use `nvidia-smi` during training to confirm that the GPU is actively utilized.

**Next Steps:**
- Proceed with the remaining tasks in the Objective 5 to-do list (Code Modification, Performance Tuning, etc.).
- The immediate priority is now **Milestone 4 of Objective 6**: running the full evaluation for the multi-product agent.

## Cross-cutting tasks (reproducibility, documentation, deliverables)

*   Maintain a single config file (`config.yaml`) that records experiment hyperparameters, simulator betas, file paths and seed values.
*   Always fix seeds for `numpy`, `random`, `torch` and make environments deterministic where possible.
*   Log all experiments to TensorBoard; optionally track runs with Weights & Biases.
*   Version control the repo and tag milestones (e.g., `v0.1_prototype`, `v1.0_results`).
*   Delivery artifacts: code, notebooks, trained models, evaluation plots, thesis document (following university structure).

## Suggested timeline (high level)

*   **Weeks 1–2**: Data consolidation, EDA, feature engineering, top-100 selection.
*   **Weeks 3–4**: Implement Gym environment and parametric demand simulator; run sanity checks.
*   **Weeks 5–7**: Implement and train DQN and PPO prototypes; tune hyperparameters.
*   **Week 8**: Full evaluation, sensitivity analysis, visualization and write-up.

## Objective 6 - Implement a Multi-Product DRL Agent
**Status: Completed**

### Goal
Transition from a "one agent per product" training strategy to a single, unified DRL agent capable of learning an optimal pricing policy for all 100 products simultaneously. This approach aims to improve training efficiency, scalability, and foster knowledge sharing between products.

### Background and Rationale
The current architecture trains an independent agent for each product. This is computationally expensive and inefficient, as it fails to leverage common patterns and behaviors that may exist across different products. A single, multi-product agent can learn a more generalized and robust policy, leading to better performance and faster learning, especially for products with sparse data.

### Optimal Strategy: Product ID as State
The chosen strategy involves modifying the environment and the agent's perception of the state to make it "product-aware."

1.  **Unified Data Handling:** The `PriceEnv` environment will be modified to accept the entire unfiltered dataset containing all products, rather than a pre-filtered, single-product DataFrame.
2.  **State Space Modification (Dict Observation & Embeddings):** The agent's observation space will be a `gymnasium.spaces.Dict` combining product and market features. The `PROD_CODE` will be mapped to a dense integer range `[0, ..., num_products-1]` and then fed through an `nn.Embedding` layer within a custom feature extractor. The embedding vector (initial dimension: 16) will be concatenated with the market features to form the final observation. A `product_registry.json` will be created during data prep to map raw `PROD_CODE`s to their corresponding indices.
3.  **Episode Sampling:** The environment's `reset` method will be updated. At the beginning of each episode, it will randomly sample not only a starting time step but also a `PROD_CODE`, dedicating that episode to the selected product.
4.  **Training Script Adaptation:**
    *   `src/models/train_agent.py`: The script will be updated to work with the new environment. The concept of a single `product_id` per run will be removed, as the agent now handles all products.
    *   `src/utils.py`: The `make_env` function will be simplified. It will no longer need to filter data for a specific product.

### Implementation To-Do List
- [x] **Environment (`PriceEnv`):**
    - [x] Modify `__init__` to handle the full, multi-product DataFrame.
    - [x] Create a mapping from `PROD_CODE` to a dense integer index.
    - [x] Update the `observation_space` to be a `Dict` including the product index and market features.
    - [x] Modify `reset()` to sample a `PROD_CODE` (and its corresponding index) for each new episode.
    - [x] Update `_get_obs()` to return a dictionary observation.
- [x] **Training Pipeline:**
    - [x] Modify `src/utils.py`'s `make_env` to no longer filter by `product_id`.
    - [x] Modify `src/models/train_agent.py` to remove the `product_id` parameter and logic, and to integrate the custom feature extractor.
    - [x] Update experiment configuration files (`configs/experiments/*.yaml`) to remove the `product_id` parameter and add configuration for the custom feature extractor (e.g., embedding_dim).
- [x] **Verification:**
    - [x] Run a training experiment with the new multi-product agent.
    - [x] Implement a **Tiered Evaluation Baseline**:
        - [x] **Tier 1 (Business Baseline):** Verify the agent beats `avg_daily_revenue` (or existing pricing logic).
        - [x] **Tier 2 (Architectural Baseline):** Compare the multi-product agent's performance against independent agents for 5 representative products (2 high-volume, 3 low-volume).
    - [x] Verify that the agent learns a reasonable policy and that the training process is efficient.

### Evaluation Plan for Multi-Product Agent
This plan outlines the steps to verify the functionality of the new multi-product agent architecture.

#### Milestone 1: Data Refactoring Verification
*   **Goal:** Verify that the `src/data_utils.py` script correctly generates the required data artifacts.
**Status: Completed**

#### Milestone 2: Environment Verification
*   **Goal:** Verify that the `MultiProductPriceEnv` can be instantiated, reset, and stepped through correctly.
**Status: Completed**

#### Milestone 3: Training Pipeline Verification (Integration Test)
*   **Goal:** Verify that a full training run can be launched with the new architecture without crashing.
*   **Action:** Run the main training script for a small number of timesteps to ensure all components are integrated correctly.
*   **Command Example:** `python run_experiment.py --config-path configs/base_config.yaml --experiment-config-path configs/experiments/ppo_baseline.yaml --run-name test_multiproduct_run --total-timesteps 1000`
*   **Checks:**
    *   The script starts and prints messages indicating the multi-product setup.
    *   The `stable-baselines3` training progress bar is displayed.
    *   The training run completes successfully and saves a final model.
**Status: Completed**

#### Milestone 4: Full Evaluation Verification
*   **Goal:** Verify that the `evaluate_multi_product_agent.py` script can load a trained model and evaluate it across all products.
*   **Action:** Run the evaluation script on a trained model. The script is designed to iterate through all products automatically.
*   **Command Example:** `venv/bin/python evaluate_multi_product_agent.py --agent-path models/<run_name>/final_model.zip --episodes 1 --use-pre-aggregated-data`
*   **Checks:**
    *   The script loads the agent and data without errors.
    *   The evaluation loop runs for all products.
    *   A final "Evaluation Summary" and "Aggregate Performance" section are printed.
**Status: Completed**
