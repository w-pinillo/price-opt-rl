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

---

## Correction: Multi-Product DRL Agent Objective Status (December 4, 2025)

**Context:** Following user feedback and a review of the codebase (specifically `src/envs/price_env.py`), it was identified that "Objective 6 - Implement a Multi-Product DRL Agent" was erroneously marked as "Completed" in `objectives.md`.

**Issue Identified:** The `PriceEnv` environment still operates on a single-product DataFrame, uses a flat observation space, and lacks the logic for product ID mapping and sampling as outlined in Objective 6's "Optimal Strategy" and "Implementation To-Do List." All checkboxes in the to-do list for this objective were found to be unchecked.

**Action Taken:**
- The status of "Objective 6 - Implement a Multi-Product DRL Agent" in `objectives.md` has been corrected from "Completed" to "In Progress".
- This entry has been added to the `GEMINI.md` log to accurately reflect the project's development history.

**Next Steps:** The implementation of Objective 6 will proceed according to the detailed plan in `objectives.md`, starting with the environment modifications.

---

## Multi-Product DRL Agent: Integration Test Completion (December 4, 2025)

**Context:** Successful completion of Milestone 3 for "Objective 6 - Implement a Multi-Product DRL Agent," verifying the training pipeline integration.

**Issues Encountered & Resolved:**
-   **Initial Error (`mat1 and mat2 shapes cannot be multiplied`):** This mismatch occurred between the output of `CustomFeatureExtractor` and the input of the policy network.
    -   **Root Cause:** The `net_arch` parameter in `policy_kwargs` was forcing a fixed, incorrect input dimension on the MLP, overriding the `features_dim` inferred from `CustomFeatureExtractor`.
    -   **Solution:** Removed the `net_arch` key from `agent_config` in `src/models/train_agent.py` to allow `stable-baselines3` to correctly infer the policy network's input dimension (34) from the feature extractor's output.
-   **Subsequent Error (`Sizes of tensors must match except in dimension 1`):** This error arose during concatenation within `CustomFeatureExtractor`, indicating an issue with batch dimensions.
    -   **Root Cause:** `stable-baselines3` was automatically one-hot encoding the `Discrete` observation for `product_id` as `(Batch_Size, 1, Num_Products)` (e.g., `[1024, 1, 100]`). The previous `argmax(dim=1)` was then incorrectly applied, resulting in mismatched tensor shapes during concatenation.
    -   **Solution:** Modified the `forward` method in `src/models/custom_feature_extractor.py`. Implemented `product_ids_raw.squeeze(1)` to remove the extra dimension, transforming the tensor to `(Batch_Size, Num_Products)`. Subsequently, `th.argmax(..., dim=1)` was applied to correctly extract product indices, ensuring `product_embed` (shape `[Batch_Size, embedding_dim]`) and `market_features` (shape `[Batch_Size, market_features_dim]`) had consistent dimensions for successful concatenation.

**Outcome:**
- The training pipeline now runs successfully without crashing, completing the integration test for the multi-product DRL agent. This resolves all major tensor shape and dimension mismatches encountered during the integration phase.

---

## Data Pipeline Memory Issue Recurrence (December 4, 2025)

**Context:** Following the implementation of memory optimization solutions for the data pipeline (detailed in `PROGRESS_LOG.md` on December 4, 2025), an attempt to re-run `src/pipeline.py` resulted in a critical Out-Of-Memory (OOM) error, causing a VSCode shutdown.

**Issue Identified:** Despite previous efforts to refactor `src/features.py` and `src/pipeline.py` to use lazy processing with Polars `LazyFrame` objects, and explicit deletion of DataFrames with garbage collection, the data pipeline continues to exhaust available RAM on resource-constrained systems (e.g., 16GB RAM). This indicates that the current memory optimization strategies are insufficient to handle the dataset size and processing requirements without OOM errors.

**Impact:** The inability to successfully run the data pipeline due to persistent OOM errors is a critical blocker for further development, including proceeding with the diagnosis of `nan` values and subsequent agent training.

**Next Steps:** Thoroughly investigate the current memory consumption patterns during pipeline execution to identify remaining bottlenecks. Further optimization of data loading, processing, and memory management is required before proceeding with other tasks.

---

## Data Pipeline Refactoring to Resolve OOM Errors (December 5, 2025)

**Context:** The data pipeline continued to fail with Out-Of-Memory (OOM) errors on a 16GB RAM machine, even after implementing lazy processing with Polars. The root cause remained the initial loading and aggregation of the 307M-row raw transaction dataset, which was too memory-intensive for the hardware.

**Solution Implemented:**
-   **Introduced Configurable Data Source:** A new `data_config` section was added to `configs/base_config.yaml`.
    -   `use_pre_aggregated_data` (boolean): A flag to switch between processing raw data and using a pre-generated, aggregated file.
    -   `pre_aggregated_data_path` (string): Path to the aggregated dataset (`data/processed/top100_daily.parquet`).
-   **Refactored `src/pipeline.py`:**
    -   The script now reads the `use_pre_aggregated_data` flag.
    -   If `true`, the pipeline bypasses the memory-intensive raw data loading, product selection, and daily aggregation steps entirely. It starts directly by loading the `top100_daily.parquet` file.
    -   If `false`, the original (but OOM-prone) raw data processing logic is executed, providing backward compatibility.
-   **Polars API Corrections:** Several minor bugs in the Polars-based feature generation code were fixed, including incorrect method calls (`.fill_inf`, `.suffix`) that caused errors during the pipeline run.

**Outcome:**
- The critical OOM blocker has been **resolved**. By setting `use_pre_aggregated_data: true`, the data pipeline now runs successfully and efficiently on resource-constrained (16GB RAM) systems, generating the necessary train, validation, and test datasets without issue.

**Next Steps:**
- With the data pipeline now stable, the project can proceed with **Milestone 4 of Objective 6**: the full evaluation of the multi-product agent.