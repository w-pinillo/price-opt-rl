
## Objective 5 - Hardware Acceleration and Experimentation

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

### Memory Optimization for Resource-Constrained Environments

**Context:** During experimentation on a system with 16GB RAM and 16GB VRAM (compared to a 32GB RAM development environment), out-of-memory (OOM) issues were encountered. Analysis revealed that excessive RAM consumption was primarily due to high parallelization settings.

**Root Causes Identified:**
-   **High `n_envs` in Experiment Configurations:** Baseline configurations for DQN (`dqn_baseline.yaml`) and PPO (`ppo_baseline.yaml`) used `n_envs` values of 4 and 16 respectively. When `n_envs > 1`, `stable-baselines3` utilizes `SubprocVecEnv`, spawning a separate Python process for each environment. Each process contributes its own memory footprint (Python interpreter overhead, environment data, etc.), leading to rapid RAM exhaustion, especially with 16 parallel processes.
-   **Large DQN Replay Buffer:** The `buffer_size` of `1,000,000` for DQN in `dqn_baseline.yaml` demanded significant system RAM to store past experiences.
-   **Parallel Hyperparameter Optimization:** The `optimize_agent.py` script's default `n_jobs=2` for Optuna trials meant that multiple `train_agent.py` processes were launched concurrently, further exacerbating memory pressure.

**Solutions Implemented (December 2, 2025):**
-   **Reduced `n_envs`:**
    -   `configs/experiments/dqn_baseline.yaml`: `n_envs` reduced from `4` to `1`.
    -   `configs/experiments/ppo_baseline.yaml`: `n_envs` reduced from `16` to `1`.
    *Rationale:* This ensures individual training runs utilize only one environment instance, drastically lowering memory consumption per training job.
-   **Reduced DQN `buffer_size`:**
    -   `configs/experiments/dqn_baseline.yaml`: `buffer_size` reduced from `1,000,000` to `100,000`.
    *Rationale:* Significantly cuts down RAM usage for the DQN agent's experience replay buffer.

**Recommendations for Future Experiments and Hyperparameter Optimization:**
-   **Limit Optuna Parallelization:** When running `optimize_agent.py`, always explicitly set `--n-jobs 1` (or a very low number if system resources permit) to prevent concurrent `train_agent.py` processes. Example:
    ```bash
    python optimize_agent.py --agent dqn --n-trials 20 --n-jobs 1
    ```
-   **Monitor GPU Usage:** Regularly use `nvidia-smi` during training to confirm that the GPU is actively utilized and VRAM is being managed effectively. Although `device: cuda` is configured, computations can silently fall back to CPU if GPU resources are not correctly accessed, leading to system RAM pressure.
-   **Further Parameter Tuning (if needed):** If OOM issues persist, consider reducing `batch_size` (e.g., from 4096) and, for DQN, exploring smaller `buffer_size` values within the Optuna search space (e.g., `10,000` to `50,000`).

**Optimality Assessment:**
- These changes address the immediate OOM issues by significantly reducing the memory footprint of individual training runs and parallel optimization trials. While some parallel training benefits are reduced, stability on resource-constrained hardware is prioritized, enabling successful completion of experiments. Further tuning can be explored as hardware capabilities improve.
