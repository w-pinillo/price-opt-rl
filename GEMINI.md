# Gemini Project Log

## Objective 1 - Data Preparation

**Status:** Completed

**Details:**
- Raw data loading, initial cleaning, top-100 product selection, daily aggregation, and time series feature generation have been successfully implemented and integrated into the `src/pipeline.py`.
- The processed data is saved to `data/processed/top100_daily.parquet` and the list of top product IDs to `data/processed/top100_ids.json`.
- Numeric features have been scaled using `StandardScaler` and saved to `models/scalers/`.
- The data has been temporally split into train, validation, and test sets, and saved as `train_scaled.parquet`, `val_scaled.parquet`, and `test_scaled.parquet` respectively in `data/processed/`.

**Optimality Assessment:**
- The data preparation pipeline is now fully implemented as per the requirements of Objective 1. It is modular, reproducible, and handles large datasets efficiently using Polars. The temporal split ensures proper evaluation for time-series data, and feature scaling prepares the data for DRL agents.

## Objective 2 — Define the MDP and implement the simulation environment

**Status:** Completed

**Details:**
- Design decisions for action type (discrete), reward formulation (revenue), and demand simulator (parametric log-linear elasticity model) have been made and configured in `config.yaml`.
- The OpenAI Gym compatible environment `src/envs/price_env.py` has been implemented, including `reset`, `step`, `render`, and `seed` methods.
- The `src/envs/simulators.py` file contains the `ParametricDemandSimulator` class.
- The environment successfully runs a full episode with a random policy without NaNs or unexpected negative rewards.
- A sanity check confirmed that increasing price tends to reduce predicted demand, validating the demand simulator.

**Optimality Assessment:**
- The simulation environment is now functional and meets the requirements of Objective 2. It provides a stable and configurable platform for training DRL agents.

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

**Status:** Completed

**Details:**
- An evaluation script, `src/evaluation.py`, was created to perform a backtest of the trained agents against a "do-nothing" baseline across all 100 top-selling products.
- The script calculates key performance metrics including total revenue, average daily revenue, average price, and price volatility for each agent and product.
- The detailed results are saved to `reports/tables/metrics_summary.csv`.
- A new Jupyter Notebook, `notebooks/02-Evaluation.ipynb`, has been created to analyze and visualize the results.
- The notebook includes a summary table of aggregated metrics, a bar chart comparing total revenues, and a bootstrapping analysis to confirm the statistical significance of the agents' performance uplift over the baseline.
- Both DQN and PPO agents demonstrated a significant, multi-fold increase in total revenue compared to the baseline, with PPO performing slightly better.

**Optimality Assessment:**
- The evaluation pipeline is complete, providing a robust and statistically validated comparison of the trained agents against a baseline. The results are clearly documented and visualized, fulfilling the requirements of Objective 4.

**Next Steps:** All primary project objectives have been met. The next logical step is to integrate these findings and methodologies into the final thesis document.

## LaTeX Thesis Template Guide

This section summarizes how to work with the LaTeX template located in the `trabajo-final/` directory.

- **Main File:** The main file is `0000.tex`. All other `.tex` files are included from here.
- **Compilation:** To compile the document, run the following commands from within the `trabajo-final/` directory:
    1. `pdflatex -shell-escape 0000.tex`
    2. `bibtex 0000`
    3. `pdflatex -shell-escape 0000.tex`
    4. `pdflatex -shell-escape 0000.tex`
- **Dependencies:**
    - The `-shell-escape` flag is required for the `minted` package (used for code highlighting).
    - The `lmodern` package is required for the `microtype` package to work correctly.
    - You may need to install language packs for `babel`, for example: `sudo apt-get install texlive-lang-spanish`.
- **Packages:** Additional LaTeX packages can be added in `0000.tex` using the `\usepackage{}` command.
- **Comments:** Use the `%` character to add comments to the `.tex` files.
- **Document Structure:** Use the following commands to structure the document:
    - `\chapter{Chapter Name}`
    - `\section{Section Name}`
    - `\subsection{Subsection Name}`
    - `\subsubsection{Subsubsection Name}`
    - `\paragraph{Paragraph Name}`
- **Paragraphs:** Use `\par` to start a new paragraph with the correct spacing.
- **Images:** 
    - Use the `figure` environment to include images that should be listed in the "Lista de Figuras".
    - Store image files in the `00Figuras/` directory.
    - `\includegraphics` to insert the image and `\caption` to add a caption.
- **Tables:**
    - Use the `table` environment to include tables that should be listed in the "Lista de Tablas".
    - The guide suggests using online tools like [Table Generator](https://www.tablesgenerator.com/) to create complex tables.
- **Lists:**
    - Use the `itemize` environment for bulleted lists.
    - Use the `enumerate` environment for numbered lists.
- **Bibliography:** Add new references to the `Referencias.bib` file.