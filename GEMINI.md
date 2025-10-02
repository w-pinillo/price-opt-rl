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

## Objective 5 — Master's Thesis Redaction

**Status:** In Progress

**Details:**
- The master's thesis will be written in English.

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