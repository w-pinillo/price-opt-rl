# Overview / Purpose

Implement a full pipeline that goes from raw transactional data to a trained DRL agent (DQN and PPO), unified DRL agent capable of learning an optimal pricing policy for all top 100 selling products simultaneously, and a robust evaluation comparing the learned pricing policies against historical baselines. The pipeline should be reproducible, modular, and documented.

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
Provide a comprehensive evaluation of the DRL pricing agents. The evaluation includes improvement, adaptability, and reliability.