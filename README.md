# Multi-Objective Microservice Placement using Deep Reinforcement Learning 

[![Python 3.7](https://img.shields.io/badge/python-3.7-blue.svg)](https://www.python.org/downloads/)
[![PaddlePaddle](https://img.shields.io/badge/PaddlePaddle-2.0+-orange.svg)](https://www.paddlepaddle.org.cn/)
[![PARL](https://img.shields.io/badge/PARL-1.3+-green.svg)](https://github.com/PaddlePaddle/PARL)

[English](#english)

---

<a name="english"></a>
## 🇬🇧 English Description

This repository implements a **Multi-Objective Deep Reinforcement Learning (DRL)** framework to optimize the placement of containerized microservices across edge/cloud nodes. Built upon **PaddlePaddle** and **PARL**, it employs a decomposed Deep Q-Network (DQN) approach to solve the placement problem by simultaneously optimizing three conflicting objectives: **Communication Cost**, **Load Balance**, and **Network Reliability**.

### 🚀 Key Features

* **Multi-Objective Optimization**: The agent learns to trade off between:
    1.  **Communication Cost**: Data transfer overhead between interacting services.
    2.  **Load Variance**: Balancing CPU and Memory usage across nodes.
    3.  **Reliability**: Minimizing network delay and packet loss.
* **Pareto-based Policy**: Implements a custom action selection strategy (`policy.py`) using **Pareto Dominance** to select optimal deployment nodes instead of a simple scalar reward sum.
* **Decomposed Architecture**: Utilizes three separate Q-networks (in `agent.py`) to estimate values for different objectives independently.
* **Custom Simulation**: Includes a flexible environment (`env.py`) modeling node resources (CPU, RAM, Bandwidth) and service dependency graphs.

### 📂 File Structure

| File | Description |
| :--- | :--- |
| `new_train.py` | **Main Entry Point**. Orchestrates the training loop, environment interaction, and logging. |
| `env.py` | **Environment**. Simulates nodes, resources, service chains, latency, and packet loss. |
| `agent.py` | **RL Agent**. Manages the interaction between the policy and the three underlying Q-networks. |
| `algorithm.py` | **DQN Implementation**. Standard DQN logic (loss calculation, target sync) using PaddlePaddle. |
| `model.py` | **Neural Network**. 3-layer fully connected network for Q-value approximation. |
| `policy.py` | **Action Selection**. Implements **Pareto Selection** and Weighted Sum strategies. |
| `pareto.py` | **Pareto Logic**. Utilities to calculate Pareto frontiers and check dominance. |
| `replay_memory.py` | **Experience Replay**. Buffer to store `(s, a, r1, r2, r3, s', done)` transitions. |
| `reward.py` | **Reward Engineering**. Normalization and calculation of specific reward signals. |
| `pareto_sum.py` | **Analysis**. Scripts to merge and analyze Pareto sets from logs. |
| `delete_script.py` | **Utility**. Helper to clean up log files and saved models. |

### 📦 Installation

Ensure you have Python 3.7 installed. Install the required dependencies:

```bash
pip install paddlepaddle parl numpy
```
### 🏃‍♂️ Usage

1. Start Training:
Run the main training script. You can optionally provide a run index argument (useful for batch experiments).

```bash
python new_train.py
```

2. Output Files:During execution, the generates several log files and artifacts:
- **feature-reward.log**: Records the reward values ($r_1, r_2, r_3$) and objective feature values for each episode.

- **pareto_set.txt**: Contains the set of non-dominated (optimal) solutions found during the training process.

- **mdqn_model.ckpt**: The saved model checkpoints after training.

- **pareto_details.txt**: A detailed log of when optimal solutions were added to or removed from the Pareto set.


3. Cleanup: To remove old log files and saved models to start a fresh training session, run the provided utility script:
```
python delete_script.py
```