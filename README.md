# COLIGHT-LITE

**Team Members:**
- Ankita Girish Aswathanarayana
- Prajnadipta Kundu

## Overview

**CoLight-Lite** is a lightweight multi-agent reinforcement learning framework for cooperative traffic signal optimization across connected intersections.
Inspired by CoLight (Wei et al., 2019), this implementation develops a simplified and reproducible version suitable for experimentation and course research.

The system integrates:

* **A custom multi-intersection simulator for College Station, Texas**
* **Independent RL controllers (DQN)**
* **Cooperative RL using network structure (CoLight-Lite)**
* **Real-time visualization and performance logging**

This project was developed as part of the course **Deep Reinforcement Learning**.

## Key Features

* **Graph-Aware RL Approximation:**
A simplified CoLight-inspired design where adjacency influences Q-values via pooled neighbor state effects instead of full attention layers.

* **Modular Multi-Intersection Simulator:**
A custom Python environment modeling stochastic vehicle arrivals, queue buildup & dissipation and phase switching & timing

* **Independent vs. Cooperative RL Baselines:**
Side-by-side training pipelines for independent DQNs and cooperative agents, enabling fair comparison.

* **Performance Metrics:**
The system logs episode-level rewards and enables quantitative comparison between fixed-time, independent RL, and CoLight-Lite controllers.

* **Visualization Tools:**
A simple Pygame interface shows queue growth and signal changes at intersections to help illustrate policy behavior


## Project Architecture

CoLight-Lite follows a modular architecture to support clarity, extensibility, and experimentation:

* **Environment Simulation:**
Custom Python environments (single_intersection_env.py, multi_intersection_cstat.py) implement queue dynamics, stochastic arrivals, and signal logic.

* **Agent Models:**
  - *Independent DQN agents (dqn_agent.py)* 
  - *CoLight-Lite cooperative agent (colight_agent_new.py, colight_network.py)* 

* **Training & Evaluation:**
Scripts for running fixed baselines, training RL models, and evaluating learned policies:
  - *run_multi_fixed.py*
  - *train_independent_multi.py, train_colight_multi.py*
  - *eval_independent_multi.py, eval_colight_multi.py*

* **Neighbor Influence (Cooperation):**
The cooperative agent aggregates state information from adjacent intersections using adjacency-conditioned state pooling to approximate CoLight-style coordination.

* **Inference Loop:**
During runtime, each intersection selects actions, updates Q-values, and synchronizes reward statistics to enable consistent evaluation across baselines.

* **Visualization:**
-*visualize_colight.py* — animated simulation viewer
-*plot_graphs.py* — produces comparison plots
-*comparison_plot.png* — generated performance figure
  
## Setup & Installation

This guide walks you through installing and running CoLight-Lite, a simplified multi-agent reinforcement learning framework for traffic signal control.

### Prerequisites
* Python 3.10 (recommended)
* Git
* (Optional) A GPU machine if you want faster RL training

1. **Clone the repository:**
```bash
git clone https://github.com/ankitaga00/drl-project.git
cd drl-project
```
2. **Create and activate a virtual environment**
```bash
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate

```
3. **Install Python dependencies**
```bash
pip install -r requirements.txt
```

## Running the Project

The project includes three traffic control approaches:

* Fixed-Time Controller
* Independent DQN Agents
* CoLight-Lite Cooperative RL

1. **Run the Fixed-Time Baseline**
```bash
python src/run_multi_fixed.py
```
This executes the non-learning baseline using simple periodic timing.

2. **Train the Independent RL Agents**
```bash
python src/train_independent_multi.py
```
This trains four independent DQN agents (one per intersection).


After training, the models are saved in:
```
models/
 ├── agent_0.pth
 ├── agent_1.pth
 ├── agent_2.pth
 └── agent_3.pth
```

3. **Evaluate the Independent RL Performance**
```bash
python src/eval_independent_multi.py
```

4. **Train the CoLight-Lite Cooperative Agent** (Optional step)
```bash
python src/train_colight_multi.py
```
This trains a graph-aware cooperative RL controller inspired by CoLight.

```
Model will be saved as:
models/trained_colight.pth

Evaluate CoLight Performance:
python src/eval_colight_multi.py
```
5. **Visualize the Traffic Flow Simulation**
```bash
python src/visualize_colight.py
```

This displays a lightweight 2D simulation showing:

1. Signal switching (red → green)

2. Queue lengths evolving over time

Note: This visualization is intentionally simple and designed for clarity—not SUMO-level fidelity.

### Comparing All Approaches

**To generate a bar plot comparing average evaluation performance of all approaches:**
```bash
python src/plot_comparison_curves.py
```

Generates a bar graph in PNG format



