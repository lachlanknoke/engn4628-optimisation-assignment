# Mine Haulage Cost Optimisation

This repository contains the code and report for a research project that formulates and solves a network-flow optimisation model to minimise the cost of transporting ore from pits to crushers in an open-pit coal mine. The model introduces a novel "wear-aware throttling" strategy, deliberately slowing trucks on non-critical routes to reduce fuel and maintenance costs without sacrificing overall throughput.

**Key Finding:** By optimising for cost instead of pure speed, the total cost of meeting crusher demand was reduced by **4.92%**.

## Project Overview

In traditional mine haulage optimisation, truck speed is often treated as a fixed parameter. However, there is a significant trade-off: higher speeds reduce cycle times but exponentially increase fuel consumption and mechanical wear. This project challenges the conventional approach by treating speed as a decision variable that directly influences the cost function.

We model the mine's road network as a directed graph where each road segment (arc) has:
*   A **cost-per-tonne** that is a function of speed, road quality, and distance.
*   A **mass-flow capacity** that is also a function of speed and road conditions.

The objective is to find the optimal flow of material (tonnes per day) through this network to meet the crushers' demands at the minimum total cost, subject to road capacity constraints.

## Methodology

### 1. Problem Formulation
The core of the project is a **Linear Programming (LP) minimum-cost network-flow model**.
*   **Objective Function:** Minimise the total cost of transport.
    \[
    \min \sum_{(i,j)\in A} V_{i\rightarrow j} \cdot c_{i\rightarrow j}
    \]
*   **Constraints:**
    *   Flow conservation at all intermediate nodes.
    *   Supply constraints at mine nodes.
    *   Demand constraints at crusher nodes.
    *   Arc capacity constraints based on road properties and speed.

### 2. Cost & Capacity Models
The innovation lies in the detailed, speed-aware models for cost and capacity.

*   **Cost Function (`c_{i→j}`):** Combines fuel and maintenance costs, each with calibrated elasticities for speed and payload. Faster speeds on a given road lead to super-linear increases in cost.
    *   **Fuel Exponent (δ):** 1.22
    *   **Maintenance Exponent (γ):** 1.70

*   **Capacity Function:** Determines the maximum daily tonnage a road can handle, influenced by speed limits, gradient, and road quality (straightness and condition).

### 3. Implementation
The model is implemented in Python using the **Gurobi** optimiser. The code sets up the network, defines the objective and constraints, and solves for the optimal flow.

### Prerequisites
*   Python 3.7+
*   [Gurobi Optimiser](https://www.gurobi.com/) with a valid license.
*   Python packages: `gurobipy`, `numpy`

### Installation & Usage

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/mine-haulage-optimisation.git
    cd mine-haulage-optimisation
    ```

2.  **Install required packages:**
    ```bash
    pip install gurobipy numpy
    ```

3.  **Run the optimisation:**
    Navigate to the `src` directory and run the script.
    ```bash
    cd src
    python cost_optimisation.py
    ```
    The script will output the optimal cost and the mass flow through each road segment.

## 📊 Results

The model was tested on a network based on the Curragh Mine in Queensland's Bowen Basin.

| Optimisation Goal | PERT Estimated Cost |
| :--- | :--- |
| **Minimum Cost (Proposed)** | **\$6,659.8** |
| Minimum Time (Baseline) | \$7,004.2 |

**Result:** The cost-optimised solution **reduced daily costs by 4.92%** compared to the speed-optimised baseline. This was achieved by rerouting traffic away from high-cost, high-speed roads and utilising "throttled" paths where slower speeds led to significant savings in fuel and wear.

## 🔮 Future Work

This model provides a strong foundation for several extensions:
*   **Discretisation:** Reformulating the problem as a Mixed-Integer Linear Program (MILP) to schedule individual trucks over time.
*   **Uncertainty Modelling:** Incorporating stochastic elements like truck breakdowns, weather, and fluctuating fuel prices.
*   **Dynamic Speed Control:** Making speed a direct decision variable in the optimisation for even finer control.
*   **New Mine Planning:** Using the model to optimise the initial layout of haul roads in new mine sites.

## Report

The full academic report is included in this repository. It provides a comprehensive background, detailed derivation of the models, a complete sensitivity analysis, and a discussion of the results and their industrial implications.

## Authors

*   **Christopher Tsirbas** (u7143682)
*   **Lachlan Knoke** (u7298351)
*   **Chithi Gunatilake** (u7637857)
*   **Joshua Dunn** (u5365782)

This project was completed for **ENGN4628 Optimisation and Control with Uncertainty and Constraints**.


Declaration: Chatgpt was used in this repository