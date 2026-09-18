# Autonomous Swarm Outdoor Car Search: Comprehensive User & Developer Guide

**Project Title**: Autonomous Multi-Robot Swarm for Missing Vehicle Search & Identification in Outdoor Urban Environments  
**Target Audience**: Graduation Thesis Committee, Evaluators, Robotics Researchers, and System Operators  
**Authors**: Mohammed & Ibrahim (Swarm Coordination Subsystem)  
**Version**: 1.0.0 (Production Release)  

---

## 1. System Requirements & Environment Setup

### 1.1 Prerequisites
- **Operating System**: macOS (Ventura / Sonoma / Sequoia) or Linux (Ubuntu 20.04 / 22.04 / 24.04).
- **Python Version**: Python 3.9 or higher.

### 1.2 Zero External Dependencies Guarantee
The entire autonomous swarm software stack—including the Ant Colony Optimization (ACO) algorithm, task manager, stigmergic pheromone fields, simulated vehicle detection, fuzzy OCR matching, live ASCII HUD, pure-Python SVG vector rendering, and statistical evaluation framework—operates with **ZERO EXTERNAL DEPENDENCIES** on standard Python (e.g. `/usr/bin/python3`).

### 1.3 Optional Graphical Extensions
For generating high-resolution Matplotlib dashboard panels (`.png`) and animated multi-frame mission GIFs (`.gif`), install the optional visualization extras:
```bash
pip install -r requirements.txt
# Or explicitly:
pip install matplotlib numpy pillow pyyaml
```

---

## 2. Package Installation & Command-Line Interface (CLI)

### 2.1 Package Installation (Optional Development Mode)
You can install the package in editable mode:
```bash
pip install -e .
```
This registers the global console commands `swarm`, `swarm-sim`, and `swarm-eval`.

### 2.2 Direct Launcher Usage (Recommended)
Alternatively, you can run all commands directly via the unified launcher script:
```bash
python3 scripts/main.py <command> [options]
```

| Command | Purpose | Example |
| :--- | :--- | :--- |
| `info` | Displays project architecture, team boundaries, and metadata | `python3 scripts/main.py info` |
| `sim` | Executes multi-robot urban search simulation with visual rendering | `python3 scripts/main.py sim --robots 4 --steps 100` |
| `eval` | Runs Monte Carlo benchmark comparing ACO against baselines | `python3 scripts/main.py eval --trials 5 --steps 60` |
| `test` | Executes the complete 96+ automated test suite | `python3 scripts/main.py test` |

---

## 3. Running Single-Mission Simulations (`swarm sim`)

### 3.1 Quickstart Simulation
To launch a 4-robot swarm search in a $50\text{m} \times 50\text{m}$ urban environment:
```bash
python3 scripts/main.py sim --robots 4 --steps 100 --output-dir output/mission_01
```

### 3.2 Key Command-Line Parameters for `sim`
- `--robots <int>`: Fleet size of autonomous ground vehicles (default: `4`).
- `--steps <int>`: Total simulation ticks to execute (default: `100`, step $\Delta t = 0.2\text{s}$).
- `--width <float>` / `--height <float>`: Dimensions of urban search area in meters (default: `50.0`).
- `--target-plate <str>`: License plate of suspect vehicle to locate (default: `DXB-88392`).
- `--target-x <float>` / `--target-y <float>`: Coordinates where target car is parked (default: `(36.0, 36.0)`).
- `--inject-failure-tick <int>`: Simulation tick at which a hardware failure is triggered (default: `35`).
- `--fail-robot <str>`: Robot ID targeted for failure injection (default: `robot_1`).
- `--render-every <int>`: Interval of ticks between saved visual frames (default: `10`).

### 3.3 Simulation Outputs
All artifacts are saved to `--output-dir`:
1. `final_mission_summary.svg`: A standalone vector graphic showing the complete urban layout, buildings, final robot positions, and target location.
2. `final_mission_summary.png` (if Matplotlib available): High-resolution 3-panel dashboard (Global Map, Tri-Layer Pheromone Map, Mission Telemetry).
3. `swarm_mission_animation.gif` (if PIL available): Animated timeline showing the swarm exploring streets and recovering from faults.

---

## 4. Running Benchmark Evaluations (`swarm eval`)

### 4.1 Quickstart Evaluation
To run 5 Monte Carlo trials comparing **ACO** against **Nearest Frontier** and **Random Walk**:
```bash
# Using standard Python (produces JSON, Markdown, and SVG curves):
python3 scripts/main.py eval --trials 5 --steps 60 --robots 4 --output-dir output/benchmark_run

# Using graphical Python (additionally produces PNG curves and bar charts):
MPLCONFIGDIR=/tmp /usr/local/bin/python3 scripts/main.py eval --trials 5 --steps 60 --robots 4 --output-dir output/benchmark_run
```

### 4.2 Benchmark Deliverables
1. `benchmark_report.md`: Pre-formatted Markdown report ready to be included directly in your graduation thesis.
2. `benchmark_results.json`: Machine-readable raw metrics, trial histories, and standard deviations.
3. `coverage_comparison.svg`: Pure vector comparative coverage curves over time.
4. `coverage_comparison.png` & `metrics_comparison.png`: Publication-grade comparative figures.

---

## 5. System Configuration Guide

Configuration files are located in `configs/`:

### 5.1 `configs/aco_params.yaml`
Controls the core Ant Colony Optimization parameters:
```yaml
# Pheromone Layer Evaporation Rates (rho in [0, 1])
evaporation_exploration: 0.01   # Low evaporation for long-term map memory
evaporation_success: 0.05       # Medium evaporation for target attraction
evaporation_avoidance: 0.30     # High evaporation for dynamic collision avoidance

# Multi-Objective Task Scorer Exponents
alpha: 1.0    # Positive pheromone sensitivity
beta: 1.5     # Information gain (frontier size) sensitivity
gamma: 1.2    # Land-use vehicle prior probability sensitivity
delta: 1.0    # Mission urgency sensitivity
epsilon: 1.5  # Distance cost penalty
zeta: 2.0     # Quadratic congestion penalty
theta: 1.5    # Avoidance pheromone repulsion sensitivity

# Action Selection
temperature: 0.5            # Boltzmann Softmax exploration temperature
epsilon_greedy: 0.05        # Epsilon-greedy exploration rate
```

---

## 6. Graduation Defense FAQ & Speaking Points

### Q1: Why did you choose Ant Colony Optimization (ACO) over Greedy Yamauchi Frontier Exploration?
**Answer**: Greedy exploration assigns robots to the physically closest frontier. In multi-robot teams, this causes severe clustering: multiple robots converge on the same street corner, creating massive trajectory overlap and redundant sensor scanning. ACO incorporates **avoidance pheromones ($\tau_a$)** and **congestion penalties ($\Omega$)**, which naturally scatter the swarm across different city blocks without requiring a central coordinator.

### Q2: What happens if a robot's motor or battery dies in the middle of a mission?
**Answer**: Our `FailureDetector` continuously monitors heartbeat silence and motion progress. If an agent goes silent for more than $3.0\text{s}$ or stalls, the `RecoveryManager` flags the robot as inactive, deposits an avoidance pheromone marker over its coordinates (so peers navigate around the wreck), releases its assigned task back to `PENDING`, and triggers immediate ACO re-allocation among healthy peers.

### Q3: How is your system decoupled from the SLAM team (Saqr & Mahmoud)?
**Answer**: We defined a strict abstract contract: `ISLAMProvider`. Our Swarm decision engine only consumes high-level `SLAMState` (poses, occupancy grid, and frontier clusters) through this interface. Saqr & Mahmoud can implement their SLAM module using any ROS 2 stack (Cartographer, RTAB-Map, Nav2) without modifying a single line of our Swarm code.
