# Autonomous Swarm Outdoor Car Search System

[![Python Version](https://img.shields.io/badge/python-3.9%20%7C%203.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue.svg)](https://www.python.org/)
[![Test Suite](https://img.shields.io/badge/tests-96%2B%20passing-brightgreen.svg)]()
[![Architecture](https://img.shields.io/badge/architecture-decentralized%20ACO-orange.svg)]()
[![Dependencies](https://img.shields.io/badge/core%20dependencies-zero%20(standard%20library)-purple.svg)]()
[![License](https://img.shields.io/badge/license-MIT-green.svg)]()

> **Graduation Project**: Decentralized AI Multi-Robot Coordination for Missing & Stolen Vehicle Search in Outdoor Urban Environments.

---

## 1. Subsystem Engineering Division

To ensure clean engineering separation and parallel development across the graduation project team:

| Subsystem | Engineers Responsible | Core Responsibilities |
| :--- | :--- | :--- |
| **Swarm Coordination & Task Allocation** | **Mohammed & Ibrahim** | Ant Colony Optimization (ACO), stigmergic pheromone fields, multi-objective task scoring, Boltzmann Softmax allocation, dynamic failure recovery, vehicle perception matching. |
| **SLAM & Localization** | **Saqr & Mahmoud** | Sensor fusion (LiDAR, IMU, wheel encoders), robot pose estimation, 2D occupancy mapping, obstacle inflation, frontier cluster extraction. |

> **Interface Contract**: The Swarm subsystem is decoupled from low-level SLAM via the [`ISLAMProvider`](interfaces/slam_interface.py) abstract interface. For integration guidelines, consult the [SLAM Integration Guide](docs/slam_integration_guide.md).

---

## 2. System Architecture & Information Flow

```
[Sensors: LiDAR / IMU / Encoders / Camera]
                  │
                  ▼
          [SLAM Subsystem]
      (SLAM Module)
                  │
                  ▼ (SLAMState via ISLAMProvider)
     [Swarm Coordination Engine]
      (ACO-Based Swarm Coordination)
        ├── Tri-Layer Stigmergic Map (Exploration, Recruitment, Avoidance)
        ├── Multi-Objective Task Scorer (Information Gain, Priors, Distance, Congestion)
        ├── Dynamic Failure Detector & Recovery Manager
        └── Boltzmann / Softmax Probabilistic Task Allocator
                  │
                  ▼ (Target Waypoints & Motion Commands)
      [Robot Onboard Controller]
        ├── Differential Drive Kinematics Model
        ├── Collision Halting & Obstacle Avoidance
        └── Perception & Fuzzy License Plate OCR Matcher
                  │
                  ▼ (Target Found & Mission Debrief)
        [Live HUD / SVG / GIF / Webhook Dashboard]
```

---

## 3. Key Algorithmic Innovations

1. **Multi-Layer Stigmergic Pheromones ($\tau_e, \tau_s, \tau_a$)**:
   Indirect spatial memory maintained across three discrete layers with customized evaporation dynamics:
   - **Exploration Field ($\tau_e$)**: Slow evaporation ($\rho = 0.01$) ensures long-term memory of mapped avenues.
   - **Recruitment Field ($\tau_s$)**: Medium evaporation ($\rho = 0.05$) attracts neighboring robots to verified target sightings.
   - **Avoidance Field ($\tau_a$)**: Fast evaporation ($\rho = 0.30$) repels robots to eliminate redundant search and prevent congestion.

2. **Scale-Invariant Multi-Objective Task Scorer**:
   $$\text{Score}(t_i, r_j) = \frac{[\tau_i]^\alpha \cdot [\eta_i]^\beta \cdot [P_i]^\gamma \cdot [G_i]^\delta}{[C(r_j, t_i)]^\varepsilon \cdot [\Omega_i]^\zeta \cdot [K_i]^\theta}$$
   Balances frontier size ($\eta$), land-use prior probabilities ($P$), travel distance ($C$), robot congestion ($\Omega$), and avoidance pheromones ($K$).

3. **Boltzmann Softmax Action Selection**:
   Prevents premature swarm convergence onto single targets via temperature-scaled probabilistic distribution with overflow protection.

4. **Fault-Tolerant Dynamic Resilience**:
   Autonomous detection of silent heartbeats, stalled wheels, and low battery. Orphaned tasks are immediately reclaimed and reassigned while depositing avoidance pheromones over immobilized wreckage.

5. **Perception & Levenshtein Fuzzy OCR Matching**:
   Combines 2D raycast field-of-view (FOV) occlusion checks with normalized Levenshtein string similarity to recognize target license plates under noisy camera conditions.

---

## 4. Quickstart Guide

### 4.1 System Prerequisites
- Python 3.9+ (Zero external dependencies required for core execution).
- Optional visualization extras: `pip install -r requirements.txt` (for Matplotlib PNG dashboards & animated GIFs).

### 4.2 Unified Command-Line Interface (CLI)
Use the unified launcher [`scripts/main.py`](scripts/main.py) to run any system command:

```bash
# 1. Display System Specification & Architecture
python3 scripts/main.py info

# 2. Run the Complete Automated Test Suite (96+ tests)
python3 scripts/main.py test

# 3. Launch an Autonomous Swarm Urban Search Simulation
python3 scripts/main.py sim --robots 4 --steps 100 --output-dir output/simulation_run

# 4. Execute Comparative Monte Carlo Benchmarks vs Baselines
python3 scripts/main.py eval --trials 5 --steps 60 --robots 4 --output-dir output/benchmark_run
```

---

## 5. Comparative Evaluation & Benchmark Results

Our proposed **ACO** coordination strategy was benchmarked against two classic baselines across 5 independent Monte Carlo trials in a $50\text{m} \times 50\text{m}$ urban environment with 4 autonomous ground vehicles and simulated hardware failure:

| Strategy | Area Coverage (%) | Trajectory Overlap (%) | Mean Travel Distance (m) | Fault Recovery Latency |
| :--- | :---: | :---: | :---: | :---: |
| **Proposed ACO** | **18.0 ± 0.9%** | **0.0 ± 0.0%** | **42.7 ± 1.8 m** | **< 0.2 s (Dynamic)** |
| **Nearest Frontier (Greedy)** | 18.2 ± 0.0% | 0.0 ± 0.0% | 46.7 ± 0.0 m | N/A (Failed) |
| **Random Walk (Uncoordinated)** | 17.3 ± 0.8% | 0.0 ± 0.0% | 37.3 ± 1.6 m | N/A |

**Key Finding**: The proposed ACO method achieved identical coverage while saving **8.6% in total travel distance** compared to Nearest Frontier, while maintaining dynamic fault recovery.

---

## 6. Project Directory Layout

```
swarm_car_search/
├── configs/                  # System, simulation, and ACO YAML configuration files
│   ├── aco_params.yaml
│   ├── default_config.yaml
│   └── simulation_config.yaml
├── docs/                     # Technical, thesis, and interface documentation
│   ├── thesis_chapter_swarm.md   # Complete Academic Graduation Thesis Chapter
│   ├── slam_integration_guide.md # Integration Guide for Saqr & Mahmoud
│   ├── user_guide.md             # Comprehensive User & Developer Guide
│   ├── architecture.md           # System Architecture & AI Specification
│   ├── aco_algorithm.md          # Theoretical Algorithmic Formulation
│   └── slam_interface_spec.md    # Formal SLAM Contract Specification
├── interfaces/               # Formal abstract base classes (ISLAMProvider, etc.)
├── slam_providers/           # Mock and concrete SLAM implementations
├── perception/               # Vehicle detection, plate OCR, and fuzzy matching
├── swarm/                    # Swarm Intelligence Core Subsystem
│   ├── aco/                  # ACO Engine, Task Scorer, Pheromone Map
│   ├── core/                 # Robot Agent, Swarm Coordinator, Swarm State
│   ├── task_allocation/      # Task Manager, Allocation Strategies
│   ├── communication/        # RF communication simulation & message models
│   ├── failure_handling/     # Failure Detector & Recovery Manager
│   └── states/               # Robot FSM & State Machine
├── simulation/               # 2D discrete-time physics, kinematics & obstacle generation
├── visualization/            # ASCII HUD, pure SVG vector rendering & Matplotlib plots
├── evaluation/               # Metrics collector, baselines, and benchmark suite
├── scripts/                  # Command-line entry points (main.py, run_simulation.py, run_evaluation.py)
├── tests/                    # Automated test suites (100% passing)
├── pyproject.toml            # PEP 517/621 package distribution specification
├── setup.py                  # Standard setuptools packaging script
├── requirements.txt          # Python dependencies (core zero-dep + optional extras)
└── README.md                 # Project landing page
```

---

## 7. Thesis Documentation References

- [Complete Graduation Thesis Chapter](docs/thesis_chapter_swarm.md)
- [SLAM Subsystem Integration Guide](docs/slam_integration_guide.md)
- [Comprehensive User & Developer Guide](docs/user_guide.md)
- [Theoretical ACO Algorithmic Design](docs/aco_algorithm.md)
- [System Architecture Specification](docs/architecture.md)

---

## 8. License

This project is licensed under the MIT License - see the `LICENSE` file for details.
