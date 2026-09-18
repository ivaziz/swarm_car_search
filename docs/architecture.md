# System Architecture & AI Engineering Specification

## 1. Executive Overview

This project implements an autonomous swarm robotics system designed to cooperatively search for, identify, and report stolen or missing vehicles in outdoor urban environments. The system relies on a decentralized multi-robot coordination framework grounded in Ant Colony Optimization (ACO), integrating perception (vehicle detection & OCR) with Simultaneous Localization and Mapping (SLAM).

## 2. Team Subsystem Boundaries

To ensure clean engineering separation and parallel development across the graduation project team:

| Subsystem | Engineers Responsible | Core Responsibilities |
| :--- | :--- | :--- |
| **Swarm Coordination & Task Allocation** | Mohammed & Ibrahim | ACO engine, task scoring, frontier selection, pheromone updates, decentralized consensus, failure recovery, dynamic re-assignment. |
| **SLAM & State Estimation** | Saqr & Mahmoud | Sensor fusion (LiDAR, IMU, Wheel Encoders), pose estimation, 2D occupancy mapping, obstacle inflation, frontier extraction. |
| **Perception & Vision** | Unified Interface | Vehicle detection (YOLO/CNN), license plate cropping, OCR engine, watchlist matching. |

### Architectural Constraint
> **Important**: The Swarm subsystem **never** implements SLAM internally and **never** relies on hardcoded internal maps. All environmental state flows exclusively through the standardized `ISLAMProvider` interface.

---

## 3. High-Level AI Pipeline Flow

```
[Sensors: LiDAR / IMU / Encoders / Camera]
                  │
                  ▼
          [SLAM Subsystem]
      (Saqr & Mahmoud's Module)
                  │
                  ▼ (SLAMState via ISLAMProvider)
      [Swarm Decision Layer]
      (Mohammed & Ibrahim's ACO Engine)
                  │
                  ▼ (Velocity & Target Commands)
      [Robot Motion Controller]
                  │
                  ▼
          [Perception Loop]
    (Vehicle Detection -> OCR -> Match Engine)
                  │
                  ▼ (MatchEvent / Priority Updates)
      [Swarm Coordinator & Cloud Dashboard]
```

---

## 4. Software Package Structure

```
swarm_car_search/
├── configs/                  # YAML configurations (ACO params, simulation, system)
├── docs/                     # Architectural, algorithmic, and interface specifications
├── interfaces/               # Formal abstract base classes (contracts)
│   ├── slam_interface.py
│   ├── perception_interface.py
│   └── server_interface.py
├── slam_providers/           # Concrete SLAM providers (MockSLAM for test, ROS2 for real)
├── perception/               # Vision events, detectors, and OCR matchers
├── swarm/                    # Swarm Intelligence Subsystem
│   ├── aco/                  # ACO Engine, Pheromone Map, Task Scorer
│   ├── core/                 # Robot Agent, Swarm Coordinator, Swarm State
│   ├── task_allocation/      # Frontier Tasks, Task Allocation Strategy
│   ├── communication/        # Inter-robot communication protocols
│   └── states/               # Robot Finite State Machine (FSM)
├── simulation/               # 2D discrete-time physics and sensor simulation
├── visualization/            # Real-time dashboard and Matplotlib visualizer
├── evaluation/               # Benchmarks, baselines (Random, Nearest Frontier), metrics
├── tests/                    # Automated unit, integration, and interface tests
├── scripts/                  # Entry points for simulation and evaluation
├── requirements.txt
└── README.md
```
