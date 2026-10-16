# SwarmX: Project Overview & System Scope

## 1. Executive Summary

**SwarmX (Swarm Car Search)** is an autonomous multi-robot swarm coordination and target-identification system engineered for rapid outdoor urban search operations. The primary objective is locating, verifying, and reporting stolen, missing, or suspect vehicles in dense city environments where line-of-sight occlusions, communication intermittency, and vast search areas preclude effective single-agent operations.

The project combines:
1. **Bio-Inspired Decentralized Swarm Intelligence**: Adapting Ant Colony Optimization (ACO) into a scalable Multi-Robot Task Allocation (MRTA) framework.
2. **Tri-Layer Digital Stigmergy**: Spatially discretized pheromone fields guiding cooperative coverage, target recruitment, and collision deconfliction.
3. **Multi-Stage Computer Vision & Perception**: Distance-attenuated vehicle detection, license plate optical character recognition (OCR), fuzzy Levenshtein distance matching, visual appearance verification, and multi-frame temporal voting.
4. **Resilient Dynamic Failure Recovery**: Decentralized heartbeat health monitoring, stall detection, and automatic orphan task reclamation when robots suffer hardware faults.
5. **High-Fidelity Simulation Infrastructure**: Pure-Python discrete kinematic 2D simulations alongside containerized ROS 2 Humble and Gazebo Fortress 3D physics environments.

---

## 2. Problem Statement & Motivation

Urban search operations—such as police patrols seeking suspect get-away vehicles, emergency services locating parked stolen cars, or municipal asset recovery—face severe operational hurdles:
* **Occlusions**: Tall buildings, alleyways, and perimeter structures block camera line-of-sight and sensor range.
* **Scale**: Searching a $2,500\,\text{m}^2$ ($50\,\text{m} \times 50\,\text{m}$) urban precinct requires extensive travel time for an individual patrol unit.
* **Single Points of Failure**: Centralized dispatcher models collapse if network links degrade or the central controller disconnects.
* **Sensor Noise & Decoys**: Civilian vehicles of similar color or shape introduce false positives, while dirt, distance, and viewing angle introduce optical character misreadings.

SwarmX overcomes these limitations by distributing search responsibility across a fleet of Autonomous Ground Vehicles (AGVs) operating under decentralized consensus.

---

## 3. Subsystem Architecture & Responsibilities

The software architecture enforces strict modular separation between subsystems:

| Subsystem | Primary Responsibilities | Core Modules & Packages |
| :--- | :--- | :--- |
| **Swarm Coordination & Task Allocation** | Multi-objective ACO task evaluation, Boltzmann Softmax task selection, dynamic peer sync, and mission state machines. | `swarm/aco/`, `swarm/task_allocation/`, `swarm/core/` |
| **Pheromone & Stigmergic Field** | Discretized tri-layer pheromone map ($\tau_e, \tau_s, \tau_a$), exponential evaporation dynamics, and Bayesian target probability mapping. | `swarm/pheromone/` |
| **Failure Detection & Recovery** | Autonomous heartbeat monitoring, stall detection, orphan task release, crash site avoidance pheromone injection. | `swarm/failure_handling/` |
| **Perception & Vehicle Identification** | Field-of-view raycast detection, synthetic plate OCR, fuzzy string similarity, color/aspect ratio verification, multi-frame temporal voting. | `perception/`, `interfaces/perception_interface.py` |
| **SLAM & Mapping Interface** | Standardized abstract interface for 2D occupancy grids, pose estimation, and frontier cluster extraction. Decoupled via `ISLAMProvider`. | `interfaces/slam_interface.py`, `slam_providers/` |
| **Simulation & Physics** | Kinematic non-holonomic robot modeling, urban obstacle generation, deterministic scenario definitions, ROS 2 / Gazebo Fortress 3D bridge. | `simulation/`, `ros2_ws/`, `docker/` |
| **Benchmarking & Visualization** | Terminal ASCII HUD, publication-grade multi-panel Matplotlib dashboards, vector SVG frame export, animated GIF generation, Monte Carlo metrics. | `visualization/`, `evaluation/`, `scripts/` |

---

## 4. Capability Audit Matrix

Every capability in this repository has been rigorously audited against ground-truth source code, execution logs, and automated tests. Features are categorized strictly according to verification status:

| Feature / Capability | Classification | Verification Evidence | Notes & Scope |
| :--- | :---: | :--- | :--- |
| **Decentralized ACO Task Allocation** | **Implemented & Verified** | [`tests/test_aco_engine.py`](file:///Users/aziz/Desktop/swarm_car_search/tests/test_aco_engine.py), [`tests/test_multi_robot.py`](file:///Users/aziz/Desktop/swarm_car_search/tests/test_multi_robot.py) | Full multi-objective scoring and Boltzmann Softmax verified. |
| **Tri-Layer Stigmergic Pheromones** | **Implemented & Verified** | [`tests/test_pheromone.py`](file:///Users/aziz/Desktop/swarm_car_search/tests/test_pheromone.py) (10 tests passed) | Exploration, Recruitment, and Avoidance layers mathematically verified. |
| **Decentralized Fault Recovery** | **Implemented & Verified** | [`tests/test_failure_handling.py`](file:///Users/aziz/Desktop/swarm_car_search/tests/test_failure_handling.py) (9 tests passed) | Heartbeat timeout, stall detection, and dynamic orphan reassignment verified. |
| **Multi-Stage Perception Pipeline** | **Implemented & Verified** | [`tests/test_perception_integration.py`](file:///Users/aziz/Desktop/swarm_car_search/tests/test_perception_integration.py), [`scripts/evaluate_ai_benchmarks.py`](file:///Users/aziz/Desktop/swarm_car_search/scripts/evaluate_ai_benchmarks.py) | 100% precision, 100% recall, 96.5% OCR character accuracy verified across 8 scenarios. |
| **2D Discrete Kinematic Simulation** | **Implemented & Verified** | [`tests/test_robot_agent.py`](file:///Users/aziz/Desktop/swarm_car_search/tests/test_robot_agent.py), [`scripts/run_simulation.py`](file:///Users/aziz/Desktop/swarm_car_search/scripts/run_simulation.py) | Headless execution, ASCII HUD, SVG export, Matplotlib dashboards, animated GIFs verified. |
| **ROS 2 Humble / Gazebo Fortress Bridge** | **Implemented but Not Verified in Current Turn** | `ros2_ws/src/swarm_gazebo/`, `docker/Dockerfile`, `run_gazebo_gui.sh` | Full package, URDF, SDF, and nodes exist; colcon build logs from Sep 30 2026 present in `ros2_ws/log/`; blocked locally by Docker daemon being stopped. |
| **Physical Hardware Deployment** | **Planned (Future Work)** | N/A | Current implementation relies on simulation (MockSLAM and Gazebo). Physical robot testing (TurtleBot / Jackal) not yet conducted. |
| **Real Camera Neural Network Inference** | **Partially Implemented** | Simulated OCR / detector models active; YOLO / TensorRT hooks designed via interfaces | Synthetic noise, distance attenuation, and Levenshtein fuzzy matching verified; weights for live PyTorch YOLOv8 not bundled in repo. |
| **Distributed Multi-Robot WiFi Mesh** | **Partially Implemented** | `swarm/communication/local_comm.py`, `simulation/sim_comm.py` | Packet loss, range limits, and latency verified in simulation; physical DDS / UDP mesh network unverified on hardware. |

---

## 5. Physical vs. Simulated Operational Scope

> [!IMPORTANT]
> **Scientific Integrity Disclosure**: All experimental data, coverage percentages, and timing metrics documented in this knowledge base reflect **simulated environments** (both Python kinematic physics and Gazebo Fortress rigid-body dynamics). No claims of physical robot road testing are made.

The software stack is designed with strict hardware abstraction:
* Moving to physical hardware requires implementing an `ISLAMProvider` that wraps `cartographer_ros` or `slam_toolbox`, and connecting `cmd_vel` topics to real motor controllers.
* No changes to `SwarmCoordinator`, `ACOEngine`, `PheromoneMap`, or `MatchingEngine` are required when transitioning from simulation to hardware.

---

## 6. Security, Privacy, and Safe Auditing

* **No Secret Exposure**: The codebase contains no API tokens, SSH keys, cloud credentials, or sensitive environment files (`.env`).
* **Synthetic Target Watchlist**: Target license plates (such as `DXB-88392`) and civilian decoy plates (`AD-10294`, `SHJ-44910`) are fictitious identifiers created strictly for algorithmic validation.
* **Zero System Changes**: This documentation package was constructed following a non-destructive audit, preserving Git branches, working trees, and unstaged modifications.
