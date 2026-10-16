# SwarmX: Development History & Git Provenance

This document reconstructs the implementation timeline, commit milestones, architectural iterations, and working tree enhancements based on Git repository metadata, commit diffs, build logs, and file timestamps.

---

## 1. Commit Log Timeline & Metadata

The repository tracks 3 historical commits on the `main` branch, supplemented by an extensive working-tree enhancement phase:

```text
* f84c7a7 (HEAD -> main, origin/main) Remove team member names from documentation (Fri Sep 18 17:41:35 2026 +0300)
* 8066dce Remove team member names from architecture (Fri Sep 18 17:39:46 2026 +0300)
* 7a5e45e Initial commit: swarm car search system (Fri Sep 18 17:29:53 2026 +0300)
```

### Commit Detail:
1. **Commit `7a5e45e` (2026-09-18 17:29:53 +03:00)**:
   * **Commit Message**: `Initial commit: swarm car search system`
   * **Scope**: 84 files created, 10,604 insertions.
   * **Core Components Introduced**:
     * Initial ACO engine, `TaskScorer`, and `PheromoneMap` (tri-layer scalar fields).
     * Abstract contracts: `ISLAMProvider`, `IPerceptionProvider`, `IServerInterface`.
     * `MockSLAMProvider` with Bresenham raycasting and 8-connected frontier extraction.
     * Simulated robot kinematics, differential drive model, and collision halting.
     * Core robot finite state machine (FSM) and heartbeat failure detector.
     * 96 initial automated tests covering unit, integration, and packaging requirements.
     * Pure-Python SVG map visualizer and terminal ASCII HUD.
2. **Commit `8066dce` (2026-09-18 17:39:46 +03:00)**:
   * **Commit Message**: `Remove team member names from architecture`
   * **Scope**: Cleaned author attribution in `README.md`.
3. **Commit `f84c7a7` (2026-09-18 17:41:35 +03:00)**:
   * **Commit Message**: `Remove team member names from documentation`
   * **Scope**: Final documentation scrub in `README.md`.

---

## 2. Working Tree Evolution & Post-Commit Development (Sep 18 – Sep 30, 2026)

Following commit `f84c7a7`, substantial engineering development occurred directly in the working tree, evolving the codebase from a baseline 2D prototype into an advanced, multi-stage AI system with Gazebo Fortress 3D simulation and ROS 2 Humble integration.

### Summary of Working Tree Modifications (17 Modified Files):
* **`interfaces/perception_interface.py` (+33 / -3 lines)**: Added `TargetAppearanceProfile`, `appearance_confidence`, `plate_detection_confidence`, `character_confidences`, and multi-stage confidence fields to `MatchEvent`.
* **`perception/matching_engine.py` (+195 / -42 lines)**: Implemented full 5-stage weighted confidence fusion, integrating vehicle detection, plate localization, Levenshtein fuzzy string distance, appearance verification, and multi-frame temporal voting.
* **`perception/plate_recognizer.py` (+65 / -18 lines)**: Added character-level confidence tracking, distance attenuation, and `OCR_CONFUSION_MAP` ambiguity simulation ($8 \leftrightarrow B, 0 \leftrightarrow O, 2 \leftrightarrow Z$).
* **`perception/vehicle_detector.py` (+57 / -14 lines)**: Integrated `VehicleAppearanceVerifier` and dynamic bounding box scaling.
* **`simulation/scenarios.py` (+205 / -76 lines)**: Expanded deterministic scenario factory to cover Scenarios A through H (central target, occluded target, civilian decoys, optical noise, failure recovery, cooperative recruitment, distant target, and low-prior Bayesian propagation).
* **`swarm/aco/aco_config.py` (+26 / -6 lines)**: Added normalized component weights ($w_{\text{info}}, w_{\text{prior}}, w_{\text{hist}}, w_{\text{explor}}, w_{\text{dist}}, w_{\text{cong}}, w_{\text{risk}}, w_{\text{redun}}, w_{\text{battery}}$).
* **`swarm/aco/task_scorer.py` (+178 / -53 lines)**: Rewrote scoring engine into a normalized multi-factor synthesis function with explainability breakdowns (`explain_score`).
* **`swarm/core/swarm_coordinator.py` (+125 / -16 lines)**: Added Bayesian target belief propagation, spatial search history recording, inter-robot collision deconfliction, and macro mission states.
* **`swarm/failure_handling/recovery_manager.py` (+71 / -17 lines)**: Enhanced orphan task reassignment with crash site reservation pheromone injection.
* **`visualization/swarm_visualizer.py` (+93 / -26 lines)**: Enhanced multi-panel dashboard generation with headless Agg configuration.

### Untracked Additions Introduced in Working Tree:
1. **Containerized ROS 2 & Gazebo Fortress 3D Stack**:
   * [`docker/Dockerfile`](file:///Users/aziz/Desktop/swarm_car_search/docker/Dockerfile): ROS 2 Humble + Gazebo Fortress + Xvfb + fluxbox + x11vnc + noVNC.
   * [`ros2_ws/src/swarm_gazebo/`](file:///Users/aziz/Desktop/swarm_car_search/ros2_ws/src/swarm_gazebo/): ROS 2 package containing:
     * `coordinator_node.py`: ROS 2 SwarmX ACO coordinator.
     * `slam_bridge.py`: Gazebo LiDAR/Odom subscriber fulfilling `ISLAMProvider`.
     * `robot_controller.py`: Proportional waypoint tracker on `/{robot_id}/cmd_vel`.
     * `perception_bridge.py`: Raycast vision and OCR matching node.
     * `mission_monitor.py`: Real-time telemetry logger.
     * `urdf/swarm_robot.urdf.xacro`: URDF with diff-drive, 2D LiDAR, RGB camera, and IMU plugins.
     * `worlds/urban_search_world.world`: 50m x 50m 3D urban world with target vehicle.
     * Launch files: `gazebo_world.launch.py`, `swarm_simulation.launch.py`.
   * [`run_gazebo_gui.sh`](file:///Users/aziz/Desktop/swarm_car_search/run_gazebo_gui.sh) and [`run_simulation.sh`](file:///Users/aziz/Desktop/swarm_car_search/run_simulation.sh): Master launchers for Apple Silicon M1 and Linux.
2. **Advanced AI Perception Modules**:
   * [`perception/appearance_verifier.py`](file:///Users/aziz/Desktop/swarm_car_search/perception/appearance_verifier.py): Color RGB Euclidean distance and body aspect ratio verification.
   * [`perception/temporal_tracker.py`](file:///Users/aziz/Desktop/swarm_car_search/perception/temporal_tracker.py): Multi-frame spatial clustering and majority voting.
   * [`swarm/pheromone/target_probability_map.py`](file:///Users/aziz/Desktop/swarm_car_search/swarm/pheromone/target_probability_map.py): Continuous Bayesian belief grid.
   * [`swarm/task_allocation/search_history.py`](file:///Users/aziz/Desktop/swarm_car_search/swarm/task_allocation/search_history.py): Spatial trajectory memory.
3. **Benchmarking & Evaluation Framework**:
   * [`evaluation/ai_benchmarks.py`](file:///Users/aziz/Desktop/swarm_car_search/evaluation/ai_benchmarks.py) and [`scripts/evaluate_ai_benchmarks.py`](file:///Users/aziz/Desktop/swarm_car_search/scripts/evaluate_ai_benchmarks.py): Scenarios A through H evaluation suite.
   * Output artifacts generated on Sep 30, 2026: `output/ai_benchmark_report.json`, `output/gazebo_simulation_run/`, and `ros2_ws/log/build_2026-09-30_*/`.

---

## 3. Provenance of Historical Code Pre-dating Git History

Because commit `7a5e45e` was an "Initial commit" containing 84 files, development history prior to September 18, 2026 cannot be established from Git metadata alone. However, code headers and docstrings indicate that the mathematical foundation originated as an academic graduation thesis project focused on decentralized swarm robotics and bio-inspired search.
