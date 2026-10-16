# SwarmX: Source Code Reference & Module Catalog

This document provides a comprehensive, file-by-file inventory of every software module, class, interface, and function in the SwarmX codebase.

---

## 1. System Interfaces (`interfaces/`)

The `interfaces/` package defines strict abstract contracts using Python's `abc.ABC`. These contracts decouple subsystems, ensuring that the swarm coordination layer never depends directly on internal SLAM, perception, or server implementations.

### 1.1 `interfaces/slam_interface.py`
* **File Path**: [`interfaces/slam_interface.py`](file:///Users/aziz/Desktop/swarm_car_search/interfaces/slam_interface.py)
* **Responsibility**: Establishes data structures and contracts for SLAM state estimation and frontier mapping.
* **Classes**:
  * `Pose2D(x: float, y: float, theta: float)` (Lines 11–26): Immutable 2D pose dataclass in global Cartesian coordinates.
  * `OccupancyGrid2D` (Lines 29–48): 2D probabilistic occupancy grid (`width`, `height`, `resolution`, `origin: Pose2D`, `data: List[int]`). Grid values: `-1` (unknown), `0` (free), `100` (occupied).
  * `FrontierCluster` (Lines 51–64): Clustered boundary dividing explored free space from unexplored unknown space (`cluster_id`, `centroid_x`, `centroid_y`, `bounding_box`, `size`, `cell_indices`).
  * `SLAMState` (Lines 67–79): Aggregated localization snapshot (`robot_id`, `pose`, `grid`, `frontiers`, `timestamp`, `is_valid`).
  * `ISLAMProvider(ABC)` (Lines 82–99): Abstract contract requiring `get_latest_state(robot_id: str) -> SLAMState` and `is_healthy(robot_id: str) -> bool`.

### 1.2 `interfaces/perception_interface.py`
* **File Path**: [`interfaces/perception_interface.py`](file:///Users/aziz/Desktop/swarm_car_search/interfaces/perception_interface.py)
* **Responsibility**: Defines vision event structures, bounding box dataclasses, and match event notifications.
* **Classes**:
  * `BoundingBox2D` (Lines 13–18): Normalized image bounding box (`ymin`, `xmin`, `ymax`, `xmax`).
  * `TargetAppearanceProfile` (Lines 21–27): Expected target signature (`vehicle_class: "sedan"`, `color: "dark_gray"`, `color_rgb: (55, 55, 60)`, `aspect_ratio: 2.2`).
  * `VehicleDetectionEvent` (Lines 30–46): Triggered when candidate vehicle is spotted in camera feed.
  * `PlateDetectionEvent` (Lines 49–62): Triggered when plate OCR extracts text from crop (`raw_ocr_text`, `ocr_confidence`, `character_confidences`).
  * `MatchEvent` (Lines 65–85): Triggered when candidate passes multi-stage verification against target watchlist (`match_confidence`, `confirmed`, `vehicle_confidence`, `plate_confidence`, `ocr_confidence`, `appearance_confidence`, `temporal_confidence`, `frame_count`, `stage_breakdown`).
  * `IPerceptionProvider(ABC)` (Lines 88–102): Abstract contract for perception pipelines (`detect_vehicles`, `recognize_plate`, `match_watchlist`).

### 1.3 `interfaces/server_interface.py`
* **File Path**: [`interfaces/server_interface.py`](file:///Users/aziz/Desktop/swarm_car_search/interfaces/server_interface.py)
* **Responsibility**: Defines ground dispatch and telemetry reporting contracts for mission operators.
* **Classes**:
  * `AlertPriority(Enum)` (Lines 11–16): `INFO`, `WARNING`, `CRITICAL`.
  * `VehicleAlert` (Lines 19–27): High-priority notification dispatched upon target identification.
  * `MissionStatusReport` (Lines 30–37): Periodic swarm health and coverage status summary.
  * `IServerInterface(ABC)` (Lines 40–49): Abstract dispatch contract (`send_alert`, `send_telemetry`).

---

## 2. SLAM Providers (`slam_providers/`)

### 2.1 `slam_providers/base_slam_provider.py`
* **File Path**: [`slam_providers/base_slam_provider.py`](file:///Users/aziz/Desktop/swarm_car_search/slam_providers/base_slam_provider.py)
* **Responsibility**: Provides thread-safe state caching and health monitoring for `ISLAMProvider` implementations.
* **Classes**:
  * `BaseSLAMProvider(ISLAMProvider)` (Lines 13–41): Maintains a thread-safe dictionary `_latest_states: Dict[str, SLAMState]` and validates update freshness.

### 2.2 `slam_providers/mock_slam_provider.py`
* **File Path**: [`slam_providers/mock_slam_provider.py`](file:///Users/aziz/Desktop/swarm_car_search/slam_providers/mock_slam_provider.py)
* **Responsibility**: High-performance simulated SLAM engine for unit testing and headless Python simulations without ROS 2.
* **Algorithms**:
  * Bresenham Line Raycasting (`raycast_line`, Lines 24–47): Traces LiDAR beams from robot pose up to `max_lidar_range = 15.0m`, marking free cells along the beam and occupied cells upon obstacle impact.
  * Frontier Extraction & Clustering (`extract_frontiers`, Lines 110–170): Scans the local occupancy grid for free cells that neighbor unknown cells (`-1`), applies 8-connected grid clustering, and produces `FrontierCluster` objects.

---

## 3. Perception & Computer Vision (`perception/`)

The perception subsystem implements a robust 5-stage verification architecture designed to eliminate false positive civilian vehicle reports and handle real-world optical noise.

### 3.1 `perception/appearance_verifier.py`
* **File Path**: [`perception/appearance_verifier.py`](file:///Users/aziz/Desktop/swarm_car_search/perception/appearance_verifier.py)
* **Responsibility**: Verifies physical vehicle attributes (color RGB distance and body aspect ratio) independent of plate readability.
* **Classes**:
  * `VehicleAppearanceVerifier` (Lines 22–78):
    * `verify_appearance(observed_class, observed_color)`: Computes normalized Euclidean color distance in RGB space against the target profile (`dark_gray` = `(55, 55, 60)`). Rejects non-matching vehicle classes (e.g., trucks/SUVs receive class penalty).

### 3.2 `perception/temporal_tracker.py`
* **File Path**: [`perception/temporal_tracker.py`](file:///Users/aziz/Desktop/swarm_car_search/perception/temporal_tracker.py)
* **Responsibility**: Multi-frame temporal consensus engine that tracks stationary vehicle detections across successive simulation ticks.
* **Classes**:
  * `CandidateTrack` (Lines 18–44): Stores history of sightings within spatial gating radius (`association_radius = 4.0m`).
  * `TemporalTracker` (Lines 47–120):
    * `add_observation(...)`: Associates observations using spatial distance gating, maintains a sliding window of readings, executes majority character voting across historical frames, and requires $k \ge 3$ consecutive confirmations before confirming the target.

### 3.3 `perception/vehicle_detector.py`
* **File Path**: [`perception/vehicle_detector.py`](file:///Users/aziz/Desktop/swarm_car_search/perception/vehicle_detector.py)
* **Responsibility**: Simulates camera field-of-view (FoV) scanning, distance-attenuated detection confidence, and raycast obstacle occlusion.
* **Classes**:
  * `TargetVehicle` (Lines 17–31): Ground-truth vehicle dataclass (`target_id`, `plate_number`, `vehicle_class`, `color`, `position`, `is_stolen`).
  * `SimulatedVehicleDetector(IVehicleDetector)` (Lines 43–198):
    * Line-of-sight check (`_has_line_of_sight`, Lines 60–92): Raycasts between robot $(x, y)$ and target $(x, y)$ against `UrbanEnvironment` obstacles.
    * FoV calculation: Verifies bearing angle $|\Delta\theta| \le \frac{\text{fov}}{2} = 45^\circ$ and Euclidean distance $\le 14.0\,\text{m}$.
    * Generates `VehicleDetectionEvent` with distance attenuation: $C_{\text{veh}} = \text{clamp}(0.50, 0.99, C_{\text{base}} \cdot (1 - \frac{d}{1.5 \cdot d_{\max}}))$.

### 3.4 `perception/plate_recognizer.py`
* **File Path**: [`perception/plate_recognizer.py`](file:///Users/aziz/Desktop/swarm_car_search/perception/plate_recognizer.py)
* **Responsibility**: Simulates license plate cropping and OCR character decoding, introducing optical blur and ambiguity at range $> 7.0\,\text{m}$.
* **Constants**: `OCR_CONFUSION_MAP = {'8': 'B', 'B': '8', '0': 'O', 'O': '0', '2': 'Z', 'Z': '2'}`.
* **Classes**:
  * `SimulatedPlateRecognizer(IPlateRecognizer)` (Lines 26–95): Generates `PlateDetectionEvent` with per-character confidence scores and distance-dependent character substitutions.

### 3.5 `perception/matching_engine.py`
* **File Path**: [`perception/matching_engine.py`](file:///Users/aziz/Desktop/swarm_car_search/perception/matching_engine.py)
* **Responsibility**: Multi-stage classification engine comparing recognized plates and visual signatures against active target watchlists.
* **Algorithms**:
  * Normalized Levenshtein Distance (`levenshtein_distance`, Lines 16–35): Dynamic programming calculation of edit distance between alphanumeric strings.
  * Levenshtein Similarity (`levenshtein_similarity`, Lines 38–43): $S_{\text{plate}} = 1.0 - \frac{D(s_1, s_2)}{\max(|s_1|, |s_2|)}$.
  * Multi-Stage Verification (`evaluate_multi_stage_match`, Lines 95–215): Fuses $C_{\text{veh}}, C_{\text{plate}}, C_{\text{ocr}}, S_{\text{plate}}, C_{\text{app}}, C_{\text{temp}}$ into composite target confidence $C_{\text{target}}$. Classifies detection as `CONFIRMED`, `CANDIDATE`, or `NON_TARGET`.

---

## 4. Swarm Intelligence & Ant Colony Optimization (`swarm/`)

### 4.1 ACO Configuration (`swarm/aco/aco_config.py`)
* **File Path**: [`swarm/aco/aco_config.py`](file:///Users/aziz/Desktop/swarm_car_search/swarm/aco/aco_config.py)
* **Responsibility**: Encapsulates all hyperparameters, exponents, and weights governing task evaluation and Boltzmann selection.
* **Dataclasses**:
  * `WeightConfig` (Lines 11–32): Exponents $\alpha, \beta, \gamma, \delta, \varepsilon, \zeta, \theta$ and normalized component weights ($w_{\text{info}}, w_{\text{prior}}, w_{\text{hist}}, w_{\text{explor}}, w_{\text{dist}}, w_{\text{cong}}, w_{\text{risk}}, w_{\text{redun}}$).
  * `PheromoneConfig` (Lines 35–45): Evaporation rates $\rho_e = 0.01, \rho_s = 0.05, \rho_a = 0.30$, bounds $[\tau_{\min}, \tau_{\max}]$, deposit quantities $Q_e, Q_s, Q_a$.
  * `StochasticConfig` (Lines 48–54): Exploration temperature $T = 0.5$, `deterministic: bool = False`.
  * `DynamicUrgencyConfig` (Lines 57–62): Multipliers for target confirmation ($3.5\times$) and vehicle investigation ($2.2\times$).
  * `ACOConfig` (Lines 65–85): Aggregated master configuration object.

### 4.2 ACO Task Scorer (`swarm/aco/task_scorer.py`)
* **File Path**: [`swarm/aco/task_scorer.py`](file:///Users/aziz/Desktop/swarm_car_search/swarm/aco/task_scorer.py)
* **Responsibility**: Computes strictly positive scalar attractiveness scores for candidate tasks evaluated from a robot's pose.
* **Classes**:
  * `TaskScorer` (Lines 22–175):
    * `explain_score(...)`: Evaluates normalized positive and negative factors, incorporating spatial search memory and Bayesian target priors.
    * `compute_score(...)`: Returns scalar attractiveness score for task allocation.

### 4.3 ACO Engine (`swarm/aco/aco_engine.py`)
* **File Path**: [`swarm/aco/aco_engine.py`](file:///Users/aziz/Desktop/swarm_car_search/swarm/aco/aco_engine.py)
* **Responsibility**: Executes Boltzmann Softmax sampling to assign candidate search tasks without greedy swarm clustering.
* **Classes**:
  * `ACOEngine` (Lines 24–165):
    * `select_task(...)`: Filters candidate tasks, computes score vector, evaluates numerical-stable Softmax probabilities $P(t_i) = \frac{\exp((\text{Score}_i - \text{Score}_{\max})/T)}{\sum_k \exp((\text{Score}_k - \text{Score}_{\max})/T)}$, and samples winning task.

---

## 5. Pheromone & Spatial Memory (`swarm/pheromone/`)

### 5.1 Evaporation Model (`swarm/pheromone/evaporation_model.py`)
* **File Path**: [`swarm/pheromone/evaporation_model.py`](file:///Users/aziz/Desktop/swarm_car_search/swarm/pheromone/evaporation_model.py)
* **Responsibility**: Mathematical decay dynamics across distinct pheromone layers.
* **Classes**:
  * `PheromoneLayerType(Enum)`: `EXPLORATION` ($\tau_e$), `SUCCESS` ($\tau_s$), `AVOIDANCE` ($\tau_a$).
  * `EvaporationModel`: Implements exponential decay $\tau(t + \Delta t) = (1 - \rho) \cdot \tau(t)$ with value clamping.

### 5.2 Pheromone Map (`swarm/pheromone/pheromone_map.py`)
* **File Path**: [`swarm/pheromone/pheromone_map.py`](file:///Users/aziz/Desktop/swarm_car_search/swarm/pheromone/pheromone_map.py)
* **Responsibility**: Discretized 2D grid storing scalar intensity fields for each pheromone layer.
* **Classes**:
  * `PheromoneMap`: Manages 2D grid layers with coordinate conversion (`world_to_grid`, `grid_to_world`) and bounding box aggregation (`get_box_aggregate`).

### 5.3 Pheromone Updater (`swarm/pheromone/pheromone_updater.py`)
* **File Path**: [`swarm/pheromone/pheromone_updater.py`](file:///Users/aziz/Desktop/swarm_car_search/swarm/pheromone/pheromone_updater.py)
* **Responsibility**: Applies Gaussian radial deposits and point deposits to pheromone layers upon robot movement or vehicle detection.
* **Classes**:
  * `PheromoneUpdater`:
    * `deposit_success(x, y, amount, radius)`: Deposits target recruitment trail.
    * `deposit_avoidance(x, y, amount, radius)`: Deposits reservation field.
    * `step_evaporation(dt)`: Advances time decay across all layers.

### 5.4 Bayesian Target Probability Map (`swarm/pheromone/target_probability_map.py`)
* **File Path**: [`swarm/pheromone/target_probability_map.py`](file:///Users/aziz/Desktop/swarm_car_search/swarm/pheromone/target_probability_map.py)
* **Responsibility**: Continuous spatial grid tracking Bayesian target belief $P(\text{target at } x, y)$.
* **Methods**:
  * `deposit_positive_evidence(x, y, confidence, radius)`: Elevates target belief upon vehicle sighting.
  * `deposit_negative_evidence(x, y, decay_factor, radius)`: Decays belief in thoroughly surveyed empty avenues.

---

## 6. Task Allocation & History (`swarm/task_allocation/`)

### 6.1 Search Task Dataclasses (`swarm/task_allocation/frontier_task.py`)
* **File Path**: [`swarm/task_allocation/frontier_task.py`](file:///Users/aziz/Desktop/swarm_car_search/swarm/task_allocation/frontier_task.py)
* **Classes**:
  * `TaskStatus(Enum)`: `PENDING`, `ASSIGNED`, `IN_PROGRESS`, `COMPLETED`, `ABANDONED`, `FAILED`.
  * `TaskType(Enum)`: `EXPLORATION`, `VEHICLE_INVESTIGATION`, `TARGET_CONFIRMATION`.
  * `SearchTask`: Represents a discrete search goal with spatial coordinates, bounding box, size, priority, and pheromone values.

### 6.2 Task Manager (`swarm/task_allocation/task_manager.py`)
* **File Path**: [`swarm/task_allocation/task_manager.py`](file:///Users/aziz/Desktop/swarm_car_search/swarm/task_allocation/task_manager.py)
* **Responsibility**: Central registry for tasks; handles frontier ingestion, spatial deduplication, lifecycle transitions, and pruning.

### 6.3 Spatial Search History (`swarm/task_allocation/search_history.py`)
* **File Path**: [`swarm/task_allocation/search_history.py`](file:///Users/aziz/Desktop/swarm_car_search/swarm/task_allocation/search_history.py)
* **Responsibility**: Tracks historical robot trajectory coverage and detects redundant search overlaps.

### 6.4 Allocation Strategies (`swarm/task_allocation/allocation_strategy.py`)
* **File Path**: [`swarm/task_allocation/allocation_strategy.py`](file:///Users/aziz/Desktop/swarm_car_search/swarm/task_allocation/allocation_strategy.py)
* **Classes**:
  * `ITaskAllocationStrategy(ABC)`: Interface contract for task allocators.
  * `ACOAllocationStrategy`: Wraps `ACOEngine` for bio-inspired stochastic task selection.
  * `NearestFrontierAllocationStrategy`: Greedy baseline allocator selecting Euclidean-nearest task.

---

## 7. Robot State Machine & Agent Core (`swarm/states/`, `swarm/core/`)

### 7.1 Robot State Machine (`swarm/states/robot_states.py`, `state_machine.py`)
* **File Paths**: [`swarm/states/robot_states.py`](file:///Users/aziz/Desktop/swarm_car_search/swarm/states/robot_states.py), [`swarm/states/state_machine.py`](file:///Users/aziz/Desktop/swarm_car_search/swarm/states/state_machine.py)
* **States**: `IDLE`, `NAVIGATING`, `SEARCHING`, `VEHICLE_DETECTED`, `VERIFYING`, `REPORTING`, `REASSIGNING`, `RETURNING`, `COMMUNICATION_LOST`, `COMPLETED`, `FAILED`.
* **Classes**: `RobotStateMachine` enforces transitions via `VALID_TRANSITIONS` lookup table.

### 7.2 Robot Agent (`swarm/core/robot_agent.py`)
* **File Path**: [`swarm/core/robot_agent.py`](file:///Users/aziz/Desktop/swarm_car_search/swarm/core/robot_agent.py)
* **Responsibility**: Autonomous entity representing an individual physical or simulated robot.
* **Classes**:
  * `RobotAgent`: Coordinates local state machine, onboard perception loop, waypoint tracking, battery monitoring, and heartbeat generation.

### 7.3 Swarm Coordinator (`swarm/core/swarm_coordinator.py`)
* **File Path**: [`swarm/core/swarm_coordinator.py`](file:///Users/aziz/Desktop/swarm_car_search/swarm/core/swarm_coordinator.py)
* **Responsibility**: Master orchestrator integrating all fleet agents, pheromone maps, failure detectors, and mission telemetry.

---

## 8. Failure Detection & Fault Recovery (`swarm/failure_handling/`)

### 8.1 Failure Detector (`swarm/failure_handling/failure_detector.py`)
* **File Path**: [`swarm/failure_handling/failure_detector.py`](file:///Users/aziz/Desktop/swarm_car_search/swarm/failure_handling/failure_detector.py)
* **Failure Triggers**:
  1. Heartbeat Timeout: Silence exceeding dynamic threshold ($10.0\,\text{s}$).
  2. Battery Depletion: Charge level falling below critical threshold ($< 0.10$).
  3. Motion Stall: Position delta $< 0.1\,\text{m}$ over $8.0\,\text{s}$ while in `NAVIGATING` state.
  4. Explicit Hardware Fault: Emergency fault alert from motor/sensor drivers.

### 8.2 Recovery Manager (`swarm/failure_handling/recovery_manager.py`)
* **File Path**: [`swarm/failure_handling/recovery_manager.py`](file:///Users/aziz/Desktop/swarm_car_search/swarm/failure_handling/recovery_manager.py)
* **Recovery Workflow**:
  1. Reclaims orphaned tasks from the failed robot.
  2. Re-inserts tasks into `TaskManager` pool with status `PENDING`.
  3. Deposits high avoidance pheromone ($\tau_a$) at the failure site to prevent peer collisions with the disabled robot.

---

## 9. Simulation & Scenarios (`simulation/`)

### 9.1 Environment & Obstacles (`simulation/environment.py`, `obstacle_generator.py`)
* **File Paths**: [`simulation/environment.py`](file:///Users/aziz/Desktop/swarm_car_search/simulation/environment.py), [`simulation/obstacle_generator.py`](file:///Users/aziz/Desktop/swarm_car_search/simulation/obstacle_generator.py)
* **Features**: 2D discrete grid, obstacle bounding boxes, navigable street avenues, and zones with prior target probabilities.

### 9.2 Scenarios Factory (`simulation/scenarios.py`)
* **File Path**: [`simulation/scenarios.py`](file:///Users/aziz/Desktop/swarm_car_search/simulation/scenarios.py)
* **Scenarios Defined**:
  * `easy_target` (Scenario A): Target at central intersection (25.0, 25.0).
  * `hidden_target` (Scenario B): Target occluded behind eastern building block (45.0, 35.0).
  * `multiple_vehicles` / `multiple_decoys` (Scenario C): Target with 3 civilian decoys.
  * `noisy_ocr` (Scenario D): Optical noise requiring multi-frame temporal voting.
  * `failure_recovery` (Scenario E): Injected robot failure at tick 20; dynamic peer reclamation.
  * `target_recruitment` / `cooperative_encounter` (Scenario F): Dual robot cooperative convergence.
  * `far_target` (Scenario G): Target in distant corner (35.0, 25.0); robots spawn at South-West origin.
  * `low_prior_target` (Scenario H): Target in low-probability zone (6.0, 25.0); Bayesian belief update verification.

---

## 10. ROS 2 & Gazebo Integration (`ros2_ws/src/swarm_gazebo/`)

### 10.1 Nodes
* `coordinator_node.py`: ROS 2 Node executing decentralized ACO coordination, task allocation, and dynamic failure handling.
* `slam_bridge.py`: Implements `ISLAMProvider` using `/robot_*/scan` LiDAR and `/robot_*/odom`.
* `robot_controller.py`: Closed-loop waypoint follower with proportional gains (`kp_linear = 0.8`, `kp_angular = 2.0`) publishing `/robot_*/cmd_vel`.
* `perception_bridge.py`: Raycast camera perception and plate matching node.
* `mission_monitor.py`: Subscribes to `/swarm/status` and logs real-time telemetry.

### 10.2 Assets & Launch Files
* `launch/swarm_simulation.launch.py`: Master launch script for all nodes.
* `urdf/swarm_robot.urdf.xacro`: Complete robot URDF model with differential drive, 2D LiDAR, RGB camera, and IMU plugins.
* `worlds/urban_search_world.world`: Gazebo Fortress world with 4 urban building blocks and target vehicle.
