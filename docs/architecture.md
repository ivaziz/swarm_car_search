# SwarmX: System Architecture & Engineering Specification

## 1. Executive Architectural Overview

**SwarmX** is a decentralized, bio-inspired multi-robot coordination architecture engineered to solve outdoor urban search and target identification problems. The system integrates:
* **Decentralized Multi-Robot Task Allocation (MRTA)** powered by Ant Colony Optimization (ACO).
* **Multi-Layer Stigmergic Interaction** via continuous digital pheromone fields ($\tau_e, \tau_s, \tau_a$).
* **Abstracted SLAM & State Estimation** decoupled via the formal `ISLAMProvider` interface.
* **5-Stage Perception & Target Verification Pipeline** (vehicle classification, plate localization, fuzzy OCR, visual appearance verification, and temporal voting).
* **Decentralized Fault-Tolerance & Autonomous Reassignment** with stall and heartbeat failure monitors.
* **Dual Execution Infrastructure**: Pure-Python discrete kinematic simulation and containerized ROS 2 Humble / Gazebo Fortress 3D physics environment.

---

## 2. Team Subsystem Boundaries & Interface Contracts

To guarantee software decoupling and modular engineering:

| Subsystem | Engineers Responsible | Core Responsibilities | Interface Contracts |
| :--- | :--- | :--- | :--- |
| **Swarm Coordination & Task Allocation** | Mohammed & Ibrahim | ACO engine, task scoring, frontier selection, pheromone updates, decentralized consensus, failure recovery, dynamic re-assignment. | Consumes `ISLAMProvider`, produces task assignments and velocity goals. |
| **SLAM & State Estimation** | Saqr & Mahmoud | Sensor fusion (LiDAR, IMU, Wheel Encoders), pose estimation, 2D occupancy mapping, obstacle inflation, frontier extraction. | Implements `ISLAMProvider`, delivers `SLAMState`. |
| **Perception & Vision** | Unified Interface | Vehicle detection, plate cropping, synthetic/live OCR engine, watchlist fuzzy matching, appearance verification. | Implements `IPerceptionProvider`, delivers `MatchEvent`. |

> [!IMPORTANT]
> **Architectural Invariant**: The Swarm Coordination subsystem **never** implements SLAM internally and **never** relies on hardcoded internal maps. All spatial awareness flows exclusively through the standardized `ISLAMProvider` interface contract defined in [`interfaces/slam_interface.py`](file:///Users/aziz/Desktop/swarm_car_search/interfaces/slam_interface.py).

---

## 3. Required Architectural Diagrams (Mermaid Models)

### Diagram 1: Overall System Architecture

```mermaid
graph TB
    subgraph SensingAndState["Sensing & State Estimation Tier (ISLAMProvider)"]
        Sensors["LiDAR / IMU / Encoders / RGB Camera"] --> SLAM["SLAM Engine / State Estimator\n(MockSLAMProvider or GazeboSLAMBridge)"]
        SLAM --> SLAMState["SLAMState\n- Pose2D (x, y, theta)\n- OccupancyGrid2D\n- FrontierCluster List"]
    end

    subgraph SwarmIntelligence["Swarm Intelligence Tier (Decentralized ACO)"]
        SLAMState --> Coordinator["SwarmCoordinator\n(Decentralized Orchestrator)"]
        Coordinator --> PheromoneField["Tri-Layer Pheromone Map\n- Exploration (tau_e)\n- Recruitment (tau_s)\n- Avoidance (tau_a)"]
        Coordinator --> TaskMgr["TaskManager\n- Ingest Frontiers\n- Deduplicate Tasks\n- Cluster Management"]
        TaskMgr --> ACOEngine["ACO Engine & TaskScorer\n- Info Gain (eta)\n- Distance Penalty\n- Congestion Penalty\n- Target Priors"]
        ACOEngine --> Softmax["Boltzmann Softmax Task Allocation\nP(select t_i | r_j)"]
        Softmax --> AssignedTask["Assigned SearchTask"]
    end

    subgraph RobotAgentTier["Robot Agent & Execution Tier"]
        AssignedTask --> Agent["RobotAgent (r_j)\n- State Machine (FSM)\n- Kinematic Waypoint Controller\n- Battery Monitor"]
        Agent --> MotionCtrl["SimulatedRobot / GazeboRobotController\n- Differential Drive Kinematics\n- Reactive Collision Halting"]
        MotionCtrl --> CmdVel["Velocity Commands (cmd_vel)"]
    end

    subgraph PerceptionTier["Perception & Target Verification Tier"]
        Agent --> Detector["SimulatedVehicleDetector\n- FoV & Line-of-Sight Raycasting"]
        Detector --> OCR["SimulatedPlateRecognizer\n- Optical Ambiguity Simulation"]
        OCR --> Matcher["MatchingEngine\n- Levenshtein Fuzzy Distance\n- AppearanceVerifier\n- TemporalTracker (Multi-Frame Voting)"]
        Matcher --> MatchEv["MatchEvent / Target Confirmed"]
        MatchEv -->|Recruitment Stigmergy| PheromoneField
        MatchEv -->|Escalate Priority| Coordinator
    end

    subgraph FaultTolerance["Resilience & Fault Tolerance Tier"]
        Agent --> FailureDet["FailureDetector\n- Heartbeat Timeout\n- Battery Monitor\n- Motion Stall Monitor"]
        FailureDet --> RecoveryMgr["RecoveryManager\n- Orphan Task Release\n- Crash Site Avoidance Deposit\n- Dynamic Reassignment"]
        RecoveryMgr --> Coordinator
    end
```

---

### Diagram 2: Module Dependency Graph

```mermaid
graph LR
    subgraph Interfaces["interfaces/"]
        slam_if["slam_interface.py\n(Pose2D, ISLAMProvider)"]
        perc_if["perception_interface.py\n(MatchEvent, IPerceptionProvider)"]
        serv_if["server_interface.py\n(IServerInterface)"]
    end

    subgraph PerceptionPkg["perception/"]
        app_ver["appearance_verifier.py"]
        temp_tr["temporal_tracker.py"]
        veh_det["vehicle_detector.py"]
        plt_rec["plate_recognizer.py"]
        mat_eng["matching_engine.py"]
    end

    subgraph SwarmPkg["swarm/"]
        aco_cfg["aco/aco_config.py"]
        task_sc["aco/task_scorer.py"]
        aco_eng["aco/aco_engine.py"]
        p_map["pheromone/pheromone_map.py"]
        p_upd["pheromone/pheromone_updater.py"]
        p_bay["pheromone/target_probability_map.py"]
        f_det["failure_handling/failure_detector.py"]
        r_mgr["failure_handling/recovery_manager.py"]
        t_mgr["task_allocation/task_manager.py"]
        s_hist["task_allocation/search_history.py"]
        a_strat["task_allocation/allocation_strategy.py"]
        fsm["states/state_machine.py"]
        r_agent["core/robot_agent.py"]
        coord["core/swarm_coordinator.py"]
    end

    subgraph SimPkg["simulation/"]
        env["environment.py"]
        scen["scenarios.py"]
        s_rob["sim_robot.py"]
        s_comm["sim_comm.py"]
    end

    slam_if --> veh_det
    slam_if --> coord
    slam_if --> r_agent
    perc_if --> mat_eng
    perc_if --> veh_det
    perc_if --> plt_rec
    app_ver --> veh_det
    app_ver --> mat_eng
    temp_tr --> mat_eng
    veh_det --> coord
    plt_rec --> coord
    mat_eng --> coord
    aco_cfg --> task_sc
    task_sc --> aco_eng
    aco_eng --> a_strat
    p_map --> p_upd
    p_upd --> coord
    p_bay --> coord
    s_hist --> coord
    t_mgr --> coord
    f_det --> coord
    r_mgr --> coord
    a_strat --> coord
    fsm --> r_agent
    s_rob --> r_agent
    r_agent --> coord
    env --> coord
```

---

### Diagram 3: Runtime Execution Flow

```mermaid
sequenceDiagram
    autonumber
    participant Main as CLI / Launcher (main.py)
    participant Coord as SwarmCoordinator
    participant SLAM as MockSLAM / GazeboSLAMBridge
    participant Agent as RobotAgent (Fleet)
    participant Scorer as ACO TaskScorer & Engine
    participant Perc as Perception & MatchingEngine
    participant Mon as Mission Monitor / Visualizer

    Main->>Coord: Initialize Fleet, Environment & Configs
    Coord->>Agent: Register Robot Agents (robot_1..4)
    loop Simulation Ticks (dt = 0.2s)
        Coord->>SLAM: Fetch Latest SLAMState (Pose, Grid, Frontiers)
        Coord->>Coord: Ingest New Frontiers into TaskManager
        Coord->>Coord: Evaporate Pheromones (tau_e, tau_s, tau_a)

        alt Robot Needs Task (IDLE / Completed)
            Coord->>Scorer: Evaluate Available Tasks via Multi-Objective Scorer
            Scorer-->>Coord: Attractiveness Scores
            Coord->>Coord: Sample via Boltzmann Softmax Distribution
            Coord->>Agent: Dispatch SearchTask Waypoint
            Coord->>Coord: Deposit Avoidance Pheromone (tau_a)
        end

        Coord->>Agent: Step Robot Agent (Kinematics & Sensors)
        Agent->>Perc: Raycast FOV Vehicle Detector & Camera Scan
        alt Candidate Vehicle Detected
            Perc->>Agent: VehicleDetectionEvent
            Agent->>Perc: Trigger Plate OCR & Multi-Stage Matcher
            alt Watchlist Match Verified
                Perc-->>Coord: MatchEvent (CONFIRMED)
                Coord->>Coord: Trigger Recruitment Pheromone (tau_s)
                Coord->>Coord: Set Mission State = TARGET_FOUND
            end
        end

        Coord->>Coord: Check Heartbeat Timeouts & Stall Detectors
        alt Robot Failure Detected
            Coord->>Coord: Release Orphan Tasks to Pool
            Coord->>Coord: Deposit Wreck Avoidance Pheromone
        end

        Coord->>Mon: Export Live Telemetry / Render Visual Frames
    end
    Coord->>Main: Mission Debrief Report & Final Visual Summary
```

---

### Diagram 4: Data-Flow Diagram

```mermaid
graph TD
    LiDAR["LiDAR Sensor (360 samples)"] -->|LaserScan| SLAM["SLAM Bridge / Provider"]
    Odom["Wheel Encoders / IMU"] -->|Odometry| SLAM
    SLAM -->|SLAMState| TaskIngest["Frontier Ingestion & Deduplication"]

    TaskIngest -->|SearchTask Candidates| TaskPool["TaskManager Active Task Pool"]

    Phero["Pheromone Map Arrays\n(50x50 Float64 Grid)"] -->|Pheromone Values| Scorer["Multi-Objective Task Scorer"]
    ProbMap["Bayesian Target Probability Map"] -->|Target Priors| Scorer
    HistMem["Spatial Search Memory"] -->|Redundancy Penalties| Scorer
    TaskPool -->|Task Geometry & Size| Scorer
    RobotPose["Current Robot Pose2D"] -->|Travel Distance| Scorer

    Scorer -->|Scores Vector| SoftmaxSelector["Boltzmann Softmax Allocator"]
    SoftmaxSelector -->|Selected Task ID| AgentFSM["Robot FSM & Controller"]

    Camera["Camera Sensor (640x480 RGB)"] -->|Image / Detection| DetStage["Vehicle Detector Stage (C_veh)"]
    DetStage -->|Vehicle BBox Crop| PlateStage["Plate Localizer Stage (C_plate)"]
    PlateStage -->|Plate Crop| OCRStage["Fuzzy OCR Stage (C_ocr x S_plate)"]
    DetStage -->|Color & Ratio| AppStage["Appearance Verifier (C_app)"]
    OCRStage -->|Spatial Observations| TempStage["Temporal Tracker (C_temp)"]

    DetStage & PlateStage & OCRStage & AppStage & TempStage -->|Weighted Fusion| MatchEngine["MatchingEngine Composite Confidence"]
    MatchEngine -->|MatchEvent| SwarmTelemetry["Swarm Coordinator Telemetry"]
```

---

### Diagram 5: Robot-to-Robot Coordination & Stigmergy

```mermaid
graph TB
    subgraph StigmergyField["Shared Environmental Memory (Stigmergy)"]
        TauE["Exploration Field (tau_e)\nrho = 0.01\nPrevents re-exploring mapped areas"]
        TauS["Recruitment Field (tau_s)\nrho = 0.05\nAttracts swarm to candidate sightings"]
        TauA["Reservation Field (tau_a)\nrho = 0.30\nPrevents multi-robot task collisions"]
    end

    subgraph Robot1["Robot Agent 1"]
        R1_Pose["Pose: (x1, y1)"] --> R1_Deposit["Deposit tau_a along trajectory"]
        R1_Deposit --> TauA
        TauE --> R1_Sensors["Sense Local Pheromones"]
        TauS --> R1_Sensors
        TauA --> R1_Sensors
    end

    subgraph Robot2["Robot Agent 2"]
        R2_Pose["Pose: (x2, y2)"] --> R2_Deposit["Deposit tau_a along trajectory"]
        R2_Deposit --> TauA
        TauE --> R2_Sensors["Sense Local Pheromones"]
        TauS --> R2_Sensors
        TauA --> R2_Sensors
    end

    subgraph DirectComm["Decentralized Peer-to-Peer Heartbeats"]
        R1_HB["Heartbeat:\n(ID, Pose, TaskID, Battery)"] <-->|Simulated Network Channel| R2_HB["Heartbeat:\n(ID, Pose, TaskID, Battery)"]
    end
```

---

### Diagram 6: Target Detection and Identification Pipeline

```mermaid
graph TD
    InputFrame["Input Camera Frame / World Sighting"] --> Step1["Step 1: Vehicle Detection (C_veh)\n- Range check (< 14.0m)\n- Bearing in FoV (+/- 45 deg)\n- Line-of-sight raycast against walls"]

    Step1 -->|Occluded or Out of Range| Discard["Reject Observation"]
    Step1 -->|Vehicle In FoV| Step2["Step 2: License Plate Localization (C_plate)\n- Sub-bounding box regression\n- Resolution check based on range"]

    Step2 --> Step3["Step 3: Character OCR & String Distance (C_ocr x S_plate)\n- Optical character recognition\n- Confusion map (8<->B, 0<->O, 2<->Z)\n- Normalized Levenshtein similarity"]

    Step2 --> Step4["Step 4: Appearance Verification (C_app)\n- Vehicle class check (Sedan vs SUV/Truck)\n- Normalized Euclidean distance in RGB color space"]

    Step3 & Step4 --> Step5["Step 5: Multi-Frame Temporal Tracker (C_temp)\n- Spatial clustering of candidate readings (< 3.0m)\n- Minimum observations threshold (k >= 3 frames)\n- Majority voting on plate string"]

    Step1 & Step2 & Step3 & Step4 & Step5 --> FusedConfidence["Composite Target Confidence Calculation:\nC = 0.15 C_veh + 0.15 C_plate + 0.30 (C_ocr x S_plate) + 0.20 C_app + 0.20 C_temp"]

    FusedConfidence --> ThresholdCheck{"C >= 0.75 (default) AND k >= 3?"}
    ThresholdCheck -->|Yes| Confirmed["CONFIRMED MATCH\nTrigger Swarm Alarm & End Mission"]
    ThresholdCheck -->|No, but C >= 0.45| Candidate["CANDIDATE DETECTION\nDeposit tau_s Recruitment Pheromone"]
    ThresholdCheck -->|No, C < 0.45| NonTarget["NON-TARGET VEHICLE\nDecoy Vehicle Discarded"]
```

---

### Diagram 7: Failure Detection and Recovery Flow

```mermaid
stateDiagram-v2
    [*] --> HealthyOperation: Robot Active

    state HealthyOperation {
        Monitoring: Periodic Health Check
        HeartbeatSent: Broadcast Heartbeat
        MotionTracking: Monitor Position Delta
    }

    HealthyOperation --> HeartbeatTimeout: Silence > 10.0s (Dynamic Timeout)
    HealthyOperation --> BatteryDepleted: Battery < 0.10
    HealthyOperation --> MotionStalled: Distance Delta < 0.1m over 8.0s in NAVIGATING

    state FaultHandling {
        MarkFailed: Coordinator Flags Robot as FAILED
        ReleaseTask: Orphaned Task Released to PENDING Pool
        AvoidanceDeposit: Deposit Avoidance Pheromone (tau_a) at Crash Site
    }

    HeartbeatTimeout --> FaultHandling
    BatteryDepleted --> FaultHandling
    MotionStalled --> FaultHandling

    FaultHandling --> PeerReassignment: Peer Evaluates Released Task
    PeerReassignment --> Reclaimed: Healthy Peer Assigned & Dispatched
    Reclaimed --> [*]
```

---

### Diagram 8: Simulation Startup, Execution, and Teardown Flow

```mermaid
sequenceDiagram
    autonumber
    participant Host as Host Shell / Launcher
    participant Docker as Docker Container (ARM64 Ubuntu 22.04)
    participant Gazebo as Gazebo Fortress 3D
    participant ROSBridge as ros_gz_bridge
    participant Nodes as SwarmX ROS 2 Nodes
    participant Artifacts as Output Directory

    Host->>Docker: bash run_gazebo_gui.sh (or run_simulation.sh)
    Docker->>Docker: Xvfb :1, fluxbox, x11vnc, noVNC (6080)
    Docker->>Gazebo: ign gazebo -v 3 -r urban_search_world.world
    Docker->>ROSBridge: parameter_bridge (/clock, /cmd_vel, /odom, /scan, /camera)
    Docker->>Nodes: Launch coordinator, slam_bridge, perception_bridge, controllers
    loop Mission Search Execution
        Nodes->>Gazebo: Send Twist commands (/robot_*/cmd_vel)
        Gazebo->>Nodes: Publish Odometry, LaserScan, Camera frames
        Nodes->>Nodes: Execute ACO task allocation & perception
        Nodes->>Artifacts: Save periodic SVG/PNG frame snapshots
    end
    alt Target Confirmed OR Max Steps Reached
        Nodes->>Artifacts: Render final_mission_summary.png & SVG
        Nodes->>Artifacts: Write ai_benchmark_report.json / debrief logs
    end
    Host->>Docker: SIGINT / Shutdown
    Docker->>Nodes: Terminate ROS 2 nodes gracefully
    Docker->>ROSBridge: Terminate parameter bridge
    Docker->>Gazebo: Terminate Gazebo process
    Docker->>Host: Exit Container cleanly
```

---

## 4. Major System Components Detail

### 4.1 Swarm Coordinator (`swarm/core/swarm_coordinator.py`)
* **What it does**: Central software hub responsible for synchronizing robots, advancing the tri-layer pheromone map, mediating inter-robot heartbeats, orchestrating task allocation, and recording mission-level telemetry.
* **Why it exists**: Serves as the high-level orchestration interface for the swarm while preserving local agent autonomy.
* **Inputs**: SLAM states, robot heartbeats, perception events.
* **Outputs**: Search task dispatches, pheromone updates, aggregated mission status.
* **Dependencies**: `TaskManager`, `ACOAllocationStrategy`, `PheromoneMap`, `PheromoneUpdater`, `FailureDetector`, `RecoveryManager`.
* **Verification Status**: **Implemented & Verified** across all test suites.

### 4.2 ACO Engine & Task Scorer (`swarm/aco/aco_engine.py`, `swarm/aco/task_scorer.py`)
* **What it does**: Evaluates candidate frontier clusters and target investigation tasks using a multi-objective scoring function and samples assignments using Boltzmann Softmax.
* **Why it exists**: Balances spatial exploration against target exploitation while preventing swarm collapse onto a single sector.
* **Mathematical Formulation**: Multi-factor synthesis detailed in [ALGORITHMS_AND_PARAMETERS.md](file:///Users/aziz/Desktop/swarm_car_search/docs/ALGORITHMS_AND_PARAMETERS.md).
* **Verification Status**: **Implemented & Verified** ([`tests/test_aco_engine.py`](file:///Users/aziz/Desktop/swarm_car_search/tests/test_aco_engine.py)).

### 4.3 Perception & Matching Engine (`perception/matching_engine.py`, `vehicle_detector.py`, `plate_recognizer.py`)
* **What it does**: Detects vehicles within camera FoV, extracts synthetic license plates, applies distance-based optical noise, verifies color/aspect ratio, tracks candidate observations over time, and calculates composite confidence against the target watchlist.
* **Verification Status**: **Implemented & Verified** ([`tests/test_perception_integration.py`](file:///Users/aziz/Desktop/swarm_car_search/tests/test_perception_integration.py), [`scripts/evaluate_ai_benchmarks.py`](file:///Users/aziz/Desktop/swarm_car_search/scripts/evaluate_ai_benchmarks.py)).

### 4.4 SLAM Bridge & Interface (`interfaces/slam_interface.py`, `ros2_ws/src/swarm_gazebo/swarm_gazebo/slam_bridge.py`)
* **What it does**: Subscribes to `/robot_*/scan` and `/robot_*/odom`, builds local 2D occupancy grids via raycasting, extracts unknown/free cell boundaries as frontier clusters, and delivers `SLAMState`.
* **Verification Status**: **Implemented & Verified in simulation** ([`tests/test_mock_slam.py`](file:///Users/aziz/Desktop/swarm_car_search/tests/test_mock_slam.py), `slam_providers/mock_slam_provider.py`).
