# SwarmX: Reproducible Setup, Installation & Execution Guide

This guide provides tested, step-by-step procedures for installing, building, and running SwarmX across both pure-Python execution modes and containerized ROS 2 / Gazebo Fortress 3D environments.

---

## 1. System Requirements & Architecture Matrix

| Platform / Environment | Supported Architectures | Python Versions | Display Modes Supported | Prerequisites |
| :--- | :---: | :---: | :--- | :--- |
| **macOS (Host)** | Apple Silicon (M1/M2/M3 ARM64), Intel x86_64 | 3.9, 3.10, 3.11, 3.12, 3.13 | Live ASCII HUD, Vector SVG, Matplotlib PNG, Web browser (noVNC), Native VNC | Python 3.9+, pip, git (Docker Desktop optional for Gazebo) |
| **Linux (Host)** | x86_64, aarch64 (Ubuntu 20.04, 22.04, 24.04) | 3.9, 3.10, 3.11, 3.12 | Live ASCII HUD, SVG, PNG, Native X11 / Wayland | Python 3.9+, pip, git |
| **Docker Simulation** | linux/arm64, linux/amd64 | 3.10 (Ubuntu 22.04 LTS) | Web 3D GUI (port 6080), VNC (port 5900), Headless CLI | Docker Desktop 20.10+ / Docker Engine 24.0+ |

---

## 2. Mode 1: Pure-Python Execution (Zero ROS / Zero Docker Required)

The core SwarmX intelligence engine—including the multi-objective ACO task scorer, tri-layer stigmergic pheromone fields, 5-stage vision and fuzzy OCR models, 2D kinematic physics, ASCII HUD, and vector SVG map generation—runs with **zero external system dependencies** on standard Python.

### 2.1 Host Environment Setup
```bash
# 1. Clone repository and navigate to workspace root
git clone <repository-url>
cd swarm_car_search

# 2. (Optional) Create and activate a clean virtual environment
python3 -m venv venv
source venv/bin/activate

# 3. Install optional visualization dependencies (for Matplotlib PNG & Animated GIF export)
pip install -r requirements.txt
# Or install directly:
pip install matplotlib numpy pillow pyyaml pytest
```

### 2.2 Verify System Installation via CLI
```bash
# Display system banner, developer specifications, and architecture overview
python3 scripts/main.py info
```
Expected output:
```text
================================================================================
   AUTONOMOUS MULTI-ROBOT SWARM: URBAN CAR SEARCH & IDENTIFICATION SYSTEM   
================================================================================
  Graduation Project: AI Swarm Robotics Subsystem
  Developers        : Mohammed & Ibrahim (Swarm Coordination & ACO Task Allocation)
  External Interface: Saqr & Mahmoud (SLAM, Odometry, & Frontier Mapping)
  Version           : 1.0.0 (Production Release)
  Architecture      : Decentralized Ant Colony Optimization (ACO) & Stigmergy
================================================================================
```

### 2.3 Run the Automated Test Suite (112 Tests)
```bash
# Execute full test suite using pytest
pytest -v

# Or run via unified CLI launcher
python3 scripts/main.py test
```
Expected output:
```text
============================= 112 passed in 5.63s =============================
```

### 2.4 Run an Autonomous Swarm Search Simulation
Launch a 4-robot swarm search in a $50\,\text{m} \times 50\,\text{m}$ urban precinct searching for stolen vehicle `DXB-88392`:
```bash
python3 scripts/run_simulation.py \
    --scenario easy_target \
    --robots 4 \
    --steps 120 \
    --output-dir output/mission_01
```

#### What happens during execution:
1. Four differential drive robots spawn at the 4 corner intersections of the city grid.
2. The SLAM provider extracts frontier boundaries between mapped free space and unexplored territory.
3. The ACO engine scores candidate clusters, depositing avoidance pheromone ($\tau_a$) along travel paths to prevent multi-robot congestion.
4. When a robot acquires line-of-sight on target `DXB-88392`, the camera detector, plate recognizer, appearance verifier, and temporal tracker fire in sequence.
5. Upon confirmation ($C_{\text{target}} \ge 0.80$), recruitment pheromone ($\tau_s$) is deposited, alerting the fleet and concluding the search.
6. Vector SVG maps (`frame_*.svg`, `final_mission_summary.svg`), Matplotlib panels (`.png`), and an animated GIF (`swarm_mission_animation.gif`) are saved to `output/mission_01/`.

### 2.5 Run AI Benchmark Suite (Scenarios A through H)
To evaluate perception precision, recall, OCR character accuracy, and swarm exploration across all 8 deterministic benchmark scenarios:
```bash
python3 scripts/evaluate_ai_benchmarks.py
```
Artifact generated: `output/ai_benchmark_report.json`.

---

## 3. Mode 2: Containerized Gazebo Fortress 3D & ROS 2 Humble Simulation

The repository includes a complete ROS 2 Humble and Gazebo Fortress (Ignition Gazebo) simulation environment containerized via Docker for cross-platform reproducibility (including native Apple Silicon M1/M2/M3 ARM64 execution).

### 3.1 Docker Prerequisites
1. Ensure Docker Desktop is installed and running:
   ```bash
   docker info
   ```
2. Verify Docker socket connection. If Docker is offline, launch Docker Desktop:
   ```bash
   open -a Docker
   ```

### 3.2 Master Launchers Overview

| Script | Display Mode | Ports Exposed | Primary Use Case |
| :--- | :--- | :--- | :--- |
| [`run_gazebo_gui.sh`](file:///Users/aziz/Desktop/swarm_car_search/run_gazebo_gui.sh) | Interactive 3D Web & VNC | `6080` (HTTP Web noVNC), `5900` (macOS Screen Sharing) | Visualizing 3D robot motion, LiDAR pointclouds, building models, and urban traffic. |
| [`run_simulation.sh`](file:///Users/aziz/Desktop/swarm_car_search/run_simulation.sh) | Headless Container CLI | None | Fast automated execution in continuous integration (CI) or headless servers. |

### 3.3 Launching the 3D Interactive Gazebo GUI
To start the full Gazebo Fortress 3D world with 4 robots, virtual X11 server, and browser view:
```bash
./run_gazebo_gui.sh
```

#### Automatic Connection Sequence:
1. The script verifies the Docker daemon and builds `swarm_gazebo:humble` if not present.
2. It compiles the ROS 2 workspace inside the container using `colcon build --symlink-install`.
3. It launches a virtual display server (`Xvfb :1` at $1600 \times 1000 \times 24$), a window manager (`fluxbox`), `x11vnc`, and `websockify` (noVNC).
4. It starts `ign gazebo -v 3 -r urban_search_world.world`.
5. It initializes `ros_gz_bridge parameter_bridge` mapping Gazebo clock, camera, LiDAR, odometry, and cmd_vel topics.
6. It automatically opens your default web browser to:
   ```text
   http://localhost:6080/vnc.html?autoconnect=true&resize=remote
   ```
7. For macOS users, it simultaneously attempts native Screen Sharing via:
   ```text
   vnc://localhost:5900
   ```

### 3.4 Launching Headless Gazebo Simulation
To run Gazebo in headless mode without spawning the web GUI:
```bash
./run_simulation.sh
```

---

## 4. Manual Docker Build & Run (Step-by-Step)

If you prefer building and executing individual container commands manually:

### Step 1: Build the Container Image
```bash
docker build -t swarm_gazebo:humble -f docker/Dockerfile .
```

### Step 2: Build the ROS 2 Colcon Workspace
```bash
docker run --rm \
    --platform linux/arm64 \
    -v "$(pwd)":/workspace \
    -w /workspace/ros2_ws \
    swarm_gazebo:humble \
    /bin/bash -c "source /opt/ros/humble/setup.bash && colcon build --symlink-install"
```

### Step 3: Run the Python ROS 2 Orchestrator
```bash
docker run --rm -it \
    --platform linux/arm64 \
    -v "$(pwd)":/workspace \
    -w /workspace \
    swarm_gazebo:humble \
    /bin/bash -c "source /opt/ros/humble/setup.bash && \
                 source /workspace/ros2_ws/install/setup.bash && \
                 export PYTHONPATH=/workspace:/workspace/ros2_ws/src/swarm_gazebo:\$PYTHONPATH && \
                 python3 scripts/run_gazebo_simulation.py --steps 100"
```

---

## 5. Inspecting Live ROS 2 Topics and Telemetry

While the simulation container is running, open a secondary shell to inspect live ROS 2 communications:

```bash
# Attach to the running simulation container
docker exec -it swarm_gazebo_gui_session /bin/bash

# Source environment
source /opt/ros/humble/setup.bash
source /workspace/ros2_ws/install/setup.bash

# List all active ROS 2 topics
ros2 topic list
```

Expected active topics:
```text
/clock
/robot_1/cmd_vel
/robot_1/odom
/robot_1/scan
/robot_1/camera/image_raw
/robot_2/cmd_vel
/robot_2/odom
/robot_2/scan
/robot_2/camera/image_raw
/robot_3/cmd_vel
/robot_3/odom
/robot_3/scan
/robot_3/camera/image_raw
/robot_4/cmd_vel
/robot_4/odom
/robot_4/scan
/robot_4/camera/image_raw
/swarm/status
/swarm/perception_events
```

### Echoing Velocity Commands & Swarm Telemetry
```bash
# Inspect velocity commands published to Robot 1
ros2 topic echo /robot_1/cmd_vel

# Inspect live swarm coordination status
ros2 topic echo /swarm/status
```

---

## 6. Simulation Teardown and Safe Cleanup

* **Stopping GUI Simulation**: Press `Ctrl+C` in the terminal executing `./run_gazebo_gui.sh`. The trap handler automatically terminates background bridge, Xvfb, and Gazebo processes.
* **Stopping Docker Container Manually**:
  ```bash
  docker stop swarm_gazebo_gui_session 2>/dev/null || true
  ```
* **Preserving Mission Artifacts**: Simulation images, debrief reports, and JSON benchmarks are saved directly to the host filesystem under `output/` via volume mounts (`-v "$(pwd)":/workspace`). No project files or outputs are deleted upon container exit.
