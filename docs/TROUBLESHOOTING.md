# SwarmX: Troubleshooting & Diagnostics Knowledge Base

This guide documents known operational issues, failure symptoms, diagnostic procedures, confirmed root causes, and verified remediations discovered during development and testing of SwarmX.

---

## 1. Quick Reference: Symptom & Diagnosis Matrix

| # | Symptom | Primary Diagnostic Command | Confirmed Cause | Safe Resolution |
| :-: | :--- | :--- | :--- | :--- |
| **1** | Docker socket connection failure | `docker info` | Docker Desktop daemon is not running on macOS host. | Run `open -a Docker` and wait for daemon initialization. |
| **2** | No display output / Gazebo GUI fails | `curl -I http://localhost:6080` | Virtual X11 server or websockify failed to bind. | Ensure ports 6080 and 5900 are free; verify Xvfb startup in container. |
| **3** | `ModuleNotFoundError: No module named 'matplotlib'` | `python3 -c "import matplotlib"` | Executing with minimal Python environment instead of environment with extras. | Run with `/opt/anaconda3/bin/python3` or run `pip install matplotlib pillow`. |
| **4** | Matplotlib font cache warning | Check stderr for `MPLCONFIGDIR` | Writing font cache to restricted root directory. | Set `export MPLCONFIGDIR=/tmp/matplotlib_swarm_cache`. |
| **5** | ROS 2 topics not communicating | `ros2 topic list` | `ros_gz_bridge parameter_bridge` not started or mismatched message signatures. | Verify parameter bridge launch in `run_gazebo_gui.sh:128-146`. |
| **6** | Robot stuck in `NAVIGATING` state | Check HUD for `[X] Failed Robot` | Physical wall deadlock triggering stall detector ($< 0.1\,\text{m}$ in $8.0\,\text{s}$). | Normal fault recovery: coordinator marks robot failed and releases task to peer. |
| **7** | Target vehicle plate misread | Check `ai_benchmark_report.json` | Optical noise at range $> 7.0\,\text{m}$ causing character confusion ($8 \leftrightarrow B$). | Handled automatically by temporal tracker: requires $k \ge 3$ close-range frames. |

---

## 2. In-Depth Problem Diagnoses & Resolutions

### Problem 1: Docker Daemon Socket Connection Failure
* **Symptom**:
  Executing `./run_gazebo_gui.sh` or `./run_simulation.sh` exits with:
  ```text
  failed to connect to the docker API at unix:///Users/aziz/.docker/run/docker.sock; check if the path is correct and if the daemon is running: dial unix /Users/aziz/.docker/run/docker.sock: connect: no such file or directory
  ```
* **Evidence**: Observed directly on macOS when Docker Desktop has not been launched prior to script invocation.
* **Confirmed Cause**: The Docker background daemon is dormant; the Unix domain socket `/Users/aziz/.docker/run/docker.sock` does not exist until Docker Desktop initializes.
* **Diagnostic Command**:
  ```bash
  docker info
  ```
* **Resolution Steps**:
  1. Start Docker Desktop from the terminal:
     ```bash
     open -a Docker
     ```
  2. Wait 15–30 seconds for the Docker icon in the macOS menu bar to stabilize.
  3. Verify connection:
     ```bash
     docker ps
     ```
  4. Re-run `./run_gazebo_gui.sh` or `./run_simulation.sh`.

---

### Problem 2: Gazebo 3D GUI Display Forwarding in Container
* **Symptom**: The container launches, but the web browser cannot connect to `http://localhost:6080/vnc.html` or native Screen Sharing fails at `vnc://localhost:5900`.
* **Confirmed Cause**:
  * Port collision: Host process already occupying port 5900 (e.g., macOS native VNC / Screen Sharing server) or port 6080.
  * Race condition: Browser launches before `Xvfb` and `websockify` have initialized.
* **Diagnostic Commands**:
  ```bash
  # Check if ports 5900 or 6080 are held by another process on host
  lsof -i :5900
  lsof -i :6080
  ```
* **Resolution Steps**:
  1. If macOS Screen Sharing is active on host port 5900, connect via browser (`http://localhost:6080/vnc.html`) which uses isolated port 6080.
  2. The launcher script [`run_gazebo_gui.sh`](file:///Users/aziz/Desktop/swarm_car_search/run_gazebo_gui.sh#L75-L82) includes an automatic 3-second sleep before opening the client, allowing `websockify` to bind cleanly.

---

### Problem 3: Headless Matplotlib Cache & Display Warnings
* **Symptom**: Terminal outputs warning: `Matplotlib is building the font cache; this may take a moment` or errors trying to open an interactive display window on headless environments.
* **Confirmed Cause**: By default, Matplotlib attempts to use interactive GUI backends (TkAgg, MacOSX, Qt) which fail when running headless or in Docker without X11.
* **Resolution**:
  [`visualization/swarm_visualizer.py`](file:///Users/aziz/Desktop/swarm_car_search/visualization/swarm_visualizer.py#L23-L36) automatically sets the headless backend and isolated cache directory:
  ```python
  if "MPLCONFIGDIR" not in os.environ:
      os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib_swarm_cache"
  import matplotlib
  matplotlib.use("Agg")
  ```

---

### Problem 4: ROS 2 <-> Gazebo Parameter Bridge Topic Mismatch
* **Symptom**: Robots in Gazebo do not move despite the coordinator publishing `/robot_*/cmd_vel`, or SLAM bridge occupancy grids remain blank.
* **Confirmed Cause**: `ros_gz_bridge` requires exact topic naming and direction delimiters:
  * `[` : Gazebo message converted to ROS 2.
  * `]` : ROS 2 message converted to Gazebo.
  * `@` : Bidirectional bridge.
* **Diagnostic Command**:
  ```bash
  # Inside container:
  ros2 topic info /robot_1/cmd_vel
  ign topic -l
  ```
* **Resolution**:
  Ensure the bridge command matches [`run_gazebo_gui.sh`](file:///Users/aziz/Desktop/swarm_car_search/run_gazebo_gui.sh#L128-L146):
  ```bash
  ros2 run ros_gz_bridge parameter_bridge \
      /clock@rosgraph_msgs/msg/Clock[ignition.msgs.Clock \
      /robot_1/cmd_vel@geometry_msgs/msg/Twist]ignition.msgs.Twist \
      /robot_1/odom@nav_msgs/msg/Odometry[ignition.msgs.Odometry \
      /robot_1/scan@sensor_msgs/msg/LaserScan[ignition.msgs.LaserScan
  ```

---

### Problem 5: Differential Robot Stalling in Tight Alleyways
* **Symptom**: Robot stops moving and is marked with an `[X]` (Failed) in the ASCII HUD.
* **Confirmed Cause**:
  * In narrow corridors, obstacles on both flanks may reduce the navigable clearance below `collision_distance = 0.5m`.
  * If the robot remains stationary ($< 0.1\,\text{m}$ progress) for more than `stall_threshold_sec = 8.0s`, [`FailureDetector`](file:///Users/aziz/Desktop/swarm_car_search/swarm/failure_handling/failure_detector.py#L125-L150) flags a motion stall.
* **System Self-Healing Verification**:
  This is an intended fault-recovery behavior, not a defect:
  1. The robot transitions to `RobotState.FAILED`.
  2. Its assigned task is stripped and re-queued as `PENDING`.
  3. A high avoidance pheromone ($\tau_a$) is deposited at the stalled coordinates.
  4. Operational peers bypass the stalled robot and continue exploring other sectors.

---

### Problem 6: Optical OCR Noise and Character Substitution
* **Symptom**: Plate reading returns `DXB-BB39Z` instead of `DXB-88392`.
* **Confirmed Cause**: At distances $> 7.0\,\text{m}$, [`SimulatedPlateRecognizer`](file:///Users/aziz/Desktop/swarm_car_search/perception/plate_recognizer.py#L55-L80) introduces optical blur, substituting ambiguous characters via `OCR_CONFUSION_MAP`.
* **Remediation & Design Safeguard**:
  The system prevents premature false-alarm dispatching by using two architectural layers:
  1. Levenshtein fuzzy string similarity computes edit distance, allowing partial match candidates ($S_{\text{plate}} \ge 0.50$) to be recognized as `CANDIDATE`.
  2. [`TemporalTracker`](file:///Users/aziz/Desktop/swarm_car_search/perception/temporal_tracker.py) accumulates sightings over time, performing majority character voting across frames as the robot navigates closer ($< 5.0\,\text{m}$), restoring 100% character accuracy.
