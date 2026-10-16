#!/usr/bin/env bash
# ==============================================================================
# SwarmX: Autonomous Multi-Robot Outdoor Car Search System
# Gazebo 3D Interactive GUI Master Launcher for macOS Apple Silicon
# ==============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=============================================================================="
echo "    SWARMX / AUTONOMOUS SWARM OUTDOOR CAR SEARCH — GAZEBO 3D GUI LAUNCHER     "
echo "=============================================================================="
echo "Host Machine       : Apple Silicon M1 (ARM64 macOS)"
echo "Simulation Engine  : Gazebo Sim (Fortress) & ROS 2 Humble"
echo "GUI Display Modes  : Web 3D View (http://localhost:6080/vnc.html)"
echo "                     macOS Native Screen Sharing (vnc://localhost:5900)"
echo "Project Path       : $SCRIPT_DIR"
echo "=============================================================================="

# Determine TTY allocation flag
if [ -t 0 ]; then
    DOCKER_INTERACTIVE="-it"
else
    DOCKER_INTERACTIVE="-i"
fi

# 1. Verify Docker Daemon Status
if ! docker info >/dev/null 2>&1; then
    echo "[!] Docker daemon is not responding. Attempting to start Docker Desktop..."
    open -a Docker || true
    echo "[*] Waiting for Docker daemon to initialize..."
    for i in {1..30}; do
        if docker info >/dev/null 2>&1; then
            echo "[+] Docker daemon is online."
            break
        fi
        sleep 2
    done
fi

if ! docker info >/dev/null 2>&1; then
    echo "[-] Error: Could not connect to Docker daemon. Please ensure Docker is running."
    exit 1
fi

# 2. Check Image
IMAGE_NAME="swarm_gazebo:humble"
if ! docker image inspect "$IMAGE_NAME" >/dev/null 2>&1; then
    echo "[*] Building Gazebo GUI container image ($IMAGE_NAME)..."
    docker build -t "$IMAGE_NAME" -f docker/Dockerfile .
    echo "[+] Container image built successfully."
else
    echo "[+] Container image '$IMAGE_NAME' is ready."
fi

# 3. Build ROS 2 Colcon Workspace
echo "[*] Building ROS 2 Workspace (colcon build)..."
docker run --rm \
    --platform linux/arm64 \
    -v "$SCRIPT_DIR":/workspace \
    -w /workspace/ros2_ws \
    "$IMAGE_NAME" \
    /bin/bash -c "source /opt/ros/humble/setup.bash && colcon build --symlink-install"

# 4. Check XQuartz / X11 setup on Host (if available)
if command -v xhost >/dev/null 2>&1; then
    xhost + 127.0.0.1 >/dev/null 2>&1 || true
    xhost + localhost >/dev/null 2>&1 || true
fi

# 5. Open Browser and Native Screen Sharing after display server is online
echo "[*] Launching 3D Display Client..."
(
    sleep 3
    # Open Web 3D view in default browser
    open "http://localhost:6080/vnc.html?autoconnect=true&resize=remote" >/dev/null 2>&1 || true
    # Try native macOS Screen Sharing app
    open "vnc://localhost:5900" >/dev/null 2>&1 || true
) &

# 6. Run Gazebo GUI & SwarmX Simulation inside Container
mkdir -p "$SCRIPT_DIR/output/gazebo_simulation_run"

# Sanitize arguments: exclude empty strings
CLEAN_ARGS=()
for a in "$@"; do
    if [ -n "$a" ]; then
        CLEAN_ARGS+=("$a")
    fi
done

echo "[*] Starting Gazebo 3D GUI, ROS 2 Bridges, and Swarm Intelligence Nodes..."
docker run --rm $DOCKER_INTERACTIVE \
    --platform linux/arm64 \
    --name swarm_gazebo_gui_session \
    -p 5900:5900 \
    -p 6080:6080 \
    -v "$SCRIPT_DIR":/workspace \
    -w /workspace \
    "$IMAGE_NAME" \
    /bin/bash -c '
        source /opt/ros/humble/setup.bash
        if [ -f /workspace/ros2_ws/install/setup.bash ]; then
            source /workspace/ros2_ws/install/setup.bash
        fi
        export PYTHONPATH=/workspace:/workspace/ros2_ws/src/swarm_gazebo:$PYTHONPATH
        export IGN_GAZEBO_RESOURCE_PATH=/workspace/ros2_ws/src/swarm_gazebo/worlds:$IGN_GAZEBO_RESOURCE_PATH

        # 1. Start Virtual X11 Display Server & Window Manager
        Xvfb :1 -screen 0 1600x1000x24 &
        export DISPLAY=:1
        sleep 1
        fluxbox &
        x11vnc -display :1 -nopw -forever -shared -rfbport 5900 -quiet &
        websockify --web=/usr/share/novnc 6080 localhost:5900 >/dev/null 2>&1 &
        sleep 1

        # 2. Start Gazebo Sim (Fortress) GUI with 50m x 50m Urban Search World & 4 Robots
        echo "[+] Launching Gazebo Fortress 3D GUI Window..."
        ign gazebo -v 3 -r /workspace/ros2_ws/src/swarm_gazebo/worlds/urban_search_world.world &
        GZ_PID=$!
        sleep 4

        # 3. Start ROS-Gazebo Parameter Bridge for all 4 Robots
        echo "[+] Initializing ROS 2 <-> Gazebo Sensor & Command Bridge..."
        ros2 run ros_gz_bridge parameter_bridge \
            /clock@rosgraph_msgs/msg/Clock[ignition.msgs.Clock \
            /robot_1/cmd_vel@geometry_msgs/msg/Twist]ignition.msgs.Twist \
            /robot_1/odom@nav_msgs/msg/Odometry[ignition.msgs.Odometry \
            /robot_1/scan@sensor_msgs/msg/LaserScan[ignition.msgs.LaserScan \
            /robot_1/camera/image_raw@sensor_msgs/msg/Image[ignition.msgs.Image \
            /robot_2/cmd_vel@geometry_msgs/msg/Twist]ignition.msgs.Twist \
            /robot_2/odom@nav_msgs/msg/Odometry[ignition.msgs.Odometry \
            /robot_2/scan@sensor_msgs/msg/LaserScan[ignition.msgs.LaserScan \
            /robot_2/camera/image_raw@sensor_msgs/msg/Image[ignition.msgs.Image \
            /robot_3/cmd_vel@geometry_msgs/msg/Twist]ignition.msgs.Twist \
            /robot_3/odom@nav_msgs/msg/Odometry[ignition.msgs.Odometry \
            /robot_3/scan@sensor_msgs/msg/LaserScan[ignition.msgs.LaserScan \
            /robot_3/camera/image_raw@sensor_msgs/msg/Image[ignition.msgs.Image \
            /robot_4/cmd_vel@geometry_msgs/msg/Twist]ignition.msgs.Twist \
            /robot_4/odom@nav_msgs/msg/Odometry[ignition.msgs.Odometry \
            /robot_4/scan@sensor_msgs/msg/LaserScan[ignition.msgs.LaserScan \
            /robot_4/camera/image_raw@sensor_msgs/msg/Image[ignition.msgs.Image >/dev/null 2>&1 &
        BRIDGE_PID=$!
        sleep 2

        # 4. Launch SwarmX Multi-Robot Intelligence, SLAM, Perception, and Dynamic Failure Recovery
        echo "[+] Launching SwarmX Decentralized ACO Coordinator & Robot Controllers..."
        if [ $# -gt 0 ]; then
            python3 scripts/run_gazebo_gui_simulation.py "$@"
        else
            python3 scripts/run_gazebo_gui_simulation.py
        fi

        # Cleanup background processes on exit
        kill $BRIDGE_PID $GZ_PID 2>/dev/null || true
    ' -- "${CLEAN_ARGS[@]}"

echo "=============================================================================="
echo "[+] Gazebo 3D GUI Simulation session completed."
echo "    Artifacts and debrief reports saved to: $SCRIPT_DIR/output/gazebo_simulation_run/"
echo "=============================================================================="
