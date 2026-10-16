#!/usr/bin/env bash
# ==============================================================================
# SwarmX: Autonomous Multi-Robot Outdoor Car Search System
# Gazebo & ROS 2 Simulation Unified Master Launcher
# ==============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Check if user requested GUI mode
if [[ "$*" == *"--gui"* ]]; then
    args=()
    for arg in "$@"; do
        if [ "$arg" != "--gui" ]; then
            args+=("$arg")
        fi
    done
    if [ ${#args[@]} -gt 0 ]; then
        exec "$SCRIPT_DIR/run_gazebo_gui.sh" "${args[@]}"
    else
        exec "$SCRIPT_DIR/run_gazebo_gui.sh"
    fi
fi

echo "=============================================================================="
echo "    SWARMX / AUTONOMOUS SWARM OUTDOOR CAR SEARCH — GAZEBO SIMULATION LAUNCHER    "
echo "=============================================================================="
echo "Host Machine       : Apple Silicon M1 (ARM64 macOS)"
echo "Simulation Engine  : Gazebo Sim (Fortress) & ROS 2 Humble"
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

# 2. Build Docker Simulation Image if not present
IMAGE_NAME="swarm_gazebo:humble"
if ! docker image inspect "$IMAGE_NAME" >/dev/null 2>&1; then
    echo "[*] Building Gazebo & ROS 2 ARM64 container image ($IMAGE_NAME)..."
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

echo "[+] ROS 2 workspace built successfully."

# 4. Run SwarmX Gazebo Simulation in Headless Mode
echo "[*] Launching SwarmX Gazebo Simulation with 4 Autonomous Ground Vehicles..."
mkdir -p "$SCRIPT_DIR/output/gazebo_simulation_run"

docker run --rm $DOCKER_INTERACTIVE \
    --platform linux/arm64 \
    -v "$SCRIPT_DIR":/workspace \
    -w /workspace \
    "$IMAGE_NAME" \
    /bin/bash -c 'source /opt/ros/humble/setup.bash && \
                 source /workspace/ros2_ws/install/setup.bash && \
                 export PYTHONPATH=/workspace:/workspace/ros2_ws/src/swarm_gazebo:$PYTHONPATH && \
                 python3 scripts/run_gazebo_simulation.py "$@"' -- "$@"

echo "=============================================================================="
echo "[+] Simulation run finished. Visual artifacts and debrief report saved to:"
echo "    $SCRIPT_DIR/output/gazebo_simulation_run/"
echo "=============================================================================="
