# SLAM Subsystem Integration Guide

**Prepared for**: Saqr & Mahmoud (SLAM, Localization & Mapping Subsystem)  
**Authored by**: Mohammed & Ibrahim (Swarm Coordination & Task Allocation Subsystem)  
**Project**: Autonomous Swarm Outdoor Car Search  
**Target Platform**: ROS 2 (Humble / Iron / Rolling) & Native Python 3.9+  

---

## 1. Overview & Architectural Contract

The Swarm Robotics layer (developed by Mohammed & Ibrahim) coordinates multi-robot search operations using decentralized Ant Colony Optimization (ACO). 

To ensure complete modularity and prevent software coupling, the Swarm subsystem **never** implements mapping algorithms or interacts directly with raw sensor hardware (LiDAR, IMU, wheel encoders). Instead, all localization and spatial data flow through the standardized **`ISLAMProvider`** abstract interface located at [`interfaces/slam_interface.py`](file:///Users/aziz/.gemini/antigravity/scratch/swarm_car_search/interfaces/slam_interface.py).

This guide provides everything needed for Saqr & Mahmoud to implement and plug their concrete SLAM provider into the Swarm coordination engine.

```
+--------------------------------------------------------------------------------+
|                        SUBSYSTEM RESPONSIBILITY MATRIX                         |
+--------------------------------------------------------------------------------+
|  Saqr & Mahmoud (SLAM Subsystem)        |  Mohammed & Ibrahim (Swarm Layer)    |
+-----------------------------------------+--------------------------------------+
|  - Sensor Driver Integration (LiDAR)    |  - ACO Frontier Task Scorer          |
|  - EKF / Odometry Fusion (IMU + Wheels) |  - Multi-Layer Stigmergic Pheromones |
|  - 2D Occupancy Grid Generation         |  - Probabilistic Softmax Allocation  |
|  - Obstacle Inflation & Raycasting      |  - Dynamic Failure Detection         |
|  - Frontier Cluster Extraction          |  - Inter-Robot Communication Loop    |
|  - ISLAMProvider Implementation         |  - Vehicle Perception & Plate OCR    |
+--------------------------------------------------------------------------------+
```

---

## 2. Standardized Data Structures

Your SLAM provider must populate and return the dataclasses defined in [`interfaces/slam_interface.py`](file:///Users/aziz/.gemini/antigravity/scratch/swarm_car_search/interfaces/slam_interface.py):

### 2.1 `Pose2D`
Represents the robot's current pose in the global world coordinate frame:
```python
@dataclass(frozen=True)
class Pose2D:
    x: float      # Global X position in meters
    y: float      # Global Y position in meters
    theta: float  # Heading angle in radians, normalized to [-pi, pi]
```
- **Conventions**: Standard ROS REP-103 (+X forward, +Y left, yaw counter-clockwise).

### 2.2 `OccupancyGrid2D`
Represents the explored 2D grid map:
```python
@dataclass
class OccupancyGrid2D:
    width: int            # Number of columns (cells)
    height: int           # Number of rows (cells)
    resolution: float     # Grid cell size in meters/cell (e.g. 0.5 or 1.0)
    origin: Pose2D        # Global coordinates of cell (0, 0)
    data: List[int]       # 1D row-major list of occupancy values
```
- **Cell Value Specifications**:
  - `-1`: Unknown / Unexplored space.
  - `0`: Free, navigable space.
  - `100`: Occupied space (wall, obstacle, parked car).

### 2.3 `FrontierCluster`
Represents clusters of frontier points dividing explored free space from unexplored unknown space:
```python
@dataclass
class FrontierCluster:
    cluster_id: str                          # Unique identifier (e.g., "frontier_0")
    centroid_x: float                        # Global X coordinate of cluster center
    centroid_y: float                        # Global Y coordinate of cluster center
    size: int                                # Number of frontier cells in cluster
    bounding_box: Tuple[float, float, float, float] # (min_x, min_y, max_x, max_y)
```

### 2.4 `SLAMState`
The complete state payload returned for a robot at any given cycle:
```python
@dataclass
class SLAMState:
    robot_id: str
    timestamp: float
    pose: Pose2D
    linear_velocity: float
    angular_velocity: float
    occupancy_grid: OccupancyGrid2D
    frontiers: List[FrontierCluster]
    exploration_ratio: float
    navigation_cost_map: Optional[List[float]] = None
```

---

## 3. Concrete Implementation: `ROS2SLAMProvider`

Below is a complete, production-ready implementation template showing how to bridge ROS 2 topics into the Swarm interface. You can save this file in `slam_providers/ros2_slam_provider.py`.

```python
"""
ROS 2 Concrete SLAM Provider for Saqr & Mahmoud.
Subscribes to standard Nav2 / Cartographer / RTAB-Map topics and implements ISLAMProvider.
"""

import math
import time
from typing import Dict, List, Optional
from interfaces.slam_interface import (
    ISLAMProvider,
    SLAMState,
    Pose2D,
    OccupancyGrid2D,
    FrontierCluster
)

try:
    import rclpy
    from rclpy.node import Node
    from nav_msgs.msg import OccupancyGrid, Odometry
    from geometry_msgs.msg import PoseStamped
    _HAS_ROS2 = True
except ImportError:
    _HAS_ROS2 = False


class ROS2SLAMProvider(ISLAMProvider):
    """
    Concrete SLAM provider that interfaces with ROS 2 navigation stacks.
    Maintains the latest state per robot and fulfills the ISLAMProvider contract.
    """

    def __init__(self, robot_ids: List[str]):
        self._robot_ids = set(robot_ids)
        self._states: Dict[str, SLAMState] = {}
        self._is_healthy: Dict[str, bool] = {rid: False for rid in robot_ids}

    def update_from_ros_messages(
        self,
        robot_id: str,
        pose_x: float,
        pose_y: float,
        yaw: float,
        v_lin: float,
        v_ang: float,
        grid_width: int,
        grid_height: int,
        resolution: float,
        grid_data: List[int],
        frontiers: List[FrontierCluster]
    ) -> None:
        """Call this callback from your ROS 2 subscription callback."""
        # Calculate exploration ratio (fraction of known cells)
        known_cells = sum(1 for c in grid_data if c != -1)
        total_cells = max(1, len(grid_data))
        exp_ratio = min(1.0, known_cells / total_cells)

        pose = Pose2D(x=pose_x, y=pose_y, theta=yaw)
        grid = OccupancyGrid2D(
            width=grid_width,
            height=grid_height,
            resolution=resolution,
            origin=Pose2D(x=0.0, y=0.0, theta=0.0),
            data=grid_data
        )

        state = SLAMState(
            robot_id=robot_id,
            timestamp=time.time(),
            pose=pose,
            linear_velocity=v_lin,
            angular_velocity=v_ang,
            occupancy_grid=grid,
            frontiers=frontiers,
            exploration_ratio=exp_ratio
        )

        self._states[robot_id] = state
        self._is_healthy[robot_id] = True

    def get_latest_state(self, robot_id: str) -> SLAMState:
        """Returns the latest estimated state of the specified robot."""
        if robot_id not in self._states:
            raise KeyError(f"No SLAM state available for robot '{robot_id}'.")
        return self._states[robot_id]

    def is_healthy(self, robot_id: str) -> bool:
        """Returns True if the robot's SLAM state is healthy and fresh."""
        if not self._is_healthy.get(robot_id, False):
            return False
        state = self._states.get(robot_id)
        if state is None:
            return False
        # Declare unhealthy if state is older than 2.0 seconds
        return (time.time() - state.timestamp) < 2.0
```

---

## 4. Verification Checklist Before Hardware Deployment

Before running the full swarm on physical robots, Saqr & Mahmoud should run the automated interface verification test:

```bash
# Verify compliance with ISLAMProvider abstract contract
/usr/bin/python3 -m unittest tests/test_interfaces.py -v
```

### Key Integration Points:
1. **Update Frequency**: The Swarm layer executes at $5\text{Hz}$ to $10\text{Hz}$ ($\Delta t = 0.1\text{s} - 0.2\text{s}$). Your SLAM node should publish updated `SLAMState` at a minimum of $5\text{Hz}$.
2. **Frontier Extraction**: If your mapping stack does not natively extract frontiers, use the helper method in `slam_providers/mock_slam_provider.py` or standard OpenCV edge detection (`cv2.Canny` or `findContours`) between $-1$ and $0$ cells.
3. **Graceful Degradation**: If localization confidence drops below threshold (e.g., LiDAR slip in long corridors), set `is_healthy(robot_id)` to `False`. The Swarm layer's `FailureDetector` will immediately take protective action.
