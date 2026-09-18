# SLAM ↔ Swarm Subsystem Interface Specification

## 1. Purpose & Guiding Principles

This document establishes the formal software contract between:
- **Provider**: The SLAM Module (engineered by Saqr and Mahmoud).
- **Consumer**: The Swarm Robotics Coordination Module (engineered by Mohammed and Ibrahim).

The interface is completely decoupled using Python Abstract Base Classes (`abc.ABC`). Neither module depends directly on internal implementation details of the other.

---

## 2. Interface Contract (`ISLAMProvider`)

Located in [`interfaces/slam_interface.py`](file:///Users/aziz/.gemini/antigravity/scratch/swarm_car_search/interfaces/slam_interface.py):

```python
from abc import ABC, abstractmethod
from typing import List, Optional

class ISLAMProvider(ABC):
    """Abstract contract that any SLAM implementation must fulfill."""

    @abstractmethod
    def get_latest_state(self, robot_id: str) -> 'SLAMState':
        """Returns the most recent estimated state of the specified robot."""
        pass

    @abstractmethod
    def is_healthy(self, robot_id: str) -> bool:
        """Indicates whether localization and mapping are currently valid."""
        pass
```

---

## 3. Data Structures

### 3.1 `Pose2D`
Represents the estimated robot coordinates and heading in the global frame:
- `x: float`: Global X coordinate in meters.
- `y: float`: Global Y coordinate in meters.
- `theta: float`: Heading angle in radians $[-\pi, \pi]$.

### 3.2 `OccupancyGrid2D`
Represents the 2D probabilistic occupancy map:
- `width: int`: Number of columns in grid cells.
- `height: int`: Number of rows in grid cells.
- `resolution: float`: Meter size per grid cell edge (e.g., 0.5 m/cell).
- `origin: Pose2D`: Global coordinates of the grid cell $(0, 0)$.
- `data: List[int]`: 1D row-major array with values:
  - `-1`: Unknown space.
  - `0`: Free navigable space.
  - `100`: Static obstacle / occupied cell.

### 3.3 `FrontierCluster`
Represents clustered frontier points dividing explored free space from unexplored unknown space:
- `cluster_id: str`: Unique identifier.
- `centroid_x: float`: Global centroid X coordinate in meters.
- `centroid_y: float`: Global centroid Y coordinate in meters.
- `size: int`: Number of frontier cells within the cluster (information potential).
- `bounding_box: Tuple[float, float, float, float]`: `(min_x, min_y, max_x, max_y)`.

### 3.4 `SLAMState`
The comprehensive snapshot delivered to the Swarm Decision Layer per cycle:
- `robot_id: str`: Unique robot identifier.
- `timestamp: float`: Epoch timestamp in seconds.
- `pose: Pose2D`: Current robot pose.
- `linear_velocity: float`: Current estimated linear speed ($m/s$).
- `angular_velocity: float`: Current estimated angular yaw rate ($rad/s$).
- `occupancy_grid: OccupancyGrid2D`: Global or merged local grid map.
- `frontiers: List[FrontierCluster]`: Actively identified frontier search areas.
- `exploration_ratio: float`: Fraction of total mission area explored $[0.0, 1.0]$.
- `navigation_cost_map: Optional[List[float]]`: Cell-level traversal costs.

---

## 4. Replacement Strategy

1. **Development & Phase 0–6**: The Swarm team uses `MockSLAMProvider` which generates realistic simulated poses, expanding circular grids, and synthesized frontiers.
2. **Phase 7 onwards**: Saqr and Mahmoud supply `ROSSLAMProvider` or `RealSLAMProvider` inheriting from `ISLAMProvider`.
3. **No Swarm code modifications**: The swap is performed purely by injecting the new provider instance at runtime.
