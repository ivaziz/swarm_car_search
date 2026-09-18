"""
Abstract Interface and Data Structures for SLAM Subsystem.

Engineered for Saqr & Mahmoud to implement and Mohammed & Ibrahim to consume.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


@dataclass(frozen=True)
class Pose2D:
    """2D Position and orientation in global coordinate frame."""
    x: float
    y: float
    theta: float  # Orientation heading in radians [-pi, pi]


@dataclass
class OccupancyGrid2D:
    """
    2D Occupancy Grid representation of explored/unexplored urban space.
    Cell values:
      -1: Unknown / Unexplored space
       0: Free space (navigable)
     100: Occupied space (obstacle/wall)
    """
    width: int
    height: int
    resolution: float  # Meters per cell edge
    origin: Pose2D
    data: List[int] = field(default_factory=list)

    @property
    def total_cells(self) -> int:
        return self.width * self.height


@dataclass(frozen=True)
class FrontierCluster:
    """
    Clustered boundary region between known free space and unknown space.
    Candidate target for multi-robot exploration.
    """
    cluster_id: str
    centroid_x: float
    centroid_y: float
    size: int  # Number of frontier cells in this cluster
    bounding_box: Tuple[float, float, float, float]  # (min_x, min_y, max_x, max_y)


@dataclass
class SLAMState:
    """
    Unified state snapshot provided by SLAM for a single robot at a specific time.
    """
    robot_id: str
    timestamp: float
    pose: Pose2D
    linear_velocity: float
    angular_velocity: float
    occupancy_grid: OccupancyGrid2D
    frontiers: List[FrontierCluster] = field(default_factory=list)
    exploration_ratio: float = 0.0  # Explored free space ratio [0.0, 1.0]
    navigation_cost_map: Optional[List[float]] = None


class ISLAMProvider(ABC):
    """
    Abstract Base Class representing the contract that any SLAM provider
    (MockSLAMProvider or real ROS2-backed SLAM) must satisfy.
    """

    @abstractmethod
    def get_latest_state(self, robot_id: str) -> SLAMState:
        """
        Retrieve the most recent SLAM estimation for the specified robot.

        Args:
            robot_id: The unique identifier of the robot.

        Returns:
            SLAMState containing pose, map, and extracted frontiers.
        """
        pass

    @abstractmethod
    def is_healthy(self, robot_id: str) -> bool:
        """
        Check if the SLAM tracking and mapping pipeline is healthy.

        Args:
            robot_id: The unique identifier of the robot.

        Returns:
            True if localization and mapping are valid, False otherwise.
        """
        pass
