"""
Frontier Search Task Model and Lifecycle Status for Swarm Task Allocation.
"""

from dataclasses import dataclass, asdict
from enum import Enum
from typing import Dict, Any, Optional, Tuple
from interfaces.slam_interface import FrontierCluster


class TaskStatus(str, Enum):
    """Lifecycle statuses for a candidate frontier search task."""
    PENDING = "PENDING"          # Created, awaiting ACO selection/allocation
    ASSIGNED = "ASSIGNED"        # Allocated to a robot, robot is navigating
    IN_PROGRESS = "IN_PROGRESS"  # Robot arrived at zone and is actively searching
    COMPLETED = "COMPLETED"      # Zone search or vehicle identification finished
    CANCELLED = "CANCELLED"      # Obsolete due to map updates or duplicate coverage
    FAILED = "FAILED"            # Assigned robot failed or abandoned task


class TaskType(str, Enum):
    """Semantic category of task for prioritized swarm allocation."""
    EXPLORATION = "EXPLORATION"
    VEHICLE_INVESTIGATION = "VEHICLE_INVESTIGATION"
    TARGET_CONFIRMATION = "TARGET_CONFIRMATION"


@dataclass
class SearchTask:
    """
    Candidate search task representing a spatial frontier cluster extracted from SLAM
    or an investigated candidate vehicle requiring confirmation.
    Carries all features needed for ACO-based multi-objective task evaluation.
    """
    task_id: str
    cluster_id: str
    centroid_x: float
    centroid_y: float
    bounding_box: Tuple[float, float, float, float]
    size: int                         # Number of frontier cells in cluster
    exploration_value: float = 1.0     # Information gain metric
    detection_probability: float = 0.5 # Urban zone prior vehicle probability
    priority: float = 1.0              # Dynamic mission priority multiplier
    pheromone_exploration: float = 0.0 # Positive exploration memory
    pheromone_success: float = 0.0     # Target car discovery indicator
    pheromone_avoidance: float = 0.0   # Current robot reservation repulsion
    task_type: TaskType = TaskType.EXPLORATION
    target_vehicle_id: Optional[str] = None
    status: TaskStatus = TaskStatus.PENDING
    assigned_robot: Optional[str] = None
    created_at: float = 0.0
    last_updated: float = 0.0
    expiration_time: Optional[float] = None

    def is_expired(self, current_time: float) -> bool:
        """Checks whether the task has exceeded its valid lifetime."""
        if self.expiration_time is None:
            return False
        return current_time > self.expiration_time

    def mark_assigned(self, robot_id: str, timestamp: float) -> None:
        """Assigns the task to a specific robot."""
        self.assigned_robot = robot_id
        self.status = TaskStatus.ASSIGNED
        self.last_updated = timestamp

    def mark_in_progress(self, timestamp: float) -> None:
        """Updates task state to actively being searched."""
        self.status = TaskStatus.IN_PROGRESS
        self.last_updated = timestamp

    def mark_completed(self, timestamp: float) -> None:
        """Marks task as fully investigated or explored."""
        self.status = TaskStatus.COMPLETED
        self.last_updated = timestamp

    def mark_released(self, timestamp: float) -> None:
        """Releases assignment so other robots can bid/claim it."""
        self.assigned_robot = None
        self.status = TaskStatus.PENDING
        self.last_updated = timestamp

    @classmethod
    def from_frontier(
        cls,
        frontier: FrontierCluster,
        timestamp: float,
        detection_probability: float = 0.5,
        priority: float = 1.0,
        ttl_seconds: Optional[float] = 120.0,
    ) -> "SearchTask":
        """
        Factory method constructing a SearchTask directly from a SLAM FrontierCluster.
        """
        task_id = f"task_{frontier.cluster_id}_{int(frontier.centroid_x)}_{int(frontier.centroid_y)}_{int(timestamp)}"
        expiration = (timestamp + ttl_seconds) if ttl_seconds is not None else None
        return cls(
            task_id=task_id,
            cluster_id=frontier.cluster_id,
            centroid_x=frontier.centroid_x,
            centroid_y=frontier.centroid_y,
            bounding_box=frontier.bounding_box,
            size=frontier.size,
            exploration_value=float(frontier.size),
            detection_probability=detection_probability,
            priority=priority,
            created_at=timestamp,
            last_updated=timestamp,
            expiration_time=expiration,
        )

    @classmethod
    def from_vehicle_detection(
        cls,
        x: float,
        y: float,
        timestamp: float,
        target_probability: float = 0.85,
        priority: float = 2.5,
        vehicle_id: Optional[str] = None,
        task_type: TaskType = TaskType.TARGET_CONFIRMATION,
        cluster_id: Optional[str] = None,
        ttl_seconds: Optional[float] = 180.0,
    ) -> "SearchTask":
        """
        Factory method constructing a SearchTask for vehicle investigation/confirmation.
        """
        cid = cluster_id or (f"veh_{vehicle_id}" if vehicle_id else "veh_candidate")
        task_id = f"task_{cid}_{int(x)}_{int(y)}_{int(timestamp)}"
        expiration = (timestamp + ttl_seconds) if ttl_seconds is not None else None
        radius = 2.5
        bbox = (x - radius, y - radius, x + radius, y + radius)
        return cls(
            task_id=task_id,
            cluster_id=cid,
            centroid_x=x,
            centroid_y=y,
            bounding_box=bbox,
            size=20,  # High information potential
            exploration_value=20.0,
            detection_probability=target_probability,
            priority=priority,
            task_type=task_type,
            target_vehicle_id=vehicle_id,
            created_at=timestamp,
            last_updated=timestamp,
            expiration_time=expiration,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Serializes SearchTask to a JSON-compatible dictionary."""
        data = asdict(self)
        data["status"] = self.status.value
        data["task_type"] = self.task_type.value
        data["bounding_box"] = list(self.bounding_box)
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SearchTask":
        """Deserializes SearchTask from a dictionary."""
        d = dict(data)
        d["status"] = TaskStatus(d["status"])
        if "task_type" in d:
            d["task_type"] = TaskType(d["task_type"])
        d["bounding_box"] = tuple(d["bounding_box"])
        return cls(**d)
