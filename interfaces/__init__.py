"""
Core abstract interfaces connecting the Swarm Subsystem with SLAM, Perception, and Server layers.
"""

from .slam_interface import (
    Pose2D,
    OccupancyGrid2D,
    FrontierCluster,
    SLAMState,
    ISLAMProvider,
)
from .perception_interface import (
    BoundingBox2D,
    VehicleDetectionEvent,
    PlateDetectionEvent,
    MatchEvent,
    IPerceptionProvider,
)
from .server_interface import (
    MissionReport,
    DetectionReport,
    IServerReporter,
)

__all__ = [
    "Pose2D",
    "OccupancyGrid2D",
    "FrontierCluster",
    "SLAMState",
    "ISLAMProvider",
    "BoundingBox2D",
    "VehicleDetectionEvent",
    "PlateDetectionEvent",
    "MatchEvent",
    "IPerceptionProvider",
    "MissionReport",
    "DetectionReport",
    "IServerReporter",
]
