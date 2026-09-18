"""
Base Abstract Class for SLAM Providers fulfilling ISLAMProvider.
Provides common caching, health validation, and registry mechanics.
"""

from typing import Dict, Optional
from interfaces.slam_interface import ISLAMProvider, SLAMState


class BaseSLAMProvider(ISLAMProvider):
    """
    Base scaffolding class for SLAM implementations (MockSLAMProvider, ROS2SLAMProvider).
    """

    def __init__(self) -> None:
        self._latest_states: Dict[str, SLAMState] = {}
        self._health_status: Dict[str, bool] = {}

    def update_cached_state(self, robot_id: str, state: SLAMState) -> None:
        """Stores the latest estimated SLAM state in internal cache."""
        self._latest_states[robot_id] = state
        self._health_status[robot_id] = True

    def set_healthy(self, robot_id: str, healthy: bool) -> None:
        """Manually override or update healthy status for a specific robot."""
        self._health_status[robot_id] = healthy

    def get_latest_state(self, robot_id: str) -> SLAMState:
        """
        Retrieves the latest known state from the cache or raises KeyError.
        Concrete subclasses can override to fetch live data from ROS2 or simulator.
        """
        if robot_id not in self._latest_states:
            raise KeyError(f"No SLAM state available for robot_id: '{robot_id}'")
        return self._latest_states[robot_id]

    def is_healthy(self, robot_id: str) -> bool:
        """Returns the health status of localization and mapping for robot_id."""
        return self._health_status.get(robot_id, False)
