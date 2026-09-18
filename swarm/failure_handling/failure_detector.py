"""
Robust Swarm Robot Failure Detection Engine.
Monitors heartbeats, battery levels, kinematic stalls, and explicit hardware faults.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
import math
from typing import Dict, List, Optional, Set, Tuple

from interfaces.slam_interface import Pose2D
from swarm.states.robot_states import RobotState


class FailureType(str, Enum):
    """Classification of robot failure modes in the field."""
    HEARTBEAT_TIMEOUT = "HEARTBEAT_TIMEOUT"      # Robot ceased RF communication / heartbeats
    BATTERY_CRITICAL = "BATTERY_CRITICAL"        # Battery dropped below critical operational reserve
    STALLED = "STALLED"                          # Robot stationary despite active motion commands (trapped)
    EXPLICIT_FAULT = "EXPLICIT_FAULT"            # Internal sensor/actuator diagnostic fault (state FAILED)


@dataclass
class RobotFailureEvent:
    """Detailed record of an identified robot failure occurrence."""
    robot_id: str
    failure_type: FailureType
    timestamp: float
    last_known_pose: Pose2D
    abandoned_task_id: Optional[str] = None
    battery_level: float = 1.0
    details: str = ""


@dataclass
class FailureDetectorConfig:
    """Threshold settings for multi-mode failure identification."""
    heartbeat_timeout_sec: float = 6.0
    battery_critical_threshold: float = 0.10
    stall_timeout_sec: float = 8.0
    stall_distance_threshold: float = 0.3


@dataclass
class _RobotTelemetryRecord:
    """Internal historical tracking for a single robot."""
    robot_id: str
    last_heartbeat: float
    last_pose: Pose2D
    battery_level: float
    state: RobotState
    assigned_task_id: Optional[str] = None
    stall_start_time: Optional[float] = None
    stall_anchor_pose: Optional[Pose2D] = None


class FailureDetector:
    """
    Centralized or localized failure detector evaluating robot health telemetry.
    Detects silent communications loss, power exhaustion, physical entrapment,
    and catastrophic FSM failures.
    """

    def __init__(self, config: Optional[FailureDetectorConfig] = None) -> None:
        self.config = config or FailureDetectorConfig()
        self._records: Dict[str, _RobotTelemetryRecord] = {}
        self._failed_robots: Dict[str, RobotFailureEvent] = {}

    def record_telemetry(
        self,
        robot_id: str,
        timestamp: float,
        pose: Pose2D,
        battery_level: float,
        state: RobotState,
        assigned_task_id: Optional[str] = None,
    ) -> None:
        """
        Updates tracking history for a robot from an incoming heartbeat or direct telemetry.
        """
        if robot_id not in self._records:
            self._records[robot_id] = _RobotTelemetryRecord(
                robot_id=robot_id,
                last_heartbeat=timestamp,
                last_pose=pose,
                battery_level=battery_level,
                state=state,
                assigned_task_id=assigned_task_id,
                stall_start_time=timestamp if state in (RobotState.NAVIGATING, RobotState.SEARCHING) else None,
                stall_anchor_pose=pose,
            )
            return

        rec = self._records[robot_id]
        rec.last_heartbeat = timestamp
        rec.battery_level = battery_level
        rec.state = state
        rec.assigned_task_id = assigned_task_id

        # Evaluate movement for stall tracking
        if state in (RobotState.NAVIGATING, RobotState.SEARCHING):
            if rec.stall_anchor_pose is None:
                rec.stall_anchor_pose = pose
                rec.stall_start_time = timestamp
            else:
                dist = math.hypot(pose.x - rec.stall_anchor_pose.x, pose.y - rec.stall_anchor_pose.y)
                if dist >= self.config.stall_distance_threshold:
                    # Made sufficient displacement; reset stall anchor
                    rec.stall_anchor_pose = pose
                    rec.stall_start_time = timestamp
        else:
            # Idle, returning, or reporting: not considered stalled
            rec.stall_anchor_pose = None
            rec.stall_start_time = None

        rec.last_pose = pose

    def check_failures(self, current_time: float) -> List[RobotFailureEvent]:
        """
        Scans all monitored robots for threshold violations.
        Returns newly detected failure events.
        """
        new_failures: List[RobotFailureEvent] = []

        for robot_id, rec in list(self._records.items()):
            if robot_id in self._failed_robots:
                continue

            # 1. Check explicit internal fault
            if rec.state == RobotState.FAILED:
                event = RobotFailureEvent(
                    robot_id=robot_id,
                    failure_type=FailureType.EXPLICIT_FAULT,
                    timestamp=current_time,
                    last_known_pose=rec.last_pose,
                    abandoned_task_id=rec.assigned_task_id,
                    battery_level=rec.battery_level,
                    details="Robot reported FAILED internal state",
                )
                self._failed_robots[robot_id] = event
                new_failures.append(event)
                continue

            # 2. Check heartbeat silence / RF loss
            silence_duration = current_time - rec.last_heartbeat
            if silence_duration > self.config.heartbeat_timeout_sec:
                event = RobotFailureEvent(
                    robot_id=robot_id,
                    failure_type=FailureType.HEARTBEAT_TIMEOUT,
                    timestamp=current_time,
                    last_known_pose=rec.last_pose,
                    abandoned_task_id=rec.assigned_task_id,
                    battery_level=rec.battery_level,
                    details=f"Heartbeat silent for {silence_duration:.2f}s (threshold: {self.config.heartbeat_timeout_sec:.2f}s)",
                )
                self._failed_robots[robot_id] = event
                new_failures.append(event)
                continue

            # 3. Check critical battery exhaustion
            if rec.battery_level < self.config.battery_critical_threshold:
                event = RobotFailureEvent(
                    robot_id=robot_id,
                    failure_type=FailureType.BATTERY_CRITICAL,
                    timestamp=current_time,
                    last_known_pose=rec.last_pose,
                    abandoned_task_id=rec.assigned_task_id,
                    battery_level=rec.battery_level,
                    details=f"Battery level {rec.battery_level * 100:.1f}% below critical limit {self.config.battery_critical_threshold * 100:.1f}%",
                )
                self._failed_robots[robot_id] = event
                new_failures.append(event)
                continue

            # 4. Check physical stall / entrapment
            if rec.stall_start_time is not None and rec.stall_anchor_pose is not None:
                stall_duration = current_time - rec.stall_start_time
                if stall_duration > self.config.stall_timeout_sec:
                    dist = math.hypot(
                        rec.last_pose.x - rec.stall_anchor_pose.x,
                        rec.last_pose.y - rec.stall_anchor_pose.y,
                    )
                    if dist < self.config.stall_distance_threshold:
                        event = RobotFailureEvent(
                            robot_id=robot_id,
                            failure_type=FailureType.STALLED,
                            timestamp=current_time,
                            last_known_pose=rec.last_pose,
                            abandoned_task_id=rec.assigned_task_id,
                            battery_level=rec.battery_level,
                            details=f"Robot stalled in place ({dist:.2f}m in {stall_duration:.2f}s) while {rec.state.value}",
                        )
                        self._failed_robots[robot_id] = event
                        new_failures.append(event)
                        continue

        return new_failures

    def is_failed(self, robot_id: str) -> bool:
        """Checks if a robot is currently recorded as failed."""
        return robot_id in self._failed_robots

    def get_failed_event(self, robot_id: str) -> Optional[RobotFailureEvent]:
        """Returns the failure event for a failed robot, or None if healthy."""
        return self._failed_robots.get(robot_id)

    def get_all_failed_robots(self) -> Set[str]:
        """Returns set of all identified failed robot IDs."""
        return set(self._failed_robots.keys())

    def reset_robot(self, robot_id: str) -> None:
        """Clears failure status for a recovered or rebooted robot."""
        if robot_id in self._failed_robots:
            del self._failed_robots[robot_id]
        if robot_id in self._records:
            del self._records[robot_id]
