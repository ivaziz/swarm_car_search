"""
Decentralized Swarm State Representation and Peer Tracking.
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any
from interfaces.slam_interface import Pose2D
from swarm.states.robot_states import RobotState
from swarm.task_allocation.frontier_task import SearchTask, TaskStatus


@dataclass
class RobotPeerState:
    """Estimated state of a peer robot in the swarm."""
    robot_id: str
    state: RobotState
    pose: Pose2D
    assigned_task_id: Optional[str] = None
    battery_level: float = 1.0
    last_heartbeat: float = 0.0
    is_active: bool = True

    def to_dict(self) -> Dict[str, Any]:
        """Convert peer state to dictionary."""
        return {
            "robot_id": self.robot_id,
            "state": self.state.value,
            "pose": {"x": self.pose.x, "y": self.pose.y, "theta": self.pose.theta},
            "assigned_task_id": self.assigned_task_id,
            "battery_level": self.battery_level,
            "last_heartbeat": self.last_heartbeat,
            "is_active": self.is_active,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RobotPeerState":
        """Reconstruct peer state from dictionary."""
        pose_data = data["pose"]
        return cls(
            robot_id=data["robot_id"],
            state=RobotState(data["state"]),
            pose=Pose2D(x=pose_data["x"], y=pose_data["y"], theta=pose_data["theta"]),
            assigned_task_id=data.get("assigned_task_id"),
            battery_level=data.get("battery_level", 1.0),
            last_heartbeat=data.get("last_heartbeat", 0.0),
            is_active=data.get("is_active", True),
        )


@dataclass
class SwarmState:
    """
    Local node's perspective of the entire multi-robot swarm.
    Maintained independently on each robot via peer heartbeats and message exchanges.
    """
    local_robot_id: str
    timestamp: float = 0.0
    peers: Dict[str, RobotPeerState] = field(default_factory=dict)
    active_tasks: Dict[str, SearchTask] = field(default_factory=dict)
    global_exploration_ratio: float = 0.0
    target_found: bool = False
    confirmed_target_location: Optional[Pose2D] = None

    def update_or_add_peer(
        self,
        robot_id: str,
        state: RobotState,
        pose: Pose2D,
        timestamp: float,
        battery_level: float = 1.0,
        assigned_task_id: Optional[str] = None,
    ) -> None:
        """Updates or registers a peer robot state from a received heartbeat."""
        self.peers[robot_id] = RobotPeerState(
            robot_id=robot_id,
            state=state,
            pose=pose,
            assigned_task_id=assigned_task_id,
            battery_level=battery_level,
            last_heartbeat=timestamp,
            is_active=True,
        )

    def mark_peer_inactive(self, robot_id: str) -> None:
        """Marks a peer as inactive / failed after heartbeat timeout."""
        if robot_id in self.peers:
            self.peers[robot_id].is_active = False
            # If the failed peer had an assigned task, release that task
            failed_task_id = self.peers[robot_id].assigned_task_id
            if failed_task_id and failed_task_id in self.active_tasks:
                self.active_tasks[failed_task_id].mark_released(self.timestamp)
            self.peers[robot_id].assigned_task_id = None

    def check_peer_timeouts(self, current_time: float, timeout_sec: float) -> List[str]:
        """
        Evaluates peer heartbeats against timeout threshold.
        Marks silent peers as inactive, releases any assigned tasks,
        and returns list of newly timed-out robot IDs.
        """
        timed_out: List[str] = []
        for peer_id, peer in self.peers.items():
            if peer.is_active:
                if (current_time - peer.last_heartbeat) > timeout_sec:
                    self.mark_peer_inactive(peer_id)
                    timed_out.append(peer_id)
        return timed_out

    def get_active_peers(self) -> List[RobotPeerState]:
        """Returns all peer robots currently considered active and communicative."""
        return [peer for peer in self.peers.values() if peer.is_active]

    def get_tasks_by_status(self, status: TaskStatus) -> List[SearchTask]:
        """Filters active tasks by their lifecycle status."""
        return [task for task in self.active_tasks.values() if task.status == status]

    def get_robot_assigned_task(self, robot_id: str) -> Optional[SearchTask]:
        """Retrieves the task currently assigned to a specific robot, if any."""
        for task in self.active_tasks.values():
            if task.assigned_robot == robot_id and task.status in (TaskStatus.ASSIGNED, TaskStatus.IN_PROGRESS):
                return task
        return None

    def to_dict(self) -> Dict[str, Any]:
        """Serializes full SwarmState to dictionary."""
        return {
            "local_robot_id": self.local_robot_id,
            "timestamp": self.timestamp,
            "peers": {rid: p.to_dict() for rid, p in self.peers.items()},
            "active_tasks": {tid: t.to_dict() for tid, t in self.active_tasks.items()},
            "global_exploration_ratio": self.global_exploration_ratio,
            "target_found": self.target_found,
            "confirmed_target_location": (
                {"x": self.confirmed_target_location.x, "y": self.confirmed_target_location.y, "theta": self.confirmed_target_location.theta}
                if self.confirmed_target_location else None
            ),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SwarmState":
        """Reconstructs SwarmState from dictionary."""
        peers = {rid: RobotPeerState.from_dict(p) for rid, p in data.get("peers", {}).items()}
        tasks = {tid: SearchTask.from_dict(t) for tid, t in data.get("active_tasks", {}).items()}
        loc = data.get("confirmed_target_location")
        target_loc = Pose2D(x=loc["x"], y=loc["y"], theta=loc["theta"]) if loc else None
        return cls(
            local_robot_id=data["local_robot_id"],
            timestamp=data.get("timestamp", 0.0),
            peers=peers,
            active_tasks=tasks,
            global_exploration_ratio=data.get("global_exploration_ratio", 0.0),
            target_found=data.get("target_found", False),
            confirmed_target_location=target_loc,
        )
