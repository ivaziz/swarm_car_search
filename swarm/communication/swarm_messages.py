"""
Inter-Robot Communication Protocols and Swarm Message Schemas.
"""

import json
import uuid
from dataclasses import dataclass, asdict, field
from enum import Enum
from typing import Dict, Any, Optional
from interfaces.slam_interface import Pose2D
from interfaces.perception_interface import MatchEvent
from swarm.states.robot_states import RobotState


class MessageType(str, Enum):
    """Types of broadcast and unicast messages in the swarm communication layer."""
    HEARTBEAT = "HEARTBEAT"              # Periodic liveness and pose broadcast
    STATE_UPDATE = "STATE_UPDATE"        # FSM state change notification
    TASK_CLAIM = "TASK_CLAIM"            # Declaration of frontier task reservation
    TASK_RELEASE = "TASK_RELEASE"        # Task completion or abandonment
    PHEROMONE_SYNC = "PHEROMONE_SYNC"    # Dispersal of local pheromone trail updates
    VEHICLE_ALERT = "VEHICLE_ALERT"      # High-priority alert of vehicle or plate match
    FAILURE_ALERT = "FAILURE_ALERT"      # Peer failure timeout broadcast


@dataclass
class SwarmMessage:
    """
    Standard message envelope exchanged between robot peers or sent to base station.
    """
    message_id: str
    sender_id: str
    timestamp: float
    msg_type: MessageType
    payload: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert message to dictionary."""
        return {
            "message_id": self.message_id,
            "sender_id": self.sender_id,
            "timestamp": self.timestamp,
            "msg_type": self.msg_type.value,
            "payload": self.payload,
        }

    def to_json(self) -> str:
        """Serialize message to JSON string."""
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SwarmMessage":
        """Reconstruct SwarmMessage from dictionary."""
        return cls(
            message_id=data["message_id"],
            sender_id=data["sender_id"],
            timestamp=data["timestamp"],
            msg_type=MessageType(data["msg_type"]),
            payload=data.get("payload", {}),
        )

    @classmethod
    def from_json(cls, json_str: str) -> "SwarmMessage":
        """Deserialize SwarmMessage from JSON string."""
        return cls.from_dict(json.loads(json_str))


# Helper factories for standardized message construction

def create_heartbeat_message(
    sender_id: str,
    timestamp: float,
    state: RobotState,
    pose: Pose2D,
    battery_level: float = 1.0,
    current_task_id: Optional[str] = None,
) -> SwarmMessage:
    """Constructs a standard periodic heartbeat message."""
    payload = {
        "state": state.value,
        "pose": {"x": pose.x, "y": pose.y, "theta": pose.theta},
        "battery_level": battery_level,
        "current_task_id": current_task_id,
    }
    return SwarmMessage(
        message_id=str(uuid.uuid4()),
        sender_id=sender_id,
        timestamp=timestamp,
        msg_type=MessageType.HEARTBEAT,
        payload=payload,
    )


def create_task_claim_message(
    sender_id: str,
    timestamp: float,
    task_id: str,
    bid_score: float,
) -> SwarmMessage:
    """Constructs a task reservation claim message."""
    payload = {
        "task_id": task_id,
        "bid_score": bid_score,
    }
    return SwarmMessage(
        message_id=str(uuid.uuid4()),
        sender_id=sender_id,
        timestamp=timestamp,
        msg_type=MessageType.TASK_CLAIM,
        payload=payload,
    )


def create_task_release_message(
    sender_id: str,
    timestamp: float,
    task_id: str,
    reason: str = "COMPLETED",
) -> SwarmMessage:
    """Constructs a task release broadcast."""
    payload = {
        "task_id": task_id,
        "reason": reason,
    }
    return SwarmMessage(
        message_id=str(uuid.uuid4()),
        sender_id=sender_id,
        timestamp=timestamp,
        msg_type=MessageType.TASK_RELEASE,
        payload=payload,
    )


def create_vehicle_alert_message(
    sender_id: str,
    timestamp: float,
    match_event: MatchEvent,
) -> SwarmMessage:
    """Constructs a high-priority vehicle detection broadcast message."""
    payload = {
        "target_id": match_event.target_id,
        "plate_number": match_event.plate_number,
        "match_confidence": match_event.match_confidence,
        "confirmed": match_event.confirmed,
        "location": list(match_event.location),
        "evidence_reference": match_event.evidence_reference,
    }
    return SwarmMessage(
        message_id=str(uuid.uuid4()),
        sender_id=sender_id,
        timestamp=timestamp,
        msg_type=MessageType.VEHICLE_ALERT,
        payload=payload,
    )
