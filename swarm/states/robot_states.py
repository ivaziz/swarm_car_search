"""
Robot Finite State Machine (FSM) States and Transition Validations.
"""

from enum import Enum
from typing import Dict, Set


class RobotState(str, Enum):
    """
    Operational states of an autonomous search robot in the swarm.
    """
    IDLE = "IDLE"                          # Awaiting initialization or mission task
    NAVIGATING = "NAVIGATING"              # Moving toward assigned frontier or waypoint
    SEARCHING = "SEARCHING"                # Actively searching/scanning within an assigned zone
    VEHICLE_DETECTED = "VEHICLE_DETECTED"  # Visual detector spotted candidate vehicle
    VERIFYING = "VERIFYING"                # Processing plate OCR and verifying against watchlist
    REPORTING = "REPORTING"                # Dispatched confirmed match to swarm and cloud server
    REASSIGNING = "REASSIGNING"            # Task ended/released, computing next frontier assignment
    RETURNING = "RETURNING"                # Mission complete or battery low, returning to rendezvous
    COMMUNICATION_LOST = "COMMUNICATION_LOST" # Heartbeat lost, running local autonomous exploration
    COMPLETED = "COMPLETED"                # Assigned search area or entire mission finalized
    FAILED = "FAILED"                      # Critical hardware or navigation error


# Directed graph of permissible state transitions
VALID_STATE_TRANSITIONS: Dict[RobotState, Set[RobotState]] = {
    RobotState.IDLE: {
        RobotState.NAVIGATING,
        RobotState.SEARCHING,
        RobotState.COMMUNICATION_LOST,
        RobotState.FAILED,
    },
    RobotState.NAVIGATING: {
        RobotState.SEARCHING,
        RobotState.VEHICLE_DETECTED,
        RobotState.REASSIGNING,
        RobotState.COMMUNICATION_LOST,
        RobotState.FAILED,
        RobotState.RETURNING,
    },
    RobotState.SEARCHING: {
        RobotState.VEHICLE_DETECTED,
        RobotState.VERIFYING,
        RobotState.REASSIGNING,
        RobotState.NAVIGATING,
        RobotState.COMMUNICATION_LOST,
        RobotState.RETURNING,
        RobotState.COMPLETED,
        RobotState.FAILED,
    },
    RobotState.VEHICLE_DETECTED: {
        RobotState.VERIFYING,
        RobotState.REPORTING,
        RobotState.SEARCHING,
        RobotState.NAVIGATING,
        RobotState.COMMUNICATION_LOST,
        RobotState.FAILED,
    },
    RobotState.VERIFYING: {
        RobotState.REPORTING,       # Confirmed match
        RobotState.SEARCHING,       # False positive / non-target vehicle
        RobotState.NAVIGATING,      # Approach candidate vehicle for closer inspection
        RobotState.COMMUNICATION_LOST,
        RobotState.FAILED,
    },
    RobotState.REPORTING: {
        RobotState.SEARCHING,
        RobotState.REASSIGNING,
        RobotState.RETURNING,
        RobotState.COMPLETED,
        RobotState.FAILED,
    },
    RobotState.REASSIGNING: {
        RobotState.NAVIGATING,
        RobotState.SEARCHING,
        RobotState.RETURNING,
        RobotState.COMPLETED,
        RobotState.FAILED,
    },
    RobotState.RETURNING: {
        RobotState.IDLE,
        RobotState.COMPLETED,
        RobotState.FAILED,
    },
    RobotState.COMMUNICATION_LOST: {
        RobotState.SEARCHING,
        RobotState.NAVIGATING,
        RobotState.RETURNING,
        RobotState.FAILED,
    },
    RobotState.COMPLETED: {
        RobotState.IDLE,
    },
    RobotState.FAILED: set(),  # Terminal state until manually reset
}


def is_valid_transition(current_state: RobotState, next_state: RobotState) -> bool:
    """
    Validates whether transitioning from current_state to next_state is logically permissible.

    Args:
        current_state: The current operational state.
        next_state: The proposed target state.

    Returns:
        True if transition is allowed, False otherwise.
    """
    if current_state == next_state:
        return True
    allowed = VALID_STATE_TRANSITIONS.get(current_state, set())
    return next_state in allowed
