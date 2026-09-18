"""
Finite State Machine Engine with Transition Validation, Callbacks, and Event History.
"""

from typing import Callable, Dict, List, Optional, Tuple
from .robot_states import RobotState, is_valid_transition


class RobotStateMachine:
    """
    Event-driven Finite State Machine governing the operational lifecycle of a robot.
    Guarantees that state transitions adhere strictly to the VALID_STATE_TRANSITIONS graph.
    """

    def __init__(
        self,
        robot_id: str,
        initial_state: RobotState = RobotState.IDLE,
        initial_timestamp: float = 0.0,
    ) -> None:
        self.robot_id = robot_id
        self._current_state = initial_state
        self._state_enter_timestamp = initial_timestamp

        # History log: (timestamp, from_state, to_state, reason)
        self.history: List[Tuple[float, RobotState, RobotState, str]] = [
            (initial_timestamp, initial_state, initial_state, "INITIAL_STATE")
        ]

        # Callbacks
        self._on_enter_callbacks: Dict[RobotState, List[Callable[[RobotState, float, str], None]]] = {
            s: [] for s in RobotState
        }
        self._on_exit_callbacks: Dict[RobotState, List[Callable[[RobotState, float, str], None]]] = {
            s: [] for s in RobotState
        }

    @property
    def current_state(self) -> RobotState:
        """Returns the current active state."""
        return self._current_state

    @property
    def time_entered(self) -> float:
        """Returns the timestamp when current state was entered."""
        return self._state_enter_timestamp

    def time_in_state(self, current_timestamp: float) -> float:
        """Returns elapsed duration in the current state."""
        return max(0.0, current_timestamp - self._state_enter_timestamp)

    def register_on_enter(
        self, state: RobotState, callback: Callable[[RobotState, float, str], None]
    ) -> None:
        """Registers a callback executed when entering the specified state."""
        self._on_enter_callbacks[state].append(callback)

    def register_on_exit(
        self, state: RobotState, callback: Callable[[RobotState, float, str], None]
    ) -> None:
        """Registers a callback executed when exiting the specified state."""
        self._on_exit_callbacks[state].append(callback)

    def transition_to(
        self,
        next_state: RobotState,
        timestamp: float,
        reason: str = "",
        strict: bool = True,
    ) -> bool:
        """
        Attempts a state transition.

        Args:
            next_state: Target RobotState.
            timestamp: Event occurrence time.
            reason: Contextual description of the trigger.
            strict: If True, raises ValueError on invalid transition. If False, returns False.

        Returns:
            True if transition succeeded, False if rejected.
        """
        if next_state == self._current_state:
            return True

        if not is_valid_transition(self._current_state, next_state):
            err_msg = (
                f"[{self.robot_id}] Invalid FSM transition: "
                f"Cannot move from {self._current_state.value} to {next_state.value} (Reason: '{reason}')"
            )
            if strict:
                raise ValueError(err_msg)
            return False

        old_state = self._current_state

        # Execute exit callbacks
        for cb in self._on_exit_callbacks[old_state]:
            cb(old_state, timestamp, reason)

        # Update state
        self._current_state = next_state
        self._state_enter_timestamp = timestamp
        self.history.append((timestamp, old_state, next_state, reason))

        # Execute enter callbacks
        for cb in self._on_enter_callbacks[next_state]:
            cb(next_state, timestamp, reason)

        return True
