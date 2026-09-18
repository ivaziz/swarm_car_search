"""
Dynamic Reassignment and Task Recovery Manager.
Safely releases orphaned tasks, updates stigmergic avoidance around failed units,
and dispatches dynamic reallocations to prevent search coverage gaps.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any, TYPE_CHECKING

from interfaces.slam_interface import Pose2D
from swarm.states.robot_states import RobotState
from swarm.task_allocation.frontier_task import SearchTask, TaskStatus
from .failure_detector import RobotFailureEvent

if TYPE_CHECKING:
    from swarm.task_allocation.task_manager import TaskManager
    from swarm.core.swarm_state import SwarmState
    from swarm.core.robot_agent import RobotAgent
    from swarm.pheromone.pheromone_updater import PheromoneUpdater
    from swarm.pheromone.pheromone_map import PheromoneMap
    from swarm.task_allocation.allocation_strategy import AllocationStrategy


class RecoveryAction(str, Enum):
    """Specific recovery interventions executed during failure handling."""
    PEER_DEACTIVATED = "PEER_DEACTIVATED"
    TASK_RELEASED = "TASK_RELEASED"
    AVOIDANCE_DEPOSITED = "AVOIDANCE_DEPOSITED"
    TASK_REASSIGNED = "TASK_REASSIGNED"


@dataclass
class RecoveryResult:
    """Summary record of recovery actions taken for a failed robot."""
    failed_robot_id: str
    orphaned_task_id: Optional[str] = None
    reassigned_to_robot_id: Optional[str] = None
    actions_taken: List[RecoveryAction] = field(default_factory=list)
    timestamp: float = 0.0


class RecoveryManager:
    """
    Coordinates multi-robot recovery when an agent fails or disconnects:
    1. Revokes task ownership from failed agent and resets task to PENDING.
    2. Deactivates failed peer in all decentralized SwarmStates.
    3. Deposits strong avoidance pheromone around disabled unit's location.
    4. Immediately triggers ACO reallocation for available active agents.
    """

    def __init__(
        self,
        wreck_avoidance_amount: float = 12.0,
        wreck_avoidance_radius: float = 2.5,
    ) -> None:
        self.wreck_avoidance_amount = wreck_avoidance_amount
        self.wreck_avoidance_radius = wreck_avoidance_radius
        self.recovery_history: List[RecoveryResult] = []

    def handle_failure(
        self,
        event: RobotFailureEvent,
        task_manager: "TaskManager",
        swarm_states: Dict[str, "SwarmState"],
        pheromone_updater: Optional["PheromoneUpdater"] = None,
        robots: Optional[Dict[str, "RobotAgent"]] = None,
        allocation_strategy: Optional["AllocationStrategy"] = None,
        pheromone_map: Optional["PheromoneMap"] = None,
    ) -> RecoveryResult:
        """
        Executes end-to-end recovery pipeline for a detected robot failure.
        """
        result = RecoveryResult(failed_robot_id=event.robot_id, timestamp=event.timestamp)

        # 1. Deactivate peer across all decentralized SwarmStates
        for sstate in swarm_states.values():
            sstate.mark_peer_inactive(event.robot_id)
        result.actions_taken.append(RecoveryAction.PEER_DEACTIVATED)

        # 2. Identify and release any task held by the failed robot
        orphaned_task: Optional[SearchTask] = None
        if event.abandoned_task_id and event.abandoned_task_id in task_manager.tasks:
            orphaned_task = task_manager.tasks[event.abandoned_task_id]
        else:
            orphaned_task = task_manager.get_robot_assigned_task(event.robot_id)

        if orphaned_task is not None and orphaned_task.status in (TaskStatus.ASSIGNED, TaskStatus.IN_PROGRESS):
            result.orphaned_task_id = orphaned_task.task_id
            task_manager.release_task(orphaned_task.task_id, event.timestamp)
            result.actions_taken.append(RecoveryAction.TASK_RELEASED)

            # Synchronize task release in decentralized swarm states
            for sstate in swarm_states.values():
                if orphaned_task.task_id in sstate.active_tasks:
                    sstate.active_tasks[orphaned_task.task_id].mark_released(event.timestamp)

        # 3. Mark failed physical robot as stationary obstacle via Avoidance Pheromone
        if pheromone_updater is not None and event.last_known_pose is not None:
            pheromone_updater.deposit_avoidance(
                x=event.last_known_pose.x,
                y=event.last_known_pose.y,
                amount=self.wreck_avoidance_amount,
                radius=self.wreck_avoidance_radius,
            )
            result.actions_taken.append(RecoveryAction.AVOIDANCE_DEPOSITED)

        # 4. Immediate Dynamic Reassignment: dispatch orphaned task to an idle peer if available
        if (
            orphaned_task is not None
            and robots is not None
            and allocation_strategy is not None
        ):
            # Find eligible active robots
            for candidate_id, candidate_agent in robots.items():
                if candidate_id == event.robot_id:
                    continue

                if candidate_agent.current_state in (RobotState.IDLE, RobotState.REASSIGNING):
                    # Check if candidate can take this task
                    candidate_swarm_state = swarm_states.get(candidate_id)
                    chosen = allocation_strategy.allocate_task(
                        robot_id=candidate_id,
                        robot_pose=candidate_agent.pose,
                        available_tasks=[orphaned_task],
                        swarm_state=candidate_swarm_state,
                        pheromone_map=pheromone_map,
                    )
                    if chosen is not None:
                        assigned = candidate_agent.assign_task(chosen, event.timestamp)
                        if assigned:
                            task_manager.assign_task(chosen.task_id, candidate_id, event.timestamp)
                            result.reassigned_to_robot_id = candidate_id
                            result.actions_taken.append(RecoveryAction.TASK_REASSIGNED)
                            break

        self.recovery_history.append(result)
        return result
