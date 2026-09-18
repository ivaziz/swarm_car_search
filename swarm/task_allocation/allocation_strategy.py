"""
Task Allocation Strategy Interfaces and Implementations (ACO, Nearest Frontier, Random).
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, List, Optional, Dict, Any
import math
import random
from interfaces.slam_interface import Pose2D
from swarm.task_allocation.frontier_task import SearchTask, TaskStatus
from swarm.pheromone.pheromone_map import PheromoneMap

if TYPE_CHECKING:
    from swarm.core.swarm_state import SwarmState
    from swarm.aco.aco_engine import ACOEngine


class ITaskAllocationStrategy(ABC):
    """Abstract interface for task allocation strategies."""

    @abstractmethod
    def allocate_task(
        self,
        robot_id: str,
        robot_pose: Pose2D,
        available_tasks: List[SearchTask],
        swarm_state: Optional[SwarmState] = None,
        pheromone_map: Optional[PheromoneMap] = None,
    ) -> Optional[SearchTask]:
        """Selects a candidate task for the specified robot."""
        pass


class ACOAllocationStrategy(ITaskAllocationStrategy):
    """
    Ant Colony Optimization (ACO) allocation strategy with pheromone stigmergy
    and peer congestion avoidance.
    """

    def __init__(self, engine: Optional[ACOEngine] = None) -> None:
        if engine is None:
            from swarm.aco.aco_engine import ACOEngine
            from swarm.aco.aco_config import ACOConfig
            self.engine = ACOEngine(ACOConfig())
        else:
            self.engine = engine

    def allocate_task(
        self,
        robot_id: str,
        robot_pose: Pose2D,
        available_tasks: List[SearchTask],
        swarm_state: Optional[SwarmState] = None,
        pheromone_map: Optional[PheromoneMap] = None,
    ) -> Optional[SearchTask]:
        """Evaluates tasks using ACO multi-objective scoring and selects via Softmax."""
        peer_commitments: Dict[str, str] = {}
        if swarm_state is not None:
            for peer_id, peer in swarm_state.peers.items():
                if peer.is_active and peer.assigned_task_id:
                    peer_commitments[peer_id] = peer.assigned_task_id

        return self.engine.select_task(
            tasks=available_tasks,
            robot_pose=robot_pose,
            pheromone_map=pheromone_map,
            peer_commitments=peer_commitments,
        )


class NearestFrontierStrategy(ITaskAllocationStrategy):
    """
    Baseline greedy strategy: selects the closest pending task by Euclidean distance.
    """

    def allocate_task(
        self,
        robot_id: str,
        robot_pose: Pose2D,
        available_tasks: List[SearchTask],
        swarm_state: Optional[SwarmState] = None,
        pheromone_map: Optional[PheromoneMap] = None,
    ) -> Optional[SearchTask]:
        pending = [t for t in available_tasks if t.status == TaskStatus.PENDING]
        if not pending:
            return None

        best_task: Optional[SearchTask] = None
        min_dist = float("inf")

        for task in pending:
            dist = math.hypot(task.centroid_x - robot_pose.x, task.centroid_y - robot_pose.y)
            if dist < min_dist:
                min_dist = dist
                best_task = task

        return best_task


class RandomAllocationStrategy(ITaskAllocationStrategy):
    """
    Baseline benchmark strategy: selects uniformly at random from pending tasks.
    """

    def __init__(self, seed: Optional[int] = None) -> None:
        self.rng = random.Random(seed)

    def allocate_task(
        self,
        robot_id: str,
        robot_pose: Pose2D,
        available_tasks: List[SearchTask],
        swarm_state: Optional[SwarmState] = None,
        pheromone_map: Optional[PheromoneMap] = None,
    ) -> Optional[SearchTask]:
        pending = [t for t in available_tasks if t.status == TaskStatus.PENDING]
        if not pending:
            return None
        return self.rng.choice(pending)
