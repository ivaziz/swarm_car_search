"""
Ant Colony Optimization (ACO) Task Allocation Engine with Softmax Action Selection.
"""

from typing import Dict, List, Optional, Tuple
import math
import random
from interfaces.slam_interface import Pose2D
from swarm.task_allocation.frontier_task import SearchTask, TaskStatus
from swarm.pheromone.pheromone_map import PheromoneMap
from .aco_config import ACOConfig
from .task_scorer import TaskScorer


class ACOEngine:
    """
    Decentralized task selection engine adapting Ant Colony Optimization for multi-robot search.
    Employs Boltzmann/Softmax probability distribution to ensure exploration diversity
    and prevent the entire swarm from collapsing onto a single frontier cluster.
    """

    def __init__(
        self,
        config: Optional[ACOConfig] = None,
        scorer: Optional[TaskScorer] = None,
    ) -> None:
        self.config = config or ACOConfig()
        self.scorer = scorer or TaskScorer(self.config)

    def _get_congestion_counts(
        self,
        tasks: List[SearchTask],
        peer_commitments: Optional[Dict[str, str]] = None,
    ) -> Dict[str, int]:
        """Calculates how many other robots have claimed or committed to each task."""
        counts = {t.task_id: 0 for t in tasks}
        if peer_commitments:
            for committed_task_id in peer_commitments.values():
                if committed_task_id in counts:
                    counts[committed_task_id] += 1
        return counts

    def score_tasks(
        self,
        tasks: List[SearchTask],
        robot_pose: Pose2D,
        pheromone_map: Optional[PheromoneMap] = None,
        peer_commitments: Optional[Dict[str, str]] = None,
    ) -> List[Tuple[SearchTask, float]]:
        """
        Computes attractiveness scores for all candidate tasks and returns them
        sorted in descending order.
        """
        congestion_counts = self._get_congestion_counts(tasks, peer_commitments)
        scored: List[Tuple[SearchTask, float]] = []

        for task in tasks:
            congestion = congestion_counts.get(task.task_id, 0)
            score = self.scorer.compute_score(
                task=task,
                robot_pose=robot_pose,
                pheromone_map=pheromone_map,
                congestion_count=congestion,
            )
            scored.append((task, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored

    def select_task(
        self,
        tasks: List[SearchTask],
        robot_pose: Pose2D,
        pheromone_map: Optional[PheromoneMap] = None,
        peer_commitments: Optional[Dict[str, str]] = None,
        rng: Optional[random.Random] = None,
    ) -> Optional[SearchTask]:
        """
        Selects a task using ACO Boltzmann exploration / exploitation strategy.

        Args:
            tasks: List of candidate tasks.
            robot_pose: Evaluating robot pose.
            pheromone_map: Optional global or local pheromone map.
            peer_commitments: Dict mapping peer_robot_id -> assigned_task_id.
            rng: Optional seeded random number generator for reproducibility.

        Returns:
            Selected SearchTask, or None if no valid candidate exists.
        """
        # Filter available / pending tasks
        available_tasks = [t for t in tasks if t.status == TaskStatus.PENDING]
        if not available_tasks:
            return None

        if len(available_tasks) == 1:
            return available_tasks[0]

        rand = rng or random

        # Deterministic top-1 choice bypasses all stochastic exploration
        if self.config.stochastic.deterministic:
            scored_tasks = self.score_tasks(
                tasks=available_tasks,
                robot_pose=robot_pose,
                pheromone_map=pheromone_map,
                peer_commitments=peer_commitments,
            )
            return scored_tasks[0][0] if scored_tasks else None

        # Epsilon-greedy exploration perturbation
        eps = self.config.stochastic.epsilon_greedy
        if eps > 0.0 and rand.random() < eps:
            return rand.choice(available_tasks)

        # Score candidates
        scored_tasks = self.score_tasks(
            tasks=available_tasks,
            robot_pose=robot_pose,
            pheromone_map=pheromone_map,
            peer_commitments=peer_commitments,
        )

        if not scored_tasks:
            return None

        # Low temperature exploitation limit
        temp = self.config.stochastic.temperature
        if temp <= 0.01:
            return scored_tasks[0][0]

        # Boltzmann / Softmax probabilistic selection with scale invariance
        scores = [s for _, s in scored_tasks]
        max_score = max(scores)

        if max_score <= 1e-9:
            return scored_tasks[0][0]

        # Normalized relative logits: (score / max_score - 1.0) / temp
        # Guaranteed to be in range [-1/temp, 0], preventing numerical underflow/overflow
        weights = [math.exp(((s / max_score) - 1.0) / temp) for s in scores]
        total_weight = sum(weights)

        if total_weight <= 0.0:
            return scored_tasks[0][0]

        probabilities = [w / total_weight for w in weights]

        # Cumulative probability sampling
        r = rand.random()
        cumulative = 0.0
        for idx, p in enumerate(probabilities):
            cumulative += p
            if r <= cumulative:
                return scored_tasks[idx][0]

        return scored_tasks[-1][0]
