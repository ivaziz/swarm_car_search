"""
Multi-Objective Objective Function Task Scorer for Ant Colony Optimization.
"""

from typing import Optional
import math
from interfaces.slam_interface import Pose2D
from swarm.task_allocation.frontier_task import SearchTask, TaskType
from swarm.pheromone.pheromone_map import PheromoneMap
from swarm.pheromone.evaporation_model import PheromoneLayerType
from .aco_config import ACOConfig


class TaskScorer:
    """
    Computes candidate frontier attractiveness score for a robot using:
        Score = (tau^alpha * eta^beta * P^gamma * G^delta) / (C^epsilon * Omega^zeta * K^theta)
    """

    def __init__(self, config: Optional[ACOConfig] = None) -> None:
        self.config = config or ACOConfig()

    def compute_score(
        self,
        task: SearchTask,
        robot_pose: Pose2D,
        pheromone_map: Optional[PheromoneMap] = None,
        congestion_count: int = 0,
    ) -> float:
        """
        Evaluates the attractiveness score of task from the perspective of robot_pose.

        Args:
            task: Candidate frontier search task.
            robot_pose: Current 2D pose of evaluating robot.
            pheromone_map: Optional live multi-layer pheromone map.
            congestion_count: Number of peer robots currently targeting this task.

        Returns:
            Strictly positive scalar attractiveness score.
        """
        w = self.config.weights

        # 1. Positive Pheromone (tau)
        if pheromone_map is not None:
            min_x, min_y, max_x, max_y = task.bounding_box
            tau = pheromone_map.get_box_aggregate(
                PheromoneLayerType.SUCCESS, min_x, min_y, max_x, max_y, mode="mean"
            )
        else:
            tau = task.pheromone_success
        tau = max(0.01, tau)

        # 2. Information Gain / Cluster Size (eta)
        eta = max(1.0, float(task.size))

        # 3. Detection Probability Prior (P)
        p_det = max(0.01, min(1.0, float(task.detection_probability)))

        # 4. Mission Urgency / Priority (G)
        task_type_factor = 1.0
        if getattr(task, "task_type", None) == TaskType.TARGET_CONFIRMATION:
            task_type_factor = 2.5
        elif getattr(task, "task_type", None) == TaskType.VEHICLE_INVESTIGATION:
            task_type_factor = 1.8
        priority = max(0.1, float(task.priority) * task_type_factor)

        # 5. Travel Cost / Distance (C)
        dx = task.centroid_x - robot_pose.x
        dy = task.centroid_y - robot_pose.y
        euclidean_dist = math.hypot(dx, dy)
        cost = max(self.config.min_distance_offset, euclidean_dist)

        # 6. Congestion Multiplier (Omega)
        omega = 1.0 + max(0, congestion_count)

        # 7. Avoidance / Negative Pheromone (K)
        if pheromone_map is not None:
            min_x, min_y, max_x, max_y = task.bounding_box
            k_avoid = pheromone_map.get_box_aggregate(
                PheromoneLayerType.AVOIDANCE, min_x, min_y, max_x, max_y, mode="mean"
            )
        else:
            k_avoid = task.pheromone_avoidance
        k_avoid = max(0.01, k_avoid)

        # Numerator: Exploitation & Information Potential
        numerator = (tau ** w.alpha) * (eta ** w.beta) * (p_det ** w.gamma) * (priority ** w.delta)

        # Denominator: Distance, Congestion, and Avoidance Penalties
        denominator = (cost ** w.epsilon) * (omega ** w.zeta) * (k_avoid ** w.theta)

        if denominator <= 0.0:
            return 0.0

        return numerator / denominator
