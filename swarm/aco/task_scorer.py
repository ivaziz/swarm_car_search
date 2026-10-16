"""
Multi-Objective Objective Function Task Scorer for Ant Colony Optimization.
Combines Information Gain, Dynamic Target Probability, Historical Memory,
Distance Cost, Congestion, and Collision Risk with explainable factor breakdowns.
"""

from typing import Optional, Dict, Any, Tuple
import math
from interfaces.slam_interface import Pose2D
from swarm.task_allocation.frontier_task import SearchTask, TaskType
from swarm.pheromone.pheromone_map import PheromoneMap
from swarm.pheromone.evaporation_model import PheromoneLayerType
from swarm.pheromone.target_probability_map import TargetProbabilityMap
from swarm.task_allocation.search_history import SearchSpatialMemory
from .aco_config import ACOConfig


class TaskScorer:
    """
    Computes candidate frontier attractiveness score for a robot using:
        Score(frontier, robot) =
            BaseMultiplier * (
                w_ig * Norm(InfoGain)
                + w_tp * Norm(TargetPrior)
                + w_he * Norm(HistoricalValue)
                + w_fs * Norm(ExplorationValue)
                - w_dist * Norm(DistanceCost)
                - w_cong * Norm(CongestionCost)
                - w_col * Norm(CollisionRisk)
                - w_red * Norm(RedundantPenalty)
            ) * BatteryHealthFactor
    """

    def __init__(self, config: Optional[ACOConfig] = None) -> None:
        self.config = config or ACOConfig()

    def explain_score(
        self,
        task: SearchTask,
        robot_pose: Pose2D,
        pheromone_map: Optional[PheromoneMap] = None,
        congestion_count: int = 0,
        target_probability_map: Optional[TargetProbabilityMap] = None,
        search_memory: Optional[SearchSpatialMemory] = None,
        battery_level: float = 1.0,
    ) -> Dict[str, Any]:
        """
        Calculates normalized scoring factors and returns a comprehensive explainable breakdown.
        """
        w = self.config.weights

        # 1. Information Gain (Normalized Frontier Size & Unknown Potential) [0.0, 1.0]
        max_fsize = getattr(self.config, "max_frontier_size", 30.0)
        norm_info_gain = min(1.0, max(0.05, float(task.size) / max_fsize))

        # 2. Dynamic Target Prior Probability [0.01, 0.99]
        if target_probability_map is not None:
            min_x, min_y, max_x, max_y = task.bounding_box
            p_target = target_probability_map.get_box_probability(min_x, min_y, max_x, max_y, mode="max")
        else:
            p_target = max(0.01, min(1.0, float(task.detection_probability)))
        norm_target_prior = p_target

        # 3. Historical Value (from Spatial Search Memory) [0.0, 1.0]
        if search_memory is not None:
            norm_hist_value = search_memory.get_historical_value(task.centroid_x, task.centroid_y)
            norm_redundant_penalty = search_memory.get_redundancy_penalty(task.centroid_x, task.centroid_y)
        else:
            norm_hist_value = 0.80
            norm_redundant_penalty = 0.0

        # 4. Positive Pheromone / Exploration Potential [0.0, 1.0]
        if pheromone_map is not None:
            min_x, min_y, max_x, max_y = task.bounding_box
            tau = pheromone_map.get_box_aggregate(
                PheromoneLayerType.SUCCESS, min_x, min_y, max_x, max_y, mode="mean"
            )
            k_avoid = pheromone_map.get_box_aggregate(
                PheromoneLayerType.AVOIDANCE, min_x, min_y, max_x, max_y, mode="mean"
            )
        else:
            tau = task.pheromone_success
            k_avoid = task.pheromone_avoidance

        norm_explor_value = min(1.0, max(0.05, tau / 20.0))
        norm_collision_risk = min(1.0, max(0.0, k_avoid / 15.0))

        # 5. Travel Cost / Distance [0.0, 1.0]
        dx = task.centroid_x - robot_pose.x
        dy = task.centroid_y - robot_pose.y
        euclidean_dist = math.hypot(dx, dy)
        max_dist = getattr(self.config, "max_normalization_distance", 70.0)
        norm_dist_cost = min(1.0, max(0.01, euclidean_dist / max_dist))

        # 6. Congestion Penalty [0.0, 1.0]
        norm_congestion = min(1.0, float(max(0, congestion_count)) / 3.0)

        # 7. Mission Urgency Multiplier
        urgency_multiplier = 1.0
        if getattr(task, "task_type", None) == TaskType.TARGET_CONFIRMATION:
            urgency_multiplier = 3.5
        elif getattr(task, "task_type", None) == TaskType.VEHICLE_INVESTIGATION:
            urgency_multiplier = 2.2
        urgency = max(0.1, float(task.priority) * urgency_multiplier)

        # 8. Normalized Multi-Factor Synthesis
        positive_components = (
            w.w_information_gain * norm_info_gain
            + w.w_target_prior * norm_target_prior
            + w.w_historical_value * norm_hist_value
            + w.w_exploration_value * norm_explor_value
        )

        negative_components = (
            w.w_distance_cost * norm_dist_cost
            + w.w_congestion_cost * norm_congestion
            + w.w_collision_risk * norm_collision_risk
            + w.w_redundant_penalty * norm_redundant_penalty
        )

        # Net normalized value bounded > 0.05
        net_utility = max(0.05, 1.0 + positive_components - negative_components)

        # Classical pheromone exponents integration for backwards-compatible ACO convergence
        tau_val = max(0.01, tau)
        cost_val = max(self.config.min_distance_offset, euclidean_dist)
        classical_ratio = ((tau_val ** (w.alpha * 0.5)) * (urgency ** w.delta)) / (cost_val ** (w.epsilon * 0.5))

        # Battery / health modulation
        battery_factor = max(0.2, min(1.0, 0.4 + 0.6 * battery_level))

        final_score = net_utility * classical_ratio * battery_factor

        return {
            "task_id": task.task_id,
            "information_gain": round(norm_info_gain, 3),
            "target_prior": round(norm_target_prior, 3),
            "historical_value": round(norm_hist_value, 3),
            "exploration_value": round(norm_explor_value, 3),
            "distance_cost": round(norm_dist_cost, 3),
            "congestion_cost": round(norm_congestion, 3),
            "collision_risk": round(norm_collision_risk, 3),
            "redundancy_penalty": round(norm_redundant_penalty, 3),
            "battery_factor": round(battery_factor, 3),
            "urgency_multiplier": round(urgency_multiplier, 3),
            "final_score": round(max(0.001, final_score), 4),
        }

    def compute_score(
        self,
        task: SearchTask,
        robot_pose: Pose2D,
        pheromone_map: Optional[PheromoneMap] = None,
        congestion_count: int = 0,
        target_probability_map: Optional[TargetProbabilityMap] = None,
        search_memory: Optional[SearchSpatialMemory] = None,
        battery_level: float = 1.0,
    ) -> float:
        """
        Evaluates the attractiveness score of task from the perspective of robot_pose.
        """
        breakdown = self.explain_score(
            task=task,
            robot_pose=robot_pose,
            pheromone_map=pheromone_map,
            congestion_count=congestion_count,
            target_probability_map=target_probability_map,
            search_memory=search_memory,
            battery_level=battery_level,
        )
        return breakdown["final_score"]
