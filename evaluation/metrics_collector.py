"""
Comprehensive Swarm Mission Metrics Collector.
Evaluates Time-to-Detection (TTD), spatial exploration rate, multi-robot trajectory overlap,
battery energy expenditure, and fault recovery latency.
"""

from __future__ import annotations
from dataclasses import dataclass, field, asdict
import json
import math
import os
from typing import Dict, List, Optional, Set, Tuple, Any

from interfaces.slam_interface import Pose2D
from simulation.environment import UrbanEnvironment
from swarm.core.swarm_coordinator import SwarmCoordinator
from swarm.task_allocation.frontier_task import TaskStatus


@dataclass
class MissionMetrics:
    """Consolidated quantitative performance telemetry for an evaluation trial."""
    strategy_name: str
    trial_seed: int
    total_simulation_time: float
    target_found: bool = False
    time_to_detection: Optional[float] = None
    final_exploration_ratio: float = 0.0
    exploration_rate_m2_per_sec: float = 0.0
    overlap_ratio: float = 0.0
    unique_cells_visited: int = 0
    redundant_visits: int = 0
    total_distance_traveled: float = 0.0
    per_robot_distance: Dict[str, float] = field(default_factory=dict)
    total_energy_consumed: float = 0.0
    tasks_completed: int = 0
    tasks_created: int = 0
    reassignment_events: int = 0
    failed_robots_count: int = 0
    recovered_tasks_count: int = 0
    failure_recovery_latency: Optional[float] = None
    failure_recovery_success: bool = False
    discovering_robot_id: Optional[str] = None
    confirmed_target_location: Optional[Tuple[float, float]] = None
    target_confirmation_confidence: Optional[float] = None
    candidate_detections_count: int = 0
    false_positives_count: int = 0
    target_confirmations_count: int = 0
    coverage_timeline: List[Tuple[float, float]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Converts metrics dataclass to serializable dictionary."""
        return asdict(self)


class MetricsCollector:
    """
    Accumulates runtime telemetry across a simulation trial and evaluates
    quantitative benchmarks comparing multi-robot coordination algorithms.
    """

    def __init__(
        self,
        environment: UrbanEnvironment,
        strategy_name: str = "ACO",
        trial_seed: int = 42,
        spatial_bin_size: float = 1.0,
    ) -> None:
        self.env = environment
        self.strategy_name = strategy_name
        self.trial_seed = trial_seed
        self.spatial_bin_size = spatial_bin_size

        # Multi-robot cell visitation tracking: (bin_x, bin_y) -> Set[robot_id]
        self.cell_visits: Dict[Tuple[int, int], Set[str]] = {}

        # Trajectory distance tracking
        self.last_poses: Dict[str, Pose2D] = {}
        self.distance_traveled: Dict[str, float] = {}

        # Battery initial state tracking
        self.initial_battery: Dict[str, float] = {}
        self.final_battery: Dict[str, float] = {}

        # Target detection tracking
        self.target_found: bool = False
        self.time_to_detection: Optional[float] = None

        # Failure recovery tracking
        self.failure_injection_time: Optional[float] = None
        self.recovery_time: Optional[float] = None
        self.initial_failed_count: int = 0

        # Timeline
        self.coverage_timeline: List[Tuple[float, float]] = []
        self.last_timestamp: float = 0.0

    def step(self, coordinator: SwarmCoordinator, timestamp: float) -> None:
        """
        Gathers telemetry for one simulation tick.
        """
        self.last_timestamp = timestamp
        status = coordinator.get_swarm_status(timestamp)

        # 1. Update distance traveled & cell visitation
        for robot_id, agent in coordinator.robots.items():
            pose = agent.pose

            # Record initial battery
            if robot_id not in self.initial_battery:
                self.initial_battery[robot_id] = agent.battery_level
            self.final_battery[robot_id] = agent.battery_level

            # Distance integration
            if robot_id in self.last_poses:
                prev = self.last_poses[robot_id]
                d = math.hypot(pose.x - prev.x, pose.y - prev.y)
                self.distance_traveled[robot_id] = self.distance_traveled.get(robot_id, 0.0) + d
            else:
                self.distance_traveled[robot_id] = 0.0
            self.last_poses[robot_id] = pose

            # Spatial bin visitation
            bx = int(pose.x / self.spatial_bin_size)
            by = int(pose.y / self.spatial_bin_size)
            bcoord = (bx, by)
            if bcoord not in self.cell_visits:
                self.cell_visits[bcoord] = set()
            self.cell_visits[bcoord].add(robot_id)

        # 2. Track Target Detection Event
        if coordinator.target_found and not self.target_found:
            self.target_found = True
            self.time_to_detection = timestamp

        # 3. Track Failure & Reassignment Latency
        current_failed = status.get("failed_robots_count", 0)
        current_recovered = status.get("recovered_tasks_count", 0)
        if current_failed > self.initial_failed_count and self.failure_injection_time is None:
            self.failure_injection_time = timestamp
        if self.failure_injection_time is not None and current_recovered > 0 and self.recovery_time is None:
            self.recovery_time = timestamp

        # 4. Record exploration ratio timeline
        avg_exp = status.get("average_exploration_ratio", 0.0)
        self.coverage_timeline.append((timestamp, avg_exp))

    def record_failure_injection(self, timestamp: float) -> None:
        """Explicit hook when a failure is injected externally."""
        if self.failure_injection_time is None:
            self.failure_injection_time = timestamp

    def compute_metrics(self, coordinator: SwarmCoordinator) -> MissionMetrics:
        """
        Compiles final mission statistics and computes mathematical overlap and exploration rates.
        """
        status = coordinator.get_swarm_status(self.last_timestamp)

        # Calculate spatial overlap ratio
        unique_cells = len(self.cell_visits)
        total_visits = sum(len(rob_set) for rob_set in self.cell_visits.values())
        redundant_visits = max(0, total_visits - unique_cells)
        overlap_ratio = (redundant_visits / total_visits) if total_visits > 0 else 0.0

        # Calculate total distance and energy
        total_dist = sum(self.distance_traveled.values())
        total_energy = sum(
            max(0.0, self.initial_battery.get(rid, 1.0) - self.final_battery.get(rid, 1.0))
            for rid in coordinator.robots.keys()
        )

        # Calculate exploration speed (m^2 / s)
        total_area = self.env.width * self.env.height
        final_coverage = status.get("average_exploration_ratio", 0.0)
        explored_area = total_area * final_coverage
        exp_rate = (explored_area / self.last_timestamp) if self.last_timestamp > 0 else 0.0

        # Failure recovery latency
        recovery_latency = None
        if self.failure_injection_time is not None and self.recovery_time is not None:
            recovery_latency = max(0.0, self.recovery_time - self.failure_injection_time)

        # Target found status and details
        target_found = self.target_found or status.get("target_found", False)
        ttd = self.time_to_detection

        # Aggregate perception detection stats across robots
        cand_detections = sum(getattr(agent, "total_detections_count", 0) for agent in coordinator.robots.values())
        false_positives = sum(getattr(agent, "total_false_positives", 0) for agent in coordinator.robots.values())
        target_confirmations = sum(getattr(agent, "total_target_confirmations", 0) for agent in coordinator.robots.values())

        failed_count = status.get("failed_robots_count", 0)
        recovered_count = status.get("recovered_tasks_count", 0)
        recovery_success = (recovered_count > 0) if failed_count > 0 else True

        discovering_robot = getattr(coordinator, "discovering_robot_id", None)
        target_conf = getattr(coordinator, "target_confirmation_confidence", None)
        target_loc = coordinator.confirmed_target_location

        return MissionMetrics(
            strategy_name=self.strategy_name,
            trial_seed=self.trial_seed,
            total_simulation_time=self.last_timestamp,
            target_found=target_found,
            time_to_detection=ttd,
            final_exploration_ratio=final_coverage,
            exploration_rate_m2_per_sec=exp_rate,
            overlap_ratio=overlap_ratio,
            unique_cells_visited=unique_cells,
            redundant_visits=redundant_visits,
            total_distance_traveled=total_dist,
            per_robot_distance=dict(self.distance_traveled),
            total_energy_consumed=total_energy,
            tasks_completed=status.get("completed_tasks_count", 0),
            tasks_created=len(coordinator.task_manager.tasks),
            reassignment_events=status.get("reassignment_events_count", 0),
            failed_robots_count=failed_count,
            recovered_tasks_count=recovered_count,
            failure_recovery_latency=recovery_latency,
            failure_recovery_success=recovery_success,
            discovering_robot_id=discovering_robot,
            confirmed_target_location=target_loc,
            target_confirmation_confidence=target_conf,
            candidate_detections_count=cand_detections,
            false_positives_count=false_positives,
            target_confirmations_count=target_confirmations,
            coverage_timeline=list(self.coverage_timeline),
        )

    def save_json(self, coordinator: SwarmCoordinator, filepath: str) -> None:
        """Saves compiled trial metrics to JSON file."""
        metrics = self.compute_metrics(coordinator)
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(metrics.to_dict(), f, indent=2)
