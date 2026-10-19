"""
Comprehensive Swarm Mission Metrics Collector.
Evaluates Time-to-Detection (TTD), spatial exploration rate, multi-robot trajectory overlap,
battery energy expenditure, and fault recovery latency.
"""

from __future__ import annotations
from dataclasses import dataclass, field, asdict, fields
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
    # Multi-robot spatial footprint redundancy (legacy + disambiguated)
    overlap_ratio: float = 0.0
    unique_cells_visited: int = 0
    redundant_visits: int = 0
    inter_robot_overlap_ratio: Optional[float] = None
    shared_cells_count: Optional[int] = None
    shared_cells_ratio: Optional[float] = None
    # Temporal trajectory revisit metrics (Optional to distinguish missing historical logs from measured 0.0)
    total_cell_visits: Optional[int] = None
    revisit_events_count: Optional[int] = None
    temporal_revisit_rate: Optional[float] = None
    intra_robot_revisits: Optional[int] = None
    intra_robot_self_revisit_rate: Optional[float] = None
    per_robot_self_revisit_rate: Optional[Dict[str, float]] = None
    # Physical and task telemetry
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

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MissionMetrics":
        """Safely instantiates MissionMetrics from dictionary, supporting legacy JSON trial logs."""
        valid_fields = {f.name for f in fields(cls)}
        filtered = {k: v for k, v in data.items() if k in valid_fields}
        return cls(**filtered)


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

        # Multi-robot cell footprint visitation tracking: (bin_x, bin_y) -> Set[robot_id]
        self.cell_visits: Dict[Tuple[int, int], Set[str]] = {}

        # Temporal cell visitation & transition tracking
        self.last_robot_bin: Dict[str, Tuple[int, int]] = {}
        self.robot_entry_counts: Dict[str, int] = {}
        self.cell_entry_counts: Dict[Tuple[int, int], int] = {}
        self.robot_visited_cells: Dict[str, Set[Tuple[int, int]]] = {}
        self.last_robot_timestamps: Dict[str, float] = {}

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

            # Deduplicate repeated identical updates: only skip if BOTH timestamp and pose match exactly
            prev_ts = self.last_robot_timestamps.get(robot_id)
            if prev_ts is not None and timestamp == prev_ts and robot_id in self.last_poses:
                prev_p = self.last_poses[robot_id]
                if pose.x == prev_p.x and pose.y == prev_p.y and pose.theta == prev_p.theta:
                    continue
            self.last_robot_timestamps[robot_id] = timestamp

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

            # Spatial bin visitation with uniform math.floor discretization
            bx = math.floor(pose.x / self.spatial_bin_size)
            by = math.floor(pose.y / self.spatial_bin_size)
            bcoord = (int(bx), int(by))

            # Inter-robot footprint tracking
            if bcoord not in self.cell_visits:
                self.cell_visits[bcoord] = set()
            self.cell_visits[bcoord].add(robot_id)

            if robot_id not in self.robot_visited_cells:
                self.robot_visited_cells[robot_id] = set()
            self.robot_visited_cells[robot_id].add(bcoord)

            # Temporal transition tracking: entry into novel or revisited cell
            prev_bin = self.last_robot_bin.get(robot_id)
            if prev_bin is None or bcoord != prev_bin:
                self.robot_entry_counts[robot_id] = self.robot_entry_counts.get(robot_id, 0) + 1
                self.cell_entry_counts[bcoord] = self.cell_entry_counts.get(bcoord, 0) + 1
                self.last_robot_bin[robot_id] = bcoord

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

        # 4. Record exploration ratio timeline (prevent duplicate timestamp entries)
        avg_exp = status.get("average_exploration_ratio", 0.0)
        if not (self.coverage_timeline and self.coverage_timeline[-1][0] == timestamp):
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

        # 1. Multi-robot inter-robot footprint redundancy
        unique_cells = len(self.cell_visits)
        total_robot_visits = sum(len(rob_set) for rob_set in self.cell_visits.values())
        redundant_visits = max(0, total_robot_visits - unique_cells)
        overlap_ratio = (redundant_visits / total_robot_visits) if total_robot_visits > 0 else 0.0

        # Shared cells across multiple distinct robots (count and proportion)
        shared_cells = sum(1 for rob_set in self.cell_visits.values() if len(rob_set) >= 2)
        shared_cells_ratio = (shared_cells / unique_cells) if unique_cells > 0 else 0.0

        # 2. Swarm temporal revisit metrics (transition-based)
        total_cell_visits = sum(self.robot_entry_counts.values())
        revisit_events = max(0, total_cell_visits - unique_cells)
        temporal_revisit_rate = (revisit_events / total_cell_visits) if total_cell_visits > 0 else 0.0

        # Intra-robot self-revisits (same robot returning to previously entered cell)
        per_robot_self_revisit_rate: Dict[str, float] = {}
        for rid, entry_cnt in self.robot_entry_counts.items():
            u_cells = len(self.robot_visited_cells.get(rid, set()))
            self_revs = max(0, entry_cnt - u_cells)
            per_robot_self_revisit_rate[rid] = (self_revs / entry_cnt) if entry_cnt > 0 else 0.0

        intra_robot_revisits = sum(
            max(0, self.robot_entry_counts.get(rid, 0) - len(self.robot_visited_cells.get(rid, set())))
            for rid in self.robot_visited_cells
        )
        intra_robot_self_revisit_rate = (intra_robot_revisits / total_cell_visits) if total_cell_visits > 0 else 0.0

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
            inter_robot_overlap_ratio=overlap_ratio,
            shared_cells_count=shared_cells,
            shared_cells_ratio=shared_cells_ratio,
            total_cell_visits=total_cell_visits,
            revisit_events_count=revisit_events,
            temporal_revisit_rate=temporal_revisit_rate,
            intra_robot_revisits=intra_robot_revisits,
            intra_robot_self_revisit_rate=intra_robot_self_revisit_rate,
            per_robot_self_revisit_rate=per_robot_self_revisit_rate,
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
