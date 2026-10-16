"""
Decentralized Swarm Coordinator Orchestrating Multi-Robot Exploration and Dynamic Task Allocation.
"""

from enum import Enum
from typing import Dict, List, Optional, Any, Tuple, Set
import math
from interfaces.slam_interface import Pose2D
from simulation.environment import UrbanEnvironment
from swarm.states.robot_states import RobotState
from swarm.task_allocation.frontier_task import SearchTask, TaskStatus, TaskType
from swarm.task_allocation.task_manager import TaskManager
from swarm.task_allocation.allocation_strategy import (
    ITaskAllocationStrategy,
    ACOAllocationStrategy,
)
from swarm.pheromone.pheromone_map import PheromoneMap
from swarm.pheromone.pheromone_updater import PheromoneUpdater
from swarm.pheromone.target_probability_map import TargetProbabilityMap
from swarm.task_allocation.search_history import SearchSpatialMemory
from swarm.failure_handling.failure_detector import (
    FailureDetector,
    FailureDetectorConfig,
    RobotFailureEvent,
    FailureType,
)
from swarm.failure_handling.recovery_manager import (
    RecoveryManager,
    RecoveryResult,
    RecoveryAction,
)
from .robot_agent import RobotAgent
from .swarm_state import SwarmState


class SwarmMissionState(str, Enum):
    """Macro state of the autonomous swarm mission."""
    SEARCHING = "SEARCHING"
    INVESTIGATING = "INVESTIGATING"
    VERIFYING = "VERIFYING"
    TARGET_FOUND = "TARGET_FOUND"


class SwarmCoordinator:
    """
    Coordinates multi-robot swarm behavior:
    - Integrates decentralized robot agents.
    - Synchronizes SLAM frontier ingestion into TaskManager.
    - Resolves dynamic task assignments via ACO with Bayesian target beliefs and spatial memory.
    - Disperses avoidance pheromones to prevent multi-robot congestion.
    - Exchanges peer state heartbeats.
    - Continuously monitors robot health and dynamically reassigns orphaned tasks on failure.
    """

    def __init__(
        self,
        task_manager: Optional[TaskManager] = None,
        allocation_strategy: Optional[ITaskAllocationStrategy] = None,
        pheromone_map: Optional[PheromoneMap] = None,
        pheromone_updater: Optional[PheromoneUpdater] = None,
        target_probability_map: Optional[TargetProbabilityMap] = None,
        search_memory: Optional[SearchSpatialMemory] = None,
        environment: Optional[UrbanEnvironment] = None,
        failure_config: Optional[FailureDetectorConfig] = None,
        failure_detector: Optional[FailureDetector] = None,
        recovery_manager: Optional[RecoveryManager] = None,
    ) -> None:
        self.env = environment
        self.task_manager = task_manager or TaskManager()
        self.allocation_strategy = allocation_strategy or ACOAllocationStrategy()

        # Shared stigmergic environment representation
        width = environment.width if environment else 100.0
        height = environment.height if environment else 100.0
        res = environment.resolution if environment else 0.5

        self.pheromone_map = pheromone_map or PheromoneMap(width=width, height=height, resolution=res)
        self.pheromone_updater = pheromone_updater or PheromoneUpdater(self.pheromone_map)

        # Dynamic Bayesian belief grid and search history memory
        self.target_probability_map = target_probability_map or TargetProbabilityMap(
            width=width, height=height, resolution=res, initial_prior=0.50
        )
        self.target_probability_map.initialize_from_environment(self.env)
        self.search_memory = search_memory or SearchSpatialMemory(
            width=width, height=height, bin_size=2.0
        )

        # Failure detection and dynamic recovery
        self.failure_detector = failure_detector or FailureDetector(failure_config)
        self.recovery_manager = recovery_manager or RecoveryManager()

        # Swarm registry: robot_id -> RobotAgent
        self.robots: Dict[str, RobotAgent] = {}
        # Per-robot decentralized state perspective
        self.robot_swarm_states: Dict[str, SwarmState] = {}

        # Swarm macro mission state
        self.mission_state = SwarmMissionState.SEARCHING

        # Aggregate telemetry metrics
        self.completed_tasks_count = 0
        self._completed_task_ids: Set[str] = set()
        self.reassignment_events_count = 0
        self.failed_robots_count = 0
        self.recovered_tasks_count = 0
        self.target_found = False
        self.confirmed_target_location: Optional[Tuple[float, float]] = None
        self.confirmed_target_plate: Optional[str] = None

        # Comprehensive Target Identification Telemetry & Stage Breakdown
        self.target_discovery_time: Optional[float] = None
        self.discovering_robot_id: Optional[str] = None
        self.target_confirmation_confidence: Optional[float] = None
        self.confidence_stage_breakdown: Dict[str, float] = {}
        self.target_distance_traveled: Optional[float] = None
        self.total_vehicle_detections_count: int = 0
        self.total_target_confirmations_count: int = 0
        self.total_false_positives_count: int = 0


    def register_robot(self, agent: RobotAgent) -> None:
        """Enrolls an autonomous robot agent into the swarm coordinator."""
        self.robots[agent.robot_id] = agent
        self.robot_swarm_states[agent.robot_id] = SwarmState(local_robot_id=agent.robot_id)
        # Register in failure detector
        self.failure_detector.record_telemetry(
            robot_id=agent.robot_id,
            timestamp=0.0,
            pose=agent.pose,
            battery_level=agent.battery_level,
            state=agent.current_state,
            assigned_task_id=agent.current_task.task_id if agent.current_task else None,
        )

    def simulate_robot_failure(
        self,
        robot_id: str,
        timestamp: float,
        reason: str = "MANUAL_SIMULATION",
    ) -> Optional[RobotFailureEvent]:
        """
        Injects failure into a specific robot and immediately triggers dynamic recovery.
        """
        if robot_id in self.robots:
            agent = self.robots[robot_id]
            agent.inject_failure(timestamp=timestamp, reason=reason)
            self.failure_detector.record_telemetry(
                robot_id=robot_id,
                timestamp=timestamp,
                pose=agent.pose,
                battery_level=agent.battery_level,
                state=RobotState.FAILED,
                assigned_task_id=agent.current_task.task_id if agent.current_task else None,
            )
            failures = self.failure_detector.check_failures(timestamp)
            for event in failures:
                rec_result = self.recovery_manager.handle_failure(
                    event=event,
                    task_manager=self.task_manager,
                    swarm_states=self.robot_swarm_states,
                    pheromone_updater=self.pheromone_updater,
                    robots=self.robots,
                    allocation_strategy=self.allocation_strategy,
                    pheromone_map=self.pheromone_map,
                )
                self.failed_robots_count += 1
                if rec_result.orphaned_task_id is not None:
                    self.recovered_tasks_count += 1
                    self.reassignment_events_count += 1
                return event
        return None

    def step_coordination(self, timestamp: float, dt: float) -> Dict[str, Any]:
        """
        Executes one synchronized coordination iteration:
        1. Propagate robot physics, SLAM, and spatial search memory.
        2. Ingest peer heartbeats and evaluate failure detection & dynamic recovery.
        3. Ingest newly observed frontier clusters into task pool with Bayesian prior.
        4. Dynamically allocate pending tasks via Intelligent ACO.
        5. Evaluate multi-stage perception & multi-frame target confirmation.
        6. Stigmergic pheromone deposits & discrete evaporation.
        7. Prune stale tasks and compile telemetry metrics.
        """
        # 1. Physics & SLAM propagation & Spatial Memory update
        for agent in self.robots.values():
            agent.step(timestamp=timestamp, dt=dt, env=self.env)

            if agent.is_alive and not self.failure_detector.is_failed(agent.robot_id):
                # Record spatial visitation memory
                self.search_memory.record_visit(
                    robot_id=agent.robot_id,
                    x=agent.pose.x,
                    y=agent.pose.y,
                    timestamp=timestamp,
                )
                # Bayesian negative evidence update for surveyed empty regions
                self.target_probability_map.deposit_negative_evidence(
                    x=agent.pose.x,
                    y=agent.pose.y,
                    decay_factor=0.08 * dt,
                    radius=2.5,
                )

        # 1.5. Inter-robot collision avoidance & motion de-confliction
        alive_robots = [a for a in self.robots.values() if a.is_alive and not self.failure_detector.is_failed(a.robot_id)]
        for i in range(len(alive_robots)):
            for j in range(i + 1, len(alive_robots)):
                r1 = alive_robots[i]
                r2 = alive_robots[j]
                dx = r2.pose.x - r1.pose.x
                dy = r2.pose.y - r1.pose.y
                dist = math.hypot(dx, dy)
                if dist < 2.2 and dist > 1e-3:
                    repel_x = -(dx / dist) * 0.15
                    repel_y = -(dy / dist) * 0.15
                    if hasattr(r1, "sim_robot") and r1.sim_robot is not None:
                        nx1 = max(1.0, min(self.env.width - 1.0, r1.sim_robot.pose.x + repel_x * dt))
                        ny1 = max(1.0, min(self.env.height - 1.0, r1.sim_robot.pose.y + repel_y * dt))
                        r1.sim_robot.pose = Pose2D(nx1, ny1, r1.sim_robot.pose.theta)
                    if hasattr(r2, "sim_robot") and r2.sim_robot is not None:
                        nx2 = max(1.0, min(self.env.width - 1.0, r2.sim_robot.pose.x - repel_x * dt))
                        ny2 = max(1.0, min(self.env.height - 1.0, r2.sim_robot.pose.y - repel_y * dt))
                        r2.sim_robot.pose = Pose2D(nx2, ny2, r2.sim_robot.pose.theta)

        # 2. Inter-robot communication / heartbeats exchange
        for agent in self.robots.values():
            hb = agent.create_heartbeat(timestamp)
            if hb is not None:
                sender = hb.sender_id
                payload = hb.payload
                pose_dict = payload["pose"]
                pose = Pose2D(x=pose_dict["x"], y=pose_dict["y"], theta=pose_dict["theta"])
                state = RobotState(payload["state"])
                battery = payload.get("battery_level", 1.0)
                task_id = payload.get("current_task_id")

                self.failure_detector.record_telemetry(
                    robot_id=sender,
                    timestamp=timestamp,
                    pose=pose,
                    battery_level=battery,
                    state=state,
                    assigned_task_id=task_id,
                )

                for recipient_id, sstate in self.robot_swarm_states.items():
                    if recipient_id != sender:
                        sstate.update_or_add_peer(
                            robot_id=sender,
                            state=state,
                            pose=pose,
                            timestamp=timestamp,
                            battery_level=battery,
                            assigned_task_id=task_id,
                        )

        # 2.5. Failure detection and dynamic recovery
        new_failures = self.failure_detector.check_failures(timestamp)
        for fail_event in new_failures:
            rec_result = self.recovery_manager.handle_failure(
                event=fail_event,
                task_manager=self.task_manager,
                swarm_states=self.robot_swarm_states,
                pheromone_updater=self.pheromone_updater,
                robots=self.robots,
                allocation_strategy=self.allocation_strategy,
                pheromone_map=self.pheromone_map,
            )
            self.failed_robots_count += 1
            if rec_result.orphaned_task_id is not None:
                self.recovered_tasks_count += 1
                self.reassignment_events_count += 1

        # Check peer timeouts across decentralized swarm states
        for sstate in self.robot_swarm_states.values():
            sstate.check_peer_timeouts(
                current_time=timestamp,
                timeout_sec=self.failure_detector.config.heartbeat_timeout_sec,
            )

        # 3. Frontier ingestion from active robots' SLAM maps
        for agent in self.robots.values():
            if self.failure_detector.is_failed(agent.robot_id):
                continue
            if agent.last_slam_state and agent.last_slam_state.frontiers:
                for frontier in agent.last_slam_state.frontiers:
                    prior = 0.5
                    if self.target_probability_map is not None:
                        prior = self.target_probability_map.get_probability(frontier.centroid_x, frontier.centroid_y)
                    elif self.env is not None:
                        prior = self.env.get_zone_prior(frontier.centroid_x, frontier.centroid_y)

                    self.task_manager.ingest_frontiers(
                        frontiers=[frontier],
                        timestamp=timestamp,
                        detection_probability=prior,
                    )

        # 3.5. Candidate vehicle detection & Bayesian Target Belief Update
        for agent in self.robots.values():
            if self.failure_detector.is_failed(agent.robot_id):
                continue
            if agent.latest_detected_candidate is not None:
                cand = agent.latest_detected_candidate
                agent.latest_detected_candidate = None
                self.total_vehicle_detections_count += 1
                cx, cy = cand.estimated_global_position
                p_target = getattr(cand, "target_probability", 0.75)
                v_id = getattr(cand, "detected_vehicle_id", None)

                # Positive Bayesian belief update & spatial memory record
                self.target_probability_map.deposit_positive_evidence(
                    x=cx,
                    y=cy,
                    confidence=p_target,
                    radius=5.0,
                )
                self.search_memory.record_sighting(
                    x=cx,
                    y=cy,
                    is_candidate=True,
                )

                # Update macro mission state
                if not self.target_found:
                    self.mission_state = SwarmMissionState.INVESTIGATING

                # Deposit target recruitment pheromone
                self.pheromone_updater.deposit_success(
                    x=cx,
                    y=cy,
                    amount=25.0 * p_target,
                    radius=5.0,
                )

                # Create or escalate target investigation/confirmation task
                task_type = TaskType.TARGET_CONFIRMATION if p_target >= 0.70 else TaskType.VEHICLE_INVESTIGATION
                self.task_manager.create_or_update_target_task(
                    x=cx,
                    y=cy,
                    timestamp=timestamp,
                    target_probability=p_target,
                    priority=2.5 + 3.5 * p_target,
                    vehicle_id=v_id,
                    task_type=task_type,
                )

        # 4. Dynamic Task Allocation for active idle, reassigning, or preemptible robots
        for agent in self.robots.values():
            if self.failure_detector.is_failed(agent.robot_id):
                continue
            should_allocate = agent.current_state in (RobotState.IDLE, RobotState.REASSIGNING)
            if (
                not should_allocate
                and agent.current_state == RobotState.NAVIGATING
                and agent.current_task is not None
                and getattr(agent.current_task, "task_type", TaskType.EXPLORATION) == TaskType.EXPLORATION
            ):
                available_targets = [
                    t for t in self.task_manager.get_available_tasks(current_time=timestamp)
                    if getattr(t, "task_type", None) == TaskType.TARGET_CONFIRMATION
                ]
                if available_targets:
                    should_allocate = True

            if should_allocate:
                available_tasks = self.task_manager.get_available_tasks(current_time=timestamp)
                if not available_tasks:
                    continue

                local_swarm_state = self.robot_swarm_states[agent.robot_id]
                chosen_task = self.allocation_strategy.allocate_task(
                    robot_id=agent.robot_id,
                    robot_pose=agent.pose,
                    available_tasks=available_tasks,
                    swarm_state=local_swarm_state,
                    pheromone_map=self.pheromone_map,
                    target_probability_map=self.target_probability_map,
                    search_memory=self.search_memory,
                    battery_level=agent.battery_level,
                )

                if chosen_task is not None:
                    assigned = agent.assign_task(chosen_task, timestamp)
                    if assigned:
                        self.task_manager.assign_task(chosen_task.task_id, agent.robot_id, timestamp)
                        self.reassignment_events_count += 1
                        for other_id, sstate in self.robot_swarm_states.items():
                            if other_id != agent.robot_id:
                                sstate.update_or_add_peer(
                                    robot_id=agent.robot_id,
                                    state=agent.current_state,
                                    pose=agent.pose,
                                    timestamp=timestamp,
                                    assigned_task_id=chosen_task.task_id,
                                    target_probability_map=self.target_probability_map,
                                    search_memory=self.search_memory,
                                    battery_level=agent.battery_level,
                                )
                        self.pheromone_updater.deposit_avoidance(
                            x=chosen_task.centroid_x,
                            y=chosen_task.centroid_y,
                            amount=6.0,
                            radius=3.0,
                        )

        # 4.5. Multi-Stage Perception evaluation & Target Confirmation
        for agent in self.robots.values():
            if self.failure_detector.is_failed(agent.robot_id):
                continue
            if agent.current_state == RobotState.VERIFYING and not self.target_found:
                self.mission_state = SwarmMissionState.VERIFYING

            if agent.last_confirmed_match is not None:
                match = agent.last_confirmed_match
                if not self.target_found:
                    self.target_found = True
                    self.mission_state = SwarmMissionState.TARGET_FOUND
                    self.confirmed_target_location = match.location
                    self.confirmed_target_plate = match.plate_number
                    self.target_discovery_time = timestamp
                    self.discovering_robot_id = match.reporting_robot_id
                    self.target_confirmation_confidence = match.match_confidence
                    self.confidence_stage_breakdown = match.stage_breakdown
                    self.target_distance_traveled = agent.sim_robot.total_distance
                    self.total_target_confirmations_count += 1

                self.pheromone_updater.deposit_success(
                    x=match.location[0],
                    y=match.location[1],
                    amount=35.0,
                    radius=8.0,
                )
                for sstate in self.robot_swarm_states.values():
                    sstate.target_found = True
                    sstate.confirmed_target_location = Pose2D(x=match.location[0], y=match.location[1], theta=0.0)

        # 5. Pheromone updates & discrete evaporation for active robots
        for agent in self.robots.values():
            if self.failure_detector.is_failed(agent.robot_id):
                continue
            self.pheromone_updater.deposit_exploration(
                x=agent.pose.x,
                y=agent.pose.y,
                amount=0.5,
                radius=1.5,
            )
            self.pheromone_updater.deposit_avoidance(
                x=agent.pose.x,
                y=agent.pose.y,
                amount=2.0,
                radius=1.5,
            )

        self.pheromone_updater.step_evaporation(dt=dt)

        # 6. Task completion accounting & pruning
        for task in list(self.task_manager.tasks.values()):
            if task.status == TaskStatus.COMPLETED and task.task_id not in self._completed_task_ids:
                self._completed_task_ids.add(task.task_id)
                self.completed_tasks_count += 1

        self.task_manager.prune_tasks(current_time=timestamp)

        return self.get_swarm_status(timestamp)

    def get_swarm_status(self, timestamp: float = 0.0) -> Dict[str, Any]:
        """Returns consolidated diagnostic summary of the swarm."""
        states = {rid: r.current_state.value for rid, r in self.robots.items()}
        active_assignments = {
            rid: r.current_task.task_id for rid, r in self.robots.items() if r.current_task
        }
        exploration_ratios = [
            r.last_slam_state.exploration_ratio for r in self.robots.values() if r.last_slam_state
        ]
        avg_exp = sum(exploration_ratios) / len(exploration_ratios) if exploration_ratios else 0.0

        total_dets = self.total_vehicle_detections_count + sum(r.total_detections_count for r in self.robots.values())
        total_fps = self.total_false_positives_count + sum(r.total_false_positives for r in self.robots.values())

        return {
            "timestamp": timestamp,
            "mission_state": self.mission_state.value,
            "robot_count": len(self.robots),
            "failed_robots_count": len(self.failure_detector.get_all_failed_robots()),
            "recovered_tasks_count": self.recovered_tasks_count,
            "robot_states": states,
            "active_assignments": active_assignments,
            "pending_tasks_count": len(self.task_manager.get_available_tasks(timestamp)),
            "completed_tasks_count": self.completed_tasks_count,
            "reassignment_events_count": self.reassignment_events_count,
            "average_exploration_ratio": avg_exp,
            "target_found": self.target_found,
            "confirmed_target_plate": self.confirmed_target_plate,
            "confirmed_target_location": self.confirmed_target_location,
            "target_discovery_time": self.target_discovery_time,
            "discovering_robot_id": self.discovering_robot_id,
            "target_confirmation_confidence": self.target_confirmation_confidence,
            "confidence_stage_breakdown": self.confidence_stage_breakdown,
            "target_distance_traveled": self.target_distance_traveled,
            "total_vehicle_detections": total_dets,
            "total_false_positives": total_fps,
        }
