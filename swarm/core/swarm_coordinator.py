"""
Decentralized Swarm Coordinator Orchestrating Multi-Robot Exploration and Dynamic Task Allocation.
"""

from typing import Dict, List, Optional, Any, Tuple
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


class SwarmCoordinator:
    """
    Coordinates multi-robot swarm behavior:
    - Integrates decentralized robot agents.
    - Synchronizes SLAM frontier ingestion into TaskManager.
    - Resolves dynamic task assignments via ACO without central point of failure.
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

        # Failure detection and dynamic recovery
        self.failure_detector = failure_detector or FailureDetector(failure_config)
        self.recovery_manager = recovery_manager or RecoveryManager()

        # Swarm registry: robot_id -> RobotAgent
        self.robots: Dict[str, RobotAgent] = {}
        # Per-robot decentralized state perspective
        self.robot_swarm_states: Dict[str, SwarmState] = {}

        # Aggregate telemetry metrics
        self.completed_tasks_count = 0
        self._completed_task_ids: Set[str] = set()
        self.reassignment_events_count = 0
        self.failed_robots_count = 0
        self.recovered_tasks_count = 0
        self.target_found = False
        self.confirmed_target_location: Optional[Tuple[float, float]] = None
        self.confirmed_target_plate: Optional[str] = None

        # Comprehensive Target Identification Telemetry
        self.target_discovery_time: Optional[float] = None
        self.discovering_robot_id: Optional[str] = None
        self.target_confirmation_confidence: Optional[float] = None
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
        1. Propagate robot physics and local SLAM updates.
        2. Ingest peer heartbeats and evaluate failure detection & dynamic recovery.
        3. Ingest newly observed frontier clusters into task pool.
        4. Dynamically allocate pending tasks to idle/reassigning active robots.
        5. Evaluate visual perception alerts & target recruitment.
        6. Stigmergic pheromone deposits (exploration + avoidance) & decay.
        7. Prune stale tasks and compile telemetry metrics.
        """
        # 1. Physics & SLAM propagation
        for agent in self.robots.values():
            agent.step(timestamp=timestamp, dt=dt, env=self.env)

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
                    if self.env is not None:
                        prior = self.env.get_zone_prior(frontier.centroid_x, frontier.centroid_y)

                    self.task_manager.ingest_frontiers(
                        frontiers=[frontier],
                        timestamp=timestamp,
                        detection_probability=prior,
                    )

        # 3.5. Candidate vehicle detection & Stigmergic recruitment
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
                    priority=2.0 + 3.0 * p_target,
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
                                )
                        self.pheromone_updater.deposit_avoidance(
                            x=chosen_task.centroid_x,
                            y=chosen_task.centroid_y,
                            amount=6.0,
                            radius=3.0,
                        )

        # 4.5. Perception detection evaluation & Target recruitment
        for agent in self.robots.values():
            if self.failure_detector.is_failed(agent.robot_id):
                continue
            if agent.last_confirmed_match is not None:
                match = agent.last_confirmed_match
                if not self.target_found:
                    self.target_found = True
                    self.confirmed_target_location = match.location
                    self.confirmed_target_plate = match.plate_number
                    self.target_discovery_time = timestamp
                    self.discovering_robot_id = match.reporting_robot_id
                    self.target_confirmation_confidence = match.match_confidence
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
            "confirmed_target_location": self.confirmed_target_location,
            "target_discovery_time": self.target_discovery_time,
            "discovering_robot_id": self.discovering_robot_id,
            "target_confirmation_confidence": self.target_confirmation_confidence,
            "target_distance_traveled": self.target_distance_traveled,
            "total_vehicle_detections": total_dets,
            "total_false_positives": total_fps,
        }
