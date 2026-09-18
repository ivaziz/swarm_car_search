"""
Autonomous Robot Agent Orchestrator Fusing SLAM, FSM, and Motion Control.
"""

from __future__ import annotations
from typing import TYPE_CHECKING, Optional, Tuple, List
from interfaces.slam_interface import Pose2D, ISLAMProvider, SLAMState
from interfaces.perception_interface import MatchEvent
from swarm.states.robot_states import RobotState
from swarm.states.state_machine import RobotStateMachine
from swarm.task_allocation.frontier_task import SearchTask, TaskStatus, TaskType
from swarm.communication.swarm_messages import (
    SwarmMessage,
    create_heartbeat_message,
    create_vehicle_alert_message,
)
from simulation.environment import UrbanEnvironment
from simulation.sim_robot import SimulatedRobot

if TYPE_CHECKING:
    from swarm.communication.comm_interface import ICommunicationChannel
    from perception.vehicle_detector import TargetVehicle, SimulatedVehicleDetector
    from perception.plate_recognizer import SimulatedPlateRecognizer
    from perception.matching_engine import MatchingEngine


class RobotAgent:
    """
    Onboard AI Agent orchestrator for a single autonomous robot.
    Consumes SLAM states, executes task navigation via FSM, and generates telemetry.
    """

    def __init__(
        self,
        robot_id: str,
        initial_pose: Pose2D,
        slam_provider: ISLAMProvider,
        max_linear_speed: float = 1.5,
        max_angular_speed: float = 1.5,
        zone_search_duration: float = 10.0,
        reporting_duration: float = 5.0,
        comm_channel: Optional["ICommunicationChannel"] = None,
        comm_timeout_sec: float = 8.0,
        detector: Optional["SimulatedVehicleDetector"] = None,
        ocr: Optional["SimulatedPlateRecognizer"] = None,
        matcher: Optional["MatchingEngine"] = None,
        known_vehicles: Optional[List["TargetVehicle"]] = None,
    ) -> None:
        self.robot_id = robot_id
        self.slam_provider = slam_provider
        self.comm_channel = comm_channel
        self.comm_timeout_sec = comm_timeout_sec
        self.detector = detector
        self.ocr = ocr
        self.matcher = matcher
        self.known_vehicles = known_vehicles or []
        self.last_confirmed_match: Optional[MatchEvent] = None
        self.fsm = RobotStateMachine(robot_id=robot_id, initial_state=RobotState.IDLE)
        self.sim_robot = SimulatedRobot(
            robot_id=robot_id,
            initial_pose=initial_pose,
            max_linear_speed=max_linear_speed,
            max_angular_speed=max_angular_speed,
        )

        self.current_task: Optional[SearchTask] = None
        self.assigned_waypoint: Optional[Tuple[float, float]] = None
        self.zone_search_duration = zone_search_duration
        self.reporting_duration = reporting_duration
        self._search_elapsed = 0.0
        self._reporting_elapsed = 0.0
        self.last_slam_state: Optional[SLAMState] = None
        self.last_peer_comm_timestamp: float = 0.0
        self.inbox_messages: list = []
        self.is_alive: bool = True
        self.is_transmitting: bool = True

        # Target candidate and perception metrics
        self.latest_detected_candidate: Optional[VehicleDetectionEvent] = None
        self.total_detections_count: int = 0
        self.total_target_confirmations: int = 0
        self.total_false_positives: int = 0

        if self.comm_channel is not None:
            self.comm_channel.register_node(self.robot_id)

    @property
    def current_state(self) -> RobotState:
        """Returns the robot's current FSM state."""
        return self.fsm.current_state

    @property
    def pose(self) -> Pose2D:
        """Returns current physical robot pose."""
        return self.sim_robot.pose

    @property
    def battery_level(self) -> float:
        """Returns normalized battery state [0.0, 1.0]."""
        return self.sim_robot.battery

    def inject_failure(self, timestamp: float, reason: str = "SIMULATED_FAILURE") -> None:
        """Injects fatal hardware/subsystem failure into the robot."""
        self.is_alive = False
        self.sim_robot.stop()
        self.fsm.transition_to(RobotState.FAILED, timestamp, reason=reason)

    def inject_communication_loss(self, active: bool = True) -> None:
        """Toggles heartbeat broadcasting (True stops comms, False restores)."""
        self.is_transmitting = not active

    def inject_battery_drain(self, level: float = 0.05) -> None:
        """Simulates rapid battery exhaustion."""
        self.sim_robot.battery = max(0.0, min(level, 1.0))

    def assign_task(self, task: SearchTask, timestamp: float) -> bool:
        """
        Assigns a new search task and transitions the robot to NAVIGATING.
        Urgent target confirmation tasks preempt background exploration.
        """
        # If currently idle, reassigning, or searching, accept assignment
        if self.current_state in (RobotState.IDLE, RobotState.REASSIGNING, RobotState.SEARCHING):
            self.fsm.transition_to(RobotState.NAVIGATING, timestamp, reason=f"ASSIGN_TASK_{task.task_id}")
            self.current_task = task
            self.assigned_waypoint = (task.centroid_x, task.centroid_y)
            self.current_task.mark_assigned(self.robot_id, timestamp)
            self._search_elapsed = 0.0
            return True
        elif (
            self.current_state == RobotState.NAVIGATING
            and getattr(task, "task_type", None) == TaskType.TARGET_CONFIRMATION
        ):
            # Target confirmation preempts generic exploration
            if (
                self.current_task is not None
                and getattr(self.current_task, "task_type", None) != TaskType.TARGET_CONFIRMATION
            ):
                self.current_task.mark_released(timestamp)
            self.current_task = task
            self.assigned_waypoint = (task.centroid_x, task.centroid_y)
            self.current_task.mark_assigned(self.robot_id, timestamp)
            self._search_elapsed = 0.0
            return True
        return False

    def on_vehicle_detected(self, timestamp: float) -> None:
        """Interrupt hook when vision detector observes candidate vehicle."""
        if self.current_state in (RobotState.SEARCHING, RobotState.NAVIGATING):
            self.sim_robot.stop()
            self.fsm.transition_to(RobotState.VEHICLE_DETECTED, timestamp, reason="VISUAL_DETECTOR_TRIGGER")

    def create_heartbeat(self, timestamp: float) -> Optional[SwarmMessage]:
        """Builds standardized periodic heartbeat message if communicative."""
        if not self.is_transmitting or not self.is_alive:
            return None
        task_id = self.current_task.task_id if self.current_task else None
        return create_heartbeat_message(
            sender_id=self.robot_id,
            timestamp=timestamp,
            state=self.current_state,
            pose=self.sim_robot.pose,
            battery_level=self.sim_robot.battery,
            current_task_id=task_id,
        )

    def run_perception_scan(
        self,
        timestamp: float,
        env: Optional[UrbanEnvironment] = None,
    ) -> Optional[MatchEvent]:
        """
        Executes camera detection, OCR, and watchlist matching pipeline.
        Distinguishes CONFIRMED, CANDIDATE, and NON_TARGET vehicle detections.
        """
        if self.detector is None or not self.known_vehicles:
            return None

        detections = self.detector.detect_vehicles(
            robot_id=self.robot_id,
            robot_pose=self.sim_robot.pose,
            timestamp=timestamp,
            vehicles=self.known_vehicles,
            env=env,
        )

        if not detections:
            return None

        det_event, ground_truth_vehicle = detections[0]
        self.total_detections_count += 1
        self.latest_detected_candidate = det_event

        dist = getattr(det_event, "distance", 10.0)

        # Run OCR
        plate_event = None
        if self.ocr is not None:
            if dist <= 5.0:
                self.sim_robot.stop()
                self.fsm.transition_to(RobotState.VERIFYING, timestamp, reason="RUN_CLOSE_RANGE_PLATE_OCR")
            else:
                if self.current_state != RobotState.VERIFYING:
                    self.fsm.transition_to(RobotState.VEHICLE_DETECTED, timestamp, reason="CANDIDATE_VEHICLE_SIGHTED")
            plate_event = self.ocr.recognize_license_plate(det_event, ground_truth_vehicle)

        # Match against watchlist
        match_event = None
        if self.matcher is not None and plate_event is not None:
            if hasattr(self.matcher, "classify_plate"):
                classification, match_event, sim = self.matcher.classify_plate(plate_event)
            else:
                match_event = self.matcher.match_plate(plate_event)
                classification = "CONFIRMED" if (match_event and match_event.confirmed) else "NON_TARGET"

            if classification == "CONFIRMED" and match_event is not None:
                self.sim_robot.stop()
                self._reporting_elapsed = 0.0
                self.fsm.transition_to(RobotState.REPORTING, timestamp, reason="CONFIRMED_TARGET_MATCH")
                self.last_confirmed_match = match_event
                self.total_target_confirmations += 1
                if self.comm_channel is not None:
                    alert_msg = create_vehicle_alert_message(self.robot_id, timestamp, match_event)
                    self.comm_channel.broadcast(alert_msg)
                return match_event
            elif classification == "CANDIDATE":
                # Candidate under investigation: navigate closer for clear close-range OCR
                cx, cy = det_event.estimated_global_position
                self.assigned_waypoint = (cx, cy)
                if self.current_state != RobotState.NAVIGATING:
                    self.fsm.transition_to(RobotState.NAVIGATING, timestamp, reason="APPROACHING_CANDIDATE_FOR_OCR")
                return None
            else:
                # Non-target / false positive rejected
                self.total_false_positives += 1
                if (
                    self.current_task is not None
                    and getattr(self.current_task, "target_vehicle_id", None) == ground_truth_vehicle.target_id
                ):
                    self.current_task.mark_completed(timestamp)
                    self.current_task = None
                    self.assigned_waypoint = None
                if self.current_state != RobotState.SEARCHING:
                    self.fsm.transition_to(RobotState.SEARCHING, timestamp, reason="NON_TARGET_OR_UNCONFIRMED")
                return None
        else:
            if self.current_state != RobotState.SEARCHING:
                self.fsm.transition_to(RobotState.SEARCHING, timestamp, reason="NO_MATCHER_OR_OCR")
            return None

    def step(
        self,
        timestamp: float,
        dt: float,
        env: Optional[UrbanEnvironment] = None,
    ) -> None:
        """
        Executes one iteration of the robot control loop:
        1. Propagate physics/kinematics.
        2. Update SLAM state.
        3. Run perception scanner.
        4. Execute state-specific motion and logic.
        """
        # Check fatal failure or dead robot
        if not self.is_alive or self.current_state == RobotState.FAILED:
            self.sim_robot.stop()
            return

        # Check total battery depletion
        if self.sim_robot.battery <= 0.0:
            self.sim_robot.stop()
            if self.current_state != RobotState.FAILED:
                self.fsm.transition_to(RobotState.FAILED, timestamp, reason="BATTERY_EXHAUSTED")
            return

        # 0. Inter-robot communication processing
        if self.comm_channel is not None:
            self.comm_channel.update_node_pose(self.robot_id, self.sim_robot.pose.x, self.sim_robot.pose.y)
            incoming = self.comm_channel.receive_inbox(self.robot_id)
            if incoming:
                self.last_peer_comm_timestamp = timestamp
                self.inbox_messages.extend(incoming)
                if self.current_state == RobotState.COMMUNICATION_LOST:
                    resume_state = RobotState.NAVIGATING if self.current_task else RobotState.SEARCHING
                    self.fsm.transition_to(resume_state, timestamp, reason="COMMUNICATION_RESTORED")
            elif self.last_peer_comm_timestamp > 0.0 and self.current_state in (RobotState.NAVIGATING, RobotState.SEARCHING):
                if (timestamp - self.last_peer_comm_timestamp) > self.comm_timeout_sec:
                    self.fsm.transition_to(RobotState.COMMUNICATION_LOST, timestamp, reason="COMMUNICATION_TIMEOUT")

        # 1. Physics integration
        self.sim_robot.step(dt=dt, env=env)

        # 2. Update SLAM
        if hasattr(self.slam_provider, "update_robot_pose"):
            self.last_slam_state = self.slam_provider.update_robot_pose(
                robot_id=self.robot_id,
                pose=self.sim_robot.pose,
                timestamp=timestamp,
                linear_velocity=self.sim_robot.velocity.linear,
                angular_velocity=self.sim_robot.velocity.angular,
            )
        elif self.slam_provider.is_healthy(self.robot_id):
            self.last_slam_state = self.slam_provider.get_latest_state(self.robot_id)

        # 3. Perception scan
        if self.current_state in (RobotState.SEARCHING, RobotState.NAVIGATING):
            self.run_perception_scan(timestamp=timestamp, env=env)

        # 3. FSM-driven behavior logic
        state = self.current_state

        if state == RobotState.NAVIGATING:
            if self.assigned_waypoint is not None:
                tx, ty = self.assigned_waypoint
                arrived = self.sim_robot.drive_towards_waypoint(tx, ty, tolerance=0.8, env=env)
                if arrived:
                    self.fsm.transition_to(RobotState.SEARCHING, timestamp, reason="ARRIVED_AT_FRONTIER_WAYPOINT")
                    self._search_elapsed = 0.0
                    if self.current_task is not None:
                        self.current_task.mark_in_progress(timestamp)
            else:
                self.sim_robot.stop()
                self.fsm.transition_to(RobotState.IDLE, timestamp, reason="NO_WAYPOINT")

        elif state == RobotState.SEARCHING:
            # Sweeping scan motion: rotate gently while inspecting zone
            self.sim_robot.set_velocity(linear=0.4, angular=0.5)
            self._search_elapsed += dt
            if self._search_elapsed >= self.zone_search_duration:
                # Finished searching current frontier zone
                self.sim_robot.stop()
                if self.current_task is not None:
                    self.current_task.mark_completed(timestamp)
                    self.current_task = None
                self.assigned_waypoint = None
                self.fsm.transition_to(RobotState.REASSIGNING, timestamp, reason="ZONE_SEARCH_COMPLETED")

        elif state == RobotState.COMMUNICATION_LOST:
            # Autonomous local fallback: continue cautious local sweep
            self.sim_robot.set_velocity(linear=0.3, angular=0.2)

        elif state == RobotState.REASSIGNING:
            self.sim_robot.stop()
            # Ready for next assignment
            self.fsm.transition_to(RobotState.IDLE, timestamp, reason="AWAITING_NEW_TASK")

        elif state == RobotState.VEHICLE_DETECTED:
            self.sim_robot.stop()
            # Move immediately to plate verification
            self.fsm.transition_to(RobotState.VERIFYING, timestamp, reason="START_OCR_VERIFICATION")

        elif state == RobotState.VERIFYING:
            self.sim_robot.stop()
            self._reporting_elapsed = 0.0
            # Default behavior after verification: transition to reporting or back to search
            self.fsm.transition_to(RobotState.REPORTING, timestamp, reason="MATCH_VERIFIED")

        elif state == RobotState.REPORTING:
            self.sim_robot.stop()
            self._reporting_elapsed += dt
            if self._reporting_elapsed >= self.reporting_duration:
                # Finished dispatching alerts, resume search or reassignment
                self._reporting_elapsed = 0.0
                self.fsm.transition_to(RobotState.SEARCHING, timestamp, reason="RESUME_SEARCH_AFTER_REPORT")

        elif state == RobotState.IDLE:
            self.sim_robot.stop()
