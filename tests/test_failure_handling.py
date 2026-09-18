"""
Comprehensive Unit and Integration Tests for Swarm Failure Detection,
Fault-Tolerant Dynamic Reassignment, and Swarm Resilience.
"""

import unittest
import math
from interfaces.slam_interface import Pose2D
from simulation.environment import UrbanEnvironment
from slam_providers.mock_slam_provider import MockSLAMProvider
from swarm.states.robot_states import RobotState
from swarm.task_allocation.frontier_task import SearchTask, TaskStatus
from swarm.task_allocation.task_manager import TaskManager
from swarm.task_allocation.allocation_strategy import ACOAllocationStrategy
from swarm.aco.aco_config import ACOConfig
from swarm.aco.aco_engine import ACOEngine
from swarm.pheromone.pheromone_map import PheromoneMap
from swarm.pheromone.pheromone_updater import PheromoneUpdater
from swarm.pheromone.evaporation_model import PheromoneLayerType
from swarm.core.swarm_state import SwarmState
from swarm.core.robot_agent import RobotAgent
from swarm.core.swarm_coordinator import SwarmCoordinator
from swarm.failure_handling.failure_detector import (
    FailureDetector,
    FailureDetectorConfig,
    FailureType,
    RobotFailureEvent,
)
from swarm.failure_handling.recovery_manager import (
    RecoveryManager,
    RecoveryAction,
)


class TestFailureDetector(unittest.TestCase):
    """Tests for individual failure detection heuristics."""

    def setUp(self):
        self.config = FailureDetectorConfig(
            heartbeat_timeout_sec=5.0,
            battery_critical_threshold=0.10,
            stall_timeout_sec=6.0,
            stall_distance_threshold=0.3,
        )
        self.detector = FailureDetector(self.config)

    def test_heartbeat_timeout(self):
        pose = Pose2D(10.0, 10.0, 0.0)
        self.detector.record_telemetry(
            robot_id="r1",
            timestamp=0.0,
            pose=pose,
            battery_level=0.9,
            state=RobotState.NAVIGATING,
        )

        # At t=3.0s: silence is 3s < 5.0s -> healthy
        failures = self.detector.check_failures(current_time=3.0)
        self.assertEqual(len(failures), 0)
        self.assertFalse(self.detector.is_failed("r1"))

        # At t=6.0s: silence is 6s > 5.0s -> HEARTBEAT_TIMEOUT
        failures = self.detector.check_failures(current_time=6.0)
        self.assertEqual(len(failures), 1)
        self.assertEqual(failures[0].failure_type, FailureType.HEARTBEAT_TIMEOUT)
        self.assertEqual(failures[0].robot_id, "r1")
        self.assertTrue(self.detector.is_failed("r1"))

    def test_battery_critical(self):
        pose = Pose2D(5.0, 5.0, 0.0)
        # Battery at 8% (below 10% threshold)
        self.detector.record_telemetry(
            robot_id="r_low_bat",
            timestamp=10.0,
            pose=pose,
            battery_level=0.08,
            state=RobotState.SEARCHING,
        )

        failures = self.detector.check_failures(current_time=10.0)
        self.assertEqual(len(failures), 1)
        self.assertEqual(failures[0].failure_type, FailureType.BATTERY_CRITICAL)
        self.assertEqual(failures[0].robot_id, "r_low_bat")
        self.assertTrue(self.detector.is_failed("r_low_bat"))

    def test_stalled_robot(self):
        initial_pose = Pose2D(20.0, 20.0, 0.0)
        self.detector.record_telemetry(
            robot_id="r_stuck",
            timestamp=0.0,
            pose=initial_pose,
            battery_level=0.8,
            state=RobotState.NAVIGATING,
        )

        # Robot stays essentially in the same spot for 7s (> stall_timeout_sec 6s)
        stuck_pose = Pose2D(20.05, 20.02, 0.1)  # moved only 0.05m
        self.detector.record_telemetry(
            robot_id="r_stuck",
            timestamp=7.0,
            pose=stuck_pose,
            battery_level=0.79,
            state=RobotState.NAVIGATING,
        )

        failures = self.detector.check_failures(current_time=7.0)
        self.assertEqual(len(failures), 1)
        self.assertEqual(failures[0].failure_type, FailureType.STALLED)

    def test_explicit_fault(self):
        pose = Pose2D(1.0, 1.0, 0.0)
        self.detector.record_telemetry(
            robot_id="r_dead",
            timestamp=5.0,
            pose=pose,
            battery_level=0.5,
            state=RobotState.FAILED,
        )

        failures = self.detector.check_failures(current_time=5.0)
        self.assertEqual(len(failures), 1)
        self.assertEqual(failures[0].failure_type, FailureType.EXPLICIT_FAULT)

    def test_robot_reset(self):
        self.detector.record_telemetry(
            robot_id="r_reboot",
            timestamp=1.0,
            pose=Pose2D(0.0, 0.0, 0.0),
            battery_level=0.0,
            state=RobotState.FAILED,
        )
        self.detector.check_failures(current_time=1.0)
        self.assertTrue(self.detector.is_failed("r_reboot"))

        # Reboot / reset
        self.detector.reset_robot("r_reboot")
        self.assertFalse(self.detector.is_failed("r_reboot"))


class TestRecoveryManager(unittest.TestCase):
    """Tests for dynamic reassignment and stigmergic obstacle marking."""

    def setUp(self):
        self.task_mgr = TaskManager()
        self.recovery = RecoveryManager(wreck_avoidance_amount=15.0, wreck_avoidance_radius=2.0)
        self.pmap = PheromoneMap(width=30.0, height=30.0, resolution=1.0)
        self.pupdater = PheromoneUpdater(self.pmap)

        # Create task
        self.task = SearchTask(
            task_id="task_rescue_01",
            cluster_id="c_01",
            centroid_x=12.0,
            centroid_y=12.0,
            bounding_box=(11.0, 11.0, 13.0, 13.0),
            size=15,
        )
        self.task_mgr.tasks[self.task.task_id] = self.task
        self.task_mgr.assign_task(self.task.task_id, "robot_fail", timestamp=0.0)

        self.sstates = {
            "robot_fail": SwarmState(local_robot_id="robot_fail"),
            "robot_idle": SwarmState(local_robot_id="robot_idle"),
        }

    def test_task_release_and_avoidance_deposit(self):
        fail_event = RobotFailureEvent(
            robot_id="robot_fail",
            failure_type=FailureType.HEARTBEAT_TIMEOUT,
            timestamp=10.0,
            last_known_pose=Pose2D(10.0, 10.0, 0.0),
            abandoned_task_id=self.task.task_id,
        )

        result = self.recovery.handle_failure(
            event=fail_event,
            task_manager=self.task_mgr,
            swarm_states=self.sstates,
            pheromone_updater=self.pupdater,
        )

        self.assertEqual(result.failed_robot_id, "robot_fail")
        self.assertEqual(result.orphaned_task_id, self.task.task_id)
        self.assertIn(RecoveryAction.TASK_RELEASED, result.actions_taken)
        self.assertIn(RecoveryAction.AVOIDANCE_DEPOSITED, result.actions_taken)

        # Task must now be PENDING
        self.assertEqual(self.task.status, TaskStatus.PENDING)
        self.assertIsNone(self.task.assigned_robot)

        # Avoidance pheromone must be deposited at wreck coordinates (10, 10)
        avoid_val = self.pmap.get_value(PheromoneLayerType.AVOIDANCE, 10.0, 10.0)
        self.assertGreater(avoid_val, 10.0)


class TestSwarmCoordinatorResilience(unittest.TestCase):
    """End-to-end integration tests for multi-robot swarm resilience and dynamic reassignment."""

    def setUp(self):
        self.env = UrbanEnvironment(width=40.0, height=40.0, resolution=1.0)
        self.slam = MockSLAMProvider(self.env)

        aco_cfg = ACOConfig()
        aco_cfg.stochastic.deterministic = True
        strategy = ACOAllocationStrategy(ACOEngine(aco_cfg))

        self.coordinator = SwarmCoordinator(
            allocation_strategy=strategy,
            environment=self.env,
            failure_config=FailureDetectorConfig(heartbeat_timeout_sec=3.0),
        )

        # Register 3 agents
        self.agent_1 = RobotAgent(
            robot_id="robot_1",
            initial_pose=Pose2D(5.0, 5.0, 0.0),
            slam_provider=self.slam,
            zone_search_duration=5.0,
        )
        self.agent_2 = RobotAgent(
            robot_id="robot_2",
            initial_pose=Pose2D(6.0, 5.0, 0.0),
            slam_provider=self.slam,
            zone_search_duration=5.0,
        )
        self.coordinator.register_robot(self.agent_1)
        self.coordinator.register_robot(self.agent_2)

    def test_dynamic_reassignment_on_simulated_failure(self):
        # Create a frontier task
        task = SearchTask(
            task_id="priority_target_frontier",
            cluster_id="c_target",
            centroid_x=15.0,
            centroid_y=15.0,
            bounding_box=(14.0, 14.0, 16.0, 16.0),
            size=25,
            priority=2.0,
        )
        self.coordinator.task_manager.tasks[task.task_id] = task

        # Step 1: Assign task to robot_1
        self.coordinator.step_coordination(timestamp=1.0, dt=0.1)
        self.assertEqual(self.agent_1.current_state, RobotState.NAVIGATING)
        self.assertEqual(self.agent_1.current_task.task_id, task.task_id)

        # Step 2: Inject catastrophic hardware failure into robot_1
        event = self.coordinator.simulate_robot_failure(
            robot_id="robot_1",
            timestamp=2.0,
            reason="MOTOR_DRIVER_BURNOUT",
        )
        self.assertIsNotNone(event)
        self.assertEqual(self.agent_1.current_state, RobotState.FAILED)
        self.assertTrue(self.coordinator.failure_detector.is_failed("robot_1"))

        # Step 3: Run coordination step -> robot_2 must immediately take over the orphaned task
        self.coordinator.step_coordination(timestamp=2.1, dt=0.1)

        # Verify task was dynamically reassigned to robot_2
        self.assertEqual(self.agent_2.current_state, RobotState.NAVIGATING)
        self.assertEqual(self.agent_2.current_task.task_id, task.task_id)
        self.assertEqual(task.assigned_robot, "robot_2")

        # Telemetry metrics reflect recovered task and failed robot
        status = self.coordinator.get_swarm_status(timestamp=2.1)
        self.assertEqual(status["failed_robots_count"], 1)
        self.assertGreaterEqual(status["recovered_tasks_count"], 1)

    def test_heartbeat_silence_dynamic_timeout_in_swarm(self):
        # Assign task to robot_1
        task = SearchTask(
            task_id="task_comms_loss",
            cluster_id="c_loss",
            centroid_x=12.0,
            centroid_y=12.0,
            bounding_box=(11.0, 11.0, 13.0, 13.0),
            size=10,
        )
        self.coordinator.task_manager.tasks[task.task_id] = task
        self.coordinator.step_coordination(timestamp=1.0, dt=0.1)
        self.assertEqual(self.agent_1.current_task.task_id, task.task_id)

        # Robot 1 drops communication completely
        self.agent_1.inject_communication_loss(active=True)

        # Advance time past heartbeat_timeout_sec (3.0s)
        # Run coordination at t=5.0s (silence = 4.0s > 3.0s)
        self.coordinator.step_coordination(timestamp=5.0, dt=0.1)

        # Robot 1 must be flagged failed due to HEARTBEAT_TIMEOUT
        self.assertTrue(self.coordinator.failure_detector.is_failed("robot_1"))
        # Task must have been recovered and reassigned to robot_2
        self.assertEqual(task.assigned_robot, "robot_2")
        self.assertEqual(self.agent_2.current_state, RobotState.NAVIGATING)


class TestDecentralizedSwarmStateTimeout(unittest.TestCase):
    """Tests peer timeout detection directly inside local SwarmState."""

    def test_peer_heartbeat_timeout_release(self):
        sstate = SwarmState(local_robot_id="r_local")
        # Add task to local state
        task = SearchTask(
            task_id="task_p2p",
            cluster_id="c_01",
            centroid_x=5.0,
            centroid_y=5.0,
            bounding_box=(4.0, 4.0, 6.0, 6.0),
            size=10,
            status=TaskStatus.ASSIGNED,
            assigned_robot="r_peer",
        )
        sstate.active_tasks[task.task_id] = task

        # Register peer with last heartbeat at t=1.0
        sstate.update_or_add_peer(
            robot_id="r_peer",
            state=RobotState.NAVIGATING,
            pose=Pose2D(5.0, 5.0, 0.0),
            timestamp=1.0,
            assigned_task_id=task.task_id,
        )

        # Check timeouts at t=3.0 (timeout=5.0) -> not timed out
        timed_out = sstate.check_peer_timeouts(current_time=3.0, timeout_sec=5.0)
        self.assertEqual(len(timed_out), 0)
        self.assertTrue(sstate.peers["r_peer"].is_active)

        # Check timeouts at t=8.0 (elapsed 7.0s > 5.0s) -> timed out
        timed_out = sstate.check_peer_timeouts(current_time=8.0, timeout_sec=5.0)
        self.assertIn("r_peer", timed_out)
        self.assertFalse(sstate.peers["r_peer"].is_active)
        # Task in local swarm state must be released back to PENDING
        self.assertEqual(task.status, TaskStatus.PENDING)
        self.assertIsNone(task.assigned_robot)


if __name__ == "__main__":
    unittest.main()
