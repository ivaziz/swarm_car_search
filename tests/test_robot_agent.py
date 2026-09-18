"""
Comprehensive unit and integration tests for State Machine, Simulated Robot Kinematics, and RobotAgent.
"""

import unittest
import math
from interfaces.slam_interface import Pose2D
from simulation.environment import UrbanEnvironment
from simulation.sim_robot import SimulatedRobot, Twist2D
from slam_providers.mock_slam_provider import MockSLAMProvider
from swarm.states.robot_states import RobotState
from swarm.states.state_machine import RobotStateMachine
from swarm.task_allocation.frontier_task import SearchTask, TaskStatus
from swarm.core.robot_agent import RobotAgent
from swarm.communication.swarm_messages import MessageType


class TestRobotStateMachine(unittest.TestCase):
    """Tests for FSM engine, transition callbacks, and history tracking."""

    def test_transition_and_history(self):
        fsm = RobotStateMachine("robot_1", initial_state=RobotState.IDLE, initial_timestamp=10.0)
        self.assertEqual(fsm.current_state, RobotState.IDLE)
        self.assertEqual(fsm.time_entered, 10.0)

        # Transition to NAVIGATING
        success = fsm.transition_to(RobotState.NAVIGATING, timestamp=15.0, reason="GOTO_WAYPOINT")
        self.assertTrue(success)
        self.assertEqual(fsm.current_state, RobotState.NAVIGATING)
        self.assertEqual(fsm.time_in_state(20.0), 5.0)

        # Check history contains both initial and transition
        self.assertEqual(len(fsm.history), 2)
        self.assertEqual(fsm.history[1][1], RobotState.IDLE)
        self.assertEqual(fsm.history[1][2], RobotState.NAVIGATING)

    def test_callbacks(self):
        fsm = RobotStateMachine("robot_1", initial_state=RobotState.IDLE, initial_timestamp=0.0)
        entered = []
        exited = []

        fsm.register_on_enter(RobotState.NAVIGATING, lambda s, t, r: entered.append(s))
        fsm.register_on_exit(RobotState.IDLE, lambda s, t, r: exited.append(s))

        fsm.transition_to(RobotState.NAVIGATING, timestamp=1.0)
        self.assertEqual(exited, [RobotState.IDLE])
        self.assertEqual(entered, [RobotState.NAVIGATING])

    def test_invalid_transition_modes(self):
        fsm = RobotStateMachine("robot_1", initial_state=RobotState.IDLE)
        # Strict mode raises ValueError
        with self.assertRaises(ValueError):
            fsm.transition_to(RobotState.REPORTING, timestamp=1.0, strict=True)

        # Permissive mode returns False
        success = fsm.transition_to(RobotState.REPORTING, timestamp=1.0, strict=False)
        self.assertFalse(success)
        self.assertEqual(fsm.current_state, RobotState.IDLE)


class TestSimulatedRobot(unittest.TestCase):
    """Tests for physical kinematics integration and waypoint controller."""

    def test_velocity_clamping(self):
        robot = SimulatedRobot("robot_1", Pose2D(0.0, 0.0, 0.0), max_linear_speed=1.5, max_angular_speed=1.0)
        robot.set_velocity(5.0, -3.0)
        self.assertEqual(robot.velocity.linear, 1.5)
        self.assertEqual(robot.velocity.angular, -1.0)

    def test_kinematic_step(self):
        robot = SimulatedRobot("robot_1", Pose2D(0.0, 0.0, 0.0))
        robot.set_velocity(1.0, 0.0)
        new_pose = robot.step(dt=1.0)
        self.assertAlmostEqual(new_pose.x, 1.0)
        self.assertAlmostEqual(new_pose.y, 0.0)

    def test_collision_halting(self):
        env = UrbanEnvironment(width=20.0, height=20.0, resolution=1.0)
        env.set_cell(2, 0, 100)  # Obstacle at x=2, y=0

        robot = SimulatedRobot("robot_1", Pose2D(1.0, 0.5, 0.0))
        robot.set_velocity(2.0, 0.0)  # Drive into obstacle

        # Step forward
        robot.step(dt=1.0, env=env)
        # Should halt before entering obstacle cell
        self.assertEqual(robot.velocity.linear, 0.0)
        self.assertLess(robot.pose.x, 2.0)

    def test_waypoint_controller_convergence(self):
        robot = SimulatedRobot("robot_1", Pose2D(0.0, 0.0, 0.0))
        target_x, target_y = 3.0, 4.0

        # Simulate closed-loop navigation
        dt = 0.1
        arrived = False
        for _ in range(100):
            arrived = robot.drive_towards_waypoint(target_x, target_y, tolerance=0.3)
            robot.step(dt=dt)
            if arrived:
                break

        self.assertTrue(arrived)
        distance = math.hypot(target_x - robot.pose.x, target_y - robot.pose.y)
        self.assertLessEqual(distance, 0.5)


class TestRobotAgent(unittest.TestCase):
    """Tests for integrated RobotAgent orchestrator."""

    def setUp(self):
        self.env = UrbanEnvironment(width=30.0, height=30.0, resolution=1.0)
        self.slam = MockSLAMProvider(self.env)
        self.agent = RobotAgent(
            robot_id="robot_alpha",
            initial_pose=Pose2D(5.0, 5.0, 0.0),
            slam_provider=self.slam,
            zone_search_duration=1.0,  # Fast search duration for test
        )

    def test_task_lifecycle_execution(self):
        self.assertEqual(self.agent.current_state, RobotState.IDLE)

        # Assign task
        task = SearchTask(
            task_id="task_test_01",
            cluster_id="c_01",
            centroid_x=7.0,
            centroid_y=5.0,
            bounding_box=(6.0, 4.0, 8.0, 6.0),
            size=10,
        )
        assigned = self.agent.assign_task(task, timestamp=0.0)
        self.assertTrue(assigned)
        self.assertEqual(self.agent.current_state, RobotState.NAVIGATING)
        self.assertEqual(task.status, TaskStatus.ASSIGNED)

        # Step until arrived at waypoint
        dt = 0.1
        sim_time = 0.0
        while self.agent.current_state == RobotState.NAVIGATING and sim_time < 10.0:
            sim_time += dt
            self.agent.step(timestamp=sim_time, dt=dt, env=self.env)

        # Must have arrived and transitioned to SEARCHING
        self.assertEqual(self.agent.current_state, RobotState.SEARCHING)
        self.assertEqual(task.status, TaskStatus.IN_PROGRESS)

        # Step through SEARCHING duration
        while self.agent.current_state == RobotState.SEARCHING and sim_time < 20.0:
            sim_time += dt
            self.agent.step(timestamp=sim_time, dt=dt, env=self.env)

        # Must have completed zone search and marked task COMPLETED
        self.assertEqual(task.status, TaskStatus.COMPLETED)

    def test_perception_interrupt_flow(self):
        self.agent.fsm.transition_to(RobotState.SEARCHING, timestamp=0.0)
        self.agent.on_vehicle_detected(timestamp=1.0)

        self.assertEqual(self.agent.current_state, RobotState.VEHICLE_DETECTED)

        # Step agent: VEHICLE_DETECTED -> VERIFYING
        self.agent.step(timestamp=2.0, dt=0.1)
        self.assertEqual(self.agent.current_state, RobotState.VERIFYING)

        # Step agent: VERIFYING -> REPORTING
        self.agent.step(timestamp=3.0, dt=0.1)
        self.assertEqual(self.agent.current_state, RobotState.REPORTING)

    def test_heartbeat_generation(self):
        hb = self.agent.create_heartbeat(timestamp=50.0)
        self.assertEqual(hb.sender_id, "robot_alpha")
        self.assertEqual(hb.msg_type, MessageType.HEARTBEAT)
        self.assertEqual(hb.payload["state"], RobotState.IDLE.value)
        self.assertAlmostEqual(hb.payload["pose"]["x"], 5.0)


if __name__ == "__main__":
    unittest.main()
