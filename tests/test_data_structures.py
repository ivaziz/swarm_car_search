"""
Comprehensive tests for core swarm data structures, state machines, messages, and serialization.
"""

import unittest
import json
from interfaces.slam_interface import Pose2D, FrontierCluster, OccupancyGrid2D, SLAMState
from interfaces.perception_interface import MatchEvent
from swarm.states.robot_states import RobotState, is_valid_transition
from swarm.task_allocation.frontier_task import SearchTask, TaskStatus
from swarm.communication.swarm_messages import (
    MessageType,
    SwarmMessage,
    create_heartbeat_message,
    create_task_claim_message,
    create_task_release_message,
    create_vehicle_alert_message,
)
from swarm.core.swarm_state import RobotPeerState, SwarmState
from slam_providers.base_slam_provider import BaseSLAMProvider


class TestRobotStates(unittest.TestCase):
    """Tests for robot state definitions and transition validity."""

    def test_valid_transitions(self):
        # Permissible transition sequence
        self.assertTrue(is_valid_transition(RobotState.IDLE, RobotState.NAVIGATING))
        self.assertTrue(is_valid_transition(RobotState.NAVIGATING, RobotState.SEARCHING))
        self.assertTrue(is_valid_transition(RobotState.SEARCHING, RobotState.VEHICLE_DETECTED))
        self.assertTrue(is_valid_transition(RobotState.VEHICLE_DETECTED, RobotState.VERIFYING))
        self.assertTrue(is_valid_transition(RobotState.VERIFYING, RobotState.REPORTING))
        self.assertTrue(is_valid_transition(RobotState.REPORTING, RobotState.SEARCHING))
        self.assertTrue(is_valid_transition(RobotState.SEARCHING, RobotState.COMPLETED))

        # Identity transition (same state) is always valid
        self.assertTrue(is_valid_transition(RobotState.SEARCHING, RobotState.SEARCHING))

    def test_invalid_transitions(self):
        # Impossible jump: IDLE directly to REPORTING
        self.assertFalse(is_valid_transition(RobotState.IDLE, RobotState.REPORTING))
        # Failed is a terminal state
        self.assertFalse(is_valid_transition(RobotState.FAILED, RobotState.NAVIGATING))


class TestSearchTask(unittest.TestCase):
    """Tests for SearchTask lifecycle, expiration, and serialization."""

    def setUp(self):
        self.frontier = FrontierCluster(
            cluster_id="cluster_42",
            centroid_x=15.0,
            centroid_y=25.0,
            size=30,
            bounding_box=(14.0, 24.0, 16.0, 26.0),
        )

    def test_creation_from_frontier(self):
        task = SearchTask.from_frontier(
            frontier=self.frontier,
            timestamp=100.0,
            detection_probability=0.75,
            priority=2.0,
            ttl_seconds=60.0,
        )
        self.assertEqual(task.cluster_id, "cluster_42")
        self.assertEqual(task.centroid_x, 15.0)
        self.assertEqual(task.centroid_y, 25.0)
        self.assertEqual(task.size, 30)
        self.assertEqual(task.status, TaskStatus.PENDING)
        self.assertEqual(task.detection_probability, 0.75)
        self.assertEqual(task.priority, 2.0)
        self.assertEqual(task.expiration_time, 160.0)

    def test_lifecycle_and_expiration(self):
        task = SearchTask.from_frontier(self.frontier, timestamp=100.0, ttl_seconds=50.0)
        self.assertFalse(task.is_expired(120.0))
        self.assertTrue(task.is_expired(151.0))

        # Assignment
        task.mark_assigned("robot_1", timestamp=110.0)
        self.assertEqual(task.status, TaskStatus.ASSIGNED)
        self.assertEqual(task.assigned_robot, "robot_1")

        # In progress
        task.mark_in_progress(timestamp=115.0)
        self.assertEqual(task.status, TaskStatus.IN_PROGRESS)

        # Release
        task.mark_released(timestamp=120.0)
        self.assertEqual(task.status, TaskStatus.PENDING)
        self.assertIsNone(task.assigned_robot)

        # Completion
        task.mark_assigned("robot_2", timestamp=125.0)
        task.mark_completed(timestamp=130.0)
        self.assertEqual(task.status, TaskStatus.COMPLETED)

    def test_task_serialization_round_trip(self):
        task = SearchTask.from_frontier(self.frontier, timestamp=100.0)
        task.mark_assigned("robot_3", timestamp=105.0)
        data = task.to_dict()

        reconstructed = SearchTask.from_dict(data)
        self.assertEqual(reconstructed.task_id, task.task_id)
        self.assertEqual(reconstructed.cluster_id, task.cluster_id)
        self.assertEqual(reconstructed.status, TaskStatus.ASSIGNED)
        self.assertEqual(reconstructed.assigned_robot, "robot_3")
        self.assertEqual(reconstructed.bounding_box, task.bounding_box)


class TestSwarmMessages(unittest.TestCase):
    """Tests for swarm communication messages and JSON serialization."""

    def test_heartbeat_message(self):
        pose = Pose2D(x=12.0, y=34.0, theta=1.57)
        msg = create_heartbeat_message(
            sender_id="robot_1",
            timestamp=500.0,
            state=RobotState.SEARCHING,
            pose=pose,
            battery_level=0.88,
            current_task_id="task_001",
        )
        self.assertEqual(msg.msg_type, MessageType.HEARTBEAT)
        self.assertEqual(msg.sender_id, "robot_1")
        self.assertEqual(msg.payload["state"], "SEARCHING")
        self.assertEqual(msg.payload["pose"]["x"], 12.0)

        # JSON Round trip
        json_str = msg.to_json()
        deserialized = SwarmMessage.from_json(json_str)
        self.assertEqual(deserialized.message_id, msg.message_id)
        self.assertEqual(deserialized.sender_id, "robot_1")
        self.assertEqual(deserialized.msg_type, MessageType.HEARTBEAT)
        self.assertEqual(deserialized.payload["battery_level"], 0.88)

    def test_vehicle_alert_message(self):
        match = MatchEvent(
            target_id="target_car_alpha",
            plate_number="DXB-77492",
            match_confidence=0.96,
            confirmed=True,
            location=(25.0, 85.0),
            reporting_robot_id="robot_2",
            timestamp=600.0,
            evidence_reference="img_evidence_001.jpg",
        )
        alert_msg = create_vehicle_alert_message("robot_2", 600.1, match)
        self.assertEqual(alert_msg.msg_type, MessageType.VEHICLE_ALERT)
        self.assertEqual(alert_msg.payload["plate_number"], "DXB-77492")
        self.assertTrue(alert_msg.payload["confirmed"])

        # JSON Round trip
        json_str = alert_msg.to_json()
        reconstructed = SwarmMessage.from_json(json_str)
        self.assertEqual(reconstructed.payload["target_id"], "target_car_alpha")


class TestSwarmState(unittest.TestCase):
    """Tests for decentralized SwarmState tracking and failure handling."""

    def test_peer_registration_and_update(self):
        swarm_state = SwarmState(local_robot_id="robot_1", timestamp=100.0)
        pose2 = Pose2D(x=10.0, y=20.0, theta=0.0)

        swarm_state.update_or_add_peer(
            robot_id="robot_2",
            state=RobotState.NAVIGATING,
            pose=pose2,
            timestamp=100.0,
            battery_level=0.95,
        )

        active_peers = swarm_state.get_active_peers()
        self.assertEqual(len(active_peers), 1)
        self.assertEqual(active_peers[0].robot_id, "robot_2")
        self.assertEqual(active_peers[0].state, RobotState.NAVIGATING)

    def test_peer_failure_and_task_release(self):
        swarm_state = SwarmState(local_robot_id="robot_1", timestamp=100.0)
        pose2 = Pose2D(x=10.0, y=20.0, theta=0.0)

        # Create task and assign to robot_2
        task = SearchTask(
            task_id="task_fail_test",
            cluster_id="c_fail",
            centroid_x=12.0,
            centroid_y=22.0,
            bounding_box=(10.0, 20.0, 14.0, 24.0),
            size=15,
        )
        task.mark_assigned("robot_2", timestamp=100.0)
        swarm_state.active_tasks[task.task_id] = task

        swarm_state.update_or_add_peer(
            robot_id="robot_2",
            state=RobotState.NAVIGATING,
            pose=pose2,
            timestamp=100.0,
            assigned_task_id="task_fail_test",
        )

        # Verify task is assigned to robot_2
        assigned = swarm_state.get_robot_assigned_task("robot_2")
        self.assertIsNotNone(assigned)
        self.assertEqual(assigned.task_id, "task_fail_test")

        # Mark robot_2 inactive (failure detected)
        swarm_state.mark_peer_inactive("robot_2")
        self.assertEqual(len(swarm_state.get_active_peers()), 0)

        # Verify task was automatically released and returned to PENDING
        self.assertEqual(task.status, TaskStatus.PENDING)
        self.assertIsNone(task.assigned_robot)

    def test_swarm_state_serialization_round_trip(self):
        swarm_state = SwarmState(
            local_robot_id="robot_1",
            timestamp=150.0,
            global_exploration_ratio=0.45,
            target_found=True,
            confirmed_target_location=Pose2D(x=25.0, y=85.0, theta=1.57),
        )
        swarm_state.update_or_add_peer(
            robot_id="robot_3",
            state=RobotState.SEARCHING,
            pose=Pose2D(15.0, 30.0, 0.5),
            timestamp=150.0,
        )

        data = swarm_state.to_dict()
        reconstructed = SwarmState.from_dict(data)

        self.assertEqual(reconstructed.local_robot_id, "robot_1")
        self.assertEqual(reconstructed.timestamp, 150.0)
        self.assertEqual(reconstructed.global_exploration_ratio, 0.45)
        self.assertTrue(reconstructed.target_found)
        self.assertIsNotNone(reconstructed.confirmed_target_location)
        self.assertEqual(reconstructed.confirmed_target_location.x, 25.0)
        self.assertIn("robot_3", reconstructed.peers)


class TestBaseSLAMProvider(unittest.TestCase):
    """Tests for BaseSLAMProvider caching and health reporting."""

    def test_caching_and_health(self):
        provider = BaseSLAMProvider()
        self.assertFalse(provider.is_healthy("robot_1"))

        with self.assertRaises(KeyError):
            provider.get_latest_state("robot_1")

        dummy_pose = Pose2D(1.0, 2.0, 0.0)
        dummy_grid = OccupancyGrid2D(width=5, height=5, resolution=1.0, origin=dummy_pose)
        state = SLAMState(
            robot_id="robot_1",
            timestamp=10.0,
            pose=dummy_pose,
            linear_velocity=0.0,
            angular_velocity=0.0,
            occupancy_grid=dummy_grid,
        )

        provider.update_cached_state("robot_1", state)
        self.assertTrue(provider.is_healthy("robot_1"))
        retrieved = provider.get_latest_state("robot_1")
        self.assertEqual(retrieved.robot_id, "robot_1")
        self.assertEqual(retrieved.pose.x, 1.0)


if __name__ == "__main__":
    unittest.main()
