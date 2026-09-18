"""
Comprehensive Unit and Integration Tests for Swarm Communication Channels and Network Simulation.
"""

import unittest
from interfaces.slam_interface import Pose2D
from simulation.environment import UrbanEnvironment
from slam_providers.mock_slam_provider import MockSLAMProvider
from swarm.states.robot_states import RobotState
from swarm.task_allocation.frontier_task import SearchTask
from swarm.communication.swarm_messages import (
    MessageType,
    SwarmMessage,
    create_heartbeat_message,
)
from swarm.communication.local_comm import LocalMessageBus
from simulation.sim_comm import SimulatedNetworkChannel
from swarm.core.robot_agent import RobotAgent


class TestLocalMessageBus(unittest.TestCase):
    """Tests for in-memory synchronous pub/sub message bus."""

    def setUp(self):
        self.bus = LocalMessageBus()
        self.bus.register_node("robot_1")
        self.bus.register_node("robot_2")
        self.bus.register_node("robot_3")

    def test_broadcast(self):
        msg = create_heartbeat_message(
            sender_id="robot_1",
            timestamp=10.0,
            state=RobotState.SEARCHING,
            pose=Pose2D(5.0, 5.0, 0.0),
        )
        self.bus.broadcast(msg)

        # Sender should have empty inbox
        self.assertEqual(len(self.bus.receive_inbox("robot_1")), 0)
        # Other robots must receive the broadcast
        inbox_2 = self.bus.receive_inbox("robot_2")
        inbox_3 = self.bus.receive_inbox("robot_3")
        self.assertEqual(len(inbox_2), 1)
        self.assertEqual(len(inbox_3), 1)
        self.assertEqual(inbox_2[0].sender_id, "robot_1")

        # Inboxes drained after receive
        self.assertEqual(len(self.bus.receive_inbox("robot_2")), 0)

    def test_unicast_send_direct(self):
        msg = SwarmMessage(
            message_id="msg_direct_01",
            sender_id="robot_1",
            timestamp=10.0,
            msg_type=MessageType.TASK_CLAIM,
            payload={"task_id": "task_42"},
        )
        self.bus.send_direct("robot_2", msg)

        self.assertEqual(len(self.bus.receive_inbox("robot_3")), 0)
        inbox_2 = self.bus.receive_inbox("robot_2")
        self.assertEqual(len(inbox_2), 1)
        self.assertEqual(inbox_2[0].message_id, "msg_direct_01")


class TestSimulatedNetworkChannel(unittest.TestCase):
    """Tests for physical range limits, packet loss, and latency queuing."""

    def setUp(self):
        self.net = SimulatedNetworkChannel(
            communication_range=20.0,  # 20 meters limit
            packet_loss_rate=0.0,      # Zero loss initially
            latency_seconds=0.1,       # 100 ms latency
            seed=42,
        )
        self.net.register_node("robot_1")
        self.net.register_node("robot_2")
        self.net.register_node("robot_3")

        # Place robot_1 at (0, 0)
        self.net.update_node_pose("robot_1", 0.0, 0.0)
        # Place robot_2 at (10, 0) -> distance = 10m (< 20m)
        self.net.update_node_pose("robot_2", 10.0, 0.0)
        # Place robot_3 at (35, 0) -> distance = 35m (> 20m, out of range)
        self.net.update_node_pose("robot_3", 35.0, 0.0)

    def test_range_limit_filtering(self):
        msg = create_heartbeat_message("robot_1", 10.0, RobotState.IDLE, Pose2D(0, 0, 0))
        self.net.broadcast(msg)

        # Advance time to allow latency delivery
        self.net.step(10.15)

        # robot_2 (within range) should receive packet
        inbox_2 = self.net.receive_inbox("robot_2")
        self.assertEqual(len(inbox_2), 1)

        # robot_3 (out of range) should not receive packet
        inbox_3 = self.net.receive_inbox("robot_3")
        self.assertEqual(len(inbox_3), 0)
        self.assertGreater(self.net.stats["dropped_range"], 0)

    def test_latency_delay_queue(self):
        msg = create_heartbeat_message("robot_1", 5.0, RobotState.IDLE, Pose2D(0, 0, 0))
        self.net.send_direct("robot_2", msg)

        # At timestamp 5.05 (before 5.0 + 0.10 latency), inbox must be empty
        self.net.step(5.05)
        self.assertEqual(len(self.net.receive_inbox("robot_2")), 0)

        # At timestamp 5.11 (after latency expiry), message is delivered
        self.net.step(5.11)
        inbox = self.net.receive_inbox("robot_2")
        self.assertEqual(len(inbox), 1)

    def test_packet_loss(self):
        lossy_net = SimulatedNetworkChannel(
            communication_range=50.0,
            packet_loss_rate=1.0,  # 100% loss
            latency_seconds=0.0,
            seed=42,
        )
        lossy_net.register_node("robot_1")
        lossy_net.register_node("robot_2")
        lossy_net.update_node_pose("robot_1", 0, 0)
        lossy_net.update_node_pose("robot_2", 5, 5)

        msg = create_heartbeat_message("robot_1", 1.0, RobotState.IDLE, Pose2D(0, 0, 0))
        lossy_net.send_direct("robot_2", msg)
        lossy_net.step(1.0)

        # Must be dropped due to loss
        self.assertEqual(len(lossy_net.receive_inbox("robot_2")), 0)
        self.assertEqual(lossy_net.stats["dropped_loss"], 1)


class TestRobotAgentCommunicationIntegration(unittest.TestCase):
    """Tests for RobotAgent communication timeout and COMMUNICATION_LOST state."""

    def setUp(self):
        self.bus = LocalMessageBus()
        self.env = UrbanEnvironment(width=30, height=30, resolution=1.0)
        self.slam = MockSLAMProvider(self.env)
        self.agent = RobotAgent(
            robot_id="robot_test",
            initial_pose=Pose2D(5.0, 5.0, 0.0),
            slam_provider=self.slam,
            comm_channel=self.bus,
            comm_timeout_sec=2.0,  # 2.0 second timeout for test
            zone_search_duration=10.0,
        )

    def test_communication_lost_and_restored(self):
        # Assign task and move to SEARCHING
        task = SearchTask("t1", "c1", 5.0, 5.0, (4, 4, 6, 6), size=10)
        self.agent.assign_task(task, timestamp=0.0)
        self.agent.fsm.transition_to(RobotState.SEARCHING, timestamp=0.5)

        # Simulate heartbeat arriving from peer at t=1.0
        peer_hb = create_heartbeat_message("robot_peer", 1.0, RobotState.SEARCHING, Pose2D(6, 6, 0))
        self.bus.send_direct("robot_test", peer_hb)

        # Step agent at t=1.0: receives heartbeat
        self.agent.step(timestamp=1.0, dt=0.1, env=self.env)
        self.assertEqual(self.agent.current_state, RobotState.SEARCHING)
        self.assertEqual(self.agent.last_peer_comm_timestamp, 1.0)

        # Advance time past timeout (1.0 + 2.0 = 3.0) to t=3.5 without any new peer messages
        self.agent.step(timestamp=3.5, dt=0.1, env=self.env)
        self.assertEqual(self.agent.current_state, RobotState.COMMUNICATION_LOST)

        # New heartbeat arrives from peer at t=4.0
        new_hb = create_heartbeat_message("robot_peer", 4.0, RobotState.SEARCHING, Pose2D(7, 7, 0))
        self.bus.send_direct("robot_test", new_hb)

        # Step agent at t=4.0: communication restored!
        self.agent.step(timestamp=4.0, dt=0.1, env=self.env)
        # Returns to active navigation or search
        self.assertIn(self.agent.current_state, (RobotState.SEARCHING, RobotState.NAVIGATING))


if __name__ == "__main__":
    unittest.main()
