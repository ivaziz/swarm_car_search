"""
Integration and Unit Tests for Multi-Robot Coordination and Distributed Task Management.
"""

import unittest
from interfaces.slam_interface import Pose2D, FrontierCluster
from simulation.environment import UrbanEnvironment
from simulation.obstacle_generator import ObstacleGenerator
from slam_providers.mock_slam_provider import MockSLAMProvider
from swarm.states.robot_states import RobotState
from swarm.task_allocation.frontier_task import SearchTask, TaskStatus
from swarm.task_allocation.task_manager import TaskManager
from swarm.task_allocation.allocation_strategy import (
    ACOAllocationStrategy,
    NearestFrontierStrategy,
    RandomAllocationStrategy,
)
from swarm.core.robot_agent import RobotAgent
from swarm.core.swarm_coordinator import SwarmCoordinator
from swarm.aco.aco_engine import ACOEngine
from swarm.aco.aco_config import ACOConfig


class TestTaskManager(unittest.TestCase):
    """Tests for frontier ingestion, spatial deduplication, and pruning in TaskManager."""

    def setUp(self):
        self.tm = TaskManager(deduplication_radius=3.0, default_ttl_seconds=50.0)

    def test_ingestion_and_deduplication(self):
        # Two frontiers far apart
        f1 = FrontierCluster("f1", 10.0, 10.0, 15, (9, 9, 11, 11))
        f2 = FrontierCluster("f2", 25.0, 25.0, 20, (24, 24, 26, 26))
        # Third frontier very close to f1 (distance = 1.0 m < 3.0 m threshold)
        f3_duplicate = FrontierCluster("f3", 10.5, 10.5, 12, (9.5, 9.5, 11.5, 11.5))

        new_tasks = self.tm.ingest_frontiers([f1, f2, f3_duplicate], timestamp=10.0)
        self.assertEqual(len(new_tasks), 2)  # f3 must be discarded as duplicate
        self.assertEqual(len(self.tm.tasks), 2)

    def test_task_lifecycle(self):
        f = FrontierCluster("f1", 15.0, 15.0, 10, (14, 14, 16, 16))
        self.tm.ingest_frontiers([f], timestamp=10.0)

        avail = self.tm.get_available_tasks(current_time=20.0)
        self.assertEqual(len(avail), 1)
        task = avail[0]

        # Assign task
        self.tm.assign_task(task.task_id, "robot_1", timestamp=25.0)
        self.assertEqual(task.status, TaskStatus.ASSIGNED)
        self.assertEqual(len(self.tm.get_available_tasks(current_time=25.0)), 0)

        # Release task
        self.tm.release_task(task.task_id, timestamp=30.0)
        self.assertEqual(task.status, TaskStatus.PENDING)
        self.assertEqual(len(self.tm.get_available_tasks(current_time=30.0)), 1)

        # Complete task
        self.tm.complete_task(task.task_id, timestamp=35.0)
        self.assertEqual(task.status, TaskStatus.COMPLETED)

    def test_task_pruning(self):
        f = FrontierCluster("f1", 10.0, 10.0, 10, (9, 9, 11, 11))
        self.tm.ingest_frontiers([f], timestamp=0.0)
        task = list(self.tm.tasks.values())[0]

        # At time 40, task not expired (TTL=50)
        pruned = self.tm.prune_tasks(current_time=40.0)
        self.assertEqual(pruned, 0)

        # At time 60, task expired (>50)
        pruned = self.tm.prune_tasks(current_time=60.0)
        self.assertEqual(pruned, 1)
        self.assertEqual(len(self.tm.tasks), 0)


class TestAllocationStrategies(unittest.TestCase):
    """Tests comparing ACO, Nearest, and Random allocation strategies."""

    def setUp(self):
        self.robot_pose = Pose2D(10.0, 10.0, 0.0)
        self.task_near = SearchTask("near", "c1", 12.0, 10.0, (11, 9, 13, 11), size=10)
        self.task_far = SearchTask("far", "c2", 30.0, 30.0, (29, 29, 31, 31), size=50)
        self.tasks = [self.task_far, self.task_near]

    def test_nearest_frontier(self):
        strat = NearestFrontierStrategy()
        chosen = strat.allocate_task("r1", self.robot_pose, self.tasks)
        self.assertIsNotNone(chosen)
        self.assertEqual(chosen.task_id, "near")

    def test_aco_allocation(self):
        cfg = ACOConfig()
        cfg.stochastic.deterministic = True
        strat = ACOAllocationStrategy(ACOEngine(cfg))
        chosen = strat.allocate_task("r1", self.robot_pose, self.tasks)
        self.assertIsNotNone(chosen)


class TestSwarmCoordinatorIntegration(unittest.TestCase):
    """Integration test coordinating 3 autonomous robots in an urban environment."""

    def setUp(self):
        self.env = UrbanEnvironment(width=50.0, height=50.0, resolution=1.0)
        ObstacleGenerator.generate_urban_city_blocks(self.env)
        self.slam = MockSLAMProvider(self.env)

        # Deterministic ACO strategy for reproducible coordination test
        cfg = ACOConfig()
        cfg.stochastic.deterministic = True
        strategy = ACOAllocationStrategy(ACOEngine(cfg))

        self.coordinator = SwarmCoordinator(
            allocation_strategy=strategy,
            environment=self.env,
        )

        # Instantiate 3 robots at distinct starting poses
        poses = [
            Pose2D(5.0, 5.0, 0.0),
            Pose2D(45.0, 5.0, 3.14),
            Pose2D(5.0, 45.0, 1.57),
        ]
        for i, pose in enumerate(poses):
            agent = RobotAgent(
                robot_id=f"robot_{i+1}",
                initial_pose=pose,
                slam_provider=self.slam,
                zone_search_duration=1.0,
            )
            self.coordinator.register_robot(agent)

    def test_multi_robot_coordination_loop(self):
        dt = 0.1
        sim_time = 0.0

        # Run coordination loop for 25 simulation ticks
        for tick in range(25):
            sim_time += dt
            status = self.coordinator.step_coordination(timestamp=sim_time, dt=dt)

        self.assertEqual(status["robot_count"], 3)
        self.assertGreater(status["reassignment_events_count"], 0)

        # Verify no duplicate task assignments across robots
        assigned_task_ids = [
            agent.current_task.task_id
            for agent in self.coordinator.robots.values()
            if agent.current_task is not None
        ]
        self.assertEqual(len(assigned_task_ids), len(set(assigned_task_ids)))

        # Verify peer state relays
        r1_peers = self.coordinator.robot_swarm_states["robot_1"].peers
        self.assertIn("robot_2", r1_peers)
        self.assertIn("robot_3", r1_peers)
        self.assertTrue(r1_peers["robot_2"].is_active)


if __name__ == "__main__":
    unittest.main()
