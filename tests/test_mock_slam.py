"""
Unit and integration tests for Urban Environment, Obstacle Generator, and MockSLAMProvider.
"""

import unittest
from interfaces.slam_interface import Pose2D, ISLAMProvider
from simulation.environment import UrbanEnvironment, UrbanZone
from simulation.obstacle_generator import ObstacleGenerator
from slam_providers.mock_slam_provider import MockSLAMProvider


class TestUrbanEnvironment(unittest.TestCase):
    """Tests for discrete 2D environment physics and raycasting."""

    def setUp(self):
        self.env = UrbanEnvironment(width=50.0, height=50.0, resolution=1.0)

    def test_dimensions_and_conversion(self):
        self.assertEqual(self.env.grid_width, 50)
        self.assertEqual(self.env.grid_height, 50)

        # Coordinate round-trip
        gx, gy = self.env.world_to_grid(10.2, 20.8)
        self.assertEqual((gx, gy), (10, 20))
        wx, wy = self.env.grid_to_world(gx, gy)
        self.assertAlmostEqual(wx, 10.5)
        self.assertAlmostEqual(wy, 20.5)

        # Out of bounds
        self.assertIsNone(self.env.world_to_grid(-1.0, 10.0))
        self.assertIsNone(self.env.world_to_grid(60.0, 10.0))

    def test_raycasting_free_and_blocked(self):
        # Raycast in completely open space
        observed = self.env.raycast_2d(origin_x=25.0, origin_y=25.0, max_range=5.0, num_rays=16)
        self.assertTrue(len(observed) > 0)
        for gx, gy, val in observed:
            self.assertEqual(val, 0)  # All free space

        # Place an obstacle block at (28, 25)
        self.env.set_cell(28, 25, 100)
        observed_blocked = self.env.raycast_2d(origin_x=25.0, origin_y=25.0, max_range=6.0, num_rays=36)
        obstacle_hits = [c for c in observed_blocked if c[2] == 100]
        self.assertTrue(len(obstacle_hits) > 0)

    def test_zone_prior_probability(self):
        zone = UrbanZone("parking", (10.0, 10.0, 20.0, 20.0), vehicle_prior_probability=0.85)
        self.env.add_zone(zone)
        self.assertEqual(self.env.get_zone_prior(15.0, 15.0), 0.85)
        self.assertEqual(self.env.get_zone_prior(35.0, 35.0), 0.30)  # default


class TestObstacleGenerator(unittest.TestCase):
    """Tests for procedural city layouts and barrier synthesis."""

    def test_perimeter_and_blocks(self):
        env = UrbanEnvironment(width=40.0, height=40.0, resolution=1.0)
        ObstacleGenerator.add_perimeter_walls(env, thickness_cells=1)

        # Corners and edges must be 100
        self.assertEqual(env.get_cell(0, 0), 100)
        self.assertEqual(env.get_cell(39, 39), 100)
        # Center should remain 0
        self.assertEqual(env.get_cell(20, 20), 0)

        # Add building block
        ObstacleGenerator.add_box_obstacle(env, 10.0, 10.0, 15.0, 15.0)
        self.assertEqual(env.get_cell(12, 12), 100)


class TestMockSLAMProvider(unittest.TestCase):
    """Tests for incremental SLAM state simulation and frontier extraction."""

    def setUp(self):
        self.env = UrbanEnvironment(width=50.0, height=50.0, resolution=1.0)
        ObstacleGenerator.generate_urban_city_blocks(self.env)
        self.slam = MockSLAMProvider(self.env, min_cluster_size=2)

    def test_contract_compliance(self):
        self.assertIsInstance(self.slam, ISLAMProvider)
        self.assertFalse(self.slam.is_healthy("robot_1"))

    def test_incremental_exploration_and_frontiers(self):
        pose1 = Pose2D(x=5.0, y=5.0, theta=0.0)

        # Initial sweep
        state1 = self.slam.update_robot_pose("robot_1", pose1, timestamp=10.0, sensor_range=10.0)

        self.assertTrue(self.slam.is_healthy("robot_1"))
        self.assertEqual(state1.robot_id, "robot_1")
        self.assertGreater(state1.exploration_ratio, 0.0)
        self.assertLess(state1.exploration_ratio, 1.0)

        # Frontier extraction check
        self.assertGreater(len(state1.frontiers), 0)
        first_frontier = state1.frontiers[0]
        self.assertGreaterEqual(first_frontier.size, 2)
        self.assertGreater(first_frontier.centroid_x, 0.0)
        self.assertLess(first_frontier.centroid_x, 50.0)

        # Move robot to new location (5, 5) -> (10, 5)
        pose2 = Pose2D(x=10.0, y=5.0, theta=0.0)
        state2 = self.slam.update_robot_pose("robot_1", pose2, timestamp=20.0, sensor_range=10.0)

        # Exploration ratio must monotonically increase
        self.assertGreater(state2.exploration_ratio, state1.exploration_ratio)

        # Verify get_latest_state reflects updated state
        latest = self.slam.get_latest_state("robot_1")
        self.assertEqual(latest.timestamp, 20.0)
        self.assertEqual(latest.pose.x, 10.0)


if __name__ == "__main__":
    unittest.main()
