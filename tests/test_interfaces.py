"""
Unit tests validating interface compliance, abstract class constraints, and dataclass schemas.
"""

import unittest
from interfaces.slam_interface import (
    Pose2D,
    OccupancyGrid2D,
    FrontierCluster,
    SLAMState,
    ISLAMProvider,
)
from interfaces.perception_interface import (
    BoundingBox2D,
    VehicleDetectionEvent,
    PlateDetectionEvent,
    MatchEvent,
    IPerceptionProvider,
)
from interfaces.server_interface import (
    DetectionReport,
    MissionReport,
    IServerReporter,
)


class TestInterfaces(unittest.TestCase):
    """Validates structural integrity and constraints of abstract interfaces and data models."""

    def test_pose2d_immutability(self):
        pose = Pose2D(x=10.5, y=20.0, theta=1.57)
        self.assertEqual(pose.x, 10.5)
        self.assertEqual(pose.y, 20.0)
        self.assertEqual(pose.theta, 1.57)
        with self.assertRaises(Exception):
            # Pose2D is frozen
            pose.x = 5.0

    def test_occupancy_grid(self):
        origin = Pose2D(0.0, 0.0, 0.0)
        grid = OccupancyGrid2D(
            width=100,
            height=50,
            resolution=0.5,
            origin=origin,
            data=[-1] * 5000,
        )
        self.assertEqual(grid.total_cells, 5000)
        self.assertEqual(len(grid.data), 5000)

    def test_slam_state_instantiation(self):
        pose = Pose2D(x=5.0, y=5.0, theta=0.0)
        grid = OccupancyGrid2D(width=10, height=10, resolution=1.0, origin=pose, data=[0] * 100)
        frontier = FrontierCluster(
            cluster_id="f_001",
            centroid_x=7.5,
            centroid_y=8.0,
            size=14,
            bounding_box=(6.0, 7.0, 9.0, 9.0),
        )
        state = SLAMState(
            robot_id="robot_1",
            timestamp=1000.0,
            pose=pose,
            linear_velocity=0.5,
            angular_velocity=0.1,
            occupancy_grid=grid,
            frontiers=[frontier],
            exploration_ratio=0.15,
        )
        self.assertEqual(state.robot_id, "robot_1")
        self.assertEqual(len(state.frontiers), 1)
        self.assertEqual(state.frontiers[0].cluster_id, "f_001")

    def test_abstract_class_instantiation_prohibited(self):
        """Ensure abstract base classes cannot be instantiated without implementing methods."""
        with self.assertRaises(TypeError):
            ISLAMProvider()

        with self.assertRaises(TypeError):
            IPerceptionProvider()

        with self.assertRaises(TypeError):
            IServerReporter()

    def test_perception_events(self):
        bbox = BoundingBox2D(ymin=0.2, xmin=0.3, ymax=0.8, xmax=0.7)
        v_event = VehicleDetectionEvent(
            robot_id="robot_1",
            timestamp=1000.5,
            vehicle_class="sedan",
            confidence=0.92,
            bounding_box=bbox,
            estimated_global_position=(25.0, 85.0),
        )
        self.assertEqual(v_event.vehicle_class, "sedan")

        p_event = PlateDetectionEvent(
            robot_id="robot_1",
            timestamp=1001.0,
            plate_number="DXB-77492",
            ocr_confidence=0.95,
            estimated_global_position=(25.0, 85.0),
        )
        self.assertEqual(p_event.plate_number, "DXB-77492")

        m_event = MatchEvent(
            target_id="target_car_alpha",
            plate_number="DXB-77492",
            match_confidence=0.98,
            confirmed=True,
            location=(25.0, 85.0),
            reporting_robot_id="robot_1",
            timestamp=1001.2,
        )
        self.assertTrue(m_event.confirmed)


if __name__ == "__main__":
    unittest.main()
