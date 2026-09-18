"""
Unit and Integration Tests for Swarm Visualization and Simulation Runner.
Ensures zero-dependency ASCII and SVG rendering work cleanly, and tests
Matplotlib dashboard generation when dependencies are present.
"""

import unittest
import os
import shutil
import tempfile

from interfaces.slam_interface import Pose2D
from simulation.environment import UrbanEnvironment
from slam_providers.mock_slam_provider import MockSLAMProvider
from swarm.states.robot_states import RobotState
from swarm.task_allocation.frontier_task import SearchTask
from swarm.core.robot_agent import RobotAgent
from swarm.core.swarm_coordinator import SwarmCoordinator
from perception.vehicle_detector import TargetVehicle
from visualization.swarm_visualizer import (
    SwarmVisualizer,
    STATE_COLOR_MAP,
    _HAS_MATPLOTLIB,
    _HAS_PIL,
)
from scripts.run_simulation import setup_simulation, run_simulation, parse_args


class TestSwarmVisualizer(unittest.TestCase):
    """Tests for ASCII, SVG, and Matplotlib visualizer backends."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp(prefix="swarm_viz_test_")
        self.env = UrbanEnvironment(width=30.0, height=30.0, resolution=1.0)
        self.env.set_cell(5, 5, 100)  # Wall obstacle
        self.slam = MockSLAMProvider(self.env)
        self.coordinator = SwarmCoordinator(environment=self.env)

        self.agent = RobotAgent(
            robot_id="robot_viz_1",
            initial_pose=Pose2D(10.0, 10.0, 0.0),
            slam_provider=self.slam,
        )
        self.coordinator.register_robot(self.agent)

        self.target = TargetVehicle(
            target_id="target_car",
            plate_number="DXB-12345",
            vehicle_class="sedan",
            color="black",
            position=(20.0, 20.0),
        )
        self.visualizer = SwarmVisualizer(self.env, title="Test Visualizer")

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_state_color_mapping(self):
        """Ensures all RobotState enum members have valid HEX color mappings."""
        for state in RobotState:
            self.assertIn(state, STATE_COLOR_MAP)
            self.assertTrue(STATE_COLOR_MAP[state].startswith("#"))

    def test_ascii_map_rendering(self):
        """Verifies pure-Python ASCII rendering produces complete grid and legend."""
        ascii_out = self.visualizer.render_ascii_map(self.coordinator, timestamp=5.0, cols=30, rows=15)
        self.assertIsInstance(ascii_out, str)
        self.assertIn("=== Swarm Urban Search Map", ascii_out)
        self.assertIn("Legend:", ascii_out)
        self.assertIn("R", ascii_out)  # Robot glyph
        self.assertIn("#", ascii_out)  # Obstacle glyph

    def test_svg_vector_rendering(self):
        """Verifies pure-Python SVG generator produces valid vector graphic file."""
        svg_file = os.path.join(self.temp_dir, "test_map.svg")
        svg_str = self.visualizer.render_svg_map(
            self.coordinator,
            timestamp=2.5,
            filepath=svg_file,
            target_vehicles=[self.target],
        )
        self.assertIsInstance(svg_str, str)
        self.assertTrue(svg_str.startswith("<svg"))
        self.assertTrue(svg_str.strip().endswith("</svg>"))
        self.assertIn("robot_viz_1", svg_str)
        self.assertIn("DXB-12345", svg_str)

        # File written to disk
        self.assertTrue(os.path.exists(svg_file))
        self.assertGreater(os.path.getsize(svg_file), 200)

    def test_trajectory_recording(self):
        """Verifies path history accumulation over multiple coordination ticks."""
        self.agent.sim_robot.set_velocity(linear=1.0, angular=0.0)
        for t in range(5):
            self.coordinator.step_coordination(timestamp=t * 0.2, dt=0.2)
            self.visualizer.update_trajectories(self.coordinator, timestamp=t * 0.2)

        trail = self.visualizer.trajectories.get("robot_viz_1", [])
        self.assertGreaterEqual(len(trail), 5)
        # Verify displacement occurred
        self.assertNotEqual(trail[0], trail[-1])

    def test_matplotlib_dashboard_generation(self):
        """Verifies Matplotlib dashboard generation if library is available, or clean skip."""
        if not _HAS_MATPLOTLIB:
            res = self.visualizer.render_matplotlib_dashboard(self.coordinator, timestamp=1.0)
            self.assertIsNone(res)
            return

        png_file = os.path.join(self.temp_dir, "test_dashboard.png")
        out_path = self.visualizer.render_matplotlib_dashboard(
            self.coordinator,
            timestamp=1.0,
            target_vehicles=[self.target],
            filepath=png_file,
            dpi=80,
        )
        self.assertEqual(out_path, png_file)
        self.assertTrue(os.path.exists(png_file))
        self.assertGreater(os.path.getsize(png_file), 5000)


class TestSimulationRunner(unittest.TestCase):
    """End-to-end integration test for scripts/run_simulation.py."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp(prefix="sim_runner_test_")

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_short_simulation_execution(self):
        """Runs a fast 15-step simulation and verifies output generation."""
        class MockArgs:
            width = 30.0
            height = 30.0
            resolution = 1.0
            robots = 2
            steps = 15
            dt = 0.2
            target_x = 22.0
            target_y = 22.0
            target_plate = "TEST-999"
            inject_failure_tick = 8
            fail_robot = "robot_1"
            output_dir = self.temp_dir
            render_every = 5
            seed = 123

        args = MockArgs()
        run_simulation(args)

        # Output dir must contain generated SVG frames and final summary
        final_svg = os.path.join(self.temp_dir, "final_mission_summary.svg")
        self.assertTrue(os.path.exists(final_svg))
        self.assertGreater(os.path.getsize(final_svg), 500)

        # Check frame_0005.svg was generated
        frame_svg = os.path.join(self.temp_dir, "frame_0005.svg")
        self.assertTrue(os.path.exists(frame_svg))


if __name__ == "__main__":
    unittest.main()
