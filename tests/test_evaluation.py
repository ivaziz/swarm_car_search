"""
Unit and Integration Tests for Swarm Evaluation Metrics and Benchmark Framework.
"""

import unittest
import os
import shutil
import tempfile

from interfaces.slam_interface import Pose2D
from simulation.environment import UrbanEnvironment
from slam_providers.mock_slam_provider import MockSLAMProvider
from swarm.states.robot_states import RobotState
from swarm.task_allocation.frontier_task import SearchTask, TaskStatus
from swarm.core.robot_agent import RobotAgent
from swarm.core.swarm_coordinator import SwarmCoordinator
from evaluation.baselines import (
    StrategyType,
    create_strategy,
    ACOAllocationStrategy,
    NearestFrontierStrategy,
    RandomAllocationStrategy,
)
from evaluation.metrics_collector import MetricsCollector, MissionMetrics
from scripts.run_evaluation import (
    run_single_trial,
    aggregate_strategy_metrics,
    generate_markdown_report,
    generate_svg_comparison_plots,
)


class TestStrategyFactory(unittest.TestCase):
    """Tests for baseline strategy instantiation."""

    def test_create_strategies(self):
        s_aco = create_strategy(StrategyType.ACO)
        self.assertIsInstance(s_aco, ACOAllocationStrategy)

        s_nf = create_strategy(StrategyType.NEAREST_FRONTIER)
        self.assertIsInstance(s_nf, NearestFrontierStrategy)

        s_rnd = create_strategy(StrategyType.RANDOM, seed=42)
        self.assertIsInstance(s_rnd, RandomAllocationStrategy)

    def test_invalid_strategy(self):
        with self.assertRaises(ValueError):
            create_strategy("INVALID_UNKNOWN_STRATEGY")


class TestMetricsCollector(unittest.TestCase):
    """Tests for quantitative metrics computation and overlap analysis."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp(prefix="metrics_test_")
        self.env = UrbanEnvironment(width=20.0, height=20.0, resolution=1.0)
        self.slam = MockSLAMProvider(self.env)
        self.coordinator = SwarmCoordinator(environment=self.env)

        self.agent_1 = RobotAgent("r1", Pose2D(5.0, 5.0, 0.0), self.slam)
        self.agent_2 = RobotAgent("r2", Pose2D(5.0, 5.0, 0.0), self.slam)
        self.coordinator.register_robot(self.agent_1)
        self.coordinator.register_robot(self.agent_2)

        self.collector = MetricsCollector(
            environment=self.env,
            strategy_name="ACO",
            trial_seed=42,
            spatial_bin_size=1.0,
        )

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_overlap_ratio_calculation(self):
        # Initial step: both robots at (5.0, 5.0) -> exactly 1 cell visited by 2 robots
        self.collector.step(self.coordinator, timestamp=1.0)
        metrics = self.collector.compute_metrics(self.coordinator)

        self.assertEqual(metrics.unique_cells_visited, 1)
        self.assertEqual(metrics.redundant_visits, 1)
        self.assertAlmostEqual(metrics.overlap_ratio, 0.5)  # (2 - 1) / 2 = 0.5

        # Move robot 2 far away to (15.0, 15.0) -> disjoint cell
        self.agent_2.sim_robot.pose = Pose2D(15.0, 15.0, 0.0)
        self.collector.step(self.coordinator, timestamp=2.0)
        metrics = self.collector.compute_metrics(self.coordinator)

        # Now unique cells: (5,5) and (15,15) = 2. Total visits: (5,5)->{r1, r2}, (15,15)->{r2} = 3 visits
        self.assertEqual(metrics.unique_cells_visited, 2)
        self.assertAlmostEqual(metrics.overlap_ratio, 1.0 / 3.0)

    def test_distance_and_energy_accumulation(self):
        # Step robot 1 forward
        self.agent_1.sim_robot.set_velocity(linear=1.0, angular=0.0)
        for t in range(5):
            self.coordinator.step_coordination(timestamp=t * 0.2, dt=0.2)
            self.collector.step(self.coordinator, timestamp=t * 0.2)

        metrics = self.collector.compute_metrics(self.coordinator)
        self.assertGreater(metrics.total_distance_traveled, 0.0)
        self.assertIn("r1", metrics.per_robot_distance)
        self.assertGreater(metrics.per_robot_distance["r1"], 0.0)

    def test_json_serialization(self):
        self.collector.step(self.coordinator, timestamp=1.0)
        json_file = os.path.join(self.temp_dir, "metrics.json")
        self.collector.save_json(self.coordinator, json_file)

        self.assertTrue(os.path.exists(json_file))
        self.assertGreater(os.path.getsize(json_file), 50)


class TestBenchmarkExecution(unittest.TestCase):
    """Integration test for single trial runner and report generator."""

    def test_single_trial_and_aggregation(self):
        class MockArgs:
            trials = 2
            steps = 15
            dt = 0.2
            robots = 2
            width = 30.0
            height = 30.0
            target_x = 20.0
            target_y = 20.0
            target_plate = "TEST-PLATE"
            inject_failure_tick = 8
            fail_robot = "robot_1"
            seed_start = 100

        args = MockArgs()
        m1 = run_single_trial(StrategyType.ACO, seed=101, args=args)
        m2 = run_single_trial(StrategyType.ACO, seed=102, args=args)

        self.assertIsInstance(m1, MissionMetrics)
        self.assertEqual(m1.strategy_name, "ACO")
        self.assertGreater(m1.final_exploration_ratio, 0.0)

        # Test Aggregation
        agg = aggregate_strategy_metrics([m1, m2])
        self.assertEqual(agg["strategy_name"], "ACO")
        self.assertEqual(agg["trials_count"], 2)
        self.assertIn("mean_coverage_pct", agg)

        # Test Report Generation
        results = {"ACO": agg}
        report = generate_markdown_report(results, args)
        self.assertIn("# Autonomous Swarm Car Search: Task Allocation Benchmark Report", report)
        self.assertIn("ACO (Proposed)", report)

        # Test Vector SVG generation
        temp_svg = tempfile.mktemp(suffix=".svg")
        generate_svg_comparison_plots(results, temp_svg)
        self.assertTrue(os.path.exists(temp_svg))
        os.remove(temp_svg)


if __name__ == "__main__":
    unittest.main()
