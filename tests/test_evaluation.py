"""
Unit and Integration Tests for Swarm Evaluation Metrics and Benchmark Framework.
"""

import unittest
import os
import json
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
from evaluation.ai_benchmarks import ScenarioBenchmarkResult, ComprehensiveAIBenchmarkReport
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

    def test_single_robot_single_visit(self):
        """Validates that a single robot visiting one bin produces zero inter-robot overlap and zero revisits."""
        coord = SwarmCoordinator(environment=self.env)
        a1 = RobotAgent("r1", Pose2D(5.0, 5.0, 0.0), self.slam)
        coord.register_robot(a1)
        collector = MetricsCollector(environment=self.env, strategy_name="ACO", spatial_bin_size=1.0)

        collector.step(coord, timestamp=1.0)
        m = collector.compute_metrics(coord)

        self.assertEqual(m.unique_cells_visited, 1)
        self.assertEqual(m.redundant_visits, 0)
        self.assertAlmostEqual(m.overlap_ratio, 0.0)
        self.assertAlmostEqual(m.inter_robot_overlap_ratio, 0.0)
        self.assertEqual(m.shared_cells_count, 0)
        self.assertAlmostEqual(m.shared_cells_ratio, 0.0)
        self.assertEqual(m.total_cell_visits, 1)
        self.assertEqual(m.revisit_events_count, 0)
        self.assertAlmostEqual(m.temporal_revisit_rate, 0.0)
        self.assertEqual(m.intra_robot_revisits, 0)

    def test_single_robot_revisit(self):
        """
        Validates that a single robot revisiting an earlier bin has zero inter-robot overlap
        but a non-zero temporal revisit rate.
        """
        coord = SwarmCoordinator(environment=self.env)
        a1 = RobotAgent("r1", Pose2D(5.0, 5.0, 0.0), self.slam)
        coord.register_robot(a1)
        collector = MetricsCollector(environment=self.env, strategy_name="ACO", spatial_bin_size=1.0)

        # Step 1: at (5.0, 5.0) -> bin (5, 5)
        collector.step(coord, timestamp=1.0)
        # Step 2: moves to (6.0, 5.0) -> bin (6, 5)
        a1.sim_robot.pose = Pose2D(6.0, 5.0, 0.0)
        collector.step(coord, timestamp=2.0)
        # Step 3: returns to (5.0, 5.0) -> bin (5, 5) [revisit]
        a1.sim_robot.pose = Pose2D(5.0, 5.0, 0.0)
        collector.step(coord, timestamp=3.0)

        m = collector.compute_metrics(coord)

        # Spatial inter-robot overlap must be 0 (only 1 robot exists)
        self.assertEqual(m.unique_cells_visited, 2)
        self.assertEqual(m.redundant_visits, 0)
        self.assertAlmostEqual(m.overlap_ratio, 0.0)
        self.assertAlmostEqual(m.inter_robot_overlap_ratio, 0.0)
        self.assertEqual(m.shared_cells_count, 0)
        self.assertAlmostEqual(m.shared_cells_ratio, 0.0)

        # Temporal transitions: 3 entries into 2 unique cells -> 1 revisit
        self.assertEqual(m.total_cell_visits, 3)
        self.assertEqual(m.revisit_events_count, 1)
        self.assertAlmostEqual(m.temporal_revisit_rate, 1.0 / 3.0)
        self.assertEqual(m.intra_robot_revisits, 1)

    def test_two_robots_visiting_same_bin(self):
        """Validates that 2 robots in the same bin produce inter-robot overlap and temporal revisit."""
        coord = SwarmCoordinator(environment=self.env)
        a1 = RobotAgent("r1", Pose2D(5.0, 5.0, 0.0), self.slam)
        a2 = RobotAgent("r2", Pose2D(5.0, 5.0, 0.0), self.slam)
        coord.register_robot(a1)
        coord.register_robot(a2)
        collector = MetricsCollector(environment=self.env, strategy_name="ACO", spatial_bin_size=1.0)

        collector.step(coord, timestamp=1.0)
        m = collector.compute_metrics(coord)

        # Inter-robot: 1 unique cell visited by 2 robots -> redundant=1, ratio=1/2=0.5
        self.assertEqual(m.unique_cells_visited, 1)
        self.assertEqual(m.redundant_visits, 1)
        self.assertAlmostEqual(m.overlap_ratio, 0.5)
        self.assertAlmostEqual(m.inter_robot_overlap_ratio, 0.5)
        self.assertEqual(m.shared_cells_count, 1)
        self.assertAlmostEqual(m.shared_cells_ratio, 1.0)

        # Temporal: 2 entries into 1 cell -> 1 revisit
        self.assertEqual(m.total_cell_visits, 2)
        self.assertEqual(m.revisit_events_count, 1)
        self.assertAlmostEqual(m.temporal_revisit_rate, 0.5)
        self.assertEqual(m.intra_robot_revisits, 0)

    def test_two_robots_visiting_disjoint_bins(self):
        """Validates that 2 robots in distinct bins produce 0 overlap and 0 revisits."""
        coord = SwarmCoordinator(environment=self.env)
        a1 = RobotAgent("r1", Pose2D(5.0, 5.0, 0.0), self.slam)
        a2 = RobotAgent("r2", Pose2D(15.0, 15.0, 0.0), self.slam)
        coord.register_robot(a1)
        coord.register_robot(a2)
        collector = MetricsCollector(environment=self.env, strategy_name="ACO", spatial_bin_size=1.0)

        collector.step(coord, timestamp=1.0)
        m = collector.compute_metrics(coord)

        self.assertEqual(m.unique_cells_visited, 2)
        self.assertEqual(m.redundant_visits, 0)
        self.assertAlmostEqual(m.overlap_ratio, 0.0)
        self.assertAlmostEqual(m.inter_robot_overlap_ratio, 0.0)
        self.assertEqual(m.shared_cells_count, 0)
        self.assertAlmostEqual(m.shared_cells_ratio, 0.0)

        self.assertEqual(m.total_cell_visits, 2)
        self.assertEqual(m.revisit_events_count, 0)
        self.assertAlmostEqual(m.temporal_revisit_rate, 0.0)
        self.assertEqual(m.intra_robot_revisits, 0)

    def test_multiple_robots_with_shared_and_unshared_bins(self):
        """Validates complex fleet scenario with 3 robots having both shared and unshared bins."""
        coord = SwarmCoordinator(environment=self.env)
        a1 = RobotAgent("r1", Pose2D(1.0, 1.0, 0.0), self.slam)
        a2 = RobotAgent("r2", Pose2D(2.0, 2.0, 0.0), self.slam)
        a3 = RobotAgent("r3", Pose2D(3.0, 3.0, 0.0), self.slam)
        coord.register_robot(a1)
        coord.register_robot(a2)
        coord.register_robot(a3)
        collector = MetricsCollector(environment=self.env, strategy_name="ACO", spatial_bin_size=1.0)

        # Step 1:
        # r1 at (1,1), r2 at (2,2), r3 at (3,3)
        collector.step(coord, timestamp=1.0)

        # Step 2:
        # r1 moves to (2,2) [now sharing with r2's previous bin]
        # r2 moves to (4,4)
        # r3 moves to (5,5)
        a1.sim_robot.pose = Pose2D(2.0, 2.0, 0.0)
        a2.sim_robot.pose = Pose2D(4.0, 4.0, 0.0)
        a3.sim_robot.pose = Pose2D(5.0, 5.0, 0.0)
        collector.step(coord, timestamp=2.0)

        m = collector.compute_metrics(coord)

        # Unique cells across fleet: (1,1), (2,2), (3,3), (4,4), (5,5) = 5
        self.assertEqual(m.unique_cells_visited, 5)
        # Total robot visits: r1 visited {(1,1),(2,2)}=2, r2 visited {(2,2),(4,4)}=2, r3 visited {(3,3),(5,5)}=2 -> 6
        # Redundant robot visits = 6 - 5 = 1 (cell (2,2) visited by r1 and r2)
        self.assertEqual(m.redundant_visits, 1)
        self.assertAlmostEqual(m.overlap_ratio, 1.0 / 6.0)
        self.assertAlmostEqual(m.inter_robot_overlap_ratio, 1.0 / 6.0)
        # Shared cells: only (2,2) visited by >=2 robots -> 1 cell
        self.assertEqual(m.shared_cells_count, 1)
        self.assertAlmostEqual(m.shared_cells_ratio, 1.0 / 5.0)

        # Total entry events: 3 robots * 2 ticks = 6
        self.assertEqual(m.total_cell_visits, 6)
        self.assertEqual(m.revisit_events_count, 1)
        self.assertAlmostEqual(m.temporal_revisit_rate, 1.0 / 6.0)
        self.assertEqual(m.intra_robot_revisits, 0)

    def test_empty_input_and_zero_denominators(self):
        """Validates that compute_metrics handles empty simulation / zero denominators without crashing."""
        coord = SwarmCoordinator(environment=self.env)
        collector = MetricsCollector(environment=self.env, strategy_name="ACO", spatial_bin_size=1.0)

        m = collector.compute_metrics(coord)

        self.assertEqual(m.unique_cells_visited, 0)
        self.assertEqual(m.redundant_visits, 0)
        self.assertAlmostEqual(m.overlap_ratio, 0.0)
        self.assertAlmostEqual(m.inter_robot_overlap_ratio, 0.0)
        self.assertEqual(m.shared_cells_count, 0)
        self.assertAlmostEqual(m.shared_cells_ratio, 0.0)
        self.assertEqual(m.total_cell_visits, 0)
        self.assertEqual(m.revisit_events_count, 0)
        self.assertAlmostEqual(m.temporal_revisit_rate, 0.0)
        self.assertEqual(m.intra_robot_revisits, 0)
        self.assertAlmostEqual(m.total_distance_traveled, 0.0)
        self.assertAlmostEqual(m.exploration_rate_m2_per_sec, 0.0)

    def test_scenario_a_one_robot_enters_A_B_A(self):
        """Scenario A: One robot enters A, then B, then A (intra-robot self-revisit)."""
        coord = SwarmCoordinator(environment=self.env)
        a1 = RobotAgent("r1", Pose2D(5.0, 5.0, 0.0), self.slam)
        coord.register_robot(a1)
        collector = MetricsCollector(environment=self.env, strategy_name="ACO", spatial_bin_size=1.0)

        # Step 1: at (5.0, 5.0) -> bin A=(5,5)
        collector.step(coord, timestamp=1.0)
        # Step 2: moves to (6.0, 5.0) -> bin B=(6,5)
        a1.sim_robot.pose = Pose2D(6.0, 5.0, 0.0)
        collector.step(coord, timestamp=2.0)
        # Step 3: returns to (5.0, 5.0) -> bin A=(5,5)
        a1.sim_robot.pose = Pose2D(5.0, 5.0, 0.0)
        collector.step(coord, timestamp=3.0)

        m = collector.compute_metrics(coord)

        self.assertEqual(m.unique_cells_visited, 2)
        self.assertEqual(m.redundant_visits, 0)
        self.assertAlmostEqual(m.inter_robot_overlap_ratio, 0.0)
        self.assertEqual(m.total_cell_visits, 3)
        self.assertEqual(m.revisit_events_count, 1)
        self.assertAlmostEqual(m.temporal_revisit_rate, 1.0 / 3.0)
        self.assertEqual(m.intra_robot_revisits, 1)
        self.assertAlmostEqual(m.intra_robot_self_revisit_rate, 1.0 / 3.0)
        self.assertAlmostEqual(m.per_robot_self_revisit_rate["r1"], 1.0 / 3.0)
        # Identity: 1 non-novel = 1 intra + 0 inter
        self.assertEqual(m.revisit_events_count, m.intra_robot_revisits + m.redundant_visits)

    def test_scenario_b_two_robots_independently_enter_A(self):
        """Scenario B: Two robots independently enter A initially (inter-robot overlap without intra self-revisit)."""
        coord = SwarmCoordinator(environment=self.env)
        a1 = RobotAgent("r1", Pose2D(5.0, 5.0, 0.0), self.slam)
        a2 = RobotAgent("r2", Pose2D(5.0, 5.0, 0.0), self.slam)
        coord.register_robot(a1)
        coord.register_robot(a2)
        collector = MetricsCollector(environment=self.env, strategy_name="ACO", spatial_bin_size=1.0)

        collector.step(coord, timestamp=1.0)
        m = collector.compute_metrics(coord)

        self.assertEqual(m.unique_cells_visited, 1)
        self.assertEqual(m.redundant_visits, 1)
        self.assertAlmostEqual(m.inter_robot_overlap_ratio, 0.5)
        self.assertEqual(m.total_cell_visits, 2)
        self.assertEqual(m.revisit_events_count, 1)
        self.assertAlmostEqual(m.temporal_revisit_rate, 0.5)
        self.assertEqual(m.intra_robot_revisits, 0)
        self.assertAlmostEqual(m.intra_robot_self_revisit_rate, 0.0)
        self.assertAlmostEqual(m.per_robot_self_revisit_rate["r1"], 0.0)
        self.assertAlmostEqual(m.per_robot_self_revisit_rate["r2"], 0.0)
        # Identity: 1 non-novel = 0 intra + 1 inter
        self.assertEqual(m.revisit_events_count, m.intra_robot_revisits + m.redundant_visits)

    def test_scenario_c_robot1_enters_A_robot2_enters_A_later(self):
        """Scenario C: Robot 1 enters A at t=1, and Robot 2 enters A later at t=2."""
        coord = SwarmCoordinator(environment=self.env)
        a1 = RobotAgent("r1", Pose2D(5.0, 5.0, 0.0), self.slam)
        a2 = RobotAgent("r2", Pose2D(15.0, 15.0, 0.0), self.slam)
        coord.register_robot(a1)
        coord.register_robot(a2)
        collector = MetricsCollector(environment=self.env, strategy_name="ACO", spatial_bin_size=1.0)

        # Tick 1: r1 in A=(5,5), r2 in B=(15,15)
        collector.step(coord, timestamp=1.0)
        # Tick 2: r2 enters A=(5,5)
        a2.sim_robot.pose = Pose2D(5.0, 5.0, 0.0)
        collector.step(coord, timestamp=2.0)

        m = collector.compute_metrics(coord)

        self.assertEqual(m.unique_cells_visited, 2)
        self.assertEqual(m.redundant_visits, 1)
        self.assertAlmostEqual(m.inter_robot_overlap_ratio, 1.0 / 3.0)
        self.assertEqual(m.total_cell_visits, 3)
        self.assertEqual(m.revisit_events_count, 1)
        self.assertAlmostEqual(m.temporal_revisit_rate, 1.0 / 3.0)
        self.assertEqual(m.intra_robot_revisits, 0)
        self.assertAlmostEqual(m.intra_robot_self_revisit_rate, 0.0)
        # Identity: 1 non-novel = 0 intra + 1 inter
        self.assertEqual(m.revisit_events_count, m.intra_robot_revisits + m.redundant_visits)

    def test_scenario_d_robot1_enters_A_B_robot2_enters_B_C(self):
        """Scenario D: Robot 1 enters A and B, while Robot 2 enters B and C."""
        coord = SwarmCoordinator(environment=self.env)
        a1 = RobotAgent("r1", Pose2D(5.0, 5.0, 0.0), self.slam)
        a2 = RobotAgent("r2", Pose2D(6.0, 5.0, 0.0), self.slam)
        coord.register_robot(a1)
        coord.register_robot(a2)
        collector = MetricsCollector(environment=self.env, strategy_name="ACO", spatial_bin_size=1.0)

        # Tick 1: r1 at A=(5,5), r2 at B=(6,5)
        collector.step(coord, timestamp=1.0)
        # Tick 2: r1 moves to B=(6,5), r2 moves to C=(7,5)
        a1.sim_robot.pose = Pose2D(6.0, 5.0, 0.0)
        a2.sim_robot.pose = Pose2D(7.0, 5.0, 0.0)
        collector.step(coord, timestamp=2.0)

        m = collector.compute_metrics(coord)

        self.assertEqual(m.unique_cells_visited, 3)
        self.assertEqual(m.redundant_visits, 1)
        self.assertAlmostEqual(m.inter_robot_overlap_ratio, 1.0 / 4.0)
        self.assertEqual(m.total_cell_visits, 4)
        self.assertEqual(m.revisit_events_count, 1)
        self.assertAlmostEqual(m.temporal_revisit_rate, 0.25)
        self.assertEqual(m.intra_robot_revisits, 0)
        self.assertAlmostEqual(m.intra_robot_self_revisit_rate, 0.0)
        # Identity: 1 non-novel = 0 intra + 1 inter
        self.assertEqual(m.revisit_events_count, m.intra_robot_revisits + m.redundant_visits)

    def test_scenario_e_stationary_robot_in_A(self):
        """Scenario E: Robot remains stationary in A across several simulation ticks."""
        coord = SwarmCoordinator(environment=self.env)
        a1 = RobotAgent("r1", Pose2D(5.0, 5.0, 0.0), self.slam)
        coord.register_robot(a1)
        collector = MetricsCollector(environment=self.env, strategy_name="ACO", spatial_bin_size=1.0)

        # 4 consecutive ticks with unchanged pose
        for t in [1.0, 2.0, 3.0, 4.0]:
            collector.step(coord, timestamp=t)

        m = collector.compute_metrics(coord)

        self.assertEqual(m.unique_cells_visited, 1)
        self.assertEqual(m.total_cell_visits, 1)  # Only initial observation is an entry
        self.assertEqual(m.redundant_visits, 0)
        self.assertAlmostEqual(m.inter_robot_overlap_ratio, 0.0)
        self.assertEqual(m.revisit_events_count, 0)
        self.assertAlmostEqual(m.temporal_revisit_rate, 0.0)
        self.assertEqual(m.intra_robot_revisits, 0)
        self.assertAlmostEqual(m.intra_robot_self_revisit_rate, 0.0)
        self.assertAlmostEqual(m.total_distance_traveled, 0.0)
        self.assertEqual(len(m.coverage_timeline), 4)

    def test_scenario_f_same_timestamp_different_positions(self):
        """
        Scenario F: Same robot receives two updates with the same timestamp but different positions.
        Verifies that legitimate position changes at the same timestamp are NOT dropped.
        """
        coord = SwarmCoordinator(environment=self.env)
        a1 = RobotAgent("r1", Pose2D(5.0, 5.0, 0.0), self.slam)
        coord.register_robot(a1)
        collector = MetricsCollector(environment=self.env, strategy_name="ACO", spatial_bin_size=1.0)

        # Update 1: at (5.0, 5.0) at t=1.0
        collector.step(coord, timestamp=1.0)

        # Update 2: moves to (6.0, 5.0) at same timestamp t=1.0 (sub-tick displacement)
        a1.sim_robot.pose = Pose2D(6.0, 5.0, 0.0)
        collector.step(coord, timestamp=1.0)

        m = collector.compute_metrics(coord)

        # Position change must NOT be dropped
        self.assertAlmostEqual(m.total_distance_traveled, 1.0)
        self.assertEqual(m.unique_cells_visited, 2)
        self.assertEqual(m.total_cell_visits, 2)
        self.assertEqual(m.revisit_events_count, 0)

        # Update 3: duplicate tick with identical pose at same timestamp t=1.0 (must be deduplicated)
        collector.step(coord, timestamp=1.0)
        m2 = collector.compute_metrics(coord)
        self.assertAlmostEqual(m2.total_distance_traveled, 1.0)
        self.assertEqual(m2.total_cell_visits, 2)

    def test_scenario_g_empty_trajectories(self):
        """Scenario G: Empty trajectories with zero steps."""
        coord = SwarmCoordinator(environment=self.env)
        collector = MetricsCollector(environment=self.env, strategy_name="ACO", spatial_bin_size=1.0)

        m = collector.compute_metrics(coord)

        self.assertEqual(m.unique_cells_visited, 0)
        self.assertEqual(m.redundant_visits, 0)
        self.assertAlmostEqual(m.inter_robot_overlap_ratio, 0.0)
        self.assertEqual(m.shared_cells_count, 0)
        self.assertAlmostEqual(m.shared_cells_ratio, 0.0)
        self.assertEqual(m.total_cell_visits, 0)
        self.assertEqual(m.revisit_events_count, 0)
        self.assertAlmostEqual(m.temporal_revisit_rate, 0.0)
        self.assertEqual(m.intra_robot_revisits, 0)
        self.assertAlmostEqual(m.intra_robot_self_revisit_rate, 0.0)
        self.assertEqual(m.per_robot_self_revisit_rate, {})
        self.assertAlmostEqual(m.total_distance_traveled, 0.0)

    def test_negative_coordinates_and_bin_boundaries(self):
        """Validates that floor discretization correctly handles negative coordinates and exact boundaries."""
        coord = SwarmCoordinator(environment=self.env)
        a1 = RobotAgent("r1", Pose2D(0.0, 0.0, 0.0), self.slam)
        coord.register_robot(a1)
        collector = MetricsCollector(environment=self.env, strategy_name="ACO", spatial_bin_size=1.0)

        # 0.0 -> bin (0,0)
        collector.step(coord, timestamp=1.0)
        # 0.999 -> bin (0,0) [within boundary, dwell]
        a1.sim_robot.pose = Pose2D(0.999, 0.0, 0.0)
        collector.step(coord, timestamp=2.0)
        # 1.0 -> bin (1,0) [boundary crossing]
        a1.sim_robot.pose = Pose2D(1.0, 0.0, 0.0)
        collector.step(coord, timestamp=3.0)
        # -0.5 -> bin (-1,0) [negative coordinate floor check]
        a1.sim_robot.pose = Pose2D(-0.5, 0.0, 0.0)
        collector.step(coord, timestamp=4.0)
        # -1.0 -> bin (-1,0) [exact negative boundary]
        a1.sim_robot.pose = Pose2D(-1.0, 0.0, 0.0)
        collector.step(coord, timestamp=5.0)
        # -1.001 -> bin (-2,0)
        a1.sim_robot.pose = Pose2D(-1.001, 0.0, 0.0)
        collector.step(coord, timestamp=6.0)

        m = collector.compute_metrics(coord)
        # Expected distinct bins: (0,0), (1,0), (-1,0), (-2,0) = 4 unique cells
        self.assertEqual(m.unique_cells_visited, 4)
        self.assertEqual(m.total_cell_visits, 4)
        self.assertIn((0, 0), collector.cell_visits)
        self.assertIn((1, 0), collector.cell_visits)
        self.assertIn((-1, 0), collector.cell_visits)
        self.assertIn((-2, 0), collector.cell_visits)

    def test_mathematical_identity_fleet_partition(self):
        """
        Validates the fundamental decomposition identity:
        Total fleet non-novel entries = intra-robot self-revisits + inter-robot footprint duplicate entries.
        """
        coord = SwarmCoordinator(environment=self.env)
        a1 = RobotAgent("r1", Pose2D(1.0, 1.0, 0.0), self.slam)
        a2 = RobotAgent("r2", Pose2D(2.0, 1.0, 0.0), self.slam)
        a3 = RobotAgent("r3", Pose2D(10.0, 10.0, 0.0), self.slam)
        coord.register_robot(a1)
        coord.register_robot(a2)
        coord.register_robot(a3)
        collector = MetricsCollector(environment=self.env, strategy_name="ACO", spatial_bin_size=1.0)

        # Robot 1: (1,1) -> (2,1) -> (1,1) -> (3,1) [4 entries, 3 unique cells, 1 self-revisit]
        # Robot 2: (2,1) -> (3,1) -> (2,1) -> (4,1) [4 entries, 3 unique cells, 1 self-revisit]
        # Robot 3: (10,10) -> (11,10) [2 entries, 2 unique cells, 0 self-revisit]
        collector.step(coord, timestamp=1.0)

        a1.sim_robot.pose = Pose2D(2.0, 1.0, 0.0)
        a2.sim_robot.pose = Pose2D(3.0, 1.0, 0.0)
        a3.sim_robot.pose = Pose2D(11.0, 10.0, 0.0)
        collector.step(coord, timestamp=2.0)

        a1.sim_robot.pose = Pose2D(1.0, 1.0, 0.0)
        a2.sim_robot.pose = Pose2D(2.0, 1.0, 0.0)
        collector.step(coord, timestamp=3.0)

        a1.sim_robot.pose = Pose2D(3.0, 1.0, 0.0)
        a2.sim_robot.pose = Pose2D(4.0, 1.0, 0.0)
        collector.step(coord, timestamp=4.0)

        m = collector.compute_metrics(coord)

        # Total entries: 4 + 4 + 2 = 10
        self.assertEqual(m.total_cell_visits, 10)
        # Fleet unique cells: (1,1), (2,1), (3,1), (4,1), (10,10), (11,10) = 6
        self.assertEqual(m.unique_cells_visited, 6)
        # Non-novel entries: 10 - 6 = 4
        self.assertEqual(m.revisit_events_count, 4)

        # Intra-robot self revisits: R1=1, R2=1, R3=0 -> sum = 2
        self.assertEqual(m.intra_robot_revisits, 2)
        self.assertAlmostEqual(m.intra_robot_self_revisit_rate, 2.0 / 10.0)
        self.assertAlmostEqual(m.per_robot_self_revisit_rate["r1"], 1.0 / 4.0)
        self.assertAlmostEqual(m.per_robot_self_revisit_rate["r2"], 1.0 / 4.0)
        self.assertAlmostEqual(m.per_robot_self_revisit_rate["r3"], 0.0)

        # Inter-robot footprint duplicate entries: sum(|C_r|) - |C| = (3 + 3 + 2) - 6 = 2
        self.assertEqual(m.redundant_visits, 2)
        self.assertAlmostEqual(m.inter_robot_overlap_ratio, 2.0 / 8.0)

        # Strict identity check: 4 == 2 + 2
        self.assertEqual(m.revisit_events_count, m.intra_robot_revisits + m.redundant_visits)


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

    def test_complete_metric_data_flow_to_json_and_markdown(self):
        """Verifies end-to-end data flow: MetricsCollector -> MissionMetrics -> aggregation -> json -> markdown."""
        class MockArgs:
            trials = 2
            steps = 20
            dt = 0.2
            robots = 2
            width = 30.0
            height = 30.0
            target_x = 20.0
            target_y = 20.0
            target_plate = "FLOW-TEST"
            inject_failure_tick = None
            fail_robot = "robot_1"
            seed_start = 100

        args = MockArgs()
        m1 = run_single_trial(StrategyType.ACO, seed=101, args=args)
        m2 = run_single_trial(StrategyType.ACO, seed=102, args=args)

        # 1. MissionMetrics fields verification
        for m in [m1, m2]:
            self.assertIsNotNone(m.overlap_ratio)
            self.assertIsNotNone(m.inter_robot_overlap_ratio)
            self.assertIsNotNone(m.shared_cells_ratio)
            self.assertIsNotNone(m.temporal_revisit_rate)
            self.assertIsNotNone(m.intra_robot_self_revisit_rate)
            self.assertIsNotNone(m.total_cell_visits)
            self.assertIsNotNone(m.revisit_events_count)
            self.assertIsNotNone(m.intra_robot_revisits)

            # Serialization to dict
            d = m.to_dict()
            self.assertIn("overlap_ratio", d)
            self.assertIn("inter_robot_overlap_ratio", d)
            self.assertIn("shared_cells_ratio", d)
            self.assertIn("temporal_revisit_rate", d)
            self.assertIn("intra_robot_self_revisit_rate", d)
            self.assertIn("total_cell_visits", d)
            self.assertIn("revisit_events_count", d)
            self.assertIn("intra_robot_revisits", d)

        # 2. Aggregation verification
        agg = aggregate_strategy_metrics([m1, m2])
        self.assertIn("mean_overlap_ratio_pct", agg)
        self.assertIn("mean_inter_robot_overlap_ratio_pct", agg)
        self.assertIn("mean_shared_cells_ratio_pct", agg)
        self.assertIn("mean_temporal_revisit_rate_pct", agg)
        self.assertIn("mean_intra_robot_self_revisit_rate_pct", agg)
        self.assertIn("mean_total_cell_visits", agg)
        self.assertIn("mean_revisit_events_count", agg)
        self.assertIn("mean_intra_robot_revisits", agg)
        self.assertEqual(len(agg["raw_trials"]), 2)

        # 3. JSON serialization round-trip
        json_str = json.dumps(agg)
        loaded = json.loads(json_str)
        self.assertIn("mean_temporal_revisit_rate_pct", loaded)
        self.assertIn("mean_intra_robot_self_revisit_rate_pct", loaded)
        self.assertIn("mean_inter_robot_overlap_ratio_pct", loaded)

        # 4. Markdown Report Labels
        results = {"ACO": agg}
        report = generate_markdown_report(results, args)
        self.assertIn("| **Trajectory Overlap Ratio (%)** |", report)
        self.assertIn("| **Inter-Robot Shared Area (%)** |", report)
        self.assertIn("| **Temporal Revisit Rate (%)** |", report)
        self.assertIn("| **Intra-Robot Self-Revisit Rate (%)** |", report)

    def test_backward_compatibility_historical_json_loading(self):
        """Verifies that historical JSON logs lacking new fields load safely and distinguish missing from 0.0."""
        # Simulated raw_trial from historical benchmark_results.json
        legacy_trial = {
            "strategy_name": "ACO",
            "trial_seed": 100,
            "total_simulation_time": 28.0,
            "target_found": True,
            "time_to_detection": 23.4,
            "final_exploration_ratio": 0.2636,
            "exploration_rate_m2_per_sec": 23.53,
            "overlap_ratio": 0.0992,
            "unique_cells_visited": 118,
            "redundant_visits": 13,
            "total_distance_traveled": 118.75,
            "per_robot_distance": {"robot_1": 28.3, "robot_2": 32.1},
            "total_energy_consumed": 0.0317,
            "tasks_completed": 5,
            "tasks_created": 36,
            "reassignment_events": 10,
            "failed_robots_count": 0,
            "recovered_tasks_count": 0,
        }

        m_legacy = MissionMetrics.from_dict(legacy_trial)
        self.assertEqual(m_legacy.strategy_name, "ACO")
        self.assertEqual(m_legacy.overlap_ratio, 0.0992)
        self.assertEqual(m_legacy.redundant_visits, 13)
        self.assertEqual(m_legacy.unique_cells_visited, 118)

        # Missing new metrics must be None to indicate "not measured in historical data"
        self.assertIsNone(m_legacy.inter_robot_overlap_ratio)
        self.assertIsNone(m_legacy.shared_cells_ratio)
        self.assertIsNone(m_legacy.temporal_revisit_rate)
        self.assertIsNone(m_legacy.intra_robot_self_revisit_rate)
        self.assertIsNone(m_legacy.total_cell_visits)
        self.assertIsNone(m_legacy.revisit_events_count)
        self.assertIsNone(m_legacy.intra_robot_revisits)

        # Distinguish from a genuinely measured 0.0
        m_measured_zero = MissionMetrics(
            strategy_name="ACO",
            trial_seed=101,
            total_simulation_time=10.0,
            temporal_revisit_rate=0.0,
            intra_robot_self_revisit_rate=0.0,
            inter_robot_overlap_ratio=0.0,
        )
        self.assertIsNotNone(m_measured_zero.temporal_revisit_rate)
        self.assertEqual(m_measured_zero.temporal_revisit_rate, 0.0)

        # Aggregation of legacy metrics
        agg_legacy = aggregate_strategy_metrics([m_legacy])
        self.assertIsNone(agg_legacy["mean_temporal_revisit_rate_pct"])
        self.assertIsNone(agg_legacy["mean_intra_robot_self_revisit_rate_pct"])
        self.assertEqual(agg_legacy["mean_overlap_ratio_pct"], 0.0992 * 100.0)

        class MockArgs:
            width = 50.0
            height = 50.0
            robots = 4
            trials = 1
            steps = 100
            dt = 0.2

        report_legacy = generate_markdown_report({"ACO": agg_legacy}, MockArgs())
        self.assertIn("N/A", report_legacy)

    def test_ai_benchmark_result_backward_compatibility(self):
        """Verifies ScenarioBenchmarkResult loads legacy JSON."""
        legacy_scenario_json = {
            "scenario_name": "scenario_a",
            "scenario_desc": "Target easy to detect.",
            "target_plate": "DXB-88392",
            "target_found": True,
            "time_to_detection": 20.8,
            "final_coverage_pct": 24.37,
            "total_distance_m": 97.51,
            "redundancy_ratio": 0.015,
            "completed_tasks": 1,
            "reassignment_count": 6,
            "failed_robots": 0,
            "recovered_tasks": 0,
            "target_confidence": 0.753,
            "vehicle_confidence": 0.626,
            "plate_confidence": 0.6,
            "ocr_confidence": 0.765,
            "appearance_match": 1.0,
            "temporal_consistency": 0.84,
            "detections_count": 82,
            "false_positives_count": 0,
            "wall_clock_sec": 0.5,
        }

        res = ScenarioBenchmarkResult.from_dict(legacy_scenario_json)
        self.assertEqual(res.scenario_name, "scenario_a")
        self.assertEqual(res.redundancy_ratio, 0.015)
        self.assertIsNone(res.temporal_revisit_rate)
        self.assertIsNone(res.intra_robot_self_revisit_rate)
        self.assertIsNone(res.inter_robot_overlap_ratio)


if __name__ == "__main__":
    unittest.main()
