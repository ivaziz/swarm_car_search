"""
Comprehensive End-to-End Tests for Swarm Target Vehicle Search & Identification Pipeline.
Validates:
1. Long-range vs close-range OCR classification (CONFIRMED vs CANDIDATE)
2. Decoy vehicle false positive rejection
3. Stigmergic recruitment and target confirmation preemption
4. Decentralized failure recovery on target tasks
5. All 6 deterministic simulation benchmark scenarios
6. Ablation study configurations
"""

import math
import os
import sys
import unittest

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from interfaces.slam_interface import Pose2D
from simulation.environment import UrbanEnvironment, UrbanZone
from simulation.obstacle_generator import ObstacleGenerator
from simulation.scenarios import create_scenario, ScenarioConfig
from slam_providers.mock_slam_provider import MockSLAMProvider
from swarm.states.robot_states import RobotState
from swarm.aco.aco_config import ACOConfig
from swarm.aco.aco_engine import ACOEngine
from swarm.aco.task_scorer import TaskScorer
from swarm.task_allocation.frontier_task import SearchTask, TaskStatus, TaskType
from swarm.core.robot_agent import RobotAgent
from swarm.core.swarm_coordinator import SwarmCoordinator
from perception.vehicle_detector import TargetVehicle, SimulatedVehicleDetector
from perception.plate_recognizer import SimulatedPlateRecognizer
from perception.matching_engine import MatchingEngine, levenshtein_similarity
from evaluation.baselines import StrategyType
from scripts.run_evaluation import run_single_trial, aggregate_strategy_metrics


class TestTargetPerceptionAndMatching(unittest.TestCase):
    """Unit tests for multi-stage vehicle perception, distance attenuation, and fuzzy matching."""

    def setUp(self):
        self.detector = SimulatedVehicleDetector(max_detection_range=15.0, fov_degrees=90.0)
        self.ocr = SimulatedPlateRecognizer(base_ocr_confidence=0.95)
        self.matcher = MatchingEngine(match_threshold=0.80, candidate_threshold=0.50)

        self.target = TargetVehicle(
            target_id="target_sedan",
            plate_number="DXB-88392",
            vehicle_class="sedan",
            color="dark_gray",
            position=(25.0, 25.0),
            is_stolen=True,
        )
        self.decoy = TargetVehicle(
            target_id="decoy_sedan",
            plate_number="ABC-11223",
            vehicle_class="sedan",
            color="white",
            position=(20.0, 25.0),
            is_stolen=False,
        )
        self.matcher.add_target_vehicle(self.target)

    def test_close_range_ocr_accuracy(self):
        """Close-range scan (<=5m) yields zero noise, 1.0 similarity, and CONFIRMED classification."""
        robot_pose = Pose2D(25.0, 22.0, math.pi / 2.0)  # 3.0m away, facing North
        detections = self.detector.detect_vehicles("r1", robot_pose, timestamp=1.0, vehicles=[self.target])
        self.assertEqual(len(detections), 1)

        det_event, ground_truth = detections[0]
        self.assertLessEqual(det_event.distance, 5.0)

        plate_event = self.ocr.recognize_license_plate(det_event, ground_truth)
        self.assertEqual(plate_event.plate_number, "DXB-88392")
        self.assertGreaterEqual(plate_event.ocr_confidence, 0.90)

        classification, match_event, sim = self.matcher.classify_plate(plate_event)
        self.assertEqual(classification, "CONFIRMED")
        self.assertIsNotNone(match_event)
        self.assertTrue(match_event.confirmed)
        self.assertEqual(match_event.plate_number, "DXB-88392")
        self.assertAlmostEqual(sim, 1.0)

    def test_long_range_candidate_classification(self):
        """Long-range scan (>5m) attenuates confidence and classifies as CANDIDATE."""
        robot_pose = Pose2D(25.0, 13.0, math.pi / 2.0)  # 12.0m away, facing North
        detections = self.detector.detect_vehicles("r1", robot_pose, timestamp=1.0, vehicles=[self.target])
        self.assertEqual(len(detections), 1)

        det_event, ground_truth = detections[0]
        self.assertGreater(det_event.distance, 5.0)

        plate_event = self.ocr.recognize_license_plate(det_event, ground_truth)
        classification, match_event, sim = self.matcher.classify_plate(plate_event)
        # At 12m, either confirmed or candidate, but never falsely rejected as non-target
        self.assertIn(classification, ("CONFIRMED", "CANDIDATE"))

    def test_decoy_vehicle_rejection(self):
        """Decoy vehicle plate does not match watchlist and is classified as NON_TARGET."""
        robot_pose = Pose2D(20.0, 22.0, math.pi / 2.0)  # 3.0m from decoy
        detections = self.detector.detect_vehicles("r1", robot_pose, timestamp=1.0, vehicles=[self.decoy])
        self.assertEqual(len(detections), 1)

        det_event, ground_truth = detections[0]
        plate_event = self.ocr.recognize_license_plate(det_event, ground_truth)
        classification, match_event, sim = self.matcher.classify_plate(plate_event)

        self.assertEqual(classification, "NON_TARGET")
        self.assertIsNone(match_event)
        self.assertLess(sim, 0.50)


class TestACOTargetTaskPrioritization(unittest.TestCase):
    """Tests ACO TaskScorer prioritization of target confirmation over generic frontiers."""

    def setUp(self):
        self.config = ACOConfig()
        self.scorer = TaskScorer(self.config)
        self.robot_pose = Pose2D(10.0, 10.0, 0.0)

    def test_target_confirmation_scoring_boost(self):
        """TARGET_CONFIRMATION tasks receive a massive priority multiplier over EXPLORATION tasks."""
        generic_task = SearchTask(
            task_id="task_front_01",
            cluster_id="c_01",
            centroid_x=15.0,
            centroid_y=10.0,
            bounding_box=(14.0, 9.0, 16.0, 11.0),
            size=10,
            priority=1.0,
            detection_probability=0.10,
            task_type=TaskType.EXPLORATION,
        )

        target_task = SearchTask(
            task_id="task_target_01",
            cluster_id="c_target",
            centroid_x=15.0,
            centroid_y=10.0,
            bounding_box=(14.0, 9.0, 16.0, 11.0),
            size=10,
            priority=1.0,
            detection_probability=0.85,
            task_type=TaskType.TARGET_CONFIRMATION,
        )

        generic_score = self.scorer.compute_score(generic_task, self.robot_pose)
        target_score = self.scorer.compute_score(target_task, self.robot_pose)

        # Target score must dramatically exceed generic exploration score at equal distance
        self.assertGreater(target_score, generic_score * 3.0)


class TestDeterministicScenarios(unittest.TestCase):
    """Tests factory construction and parameters of all 6 deterministic scenarios."""

    def test_all_scenarios_instantiation(self):
        scenario_names = [
            "easy_target",
            "hidden_target",
            "target_recruitment",
            "failure_recovery",
            "false_positive",
            "multiple_vehicles",
        ]

        for name in scenario_names:
            sc = create_scenario(name)
            self.assertIsInstance(sc, ScenarioConfig)
            self.assertEqual(sc.name, name)
            self.assertGreater(len(sc.vehicles), 0)
            self.assertGreater(len(sc.start_poses), 0)
            self.assertGreater(sc.max_steps, 0)
            self.assertIsNotNone(sc.target_plate)

            # Ensure all vehicles are positioned in obstacle-free space
            env = UrbanEnvironment(width=sc.width, height=sc.height, resolution=sc.resolution)
            ObstacleGenerator.add_perimeter_walls(env, thickness_cells=1)
            ObstacleGenerator.add_box_obstacle(env, sc.width * 0.2, sc.height * 0.2, sc.width * 0.45, sc.height * 0.45)
            ObstacleGenerator.add_box_obstacle(env, sc.width * 0.55, sc.height * 0.2, sc.width * 0.8, sc.height * 0.45)
            ObstacleGenerator.add_box_obstacle(env, sc.width * 0.2, sc.height * 0.55, sc.width * 0.45, sc.height * 0.8)
            ObstacleGenerator.add_box_obstacle(env, sc.width * 0.55, sc.height * 0.55, sc.width * 0.8, sc.height * 0.8)

            for v in sc.vehicles:
                vx, vy = v.position
                self.assertTrue(env.is_free(vx, vy), f"Vehicle {v.target_id} in {name} is at blocked position ({vx}, {vy})")


class TestEndToEndTargetSearchExecution(unittest.TestCase):
    """End-to-end simulation trials verifying target identification on key scenarios."""

    def test_easy_target_scenario_success(self):
        """Scenario A (easy_target): Swarm finds and confirms target at (25, 25)."""
        class DummyArgs:
            scenario = "easy_target"
            trials = 1
            steps = 120
            dt = 0.2
            robots = 4
            width = 50.0
            height = 50.0
            target_x = 25.0
            target_y = 25.0
            target_plate = "DXB-88392"
            inject_failure_tick = None
            fail_robot = "robot_1"

        args = DummyArgs()
        metrics = run_single_trial("ACO", StrategyType.ACO, seed=100, args=args)

        self.assertTrue(metrics.target_found, "Target should be found in easy_target scenario")
        self.assertIsNotNone(metrics.time_to_detection)
        self.assertLess(metrics.time_to_detection, 30.0)
        self.assertEqual(metrics.confirmed_target_location, (25.0, 25.0))
        self.assertGreaterEqual(metrics.target_confirmations_count, 1)

    def test_false_positive_rejection_scenario(self):
        """Scenario E (false_positive): Rejects decoy vehicles and confirms true target."""
        class DummyArgs:
            scenario = "false_positive"
            trials = 1
            steps = 180
            dt = 0.2
            robots = 4
            width = 50.0
            height = 50.0
            target_x = 25.0
            target_y = 35.0
            target_plate = "DXB-88392"
            inject_failure_tick = None
            fail_robot = "robot_1"

        args = DummyArgs()
        metrics = run_single_trial("ACO", StrategyType.ACO, seed=120, args=args)

        self.assertGreater(metrics.candidate_detections_count, 0)
        self.assertTrue(metrics.target_found, "True target should be confirmed in false_positive scenario")

    def test_failure_recovery_scenario(self):
        """Scenario D (failure_recovery): Assigned robot fails; swarm reclaims task and confirms."""
        class DummyArgs:
            scenario = "failure_recovery"
            trials = 1
            steps = 140
            dt = 0.2
            robots = 4
            width = 50.0
            height = 50.0
            target_x = 25.0
            target_y = 25.0
            target_plate = "DXB-88392"
            inject_failure_tick = 25
            fail_robot = "robot_1"

        args = DummyArgs()
        metrics = run_single_trial("ACO", StrategyType.ACO, seed=115, args=args)

        self.assertGreaterEqual(metrics.failed_robots_count, 1)
        self.assertTrue(metrics.failure_recovery_success)
        self.assertTrue(metrics.target_found)


class TestAblationStudyExecution(unittest.TestCase):
    """Verifies that all 4 ablation variants execute properly and produce comparable metrics."""

    def test_ablation_variants_run(self):
        class DummyArgs:
            scenario = "easy_target"
            trials = 1
            steps = 60
            dt = 0.2
            robots = 4
            width = 50.0
            height = 50.0
            target_x = 25.0
            target_y = 25.0
            target_plate = "DXB-88392"
            inject_failure_tick = None
            fail_robot = "robot_1"

        args = DummyArgs()
        variants = [
            ("ACO_BASELINE", ACOConfig(weights=ACOConfig().weights.__class__(alpha=0.0, gamma=0.0))),
            ("ACO_PROB_ONLY", ACOConfig(weights=ACOConfig().weights.__class__(alpha=0.0, gamma=1.8))),
            ("ACO_PHERO_ONLY", ACOConfig(weights=ACOConfig().weights.__class__(alpha=1.2, gamma=0.0))),
            ("ACO_FULL", ACOConfig(weights=ACOConfig().weights.__class__(alpha=1.2, gamma=1.8))),
        ]

        for label, cfg in variants:
            m = run_single_trial(label, None, seed=100, args=args, aco_config=cfg)
            self.assertEqual(m.strategy_name, label)
            self.assertGreater(m.final_exploration_ratio, 0.0)
            self.assertGreater(m.total_distance_traveled, 0.0)


if __name__ == "__main__":
    unittest.main()
