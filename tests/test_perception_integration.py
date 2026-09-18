"""
Comprehensive Unit and Integration Tests for Vehicle Detection, OCR, Matching, and Swarm Reaction.
"""

import unittest
import math
from interfaces.slam_interface import Pose2D
from simulation.environment import UrbanEnvironment
from slam_providers.mock_slam_provider import MockSLAMProvider
from swarm.states.robot_states import RobotState
from swarm.task_allocation.frontier_task import SearchTask
from swarm.communication.local_comm import LocalMessageBus
from swarm.core.robot_agent import RobotAgent
from swarm.core.swarm_coordinator import SwarmCoordinator
from swarm.pheromone.evaporation_model import PheromoneLayerType
from perception.vehicle_detector import TargetVehicle, SimulatedVehicleDetector
from perception.plate_recognizer import SimulatedPlateRecognizer
from perception.matching_engine import MatchingEngine, levenshtein_similarity, normalize_plate


class TestPerceptionModels(unittest.TestCase):
    """Tests for standalone vision detection, plate OCR, and fuzzy matching."""

    def setUp(self):
        self.detector = SimulatedVehicleDetector(max_detection_range=15.0, fov_degrees=90.0)
        self.ocr = SimulatedPlateRecognizer(base_ocr_confidence=0.95)
        self.matcher = MatchingEngine(match_threshold=0.80)

        # Register target car in watchlist
        self.target = TargetVehicle(
            target_id="target_01",
            plate_number="DXB-77492",
            vehicle_class="sedan",
            color="black",
            position=(10.0, 10.0),
        )
        self.matcher.add_target_vehicle(self.target)

    def test_plate_normalization_and_fuzzy_similarity(self):
        self.assertEqual(normalize_plate("dxb-77492"), "DXB77492")
        self.assertEqual(normalize_plate("DXB  77492 "), "DXB77492")

        # Identical
        self.assertAlmostEqual(levenshtein_similarity("DXB77492", "DXB77492"), 1.0)
        # Minor typo (1 character replaced)
        sim = levenshtein_similarity("DXB77492", "DXB77493")
        self.assertGreater(sim, 0.85)
        # Completely different
        self.assertLess(levenshtein_similarity("DXB77492", "AUH12345"), 0.3)

    def test_vehicle_detection_in_fov(self):
        # Robot at (10.0, 5.0) facing North (theta = pi/2)
        robot_pose = Pose2D(10.0, 5.0, math.pi / 2.0)
        detections = self.detector.detect_vehicles("r1", robot_pose, timestamp=1.0, vehicles=[self.target])

        self.assertEqual(len(detections), 1)
        det_event, ground_truth = detections[0]
        self.assertEqual(ground_truth.target_id, "target_01")
        self.assertGreater(det_event.confidence, 0.70)

    def test_vehicle_detection_out_of_range(self):
        # Robot at (10.0, -10.0) -> distance = 20m (> 15m range)
        robot_pose = Pose2D(10.0, -10.0, math.pi / 2.0)
        detections = self.detector.detect_vehicles("r1", robot_pose, timestamp=1.0, vehicles=[self.target])
        self.assertEqual(len(detections), 0)

    def test_vehicle_detection_out_of_fov(self):
        # Robot at (10.0, 5.0) facing South (theta = -pi/2), vehicle is North
        robot_pose = Pose2D(10.0, 5.0, -math.pi / 2.0)
        detections = self.detector.detect_vehicles("r1", robot_pose, timestamp=1.0, vehicles=[self.target])
        self.assertEqual(len(detections), 0)

    def test_obstacle_line_of_sight_occlusion(self):
        env = UrbanEnvironment(width=30.0, height=30.0, resolution=1.0)
        # Put solid wall between robot (10, 5) and target (10, 10) at (10, 7)
        env.set_cell(10, 7, 100)

        robot_pose = Pose2D(10.0, 5.0, math.pi / 2.0)
        detections = self.detector.detect_vehicles("r1", robot_pose, timestamp=1.0, vehicles=[self.target], env=env)
        # Should be occluded by wall
        self.assertEqual(len(detections), 0)

    def test_full_pipeline_matching(self):
        robot_pose = Pose2D(10.0, 8.0, math.pi / 2.0)
        detections = self.detector.detect_vehicles("r1", robot_pose, 1.0, [self.target])
        self.assertEqual(len(detections), 1)

        det_event, target_veh = detections[0]
        plate_event = self.ocr.recognize_license_plate(det_event, target_veh)
        self.assertEqual(plate_event.plate_number, "DXB-77492")

        match = self.matcher.match_plate(plate_event)
        self.assertIsNotNone(match)
        self.assertTrue(match.confirmed)
        self.assertEqual(match.target_id, "target_01")


class TestPerceptionSwarmIntegration(unittest.TestCase):
    """Integration test verifying swarm state escalation upon target vehicle confirmation."""

    def setUp(self):
        self.env = UrbanEnvironment(width=40.0, height=40.0, resolution=1.0)
        self.slam = MockSLAMProvider(self.env)
        self.bus = LocalMessageBus()

        self.detector = SimulatedVehicleDetector(max_detection_range=15.0)
        self.ocr = SimulatedPlateRecognizer()
        self.matcher = MatchingEngine(match_threshold=0.80)

        self.target = TargetVehicle(
            target_id="stolen_sedan_alpha",
            plate_number="DXB-77492",
            vehicle_class="sedan",
            color="black",
            position=(10.0, 12.0),
        )
        self.matcher.add_target_vehicle(self.target)

        self.coordinator = SwarmCoordinator(environment=self.env)

        # Register Robot 1 near target
        self.agent_1 = RobotAgent(
            robot_id="robot_1",
            initial_pose=Pose2D(10.0, 10.0, math.pi / 2.0),  # Facing target North
            slam_provider=self.slam,
            comm_channel=self.bus,
            detector=self.detector,
            ocr=self.ocr,
            matcher=self.matcher,
            known_vehicles=[self.target],
        )
        # Register Robot 2 elsewhere
        self.agent_2 = RobotAgent(
            robot_id="robot_2",
            initial_pose=Pose2D(30.0, 30.0, 0.0),
            slam_provider=self.slam,
            comm_channel=self.bus,
            detector=self.detector,
            ocr=self.ocr,
            matcher=self.matcher,
            known_vehicles=[self.target],
        )

        self.coordinator.register_robot(self.agent_1)
        self.coordinator.register_robot(self.agent_2)

    def test_target_detection_escalation_in_swarm(self):
        # Set robot_1 in SEARCHING state
        self.agent_1.fsm.transition_to(RobotState.SEARCHING, timestamp=0.0)

        # Step coordination loop
        status = self.coordinator.step_coordination(timestamp=1.0, dt=0.1)

        # Robot 1 should have transitioned: SEARCHING -> VEHICLE_DETECTED -> VERIFYING -> REPORTING
        self.assertEqual(self.agent_1.current_state, RobotState.REPORTING)
        self.assertIsNotNone(self.agent_1.last_confirmed_match)
        self.assertEqual(self.agent_1.last_confirmed_match.target_id, "stolen_sedan_alpha")

        # Coordinator must have registered target discovery
        self.assertTrue(self.coordinator.target_found)
        self.assertEqual(self.coordinator.confirmed_target_location, (10.0, 12.0))

        # Check SUCCESS pheromone was heavily deposited around target position (10, 12)
        success_pheromone = self.coordinator.pheromone_map.get_value(
            layer=PheromoneLayerType.SUCCESS,
            x=10.0,
            y=12.0,
        )
        self.assertGreater(success_pheromone, 15.0)

        # Check peer robot 2's decentralized SwarmState has target_found marked True
        r2_swarm_state = self.coordinator.robot_swarm_states["robot_2"]
        self.assertTrue(r2_swarm_state.target_found)
        self.assertIsNotNone(r2_swarm_state.confirmed_target_location)
        self.assertEqual(r2_swarm_state.confirmed_target_location.x, 10.0)


if __name__ == "__main__":
    unittest.main()
