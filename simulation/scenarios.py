"""
Deterministic End-to-End Simulation Scenarios for Swarm Urban Vehicle Search.
Provides pre-configured, reproducible benchmark scenarios validating target identification,
multi-robot recruitment, false-positive rejection, and fault recovery.
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any
import math

from interfaces.slam_interface import Pose2D
from simulation.environment import UrbanEnvironment, UrbanZone
from simulation.obstacle_generator import ObstacleGenerator
from perception.vehicle_detector import TargetVehicle


@dataclass
class ScenarioConfig:
    """Configuration specification for a deterministic swarm search scenario."""
    name: str
    description: str
    width: float = 50.0
    height: float = 50.0
    resolution: float = 1.0
    fleet_size: int = 4
    max_steps: int = 100
    dt: float = 0.2
    target_plate: str = "DXB-88392"
    vehicles: List[TargetVehicle] = field(default_factory=list)
    start_poses: List[Tuple[float, float, float]] = field(default_factory=list)
    zones: List[UrbanZone] = field(default_factory=list)
    inject_failure_tick: Optional[int] = None
    fail_robot_id: Optional[str] = None
    seed: int = 42


def get_default_environment(width: float = 50.0, height: float = 50.0, resolution: float = 1.0) -> UrbanEnvironment:
    """Constructs the canonical 4-block urban city grid with clear navigable avenues."""
    env = UrbanEnvironment(width=width, height=height, resolution=resolution)
    ObstacleGenerator.add_perimeter_walls(env, thickness_cells=1)

    # 4 Solid City Blocks (Boxes)
    w, h = width, height
    # Box 1 (South-West Block): [10.0, 22.5] x [10.0, 22.5]
    ObstacleGenerator.add_box_obstacle(env, w * 0.2, h * 0.2, w * 0.45, h * 0.45)
    # Box 2 (South-East Block): [27.5, 40.0] x [10.0, 22.5]
    ObstacleGenerator.add_box_obstacle(env, w * 0.55, h * 0.2, w * 0.8, h * 0.45)
    # Box 3 (North-West Block): [10.0, 22.5] x [27.5, 40.0]
    ObstacleGenerator.add_box_obstacle(env, w * 0.2, h * 0.55, w * 0.45, h * 0.8)
    # Box 4 (North-East Block): [27.5, 40.0] x [27.5, 40.0]
    ObstacleGenerator.add_box_obstacle(env, w * 0.55, h * 0.55, w * 0.8, h * 0.8)

    return env


def create_scenario(scenario_name: str) -> ScenarioConfig:
    """
    Factory creating a named deterministic simulation scenario.

    Supported Scenarios:
    - 'easy_target'        (Scenario A: Central plaza reachable target)
    - 'hidden_target'      (Scenario B: Peripheral corridor target behind building)
    - 'target_recruitment' (Scenario C: Multi-robot detection and recruitment)
    - 'failure_recovery'   (Scenario D: Assigned robot fails, peer reclaims and confirms)
    - 'false_positive'     (Scenario E: Non-target vehicles rejected, target confirmed)
    - 'multiple_vehicles'  (Scenario F: Target identified among fleet of decoy vehicles)
    """
    name = scenario_name.lower().strip()
    target_plate = "DXB-88392"

    # Canonical 4 corner starting positions (free of obstacles)
    default_starts = [
        (6.0, 6.0, 0.0),                  # Robot 1 (South-West corner, facing East)
        (6.0, 44.0, -math.pi / 2.0),       # Robot 2 (North-West corner, facing South)
        (44.0, 6.0, math.pi / 2.0),        # Robot 3 (South-East corner, facing North)
        (44.0, 44.0, math.pi),             # Robot 4 (North-East corner, facing West)
    ]

    if name in ("easy_target", "scenario_a"):
        # Scenario A: Target parked in central plaza intersection (25.0, 25.0)
        target = TargetVehicle(
            target_id="target_car_alpha",
            plate_number=target_plate,
            vehicle_class="sedan",
            color="dark_gray",
            position=(25.0, 25.0),
            is_stolen=True,
        )
        return ScenarioConfig(
            name=name,
            description="Scenario A: Target easy to detect in central intersection.",
            target_plate=target_plate,
            vehicles=[target],
            start_poses=default_starts,
            max_steps=500,
            seed=100,
            zones=[UrbanZone("central_plaza", (20.0, 20.0, 30.0, 30.0), 0.85)],
        )

    elif name in ("hidden_target", "partially_occluded", "scenario_b"):
        # Scenario B: Target parked on East street corridor (45.0, 35.0), behind building block
        target = TargetVehicle(
            target_id="target_car_hidden",
            plate_number=target_plate,
            vehicle_class="sedan",
            color="dark_gray",
            position=(45.0, 35.0),
            is_stolen=True,
        )
        return ScenarioConfig(
            name=name,
            description="Scenario B: Target partially occluded behind eastern building block corridor.",
            target_plate=target_plate,
            vehicles=[target],
            start_poses=default_starts,
            max_steps=600,
            seed=105,
            zones=[UrbanZone("east_corridor", (40.0, 28.0, 49.0, 42.0), 0.80)],
        )

    elif name in ("multiple_decoys", "multiple_vehicles", "scenario_c", "false_positive"):
        # Scenario C: Multiple decoy vehicles with distinct colors, classes, and plates
        true_target = TargetVehicle(
            target_id="target_stolen_car",
            plate_number=target_plate,
            vehicle_class="sedan",
            color="dark_gray",
            position=(25.0, 25.0),
            is_stolen=True,
        )
        decoy_1 = TargetVehicle(
            target_id="decoy_suv_1",
            plate_number="AD-10294",
            vehicle_class="suv",
            color="white",
            position=(10.0, 25.0),
            is_stolen=False,
        )
        decoy_2 = TargetVehicle(
            target_id="decoy_sedan_2",
            plate_number="SHJ-44910",
            vehicle_class="sedan",
            color="red",
            position=(40.0, 25.0),
            is_stolen=False,
        )
        decoy_3 = TargetVehicle(
            target_id="decoy_truck_3",
            plate_number="DXB-99112",
            vehicle_class="truck",
            color="blue",
            position=(25.0, 10.0),
            is_stolen=False,
        )
        return ScenarioConfig(
            name=name,
            description="Scenario C: Target verified among fleet of multiple decoy civilian vehicles.",
            target_plate=target_plate,
            vehicles=[true_target, decoy_1, decoy_2, decoy_3],
            start_poses=default_starts,
            max_steps=500,
            seed=110,
            zones=[UrbanZone("central_hub", (20.0, 20.0, 30.0, 30.0), 0.85)],
        )

    elif name in ("noisy_ocr", "poor_ocr", "scenario_d"):
        # Scenario D: Poor OCR / partial plate reading requiring multi-frame accumulation
        target = TargetVehicle(
            target_id="target_car_noisy",
            plate_number=target_plate,
            vehicle_class="sedan",
            color="dark_gray",
            position=(25.0, 35.0),
            is_stolen=True,
        )
        return ScenarioConfig(
            name=name,
            description="Scenario D: Optical noise and character confusion requiring multi-frame temporal voting.",
            target_plate=target_plate,
            vehicles=[target],
            start_poses=default_starts,
            max_steps=500,
            seed=115,
            zones=[UrbanZone("north_lane", (20.0, 30.0, 30.0, 40.0), 0.80)],
        )

    elif name in ("failure_recovery", "scenario_e"):
        # Scenario E: One robot fails during search; swarm recovers orphaned tasks dynamically
        target = TargetVehicle(
            target_id="target_car_fault",
            plate_number=target_plate,
            vehicle_class="sedan",
            color="dark_gray",
            position=(25.0, 25.0),
            is_stolen=True,
        )
        return ScenarioConfig(
            name=name,
            description="Scenario E: Robot failure during search; scored dynamic reassignment recovers task.",
            target_plate=target_plate,
            vehicles=[target],
            start_poses=default_starts,
            inject_failure_tick=20,
            fail_robot_id="robot_1",
            max_steps=500,
            seed=120,
            zones=[UrbanZone("central_plaza", (20.0, 20.0, 30.0, 30.0), 0.85)],
        )

    elif name in ("cooperative_encounter", "target_recruitment", "scenario_f"):
        # Scenario F: Two robots encounter candidate area; cooperative convergence & verification
        target = TargetVehicle(
            target_id="target_car_coop",
            plate_number=target_plate,
            vehicle_class="sedan",
            color="dark_gray",
            position=(25.0, 35.0),
            is_stolen=True,
        )
        return ScenarioConfig(
            name=name,
            description="Scenario F: Two robots encounter candidate; cooperative deconfliction & verification.",
            target_plate=target_plate,
            vehicles=[target],
            start_poses=default_starts,
            max_steps=600,
            seed=125,
            zones=[UrbanZone("north_avenue", (15.0, 25.0, 35.0, 45.0), 0.85)],
        )

    elif name in ("far_target", "scenario_g"):
        # Scenario G: Target located far from initial robot positions (North-East sector, 40.0, 40.0)
        target = TargetVehicle(
            target_id="target_car_far",
            plate_number=target_plate,
            vehicle_class="sedan",
            color="dark_gray",
            position=(35.0, 25.0),
            is_stolen=True,
        )
        # All robots start in South-West corner to maximize travel distance
        south_starts = [
            (6.0, 6.0, 0.0),
            (8.0, 6.0, 0.0),
            (6.0, 8.0, 0.0),
            (8.0, 8.0, 0.0),
        ]
        return ScenarioConfig(
            name=name,
            description="Scenario G: Target located in distant corner far from robot spawn origins.",
            target_plate=target_plate,
            vehicles=[target],
            start_poses=south_starts,
            max_steps=1500,
            seed=130,
            zones=[UrbanZone("distant_corner", (30.0, 30.0, 48.0, 48.0), 0.80)],
        )

    elif name in ("low_prior_target", "scenario_h"):
        # Scenario H: Target located in a low-prior region; Bayesian belief updates from visual evidence
        target = TargetVehicle(
            target_id="target_car_low_prior",
            plate_number=target_plate,
            vehicle_class="sedan",
            color="dark_gray",
            position=(6.0, 25.0),
            is_stolen=True,
        )
        return ScenarioConfig(
            name=name,
            description="Scenario H: Target in low-probability zone; swarm discovers via Bayesian belief propagation.",
            target_plate=target_plate,
            vehicles=[target],
            start_poses=default_starts,
            max_steps=1000,
            seed=135,
            zones=[
                UrbanZone("central_distractor", (20.0, 20.0, 30.0, 30.0), 0.90),
                UrbanZone("west_low_prior", (2.0, 20.0, 10.0, 30.0), 0.20),
            ],
        )

    else:
        # Default scenario fallback
        return create_scenario("easy_target")
