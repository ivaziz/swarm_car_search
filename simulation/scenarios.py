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
            name="easy_target",
            description="Scenario A: Reachable target in central intersection. Expect fast discovery and confirmation.",
            target_plate=target_plate,
            vehicles=[target],
            start_poses=default_starts,
            max_steps=80,
            seed=100,
            zones=[UrbanZone("central_plaza", (20.0, 20.0, 30.0, 30.0), 0.85)],
        )

    elif name in ("hidden_target", "scenario_b"):
        # Scenario B: Target parked on East street corridor (45.0, 35.0), behind building 4
        target = TargetVehicle(
            target_id="target_car_hidden",
            plate_number=target_plate,
            vehicle_class="sedan",
            color="black",
            position=(45.0, 35.0),
            is_stolen=True,
        )
        return ScenarioConfig(
            name="hidden_target",
            description="Scenario B: Hidden target behind eastern building block. Requires street corridor exploration.",
            target_plate=target_plate,
            vehicles=[target],
            start_poses=default_starts,
            max_steps=120,
            seed=105,
            zones=[UrbanZone("east_corridor", (40.0, 28.0, 49.0, 42.0), 0.80)],
        )

    elif name in ("target_recruitment", "scenario_c"):
        # Scenario C: Target spotted in North avenue (25.0, 45.0)
        target = TargetVehicle(
            target_id="target_car_recruit",
            plate_number=target_plate,
            vehicle_class="sedan",
            color="silver",
            position=(25.0, 45.0),
            is_stolen=True,
        )
        return ScenarioConfig(
            name="target_recruitment",
            description="Scenario C: Candidate detection recruits nearby swarm robots via stigmergic pheromones.",
            target_plate=target_plate,
            vehicles=[target],
            start_poses=default_starts,
            max_steps=90,
            seed=110,
            zones=[UrbanZone("north_street", (15.0, 40.0, 35.0, 49.0), 0.85)],
        )

    elif name in ("failure_recovery", "scenario_d"):
        # Scenario D: Robot 1 assigned to target at (25.0, 25.0), fails at step 25, robot 2 recovers
        target = TargetVehicle(
            target_id="target_car_fault",
            plate_number=target_plate,
            vehicle_class="sedan",
            color="dark_gray",
            position=(25.0, 25.0),
            is_stolen=True,
        )
        return ScenarioConfig(
            name="failure_recovery",
            description="Scenario D: Robot failure during search. Swarm reclaims orphaned task and confirms target.",
            target_plate=target_plate,
            vehicles=[target],
            start_poses=default_starts,
            inject_failure_tick=25,
            fail_robot_id="robot_1",
            max_steps=90,
            seed=115,
            zones=[UrbanZone("central_plaza", (20.0, 20.0, 30.0, 30.0), 0.85)],
        )

    elif name in ("false_positive", "scenario_e"):
        # Scenario E: Multiple non-target vehicles rejected before target confirmed
        true_target = TargetVehicle(
            target_id="true_target",
            plate_number=target_plate,
            vehicle_class="sedan",
            color="dark_gray",
            position=(25.0, 35.0),
            is_stolen=True,
        )
        decoy_1 = TargetVehicle(
            target_id="decoy_car_1",
            plate_number="ABC-11223",
            vehicle_class="sedan",
            color="white",
            position=(16.0, 25.0),
            is_stolen=False,
        )
        decoy_2 = TargetVehicle(
            target_id="decoy_car_2",
            plate_number="KSA-77665",
            vehicle_class="sedan",
            color="red",
            position=(34.0, 25.0),
            is_stolen=False,
        )
        return ScenarioConfig(
            name="false_positive",
            description="Scenario E: Two decoy vehicles rejected via fuzzy plate matching; target vehicle confirmed.",
            target_plate=target_plate,
            vehicles=[true_target, decoy_1, decoy_2],
            start_poses=default_starts,
            max_steps=160,
            seed=120,
            zones=[UrbanZone("north_avenue", (20.0, 30.0, 30.0, 40.0), 0.85)],
        )

    elif name in ("multiple_vehicles", "scenario_f"):
        # Scenario F: Urban traffic fleet with target hidden among 4 total vehicles
        true_target = TargetVehicle(
            target_id="target_stolen_car",
            plate_number=target_plate,
            vehicle_class="sedan",
            color="dark_gray",
            position=(25.0, 25.0),
            is_stolen=True,
        )
        decoy_1 = TargetVehicle(
            target_id="fleet_car_1",
            plate_number="DXB-10101",
            vehicle_class="sedan",
            color="white",
            position=(25.0, 5.0),
            is_stolen=False,
        )
        decoy_2 = TargetVehicle(
            target_id="fleet_car_2",
            plate_number="AUH-99881",
            vehicle_class="suv",
            color="black",
            position=(45.0, 25.0),
            is_stolen=False,
        )
        decoy_3 = TargetVehicle(
            target_id="fleet_car_3",
            plate_number="SHJ-33442",
            vehicle_class="sedan",
            color="blue",
            position=(5.0, 25.0),
            is_stolen=False,
        )
        return ScenarioConfig(
            name="multiple_vehicles",
            description="Scenario F: Target correctly identified among a fleet of 4 urban vehicles.",
            target_plate=target_plate,
            vehicles=[true_target, decoy_1, decoy_2, decoy_3],
            start_poses=default_starts,
            max_steps=100,
            seed=125,
            zones=[UrbanZone("central_hub", (20.0, 20.0, 30.0, 30.0), 0.85)],
        )

    else:
        # Default scenario fallback
        return create_scenario("easy_target")
