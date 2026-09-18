#!/usr/bin/env python3
"""
Full Autonomous Swarm Simulation and Mission Visualization Script.
Simulates a multi-robot swarm searching an urban environment for a stolen vehicle.
Supports dynamic failure injection, live terminal HUD, SVG vector snapshots,
and publication-grade Matplotlib dashboard plots.
"""

import argparse
import math
import os
import random
import sys
import time

# Ensure project root is in sys.path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from interfaces.slam_interface import Pose2D
from simulation.environment import UrbanEnvironment, UrbanZone
from simulation.obstacle_generator import ObstacleGenerator
from slam_providers.mock_slam_provider import MockSLAMProvider
from swarm.states.robot_states import RobotState
from swarm.aco.aco_config import ACOConfig
from swarm.aco.aco_engine import ACOEngine
from swarm.task_allocation.allocation_strategy import ACOAllocationStrategy
from swarm.core.robot_agent import RobotAgent
from swarm.core.swarm_coordinator import SwarmCoordinator
from perception.vehicle_detector import TargetVehicle, SimulatedVehicleDetector
from perception.plate_recognizer import SimulatedPlateRecognizer
from perception.matching_engine import MatchingEngine
from perception.vehicle_detector import TargetVehicle, SimulatedVehicleDetector
from perception.plate_recognizer import SimulatedPlateRecognizer
from perception.matching_engine import MatchingEngine
from simulation.scenarios import create_scenario, ScenarioConfig
from visualization.swarm_visualizer import SwarmVisualizer, _HAS_MATPLOTLIB, _HAS_PIL

try:
    from PIL import Image
except ImportError:
    Image = None


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run Autonomous Swarm Urban Search & Target Identification Simulation"
    )
    parser.add_argument("--scenario", type=str, default="easy_target",
                        help="Named deterministic scenario: easy_target, hidden_target, target_recruitment, failure_recovery, false_positive, multiple_vehicles, or custom (default: easy_target)")
    parser.add_argument("--width", type=float, default=50.0, help="Urban grid width in meters (default: 50.0)")
    parser.add_argument("--height", type=float, default=50.0, help="Urban grid height in meters (default: 50.0)")
    parser.add_argument("--resolution", type=float, default=1.0, help="Grid resolution in meters (default: 1.0)")
    parser.add_argument("--robots", type=int, default=4, help="Number of ground search robots (default: 4)")
    parser.add_argument("--steps", type=int, default=100, help="Max simulation steps (default: 100)")
    parser.add_argument("--dt", type=float, default=0.2, help="Physics integration step dt in seconds (default: 0.2)")
    parser.add_argument("--target-x", type=float, default=25.0, help="Target car X coordinate (default: 25.0)")
    parser.add_argument("--target-y", type=float, default=25.0, help="Target car Y coordinate (default: 25.0)")
    parser.add_argument("--target-plate", type=str, default="DXB-88392", help="Stolen car license plate (default: DXB-88392)")
    parser.add_argument("--inject-failure-tick", type=int, default=None, help="Simulation tick to inject robot failure (default: None or scenario default)")
    parser.add_argument("--fail-robot", type=str, default="robot_1", help="ID of robot to inject failure into (default: robot_1)")
    parser.add_argument("--output-dir", type=str, default="output/simulation_run", help="Directory to save visual outputs")
    parser.add_argument("--render-every", type=int, default=10, help="Save frame every N ticks (default: 10)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    return parser.parse_args()


def setup_simulation(args):
    # Check if a named scenario is requested
    scenario: Optional[ScenarioConfig] = None
    scenario_arg = getattr(args, "scenario", None)
    if scenario_arg and scenario_arg.lower() != "custom":
        scenario = create_scenario(scenario_arg)
        seed = scenario.seed if getattr(args, "seed", 42) == 42 else args.seed
        steps = scenario.max_steps if getattr(args, "steps", 100) == 100 else args.steps
        inject_tick = scenario.inject_failure_tick if getattr(args, "inject_failure_tick", None) is None else args.inject_failure_tick
        fail_robot = scenario.fail_robot_id or getattr(args, "fail_robot", "robot_1")
    else:
        seed = getattr(args, "seed", 42)
        steps = getattr(args, "steps", 100)
        inject_tick = getattr(args, "inject_failure_tick", None)
        fail_robot = getattr(args, "fail_robot", "robot_1")

    random.seed(seed)

    # 1. Environment & Obstacles
    env = UrbanEnvironment(width=args.width, height=args.height, resolution=args.resolution)
    ObstacleGenerator.add_perimeter_walls(env, thickness_cells=1)
    w, h = args.width, args.height
    ObstacleGenerator.add_box_obstacle(env, w * 0.2, h * 0.2, w * 0.45, h * 0.45)
    ObstacleGenerator.add_box_obstacle(env, w * 0.55, h * 0.2, w * 0.8, h * 0.45)
    ObstacleGenerator.add_box_obstacle(env, w * 0.2, h * 0.55, w * 0.45, h * 0.8)
    ObstacleGenerator.add_box_obstacle(env, w * 0.55, h * 0.55, w * 0.8, h * 0.8)

    # Zones setup
    if scenario and scenario.zones:
        for z in scenario.zones:
            env.add_zone(z)
    else:
        env.add_zone(UrbanZone(
            name="target_search_zone",
            bounds=(args.target_x - 8.0, args.target_y - 8.0, args.target_x + 8.0, args.target_y + 8.0),
            vehicle_prior_probability=0.85,
        ))

    # 2. Vehicles & Target Watchlist Setup
    matcher = MatchingEngine(match_threshold=0.80)
    vehicles_in_world: List[TargetVehicle] = []
    primary_target: Optional[TargetVehicle] = None

    if scenario and scenario.vehicles:
        vehicles_in_world = scenario.vehicles
        for v in vehicles_in_world:
            if getattr(v, "is_target", False) or getattr(v, "is_stolen", False) or v.plate_number == scenario.target_plate:
                matcher.add_target_vehicle(v)
                if primary_target is None:
                    primary_target = v
    else:
        primary_target = TargetVehicle(
            target_id="target_stolen_sedan",
            plate_number=args.target_plate,
            vehicle_class="sedan",
            color="dark_gray",
            position=(args.target_x, args.target_y),
            is_stolen=True,
        )
        vehicles_in_world = [primary_target]
        matcher.add_target_vehicle(primary_target)

    # 3. ACO Coordination Setup
    aco_config = ACOConfig()
    aco_config.stochastic.deterministic = False
    aco_config.stochastic.temperature = 0.5
    strategy = ACOAllocationStrategy(ACOEngine(aco_config))

    coordinator = SwarmCoordinator(
        allocation_strategy=strategy,
        environment=env,
    )

    # 4. Robot Fleet Registration
    default_start_positions = [
        (6.0, 6.0, 0.0),
        (6.0, args.height - 6.0, -math.pi / 2.0),
        (args.width - 6.0, 6.0, math.pi / 2.0),
        (args.width - 6.0, args.height - 6.0, math.pi),
    ]
    start_positions = (scenario.start_poses if scenario and scenario.start_poses else default_start_positions)

    fleet_size = scenario.fleet_size if scenario else args.robots
    for i in range(fleet_size):
        rid = f"robot_{i+1}"
        sp = start_positions[i % len(start_positions)]
        slam = MockSLAMProvider(env)
        detector = SimulatedVehicleDetector(max_detection_range=14.0, fov_degrees=90.0)
        ocr = SimulatedPlateRecognizer(base_ocr_confidence=0.95)

        agent = RobotAgent(
            robot_id=rid,
            initial_pose=Pose2D(sp[0], sp[1], sp[2]),
            slam_provider=slam,
            detector=detector,
            ocr=ocr,
            matcher=matcher,
            known_vehicles=vehicles_in_world,
            zone_search_duration=4.0,
            reporting_duration=5.0,
        )
        coordinator.register_robot(agent)

    sim_params = {
        "steps": steps,
        "inject_failure_tick": inject_tick,
        "fail_robot": fail_robot,
        "primary_target": primary_target,
        "vehicles": vehicles_in_world,
        "scenario_name": scenario.name if scenario else "custom",
        "description": scenario.description if scenario else "Custom configuration",
    }
    return env, coordinator, vehicles_in_world, primary_target, sim_params


def run_simulation(args):
    os.makedirs(args.output_dir, exist_ok=True)
    env, coordinator, vehicles, primary_target, sim_params = setup_simulation(args)
    visualizer = SwarmVisualizer(env, title="Autonomous Swarm Search & Vehicle ID")

    steps = sim_params["steps"]
    inject_tick = sim_params["inject_failure_tick"]
    fail_robot = sim_params["fail_robot"]

    target_info = f"{primary_target.plate_number} at ({primary_target.position[0]:.1f}, {primary_target.position[1]:.1f})" if primary_target else "N/A"

    print("=" * 76)
    print("      AUTONOMOUS SWARM URBAN CAR SEARCH & IDENTIFICATION SIMULATION      ")
    print("=" * 76)
    print(f"Scenario Name   : {sim_params['scenario_name']} ({sim_params['description']})")
    print(f"Grid Dimensions : {args.width}m x {args.height}m (Resolution: {args.resolution}m)")
    print(f"Swarm Fleet Size: {len(coordinator.robots)} Autonomous Ground Vehicles")
    print(f"Target Vehicle  : {target_info}")
    print(f"World Vehicles  : {len(vehicles)} vehicle(s) present in urban simulation")
    if inject_tick is not None:
        print(f"Failure Injected: At tick {inject_tick} on {fail_robot}")
    else:
        print("Failure Injected: None")
    print(f"Output Directory: {args.output_dir}")
    print("=" * 76)

    saved_frames = []
    failure_injected = False
    target_announced = False
    start_wall_time = time.time()

    for tick in range(1, steps + 1):
        sim_time = tick * args.dt

        # Inject failure at designated tick
        if not failure_injected and inject_tick is not None and tick == inject_tick:
            print(f"\n[! ALERT] Injecting catastrophic failure into '{fail_robot}' at t={sim_time:.1f}s...")
            event = coordinator.simulate_robot_failure(
                robot_id=fail_robot,
                timestamp=sim_time,
                reason="SIMULATED_MOTOR_FAILURE",
            )
            failure_injected = True
            print(f"[+ RECOVERY] Swarm dynamically recovered orphaned tasks & marked wreck obstacle.\n")

        # Step coordination loop
        status = coordinator.step_coordination(timestamp=sim_time, dt=args.dt)

        # Check for target confirmation event
        if coordinator.target_found and not target_announced:
            target_announced = True
            conf_str = f"{coordinator.target_confirmation_confidence * 100:.1f}%" if coordinator.target_confirmation_confidence else "CONFIRMED"
            print("\n" + "*" * 76)
            print(f"🎯 TARGET CONFIRMED & REPORTED!")
            print(f"   Plate:      {coordinator.confirmed_target_plate or (primary_target.plate_number if primary_target else 'UNKNOWN')}")
            print(f"   Discovered: by '{coordinator.discovering_robot_id}' at t={coordinator.target_discovery_time:.1f}s")
            print(f"   Confidence: {conf_str}")
            print(f"   Location:   {coordinator.confirmed_target_location}")
            print("*" * 76 + "\n")

        # Periodic terminal logging
        if tick % args.render_every == 0 or (coordinator.target_found and tick == steps):
            active_cnt = status["robot_count"] - status["failed_robots_count"]
            exp_pct = status["average_exploration_ratio"] * 100.0
            found_str = f"CONFIRMED (t={coordinator.target_discovery_time:.1f}s)" if status["target_found"] else "SEARCHING"
            print(
                f"[t={sim_time:5.1f}s | Step {tick:3d}/{steps}] "
                f"Active: {active_cnt}/{status['robot_count']} | "
                f"Coverage: {exp_pct:5.1f}% | "
                f"Tasks: {status['completed_tasks_count']} done, {status['pending_tasks_count']} pend | "
                f"Target: {found_str}"
            )

        # Save visualization frame
        if tick % args.render_every == 0:
            frame_base = os.path.join(args.output_dir, f"frame_{tick:04d}")
            # Always save SVG (pure Python, vector)
            visualizer.render_svg_map(
                coordinator,
                sim_time,
                filepath=f"{frame_base}.svg",
                target_vehicles=vehicles,
            )
            # Save Matplotlib PNG if available
            if _HAS_MATPLOTLIB:
                png_path = f"{frame_base}.png"
                visualizer.render_matplotlib_dashboard(
                    coordinator,
                    sim_time,
                    target_vehicles=vehicles,
                    filepath=png_path,
                )
                saved_frames.append(png_path)

    total_wall_time = time.time() - start_wall_time
    final_status = coordinator.get_swarm_status(steps * args.dt)

    # Save final publication-grade summary figures
    final_svg = os.path.join(args.output_dir, "final_mission_summary.svg")
    visualizer.render_svg_map(
        coordinator,
        steps * args.dt,
        filepath=final_svg,
        target_vehicles=vehicles,
    )

    if _HAS_MATPLOTLIB:
        final_png = os.path.join(args.output_dir, "final_mission_summary.png")
        visualizer.render_matplotlib_dashboard(
            coordinator,
            steps * args.dt,
            target_vehicles=vehicles,
            filepath=final_png,
            dpi=160,
        )
        print(f"\n[+] Saved final summary dashboard: {final_png}")

    print(f"[+] Saved final vector SVG map    : {final_svg}")

    # Generate Animated GIF if PIL and Matplotlib frames are available
    if _HAS_PIL and saved_frames:
        gif_path = os.path.join(args.output_dir, "swarm_mission_animation.gif")
        try:
            images = [Image.open(f) for f in saved_frames]
            images[0].save(
                gif_path,
                save_all=True,
                append_images=images[1:],
                optimize=False,
                duration=250,
                loop=0,
            )
            print(f"[+] Saved animated mission GIF   : {gif_path}")
        except Exception as e:
            print(f"[-] Could not build GIF: {e}")

    # Print ASCII summary map
    ascii_summary = visualizer.render_ascii_map(
        coordinator,
        steps * args.dt,
        cols=45,
        rows=18,
        target_vehicles=vehicles,
    )
    print("\n" + ascii_summary)

    # Aggregate perception statistics
    tot_detections = sum(getattr(a, "total_detections_count", 0) for a in coordinator.robots.values())
    tot_false_pos = sum(getattr(a, "total_false_positives", 0) for a in coordinator.robots.values())
    tot_confirmations = sum(getattr(a, "total_target_confirmations", 0) for a in coordinator.robots.values())

    print("=" * 76)
    print("                         MISSION DEBRIEF REPORT                         ")
    print("=" * 76)
    print(f"Scenario              : {sim_params['scenario_name']}")
    print(f"Total Simulation Time : {steps * args.dt:.1f} seconds (Wall clock: {total_wall_time:.2f}s)")
    print(f"Exploration Coverage  : {final_status['average_exploration_ratio'] * 100.0:.1f}%")
    print(f"Completed Tasks       : {final_status['completed_tasks_count']}")
    print(f"Reassignment Events   : {final_status['reassignment_events_count']}")
    print(f"Failed Robots Handled : {final_status['failed_robots_count']}")
    print(f"Recovered Orphan Tasks: {final_status['recovered_tasks_count']}")
    print(f"Vehicle Detections    : {tot_detections} visual sighting(s)")
    print(f"False Positives Disc. : {tot_false_pos} non-target vehicle(s) rejected")
    print(f"Target Confirmations  : {tot_confirmations}")
    if final_status["target_found"]:
        conf_pct = f"{coordinator.target_confirmation_confidence * 100:.1f}%" if coordinator.target_confirmation_confidence else "100.0%"
        print(f"Target Vehicle Status : 🎯 TARGET CONFIRMED & REPORTED")
        print(f"  - Target Plate      : {coordinator.confirmed_target_plate or (primary_target.plate_number if primary_target else 'UNKNOWN')}")
        print(f"  - Discovered By     : {coordinator.discovering_robot_id}")
        print(f"  - Discovery Time    : {coordinator.target_discovery_time:.1f} s")
        print(f"  - Match Confidence  : {conf_pct}")
        print(f"  - Physical Location : {final_status['confirmed_target_location']}")
    else:
        print(f"Target Vehicle Status : NOT FOUND")
    print("=" * 76 + "\n")


def main():
    cli_args = parse_args()
    run_simulation(cli_args)


if __name__ == "__main__":
    main()
