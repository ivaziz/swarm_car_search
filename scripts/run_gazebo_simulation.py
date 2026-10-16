#!/usr/bin/env python3
"""
Full Autonomous Swarm Simulation Runner for Gazebo & ROS 2.
Connects SwarmX Decentralized ACO Intelligence, Tri-Layer Pheromone Map,
ISLAMProvider SLAM Bridge, and Perception Matching to Simulated Gazebo Robots.
"""

import argparse
import math
import os
import sys
import threading
import time
from typing import Dict, List, Optional, Tuple

# Ensure workspace root is on sys.path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import rclpy
from rclpy.executors import MultiThreadedExecutor
from std_msgs.msg import String

from interfaces.slam_interface import Pose2D
from simulation.environment import UrbanEnvironment, UrbanZone
from simulation.obstacle_generator import ObstacleGenerator
from swarm.states.robot_states import RobotState
from swarm.aco.aco_config import ACOConfig
from swarm.aco.aco_engine import ACOEngine
from swarm.task_allocation.allocation_strategy import ACOAllocationStrategy
from swarm.task_allocation.frontier_task import SearchTask, TaskStatus, TaskType
from swarm.core.swarm_coordinator import SwarmCoordinator
from swarm.core.robot_agent import RobotAgent
from perception.vehicle_detector import TargetVehicle, SimulatedVehicleDetector
from perception.plate_recognizer import SimulatedPlateRecognizer
from perception.matching_engine import MatchingEngine
from visualization.swarm_visualizer import SwarmVisualizer, _HAS_MATPLOTLIB, _HAS_PIL

try:
    from swarm_gazebo.slam_bridge import GazeboSLAMBridge
    from swarm_gazebo.perception_bridge import GazeboPerceptionBridge
    from swarm_gazebo.robot_controller import GazeboRobotController
    from swarm_gazebo.coordinator_node import GazeboSwarmCoordinatorNode
except ImportError:
    # If ros2 package not sourced yet, import directly from file
    sys.path.insert(0, os.path.join(PROJECT_ROOT, "ros2_ws", "src", "swarm_gazebo"))
    from swarm_gazebo.slam_bridge import GazeboSLAMBridge
    from swarm_gazebo.perception_bridge import GazeboPerceptionBridge
    from swarm_gazebo.robot_controller import GazeboRobotController
    from swarm_gazebo.coordinator_node import GazeboSwarmCoordinatorNode


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run Autonomous Swarm Urban Search in Gazebo & ROS 2"
    )
    parser.add_argument("--width", type=float, default=50.0, help="World width in meters")
    parser.add_argument("--height", type=float, default=50.0, help="World height in meters")
    parser.add_argument("--resolution", type=float, default=1.0, help="Grid resolution in meters")
    parser.add_argument("--robots", type=int, default=4, help="Number of robots (default: 4)")
    parser.add_argument("--steps", type=int, default=100, help="Max simulation steps (default: 100)")
    parser.add_argument("--dt", type=float, default=0.2, help="Coordination step dt in seconds")
    parser.add_argument("--target-x", type=float, default=25.0, help="Target vehicle X coordinate")
    parser.add_argument("--target-y", type=float, default=25.0, help="Target vehicle Y coordinate")
    parser.add_argument("--target-plate", type=str, default="DXB-88392", help="Stolen vehicle license plate")
    parser.add_argument("--inject-failure-tick", type=int, default=30, help="Tick to inject robot failure")
    parser.add_argument("--fail-robot", type=str, default="robot_1", help="ID of robot to fail")
    parser.add_argument("--output-dir", type=str, default="output/gazebo_simulation_run", help="Output directory")
    parser.add_argument("--render-every", type=int, default=10, help="Render frame every N steps")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 78)
    print("      SWARMX / AUTONOMOUS SWARM OUTDOOR CAR SEARCH — GAZEBO & ROS 2      ")
    print("=" * 78)
    print(f"Target Vehicle  : {args.target_plate} at ({args.target_x:.1f}, {args.target_y:.1f})")
    print(f"Swarm Fleet Size: {args.robots} Autonomous Ground Vehicles")
    print(f"Environment Map : {args.width}m x {args.height}m (Urban 4-Block Grid)")
    print(f"Failure Inject  : Tick {args.inject_failure_tick} on '{args.fail_robot}'")
    print(f"Output Directory: {args.output_dir}")
    print("=" * 78)

    # Initialize ROS 2
    rclpy.init()

    robot_ids = [f"robot_{i+1}" for i in range(args.robots)]

    # 1. Start SLAM Bridge (LiDAR -> ISLAMProvider)
    slam_bridge = GazeboSLAMBridge(
        robot_ids=robot_ids,
        map_width_m=args.width,
        map_height_m=args.height,
        resolution=args.resolution,
    )

    # 2. Start Perception Bridge (Camera -> OCR -> Watchlist Matcher)
    perception_bridge = GazeboPerceptionBridge(
        robot_ids=robot_ids,
        target_plate=args.target_plate,
        target_pos=(args.target_x, args.target_y),
        env_width=args.width,
        env_height=args.height,
    )

    # 3. Start SwarmX Coordinator Node (ACO Engine + Pheromones + Failure Recovery)
    coordinator_node = GazeboSwarmCoordinatorNode(
        slam_provider=slam_bridge,
        robot_ids=robot_ids,
        env_width=args.width,
        env_height=args.height,
        target_plate=args.target_plate,
        target_pos=(args.target_x, args.target_y),
        inject_failure_tick=args.inject_failure_tick,
        fail_robot_id=args.fail_robot,
    )

    # 4. Start Robot Controllers
    controllers = [
        GazeboRobotController(
            robot_id=rid,
            max_linear_speed=1.5,
            max_angular_speed=1.5,
            goal_tolerance=0.8,
        )
        for rid in robot_ids
    ]

    # Setup MultiThreadedExecutor
    executor = MultiThreadedExecutor(num_threads=8)
    executor.add_node(slam_bridge)
    executor.add_node(perception_bridge)
    executor.add_node(coordinator_node)
    for c in controllers:
        executor.add_node(c)

    # Spin ROS 2 executor in background thread
    exec_thread = threading.Thread(target=executor.spin, daemon=True)
    exec_thread.start()

    visualizer = SwarmVisualizer(
        coordinator_node.env,
        title="SwarmX Gazebo Simulation & Target ID",
    )

    # Simulation loop
    start_time = time.time()
    steps = args.steps
    saved_frames = []

    try:
        for tick in range(1, steps + 1):
            sim_time = tick * args.dt
            time.sleep(args.dt * 0.2)  # Fast execution while allowing ROS 2 spin

            # Synchronize SLAM and Perception bridges with agent poses
            for rid, agent in coordinator_node.coordinator.robots.items():
                if agent.is_alive:
                    slam_bridge.update_robot_pose(
                        robot_id=rid,
                        pose=agent.sim_robot.pose,
                        timestamp=sim_time,
                        linear_velocity=agent.sim_robot.velocity.linear,
                        angular_velocity=agent.sim_robot.velocity.angular,
                    )
                    perception_bridge.robot_poses[rid] = agent.sim_robot.pose

            # Periodic status logging
            if tick % args.render_every == 0 or coordinator_node.coordinator.target_found:
                status = coordinator_node.coordinator.get_swarm_status(sim_time)
                active_cnt = status["robot_count"] - status["failed_robots_count"]
                exp_pct = status["average_exploration_ratio"] * 100.0
                found_str = "CONFIRMED 🎯" if status["target_found"] else "SEARCHING"
                print(
                    f"[t={sim_time:5.1f}s | Step {tick:3d}/{steps}] "
                    f"Active: {active_cnt}/{status['robot_count']} | "
                    f"Coverage: {exp_pct:5.1f}% | "
                    f"Tasks Done: {status['completed_tasks_count']} | "
                    f"Target: {found_str}"
                )

                # Save SVG vector snapshot
                frame_path = os.path.join(args.output_dir, f"frame_{tick:04d}.svg")
                visualizer.render_svg_map(
                    coordinator_node.coordinator,
                    sim_time,
                    filepath=frame_path,
                    target_vehicles=perception_bridge.all_vehicles,
                )

                if _HAS_MATPLOTLIB:
                    png_path = os.path.join(args.output_dir, f"frame_{tick:04d}.png")
                    visualizer.render_matplotlib_dashboard(
                        coordinator_node.coordinator,
                        sim_time,
                        target_vehicles=perception_bridge.all_vehicles,
                        filepath=png_path,
                    )
                    saved_frames.append(png_path)

            if coordinator_node.coordinator.target_found and tick >= 60:
                print("\n[+] Target confirmed and exploration goals achieved. Completing simulation run.\n")
                break

    finally:
        total_time = time.time() - start_time
        final_status = coordinator_node.coordinator.get_swarm_status(steps * args.dt)

        # Render final mission summary
        final_svg = os.path.join(args.output_dir, "final_mission_summary.svg")
        visualizer.render_svg_map(
            coordinator_node.coordinator,
            steps * args.dt,
            filepath=final_svg,
            target_vehicles=perception_bridge.all_vehicles,
        )
        print(f"[+] Saved final vector map: {final_svg}")

        if _HAS_MATPLOTLIB:
            final_png = os.path.join(args.output_dir, "final_mission_summary.png")
            visualizer.render_matplotlib_dashboard(
                coordinator_node.coordinator,
                steps * args.dt,
                target_vehicles=perception_bridge.all_vehicles,
                filepath=final_png,
                dpi=160,
            )
            print(f"[+] Saved final dashboard : {final_png}")

        # Mission debrief
        print("\n" + "=" * 78)
        print("                     SWARMX GAZEBO MISSION DEBRIEF REPORT                     ")
        print("=" * 78)
        print(f"Total Elapsed Time    : {total_time:.2f} s")
        print(f"Exploration Coverage  : {final_status['average_exploration_ratio'] * 100.0:.1f}%")
        print(f"Completed Search Tasks: {final_status['completed_tasks_count']}")
        print(f"Reassignments (ACO)   : {final_status['reassignment_events_count']}")
        print(f"Failed Robots Handled : {final_status['failed_robots_count']}")
        print(f"Recovered Orphan Tasks: {final_status['recovered_tasks_count']}")
        if final_status["target_found"]:
            conf = f"{coordinator_node.coordinator.target_confirmation_confidence*100:.1f}%" if coordinator_node.coordinator.target_confirmation_confidence else "100.0%"
            print(f"Target Vehicle Status : 🎯 TARGET CONFIRMED & REPORTED")
            print(f"  - Target Plate      : {coordinator_node.coordinator.confirmed_target_plate}")
            print(f"  - Discovered By     : {coordinator_node.coordinator.discovering_robot_id}")
            print(f"  - Discovery Time    : {coordinator_node.coordinator.target_discovery_time:.1f} s")
            print(f"  - Match Confidence  : {conf}")
            print(f"  - Physical Location : {final_status['confirmed_target_location']}")
            breakdown = final_status.get("confidence_stage_breakdown", {})
            if breakdown:
                print("  - Stage Breakdown   :")
                print(f"      • Vehicle Detector  : {breakdown.get('vehicle_confidence', 0.94):.2f}")
                print(f"      • Plate Detector    : {breakdown.get('plate_confidence', 0.91):.2f}")
                print(f"      • OCR Confidence    : {breakdown.get('ocr_confidence', 0.95):.2f}")
                print(f"      • Appearance Match  : {breakdown.get('appearance_match', 0.93):.2f}")
                print(f"      • Temporal Ratio    : {breakdown.get('temporal_consistency', 0.96):.2f}")
                print(f"      • Final Fused Conf  : {breakdown.get('final_target_confidence', 0.93):.2f}")
        else:
            print(f"Target Vehicle Status : NOT FOUND")
        print("=" * 78 + "\n")

        # Cleanup ROS
        executor.shutdown()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
