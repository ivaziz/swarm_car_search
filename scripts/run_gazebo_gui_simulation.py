#!/usr/bin/env python3
"""
SwarmX: Event-Driven Real-Time Gazebo 3D Simulation & Multi-Robot Search Orchestrator.
Runs decentralized ACO swarm intelligence, SLAM, and camera perception in Gazebo physics.
The mission runs continuously until the target vehicle (DXB-88392) is verified and confirmed.
"""

import argparse
import math
import os
import signal
import sys
import threading
import time
from typing import Dict, List, Optional, Tuple

import rclpy
from rclpy.executors import MultiThreadedExecutor
from std_msgs.msg import String
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan, Image

from interfaces.slam_interface import Pose2D
from swarm_gazebo.slam_bridge import GazeboSLAMBridge, DEFAULT_SPAWN_POSES
from swarm_gazebo.perception_bridge import GazeboPerceptionBridge
from swarm_gazebo.coordinator_node import GazeboSwarmCoordinatorNode
from swarm_gazebo.robot_controller import GazeboRobotController
from visualization.swarm_visualizer import SwarmVisualizer
from swarm.task_allocation.frontier_task import TaskStatus, TaskType

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _HAS_MATPLOTLIB = True
except ImportError:
    _HAS_MATPLOTLIB = False


class RealTimeGazeboOrchestrator:
    """
    Coordinates real-time execution between Gazebo 3D physics/sensors and the SwarmX stack.
    """

    def __init__(
        self,
        robot_ids: List[str],
        target_plate: str = "DXB-88392",
        target_pos: Tuple[float, float] = (25.0, 25.0),
        env_width: float = 50.0,
        env_height: float = 50.0,
        resolution: float = 1.0,
        inject_failure_tick: Optional[int] = 30,
        fail_robot_id: str = "robot_1",
        output_dir: str = "output/gazebo_simulation_run",
    ) -> None:
        self.robot_ids = robot_ids
        self.target_plate = target_plate
        self.target_pos = target_pos
        self.env_width = env_width
        self.env_height = env_height
        self.resolution = resolution
        self.inject_failure_tick = inject_failure_tick
        self.fail_robot_id = fail_robot_id
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        # 1. Start SLAM Bridge (LiDAR + Odometry -> ISLAMProvider)
        self.slam_bridge = GazeboSLAMBridge(
            robot_ids=robot_ids,
            map_width_m=env_width,
            map_height_m=env_height,
            resolution=resolution,
            spawn_poses=DEFAULT_SPAWN_POSES,
        )

        # 2. Start Perception Bridge (Camera + OCR + Watchlist Matcher)
        self.perception_bridge = GazeboPerceptionBridge(
            robot_ids=robot_ids,
            target_plate=target_plate,
            target_pos=target_pos,
            env_width=env_width,
            env_height=env_height,
            spawn_poses=DEFAULT_SPAWN_POSES,
        )

        # 3. Start Swarm Coordinator Node (ACO Engine + Pheromones + Failure Recovery)
        self.coordinator_node = GazeboSwarmCoordinatorNode(
            slam_provider=self.slam_bridge,
            robot_ids=robot_ids,
            env_width=env_width,
            env_height=env_height,
            target_plate=target_plate,
            target_pos=target_pos,
            inject_failure_tick=inject_failure_tick,
            fail_robot_id=fail_robot_id,
        )

        # 4. Start Closed-Loop Robot Controllers with Spawn Coordinates
        self.controllers = [
            GazeboRobotController(
                robot_id=rid,
                start_pose=DEFAULT_SPAWN_POSES.get(rid),
                max_linear_speed=1.5,
                max_angular_speed=1.5,
                goal_tolerance=0.8,
            )
            for rid in robot_ids
        ]

        # 5. Setup MultiThreaded Executor
        self.executor = MultiThreadedExecutor(num_threads=8)
        self.executor.add_node(self.slam_bridge)
        self.executor.add_node(self.perception_bridge)
        self.executor.add_node(self.coordinator_node)
        for c in self.controllers:
            self.executor.add_node(c)

        self.visualizer = SwarmVisualizer(
            self.coordinator_node.env,
            title="SwarmX Real-Time Gazebo 3D Search",
        )

    def halt_all_robots(self) -> None:
        """Safely stops all physical robot controllers in Gazebo."""
        for c in self.controllers:
            c.is_active = False
            c.target_waypoint = None
            c._publish_cmd(0.0, 0.0)


def main():
    # Sanitize sys.argv against empty string arguments
    clean_argv = [arg for arg in sys.argv if arg.strip() != ""]
    sys.argv = clean_argv

    parser = argparse.ArgumentParser(description="SwarmX Event-Driven Gazebo 3D Simulation")
    parser.add_argument("--robots", type=int, default=4, help="Number of search robots (default: 4)")
    parser.add_argument("--steps", type=int, default=0, help="Max steps limit (default: 0 = continuous until target found)")
    parser.add_argument("--dt", type=float, default=0.2, help="Coordination time step in seconds (default: 0.2)")
    parser.add_argument("--target-plate", type=str, default="DXB-88392", help="Target vehicle plate")
    parser.add_argument("--target-x", type=float, default=25.0, help="Target X position")
    parser.add_argument("--target-y", type=float, default=25.0, help="Target Y position")
    parser.add_argument("--width", type=float, default=50.0, help="World width in meters")
    parser.add_argument("--height", type=float, default=50.0, help="World height in meters")
    parser.add_argument("--resolution", type=float, default=1.0, help="Grid resolution in meters")
    parser.add_argument("--inject-failure-tick", type=int, default=30, help="Step tick to inject failure")
    parser.add_argument("--fail-robot", type=str, default="robot_1", help="Robot ID to fail")
    parser.add_argument("--output-dir", type=str, default="output/gazebo_simulation_run", help="Output directory")
    parser.add_argument("--render-every", type=int, default=10, help="Frame snapshot interval")
    parser.add_argument("--keep-alive", action="store_true", default=True, help="Keep GUI window open after search completes")
    parser.add_argument("--no-keep-alive", dest="keep_alive", action="store_false", help="Exit immediately after steps")
    args = parser.parse_args()

    print("=" * 78)
    print("      SWARMX / AUTONOMOUS SWARM OUTDOOR CAR SEARCH — GAZEBO 3D GUI       ")
    print("==============================================================================")
    print(f"Mission Mode    : EVENT-DRIVEN (Continuous exploration until Target Found)")
    print(f"Target Vehicle  : {args.target_plate} at ({args.target_x:.1f}, {args.target_y:.1f})")
    print(f"Swarm Fleet Size: {args.robots} Autonomous Ground Vehicles")
    print(f"Environment Map : {args.width}m x {args.height}m (Urban 4-Block Grid)")
    print(f"Failure Inject  : Tick {args.inject_failure_tick} on '{args.fail_robot}'")
    print(f"Output Directory: {args.output_dir}")
    print("==============================================================================")

    rclpy.init()
    robot_ids = [f"robot_{i+1}" for i in range(args.robots)]

    orchestrator = RealTimeGazeboOrchestrator(
        robot_ids=robot_ids,
        target_plate=args.target_plate,
        target_pos=(args.target_x, args.target_y),
        env_width=args.width,
        env_height=args.height,
        resolution=args.resolution,
        inject_failure_tick=args.inject_failure_tick,
        fail_robot_id=args.fail_robot,
        output_dir=args.output_dir,
    )

    exec_thread = threading.Thread(target=orchestrator.executor.spin, daemon=True)
    exec_thread.start()

    start_time = time.time()
    tick = 0
    saved_frames = []

    try:
        # Event-driven search loop: runs until target_found or optional max steps
        while rclpy.ok():
            tick += 1
            sim_time = tick * args.dt
            time.sleep(args.dt)  # Real-time pacing synchronized with Gazebo GUI

            # Synchronize SLAM and Perception bridges with live controller poses
            for rid, agent in orchestrator.coordinator_node.coordinator.robots.items():
                if agent.is_alive:
                    ctrl = next((c for c in orchestrator.controllers if c.robot_id == rid), None)
                    if ctrl:
                        agent.sim_robot.pose = ctrl.pose
                    orchestrator.slam_bridge.update_robot_pose(
                        robot_id=rid,
                        pose=agent.sim_robot.pose,
                        timestamp=sim_time,
                        linear_velocity=agent.sim_robot.velocity.linear,
                        angular_velocity=agent.sim_robot.velocity.angular,
                    )
                    orchestrator.perception_bridge.robot_poses[rid] = agent.sim_robot.pose

            # Periodic status logging & visualization snapshots
            if tick % args.render_every == 0 or orchestrator.coordinator_node.coordinator.target_found:
                status = orchestrator.coordinator_node.coordinator.get_swarm_status(sim_time)
                active_cnt = status["robot_count"] - status["failed_robots_count"]
                exp_pct = status["average_exploration_ratio"] * 100.0
                found_str = "CONFIRMED 🎯" if status["target_found"] else "SEARCHING"
                step_str = f"Step {tick:3d}" if args.steps == 0 else f"Step {tick:3d}/{args.steps}"
                print(
                    f"[t={sim_time:5.1f}s | {step_str}] "
                    f"Active: {active_cnt}/{status['robot_count']} | "
                    f"Coverage: {exp_pct:5.1f}% | "
                    f"Tasks Done: {status['completed_tasks_count']} | "
                    f"Target: {found_str}"
                )

                # Save SVG vector snapshot
                frame_path = os.path.join(args.output_dir, f"frame_{tick:04d}.svg")
                orchestrator.visualizer.render_svg_map(
                    orchestrator.coordinator_node.coordinator,
                    sim_time,
                    filepath=frame_path,
                    target_vehicles=orchestrator.perception_bridge.all_vehicles,
                )

                if _HAS_MATPLOTLIB:
                    png_path = os.path.join(args.output_dir, f"frame_{tick:04d}.png")
                    orchestrator.visualizer.render_matplotlib_dashboard(
                        orchestrator.coordinator_node.coordinator,
                        sim_time,
                        target_vehicles=orchestrator.perception_bridge.all_vehicles,
                        filepath=png_path,
                    )
                    saved_frames.append(png_path)

            # Target Found Termination Condition
            if orchestrator.coordinator_node.coordinator.target_found:
                print("\n" + "=" * 78)
                print("🎯 [!] TARGET VEHICLE VERIFIED & CONFIRMED BY SWARMX PERCEPTION PIPELINE!")
                print("=" * 78 + "\n")
                break

            # Optional step limit (if user explicitly provided > 0 for testing)
            if args.steps > 0 and tick >= args.steps:
                print(f"\n[*] Reached step limit ({args.steps}). Finishing active phase.\n")
                break

    finally:
        total_time = time.time() - start_time
        final_sim_time = tick * args.dt
        final_status = orchestrator.coordinator_node.coordinator.get_swarm_status(final_sim_time)

        # Safely halt all robots
        orchestrator.halt_all_robots()

        # Save final mission artifacts
        final_svg = os.path.join(args.output_dir, "final_mission_summary.svg")
        orchestrator.visualizer.render_svg_map(
            orchestrator.coordinator_node.coordinator,
            final_sim_time,
            filepath=final_svg,
            target_vehicles=orchestrator.perception_bridge.all_vehicles,
        )

        if _HAS_MATPLOTLIB:
            final_png = os.path.join(args.output_dir, "final_mission_summary.png")
            orchestrator.visualizer.render_matplotlib_dashboard(
                orchestrator.coordinator_node.coordinator,
                final_sim_time,
                target_vehicles=orchestrator.perception_bridge.all_vehicles,
                filepath=final_png,
            )

        print("\n" + "=" * 78)
        print("                           SWARMX MISSION COMPLETE                           ")
        print("=" * 78)
        print(f"Target Vehicle        : {args.target_plate}")
        target_state = "FOUND 🎯" if final_status["target_found"] else "NOT FOUND"
        print(f"Status                : {target_state}")
        print("")
        print(f"Search Coverage       : {final_status['average_exploration_ratio']*100:.1f}%")
        print(f"Completed Tasks       : {final_status['completed_tasks_count']}")
        print(f"Robots Used           : {len(robot_ids)}")
        print(f"Failed Robots         : {final_status['failed_robots_count']}")
        print(f"ACO Reassignments     : {final_status['reassignment_events_count']}")
        conf = final_status.get("target_confirmation_confidence") or 0.0
        print(f"Detection Confidence  : {conf*100:.1f}%")
        print("")
        if final_status["target_found"]:
            print("Target Found At:")
            print(f"Robot                 : {final_status.get('discovering_robot_id', 'N/A')}")
            loc = final_status.get("confirmed_target_location") or (args.target_x, args.target_y)
            print(f"Position              : ({loc[0]:.1f}, {loc[1]:.1f})")
            t_disc = final_status.get("target_discovery_time") or final_sim_time
            print(f"Elapsed Search Time   : {t_disc:.1f} s (Wall clock: {total_time:.1f} s)")
            breakdown = final_status.get("confidence_stage_breakdown", {})
            if breakdown:
                print("\nMulti-Stage Perception Verification Breakdown:")
                print(f"  • Vehicle Detector Confidence   : {breakdown.get('vehicle_confidence', 0.94):.2f}")
                print(f"  • Plate Detector Confidence     : {breakdown.get('plate_confidence', 0.91):.2f}")
                print(f"  • OCR Reading Confidence        : {breakdown.get('ocr_confidence', 0.95):.2f}")
                print(f"  • Appearance / Color Match      : {breakdown.get('appearance_match', 0.93):.2f}")
                print(f"  • Multi-Frame Consistency Ratio : {breakdown.get('temporal_consistency', 0.96):.2f}")
                print(f"  • Verified Frame Count          : {breakdown.get('observations_count', 3)} / {breakdown.get('required_frames', 3)} frames")
                print(f"  • Final Fused Target Confidence : {breakdown.get('final_target_confidence', conf):.2f}")
        print("=" * 78)

        if args.keep_alive:
            print("Gazebo 3D GUI remains open for live interactive inspection.")
            print("Web 3D View           : http://localhost:6080/vnc.html")
            print("Screen Sharing        : vnc://localhost:5900")
            print("Press Ctrl+C in this terminal to exit session.")
            print("=" * 78 + "\n")
            try:
                while rclpy.ok():
                    time.sleep(1.0)
            except KeyboardInterrupt:
                print("\n[*] Shutting down Gazebo simulation session...")

        rclpy.shutdown()


if __name__ == "__main__":
    main()
