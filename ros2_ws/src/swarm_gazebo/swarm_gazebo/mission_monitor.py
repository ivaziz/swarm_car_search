"""
ROS 2 Mission Monitor and Telemetry Dashboard Node for SwarmX.
Visualizes multi-robot trajectories, logs mission statistics, and generates debrief reports.
"""

import math
import os
import time
from typing import Dict, List, Optional, Tuple

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from nav_msgs.msg import Odometry

from interfaces.slam_interface import Pose2D
from simulation.environment import UrbanEnvironment
from simulation.obstacle_generator import ObstacleGenerator
from visualization.swarm_visualizer import SwarmVisualizer


class GazeboMissionMonitor(Node):
    """
    Monitors Gazebo swarm progress, displays terminal HUD,
    and saves vector SVG map snapshots and mission debrief logs.
    """

    def __init__(
        self,
        robot_ids: List[str],
        output_dir: str = "output/gazebo_simulation_run",
        env_width: float = 50.0,
        env_height: float = 50.0,
    ) -> None:
        super().__init__("mission_monitor_node")
        self.robot_ids = robot_ids
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        self.env = UrbanEnvironment(width=env_width, height=env_height, resolution=1.0)
        ObstacleGenerator.add_perimeter_walls(self.env, thickness_cells=1)
        w, h = env_width, env_height
        ObstacleGenerator.add_box_obstacle(self.env, w * 0.2, h * 0.2, w * 0.45, h * 0.45)
        ObstacleGenerator.add_box_obstacle(self.env, w * 0.55, h * 0.2, w * 0.8, h * 0.45)
        ObstacleGenerator.add_box_obstacle(self.env, w * 0.2, h * 0.55, w * 0.45, h * 0.8)
        ObstacleGenerator.add_box_obstacle(self.env, w * 0.55, h * 0.55, w * 0.8, h * 0.8)

        self.robot_poses: Dict[str, Pose2D] = {}
        for rid in robot_ids:
            self.robot_poses[rid] = Pose2D(0.0, 0.0, 0.0)
            self.create_subscription(
                Odometry,
                f"/{rid}/odom",
                lambda msg, r=rid: self._on_odom(r, msg),
                10,
            )

        self.create_subscription(String, "/swarm/status", self._on_status, 10)
        self.create_subscription(String, "/swarm/perception_events", self._on_event, 10)

        self.last_status_str = ""
        self.target_confirmed = False
        self.get_logger().info(f"GazeboMissionMonitor active. Output dir: {output_dir}")

    def _on_odom(self, robot_id: str, msg: Odometry) -> None:
        pos = msg.pose.pose.position
        ori = msg.pose.pose.orientation
        siny_cosp = 2.0 * (ori.w * ori.z + ori.x * ori.y)
        cosy_cosp = 1.0 - 2.0 * (ori.y * ori.y + ori.z * ori.z)
        yaw = math.atan2(siny_cosp, cosy_cosp)
        self.robot_poses[robot_id] = Pose2D(x=float(pos.x), y=float(pos.y), theta=yaw)

    def _on_status(self, msg: String) -> None:
        self.last_status_str = msg.data

    def _on_event(self, msg: String) -> None:
        if msg.data.startswith("TARGET_MATCH:"):
            self.target_confirmed = True
