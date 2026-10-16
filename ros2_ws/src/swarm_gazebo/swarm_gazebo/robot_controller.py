"""
ROS 2 Robot Controller Node for SwarmX.
Implements proportional closed-loop waypoint tracking, reactive obstacle avoidance,
and publishes velocity commands to Gazebo via /{robot_id}/cmd_vel.
"""

import math
import time
from typing import Optional, Tuple, Dict

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from std_msgs.msg import String

from interfaces.slam_interface import Pose2D
from simulation.environment import UrbanEnvironment
from simulation.obstacle_generator import ObstacleGenerator


# Default spawn poses matching urban_search_world.world
DEFAULT_SPAWN_POSES: Dict[str, Pose2D] = {
    "robot_1": Pose2D(6.0, 6.0, 0.0),
    "robot_2": Pose2D(6.0, 44.0, -math.pi / 2.0),
    "robot_3": Pose2D(44.0, 6.0, math.pi / 2.0),
    "robot_4": Pose2D(44.0, 44.0, math.pi),
}


class GazeboRobotController(Node):
    """
    Onboard motion controller for an individual robot in Gazebo.
    Steers towards waypoints assigned by the SwarmX ACO Coordinator,
    handles reactive collision avoidance, and reports task completion.
    """

    def __init__(
        self,
        robot_id: str,
        start_pose: Optional[Pose2D] = None,
        max_linear_speed: float = 1.2,
        max_angular_speed: float = 1.5,
        goal_tolerance: float = 0.8,
    ) -> None:
        super().__init__(f"{robot_id}_controller")
        self.robot_id = robot_id
        self.start_pose = start_pose or DEFAULT_SPAWN_POSES.get(robot_id, Pose2D(0.0, 0.0, 0.0))
        self.max_linear_speed = max_linear_speed
        self.max_angular_speed = max_angular_speed
        self.goal_tolerance = goal_tolerance

        # Kinematic gains
        self.kp_linear = 0.8
        self.kp_angular = 2.0

        # State
        self.pose = Pose2D(self.start_pose.x, self.start_pose.y, self.start_pose.theta)
        self.target_waypoint: Optional[Tuple[float, float]] = None
        self.current_task_id: Optional[str] = None
        self.is_active = True
        self.latest_scan_ranges: list = []

        # Environment model for collision lookahead
        self.env = UrbanEnvironment(width=50.0, height=50.0, resolution=1.0)
        ObstacleGenerator.add_perimeter_walls(self.env, thickness_cells=1)
        w, h = 50.0, 50.0
        ObstacleGenerator.add_box_obstacle(self.env, w * 0.2, h * 0.2, w * 0.45, h * 0.45)
        ObstacleGenerator.add_box_obstacle(self.env, w * 0.55, h * 0.2, w * 0.8, h * 0.45)
        ObstacleGenerator.add_box_obstacle(self.env, w * 0.2, h * 0.55, w * 0.45, h * 0.8)
        ObstacleGenerator.add_box_obstacle(self.env, w * 0.55, h * 0.55, w * 0.8, h * 0.8)

        # ROS 2 Pub/Sub
        self.cmd_pub = self.create_publisher(Twist, f"/{robot_id}/cmd_vel", 10)
        self.status_pub = self.create_publisher(String, f"/{robot_id}/task_status", 10)

        self.create_subscription(
            Odometry,
            f"/{robot_id}/odom",
            self._on_odom,
            10,
        )

        qos_sensor = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )
        self.create_subscription(
            LaserScan,
            f"/{robot_id}/scan",
            self._on_scan,
            qos_sensor,
        )

        self.create_subscription(
            String,
            f"/{robot_id}/assigned_task",
            self._on_task_assignment,
            10,
        )

        # Control loop at 20Hz
        self.timer = self.create_timer(0.05, self._control_step)
        self.get_logger().info(f"GazeboRobotController started for '{robot_id}'.")

    def _on_odom(self, msg: Odometry) -> None:
        pos = msg.pose.pose.position
        ori = msg.pose.pose.orientation
        siny_cosp = 2.0 * (ori.w * ori.z + ori.x * ori.y)
        cosy_cosp = 1.0 - 2.0 * (ori.y * ori.y + ori.z * ori.z)
        yaw = math.atan2(siny_cosp, cosy_cosp)

        # Transform local odometry to world frame using robot initial spawn pose
        dx = float(pos.x)
        dy = float(pos.y)
        world_x = self.start_pose.x + dx * math.cos(self.start_pose.theta) - dy * math.sin(self.start_pose.theta)
        world_y = self.start_pose.y + dx * math.sin(self.start_pose.theta) + dy * math.cos(self.start_pose.theta)
        world_theta = (self.start_pose.theta + yaw + math.pi) % (2.0 * math.pi) - math.pi
        self.pose = Pose2D(x=world_x, y=world_y, theta=world_theta)

    def _on_scan(self, msg: LaserScan) -> None:
        self.latest_scan_ranges = list(msg.ranges)

    def _on_task_assignment(self, msg: String) -> None:
        """Parses task assignment: 'TASK:<task_id>:<target_x>:<target_y>' or 'HALT'."""
        text = msg.data.strip()
        if text.startswith("HALT"):
            self.is_active = False
            self.target_waypoint = None
            self._publish_cmd(0.0, 0.0)
            return

        parts = text.split(":")
        if len(parts) >= 4 and parts[0] == "TASK":
            self.current_task_id = parts[1]
            tx = float(parts[2])
            ty = float(parts[3])
            self.target_waypoint = (tx, ty)
            self.is_active = True
            self.get_logger().info(
                f"[{self.robot_id}] Assigned Task '{self.current_task_id}' -> ({tx:.1f}, {ty:.1f})"
            )

    def _control_step(self) -> None:
        if not self.is_active or self.target_waypoint is None:
            self._publish_cmd(0.0, 0.0)
            return

        tx, ty = self.target_waypoint
        dx = tx - self.pose.x
        dy = ty - self.pose.y
        dist = math.hypot(dx, dy)

        if dist <= self.goal_tolerance:
            # Reached goal
            self._publish_cmd(0.0, 0.0)
            status_msg = String()
            status_msg.data = f"COMPLETED:{self.current_task_id}"
            self.status_pub.publish(status_msg)
            self.get_logger().info(
                f"[{self.robot_id}] Reached waypoint for task '{self.current_task_id}'!"
            )
            self.target_waypoint = None
            return

        desired_heading = math.atan2(dy, dx)
        effective_heading = desired_heading

        # Reactive obstacle avoidance sweep using environment map & LiDAR
        lookahead = 1.6
        probe_x = self.pose.x + lookahead * math.cos(desired_heading)
        probe_y = self.pose.y + lookahead * math.sin(desired_heading)

        if not self.env.is_free(probe_x, probe_y):
            candidate_offsets = [
                math.pi / 4.0, -math.pi / 4.0,
                math.pi / 2.0, -math.pi / 2.0,
                3.0 * math.pi / 4.0, -3.0 * math.pi / 4.0,
            ]
            for offset in candidate_offsets:
                cand_h = desired_heading + offset
                cx = self.pose.x + lookahead * math.cos(cand_h)
                cy = self.pose.y + lookahead * math.sin(cand_h)
                if self.env.is_free(cx, cy):
                    if self.env.is_free(
                        self.pose.x + 0.8 * math.cos(cand_h),
                        self.pose.y + 0.8 * math.sin(cand_h),
                    ):
                        effective_heading = cand_h
                        break

        # Check immediate LiDAR forward obstacle proximity (< 0.6m)
        if self.latest_scan_ranges:
            num_samples = len(self.latest_scan_ranges)
            # Check rays in +/- 30 deg sector
            sector = num_samples // 12
            front_min = min(
                [self.latest_scan_ranges[i] for i in range(-sector, sector + 1)
                 if not math.isinf(self.latest_scan_ranges[i]) and not math.isnan(self.latest_scan_ranges[i])]
                or [10.0]
            )
            if front_min < 0.6:
                # Urgent obstacle avoidance turning
                self._publish_cmd(0.05, 1.2)
                return

        heading_err = (effective_heading - self.pose.theta + math.pi) % (2.0 * math.pi) - math.pi

        if abs(heading_err) > 0.6:
            cmd_lin = 0.1
            cmd_ang = max(min(self.kp_angular * heading_err, self.max_angular_speed), -self.max_angular_speed)
        else:
            cmd_lin = min(self.kp_linear * dist, self.max_linear_speed)
            cmd_ang = max(min(self.kp_angular * heading_err, self.max_angular_speed), -self.max_angular_speed)

        self._publish_cmd(cmd_lin, cmd_ang)

    def _publish_cmd(self, lin: float, ang: float) -> None:
        cmd = Twist()
        cmd.linear.x = float(lin)
        cmd.angular.z = float(ang)
        self.cmd_pub.publish(cmd)
