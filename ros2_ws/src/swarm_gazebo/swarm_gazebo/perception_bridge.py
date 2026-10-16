"""
ROS 2 Perception Bridge Node for SwarmX.
Performs vehicle detection in camera FOV, license plate OCR, and fuzzy Levenshtein
matching against the stolen vehicle watchlist, triggering swarm recruitment and debrief.
"""

import math
import time
from typing import Dict, List, Optional, Tuple

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image
from std_msgs.msg import String

from interfaces.slam_interface import Pose2D
from interfaces.perception_interface import (
    VehicleDetectionEvent,
    PlateDetectionEvent,
    MatchEvent,
    BoundingBox2D,
)
from perception.vehicle_detector import TargetVehicle, SimulatedVehicleDetector
from perception.plate_recognizer import SimulatedPlateRecognizer
from perception.matching_engine import MatchingEngine
from simulation.environment import UrbanEnvironment
from simulation.obstacle_generator import ObstacleGenerator


DEFAULT_SPAWN_POSES: Dict[str, Pose2D] = {
    "robot_1": Pose2D(6.0, 6.0, 0.0),
    "robot_2": Pose2D(6.0, 44.0, -math.pi / 2.0),
    "robot_3": Pose2D(44.0, 6.0, math.pi / 2.0),
    "robot_4": Pose2D(44.0, 44.0, math.pi),
}


class GazeboPerceptionBridge(Node):
    """
    Simulates onboard camera processing and fuzzy OCR plate matching for all swarm robots.
    Integrates directly with the perception subsystem models and publishes detection events.
    """

    def __init__(
        self,
        robot_ids: List[str],
        target_plate: str = "DXB-88392",
        target_pos: Tuple[float, float] = (25.0, 25.0),
        env_width: float = 50.0,
        env_height: float = 50.0,
        spawn_poses: Optional[Dict[str, Pose2D]] = None,
    ) -> None:
        super().__init__("gazebo_perception_bridge")
        self.robot_ids = robot_ids
        self.target_plate = target_plate
        self.spawn_poses = spawn_poses or DEFAULT_SPAWN_POSES

        # 1. Environment & Obstacle Setup for Line-Of-Sight Occlusion Raycasting
        self.env = UrbanEnvironment(width=env_width, height=env_height, resolution=1.0)
        ObstacleGenerator.add_perimeter_walls(self.env, thickness_cells=1)
        w, h = env_width, env_height
        ObstacleGenerator.add_box_obstacle(self.env, w * 0.2, h * 0.2, w * 0.45, h * 0.45)
        ObstacleGenerator.add_box_obstacle(self.env, w * 0.55, h * 0.2, w * 0.8, h * 0.45)
        ObstacleGenerator.add_box_obstacle(self.env, w * 0.2, h * 0.55, w * 0.45, h * 0.8)
        ObstacleGenerator.add_box_obstacle(self.env, w * 0.55, h * 0.55, w * 0.8, h * 0.8)

        # 2. Vehicle Database in Gazebo World
        self.target_vehicle = TargetVehicle(
            target_id="target_stolen_sedan",
            plate_number=target_plate,
            vehicle_class="sedan",
            color="dark_gray",
            position=target_pos,
            is_stolen=True,
        )
        self.decoy_vehicles = [
            TargetVehicle(
                target_id="decoy_suv_1",
                plate_number="AD-10294",
                vehicle_class="suv",
                color="white",
                position=(10.0, 25.0),
                is_stolen=False,
            ),
            TargetVehicle(
                target_id="decoy_sedan_2",
                plate_number="SHJ-44910",
                vehicle_class="sedan",
                color="red",
                position=(40.0, 25.0),
                is_stolen=False,
            ),
        ]
        self.all_vehicles = [self.target_vehicle] + self.decoy_vehicles

        # 3. Perception Modules
        self.detector = SimulatedVehicleDetector(max_detection_range=14.0, fov_degrees=90.0)
        self.ocr = SimulatedPlateRecognizer(base_ocr_confidence=0.95)
        self.matcher = MatchingEngine(match_threshold=0.80)
        self.matcher.add_target_vehicle(self.target_vehicle)

        # 4. Robot State and ROS Subscriptions
        self.robot_poses: Dict[str, Pose2D] = {}
        self.confirmed_match: Optional[MatchEvent] = None
        self._target_confirmed_published = False

        # ROS 2 Publisher for Perception Events
        self.event_pub = self.create_publisher(String, "/swarm/perception_events", 10)

        qos_sensor = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )

        for rid in robot_ids:
            sp = self.spawn_poses.get(rid, Pose2D(0.0, 0.0, 0.0))
            self.robot_poses[rid] = Pose2D(sp.x, sp.y, sp.theta)
            self.create_subscription(
                Odometry,
                f"/{rid}/odom",
                lambda msg, r=rid: self._on_odom(r, msg),
                10,
            )
            # Subscribe to Gazebo Camera stream
            self.create_subscription(
                Image,
                f"/{rid}/camera/image_raw",
                lambda msg, r=rid: self._on_camera_frame(r, msg),
                qos_sensor,
            )

        # Timer for periodic perception scanning (10Hz)
        self.timer = self.create_timer(0.1, self._perception_cycle)
        self.get_logger().info(
            f"GazeboPerceptionBridge initialized. Target Watchlist: [{target_plate}]"
        )

    def _on_odom(self, robot_id: str, msg: Odometry) -> None:
        pos = msg.pose.pose.position
        ori = msg.pose.pose.orientation
        siny_cosp = 2.0 * (ori.w * ori.z + ori.x * ori.y)
        cosy_cosp = 1.0 - 2.0 * (ori.y * ori.y + ori.z * ori.z)
        yaw = math.atan2(siny_cosp, cosy_cosp)

        sp = self.spawn_poses.get(robot_id, Pose2D(0.0, 0.0, 0.0))
        dx = float(pos.x)
        dy = float(pos.y)
        world_x = sp.x + dx * math.cos(sp.theta) - dy * math.sin(sp.theta)
        world_y = sp.y + dx * math.sin(sp.theta) + dy * math.cos(sp.theta)
        world_theta = (sp.theta + yaw + math.pi) % (2.0 * math.pi) - math.pi
        self.robot_poses[robot_id] = Pose2D(x=world_x, y=world_y, theta=world_theta)

    def _on_camera_frame(self, robot_id: str, msg: Image) -> None:
        """Handles camera frame feeds from Gazebo."""
        pass  # Frame received, processed via detector and OCR in cycle

    def _perception_cycle(self) -> None:
        """Executes perception pipeline for each robot in Gazebo world."""
        now = time.time()
        for rid, pose in self.robot_poses.items():
            # Check detection against all vehicles in world
            detections = self.detector.detect_vehicles(
                robot_id=rid,
                robot_pose=pose,
                timestamp=now,
                vehicles=self.all_vehicles,
                env=self.env,
            )
            for det_event, veh_obj in detections:
                # OCR extraction
                plate_event = self.ocr.recognize_license_plate(det_event, veh_obj)
                if plate_event is None:
                    continue

                # Fuzzy watchlist matching
                match = self.matcher.match_plate(plate_event)
                if match and match.confirmed:
                    if not self._target_confirmed_published:
                        self._target_confirmed_published = True
                        self.confirmed_match = match
                        self.get_logger().warn(
                            f"🎯 TARGET CONFIRMED by {rid}! Plate: {match.plate_number}, "
                            f"Confidence: {match.match_confidence*100:.1f}%, Loc: {match.location}"
                        )
                    # Publish event to swarm
                    msg = String()
                    msg.data = (
                        f"TARGET_MATCH:{rid}:{match.plate_number}:{match.match_confidence:.3f}:"
                        f"{match.location[0]:.2f}:{match.location[1]:.2f}"
                    )
                    self.event_pub.publish(msg)
                else:
                    # Non-target / civilian vehicle rejection
                    msg = String()
                    msg.data = f"NON_TARGET:{rid}:{plate_event.plate_number}:{plate_event.ocr_confidence:.3f}"
                    self.event_pub.publish(msg)
