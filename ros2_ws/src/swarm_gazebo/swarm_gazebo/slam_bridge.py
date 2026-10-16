"""
ROS 2 SLAM Bridge Node for SwarmX.
Subscribes to Gazebo LiDAR (/robot_*/scan) and Odometry (/robot_*/odom),
maintains 2D occupancy grids per robot, extracts frontier clusters,
and fulfills the ISLAMProvider contract for the SwarmX coordination subsystem.
"""

import math
import time
from collections import deque
from typing import Dict, List, Set, Tuple, Optional

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry, OccupancyGrid
from std_msgs.msg import Header

from interfaces.slam_interface import (
    ISLAMProvider,
    SLAMState,
    Pose2D,
    OccupancyGrid2D,
    FrontierCluster,
)
from simulation.environment import UrbanEnvironment
from simulation.obstacle_generator import ObstacleGenerator


DEFAULT_SPAWN_POSES: Dict[str, Pose2D] = {
    "robot_1": Pose2D(6.0, 6.0, 0.0),
    "robot_2": Pose2D(6.0, 44.0, -math.pi / 2.0),
    "robot_3": Pose2D(44.0, 6.0, math.pi / 2.0),
    "robot_4": Pose2D(44.0, 44.0, math.pi),
}


class GazeboSLAMBridge(ISLAMProvider, Node):
    """
    Bridge that subscribes to Gazebo sensor topics for multiple robots,
    integrates LiDAR scan hits/misses into local occupancy grids,
    detects exploration frontiers, and serves as an ISLAMProvider.
    """

    def __init__(
        self,
        robot_ids: List[str],
        map_width_m: float = 50.0,
        map_height_m: float = 50.0,
        resolution: float = 1.0,
        spawn_poses: Optional[Dict[str, Pose2D]] = None,
    ) -> None:
        Node.__init__(self, "gazebo_slam_bridge")
        self.robot_ids = robot_ids
        self.map_width_m = map_width_m
        self.map_height_m = map_height_m
        self.resolution = resolution
        self.grid_width = int(math.ceil(map_width_m / resolution))
        self.grid_height = int(math.ceil(map_height_m / resolution))
        self.spawn_poses = spawn_poses or DEFAULT_SPAWN_POSES

        # Ground truth environment representation for simulation raycast updates
        self.env = UrbanEnvironment(width=map_width_m, height=map_height_m, resolution=resolution)
        ObstacleGenerator.add_perimeter_walls(self.env, thickness_cells=1)
        w, h = map_width_m, map_height_m
        ObstacleGenerator.add_box_obstacle(self.env, w * 0.2, h * 0.2, w * 0.45, h * 0.45)
        ObstacleGenerator.add_box_obstacle(self.env, w * 0.55, h * 0.2, w * 0.8, h * 0.45)
        ObstacleGenerator.add_box_obstacle(self.env, w * 0.2, h * 0.55, w * 0.45, h * 0.8)
        ObstacleGenerator.add_box_obstacle(self.env, w * 0.55, h * 0.55, w * 0.8, h * 0.8)

        # Internal state stores
        self._states: Dict[str, SLAMState] = {}
        self._grids: Dict[str, OccupancyGrid2D] = {}
        self._latest_poses: Dict[str, Pose2D] = {}
        self._latest_vels: Dict[str, Tuple[float, float]] = {}
        self._last_update_times: Dict[str, float] = {}

        # ROS 2 Subscriptions & Publishers
        self._scan_subs = {}
        self._odom_subs = {}
        self._map_pubs = {}

        qos_sensor = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )

        for rid in robot_ids:
            total_cells = self.grid_width * self.grid_height
            self._grids[rid] = OccupancyGrid2D(
                width=self.grid_width,
                height=self.grid_height,
                resolution=self.resolution,
                origin=Pose2D(0.0, 0.0, 0.0),
                data=[-1] * total_cells,
            )
            sp = self.spawn_poses.get(rid, Pose2D(0.0, 0.0, 0.0))
            self._latest_poses[rid] = Pose2D(sp.x, sp.y, sp.theta)
            self._latest_vels[rid] = (0.0, 0.0)
            self._last_update_times[rid] = time.time()

            # Subscribe to odometry
            odom_topic = f"/{rid}/odom"
            self._odom_subs[rid] = self.create_subscription(
                Odometry,
                odom_topic,
                lambda msg, r=rid: self._on_odom(r, msg),
                10,
            )

            # Subscribe to laser scan
            scan_topic = f"/{rid}/scan"
            self._scan_subs[rid] = self.create_subscription(
                LaserScan,
                scan_topic,
                lambda msg, r=rid: self._on_scan(r, msg),
                qos_sensor,
            )

            # Publish ROS 2 occupancy grid
            map_topic = f"/{rid}/map"
            self._map_pubs[rid] = self.create_publisher(OccupancyGrid, map_topic, 10)

        self.get_logger().info(
            f"GazeboSLAMBridge initialized for {len(robot_ids)} robots: {robot_ids}"
        )

    def _on_odom(self, robot_id: str, msg: Odometry) -> None:
        """Extracts 2D pose and linear/angular velocity from Gazebo Odometry and maps to world frame."""
        if msg is None or not hasattr(msg, "pose"):
            return
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

        self._latest_poses[robot_id] = Pose2D(x=world_x, y=world_y, theta=world_theta)
        self._latest_vels[robot_id] = (
            float(msg.twist.twist.linear.x),
            float(msg.twist.twist.angular.z),
        )
        self._last_update_times[robot_id] = time.time()

    def update_robot_pose(
        self,
        robot_id: str,
        pose: Pose2D,
        timestamp: float,
        linear_velocity: float = 0.0,
        angular_velocity: float = 0.0,
        sensor_range: float = 14.0,
    ) -> SLAMState:
        """Updates robot state, performs raycast mapping, and extracts frontiers."""
        self._latest_poses[robot_id] = pose
        self._latest_vels[robot_id] = (linear_velocity, angular_velocity)
        self._last_update_times[robot_id] = timestamp
        grid = self._grids[robot_id]

        # Sensor raycasting update
        observed = self.env.raycast_2d(
            origin_x=pose.x,
            origin_y=pose.y,
            max_range=sensor_range,
        )
        for gx, gy, val in observed:
            idx = gy * grid.width + gx
            if 0 <= idx < len(grid.data):
                grid.data[idx] = val

        # Extract frontiers
        frontiers = self._extract_frontiers(grid, robot_id)

        # Exploration metric
        known_cells = sum(1 for c in grid.data if c != -1)
        exp_ratio = known_cells / float(grid.total_cells)

        state = SLAMState(
            robot_id=robot_id,
            timestamp=timestamp,
            pose=pose,
            linear_velocity=linear_velocity,
            angular_velocity=angular_velocity,
            occupancy_grid=grid,
            frontiers=frontiers,
            exploration_ratio=exp_ratio,
        )
        self._states[robot_id] = state
        return state

    def _on_scan(self, robot_id: str, msg: LaserScan) -> None:
        """Processes 2D LiDAR rays from Gazebo to update the local occupancy grid."""
        pose = self._latest_poses.get(robot_id, self.spawn_poses.get(robot_id, Pose2D(0.0, 0.0, 0.0)))
        grid = self._grids[robot_id]

        angle = msg.angle_min + pose.theta
        angle_inc = msg.angle_increment
        max_r = min(msg.range_max, 14.0)

        for i in range(0, len(msg.ranges), 4):
            r = msg.ranges[i]
            cur_angle = angle + i * angle_inc
            if math.isinf(r) or math.isnan(r) or r > max_r:
                ray_dist = max_r
                hit = False
            elif r < msg.range_min:
                continue
            else:
                ray_dist = r
                hit = True

            steps = max(1, int(ray_dist / (self.resolution * 0.5)))
            for s in range(1, steps):
                dist = s * (self.resolution * 0.5)
                if dist >= ray_dist:
                    break
                px = pose.x + dist * math.cos(cur_angle)
                py = pose.y + dist * math.sin(cur_angle)
                gx = int(px / self.resolution)
                gy = int(py / self.resolution)
                if 0 <= gx < self.grid_width and 0 <= gy < self.grid_height:
                    idx = gy * self.grid_width + gx
                    if grid.data[idx] != 100:
                        grid.data[idx] = 0

            if hit:
                hx = pose.x + ray_dist * math.cos(cur_angle)
                hy = pose.y + ray_dist * math.sin(cur_angle)
                gx = int(hx / self.resolution)
                gy = int(hy / self.resolution)
                if 0 <= gx < self.grid_width and 0 <= gy < self.grid_height:
                    grid.data[gy * self.grid_width + gx] = 100

        frontiers = self._extract_frontiers(grid, robot_id)
        known_cells = sum(1 for c in grid.data if c != -1)
        exp_ratio = min(1.0, known_cells / float(grid.total_cells))
        v_lin, v_ang = self._latest_vels.get(robot_id, (0.0, 0.0))

        self._states[robot_id] = SLAMState(
            robot_id=robot_id,
            timestamp=time.time(),
            pose=pose,
            linear_velocity=v_lin,
            angular_velocity=v_ang,
            occupancy_grid=grid,
            frontiers=frontiers,
            exploration_ratio=exp_ratio,
        )

    def _extract_frontiers(self, grid: OccupancyGrid2D, robot_id: str) -> List[FrontierCluster]:
        """Extracts and clusters boundary cells between free (0) and unknown (-1) regions."""
        width = grid.width
        height = grid.height
        frontier_cells: Set[Tuple[int, int]] = set()
        neighbors = [(1, 0), (-1, 0), (0, 1), (0, -1)]

        for gy in range(1, height - 1):
            row_offset = gy * width
            for gx in range(1, width - 1):
                if grid.data[row_offset + gx] == 0:
                    for dx, dy in neighbors:
                        if grid.data[(gy + dy) * width + (gx + dx)] == -1:
                            frontier_cells.add((gx, gy))
                            break

        visited: Set[Tuple[int, int]] = set()
        clusters: List[List[Tuple[int, int]]] = []

        for cell in frontier_cells:
            if cell in visited:
                continue
            cluster: List[Tuple[int, int]] = []
            queue = deque([cell])
            visited.add(cell)

            while queue:
                cgx, cgy = queue.popleft()
                cluster.append((cgx, cgy))
                for dx in (-1, 0, 1):
                    for dy in (-1, 0, 1):
                        if dx == 0 and dy == 0:
                            continue
                        neighbor = (cgx + dx, cgy + dy)
                        if neighbor in frontier_cells and neighbor not in visited:
                            visited.add(neighbor)
                            queue.append(neighbor)

            if len(cluster) >= 3:
                clusters.append(cluster)

        frontier_clusters: List[FrontierCluster] = []
        for idx, cluster in enumerate(clusters):
            sum_x = sum((gx + 0.5) * self.resolution for gx, gy in cluster)
            sum_y = sum((gy + 0.5) * self.resolution for gx, gy in cluster)
            csize = len(cluster)
            min_x = min(gx * self.resolution for gx, gy in cluster)
            max_x = max((gx + 1) * self.resolution for gx, gy in cluster)
            min_y = min(gy * self.resolution for gx, gy in cluster)
            max_y = max((gy + 1) * self.resolution for gx, gy in cluster)

            frontier_clusters.append(
                FrontierCluster(
                    cluster_id=f"frontier_{robot_id}_{idx:03d}",
                    centroid_x=sum_x / csize,
                    centroid_y=sum_y / csize,
                    size=csize,
                    bounding_box=(min_x, min_y, max_x, max_y),
                )
            )
        return frontier_clusters

    # ISLAMProvider Contract Methods
    def get_latest_state(self, robot_id: str) -> SLAMState:
        if robot_id in self._states:
            return self._states[robot_id]

        pose = self._latest_poses.get(robot_id, self.spawn_poses.get(robot_id, Pose2D(0.0, 0.0, 0.0)))
        grid = self._grids.get(
            robot_id,
            OccupancyGrid2D(
                width=self.grid_width,
                height=self.grid_height,
                resolution=self.resolution,
                origin=Pose2D(0.0, 0.0, 0.0),
                data=[-1] * (self.grid_width * self.grid_height),
            ),
        )
        return SLAMState(
            robot_id=robot_id,
            timestamp=time.time(),
            pose=pose,
            linear_velocity=0.0,
            angular_velocity=0.0,
            occupancy_grid=grid,
            frontiers=[],
            exploration_ratio=0.0,
        )

    def is_healthy(self, robot_id: str) -> bool:
        last_t = self._last_update_times.get(robot_id, 0.0)
        return (time.time() - last_t) < 5.0
