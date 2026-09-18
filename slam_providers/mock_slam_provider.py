"""
Mock SLAM Provider Simulating Incremental Mapping and Frontier Extraction.
Fulfills ISLAMProvider contract to decouple Swarm from the SLAM subsystem.
"""

from typing import Dict, List, Set, Tuple, Optional
from collections import deque
import math
from interfaces.slam_interface import (
    Pose2D,
    OccupancyGrid2D,
    FrontierCluster,
    SLAMState,
)
from simulation.environment import UrbanEnvironment
from .base_slam_provider import BaseSLAMProvider


class MockSLAMProvider(BaseSLAMProvider):
    """
    Simulates a 2D SLAM system (e.g. Cartographer / SLAM Toolbox) by incrementally
    revealing the ground-truth UrbanEnvironment as robots navigate, updating occupancy
    grids and extracting active frontiers.
    """

    def __init__(
        self,
        environment: UrbanEnvironment,
        min_cluster_size: int = 3,
        cluster_merge_distance: float = 3.0,
    ) -> None:
        super().__init__()
        self.environment = environment
        self.min_cluster_size = min_cluster_size
        self.cluster_merge_distance = cluster_merge_distance

        # Per-robot local occupancy grids
        self._robot_grids: Dict[str, OccupancyGrid2D] = {}

    def _get_or_create_grid(self, robot_id: str) -> OccupancyGrid2D:
        """Initializes an unknown occupancy grid for a newly registered robot."""
        if robot_id not in self._robot_grids:
            total_cells = self.environment.grid_width * self.environment.grid_height
            grid = OccupancyGrid2D(
                width=self.environment.grid_width,
                height=self.environment.grid_height,
                resolution=self.environment.resolution,
                origin=Pose2D(0.0, 0.0, 0.0),
                data=[-1] * total_cells,  # All unknown (-1) initially
            )
            self._robot_grids[robot_id] = grid
        return self._robot_grids[robot_id]

    def update_robot_pose(
        self,
        robot_id: str,
        pose: Pose2D,
        timestamp: float,
        linear_velocity: float = 0.0,
        angular_velocity: float = 0.0,
        sensor_range: float = 15.0,
    ) -> SLAMState:
        """
        Simulates one SLAM cycle for the robot:
        1. Raycasts from the robot's current pose.
        2. Updates local occupancy grid (unknown -> free / occupied).
        3. Extracts and clusters frontier regions.
        4. Calculates exploration ratio.
        5. Updates cached SLAMState.
        """
        grid = self._get_or_create_grid(robot_id)

        # 1 & 2: Sensor raycasting update
        observed = self.environment.raycast_2d(
            origin_x=pose.x,
            origin_y=pose.y,
            max_range=sensor_range,
        )
        for gx, gy, val in observed:
            idx = gy * grid.width + gx
            grid.data[idx] = val

        # 3: Extract frontiers
        frontiers = self._extract_frontiers(grid, robot_id=robot_id)

        # 4: Exploration metric
        known_cells = sum(1 for c in grid.data if c != -1)
        exploration_ratio = known_cells / float(grid.total_cells)

        # 5: Construct state snapshot
        state = SLAMState(
            robot_id=robot_id,
            timestamp=timestamp,
            pose=pose,
            linear_velocity=linear_velocity,
            angular_velocity=angular_velocity,
            occupancy_grid=grid,
            frontiers=frontiers,
            exploration_ratio=exploration_ratio,
        )

        self.update_cached_state(robot_id, state)
        return state

    def _extract_frontiers(self, grid: OccupancyGrid2D, robot_id: str = "robot") -> List[FrontierCluster]:
        """
        Identifies frontier cells (navigable cells adjacent to unknown territory)
        and clusters them into coherent FrontierCluster objects via Breadth-First Search (BFS).
        """
        width = grid.width
        height = grid.height
        frontier_cells: Set[Tuple[int, int]] = set()

        # 4-connectivity neighbor offsets
        neighbors = [(1, 0), (-1, 0), (0, 1), (0, -1)]

        for gy in range(1, height - 1):
            row_offset = gy * width
            for gx in range(1, width - 1):
                if grid.data[row_offset + gx] == 0:  # Free cell
                    # Check if adjacent to at least one unknown (-1) cell
                    is_frontier = False
                    for dx, dy in neighbors:
                        if grid.data[(gy + dy) * width + (gx + dx)] == -1:
                            is_frontier = True
                            break
                    if is_frontier:
                        frontier_cells.add((gx, gy))

        # BFS connected-components clustering
        visited: Set[Tuple[int, int]] = set()
        clusters: List[List[Tuple[int, int]]] = []

        for cell in frontier_cells:
            if cell in visited:
                continue

            cluster: List[Tuple[int, int]] = []
            queue = deque([cell])
            visited.add(cell)

            while queue:
                curr_gx, curr_gy = queue.popleft()
                cluster.append((curr_gx, curr_gy))

                # 8-connectivity search for contiguous frontier segments
                for dx in (-1, 0, 1):
                    for dy in (-1, 0, 1):
                        if dx == 0 and dy == 0:
                            continue
                        neighbor = (curr_gx + dx, curr_gy + dy)
                        if neighbor in frontier_cells and neighbor not in visited:
                            visited.add(neighbor)
                            queue.append(neighbor)

            if len(cluster) >= self.min_cluster_size:
                clusters.append(cluster)

        # Convert clustered grid coordinates to world-space FrontierCluster objects
        frontier_clusters: List[FrontierCluster] = []
        for idx, cluster in enumerate(clusters):
            sum_x = 0.0
            sum_y = 0.0
            min_x = float("inf")
            min_y = float("inf")
            max_x = float("-inf")
            max_y = float("-inf")

            for gx, gy in cluster:
                wx, wy = self.environment.grid_to_world(gx, gy)
                sum_x += wx
                sum_y += wy
                min_x = min(min_x, wx)
                min_y = min(min_y, wy)
                max_x = max(max_x, wx)
                max_y = max(max_y, wy)

            cluster_size = len(cluster)
            centroid_x = sum_x / cluster_size
            centroid_y = sum_y / cluster_size

            frontier_clusters.append(FrontierCluster(
                cluster_id=f"frontier_{robot_id}_{idx:03d}",
                centroid_x=centroid_x,
                centroid_y=centroid_y,
                size=cluster_size,
                bounding_box=(min_x, min_y, max_x, max_y),
            ))

        return frontier_clusters
