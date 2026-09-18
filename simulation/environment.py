"""
2D Urban Grid Environment Simulation for Swarm Exploration.
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict
import math
from interfaces.slam_interface import Pose2D


@dataclass
class UrbanZone:
    """Designated spatial zone within the urban environment."""
    name: str
    bounds: Tuple[float, float, float, float]  # (min_x, min_y, max_x, max_y)
    vehicle_prior_probability: float = 0.5

    def contains(self, x: float, y: float) -> bool:
        """Check if coordinates fall inside the zone."""
        min_x, min_y, max_x, max_y = self.bounds
        return min_x <= x <= max_x and min_y <= y <= max_y


class UrbanEnvironment:
    """
    2D Discrete Urban Grid Environment representing ground-truth physics and obstacles.
    Cell values:
        0: Free navigable space
      100: Static obstacle (building wall, barrier)
    """

    def __init__(
        self,
        width: float = 100.0,
        height: float = 100.0,
        resolution: float = 0.5,
    ) -> None:
        self.width = width
        self.height = height
        self.resolution = resolution

        self.grid_width = int(math.ceil(width / resolution))
        self.grid_height = int(math.ceil(height / resolution))

        # Ground truth grid initialized to all free space (0)
        self.grid: List[int] = [0] * (self.grid_width * self.grid_height)
        self.zones: List[UrbanZone] = []
        self.targets: Dict[str, Tuple[float, float]] = {}

    def world_to_grid(self, x: float, y: float) -> Optional[Tuple[int, int]]:
        """Converts continuous world coordinates (meters) to discrete grid coordinates."""
        if not (0.0 <= x < self.width and 0.0 <= y < self.height):
            return None
        gx = int(x / self.resolution)
        gy = int(y / self.resolution)
        gx = min(max(gx, 0), self.grid_width - 1)
        gy = min(max(gy, 0), self.grid_height - 1)
        return (gx, gy)

    def grid_to_world(self, gx: int, gy: int) -> Tuple[float, float]:
        """Converts discrete grid coordinates to center of cell in world coordinates."""
        wx = (gx + 0.5) * self.resolution
        wy = (gy + 0.5) * self.resolution
        return (wx, wy)

    def _index(self, gx: int, gy: int) -> int:
        return gy * self.grid_width + gx

    def set_cell(self, gx: int, gy: int, value: int) -> None:
        """Sets the state of a specific grid cell."""
        if 0 <= gx < self.grid_width and 0 <= gy < self.grid_height:
            self.grid[self._index(gx, gy)] = value

    def get_cell(self, gx: int, gy: int) -> int:
        """Retrieves the state of a specific grid cell."""
        if 0 <= gx < self.grid_width and 0 <= gy < self.grid_height:
            return self.grid[self._index(gx, gy)]
        return 100  # Outside bounds is treated as obstacle

    def is_free(self, x: float, y: float) -> bool:
        """Checks whether the given world coordinate is navigable free space."""
        g = self.world_to_grid(x, y)
        if g is None:
            return False
        return self.get_cell(g[0], g[1]) == 0

    def add_zone(self, zone: UrbanZone) -> None:
        """Registers an urban functional zone."""
        self.zones.append(zone)

    def get_zone_prior(self, x: float, y: float) -> float:
        """Returns vehicle prior detection probability for given world coordinates."""
        for zone in self.zones:
            if zone.contains(x, y):
                return zone.vehicle_prior_probability
        return 0.3  # Default background prior

    def add_target_vehicle(self, target_id: str, x: float, y: float) -> None:
        """Places a target vehicle in ground-truth space."""
        self.targets[target_id] = (x, y)

    def raycast_2d(
        self,
        origin_x: float,
        origin_y: float,
        max_range: float = 15.0,
        num_rays: int = 72,
    ) -> List[Tuple[int, int, int]]:
        """
        Simulates 2D range-bearing sensor (e.g. planar LiDAR) sweeping 360 degrees.
        Returns a list of observed grid cells: (gx, gy, observed_value).
        """
        observed_cells: Dict[Tuple[int, int], int] = {}
        step_size = self.resolution * 0.5
        max_steps = int(max_range / step_size)

        for ray_idx in range(num_rays):
            angle = (2.0 * math.pi * ray_idx) / num_rays
            cos_a = math.cos(angle)
            sin_a = math.sin(angle)

            for step in range(1, max_steps + 1):
                curr_x = origin_x + step * step_size * cos_a
                curr_y = origin_y + step * step_size * sin_a

                g = self.world_to_grid(curr_x, curr_y)
                if g is None:
                    break

                gx, gy = g
                val = self.get_cell(gx, gy)
                if val == 100:
                    # Struck an obstacle cell
                    observed_cells[(gx, gy)] = 100
                    break
                else:
                    observed_cells[(gx, gy)] = 0

        return [(gx, gy, val) for (gx, gy), val in observed_cells.items()]
