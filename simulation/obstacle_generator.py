"""
Urban Obstacle Generator for City Blocks, Perimeter Walls, and Roadways.
"""

from typing import Optional
from .environment import UrbanEnvironment, UrbanZone


class ObstacleGenerator:
    """Utility class to synthesize realistic urban environments with obstacles."""

    @staticmethod
    def add_perimeter_walls(env: UrbanEnvironment, thickness_cells: int = 1) -> None:
        """Surrounds the environment with solid boundary walls."""
        for t in range(thickness_cells):
            for gx in range(env.grid_width):
                env.set_cell(gx, t, 100)
                env.set_cell(gx, env.grid_height - 1 - t, 100)
            for gy in range(env.grid_height):
                env.set_cell(t, gy, 100)
                env.set_cell(env.grid_width - 1 - t, gy, 100)

    @staticmethod
    def add_box_obstacle(
        env: UrbanEnvironment,
        min_x: float,
        min_y: float,
        max_x: float,
        max_y: float,
    ) -> None:
        """Adds a solid rectangular obstacle (e.g. building structure)."""
        g_min = env.world_to_grid(min_x, min_y)
        g_max = env.world_to_grid(max_x, max_y)
        if g_min is None or g_max is None:
            return

        min_gx, min_gy = min(g_min[0], g_max[0]), min(g_min[1], g_max[1])
        max_gx, max_gy = max(g_min[0], g_max[0]), max(g_min[1], g_max[1])

        for gx in range(min_gx, max_gx + 1):
            for gy in range(min_gy, max_gy + 1):
                env.set_cell(gx, gy, 100)

    @classmethod
    def generate_urban_city_blocks(cls, env: UrbanEnvironment) -> None:
        """
        Creates a realistic urban block layout:
        - Outer perimeter
        - 4 Major Building blocks
        - Cross-cutting avenues (roads)
        - Parking area with static parked vehicles
        """
        cls.add_perimeter_walls(env, thickness_cells=2)

        # Building Block 1 (South-West)
        cls.add_box_obstacle(env, 15.0, 15.0, 40.0, 40.0)

        # Building Block 2 (South-East)
        cls.add_box_obstacle(env, 60.0, 15.0, 85.0, 40.0)

        # Building Block 3 (North-West)
        cls.add_box_obstacle(env, 15.0, 60.0, 40.0, 85.0)

        # Building Block 4 (North-East)
        cls.add_box_obstacle(env, 60.0, 60.0, 85.0, 85.0)

        # Register functional urban zones
        env.add_zone(UrbanZone(
            name="central_intersection",
            bounds=(40.0, 40.0, 60.0, 60.0),
            vehicle_prior_probability=0.70,
        ))
        env.add_zone(UrbanZone(
            name="north_parking_lot",
            bounds=(10.0, 85.0, 50.0, 95.0),
            vehicle_prior_probability=0.85,
        ))
        env.add_zone(UrbanZone(
            name="east_residential",
            bounds=(85.0, 10.0, 95.0, 90.0),
            vehicle_prior_probability=0.40,
        ))
