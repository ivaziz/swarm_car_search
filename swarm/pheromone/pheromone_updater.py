"""
Pheromone Updater with Kernel Diffusion, Evaporation Loops, and Stigmergic Sync.
"""

from typing import Optional
import math
from .evaporation_model import PheromoneLayerType, EvaporationModel
from .pheromone_map import PheromoneMap


class PheromoneUpdater:
    """
    Manages deposition, spatial Gaussian diffusion, and discrete evaporation steps
    across all pheromone layers in the PheromoneMap.
    """

    def __init__(
        self,
        pheromone_map: PheromoneMap,
        evaporation_model: Optional[EvaporationModel] = None,
    ) -> None:
        self.map = pheromone_map
        self.evaporation_model = evaporation_model or EvaporationModel(self.map.config)

    def deposit(
        self,
        layer: PheromoneLayerType,
        x: float,
        y: float,
        amount: float,
        radius: float = 0.0,
    ) -> None:
        """
        Deposits pheromone amount around continuous coordinates (x, y).
        If radius > 0, applies radial Gaussian kernel diffusion to neighbor cells.
        """
        center_g = self.map.world_to_grid(x, y)
        if center_g is None:
            return

        cx, cy = center_g

        if radius <= 0.0 or radius < self.map.resolution:
            # Single-cell discrete deposit
            curr = self.map.get_value_grid(layer, cx, cy)
            self.map.set_value_grid(layer, cx, cy, curr + amount)
            return

        # Multi-cell radial kernel
        cell_radius = int(math.ceil(radius / self.map.resolution))
        sigma = radius / 2.0
        two_sigma_sq = 2.0 * sigma * sigma

        for dy in range(-cell_radius, cell_radius + 1):
            gy = cy + dy
            if not (0 <= gy < self.map.grid_height):
                continue

            for dx in range(-cell_radius, cell_radius + 1):
                gx = cx + dx
                if not (0 <= gx < self.map.grid_width):
                    continue

                dist = math.hypot(dx * self.map.resolution, dy * self.map.resolution)
                if dist <= radius:
                    weight = math.exp(-(dist * dist) / two_sigma_sq)
                    curr = self.map.get_value_grid(layer, gx, gy)
                    self.map.set_value_grid(layer, gx, gy, curr + amount * weight)

    def deposit_exploration(self, x: float, y: float, amount: float = 1.0, radius: float = 1.0) -> None:
        """Deposits positive exploration pheromone along surveyed paths."""
        self.deposit(PheromoneLayerType.EXPLORATION, x, y, amount, radius)

    def deposit_success(self, x: float, y: float, amount: float = 10.0, radius: float = 3.0) -> None:
        """Deposits high-priority recruitment pheromone at candidate or confirmed target vehicles."""
        self.deposit(PheromoneLayerType.SUCCESS, x, y, amount, radius)

    def deposit_avoidance(self, x: float, y: float, amount: float = 5.0, radius: float = 2.0) -> None:
        """Deposits temporary avoidance / reservation pheromone around active robots."""
        self.deposit(PheromoneLayerType.AVOIDANCE, x, y, amount, radius)

    def step_evaporation(self, dt: float = 1.0) -> None:
        """
        Executes one discrete evaporation step across all layers:
            tau(t + dt) = clamp(tau(t) * (1 - rho)^dt)
        """
        for layer in PheromoneLayerType:
            decay_factor = self.evaporation_model.get_decay_factor(layer, dt=dt)
            grid_data = self.map.layers[layer]
            min_val = self.map.config.min_pheromone
            max_val = self.map.config.max_pheromone

            for idx in range(len(grid_data)):
                val = grid_data[idx] * decay_factor
                if val < min_val:
                    grid_data[idx] = min_val
                elif val > max_val:
                    grid_data[idx] = max_val
                else:
                    grid_data[idx] = val

    def sync_from_peer_update(
        self,
        layer: PheromoneLayerType,
        x: float,
        y: float,
        amount: float,
        radius: float = 0.0,
    ) -> None:
        """Applies stigmergic update received from a peer robot."""
        self.deposit(layer, x, y, amount, radius)
