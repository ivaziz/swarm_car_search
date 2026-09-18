"""
Multi-Layer Spatially Discretized 2D Pheromone Map.
"""

from typing import Dict, List, Optional, Tuple
import math
from .evaporation_model import PheromoneLayerType, EvaporationConfig


class PheromoneMap:
    """
    Maintains continuous 2D scalar fields across urban coordinates for each pheromone layer.
    """

    def __init__(
        self,
        width: float = 100.0,
        height: float = 100.0,
        resolution: float = 0.5,
        config: Optional[EvaporationConfig] = None,
    ) -> None:
        self.width = width
        self.height = height
        self.resolution = resolution
        self.config = config or EvaporationConfig()

        self.grid_width = int(math.ceil(width / resolution))
        self.grid_height = int(math.ceil(height / resolution))
        self.total_cells = self.grid_width * self.grid_height

        # Initialize layers with minimum base pheromone
        self.layers: Dict[PheromoneLayerType, List[float]] = {
            layer: [self.config.min_pheromone] * self.total_cells
            for layer in PheromoneLayerType
        }

    def world_to_grid(self, x: float, y: float) -> Optional[Tuple[int, int]]:
        """Maps world coordinates to discrete grid indices."""
        if not (0.0 <= x < self.width and 0.0 <= y < self.height):
            return None
        gx = min(max(int(x / self.resolution), 0), self.grid_width - 1)
        gy = min(max(int(y / self.resolution), 0), self.grid_height - 1)
        return (gx, gy)

    def grid_to_world(self, gx: int, gy: int) -> Tuple[float, float]:
        """Maps discrete grid indices to continuous world coordinates (cell center)."""
        wx = (gx + 0.5) * self.resolution
        wy = (gy + 0.5) * self.resolution
        return (wx, wy)

    def _index(self, gx: int, gy: int) -> int:
        return gy * self.grid_width + gx

    def get_value_grid(self, layer: PheromoneLayerType, gx: int, gy: int) -> float:
        """Retrieves pheromone level at discrete cell."""
        if 0 <= gx < self.grid_width and 0 <= gy < self.grid_height:
            return self.layers[layer][self._index(gx, gy)]
        return self.config.min_pheromone

    def set_value_grid(self, layer: PheromoneLayerType, gx: int, gy: int, value: float) -> None:
        """Sets clamped pheromone level at discrete cell."""
        if 0 <= gx < self.grid_width and 0 <= gy < self.grid_height:
            clamped = max(self.config.min_pheromone, min(value, self.config.max_pheromone))
            self.layers[layer][self._index(gx, gy)] = clamped

    def get_value(self, layer: PheromoneLayerType, x: float, y: float) -> float:
        """Retrieves pheromone level at world coordinates."""
        g = self.world_to_grid(x, y)
        if g is None:
            return self.config.min_pheromone
        return self.get_value_grid(layer, g[0], g[1])

    def get_all_layers_at(self, x: float, y: float) -> Dict[PheromoneLayerType, float]:
        """Returns readings of all pheromone layers at specified world coordinates."""
        return {layer: self.get_value(layer, x, y) for layer in PheromoneLayerType}

    def get_box_aggregate(
        self,
        layer: PheromoneLayerType,
        min_x: float,
        min_y: float,
        max_x: float,
        max_y: float,
        mode: str = "mean",
    ) -> float:
        """
        Computes aggregate pheromone intensity across a spatial bounding box.
        Supports 'mean' (default) and 'max'.
        """
        g_min = self.world_to_grid(min_x, min_y) or (0, 0)
        g_max = self.world_to_grid(max_x, max_y) or (self.grid_width - 1, self.grid_height - 1)

        start_x, end_x = min(g_min[0], g_max[0]), max(g_min[0], g_max[0])
        start_y, end_y = min(g_min[1], g_max[1]), max(g_min[1], g_max[1])

        values: List[float] = []
        for gy in range(start_y, end_y + 1):
            row_offset = gy * self.grid_width
            for gx in range(start_x, end_x + 1):
                values.append(self.layers[layer][row_offset + gx])

        if not values:
            return self.config.min_pheromone

        if mode == "max":
            return max(values)
        return sum(values) / len(values)

    def reset(self, initial_value: Optional[float] = None) -> None:
        """Resets all layers to minimum or specified baseline."""
        val = self.config.min_pheromone if initial_value is None else initial_value
        for layer in PheromoneLayerType:
            self.layers[layer] = [val] * self.total_cells
