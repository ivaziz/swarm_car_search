"""
Live visualization dashboards, pheromone heatmaps, SVG export, and metric plotting.
"""

from .swarm_visualizer import (
    SwarmVisualizer,
    STATE_COLOR_MAP,
    _HAS_MATPLOTLIB,
    _HAS_PIL,
)

__all__ = [
    "SwarmVisualizer",
    "STATE_COLOR_MAP",
    "_HAS_MATPLOTLIB",
    "_HAS_PIL",
]
