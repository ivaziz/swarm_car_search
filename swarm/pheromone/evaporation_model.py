"""
Mathematical Evaporation Models and Bounds Clamping for Pheromone Layers.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict


class PheromoneLayerType(str, Enum):
    """Types of stigmergic pheromone fields maintained across the urban grid."""
    EXPLORATION = "EXPLORATION"  # Long-term memory of visited/mapped territory
    SUCCESS = "SUCCESS"          # Recruitment beacon for candidate/confirmed vehicles
    AVOIDANCE = "AVOIDANCE"      # Short-term spatial reservation to prevent robot clustering


@dataclass
class EvaporationConfig:
    """Configuration parameters for multi-layer evaporation and numerical stability."""
    evaporation_rates: Dict[PheromoneLayerType, float] = field(default_factory=lambda: {
        PheromoneLayerType.EXPLORATION: 0.01,  # 1% decay per update step
        PheromoneLayerType.SUCCESS: 0.05,      # 5% decay per update step
        PheromoneLayerType.AVOIDANCE: 0.30,    # 30% decay per update step (fast dissipation)
    })
    min_pheromone: float = 0.01
    max_pheromone: float = 50.0


class EvaporationModel:
    """
    Applies discrete exponential decay:
        tau_k(t + dt) = (1 - rho_k)^dt * tau_k(t)
    subject to bounds clamping [tau_min, tau_max].
    """

    def __init__(self, config: EvaporationConfig = None) -> None:
        self.config = config or EvaporationConfig()

    def get_decay_factor(self, layer: PheromoneLayerType, dt: float = 1.0) -> float:
        """Computes discrete decay factor (1 - rho)^dt for the given layer."""
        rate = self.config.evaporation_rates.get(layer, 0.05)
        # Numerical safeguard: factor cannot become negative
        base = max(0.0, 1.0 - rate)
        return math_pow(base, dt)

    def clamp(self, value: float) -> float:
        """Restricts pheromone intensity within [min_pheromone, max_pheromone]."""
        return max(self.config.min_pheromone, min(value, self.config.max_pheromone))


def math_pow(base: float, exp: float) -> float:
    """Safeguarded power computation."""
    if base <= 0.0:
        return 0.0
    return base ** exp
