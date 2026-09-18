"""
Configuration Parameters and Weights for Ant Colony Optimization (ACO) Engine.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional
import os


@dataclass
class ACOWeights:
    """Multi-objective exponents for task scoring."""
    alpha: float = 1.2    # Positive pheromone (exploitation)
    beta: float = 1.5     # Information gain / unexplored frontier size
    gamma: float = 1.8    # Vehicle detection prior probability
    delta: float = 1.0    # Dynamic mission urgency / priority
    epsilon: float = 1.3  # Distance / travel cost penalty
    zeta: float = 2.0     # Congestion penalty (peer robot overlap)
    theta: float = 1.5    # Avoidance / recent visit pheromone penalty


@dataclass
class StochasticParams:
    """Stochastic exploration and Boltzmann temperature parameters."""
    temperature: float = 1.0      # Softmax exploration temperature
    epsilon_greedy: float = 0.05  # Exploratory perturbation probability
    deterministic: bool = False   # If True, strictly picks max score


@dataclass
class ACOConfig:
    """Unified configuration for ACO task selection."""
    weights: ACOWeights = field(default_factory=ACOWeights)
    stochastic: StochasticParams = field(default_factory=StochasticParams)
    min_distance_offset: float = 1.0

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ACOConfig":
        """Builds ACOConfig from nested dictionary."""
        weights_dict = data.get("objective_function_weights", {})
        stoch_dict = data.get("stochastic_selection", {})

        weights = ACOWeights(
            alpha=float(weights_dict.get("alpha", 1.2)),
            beta=float(weights_dict.get("beta", 1.5)),
            gamma=float(weights_dict.get("gamma", 1.8)),
            delta=float(weights_dict.get("delta", 1.0)),
            epsilon=float(weights_dict.get("epsilon", 1.3)),
            zeta=float(weights_dict.get("zeta", 2.0)),
            theta=float(weights_dict.get("theta", 1.5)),
        )
        stochastic = StochasticParams(
            temperature=float(stoch_dict.get("temperature", 1.0)),
            epsilon_greedy=float(stoch_dict.get("epsilon_greedy", 0.05)),
            deterministic=bool(stoch_dict.get("deterministic", False)),
        )
        return cls(weights=weights, stochastic=stochastic)

    @classmethod
    def from_yaml(cls, file_path: str) -> "ACOConfig":
        """Loads ACO configuration from a YAML file."""
        if not os.path.exists(file_path):
            return cls()

        try:
            import yaml
            with open(file_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
            return cls.from_dict(data)
        except Exception:
            # Fallback to default if YAML parser unavailable
            return cls()
