"""
Baseline Benchmark Algorithms for Multi-Robot Task Allocation Comparison.
Includes Ant Colony Optimization (ACO), Greedy Nearest Frontier, and Random Walk.
"""

from __future__ import annotations
from enum import Enum
from typing import Optional, Any

from swarm.task_allocation.allocation_strategy import (
    ITaskAllocationStrategy,
    ACOAllocationStrategy,
    NearestFrontierStrategy,
    RandomAllocationStrategy,
)
from swarm.aco.aco_config import ACOConfig
from swarm.aco.aco_engine import ACOEngine


class StrategyType(str, Enum):
    """Enumeration of evaluated multi-robot allocation strategies."""
    ACO = "ACO"                               # Proposed Ant Colony Optimization with multi-objective stigmergy
    NEAREST_FRONTIER = "NEAREST_FRONTIER"     # Classical Greedy Nearest Frontier (Yamauchi baseline)
    RANDOM = "RANDOM"                         # Uncoordinated stochastic exploration baseline


def create_strategy(
    strategy_type: StrategyType | str,
    aco_config: Optional[ACOConfig] = None,
    seed: Optional[int] = None,
) -> ITaskAllocationStrategy:
    """
    Factory creating task allocation strategies for benchmark experiments.
    """
    stype = StrategyType(strategy_type) if isinstance(strategy_type, str) else strategy_type

    if stype == StrategyType.ACO:
        cfg = aco_config or ACOConfig()
        return ACOAllocationStrategy(ACOEngine(cfg))
    elif stype == StrategyType.NEAREST_FRONTIER:
        return NearestFrontierStrategy()
    elif stype == StrategyType.RANDOM:
        return RandomAllocationStrategy(seed=seed)
    else:
        raise ValueError(f"Unknown allocation strategy: {strategy_type}")


__all__ = [
    "StrategyType",
    "create_strategy",
    "ITaskAllocationStrategy",
    "ACOAllocationStrategy",
    "NearestFrontierStrategy",
    "RandomAllocationStrategy",
]
