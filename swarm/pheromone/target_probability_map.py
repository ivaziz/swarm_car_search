"""
Dynamic Target Probability Map for Multi-Robot Search.
Maintains a recursive Bayesian spatial belief grid P(target | location)
updated via positive (candidate sightings, appearance cues) and
negative (explored empty space, confirmed decoy rejection) evidence.
"""

from typing import Dict, List, Optional, Tuple, Any
import math
import numpy as np


class TargetProbabilityMap:
    """
    2D spatial belief grid representing P(target | location) in range [0.01, 0.99].
    Uses log-odds representation for numerical stability under recursive updates:
        L(x, y) = log( P(x, y) / (1 - P(x, y)) )
    """

    def __init__(
        self,
        width: float = 50.0,
        height: float = 50.0,
        resolution: float = 1.0,
        initial_prior: float = 0.50,
    ) -> None:
        self.width = width
        self.height = height
        self.resolution = resolution
        self.cols = max(1, int(math.ceil(width / resolution)))
        self.rows = max(1, int(math.ceil(height / resolution)))

        self.initial_prior = max(0.01, min(0.99, initial_prior))
        init_log_odds = math.log(self.initial_prior / (1.0 - self.initial_prior))
        self.log_odds = np.full((self.rows, self.cols), init_log_odds, dtype=np.float64)

    def world_to_grid(self, x: float, y: float) -> Tuple[int, int]:
        """Converts continuous world coordinates (x, y) to grid indices (col, row)."""
        col = max(0, min(self.cols - 1, int(x / self.resolution)))
        row = max(0, min(self.rows - 1, int(y / self.resolution)))
        return col, row

    def grid_to_world(self, col: int, row: int) -> Tuple[float, float]:
        """Converts grid indices (col, row) to world center coordinates (x, y)."""
        x = (col + 0.5) * self.resolution
        y = (row + 0.5) * self.resolution
        return x, y

    def get_probability(self, x: float, y: float) -> float:
        """Queries posterior target probability at world point (x, y)."""
        c, r = self.world_to_grid(x, y)
        lo = self.log_odds[r, c]
        # Sigmoid conversion
        p = 1.0 / (1.0 + math.exp(-max(-10.0, min(10.0, lo))))
        return round(float(p), 4)

    def get_box_probability(
        self,
        min_x: float,
        min_y: float,
        max_x: float,
        max_y: float,
        mode: str = "mean",
    ) -> float:
        """
        Computes aggregate target probability over a bounding box region.
        mode: 'mean' or 'max'
        """
        c_min, r_min = self.world_to_grid(min_x, min_y)
        c_max, r_max = self.world_to_grid(max_x, max_y)

        c0, c1 = min(c_min, c_max), max(c_min, c_max) + 1
        r0, r1 = min(r_min, r_max), max(r_min, r_max) + 1

        sub_lo = self.log_odds[r0:r1, c0:c1]
        if sub_lo.size == 0:
            return self.initial_prior

        # Sigmoid on array
        probs = 1.0 / (1.0 + np.exp(-np.clip(sub_lo, -10.0, 10.0)))
        if mode == "max":
            return round(float(np.max(probs)), 4)
        return round(float(np.mean(probs)), 4)

    def deposit_positive_evidence(
        self,
        x: float,
        y: float,
        confidence: float,
        radius: float = 4.0,
    ) -> None:
        """
        Positive Bayesian evidence update (vehicle detected, color match, candidate sighting).
        Deposits a 2D Gaussian kernel of positive log-odds.
        """
        # delta_l = log(P(obs|target) / P(obs|~target))
        # Scaled by confidence
        delta_lo = math.log((max(0.01, confidence) / (1.0 - min(0.99, confidence))))
        delta_lo = max(0.2, min(3.5, delta_lo))

        c_center, r_center = self.world_to_grid(x, y)
        radius_cells = max(1, int(math.ceil(radius / self.resolution)))

        r0 = max(0, r_center - radius_cells)
        r1 = min(self.rows, r_center + radius_cells + 1)
        c0 = max(0, c_center - radius_cells)
        c1 = min(self.cols, c_center + radius_cells + 1)

        sigma = max(1.0, radius_cells / 2.0)
        for r in range(r0, r1):
            for c in range(c0, c1):
                d2 = (c - c_center) ** 2 + (r - r_center) ** 2
                weight = math.exp(-d2 / (2.0 * sigma * sigma))
                self.log_odds[r, c] = min(8.0, self.log_odds[r, c] + delta_lo * weight)

    def deposit_negative_evidence(
        self,
        x: float,
        y: float,
        decay_factor: float = 0.15,
        radius: float = 2.5,
    ) -> None:
        """
        Negative Bayesian evidence update (explored region with no target, confirmed decoy).
        Decreases target log-odds in the surveyed region.
        """
        c_center, r_center = self.world_to_grid(x, y)
        radius_cells = max(1, int(math.ceil(radius / self.resolution)))

        r0 = max(0, r_center - radius_cells)
        r1 = min(self.rows, r_center + radius_cells + 1)
        c0 = max(0, c_center - radius_cells)
        c1 = min(self.cols, c_center + radius_cells + 1)

        sigma = max(1.0, radius_cells / 2.0)
        for r in range(r0, r1):
            for c in range(c0, c1):
                d2 = (c - c_center) ** 2 + (r - r_center) ** 2
                weight = math.exp(-d2 / (2.0 * sigma * sigma))
                self.log_odds[r, c] = max(-6.0, self.log_odds[r, c] - decay_factor * weight)

    def get_probability_grid(self) -> np.ndarray:
        """Returns the full (rows, cols) 2D probability matrix in [0.0, 1.0]."""
        return 1.0 / (1.0 + np.exp(-np.clip(self.log_odds, -10.0, 10.0)))

    def initialize_from_environment(self, env: Any) -> None:
        """Initializes spatial target prior distribution from urban zones."""
        if env is None or not hasattr(env, "zones"):
            return
        for zone in env.zones:
            min_x, min_y, max_x, max_y = zone.bounds
            prior = max(0.05, min(0.95, zone.vehicle_prior_probability))
            lo = math.log(prior / (1.0 - prior))
            c_min, r_min = self.world_to_grid(min_x, min_y)
            c_max, r_max = self.world_to_grid(max_x, max_y)
            c0, c1 = min(c_min, c_max), max(c_min, c_max) + 1
            r0, r1 = min(r_min, r_max), max(r_min, r_max) + 1
            self.log_odds[r0:r1, c0:c1] = lo
