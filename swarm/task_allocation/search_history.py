"""
Spatial Search History & Memory Tracker.
Maintains spatio-temporal logs of visited zones, candidate sighting clusters,
and unpromising empty sweeps to prevent redundant re-exploration.
"""

from typing import Dict, List, Optional, Tuple, Set
import math
import time


class SearchSpatialMemory:
    """
    Spatio-temporal memory maintaining search visitation counts, dwell times,
    and visual detection histories per spatial grid cell.
    """

    def __init__(
        self,
        width: float = 50.0,
        height: float = 50.0,
        bin_size: float = 2.0,
    ) -> None:
        self.width = width
        self.height = height
        self.bin_size = bin_size
        self.cols = max(1, int(math.ceil(width / bin_size)))
        self.rows = max(1, int(math.ceil(height / bin_size)))

        # (col, row) -> visit count
        self.visit_counts: Dict[Tuple[int, int], int] = {}
        # (col, row) -> last visited timestamp
        self.last_visit_time: Dict[Tuple[int, int], float] = {}
        # (col, row) -> set of visiting robots
        self.visiting_robots: Dict[Tuple[int, int], Set[str]] = {}
        # (col, row) -> candidate sightings count
        self.candidate_counts: Dict[Tuple[int, int], int] = {}
        # (col, row) -> confirmed decoy rejections count
        self.decoy_counts: Dict[Tuple[int, int], int] = {}

    def _to_bin(self, x: float, y: float) -> Tuple[int, int]:
        c = max(0, min(self.cols - 1, int(x / self.bin_size)))
        r = max(0, min(self.rows - 1, int(y / self.bin_size)))
        return c, r

    def record_visit(self, robot_id: str, x: float, y: float, timestamp: float) -> None:
        """Records a robot traversal through spatial bin (x, y)."""
        b = self._to_bin(x, y)
        self.visit_counts[b] = self.visit_counts.get(b, 0) + 1
        self.last_visit_time[b] = timestamp
        if b not in self.visiting_robots:
            self.visiting_robots[b] = set()
        self.visiting_robots[b].add(robot_id)

    def record_sighting(
        self,
        x: float,
        y: float,
        is_candidate: bool = True,
        is_decoy: bool = False,
    ) -> None:
        """Records a candidate sighting or decoy rejection event."""
        b = self._to_bin(x, y)
        if is_candidate:
            self.candidate_counts[b] = self.candidate_counts.get(b, 0) + 1
        if is_decoy:
            self.decoy_counts[b] = self.decoy_counts.get(b, 0) + 1

    def get_redundancy_penalty(self, x: float, y: float) -> float:
        """
        Computes redundancy penalty in [0.0, 1.0].
        High if cell has been swept multiple times with zero candidates.
        """
        b = self._to_bin(x, y)
        visits = self.visit_counts.get(b, 0)
        candidates = self.candidate_counts.get(b, 0)

        if visits <= 1:
            return 0.0

        # If candidates were sighted here, do not penalize revisiting for confirmation
        if candidates > 0:
            return 0.05

        # Penalty rises smoothly with repeat unpromising visits
        penalty = 1.0 - math.exp(-0.35 * (visits - 1))
        return round(float(min(0.95, penalty)), 3)

    def get_historical_value(self, x: float, y: float) -> float:
        """
        Computes historical exploration attractiveness in [0.0, 1.0].
        Highest for unvisited cells and cells with candidate history.
        """
        b = self._to_bin(x, y)
        visits = self.visit_counts.get(b, 0)
        candidates = self.candidate_counts.get(b, 0)
        decoys = self.decoy_counts.get(b, 0)

        if decoys > 0 and candidates == 0:
            # Confirmed decoy area with no other candidate
            return 0.15

        if candidates > 0:
            # High priority to verify candidate sightings
            return min(1.0, 0.80 + 0.10 * candidates)

        if visits == 0:
            # Fresh pristine frontier
            return 1.0

        # Decays with repeat visits
        return round(float(max(0.10, math.exp(-0.40 * visits))), 3)
