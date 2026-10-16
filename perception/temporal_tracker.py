"""
Multi-Frame Temporal Target Tracker & Consistency Accumulator.
Aggregates perception, OCR, and appearance evidence across multiple consecutive
observation frames to prevent single-frame false alarms and verify persistent targets.
"""

from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
import math
import time


def levenshtein_similarity(s1: str, s2: str) -> float:
    """Computes normalized string similarity in range [0.0, 1.0] using Levenshtein distance."""
    if s1 == s2:
        return 1.0
    if not s1 or not s2:
        return 0.0

    len1, len2 = len(s1), len(s2)
    dp = [[0] * (len2 + 1) for _ in range(len1 + 1)]

    for i in range(len1 + 1):
        dp[i][0] = i
    for j in range(len2 + 1):
        dp[0][j] = j

    for i in range(1, len1 + 1):
        for j in range(1, len2 + 1):
            cost = 0 if s1[i - 1] == s2[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
                dp[i - 1][j - 1] + cost
            )

    dist = dp[len1][len2]
    max_len = max(len1, len2)
    return max(0.0, 1.0 - (dist / float(max_len)))


@dataclass
class ObservationFrame:
    """Single-frame observation entry for a tracked vehicle candidate."""
    timestamp: float
    robot_id: str
    plate_reading: str
    ocr_confidence: float
    vehicle_confidence: float
    appearance_confidence: float
    estimated_position: Tuple[float, float]
    raw_ocr_similarity: float = 1.0


@dataclass
class CandidateTrack:
    """Persistent temporal track for a vehicle candidate across successive camera frames."""
    track_id: str
    first_seen: float
    last_seen: float
    position: Tuple[float, float]
    observations: List[ObservationFrame] = field(default_factory=list)
    accumulated_confidence: float = 0.0
    confirmed: bool = False
    target_plate: str = "DXB-88392"

    @property
    def frame_count(self) -> int:
        return len(self.observations)

    def get_modal_plate(self) -> Tuple[str, float]:
        """Returns the most frequent plate reading in the track and its vote frequency [0.0, 1.0]."""
        if not self.observations:
            return "", 0.0
        counts: Dict[str, int] = {}
        for obs in self.observations:
            norm = "".join(ch for ch in obs.plate_reading.upper() if ch.isalnum())
            counts[norm] = counts.get(norm, 0) + 1
        best_plate, best_cnt = max(counts.items(), key=lambda x: x[1])
        ratio = best_cnt / float(len(self.observations))
        return best_plate, ratio


class MultiFrameTargetTracker:
    """
    Multi-frame temporal tracking & Bayesian confidence accumulator:
    - Associates detections with tracks using spatial gating (Euclidean distance threshold).
    - Accumulates multi-stage confidence scores across time.
    - Requires N consecutive/consistent frames before declaring TARGET_VERIFIED.
    """

    def __init__(
        self,
        required_confirmations: int = 3,
        association_radius: float = 3.0,
        temporal_decay_half_life: float = 15.0,
        min_temporal_confidence: float = 0.82,
        ewma_alpha: float = 0.40,
    ) -> None:
        self.required_confirmations = required_confirmations
        self.association_radius = association_radius
        self.temporal_decay_half_life = temporal_decay_half_life
        self.min_temporal_confidence = min_temporal_confidence
        self.ewma_alpha = ewma_alpha
        self.tracks: Dict[str, CandidateTrack] = {}
        self._next_track_id = 1

    def _find_or_create_track(
        self,
        pos: Tuple[float, float],
        timestamp: float,
        target_plate: str,
    ) -> CandidateTrack:
        """Finds closest existing track within association radius or creates a new track."""
        best_track: Optional[CandidateTrack] = None
        best_dist = float("inf")

        for track in self.tracks.values():
            dx = pos[0] - track.position[0]
            dy = pos[1] - track.position[1]
            dist = math.hypot(dx, dy)
            if dist < self.association_radius and dist < best_dist:
                best_dist = dist
                best_track = track

        if best_track is not None:
            # Update running centroid position
            n = len(best_track.observations)
            new_x = (best_track.position[0] * n + pos[0]) / (n + 1)
            new_y = (best_track.position[1] * n + pos[1]) / (n + 1)
            best_track.position = (round(new_x, 2), round(new_y, 2))
            best_track.last_seen = timestamp
            return best_track

        # Create new track
        tid = f"track_{self._next_track_id:03d}"
        self._next_track_id += 1
        new_track = CandidateTrack(
            track_id=tid,
            first_seen=timestamp,
            last_seen=timestamp,
            position=pos,
            target_plate=target_plate,
        )
        self.tracks[tid] = new_track
        return new_track

    def add_observation(
        self,
        timestamp: float,
        robot_id: str,
        plate_reading: str,
        ocr_confidence: float,
        vehicle_confidence: float,
        appearance_confidence: float,
        estimated_position: Tuple[float, float],
        target_plate: str = "DXB-88392",
        raw_ocr_similarity: float = 1.0,
    ) -> Tuple[bool, float, Dict[str, Any]]:
        """
        Ingests a frame observation and updates the temporal track.

        Returns:
            (is_confirmed, temporal_confidence, track_diagnostics)
        """
        track = self._find_or_create_track(estimated_position, timestamp, target_plate)
        obs = ObservationFrame(
            timestamp=timestamp,
            robot_id=robot_id,
            plate_reading=plate_reading,
            ocr_confidence=ocr_confidence,
            vehicle_confidence=vehicle_confidence,
            appearance_confidence=appearance_confidence,
            estimated_position=estimated_position,
            raw_ocr_similarity=raw_ocr_similarity,
        )
        track.observations.append(obs)

        # 1. Compute frame-level instantaneous fused score
        instant_score = (
            0.20 * vehicle_confidence
            + 0.35 * (ocr_confidence * raw_ocr_similarity)
            + 0.25 * appearance_confidence
            + 0.20 * raw_ocr_similarity
        )

        # 2. Update EWMA accumulated confidence
        if track.accumulated_confidence <= 0.0:
            track.accumulated_confidence = instant_score
        else:
            track.accumulated_confidence = (
                (1.0 - self.ewma_alpha) * track.accumulated_confidence
                + self.ewma_alpha * instant_score
            )

        # 3. Calculate temporal consistency ratio using fuzzy Levenshtein alignment
        modal_norm, modal_ratio = track.get_modal_plate()
        norm_target = "".join(ch for ch in target_plate.upper() if ch.isalnum())
        modal_sim = levenshtein_similarity(modal_norm, norm_target) if modal_norm else 0.0
        target_match_ratio = modal_ratio * modal_sim

        # 4. Multi-frame count factor: increases towards 1.0 as frame count reaches required
        frame_factor = min(1.0, float(track.frame_count) / float(self.required_confirmations))

        # 5. Composite Temporal Confidence
        temporal_conf = (
            0.50 * track.accumulated_confidence
            + 0.30 * target_match_ratio
            + 0.20 * frame_factor
        )
        temporal_conf = max(0.0, min(0.99, round(temporal_conf, 3)))

        # 6. Verification test: requires at least required_confirmations, confidence threshold, and strong fuzzy match
        is_confirmed = (
            track.frame_count >= self.required_confirmations
            and temporal_conf >= self.min_temporal_confidence
            and modal_sim >= 0.75
        )

        if is_confirmed:
            track.confirmed = True

        diagnostics = {
            "track_id": track.track_id,
            "frame_count": track.frame_count,
            "required_frames": self.required_confirmations,
            "accumulated_confidence": round(track.accumulated_confidence, 3),
            "temporal_consistency_ratio": round(target_match_ratio, 3),
            "temporal_confidence": temporal_conf,
            "is_confirmed": is_confirmed,
            "reporting_robots": list(set(o.robot_id for o in track.observations)),
        }

        return is_confirmed, temporal_conf, diagnostics

    def get_track_history(self, track_id: str) -> Optional[CandidateTrack]:
        return self.tracks.get(track_id)

    def prune_stale_tracks(self, current_time: float, max_age: float = 60.0) -> None:
        """Removes old inactive tracks that were not confirmed."""
        to_del = []
        for tid, t in self.tracks.items():
            if not t.confirmed and (current_time - t.last_seen) > max_age:
                to_del.append(tid)
        for tid in to_del:
            del self.tracks[tid]
