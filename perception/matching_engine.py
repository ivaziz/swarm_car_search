"""
Target Vehicle Watchlist Matching Engine with Exact and Levenshtein Fuzzy Alignment.
"""

from typing import Dict, List, Optional, Tuple
from interfaces.perception_interface import (
    PlateDetectionEvent,
    MatchEvent,
)
from .vehicle_detector import TargetVehicle


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
                dp[i - 1][j] + 1,      # Deletion
                dp[i][j - 1] + 1,      # Insertion
                dp[i - 1][j - 1] + cost # Substitution
            )

    dist = dp[len1][len2]
    max_len = max(len1, len2)
    return max(0.0, 1.0 - (dist / float(max_len)))


def normalize_plate(plate: str) -> str:
    """Strips hyphens, spaces, and standardizes to uppercase."""
    return "".join(ch for ch in plate.upper() if ch.isalnum())


class MatchingEngine:
    """
    Evaluates detected license plates against missing/stolen vehicle watchlist records.
    Produces formal MatchEvents upon verified alignment.
    """

    def __init__(self, match_threshold: float = 0.80, candidate_threshold: float = 0.50) -> None:
        self.match_threshold = match_threshold
        self.candidate_threshold = candidate_threshold
        # Watchlist indexed by normalized plate string
        self.watchlist: Dict[str, TargetVehicle] = {}

    def add_target_vehicle(self, target: TargetVehicle) -> None:
        """Registers a target vehicle into the missing/stolen car watchlist."""
        norm_key = normalize_plate(target.plate_number)
        self.watchlist[norm_key] = target

    def match_plate(self, plate_event: PlateDetectionEvent) -> Optional[MatchEvent]:
        """
        Compares recognized plate against active watchlist.
        Returns MatchEvent if similarity exceeds threshold, None otherwise.
        """
        norm_detected = normalize_plate(plate_event.plate_number)
        best_match: Optional[TargetVehicle] = None
        best_similarity = 0.0

        for norm_target, vehicle in self.watchlist.items():
            sim = levenshtein_similarity(norm_detected, norm_target)
            if sim > best_similarity:
                best_similarity = sim
                best_match = vehicle

        if best_match is not None and best_similarity >= self.match_threshold:
            composite_confidence = round(plate_event.ocr_confidence * best_similarity, 3)
            is_confirmed = composite_confidence >= self.match_threshold

            return MatchEvent(
                target_id=best_match.target_id,
                plate_number=best_match.plate_number,
                match_confidence=composite_confidence,
                confirmed=is_confirmed,
                location=plate_event.estimated_global_position,
                reporting_robot_id=plate_event.robot_id,
                timestamp=plate_event.timestamp,
                evidence_reference=plate_event.evidence_reference,
            )

        return None

    def classify_plate(self, plate_event: PlateDetectionEvent) -> Tuple[str, Optional[MatchEvent], float]:
        """
        Classifies plate detection into ('CONFIRMED', 'CANDIDATE', 'NON_TARGET').
        Returns (classification_label, match_event_or_none, similarity_score).
        """
        norm_detected = normalize_plate(plate_event.plate_number)
        best_match: Optional[TargetVehicle] = None
        best_similarity = 0.0

        for norm_target, vehicle in self.watchlist.items():
            sim = levenshtein_similarity(norm_detected, norm_target)
            if sim > best_similarity:
                best_similarity = sim
                best_match = vehicle

        if best_match is None:
            return "NON_TARGET", None, 0.0

        composite_confidence = round(plate_event.ocr_confidence * best_similarity, 3)

        match_ev = MatchEvent(
            target_id=best_match.target_id,
            plate_number=best_match.plate_number,
            match_confidence=composite_confidence,
            confirmed=(best_similarity >= self.match_threshold and composite_confidence >= self.match_threshold),
            location=plate_event.estimated_global_position,
            reporting_robot_id=plate_event.robot_id,
            timestamp=plate_event.timestamp,
            evidence_reference=plate_event.evidence_reference,
        )

        if match_ev.confirmed:
            return "CONFIRMED", match_ev, best_similarity
        elif best_similarity >= self.candidate_threshold:
            return "CANDIDATE", match_ev, best_similarity
        else:
            return "NON_TARGET", None, best_similarity
