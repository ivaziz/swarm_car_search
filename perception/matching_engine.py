"""
Target Vehicle Watchlist Matching Engine with Exact and Levenshtein Fuzzy Alignment.
"""

from typing import Dict, List, Optional, Tuple, Any
from interfaces.perception_interface import (
    VehicleDetectionEvent,
    PlateDetectionEvent,
    MatchEvent,
    TargetAppearanceProfile,
)
from .vehicle_detector import TargetVehicle
from .appearance_verifier import VehicleAppearanceVerifier
from .temporal_tracker import MultiFrameTargetTracker


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
    Multi-Stage Target Vehicle Watchlist Matching & Verification Engine.
    Executes:
    1. Vehicle Detection Confidence (C_veh)
    2. Plate Detection Confidence (C_plate)
    3. OCR String Recognition & Confusion-Tolerant Similarity (C_ocr * S_plate)
    4. Vehicle Appearance Signature Match (C_app: color, class, aspect ratio)
    5. Multi-Frame Temporal Consistency & Consensus (C_temp)

    Produces formal MatchEvents with granular confidence breakdown only upon temporal verification.
    """

    def __init__(
        self,
        match_threshold: float = 0.75,
        candidate_threshold: float = 0.45,
        required_confirmations: int = 1,
        appearance_verifier: Optional[VehicleAppearanceVerifier] = None,
        temporal_tracker: Optional[MultiFrameTargetTracker] = None,
        # Multi-stage stage fusion weights
        w_vehicle: float = 0.15,
        w_plate: float = 0.15,
        w_ocr: float = 0.30,
        w_appearance: float = 0.20,
        w_temporal: float = 0.20,
    ) -> None:
        self.match_threshold = match_threshold
        self.candidate_threshold = candidate_threshold
        self.w_vehicle = w_vehicle
        self.w_plate = w_plate
        self.w_ocr = w_ocr
        self.w_appearance = w_appearance
        self.w_temporal = w_temporal

        self.appearance_verifier = appearance_verifier or VehicleAppearanceVerifier()
        self.temporal_tracker = temporal_tracker or MultiFrameTargetTracker(
            required_confirmations=required_confirmations,
            min_temporal_confidence=match_threshold,
        )
        # Watchlist indexed by normalized plate string
        self.watchlist: Dict[str, TargetVehicle] = {}

    def add_target_vehicle(self, target: TargetVehicle) -> None:
        """Registers a target vehicle into the missing/stolen car watchlist."""
        norm_key = normalize_plate(target.plate_number)
        self.watchlist[norm_key] = target

    def evaluate_multi_stage_match(
        self,
        plate_event: PlateDetectionEvent,
        detection_event: Optional[VehicleDetectionEvent] = None,
    ) -> Tuple[str, Optional[MatchEvent], float, Dict[str, float]]:
        """
        Executes the full multi-stage verification pipeline across perception signals.
        Returns:
            (classification_label, match_event, composite_confidence, stage_breakdown)
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
            return "NON_TARGET", None, 0.0, {}

        # 1. Vehicle Detection Confidence (C_veh)
        c_veh = getattr(detection_event, "confidence", 0.90) if detection_event else 0.90

        # 2. Plate Detection Confidence (C_plate)
        c_plate = getattr(detection_event, "plate_detection_confidence", 0.90) if detection_event else 0.90

        # 3. OCR Confidence & Plate Similarity (C_ocr * S_plate)
        c_ocr = plate_event.ocr_confidence
        ocr_composite = c_ocr * best_similarity

        # 4. Appearance Confidence (C_app)
        if detection_event is not None:
            c_app = getattr(detection_event, "appearance_confidence", 0.90)
        else:
            c_app, _ = self.appearance_verifier.verify_appearance(
                observed_class=best_match.vehicle_class,
                observed_color=best_match.color,
            )

        # 5. Multi-Frame Temporal Accumulation (C_temp)
        pos = plate_event.estimated_global_position
        is_temp_confirmed, c_temp, track_diag = self.temporal_tracker.add_observation(
            timestamp=plate_event.timestamp,
            robot_id=plate_event.robot_id,
            plate_reading=plate_event.plate_number,
            ocr_confidence=c_ocr,
            vehicle_confidence=c_veh,
            appearance_confidence=c_app,
            estimated_position=pos,
            target_plate=best_match.plate_number,
            raw_ocr_similarity=best_similarity,
        )

        # 6. Fused Composite Target Confidence
        composite_confidence = (
            self.w_vehicle * c_veh
            + self.w_plate * c_plate
            + self.w_ocr * ocr_composite
            + self.w_appearance * c_app
            + self.w_temporal * c_temp
        )
        composite_confidence = max(0.0, min(0.99, round(composite_confidence, 3)))

        stage_breakdown = {
            "vehicle_confidence": round(c_veh, 3),
            "plate_confidence": round(c_plate, 3),
            "ocr_confidence": round(c_ocr, 3),
            "plate_similarity": round(best_similarity, 3),
            "appearance_match": round(c_app, 3),
            "temporal_consistency": round(c_temp, 3),
            "final_target_confidence": composite_confidence,
            "observations_count": track_diag.get("frame_count", 1),
            "required_frames": self.temporal_tracker.required_confirmations,
        }

        # 7. Classification logic
        is_confirmed = (
            is_temp_confirmed
            and composite_confidence >= self.match_threshold
            and best_similarity >= self.candidate_threshold
        )

        match_ev = MatchEvent(
            target_id=best_match.target_id,
            plate_number=best_match.plate_number,
            match_confidence=composite_confidence,
            confirmed=is_confirmed,
            location=pos,
            reporting_robot_id=plate_event.robot_id,
            timestamp=plate_event.timestamp,
            evidence_reference=plate_event.evidence_reference,
            vehicle_confidence=round(c_veh, 3),
            plate_confidence=round(c_plate, 3),
            ocr_confidence=round(c_ocr, 3),
            appearance_confidence=round(c_app, 3),
            temporal_confidence=round(c_temp, 3),
            frame_count=track_diag.get("frame_count", 1),
            stage_breakdown=stage_breakdown,
        )

        if is_confirmed:
            return "CONFIRMED", match_ev, composite_confidence, stage_breakdown
        elif best_similarity >= self.candidate_threshold and composite_confidence >= 0.40:
            return "CANDIDATE", match_ev, composite_confidence, stage_breakdown
        else:
            return "NON_TARGET", None, composite_confidence, stage_breakdown

    def match_plate(
        self,
        plate_event: PlateDetectionEvent,
        detection_event: Optional[VehicleDetectionEvent] = None,
    ) -> Optional[MatchEvent]:
        """Compares recognized plate against active watchlist using multi-stage verification."""
        classification, match_ev, _, _ = self.evaluate_multi_stage_match(plate_event, detection_event)
        if classification == "CONFIRMED" and match_ev is not None:
            return match_ev
        return None

    def classify_plate(
        self,
        plate_event: PlateDetectionEvent,
        detection_event: Optional[VehicleDetectionEvent] = None,
    ) -> Tuple[str, Optional[MatchEvent], float]:
        """
        Classifies plate detection into ('CONFIRMED', 'CANDIDATE', 'NON_TARGET').
        Returns (classification_label, match_event_or_none, similarity_score).
        """
        classification, match_ev, comp_conf, _ = self.evaluate_multi_stage_match(plate_event, detection_event)
        norm_detected = normalize_plate(plate_event.plate_number)
        best_sim = 0.0
        for norm_target in self.watchlist.keys():
            sim = levenshtein_similarity(norm_detected, norm_target)
            if sim > best_sim:
                best_sim = sim
        return classification, match_ev, best_sim
