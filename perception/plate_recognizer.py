"""
Simulated Optical Character Recognition (OCR) Engine for License Plate Verification.
"""

from typing import Optional, Tuple
from interfaces.perception_interface import (
    BoundingBox2D,
    VehicleDetectionEvent,
    PlateDetectionEvent,
)
from .vehicle_detector import TargetVehicle


# Optical Character Confusion Mapping under noise/blur
OCR_CONFUSION_MAP = {
    "8": "B", "B": "8",
    "0": "O", "O": "0",
    "2": "Z", "Z": "2",
    "5": "S", "S": "5",
    "1": "I", "I": "1",
}


class SimulatedPlateRecognizer:
    """
    Simulates high-resolution license plate cropping and deep learning OCR:
    - Crops detected vehicle image region.
    - Performs optical character recognition on license plate.
    - Yields recognized alphanumeric string, per-character confidence, and overall OCR confidence.
    - Attenuates confidence with distance and simulates OCR character noise when far.
    """

    def __init__(
        self,
        base_ocr_confidence: float = 0.95,
        max_ocr_distance: float = 12.0,
        character_noise_rate: float = 0.0,
    ) -> None:
        self.base_ocr_confidence = base_ocr_confidence
        self.max_ocr_distance = max_ocr_distance
        self.character_noise_rate = character_noise_rate

    def recognize_license_plate(
        self,
        detection: VehicleDetectionEvent,
        vehicle: TargetVehicle,
    ) -> PlateDetectionEvent:
        """
        Processes vehicle detection crop and produces plate reading with character confidences.
        """
        dist = getattr(detection, "distance", 0.0)
        if dist <= 5.0:
            dist_factor = 1.0
        else:
            dist_factor = max(0.40, 1.0 - ((dist - 5.0) / (self.max_ocr_distance * 1.5)))

        overall_confidence = min(
            0.99,
            self.base_ocr_confidence * (0.75 + 0.25 * detection.confidence) * dist_factor
        )

        ground_truth_plate = vehicle.plate_number.upper().strip()
        plate_chars = list(ground_truth_plate)
        char_confidences = []

        # Simulate character ambiguity and noise if far away or noisy environment
        should_corrupt = (dist > 7.0 or self.character_noise_rate > 0.0)
        corrupted = False

        for i, ch in enumerate(plate_chars):
            # Per-character confidence slightly fluctuates
            char_conf = max(0.40, min(0.99, overall_confidence - (0.02 * (i % 3))))

            if should_corrupt and not corrupted and ch in OCR_CONFUSION_MAP:
                # Apply character substitution
                plate_chars[i] = OCR_CONFUSION_MAP[ch]
                char_conf = max(0.35, char_conf * 0.70)
                corrupted = True

            char_confidences.append(round(char_conf, 3))

        plate_str = "".join(plate_chars)
        evidence_ref = f"evidence_{detection.robot_id}_{plate_str}_{int(detection.timestamp)}.jpg"

        # Plate sub-bounding box within vehicle crop
        plate_box = BoundingBox2D(ymin=0.60, xmin=0.35, ymax=0.80, xmax=0.65)

        return PlateDetectionEvent(
            robot_id=detection.robot_id,
            timestamp=detection.timestamp,
            plate_number=plate_str,
            ocr_confidence=round(overall_confidence, 3),
            estimated_global_position=detection.estimated_global_position,
            evidence_reference=evidence_ref,
            plate_bounding_box=plate_box,
            raw_ocr_text=plate_str,
            character_confidences=tuple(char_confidences),
        )
