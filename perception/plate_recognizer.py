"""
Simulated Optical Character Recognition (OCR) Engine for License Plate Verification.
"""

from typing import Optional
from interfaces.perception_interface import (
    VehicleDetectionEvent,
    PlateDetectionEvent,
)
from .vehicle_detector import TargetVehicle


class SimulatedPlateRecognizer:
    """
    Simulates high-resolution license plate cropping and deep learning OCR:
    - Crops detected vehicle image region.
    - Performs optical character recognition on license plate.
    - Yields recognized alphanumeric string and OCR confidence score.
    - Attenuates confidence with distance and simulates OCR character noise when far.
    """

    def __init__(
        self,
        base_ocr_confidence: float = 0.95,
        max_ocr_distance: float = 10.0,
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
        Processes vehicle detection crop and produces plate reading.
        """
        dist = getattr(detection, "distance", 0.0)
        if dist <= 5.0:
            dist_factor = 1.0
        else:
            dist_factor = max(0.4, 1.0 - ((dist - 5.0) / (self.max_ocr_distance * 1.5)))
        confidence = min(0.99, self.base_ocr_confidence * (0.8 + 0.2 * detection.confidence) * dist_factor)

        plate_str = vehicle.plate_number.upper().strip()

        # At long range (> 7.0m) or when noise explicitly configured, simulate optical blur/character ambiguity
        if (dist > 7.0 or self.character_noise_rate > 0.0) and len(plate_str) > 2:
            substitutions = {"8": "B", "B": "8", "0": "O", "O": "0", "2": "Z", "Z": "2"}
            chars = list(plate_str)
            for idx in range(len(chars) - 1, -1, -1):
                if chars[idx] in substitutions:
                    chars[idx] = substitutions[chars[idx]]
                    break
            plate_str = "".join(chars)

        evidence_ref = f"evidence_{detection.robot_id}_{plate_str}_{int(detection.timestamp)}.jpg"

        return PlateDetectionEvent(
            robot_id=detection.robot_id,
            timestamp=detection.timestamp,
            plate_number=plate_str,
            ocr_confidence=round(confidence, 3),
            estimated_global_position=detection.estimated_global_position,
            evidence_reference=evidence_ref,
        )
