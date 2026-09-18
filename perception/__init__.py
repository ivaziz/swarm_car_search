"""
Perception and Computer Vision Subsystem (Vehicle Detection, OCR, Matching).
"""

from .vehicle_detector import TargetVehicle, SimulatedVehicleDetector
from .plate_recognizer import SimulatedPlateRecognizer
from .matching_engine import MatchingEngine, levenshtein_similarity, normalize_plate

__all__ = [
    "TargetVehicle",
    "SimulatedVehicleDetector",
    "SimulatedPlateRecognizer",
    "MatchingEngine",
    "levenshtein_similarity",
    "normalize_plate",
]
