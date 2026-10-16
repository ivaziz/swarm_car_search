"""
Abstract Interface and Event Definitions for Perception & Vision Subsystem.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, Tuple, Dict, Any
from .slam_interface import Pose2D


@dataclass(frozen=True)
class BoundingBox2D:
    """Bounding box pixel coordinates [ymin, xmin, ymax, xmax]."""
    ymin: float
    xmin: float
    ymax: float
    xmax: float


@dataclass(frozen=True)
class TargetAppearanceProfile:
    """Target visual signature specification (class, color, aspect ratio)."""
    vehicle_class: str = "sedan"
    color: str = "dark_gray"
    color_rgb: Tuple[int, int, int] = (55, 55, 60)
    aspect_ratio: float = 2.2  # length / width


@dataclass(frozen=True)
class VehicleDetectionEvent:
    """Triggered when vehicle detection model identifies a vehicle in camera feed."""
    robot_id: str
    timestamp: float
    vehicle_class: str
    confidence: float
    bounding_box: BoundingBox2D
    estimated_global_position: Tuple[float, float]
    image_reference: Optional[str] = None
    detected_vehicle_id: Optional[str] = None
    distance: float = 0.0
    plate_candidate: Optional[str] = None
    target_probability: float = 0.5
    observed_color: str = "dark_gray"
    appearance_confidence: float = 0.90
    plate_detection_confidence: float = 0.90


@dataclass(frozen=True)
class PlateDetectionEvent:
    """Triggered when OCR engine processes vehicle crop and extracts license plate."""
    robot_id: str
    timestamp: float
    plate_number: str
    ocr_confidence: float
    estimated_global_position: Tuple[float, float]
    evidence_reference: Optional[str] = None
    plate_bounding_box: Optional[BoundingBox2D] = None
    raw_ocr_text: Optional[str] = None
    character_confidences: Optional[Tuple[float, ...]] = None


@dataclass(frozen=True)
class MatchEvent:
    """
    Triggered when candidate passes multi-stage verification against target watchlist.
    Contains granular confidence scores from every perception stage.
    """
    target_id: str
    plate_number: str
    match_confidence: float
    confirmed: bool
    location: Tuple[float, float]
    reporting_robot_id: str
    timestamp: float
    evidence_reference: Optional[str] = None
    # Granular multi-stage confidence breakdown
    vehicle_confidence: float = 0.90
    plate_confidence: float = 0.90
    ocr_confidence: float = 0.90
    appearance_confidence: float = 0.90
    temporal_confidence: float = 0.90
    frame_count: int = 1
    stage_breakdown: Dict[str, float] = field(default_factory=dict)


class IPerceptionProvider(ABC):
    """Abstract interface for vehicle detection and license plate recognition."""

    @abstractmethod
    def detect_vehicles(self, robot_id: str, camera_frame_id: str) -> Optional[VehicleDetectionEvent]:
        """Runs vehicle detector model on current sensor frame."""
        pass

    @abstractmethod
    def recognize_license_plate(self, detection: VehicleDetectionEvent) -> Optional[PlateDetectionEvent]:
        """Runs OCR model on cropped vehicle region."""
        pass

    @abstractmethod
    def match_watchlist(self, plate_event: PlateDetectionEvent) -> Optional[MatchEvent]:
        """Matches extracted plate against target car database."""
        pass
