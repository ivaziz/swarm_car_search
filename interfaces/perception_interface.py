"""
Abstract Interface and Event Definitions for Perception & Vision Subsystem.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Tuple
from .slam_interface import Pose2D


@dataclass(frozen=True)
class BoundingBox2D:
    """Bounding box pixel coordinates [ymin, xmin, ymax, xmax]."""
    ymin: float
    xmin: float
    ymax: float
    xmax: float


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


@dataclass(frozen=True)
class PlateDetectionEvent:
    """Triggered when OCR engine processes vehicle crop and extracts license plate."""
    robot_id: str
    timestamp: float
    plate_number: str
    ocr_confidence: float
    estimated_global_position: Tuple[float, float]
    evidence_reference: Optional[str] = None


@dataclass(frozen=True)
class MatchEvent:
    """Triggered when plate number matches a target in the missing/stolen car watchlist."""
    target_id: str
    plate_number: str
    match_confidence: float
    confirmed: bool
    location: Tuple[float, float]
    reporting_robot_id: str
    timestamp: float
    evidence_reference: Optional[str] = None


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
