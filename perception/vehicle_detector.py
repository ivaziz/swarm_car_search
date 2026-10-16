from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional, Tuple
import math
from interfaces.slam_interface import Pose2D
from interfaces.perception_interface import (
    BoundingBox2D,
    VehicleDetectionEvent,
    TargetAppearanceProfile,
)
from simulation.environment import UrbanEnvironment
from .appearance_verifier import VehicleAppearanceVerifier


@dataclass
class TargetVehicle:
    """Ground-truth representation of a vehicle located in the urban environment."""
    target_id: str
    plate_number: str
    vehicle_class: str
    color: str
    position: Tuple[float, float]
    is_stolen: bool = True
    orientation: float = 0.0
    detection_radius: float = 15.0

    @property
    def is_target(self) -> bool:
        """Alias for is_stolen indicating if vehicle is the missing target."""
        return self.is_stolen


class IVehicleDetector(ABC):
    """Abstract interface for onboard visual vehicle detectors (YOLO / CNN)."""

    @abstractmethod
    def detect_vehicles(
        self,
        robot_id: str,
        robot_pose: Pose2D,
        timestamp: float,
        vehicles: List[TargetVehicle],
        env: Optional[UrbanEnvironment] = None,
    ) -> List[Tuple[VehicleDetectionEvent, TargetVehicle]]:
        """Executes camera detection cycle, returning observed vehicles and ground-truth reference."""
        pass


class SimulatedVehicleDetector(IVehicleDetector):
    """
    Simulates onboard YOLO/CNN visual object detection:
    - Verifies range to target within camera optical limits.
    - Verifies angular bearing within camera Field-of-View (FoV).
    - Checks line-of-sight against urban obstacles (walls, buildings).
    - Produces realistic BoundingBox2D, distance-attenuated confidence, and appearance signature.
    """

    def __init__(
        self,
        max_detection_range: float = 14.0,
        fov_degrees: float = 90.0,
        base_confidence: float = 0.94,
        target_profile: Optional[TargetAppearanceProfile] = None,
    ) -> None:
        self.max_detection_range = max_detection_range
        self.fov_radians = math.radians(fov_degrees)
        self.base_confidence = base_confidence
        self.appearance_verifier = VehicleAppearanceVerifier(default_target_profile=target_profile)

    def _has_line_of_sight(
        self,
        rx: float,
        ry: float,
        tx: float,
        ty: float,
        env: Optional[UrbanEnvironment] = None,
    ) -> bool:
        """Traces ray between robot and target to check for static obstacle obstruction."""
        if env is None:
            return True

        dx = tx - rx
        dy = ty - ry
        dist = math.hypot(dx, dy)
        if dist < 1e-3:
            return True

        steps = int(math.ceil(dist / (env.resolution * 0.5)))
        step_x = dx / steps
        step_y = dy / steps

        for i in range(1, steps):
            check_x = rx + i * step_x
            check_y = ry + i * step_y
            g = env.world_to_grid(check_x, check_y)
            if g is not None and env.get_cell(g[0], g[1]) == 100:
                # Obstacle blocks line of sight
                return False

        return True

    def detect_vehicles(
        self,
        robot_id: str,
        robot_pose: Pose2D,
        timestamp: float,
        vehicles: List[TargetVehicle],
        env: Optional[UrbanEnvironment] = None,
    ) -> List[Tuple[VehicleDetectionEvent, TargetVehicle]]:
        """
        Executes camera detection cycle, returning observed vehicles and their ground-truth reference.
        """
        detected: List[Tuple[VehicleDetectionEvent, TargetVehicle]] = []

        for vehicle in vehicles:
            vx, vy = vehicle.position
            dx = vx - robot_pose.x
            dy = vy - robot_pose.y
            dist = math.hypot(dx, dy)

            # Range test
            effective_range = min(self.max_detection_range, getattr(vehicle, "detection_radius", self.max_detection_range))
            if dist > effective_range:
                continue

            # Field-of-view angular test
            bearing_to_target = math.atan2(dy, dx)
            heading_error = (bearing_to_target - robot_pose.theta + math.pi) % (2.0 * math.pi) - math.pi
            if abs(heading_error) > (self.fov_radians / 2.0):
                continue

            # Line of sight test
            if not self._has_line_of_sight(robot_pose.x, robot_pose.y, vx, vy, env):
                continue

            # 1. Distance-attenuated vehicle detection confidence
            dist_factor = max(0.0, 1.0 - (dist / (self.max_detection_range * 1.5)))
            vehicle_confidence = min(0.99, max(0.50, self.base_confidence * dist_factor))

            # 2. Appearance Verification (Color & Shape)
            app_conf, _ = self.appearance_verifier.verify_appearance(
                observed_class=vehicle.vehicle_class,
                observed_color=vehicle.color,
            )

            # 3. Plate detection likelihood & confidence (plate detector stage)
            if dist <= 6.0:
                plate_det_conf = min(0.98, max(0.85, 0.95 - (dist / 20.0)))
            elif dist <= 10.0:
                plate_det_conf = min(0.85, max(0.60, 0.90 - (dist / 15.0)))
            else:
                plate_det_conf = max(0.35, 0.70 - (dist / 12.0))

            # 4. Synthesize 2D bounding box
            # Projected box scales with distance
            box_scale = max(0.15, min(0.60, 3.0 / max(1.0, dist)))
            bbox = BoundingBox2D(
                ymin=max(0.1, 0.5 - box_scale * 0.5),
                xmin=max(0.1, 0.5 - box_scale * 0.8),
                ymax=min(0.9, 0.5 + box_scale * 0.5),
                xmax=min(0.9, 0.5 + box_scale * 0.8),
            )

            # 5. Candidate plate visibility based on distance
            if dist <= 5.0:
                plate_candidate = vehicle.plate_number
            elif dist <= 9.0:
                plate_candidate = vehicle.plate_number[:3] + "-???"
            else:
                plate_candidate = None

            # 6. Calculate target probability
            p_prior = 0.5
            if env is not None:
                p_prior = env.get_zone_prior(vx, vy)

            target_prob = min(
                0.99,
                max(0.05, 0.30 * p_prior + 0.35 * vehicle_confidence + 0.35 * app_conf)
            )

            event = VehicleDetectionEvent(
                robot_id=robot_id,
                timestamp=timestamp,
                vehicle_class=vehicle.vehicle_class,
                confidence=round(vehicle_confidence, 3),
                bounding_box=bbox,
                estimated_global_position=(round(vx, 2), round(vy, 2)),
                image_reference=f"frame_{robot_id}_{int(timestamp * 1000)}.jpg",
                detected_vehicle_id=vehicle.target_id,
                distance=round(dist, 2),
                plate_candidate=plate_candidate,
                target_probability=round(target_prob, 3),
                observed_color=vehicle.color,
                appearance_confidence=round(app_conf, 3),
                plate_detection_confidence=round(plate_det_conf, 3),
            )
            detected.append((event, vehicle))

        return detected
