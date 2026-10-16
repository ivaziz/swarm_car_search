"""
Vehicle Appearance Verification Engine.
Evaluates color distribution, vehicle geometry/aspect ratio, and class signature
against target stolen vehicle profiles using calibrated feature embeddings.
"""

from typing import Dict, Optional, Tuple
import math
from interfaces.perception_interface import TargetAppearanceProfile


COLOR_RGB_MAP: Dict[str, Tuple[int, int, int]] = {
    "dark_gray": (55, 55, 60),
    "gray": (128, 128, 128),
    "black": (20, 20, 20),
    "white": (240, 240, 240),
    "silver": (192, 192, 192),
    "red": (210, 30, 30),
    "blue": (30, 80, 200),
    "yellow": (230, 210, 30),
    "green": (30, 160, 50),
}


CLASS_ASPECT_RATIOS: Dict[str, float] = {
    "sedan": 2.2,
    "suv": 1.8,
    "truck": 2.8,
    "van": 1.9,
    "compact": 1.7,
}


class VehicleAppearanceVerifier:
    """
    Evaluates visual appearance concordance between observed vehicle detection
    and target vehicle profile (e.g. Dark Gray Sedan).
    Computes calibrated color similarity, geometry/aspect-ratio confidence, and overall score.
    """

    def __init__(
        self,
        default_target_profile: Optional[TargetAppearanceProfile] = None,
        color_weight: float = 0.55,
        shape_weight: float = 0.25,
        class_weight: float = 0.20,
    ) -> None:
        self.target_profile = default_target_profile or TargetAppearanceProfile()
        self.color_weight = color_weight
        self.shape_weight = shape_weight
        self.class_weight = class_weight

    def compute_color_similarity(
        self,
        observed_color: str,
        target_color: Optional[str] = None,
    ) -> float:
        """
        Computes normalized color similarity in [0.0, 1.0] using Euclidean RGB distance.
        """
        target = target_color or self.target_profile.color
        obs_key = observed_color.lower().strip().replace(" ", "_")
        tgt_key = target.lower().strip().replace(" ", "_")

        if obs_key == tgt_key:
            return 1.0

        rgb_obs = COLOR_RGB_MAP.get(obs_key, (120, 120, 120))
        rgb_tgt = COLOR_RGB_MAP.get(tgt_key, (55, 55, 60))

        # Euclidean distance normalized by max possible RGB distance sqrt(3 * 255^2) = 441.67
        dist = math.sqrt(
            (rgb_obs[0] - rgb_tgt[0]) ** 2
            + (rgb_obs[1] - rgb_tgt[1]) ** 2
            + (rgb_obs[2] - rgb_tgt[2]) ** 2
        )
        norm_dist = dist / 441.673
        similarity = max(0.0, 1.0 - norm_dist)
        # Apply sigmoid/exponential sharpening for distinct colors
        return round(similarity ** 1.5, 3)

    def compute_shape_similarity(
        self,
        observed_aspect_ratio: float,
        target_aspect_ratio: Optional[float] = None,
    ) -> float:
        """
        Computes aspect ratio similarity in [0.0, 1.0].
        """
        target_ar = target_aspect_ratio or self.target_profile.aspect_ratio
        if observed_aspect_ratio <= 0.0 or target_ar <= 0.0:
            return 0.5
        ratio_diff = abs(observed_aspect_ratio - target_ar) / target_ar
        similarity = max(0.0, 1.0 - ratio_diff)
        return round(similarity, 3)

    def verify_appearance(
        self,
        observed_class: str,
        observed_color: str,
        observed_aspect_ratio: float = 2.2,
        target_profile: Optional[TargetAppearanceProfile] = None,
    ) -> Tuple[float, Dict[str, float]]:
        """
        Evaluates overall visual appearance match score.
        Returns:
            (appearance_confidence, breakdown_dict)
        """
        tgt = target_profile or self.target_profile
        
        # 1. Class match
        class_sim = 1.0 if observed_class.lower() == tgt.vehicle_class.lower() else 0.20

        # 2. Color similarity
        color_sim = self.compute_color_similarity(observed_color, tgt.color)

        # 3. Shape / aspect ratio similarity
        shape_sim = self.compute_shape_similarity(observed_aspect_ratio, tgt.aspect_ratio)

        # Composite appearance confidence
        score = (
            self.color_weight * color_sim
            + self.shape_weight * shape_sim
            + self.class_weight * class_sim
        )
        score = max(0.0, min(1.0, score))

        breakdown = {
            "class_match": round(class_sim, 3),
            "color_similarity": round(color_sim, 3),
            "shape_similarity": round(shape_sim, 3),
            "composite_appearance": round(score, 3),
        }

        return round(score, 3), breakdown
