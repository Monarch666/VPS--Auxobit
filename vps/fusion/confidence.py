"""Confidence and health evaluation for vision height estimation.

Computes a multi-factor confidence score [0.0, 1.0] based on:
1. Inlier feature count (penalizes < 15 features)
2. Inlier ratio (penalizes high RANSAC rejection rates)
3. Median optical flow displacement (full credit >= 10 px)
4. Horizontal flight speed (detects hover / stationary flight < 0.5 m/s)
5. Angular disturbance rate (penalizes excessive rotational rates > 0.5 rad/s)
6. Image texture variance (detects featureless fog/water/low-contrast patches)
"""

from dataclasses import dataclass
from typing import Optional, Dict, Any
import numpy as np


@dataclass
class ConfidenceFactors:
    """Individual breakdown of confidence sub-metrics."""
    total_confidence: float
    c_inlier_count: float
    c_inlier_ratio: float
    c_flow_magnitude: float
    c_ground_speed: float
    c_angular_rate: float
    c_texture_contrast: float
    is_acceptable: bool
    status_code: str


class ConfidenceEvaluator:
    """Evaluates multi-factor health and confidence metrics for vision height updates."""

    def __init__(
        self,
        min_features: int = 15,
        target_features: int = 150,
        min_inlier_ratio: float = 0.40,
        target_flow_px: float = 10.0,
        hover_speed_thresh_mps: float = 0.5,
        target_speed_mps: float = 3.0,
        max_angular_rate_rad_s: float = 0.60,  # ~35 deg/s
        min_texture_std: float = 6.0,
        min_overall_confidence: float = 0.30,
    ) -> None:
        """Initialize confidence evaluator with operational thresholds."""
        self.min_features = min_features
        self.target_features = target_features
        self.min_inlier_ratio = min_inlier_ratio
        self.target_flow_px = target_flow_px
        self.hover_thresh = hover_speed_thresh_mps
        self.target_speed = target_speed_mps
        self.max_ang_rate = max_angular_rate_rad_s
        self.min_texture_std = min_texture_std
        self.min_conf = min_overall_confidence

    def evaluate(
        self,
        num_inliers: int,
        inlier_ratio: float,
        median_flow_px: float,
        ground_speed_mps: float,
        angular_rate_rad_s: float,
        image_gray: Optional[np.ndarray] = None,
    ) -> ConfidenceFactors:
        """Compute consolidated confidence score and sub-factors.

        Args:
            num_inliers: Number of geometrically verified feature points.
            inlier_ratio: Fraction of tracked points that are inliers.
            median_flow_px: Median optical flow displacement in pixels.
            ground_speed_mps: Horizontal speed (m/s).
            angular_rate_rad_s: Total angular rate norm ||omega|| (rad/s).
            image_gray: Optional grayscale image to evaluate texture contrast.

        Returns:
            ConfidenceFactors with overall score and sub-factor metrics.
        """
        # 1. Feature count factor [0, 1]
        if num_inliers < self.min_features:
            c_count = 0.0
        else:
            c_count = float(np.clip((num_inliers - self.min_features) / (self.target_features - self.min_features), 0.0, 1.0))

        # 2. Inlier ratio factor [0, 1]
        c_ratio = float(np.clip((inlier_ratio - self.min_inlier_ratio) / (0.90 - self.min_inlier_ratio), 0.0, 1.0))

        # 3. Flow magnitude factor [0, 1] (at least 10 px for full score)
        c_flow = float(np.clip(median_flow_px / self.target_flow_px, 0.0, 1.0))

        # 4. Ground speed factor [0, 1] (0 in hover < 0.5 m/s)
        if ground_speed_mps < self.hover_thresh:
            c_speed = 0.0
        else:
            c_speed = float(np.clip((ground_speed_mps - self.hover_thresh) / (self.target_speed - self.hover_thresh), 0.0, 1.0))

        # 5. Angular rate factor [0, 1] (penalize extreme rotational motion)
        c_rot = float(np.clip(1.0 - (angular_rate_rad_s / self.max_ang_rate), 0.0, 1.0))

        # 6. Texture contrast factor [0, 1]
        if image_gray is not None:
            sample_gray = image_gray[::4, ::4] if image_gray.size > 200_000 else image_gray
            img_std = float(np.std(sample_gray))
            c_tex = float(np.clip((img_std - self.min_texture_std) / 20.0, 0.0, 1.0))
        else:
            c_tex = 1.0

        # Weighted Consolidation
        # Weights: Flow (0.25), Inlier Ratio (0.20), Count (0.20), Speed (0.15), Rotation (0.10), Texture (0.10)
        overall = (
            0.25 * c_flow +
            0.20 * c_ratio +
            0.20 * c_count +
            0.15 * c_speed +
            0.10 * c_rot +
            0.10 * c_tex
        )

        # Primary reason detection
        if c_tex < 0.1:
            status = "TEXTURELESS_SURFACE"
        elif c_speed == 0.0:
            status = "HOVER_DEGENERATE"
        elif c_count == 0.0:
            status = "LOW_FEATURE_COUNT"
        elif c_rot < 0.2:
            status = "HIGH_ANGULAR_RATES"
        elif overall >= self.min_conf:
            status = "HIGH_CONFIDENCE"
        else:
            status = "LOW_CONFIDENCE"

        is_acceptable = (overall >= self.min_conf) and (c_speed > 0.0) and (c_count > 0.0)

        return ConfidenceFactors(
            total_confidence=float(np.clip(overall, 0.0, 1.0)),
            c_inlier_count=c_count,
            c_inlier_ratio=c_ratio,
            c_flow_magnitude=c_flow,
            c_ground_speed=c_speed,
            c_angular_rate=c_rot,
            c_texture_contrast=c_tex,
            is_acceptable=is_acceptable,
            status_code=status,
        )
