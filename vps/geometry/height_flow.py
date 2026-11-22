"""Height estimation from de-rotated optical flow and metric ground displacement.

Mathematical foundation:
    Given camera intrinsics (fx, fy), metric horizontal ground displacement
    d_ground = [dx, dy]^T (meters) accumulated between keyframe and current frame,
    and de-rotated translational pixel flow delta_p = [du, dv]^T:
        du = fx * dx / H
        dv = fy * dy / H
    Flow magnitude:
        ||delta_p|| = sqrt((fx * dx / H)^2 + (fy * dy / H)^2)
                    = (1 / H) * sqrt((fx * dx)^2 + (fy * dy)^2)
    Therefore, for each valid feature point i:
        H_i = sqrt((fx * dx)^2 + (fy * dy)^2) / ||delta_p_i||

At least 10 px average displacement is required for reliable triangulation.
Directional consistency (cosine similarity) and robust median/RANSAC filter outliers.
"""

from dataclasses import dataclass
from typing import Optional, List
import numpy as np


@dataclass
class FlowHeightResult:
    """Result of height estimation from optical flow."""
    height_m: float
    confidence: float
    inlier_ratio: float
    num_inliers: int
    median_flow_px: float
    is_valid: bool
    status_message: str


class HeightFromFlowEngine:
    """Estimates AGL height from de-rotated feature displacements and metric velocity."""

    def __init__(
        self,
        fx: float,
        fy: float,
        min_displacement_px: float = 10.0,
        min_inlier_ratio: float = 0.4,
        min_inlier_count: int = 8,
        min_ground_displacement_m: float = 0.05,
        min_altitude_m: float = 0.05,
        max_altitude_m: float = 350.0,
        direction_cosine_threshold: float = 0.707,  # cos(45 deg)
    ) -> None:
        """Initialize height-from-flow engine.

        Args:
            fx: Focal length in x (pixels).
            fy: Focal length in y (pixels).
            min_displacement_px: Minimum recommended average flow displacement (pixels).
            min_inlier_ratio: Minimum ratio of consistent features to mark result valid.
            min_inlier_count: Minimum number of inliers required.
            min_ground_displacement_m: Minimum ground distance traveled to estimate height.
            min_altitude_m: Lower altitude sanity bound (m).
            max_altitude_m: Upper altitude sanity bound (m).
            direction_cosine_threshold: Cosine threshold for flow directional agreement.
        """
        self.fx = float(fx)
        self.fy = float(fy)
        self.min_displacement_px = float(min_displacement_px)
        self.min_inlier_ratio = float(min_inlier_ratio)
        self.min_inlier_count = int(min_inlier_count)
        self.min_ground_displacement_m = float(min_ground_displacement_m)
        self.min_altitude_m = float(min_altitude_m)
        self.max_altitude_m = float(max_altitude_m)
        self.direction_cos_thresh = float(direction_cosine_threshold)

    def estimate_height(
        self,
        flow_trans: np.ndarray,
        ground_displacement_cam_m: np.ndarray,
    ) -> FlowHeightResult:
        """Estimate AGL height from translational optical flow and metric ground displacement.

        Args:
            flow_trans: Array of shape (N, 2) containing de-rotated pixel flow [du, dv] per feature.
            ground_displacement_cam_m: Array of shape (2,) or (3,) containing [dx, dy] translation
                                       in camera frame (meters).

        Returns:
            FlowHeightResult containing height estimate, confidence score, and metrics.
        """
        flow = np.atleast_2d(flow_trans).astype(np.float64)
        n_features = flow.shape[0]

        if n_features < self.min_inlier_count:
            return FlowHeightResult(
                height_m=0.0,
                confidence=0.0,
                inlier_ratio=0.0,
                num_inliers=0,
                median_flow_px=0.0,
                is_valid=False,
                status_message=f"Insufficient features: {n_features} < {self.min_inlier_count}"
            )

        dx = float(ground_displacement_cam_m[0])
        dy = float(ground_displacement_cam_m[1])
        ground_dist = np.hypot(dx, dy)

        if ground_dist < self.min_ground_displacement_m:
            return FlowHeightResult(
                height_m=0.0,
                confidence=0.0,
                inlier_ratio=0.0,
                num_inliers=0,
                median_flow_px=0.0,
                is_valid=False,
                status_message=f"Ground displacement too small: {ground_dist:.3f} m < {self.min_ground_displacement_m} m"
            )

        # Expected flow direction vector scaled by focal lengths
        pred_flow_x = self.fx * dx
        pred_flow_y = self.fy * dy
        metric_norm = np.hypot(pred_flow_x, pred_flow_y)

        # Magnitudes of measured pixel flows
        flow_mags = np.hypot(flow[:, 0], flow[:, 1])
        median_flow_px = float(np.median(flow_mags))

        # Check collinearity via absolute cosine similarity: flow must align with translation axis
        dot_products = flow[:, 0] * pred_flow_x + flow[:, 1] * pred_flow_y
        denoms = flow_mags * metric_norm
        # Guard against zero flow magnitudes
        valid_denoms = denoms > 1e-6
        cos_sim = np.zeros(n_features, dtype=np.float64)
        cos_sim[valid_denoms] = np.abs(dot_products[valid_denoms]) / denoms[valid_denoms]

        # Compute point-wise height estimates: H_i = metric_norm / flow_mag_i
        valid_mag = flow_mags > 0.5  # at least 0.5 px flow to avoid division by near-zero
        height_candidates = np.zeros(n_features, dtype=np.float64)
        height_candidates[valid_mag] = metric_norm / flow_mags[valid_mag]

        # Inlier filtering
        inlier_mask = (
            valid_mag &
            (cos_sim >= self.direction_cos_thresh) &
            (height_candidates >= self.min_altitude_m) &
            (height_candidates <= self.max_altitude_m)
        )

        n_inliers = int(np.sum(inlier_mask))
        inlier_ratio = float(n_inliers / n_features)

        if n_inliers < self.min_inlier_count or inlier_ratio < self.min_inlier_ratio:
            return FlowHeightResult(
                height_m=0.0,
                confidence=0.0,
                inlier_ratio=inlier_ratio,
                num_inliers=n_inliers,
                median_flow_px=median_flow_px,
                is_valid=False,
                status_message=(
                    f"Low inliers: {n_inliers}/{n_features} ({inlier_ratio:.1%}), "
                    f"median flow: {median_flow_px:.1f} px"
                )
            )

        inlier_heights = height_candidates[inlier_mask]
        est_height = float(np.median(inlier_heights))

        # Confidence calculation
        # 1. Flow baseline score (1.0 when median flow >= min_displacement_px, linear drop below)
        flow_score = min(1.0, max(0.0, median_flow_px / self.min_displacement_px))
        # 2. Inlier ratio score
        inlier_score = min(1.0, max(0.0, inlier_ratio))
        # 3. Normalized spread (MAD / median)
        mad = float(np.median(np.abs(inlier_heights - est_height)))
        relative_mad = mad / max(est_height, 1e-3)
        spread_score = max(0.0, 1.0 - 2.0 * relative_mad)

        confidence = float(np.clip(flow_score * 0.4 + inlier_score * 0.35 + spread_score * 0.25, 0.0, 1.0))

        return FlowHeightResult(
            height_m=est_height,
            confidence=confidence,
            inlier_ratio=inlier_ratio,
            num_inliers=n_inliers,
            median_flow_px=median_flow_px,
            is_valid=True,
            status_message="Valid flow height estimate"
        )
