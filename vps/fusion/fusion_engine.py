"""Consolidated State Estimation and Sensor Fusion Engine for VPS.

Combines:
- 2-state terrain-relative Kalman Filter
- De-rotated keyframe optical flow height
- Vertical scale expansion during climb/descent
- Barometric altimetry fusion
- Multi-factor confidence evaluation and fallback controller
"""

from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple
import time
import numpy as np
import cv2

from vps.config import VPSConfig
from vps.tracker.flow_pipeline import VisionHeightPipeline, FlowPipelineResult
from vps.geometry.scale_expansion import ScaleFromExpansionEngine, ScaleExpansionResult
from .confidence import ConfidenceEvaluator, ConfidenceFactors
from .kalman_filter import TerrainRelativeKalmanFilter, KalmanStateEstimate


@dataclass
class FusionOutput:
    """Consolidated estimator output delivered to MAVLink sender and logger."""
    timestamp_s: float
    frame_idx: int
    height_agl_m: float
    terrain_elevation_m: float
    confidence: float
    variance_m2: float
    is_vision_fused: bool
    is_hovering: bool
    mode_status: str
    processing_time_ms: float


class VPSFusionEngine:
    """Master sensor fusion coordinator for VPS."""

    def __init__(
        self,
        config: Optional[VPSConfig] = None,
        initial_height_m: float = 30.0,
        initial_terrain_m: float = 0.0,
    ) -> None:
        """Initialize all sub-modules of the fusion engine."""
        self.cfg = config if config is not None else VPSConfig()

        self.pipeline = VisionHeightPipeline(config=self.cfg)
        self.scale_engine = ScaleFromExpansionEngine()
        self.confidence_eval = ConfidenceEvaluator(
            min_features=self.cfg.tracker.min_features_threshold,
            min_overall_confidence=self.cfg.fusion.min_confidence,
            hover_speed_thresh_mps=self.cfg.fusion.hover_speed_threshold_mps,
        )
        self.filter = TerrainRelativeKalmanFilter(
            config=self.cfg.fusion,
            initial_height_m=initial_height_m,
            initial_terrain_m=initial_terrain_m,
            baseline_ground_offset_m=self.cfg.camera.baseline_ground_offset_m,
        )

        self._last_timestamp_s: Optional[float] = None
        self._last_baro_alt_m: Optional[float] = None
        self._last_keyframe_pts: Optional[np.ndarray] = None

    def process_telemetry_and_frame(
        self,
        frame: np.ndarray,
        timestamp_s: float,
        pos_ned_m: np.ndarray,
        vel_ned_mps: np.ndarray,
        attitude_rpy_rad: np.ndarray,
        gyro_rates_rad_s: np.ndarray,
        baro_alt_m: float,
        frame_idx: int,
    ) -> FusionOutput:
        """Process synchronous camera frame and telemetry packet through the fusion filter.

        Args:
            frame: Camera frame image (BGR or grayscale).
            timestamp_s: Monotonic frame exposure timestamp (seconds).
            pos_ned_m: Local NED position (meters) [X, Y, Z].
            vel_ned_mps: Local NED velocity (m/s) [Vn, Ve, Vd].
            attitude_rpy_rad: Body Euler angles [roll, pitch, yaw] in radians.
            gyro_rates_rad_s: Body angular rates [p, q, r] in rad/s.
            baro_alt_m: Takeoff-relative barometric altitude (meters).
            frame_idx: Monotonic frame index.

        Returns:
            FusionOutput dataclass.
        """
        t_start = time.perf_counter()

        # Step 1: Compute time step dt
        if self._last_timestamp_s is not None:
            dt_s = max(0.001, timestamp_s - self._last_timestamp_s)
        else:
            dt_s = 1.0 / self.cfg.camera.fps
        self._last_timestamp_s = timestamp_s

        # Step 2: Check hover condition
        horiz_speed = float(np.hypot(vel_ned_mps[0], vel_ned_mps[1]))
        is_hover = horiz_speed < self.cfg.fusion.hover_speed_threshold_mps

        # Step 3: Kalman Filter Predict Step
        self.filter.predict(dt_s=dt_s, vel_d_mps=float(vel_ned_mps[2]), is_hover=is_hover)

        # Step 4: Kalman Filter Barometer Update
        self.filter.update_barometer(baro_alt_m=baro_alt_m)

        # Step 5: Vision Processing (Optical Flow)
        flow_res = self.pipeline.process_frame(
            frame=frame,
            timestamp_s=timestamp_s,
            pos_ned_m=pos_ned_m,
            vel_ned_mps=vel_ned_mps,
            attitude_rpy_rad=attitude_rpy_rad,
            frame_idx=frame_idx,
        )

        # Step 6: Multi-Factor Confidence Evaluation
        ang_rate_norm = float(np.linalg.norm(gyro_rates_rad_s))
        inlier_ratio = float(flow_res.num_inliers / max(1, self.cfg.tracker.max_features))
        gray_eval = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame

        conf_factors = self.confidence_eval.evaluate(
            num_inliers=flow_res.num_inliers,
            inlier_ratio=inlier_ratio,
            median_flow_px=flow_res.median_flow_px,
            ground_speed_mps=horiz_speed,
            angular_rate_rad_s=ang_rate_norm,
            image_gray=gray_eval,
        )

        # Step 7: Kalman Filter Vision Update (if acceptable)
        vision_fused = False
        if flow_res.is_valid and flow_res.height_agl_m is not None and conf_factors.is_acceptable:
            vision_fused = self.filter.update_vision_height(
                vision_height_m=flow_res.height_agl_m,
                confidence=conf_factors.total_confidence,
                measurement_type="FLOW",
            )
        elif is_hover:
            self.filter._mode_status = "HOVER_TERRAIN_HOLD"
        elif not conf_factors.is_acceptable:
            self.filter._mode_status = f"FALLBACK_{conf_factors.status_code}"

        # Step 8: Get Filtered State Output
        state_est = self.filter.get_estimate()

        t_elapsed = (time.perf_counter() - t_start) * 1000.0

        return FusionOutput(
            timestamp_s=timestamp_s,
            frame_idx=frame_idx,
            height_agl_m=state_est.height_agl_m,
            terrain_elevation_m=state_est.terrain_elevation_m,
            confidence=state_est.confidence,
            variance_m2=state_est.var_height_m2,
            is_vision_fused=vision_fused,
            is_hovering=is_hover,
            mode_status=state_est.mode_status,
            processing_time_ms=t_elapsed,
        )
