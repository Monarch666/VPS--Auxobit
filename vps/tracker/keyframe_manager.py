"""Adaptive keyframe manager for vision-based AGL height estimation.

Maintains keyframe history and triggers height estimation when accumulated
ground displacement guarantees >= 10 px de-rotated optical flow at the current
altitude.

Policy:
    1. Keyframe creation: stores grayscale image, Shi-Tomasi corners, timestamp,
       drone 3D position, and attitude.
    2. Baseline check: computes ground displacement D = ||pos - pos_keyframe||.
       Predicts expected flow: delta_p = fx * D / H_prior.
       Triggers height estimation when delta_p >= min_keyframe_displacement_px (default 10.0 px).
    3. Keyframe replenishment: promotes current frame to keyframe when:
       - Height estimation completes successfully.
       - Tracked inlier count drops below min_features_threshold.
       - Maximum keyframe age (e.g. 2.0 s) is exceeded.
    4. Hover guard: if horizontal ground speed < hover_speed_threshold (0.5 m/s),
       baseline accumulation is flagged as stagnant/hover.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Tuple
import time
import numpy as np
import cv2

from vps.config import TrackerConfig


@dataclass
class Keyframe:
    """Stored keyframe state."""
    image_gray: np.ndarray
    features: np.ndarray  # (N, 2)
    timestamp_s: float
    pos_ned_m: np.ndarray  # shape (3,)
    attitude_rpy_rad: np.ndarray  # shape (3,)
    prior_altitude_m: float
    frame_idx: int


@dataclass
class KeyframeStatus:
    """Evaluation status of current frame relative to active keyframe."""
    should_estimate_height: bool
    should_reset_keyframe: bool
    ground_displacement_m: float
    predicted_flow_px: float
    keyframe_age_s: float
    is_hovering: bool
    reason: str


class KeyframeManager:
    """Manages adaptive keyframing based on ground displacement and altitude."""

    def __init__(
        self,
        fx: float,
        config: Optional[TrackerConfig] = None,
        hover_speed_thresh_mps: float = 0.5,
        max_keyframe_age_s: float = 2.0,
        default_initial_altitude_m: float = 30.0,
    ) -> None:
        """Initialize keyframe manager.

        Args:
            fx: Camera horizontal focal length in pixels.
            config: Tracker configuration.
            hover_speed_thresh_mps: Minimum horizontal speed to consider drone in motion.
            max_keyframe_age_s: Maximum duration to hold a keyframe before forcing a reset.
            default_initial_altitude_m: Altitude prior before first valid height estimate.
        """
        self.fx = float(fx)
        self.cfg = config if config is not None else TrackerConfig()
        self.hover_thresh = float(hover_speed_thresh_mps)
        self.max_age_s = float(max_keyframe_age_s)
        self.current_alt_prior = float(default_initial_altitude_m)

        self.active_keyframe: Optional[Keyframe] = None
        self._keyframe_counter = 0

    def update_altitude_prior(self, altitude_m: float) -> None:
        """Update current estimated altitude prior."""
        if altitude_m > 0.1:
            self.current_alt_prior = float(altitude_m)

    def initialize_keyframe(
        self,
        image_gray: np.ndarray,
        features: np.ndarray,
        timestamp_s: float,
        pos_ned_m: np.ndarray,
        attitude_rpy_rad: np.ndarray,
        frame_idx: int,
    ) -> Keyframe:
        """Create and set a new active keyframe.

        Args:
            image_gray: Grayscale image frame.
            features: Detected Shi-Tomasi feature points (N, 2).
            timestamp_s: Monotonic frame exposure timestamp.
            pos_ned_m: 3D position vector in Local NED (meters).
            attitude_rpy_rad: 3-axis attitude in radians [roll, pitch, yaw].
            frame_idx: Frame index counter.

        Returns:
            The created Keyframe object.
        """
        self.active_keyframe = Keyframe(
            image_gray=image_gray.copy(),
            features=np.asarray(features, dtype=np.float32).copy(),
            timestamp_s=float(timestamp_s),
            pos_ned_m=np.asarray(pos_ned_m, dtype=np.float64).copy(),
            attitude_rpy_rad=np.asarray(attitude_rpy_rad, dtype=np.float64).copy(),
            prior_altitude_m=self.current_alt_prior,
            frame_idx=frame_idx,
        )
        self._keyframe_counter += 1
        return self.active_keyframe

    def check_keyframe_status(
        self,
        current_timestamp_s: float,
        current_pos_ned_m: np.ndarray,
        current_vel_ned_mps: np.ndarray,
        num_tracked_features: int,
    ) -> KeyframeStatus:
        """Evaluate whether current frame should trigger height calculation or reset keyframe.

        Args:
            current_timestamp_s: Current frame timestamp (s).
            current_pos_ned_m: Current drone NED position (m).
            current_vel_ned_mps: Current drone NED velocity (m/s).
            num_tracked_features: Count of active inlier features tracked from keyframe.

        Returns:
            KeyframeStatus with action flags and metrics.
        """
        if self.active_keyframe is None:
            return KeyframeStatus(
                should_estimate_height=False,
                should_reset_keyframe=True,
                ground_displacement_m=0.0,
                predicted_flow_px=0.0,
                keyframe_age_s=0.0,
                is_hovering=False,
                reason="No active keyframe"
            )

        # 1. Compute ground displacement
        kf = self.active_keyframe
        delta_pos = current_pos_ned_m - kf.pos_ned_m
        ground_disp = float(np.hypot(delta_pos[0], delta_pos[1]))
        age_s = float(current_timestamp_s - kf.timestamp_s)

        # 2. Check speed / hover condition
        horiz_speed = float(np.hypot(current_vel_ned_mps[0], current_vel_ned_mps[1]))
        is_hover = horiz_speed < self.hover_thresh

        # 3. Predict expected pixel displacement at current altitude
        pred_flow_px = (self.fx * ground_disp) / max(self.current_alt_prior, 1.0)

        # 4. Check feature loss
        if num_tracked_features < self.cfg.min_features_threshold:
            return KeyframeStatus(
                should_estimate_height=False,
                should_reset_keyframe=True,
                ground_displacement_m=ground_disp,
                predicted_flow_px=pred_flow_px,
                keyframe_age_s=age_s,
                is_hovering=is_hover,
                reason=f"Feature attrition: {num_tracked_features} < {self.cfg.min_features_threshold}"
            )

        # 5. Check baseline displacement threshold (>= 10 px)
        if pred_flow_px >= self.cfg.min_keyframe_displacement_px:
            return KeyframeStatus(
                should_estimate_height=True,
                should_reset_keyframe=True,
                ground_displacement_m=ground_disp,
                predicted_flow_px=pred_flow_px,
                keyframe_age_s=age_s,
                is_hovering=is_hover,
                reason=f"Target baseline reached ({pred_flow_px:.1f} px >= {self.cfg.min_keyframe_displacement_px:.1f} px)"
            )

        # 6. Check timeout / maximum age
        if age_s >= self.max_age_s:
            return KeyframeStatus(
                should_estimate_height=False,
                should_reset_keyframe=True,
                ground_displacement_m=ground_disp,
                predicted_flow_px=pred_flow_px,
                keyframe_age_s=age_s,
                is_hovering=is_hover,
                reason=f"Keyframe age exceeded ({age_s:.2f} s >= {self.max_age_s:.2f} s)"
            )

        # Baseline still accumulating
        return KeyframeStatus(
            should_estimate_height=False,
            should_reset_keyframe=False,
            ground_displacement_m=ground_disp,
            predicted_flow_px=pred_flow_px,
            keyframe_age_s=age_s,
            is_hovering=is_hover,
            reason="Accumulating baseline"
        )
