"""2-State Kalman Filter for terrain-relative height fusion.

State vector:
    x = [h_AGL, z_terrain]^T
Baro-referenced altitude:
    a_baro = h_AGL + z_terrain

Measurement Models:
    1. Barometer:  z_baro = [1, 1] * x + v_baro
    2. Vision:     z_vis  = [1, 0] * x + v_vis (R_vis dynamically scaled by inverse confidence)

Fallback & Hover Mode:
    When visual updates are unavailable (stationary hover, low texture, or high angular rates):
    - Vision update is gated.
    - Terrain elevation z_terrain is held constant.
    - Terrain covariance is inflated to model drift.
    - Height propagates as h = a_baro - z_terrain with growing covariance and decaying confidence.
"""

from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np

from vps.config import FusionConfig


@dataclass
class KalmanStateEstimate:
    """Estimated state vector and covariance."""
    height_agl_m: float
    terrain_elevation_m: float
    var_height_m2: float
    var_terrain_m2: float
    confidence: float
    is_vision_active: bool
    mode_status: str


class TerrainRelativeKalmanFilter:
    """2-state EKF fusing barometric altimetry and metric vision height."""

    def __init__(
        self,
        config: Optional[FusionConfig] = None,
        initial_height_m: float = 30.0,
        initial_terrain_m: float = 0.0,
        baseline_ground_offset_m: float = 0.06,
    ) -> None:
        """Initialize filter state and covariance matrices."""
        self.cfg = config if config is not None else FusionConfig()
        self.baseline_offset = float(baseline_ground_offset_m)

        # State vector: [h_agl, z_terrain]
        self.x = np.array([float(initial_height_m), float(initial_terrain_m)], dtype=np.float64)

        # Error Covariance Matrix P
        self.P = np.array([
            [1.0, 0.0],
            [0.0, 1.0]
        ], dtype=np.float64)

        # Process Noise base parameters
        self.q_h = self.cfg.q_height
        self.q_terrain_base = self.cfg.q_terrain
        self.r_baro = self.cfg.r_baro
        self.r_vis_base = self.cfg.r_vision_base

        self._confidence = 0.5
        self._last_timestamp: Optional[float] = None
        self._is_vision_active = False
        self._mode_status = "INITIALIZING"

    def predict(
        self,
        dt_s: float,
        vel_d_mps: float = 0.0,
        is_hover: bool = False,
    ) -> None:
        """Propagate state forward using down-velocity kinematics.

        Args:
            dt_s: Time step in seconds.
            vel_d_mps: Vertical velocity in Local NED frame (positive down).
            is_hover: If True, drone is hovering and terrain covariance is inflated.
        """
        if dt_s <= 0:
            return

        # Kinematic model: dot(h) = -vel_d, dot(z_terrain) = 0
        climb_rate = -float(vel_d_mps)
        self.x[0] += climb_rate * dt_s

        # Process noise covariance Q
        # When hovering without vision, inflate terrain process noise to reflect drift
        q_terr = self.q_terrain_base * (10.0 if is_hover else 1.0)
        Q = np.array([
            [self.q_h * dt_s, 0.0],
            [0.0, q_terr * dt_s]
        ], dtype=np.float64)

        # State transition F = Identity(2)
        self.P += Q

        # Decay confidence during prediction step if vision inactive
        if not self._is_vision_active:
            self._confidence = float(max(0.1, self._confidence * np.exp(-0.15 * dt_s)))

    def update_barometer(self, baro_alt_m: float) -> None:
        """Incorporate takeoff-relative barometric altitude measurement.

        Model: a_baro = h_AGL + z_terrain
        """
        z = float(baro_alt_m)
        H = np.array([[1.0, 1.0]], dtype=np.float64)

        # Innovation: nu = z - H * x
        pred_z = float(self.x[0] + self.x[1])
        nu = z - pred_z

        # Innovation covariance: S = H * P * H^T + R
        S = float(self.P[0, 0] + self.P[0, 1] + self.P[1, 0] + self.P[1, 1] + self.r_baro)

        # Kalman Gain: K = P * H^T / S
        K = np.array([
            [(self.P[0, 0] + self.P[0, 1]) / S],
            [(self.P[1, 0] + self.P[1, 1]) / S]
        ], dtype=np.float64)

        # State update
        self.x += (K * nu).ravel()

        # Covariance update: P = (I - K * H) * P
        I_KH = np.eye(2) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K * self.r_baro @ K.T

    def update_vision_height(
        self,
        vision_height_m: float,
        confidence: float,
        measurement_type: str = "FLOW",
    ) -> bool:
        """Incorporate vision height measurement (flow or expansion).

        Model: z_vis = h_AGL

        Args:
            vision_height_m: Metric height from optical flow or scale expansion.
            confidence: Normalized confidence score [0.0, 1.0].
            measurement_type: "FLOW" or "EXPANSION".

        Returns:
            True if measurement passed innovation gating and was applied; False otherwise.
        """
        if confidence < self.cfg.min_confidence:
            self._is_vision_active = False
            self._mode_status = "VISION_REJECTED_LOW_CONFIDENCE"
            return False

        z = float(vision_height_m)
        H = np.array([[1.0, 0.0]], dtype=np.float64)

        # Dynamic measurement variance inversely scaled by confidence squared
        # R_vis = R_base / (confidence^2)
        conf_clamped = max(0.05, float(confidence))
        R_vis = self.r_vis_base / (conf_clamped * conf_clamped)

        # Innovation: nu = z - h_agl
        nu = z - self.x[0]
        S = float(self.P[0, 0] + R_vis)

        # Chi-Square Innovation Gating (threshold = 9.21 for 99% 1-DOF gate)
        mahalanobis_sq = (nu * nu) / S
        if mahalanobis_sq > 9.21:
            self._is_vision_active = False
            self._mode_status = f"VISION_GATED_OUTLIER (d2={mahalanobis_sq:.1f})"
            return False

        # Kalman Gain: K = P * H^T / S
        K = np.array([
            [self.P[0, 0] / S],
            [self.P[1, 0] / S]
        ], dtype=np.float64)

        # State update
        self.x += (K * nu).ravel()

        # Covariance update (Joseph form for numerical symmetry and positive-definiteness)
        I_KH = np.eye(2) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K * R_vis @ K.T

        self._is_vision_active = True
        self._confidence = float(np.clip(confidence, 0.0, 1.0))
        self._mode_status = f"FUSION_ACTIVE_{measurement_type}"
        return True

    def get_estimate(self) -> KalmanStateEstimate:
        """Get current filtered AGL height with ground baseline offset applied."""
        raw_h = float(self.x[0])
        # Subtract physical baseline offset (e.g. 15 cm landing gear clearance)
        corrected_agl = max(0.0, raw_h - self.baseline_offset)

        return KalmanStateEstimate(
            height_agl_m=corrected_agl,
            terrain_elevation_m=float(self.x[1]),
            var_height_m2=float(self.P[0, 0]),
            var_terrain_m2=float(self.P[1, 1]),
            confidence=self._confidence,
            is_vision_active=self._is_vision_active,
            mode_status=self._mode_status,
        )

