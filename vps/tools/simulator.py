"""Synthetic aerial pinhole camera simulator for UAV height estimation.

Renders realistic camera frames from an aerial orthophoto with:
- 6-DOF drone kinematics (position, attitude, velocity, angular rates)
- Exact pinhole camera geometry and projective homography
- Environmental degradations:
  - Directional motion blur (exposure time and velocity vector)
  - Atmospheric haze (Koschmieder scattering model with altitude-dependent contrast attenuation)
  - Sensor readout and shot noise
- Synchronized synthetic telemetry generation:
  - High-rate gyro with bias and noise
  - Barometric altitude with drift
  - NED ground velocity
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, Optional, Dict, Any, List
import numpy as np
import cv2

from vps.config import CameraConfig
from vps.geometry.pinhole import PinholeModel
from vps.tools.logger import TelemetryRecord


@dataclass
class DroneState:
    """6-DOF kinematic state of the drone."""
    timestamp_s: float
    # Position in Local NED frame (meters): X north, Y east, Z down (Z = -altitude)
    pos_ned_m: np.ndarray  # shape (3,)
    # Velocity in Local NED frame (m/s)
    vel_ned_mps: np.ndarray  # shape (3,)
    # Attitude: Euler angles in radians [roll, pitch, yaw]
    attitude_rpy_rad: np.ndarray  # shape (3,)
    # Angular rates in Body frame (rad/s) [p, q, r]
    angular_rates_rad_s: np.ndarray  # shape (3,)
    # Terrain elevation directly beneath drone (m)
    terrain_elevation_m: float = 0.0

    @property
    def altitude_agl_m(self) -> float:
        """Height above ground level (m)."""
        return float(-self.pos_ned_m[2] - self.terrain_elevation_m)


@dataclass
class SimulatorDegradationConfig:
    """Configures camera sensor and atmospheric noise parameters."""
    exposure_time_s: float = 0.002  # 2 ms exposure (1/500 s)
    haze_beta: float = 0.0015  # scattering coefficient (1/m)
    haze_airlight_bgr: Tuple[int, int, int] = (220, 215, 210)  # atmospheric sky light
    sensor_noise_sigma: float = 2.0  # Gaussian noise standard deviation in pixel intensity
    gyro_bias_rad_s: float = 0.005  # static gyro bias
    gyro_noise_sigma: float = 0.002  # gyro white noise std (rad/s)
    baro_noise_sigma: float = 0.25  # baro altitude noise std (meters)
    baro_drift_rate_mps: float = 0.01  # slow barometric thermal drift (m/s)


class DroneCameraSimulator:
    """Simulates downward-facing camera imagery and telemetry from an orthophoto."""

    def __init__(
        self,
        orthophoto: np.ndarray,
        meters_per_pixel: float,
        camera_config: Optional[CameraConfig] = None,
        degradation_config: Optional[SimulatorDegradationConfig] = None,
        origin_ned_m: Tuple[float, float] = (0.0, 0.0),
        R_cam_to_body: Optional[np.ndarray] = None,
    ) -> None:
        """Initialize simulator.

        Args:
            orthophoto: Ground texture image (BGR, shape (H, W, 3)).
            meters_per_pixel: Scale of the orthophoto in meters per pixel.
            camera_config: Intrinsics and resolution of the drone camera.
            degradation_config: Environmental degradation parameters.
            origin_ned_m: NED coordinate of orthophoto top-left corner (meters).
            R_cam_to_body: 3x3 extrinsic rotation matrix mapping camera vectors to body frame.
        """
        self.ortho = orthophoto
        self.ortho_h, self.ortho_w = self.ortho.shape[:2]
        self.m_per_px = float(meters_per_pixel)
        self.cam_cfg = camera_config if camera_config is not None else CameraConfig()
        self.deg_cfg = degradation_config if degradation_config is not None else SimulatorDegradationConfig()
        self.origin_x, self.origin_y = origin_ned_m
        self.pinhole = PinholeModel(self.cam_cfg.K, self.cam_cfg.D, self.cam_cfg.width, self.cam_cfg.height)

        # Standard downward mount:
        # Camera Z_c is optical axis (pointing down || Body Z_b)
        # Camera X_c is right across image (Body Y_b)
        # Camera Y_c is down across image (Body -X_b, so top of image is drone forward)
        if R_cam_to_body is not None:
            self.R_c2b = np.array(R_cam_to_body, dtype=np.float64).reshape((3, 3))
        else:
            self.R_c2b = np.array([
                [0.0, -1.0, 0.0],
                [1.0,  0.0, 0.0],
                [0.0,  0.0, 1.0]
            ], dtype=np.float64)

        self.R_b2c = self.R_c2b.T
        self.rng = np.random.default_rng(101)
        self._gyro_bias = self.rng.normal(0, self.deg_cfg.gyro_bias_rad_s, size=3)

    def _euler_to_rotation_matrix(self, roll: float, pitch: float, yaw: float) -> np.ndarray:
        """Compute body-to-world rotation matrix R_w_b (Z-Y-X sequence)."""
        cr, sr = np.cos(roll), np.sin(roll)
        cp, sp = np.cos(pitch), np.sin(pitch)
        cy, sy = np.cos(yaw), np.sin(yaw)

        # R_w_b = Rz(yaw) * Ry(pitch) * Rx(roll)
        R_w_b = np.array([
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp,     cp * sr,                cp * cr]
        ], dtype=np.float64)
        return R_w_b

    def render_frame(self, state: DroneState) -> np.ndarray:
        """Render a camera frame for a given drone kinematic state.

        Args:
            state: DroneState with position, attitude, and velocity.

        Returns:
            Rendered image frame (BGR, uint8) matching camera resolution.
        """
        alt = state.altitude_agl_m
        if alt <= 0.5:
            raise ValueError(f"Altitude too low for aerial simulation: {alt:.2f} m")

        # 1. Compute World to Camera transformation
        # R_w_b: Body to World
        R_w_b = self._euler_to_rotation_matrix(*state.attitude_rpy_rad)
        # R_w_c = R_w_b * R_b_c -> World to Camera is R_c_w = (R_w_c)^T
        R_w_c = R_w_b @ self.R_c2b
        R_c_w = R_w_c.T

        # Camera optical center in World NED
        # C_w = pos_ned
        C_w = state.pos_ned_m
        t_c = -R_c_w @ C_w

        # 2. Homography mapping Ground Plane (Z=0) to Camera Pixels
        # For ground points X_w = [x_g, y_g, 0]^T:
        # X_c = R_c_w[:, 0]*x_g + R_c_w[:, 1]*y_g + t_c
        # p_img = K * [R_c_w[:, 0], R_c_w[:, 1], t_c] * [x_g, y_g, 1]^T
        H_ground_to_cam = self.cam_cfg.K @ np.column_stack([R_c_w[:, 0], R_c_w[:, 1], t_c])

        # Convert ground metric coordinates to orthophoto pixel coordinates
        # x_g = origin_x + u_ortho * m_per_px
        # y_g = origin_y + v_ortho * m_per_px
        T_ortho_to_ground = np.array([
            [self.m_per_px, 0.0, self.origin_x],
            [0.0, self.m_per_px, self.origin_y],
            [0.0, 0.0, 1.0]
        ], dtype=np.float64)

        # H_ortho_to_cam maps (u_ortho, v_ortho) -> (u_cam, v_cam)
        H_ortho_to_cam = H_ground_to_cam @ T_ortho_to_ground

        # Invert to map camera pixels back to orthophoto: (u_cam, v_cam) -> (u_ortho, v_ortho)
        try:
            H_cam_to_ortho = np.linalg.inv(H_ortho_to_cam)
        except np.linalg.LinAlgError:
            raise RuntimeError("Singular homography matrix encountered in simulation")

        # Warp orthophoto to camera view
        cam_w, cam_h = self.cam_cfg.width, self.cam_cfg.height
        frame = cv2.warpPerspective(
            self.ortho,
            H_cam_to_ortho,
            (cam_w, cam_h),
            flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
            borderMode=cv2.BORDER_REFLECT_101
        )

        # 3. Apply Degradations
        frame = self._apply_degradations(frame, state)
        return frame

    def _apply_degradations(self, frame: np.ndarray, state: DroneState) -> np.ndarray:
        """Apply motion blur, atmospheric haze, and sensor noise."""
        img = frame.copy().astype(np.float32)
        alt = state.altitude_agl_m

        # A. Motion Blur
        # Velocity in camera frame
        vel_body = self._euler_to_rotation_matrix(*state.attitude_rpy_rad).T @ state.vel_ned_mps
        vel_cam = self.R_b2c @ vel_body
        gsd_m_px = alt / self.pinhole.f_mean
        # Pixel displacement during exposure: v_cam * t_exp / gsd
        dx_px = vel_cam[0] * self.deg_cfg.exposure_time_s / gsd_m_px
        dy_px = vel_cam[1] * self.deg_cfg.exposure_time_s / gsd_m_px
        blur_len = np.hypot(dx_px, dy_px)

        if blur_len >= 1.5:
            ksize = int(max(3, min(31, int(round(blur_len)))))
            if ksize % 2 == 0:
                ksize += 1
            kernel = np.zeros((ksize, ksize), dtype=np.float32)
            angle = np.arctan2(dy_px, dx_px)
            # Create directional line kernel
            center = ksize // 2
            cos_a, sin_a = np.cos(angle), np.sin(angle)
            for r in np.linspace(-center, center, ksize * 2):
                x = int(round(center + r * cos_a))
                y = int(round(center + r * sin_a))
                if 0 <= x < ksize and 0 <= y < ksize:
                    kernel[y, x] += 1.0
            kernel_sum = np.sum(kernel)
            if kernel_sum > 0:
                kernel /= kernel_sum
                img = cv2.filter2D(img, -1, kernel)

        # B. Atmospheric Haze (Koschmieder Model)
        # Transmittance t = exp(-beta * alt)
        transmittance = float(np.exp(-self.deg_cfg.haze_beta * alt))
        airlight = np.array(self.deg_cfg.haze_airlight_bgr, dtype=np.float32)
        # I_hazy = I * t + A * (1 - t)
        img = img * transmittance + airlight * (1.0 - transmittance)

        # C. Sensor Noise (Gaussian Readout Noise)
        noise = self.rng.normal(0, self.deg_cfg.sensor_noise_sigma, size=img.shape)
        img += noise

        return np.clip(img, 0, 255).astype(np.uint8)

    def generate_telemetry_record(self, state: DroneState, frame_idx: int) -> TelemetryRecord:
        """Synthesize synchronized flight controller telemetry packet."""
        # Gyro: true angular rates + bias + Gaussian noise
        gyro_noise = self.rng.normal(0, self.deg_cfg.gyro_noise_sigma, size=3)
        gyro_noisy = state.angular_rates_rad_s + self._gyro_bias + gyro_noise

        # Barometer: true altitude + thermal drift + noise
        drift = self.deg_cfg.baro_drift_rate_mps * state.timestamp_s
        baro_noise = float(self.rng.normal(0, self.deg_cfg.baro_noise_sigma))
        baro_alt = state.altitude_agl_m + state.terrain_elevation_m + drift + baro_noise

        return TelemetryRecord(
            timestamp_s=state.timestamp_s,
            frame_idx=frame_idx,
            roll_rad=float(state.attitude_rpy_rad[0]),
            pitch_rad=float(state.attitude_rpy_rad[1]),
            yaw_rad=float(state.attitude_rpy_rad[2]),
            gyro_x_rad_s=float(gyro_noisy[0]),
            gyro_y_rad_s=float(gyro_noisy[1]),
            gyro_z_rad_s=float(gyro_noisy[2]),
            vel_n_mps=float(state.vel_ned_mps[0]),
            vel_e_mps=float(state.vel_ned_mps[1]),
            vel_d_mps=float(state.vel_ned_mps[2]),
            baro_alt_m=baro_alt,
            groundtruth_h_m=state.altitude_agl_m,
        )

