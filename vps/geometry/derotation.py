"""Inter-frame gyro de-rotation engine using infinite homography.

Removes apparent optical flow caused by drone attitude changes (roll, pitch, yaw)
between frames using integrated angular rates or attitude deltas.

Mathematical foundation:
    Given relative rotation R_k2c from keyframe (k) to current frame (c):
        H_rot = K * R_k2c * K^-1
    For keyframe pixel p_k:
        p_k_derot = H_rot * p_k
    The residual flow between tracked current point p_c and p_k_derot is purely translational:
        delta_p_trans = p_c - p_k_derot
"""

from typing import Tuple, Optional
import numpy as np
import cv2


class GyroDerotationEngine:
    """Computes rotational homographies and compensates optical flow for camera rotation."""

    def __init__(
        self,
        K: np.ndarray,
        R_body_to_cam: Optional[np.ndarray] = None,
    ) -> None:
        """Initialize de-rotation engine.

        Args:
            K: 3x3 camera intrinsic matrix.
            R_body_to_cam: 3x3 extrinsic rotation matrix mapping body vectors to camera frame.
                           Defaults to identity if IMU is aligned with camera frame.
        """
        self.K = np.array(K, dtype=np.float64).reshape((3, 3))
        self.K_inv = np.linalg.inv(self.K)
        if R_body_to_cam is not None:
            self.R_b2c = np.array(R_body_to_cam, dtype=np.float64).reshape((3, 3))
        else:
            self.R_b2c = np.eye(3, dtype=np.float64)

    def integrate_gyro_to_rotation(
        self,
        gyro_samples_rad_s: np.ndarray,
        timestamps_s: np.ndarray,
    ) -> np.ndarray:
        """Integrate high-rate gyro angular velocities over time interval into relative rotation R_k2c.

        Args:
            gyro_samples_rad_s: Array of shape (N, 3) containing [wx, wy, wz] in body frame (rad/s).
            timestamps_s: Array of shape (N,) containing monotonic timestamps in seconds.

        Returns:
            3x3 relative rotation matrix R_k2c in camera frame.
        """
        gyro = np.atleast_2d(gyro_samples_rad_s).astype(np.float64)
        t = np.asarray(timestamps_s, dtype=np.float64)
        n = len(t)
        if n < 2:
            return np.eye(3, dtype=np.float64)

        R_total_cam = np.eye(3, dtype=np.float64)
        for i in range(n - 1):
            dt = t[i + 1] - t[i]
            if dt <= 0:
                continue
            # Trapezoidal integration for angular step in body frame
            omega_b = 0.5 * (gyro[i] + gyro[i + 1])
            # Transform angular velocity to camera frame: omega_c = R_b2c * omega_b
            omega_c = self.R_b2c @ omega_b
            angle_axis = omega_c * dt
            angle = float(np.linalg.norm(angle_axis))
            if angle > 1e-8:
                dR, _ = cv2.Rodrigues(angle_axis)
                R_total_cam = dR @ R_total_cam

        return R_total_cam

    def rotation_matrix_from_rpy_delta(
        self,
        delta_roll_rad: float,
        delta_pitch_rad: float,
        delta_yaw_rad: float,
    ) -> np.ndarray:
        """Compute relative rotation matrix from Euler attitude differences (body frame).

        Uses standard Z-Y-X (yaw, pitch, roll) aerospace rotation order.

        Args:
            delta_roll_rad: Change in roll angle (rad).
            delta_pitch_rad: Change in pitch angle (rad).
            delta_yaw_rad: Change in yaw angle (rad).

        Returns:
            3x3 relative rotation matrix in camera frame.
        """
        cr, sr = np.cos(delta_roll_rad), np.sin(delta_roll_rad)
        cp, sp = np.cos(delta_pitch_rad), np.sin(delta_pitch_rad)
        cy, sy = np.cos(delta_yaw_rad), np.sin(delta_yaw_rad)

        # R_b = Rz(yaw) * Ry(pitch) * Rx(roll)
        R_b = np.array([
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp,     cp * sr,                cp * cr]
        ], dtype=np.float64)

        # Convert to camera frame: R_c = R_b2c * R_b * R_b2c^T
        R_c = self.R_b2c @ R_b @ self.R_b2c.T
        return R_c

    def compute_rotational_homography(self, R_cam: np.ndarray) -> np.ndarray:
        """Compute infinite rotational homography H = K * R * K^-1.

        Args:
            R_cam: 3x3 rotation matrix mapping keyframe camera frame to current camera frame.

        Returns:
            3x3 homography matrix H.
        """
        return self.K @ R_cam @ self.K_inv

    def derotate_points(
        self,
        points_keyframe: np.ndarray,
        R_cam: np.ndarray,
    ) -> np.ndarray:
        """Apply rotational homography to keyframe pixel coordinates.

        Predicts where keyframe points would appear in the current frame if the camera
        only rotated without translating.

        Args:
            points_keyframe: Array of shape (N, 2) in keyframe pixel coordinates (u, v).
            R_cam: 3x3 rotation matrix from keyframe to current frame.

        Returns:
            Array of shape (N, 2) of de-rotated pixel coordinates in current frame.
        """
        pts = np.atleast_2d(points_keyframe).astype(np.float64)
        n = pts.shape[0]
        homogeneous = np.column_stack([pts, np.ones(n)])

        H = self.compute_rotational_homography(R_cam)
        transformed = (H @ homogeneous.T).T

        # Normalize by homogeneous scale z
        z = transformed[:, 2:3]
        # Guard against points behind camera (z <= 0)
        z_safe = np.where(np.abs(z) < 1e-8, 1e-8, z)
        derotated = transformed[:, :2] / z_safe
        return derotated

    def compute_translational_flow(
        self,
        pts_keyframe: np.ndarray,
        pts_current: np.ndarray,
        R_cam: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute purely translational optical flow vectors by de-rotating keyframe points.

        Args:
            pts_keyframe: Tracked pixel positions in keyframe (N, 2).
            pts_current: Tracked pixel positions in current frame (N, 2).
            R_cam: 3x3 rotation matrix from keyframe to current frame.

        Returns:
            Tuple of:
                pts_key_derot: De-rotated keyframe positions (N, 2).
                flow_trans: Pure translational displacement vectors delta_p_trans = pts_curr - pts_key_derot (N, 2).
        """
        pts_k = np.atleast_2d(pts_keyframe).astype(np.float64)
        pts_c = np.atleast_2d(pts_current).astype(np.float64)
        if pts_k.shape != pts_c.shape:
            raise ValueError(f"Shape mismatch: pts_keyframe {pts_k.shape} vs pts_current {pts_c.shape}")

        pts_k_derot = self.derotate_points(pts_k, R_cam)
        flow_trans = pts_c - pts_k_derot
        return pts_k_derot, flow_trans
