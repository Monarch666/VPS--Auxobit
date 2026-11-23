"""15-State Error-State Extended Kalman Filter (ES-EKF) for GNSS-Denied Navigation.

States:
- Position (NED): p in R^3
- Velocity (NED): v in R^3
- Attitude: Unit quaternion q representing Body to NED rotation
- Accelerometer Bias (Body): b_a in R^3
- Gyroscope Bias (Body): b_g in R^3

Total nominal state: 16 parameters.
Error state: delta_x = [delta_p, delta_v, delta_theta, delta_ba, delta_bg]^T in R^15.

Sensors fused:
1. High-rate IMU (100 Hz): Specific force & angular rates strapdown integration
2. Compass / FC Magnetometer (10-50 Hz): Yaw heading correction
3. VIO Metric Velocity (30 Hz): Optical flow horizontal velocity
4. Phase 1 Height / Barometer (15-30 Hz): Vertical altitude constraint
5. Delayed-State Map Matching interface (1-2 Hz, Milestone 9 ready)
"""

from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Dict, Any
import numpy as np


@dataclass
class ESEKFConfig:
    """Configures noise parameters and initial covariances for ES-EKF."""
    # Continuous IMU noise densities
    sigma_accel_noise: float = 0.08  # m/s^2 / sqrt(Hz)
    sigma_gyro_noise: float = 0.005  # rad/s / sqrt(Hz)
    sigma_accel_bias_walk: float = 0.002  # m/s^3 / sqrt(Hz)
    sigma_gyro_bias_walk: float = 0.0001  # rad/s^2 / sqrt(Hz)

    # Measurement noise standard deviations
    sigma_vio_vel_mps: float = 0.35  # VIO velocity noise std (m/s)
    sigma_compass_yaw_rad: float = 0.035  # ~2.0 deg compass noise std
    sigma_height_m: float = 0.20  # Height / baro noise std (m)

    # Innovation chi-square gating thresholds
    gate_vio_vel_chi2: float = 35.0  # 3-DOF chi^2 gating
    gate_compass_chi2: float = 15.0  # 1-DOF chi^2 gating
    gate_height_chi2: float = 15.0   # 1-DOF chi^2 gating

    # Gravity constant in local NED
    gravity_mps2: float = 9.80665


@dataclass
class NavState:
    """Nominal kinematic navigation state."""
    timestamp_s: float
    pos_ned_m: np.ndarray  # shape (3,) [X_n, Y_e, Z_d]
    vel_ned_mps: np.ndarray  # shape (3,) [V_n, V_e, V_d]
    quat_wb: np.ndarray  # shape (4,) [w, x, y, z] (Body to NED)
    accel_bias_mps2: np.ndarray  # shape (3,) [bx, by, bz] in Body frame
    gyro_bias_rad_s: np.ndarray  # shape (3,) [bx, by, bz] in Body frame

    @property
    def roll_pitch_yaw_rad(self) -> np.ndarray:
        """Extract Euler angles [roll, pitch, yaw] in radians."""
        w, x, y, z = self.quat_wb
        # Roll (x-axis rotation)
        sinr_cosp = 2.0 * (w * x + y * z)
        cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
        roll = np.arctan2(sinr_cosp, cosr_cosp)

        # Pitch (y-axis rotation)
        sinp = 2.0 * (w * y - z * x)
        if abs(sinp) >= 1.0:
            pitch = np.sign(sinp) * (np.pi / 2.0)
        else:
            pitch = np.arcsin(sinp)

        # Yaw (z-axis rotation)
        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        yaw = np.arctan2(siny_cosp, cosy_cosp)

        return np.array([roll, pitch, yaw], dtype=np.float64)

    @property
    def rotation_matrix(self) -> np.ndarray:
        """3x3 Rotation matrix mapping vectors from Body to World NED: R_w_b."""
        w, x, y, z = self.quat_wb
        return np.array([
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - w * z),       2.0 * (x * z + w * y)],
            [2.0 * (x * y + w * z),       1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - w * x)],
            [2.0 * (x * z - w * y),       2.0 * (y * z + w * x),       1.0 - 2.0 * (x * x + y * y)]
        ], dtype=np.float64)


class ErrorStateEKF:
    """15-State Error-State Extended Kalman Filter for drone navigation."""

    def __init__(
        self,
        config: Optional[ESEKFConfig] = None,
        initial_pos_ned: Optional[np.ndarray] = None,
        initial_vel_ned: Optional[np.ndarray] = None,
        initial_yaw_rad: float = 0.0,
    ) -> None:
        """Initialize ES-EKF with nominal state and error covariance."""
        self.cfg = config if config is not None else ESEKFConfig()

        p0 = np.array(initial_pos_ned if initial_pos_ned is not None else [0.0, 0.0, 0.0], dtype=np.float64)
        v0 = np.array(initial_vel_ned if initial_vel_ned is not None else [0.0, 0.0, 0.0], dtype=np.float64)

        # Initialize quaternion from initial yaw heading
        cy = np.cos(initial_yaw_rad * 0.5)
        sy = np.sin(initial_yaw_rad * 0.5)
        q0 = np.array([cy, 0.0, 0.0, sy], dtype=np.float64)

        self.state = NavState(
            timestamp_s=0.0,
            pos_ned_m=p0,
            vel_ned_mps=v0,
            quat_wb=q0,
            accel_bias_mps2=np.zeros(3, dtype=np.float64),
            gyro_bias_rad_s=np.zeros(3, dtype=np.float64),
        )

        # 15x15 Error state covariance P
        # Indices: [0:3: pos, 3:6: vel, 6:9: theta, 9:12: b_a, 12:15: b_g]
        self.P = np.zeros((15, 15), dtype=np.float64)
        self.P[0:3, 0:3] = np.eye(3) * (2.0 ** 2)      # 2.0 m pos std
        self.P[3:6, 3:6] = np.eye(3) * (2.0 ** 2)      # 2.0 m/s vel std
        self.P[6:9, 6:9] = np.eye(3) * (0.3 ** 2)      # ~17 deg att std
        self.P[9:12, 9:12] = np.eye(3) * (0.1 ** 2)    # 0.1 m/s^2 accel bias std
        self.P[12:15, 12:15] = np.eye(3) * (0.01 ** 2) # 0.01 rad/s gyro bias std

        self._last_imu_timestamp_s: Optional[float] = None
        self._history_buffer: List[Tuple[float, NavState, np.ndarray]] = []
        self._max_history_len = 150  # 1.5 seconds @ 100 Hz
        self._consecutive_rejected_vio: int = 0

    @staticmethod
    def skew_symmetric(v: np.ndarray) -> np.ndarray:
        """Compute 3x3 skew-symmetric cross-product matrix [v]_x."""
        return np.array([
            [0.0,   -v[2],  v[1]],
            [v[2],   0.0,  -v[0]],
            [-v[1],  v[0],  0.0]
        ], dtype=np.float64)

    @staticmethod
    def quat_mult(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
        """Quaternion product q = q1 (x) q2."""
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2
        return np.array([
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ], dtype=np.float64)

    @staticmethod
    def exp_rot_vector(theta_vec: np.ndarray) -> np.ndarray:
        """Exponential map mapping rotation vector theta in R^3 to unit quaternion."""
        angle = float(np.linalg.norm(theta_vec))
        if angle < 1e-8:
            return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        half_angle = 0.5 * angle
        axis = theta_vec / angle
        return np.array([
            np.cos(half_angle),
            axis[0] * np.sin(half_angle),
            axis[1] * np.sin(half_angle),
            axis[2] * np.sin(half_angle),
        ], dtype=np.float64)

    def predict_imu(
        self,
        accel_meas_mps2: np.ndarray,
        gyro_meas_rad_s: np.ndarray,
        timestamp_s: float,
    ) -> NavState:
        """Propagate nominal kinematics and error covariance using IMU measurements.

        Args:
            accel_meas_mps2: Measured specific force in Body frame [ax, ay, az] (m/s^2).
            gyro_meas_rad_s: Measured angular rate in Body frame [p, q, r] (rad/s).
            timestamp_s: Monotonic measurement timestamp (seconds).

        Returns:
            Propagated NavState.
        """
        if self._last_imu_timestamp_s is not None:
            dt = max(0.0005, min(0.05, timestamp_s - self._last_imu_timestamp_s))
        else:
            dt = 0.01  # default 100 Hz
        self._last_imu_timestamp_s = timestamp_s
        self.state.timestamp_s = timestamp_s

        # 1. Unbiased specific force and angular velocity in Body frame
        f_b = accel_meas_mps2 - self.state.accel_bias_mps2
        omega_b = gyro_meas_rad_s - self.state.gyro_bias_rad_s

        R_wb = self.state.rotation_matrix

        # 2. Specific force rotated to World NED + Gravity
        g_ned = np.array([0.0, 0.0, self.cfg.gravity_mps2], dtype=np.float64)
        a_w = R_wb @ f_b + g_ned

        # 3. Propagate nominal position and velocity (strapdown kinematics)
        self.state.pos_ned_m += self.state.vel_ned_mps * dt + 0.5 * a_w * (dt ** 2)
        self.state.vel_ned_mps += a_w * dt

        # 4. Propagate nominal quaternion orientation
        delta_q = self.exp_rot_vector(omega_b * dt)
        q_new = self.quat_mult(self.state.quat_wb, delta_q)
        q_norm = np.linalg.norm(q_new)
        self.state.quat_wb = q_new / max(1e-12, q_norm)

        # 5. Continuous error-state Jacobian F_x (15x15)
        F_x = np.eye(15, dtype=np.float64)
        F_x[0:3, 3:6] = np.eye(3) * dt
        F_x[3:6, 6:9] = -self.skew_symmetric(R_wb @ f_b) * dt
        F_x[3:6, 9:12] = -R_wb * dt
        F_x[6:9, 6:9] = np.eye(3) - self.skew_symmetric(omega_b) * dt
        F_x[6:9, 12:15] = -np.eye(3) * dt

        # 6. Discrete process noise covariance Q_d (15x15)
        Q_d = np.zeros((15, 15), dtype=np.float64)
        sa2 = self.cfg.sigma_accel_noise ** 2
        sg2 = self.cfg.sigma_gyro_noise ** 2
        sba2 = self.cfg.sigma_accel_bias_walk ** 2
        sbg2 = self.cfg.sigma_gyro_bias_walk ** 2

        Q_d[0:3, 0:3] = np.eye(3) * (0.25 * sa2 * (dt ** 4))
        Q_d[0:3, 3:6] = np.eye(3) * (0.5 * sa2 * (dt ** 3))
        Q_d[3:6, 0:3] = np.eye(3) * (0.5 * sa2 * (dt ** 3))
        Q_d[3:6, 3:6] = np.eye(3) * (sa2 * (dt ** 2))
        Q_d[6:9, 6:9] = np.eye(3) * (sg2 * (dt ** 2))
        Q_d[9:12, 9:12] = np.eye(3) * (sba2 * dt)
        Q_d[12:15, 12:15] = np.eye(3) * (sbg2 * dt)

        # 7. Covariance propagation: P = F_x P F_x^T + Q_d
        self.P = F_x @ self.P @ F_x.T + Q_d
        # Ensure symmetry
        self.P = 0.5 * (self.P + self.P.T)

        # 8. Record state history for delayed-state fusion
        self._history_buffer.append((timestamp_s, self._clone_nav_state(self.state), self.P.copy()))
        if len(self._history_buffer) > self._max_history_len:
            self._history_buffer.pop(0)

        return self.state

    def update_vio_velocity(
        self,
        vel_vio_ned_mps: np.ndarray,
        confidence: float = 1.0,
        sigma_override: Optional[float] = None,
        bypass_gate: bool = False,
    ) -> bool:
        """Fuse 3D metric velocity from VIO into the ES-EKF.

        Args:
            vel_vio_ned_mps: VIO metric velocity in NED frame [Vn, Ve, Vd].
            confidence: Vision tracking confidence in [0.0, 1.0].
            sigma_override: Optional explicit measurement standard deviation.

        Returns:
            True if measurement was accepted and fused, False if gated out.
        """
        if confidence < 0.2:
            return False

        # Dynamically scale noise covariance based on confidence
        base_sigma = sigma_override if sigma_override is not None else self.cfg.sigma_vio_vel_mps
        effective_sigma = base_sigma / max(0.2, confidence)
        R_v = np.eye(3, dtype=np.float64) * (effective_sigma ** 2)

        # Measurement model: observes nominal velocity state (indices 3:6)
        # H is 3x15
        H = np.zeros((3, 15), dtype=np.float64)
        H[0:3, 3:6] = np.eye(3)

        # Innovation: nu = z - v_est
        nu = vel_vio_ned_mps - self.state.vel_ned_mps

        # Innovation covariance: S = H P H^T + R
        S = H @ self.P @ H.T + R_v

        # Chi-square gating test
        try:
            S_inv = np.linalg.inv(S)
            if not bypass_gate and self.cfg.gate_vio_vel_chi2 > 0:
                d2 = float(nu.T @ S_inv @ nu)
                if d2 > self.cfg.gate_vio_vel_chi2:
                    self._consecutive_rejected_vio += 1
                    if self._consecutive_rejected_vio >= 3:
                        # Recovery: state diverged from vision stream, perform soft reset to measurement
                        self.state.vel_ned_mps = vel_vio_ned_mps.copy()
                        self.P[3:6, 3:6] = np.eye(3) * (effective_sigma * 2.0) ** 2
                        self._consecutive_rejected_vio = 0
                        return True
                    return False
        except np.linalg.LinAlgError:
            return False

        self._consecutive_rejected_vio = 0

        # Kalman gain: K = P H^T S^-1
        K = self.P @ H.T @ S_inv

        # Error state correction
        delta_x = K @ nu

        # Joseph-form covariance update: P = (I - KH) P (I - KH)^T + K R K^T
        I_KH = np.eye(15, dtype=np.float64) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ R_v @ K.T
        self.P = 0.5 * (self.P + self.P.T)

        # Inject error state into nominal state and reset error state
        self._inject_and_reset(delta_x)
        return True

    def update_compass_yaw(
        self,
        yaw_compass_rad: float,
        sigma_override: Optional[float] = None,
        bypass_gate: bool = False,
    ) -> bool:
        """Fuse compass / FC magnetometer yaw heading into the ES-EKF.

        Args:
            yaw_compass_rad: Compass yaw heading in radians in [-pi, pi].
            sigma_override: Optional explicit measurement standard deviation.

        Returns:
            True if accepted and fused.
        """
        rpy = self.state.roll_pitch_yaw_rad
        yaw_est = rpy[2]

        # Wrap angular innovation to [-pi, pi]
        nu_yaw = float((yaw_compass_rad - yaw_est + np.pi) % (2.0 * np.pi) - np.pi)

        sigma = sigma_override if sigma_override is not None else self.cfg.sigma_compass_yaw_rad
        R_yaw = sigma ** 2

        # Measurement matrix: observes attitude error theta_z (index 8)
        H = np.zeros((1, 15), dtype=np.float64)
        H[0, 8] = 1.0

        S = (H @ self.P @ H.T + R_yaw).item()
        if S <= 0.0:
            return False

        if not bypass_gate and self.cfg.gate_compass_chi2 > 0:
            d2 = (nu_yaw ** 2) / S
            if d2 > self.cfg.gate_compass_chi2:
                return False

        K = (self.P @ H.T) / S  # shape (15, 1)
        delta_x = (K * nu_yaw).ravel()

        # Joseph-form update
        I_KH = np.eye(15, dtype=np.float64) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + (K * R_yaw) @ K.T
        self.P = 0.5 * (self.P + self.P.T)

        self._inject_and_reset(delta_x)
        return True

    def update_height(
        self,
        height_agl_m: float,
        sigma_override: Optional[float] = None,
    ) -> bool:
        """Fuse Phase 1 AGL height or baro altitude into Down position state.

        Args:
            height_agl_m: Estimated height AGL (meters).
            sigma_override: Optional measurement noise std.

        Returns:
            True if accepted and fused.
        """
        target_down = -height_agl_m
        nu_h = float(target_down - self.state.pos_ned_m[2])

        sigma = sigma_override if sigma_override is not None else self.cfg.sigma_height_m
        R_h = sigma ** 2

        H = np.zeros((1, 15), dtype=np.float64)
        H[0, 2] = 1.0  # observes pos_D (index 2)

        S = (H @ self.P @ H.T + R_h).item()
        if S <= 0.0:
            return False

        d2 = (nu_h ** 2) / S
        if d2 > self.cfg.gate_height_chi2:
            return False

        K = (self.P @ H.T) / S
        delta_x = (K * nu_h).ravel()

        I_KH = np.eye(15, dtype=np.float64) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + (K * R_h) @ K.T
        self.P = 0.5 * (self.P + self.P.T)

        self._inject_and_reset(delta_x)
        return True

    def _inject_and_reset(self, delta_x: np.ndarray) -> None:
        """Inject error state into nominal state and reset error state to zero.

        Nominal state updates:
            p <- p + delta_p
            v <- v + delta_v
            q <- q (x) exp(0.5 * delta_theta)
            b_a <- b_a + delta_ba
            b_g <- b_g + delta_bg
        Covariance reset:
            P <- G P G^T
        """
        dp = delta_x[0:3]
        dv = delta_x[3:6]
        dtheta = delta_x[6:9]
        dba = delta_x[9:12]
        dbg = delta_x[12:15]

        # 1. Position and Velocity corrections
        self.state.pos_ned_m += dp
        self.state.vel_ned_mps += dv

        # 2. Attitude quaternion correction
        dq = self.exp_rot_vector(dtheta)
        q_corrected = self.quat_mult(self.state.quat_wb, dq)
        self.state.quat_wb = q_corrected / np.linalg.norm(q_corrected)

        # 3. Biases corrections
        self.state.accel_bias_mps2 += dba
        self.state.gyro_bias_rad_s += dbg

        # 4. Error state covariance reset Jacobian G (15x15)
        # G is identity everywhere except for attitude error block:
        # G[6:9, 6:9] = I - 0.5 * [dtheta]_x
        G = np.eye(15, dtype=np.float64)
        G[6:9, 6:9] = np.eye(3) - 0.5 * self.skew_symmetric(dtheta)

        self.P = G @ self.P @ G.T
        self.P = 0.5 * (self.P + self.P.T)

    @staticmethod
    def _clone_nav_state(s: NavState) -> NavState:
        """Deep copy of a NavState instance."""
        return NavState(
            timestamp_s=s.timestamp_s,
            pos_ned_m=s.pos_ned_m.copy(),
            vel_ned_mps=s.vel_ned_mps.copy(),
            quat_wb=s.quat_wb.copy(),
            accel_bias_mps2=s.accel_bias_mps2.copy(),
            gyro_bias_rad_s=s.gyro_bias_rad_s.copy(),
        )
