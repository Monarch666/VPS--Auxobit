"""MAVLink ODOMETRY and VISION_POSITION_ESTIMATE Dispatcher for ArduPilot ExternalNav.

Constructs and streams high-rate navigation packets to ArduPilot 4.5.7:
1. Primary protocol: MAVLink ODOMETRY (#331) at 20-30 Hz (default 25 Hz)
   - Frame ID: MAV_FRAME_LOCAL_NED
   - Child Frame ID: MAV_FRAME_BODY_NED
   - 6-DOF Pose (X, Y, Z meters, Quaternion [w, x, y, z])
   - 6-DOF Twist (Vx, Vy, Vz m/s, Roll, Pitch, Yaw angular rates rad/s)
   - 21-element row-major upper-triangular Pose and Velocity covariance matrices
   - Estimator Type: MAV_ESTIMATOR_TYPE_VISION (3)
   - Reset counter incrementing on relocalization / discrete step updates
2. Secondary protocol: VISION_POSITION_ESTIMATE (#102) and VISION_SPEED_ESTIMATE (#103)
3. Tertiary protocol: GPS_INPUT (#232) for virtual GNSS emulation
4. Safety Interlock: Controlled via HealthMonitor; logs advisory warnings and inflates
   covariances or suppresses active transmission when health gates fail.
"""

import os
os.environ["MAVLINK20"] = "1"

from dataclasses import dataclass, field
from typing import Optional, List, Tuple
import threading
import time
import numpy as np
from pymavlink import mavutil

from vps.config import MAVLinkConfig
from vps.nav_ekf.es_ekf import NavState
from vps.health.health_monitor import HealthMonitor, HealthStatus


def cov6x6_to_upper_triangle(cov6: np.ndarray) -> List[float]:
    """Extract 21-element upper triangular representation of a 6x6 covariance matrix."""
    out: List[float] = []
    for i in range(6):
        for j in range(i, 6):
            out.append(float(cov6[i, j]))
    return out


@dataclass
class OdometrySnapshot:
    """Navigation snapshot ready for MAVLink packing."""
    timestamp_usec: int
    pos_ned_m: np.ndarray  # shape (3,) [X_n, Y_e, Z_d]
    vel_ned_mps: np.ndarray  # shape (3,) [V_n, V_e, V_d]
    quat_wb: np.ndarray  # shape (4,) [w, x, y, z]
    gyro_rad_s: np.ndarray  # shape (3,) [p, q, r]
    pose_cov_21: List[float]  # 21 elements
    vel_cov_21: List[float]   # 21 elements
    confidence: float
    health: HealthStatus
    reset_counter: int = 0


class OdometrySender:
    """Streams 6-DOF ODOMETRY or VISION messages to ArduPilot via MAVLink."""

    def __init__(
        self,
        mav_connection: mavutil.mavlink_connection,
        config: Optional[MAVLinkConfig] = None,
        health_monitor: Optional[HealthMonitor] = None,
        rate_hz: float = 25.0,
    ) -> None:
        """Initialize odometry sender.

        Args:
            mav_connection: Active pymavlink connection.
            config: MAVLinkConfig instance.
            health_monitor: HealthMonitor instance enforcing safety interlocks.
            rate_hz: Streaming frequency in Hz (20 to 30 Hz).
        """
        self.mav = mav_connection
        self.cfg = config if config is not None else MAVLinkConfig()
        self.health_mon = health_monitor if health_monitor is not None else HealthMonitor(config=self.cfg)
        self.rate_hz = max(10.0, min(50.0, float(rate_hz)))
        self.period_s = 1.0 / self.rate_hz

        self._lock = threading.Lock()
        self._current_snapshot: Optional[OdometrySnapshot] = None
        self._reset_counter: int = 0
        self._is_running = False
        self._thread: Optional[threading.Thread] = None
        self._packets_sent = 0
        self._boot_time_s = time.monotonic()

    def update_from_ekf(
        self,
        nav_state: NavState,
        P_15x15: np.ndarray,
        confidence: float = 1.0,
        innovation_chi2: float = 0.0,
        frame_timestamp_s: float = 0.0,
        reset_counter: int = 0,
        gyro_rad_s: Optional[np.ndarray] = None,
    ) -> HealthStatus:
        """Update current navigation state from ES-EKF.

        Args:
            nav_state: Current NavState instance.
            P_15x15: Full 15x15 ES-EKF error covariance matrix.
            confidence: Inlier-based vision confidence [0.0, 1.0].
            innovation_chi2: Normalized innovation squared from last update.
            frame_timestamp_s: Frame capture timestamp.
            reset_counter: Counter incrementing upon discrete jumps/relocalization.

        Returns:
            Evaluated HealthStatus.
        """
        now = time.monotonic()
        t_capture = frame_timestamp_s if frame_timestamp_s > 0.0 else now

        # Evaluate safety gates
        health = self.health_mon.evaluate_health(
            confidence=confidence,
            innovation_chi2=innovation_chi2,
            observation_timestamp_s=t_capture,
            current_timestamp_s=now,
        )

        # Build 6x6 Pose Covariance [X, Y, Z, Roll, Pitch, Yaw]
        cov_pose_6x6 = np.zeros((6, 6), dtype=np.float64)
        cov_pose_6x6[0:3, 0:3] = P_15x15[0:3, 0:3]  # Position covariance
        cov_pose_6x6[3:6, 3:6] = P_15x15[6:9, 6:9]  # Attitude error covariance

        # Build 6x6 Velocity Covariance [Vx, Vy, Vz, P, Q, R]
        cov_vel_6x6 = np.zeros((6, 6), dtype=np.float64)
        cov_vel_6x6[0:3, 0:3] = P_15x15[3:6, 3:6]   # Linear velocity covariance
        cov_vel_6x6[3:6, 3:6] = np.eye(3) * (0.02 ** 2)  # Angular rate variance

        # Apply health-based covariance inflation
        cov_pose_inflated = self.health_mon.inflate_covariance_matrix(cov_pose_6x6, health)
        cov_vel_inflated = self.health_mon.inflate_covariance_matrix(cov_vel_6x6, health)

        # Convert to 21-element upper triangles
        pose_cov_21 = cov6x6_to_upper_triangle(cov_pose_inflated)
        vel_cov_21 = cov6x6_to_upper_triangle(cov_vel_inflated)

        # Timestamp in microseconds
        time_usec = int(round(now * 1e6))

        snapshot = OdometrySnapshot(
            timestamp_usec=time_usec,
            pos_ned_m=nav_state.pos_ned_m.copy(),
            vel_ned_mps=nav_state.vel_ned_mps.copy(),
            quat_wb=nav_state.quat_wb.copy(),
            gyro_rad_s=gyro_rad_s.copy() if gyro_rad_s is not None else np.zeros(3, dtype=np.float64),
            pose_cov_21=pose_cov_21,
            vel_cov_21=vel_cov_21,
            confidence=confidence,
            health=health,
            reset_counter=reset_counter,
        )

        with self._lock:
            self._current_snapshot = snapshot
            self._reset_counter = reset_counter

        return health

    def send_current_packet(self) -> bool:
        """Transmit a single packet based on configured mode and safety interlocks.

        Returns:
            True if packet was dispatched over MAVLink, False otherwise.
        """
        with self._lock:
            snap = self._current_snapshot

        if snap is None:
            return False

        # Mode check: if ADVISORY_LOG_ONLY and user disabled feeding, do not dispatch
        # unless advisory streaming is explicitly allowed
        if not snap.health.can_feed_fc and not self.cfg.feed_fc_ekf:
            # Advisory mode: suppress active control injection into ArduPilot EKF
            return False

        mode = self.cfg.odometry_mode.upper()
        if mode == "ODOMETRY":
            return self._send_odometry(snap)
        elif mode == "VISION_POSITION_ESTIMATE":
            return self._send_vision_position(snap)
        elif mode == "GPS_INPUT":
            return self._send_gps_input(snap)
        else:
            return self._send_odometry(snap)

    def _send_odometry(self, snap: OdometrySnapshot) -> bool:
        """Send MAVLink ODOMETRY (#331) message."""
        try:
            self.mav.mav.odometry_send(
                time_usec=snap.timestamp_usec,
                frame_id=mavutil.mavlink.MAV_FRAME_LOCAL_NED,
                child_frame_id=mavutil.mavlink.MAV_FRAME_BODY_NED,
                x=float(snap.pos_ned_m[0]),
                y=float(snap.pos_ned_m[1]),
                z=float(snap.pos_ned_m[2]),
                q=[float(x) for x in snap.quat_wb],
                vx=float(snap.vel_ned_mps[0]),
                vy=float(snap.vel_ned_mps[1]),
                vz=float(snap.vel_ned_mps[2]),
                rollspeed=float(snap.gyro_rad_s[0]),
                pitchspeed=float(snap.gyro_rad_s[1]),
                yawspeed=float(snap.gyro_rad_s[2]),
                pose_covariance=snap.pose_cov_21,
                velocity_covariance=snap.vel_cov_21,
                reset_counter=int(snap.reset_counter % 256),
                estimator_type=mavutil.mavlink.MAV_ESTIMATOR_TYPE_VISION,
            )
            self._packets_sent += 1
            return True
        except Exception:
            return False

    def _send_vision_position(self, snap: OdometrySnapshot) -> bool:
        """Send VISION_POSITION_ESTIMATE (#102) and VISION_SPEED_ESTIMATE (#103)."""
        try:
            # Extract roll, pitch, yaw from quaternion
            w, x, y, z = snap.quat_wb
            roll = float(np.arctan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y)))
            pitch = float(np.arcsin(np.clip(2.0 * (w * y - z * x), -1.0, 1.0)))
            yaw = float(np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))

            # VISION_POSITION_ESTIMATE
            self.mav.mav.vision_position_estimate_send(
                usec=snap.timestamp_usec,
                x=float(snap.pos_ned_m[0]),
                y=float(snap.pos_ned_m[1]),
                z=float(snap.pos_ned_m[2]),
                roll=roll,
                pitch=pitch,
                yaw=yaw,
                covariance=snap.pose_cov_21,
                reset_counter=int(snap.reset_counter % 256),
            )

            # VISION_SPEED_ESTIMATE
            vel_cov_9 = [
                snap.vel_cov_21[0], snap.vel_cov_21[1], snap.vel_cov_21[2],
                snap.vel_cov_21[1], snap.vel_cov_21[6], snap.vel_cov_21[7],
                snap.vel_cov_21[2], snap.vel_cov_21[7], snap.vel_cov_21[11],
            ]
            self.mav.mav.vision_speed_estimate_send(
                usec=snap.timestamp_usec,
                x=float(snap.vel_ned_mps[0]),
                y=float(snap.vel_ned_mps[1]),
                z=float(snap.vel_ned_mps[2]),
                covariance=vel_cov_9,
                reset_counter=int(snap.reset_counter % 256),
            )
            self._packets_sent += 1
            return True
        except Exception:
            return False

    def _send_gps_input(
        self,
        snap: OdometrySnapshot,
        home_lat_deg: float = 19.8762,
        home_lon_deg: float = 75.3433,
        home_alt_m: float = 580.0,
    ) -> bool:
        """Send virtual GPS_INPUT (#232) message."""
        try:
            # Tangent plane flat-Earth conversion: North/East meters to lat/lon
            r_earth = 6378137.0
            dlat = snap.pos_ned_m[0] / r_earth
            dlon = snap.pos_ned_m[1] / (r_earth * np.cos(np.radians(home_lat_deg)))

            lat_deg = home_lat_deg + np.degrees(dlat)
            lon_deg = home_lon_deg + np.degrees(dlon)
            alt_m = home_alt_m - snap.pos_ned_m[2]  # z is Down

            horiz_acc = float(np.sqrt(snap.pose_cov_21[0]))
            vert_acc = float(np.sqrt(snap.pose_cov_21[11]))
            speed_acc = float(np.sqrt(snap.vel_cov_21[0]))

            self.mav.mav.gps_input_send(
                time_usec=snap.timestamp_usec,
                gps_id=0,
                ignore_flags=0,
                time_week_ms=0,
                time_week=0,
                fix_type=3,  # 3D Fix
                lat=int(round(lat_deg * 1e7)),
                lon=int(round(lon_deg * 1e7)),
                alt=float(alt_m),
                hdop=float(max(0.5, horiz_acc)),
                vdop=float(max(0.8, vert_acc)),
                vn=float(snap.vel_ned_mps[0]),
                ve=float(snap.vel_ned_mps[1]),
                vd=float(snap.vel_ned_mps[2]),
                speed_accuracy=speed_acc,
                horiz_accuracy=horiz_acc,
                vert_accuracy=vert_acc,
                satellites_visible=14,
                yaw=0,
            )
            self._packets_sent += 1
            return True
        except Exception:
            return False

    def start(self) -> None:
        """Start asynchronous transmission thread at target rate_hz."""
        if self._is_running:
            return
        self._is_running = True
        self._thread = threading.Thread(target=self._run_loop, daemon=True, name="VPS_OdometrySender")
        self._thread.start()

    def stop(self) -> None:
        """Stop transmission thread."""
        self._is_running = False
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None

    def _run_loop(self) -> None:
        """Background loop streaming packets at fixed period."""
        while self._is_running:
            t_start = time.monotonic()
            self.send_current_packet()
            elapsed = time.monotonic() - t_start
            sleep_time = max(0.001, self.period_s - elapsed)
            time.sleep(sleep_time)

    @property
    def packets_sent(self) -> int:
        """Total number of packets successfully sent."""
        return self._packets_sent

