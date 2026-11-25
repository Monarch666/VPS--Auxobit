"""MAVLink telemetry bridge for ArduPilot flight controllers.

Manages bidirectional communication:
- Reads high-rate ATTITUDE, HIGHRES_IMU, LOCAL_POSITION_NED, VFR_HUD into TelemetryRingBuffer.
- Broadcasts downward DISTANCE_SENSOR packets at >= 10 Hz.
"""

from typing import Optional, Callable, Dict, Any
import threading
import time
import numpy as np
from pymavlink import mavutil

from vps.config import MAVLinkConfig
from .telemetry_buffer import TelemetryRingBuffer, TelemetrySample
from .distance_sender import DistanceSensorSender
from .odometry_sender import OdometrySender
from .safety_chokepoint import ShadowMAVLinkSafetyChokepoint


class MAVLinkBridge:
    """Thread-safe MAVLink telemetry receiver and rangefinder transmitter."""

    def __init__(
        self,
        config: Optional[MAVLinkConfig] = None,
        ring_buffer: Optional[TelemetryRingBuffer] = None,
        mode: str = "flight",
    ) -> None:
        """Initialize MAVLink bridge.

        Args:
            config: MAVLink connection and stream configuration.
            ring_buffer: Shared telemetry buffer. Created if not provided.
            mode: Operating mode ("flight", "shadow", "handheld", "sitl").
        """
        self.cfg = config if config is not None else MAVLinkConfig()
        self.buffer = ring_buffer if ring_buffer is not None else TelemetryRingBuffer()
        self.mode = str(mode).lower()

        self.mav: Optional[mavutil.mavlink_connection] = None
        self.chokepoint: Optional[ShadowMAVLinkSafetyChokepoint] = None
        self.distance_sender: Optional[DistanceSensorSender] = None
        self.odometry_sender: Optional[OdometrySender] = None

        self._running = False
        self._recv_thread: Optional[threading.Thread] = None

        # Cached latest telemetry states
        self._state_lock = threading.Lock()
        self._latest_gyro = np.zeros(3, dtype=np.float64)
        self._latest_accel = np.array([0.0, 0.0, -9.80665], dtype=np.float64)
        self._latest_att = np.zeros(3, dtype=np.float64)
        self._latest_vel = np.zeros(3, dtype=np.float64)
        self._latest_pos = np.zeros(3, dtype=np.float64)
        self._latest_baro = 0.0
        self._fc_status: Dict[str, Any] = {
            "heartbeat_seen": False,
            "is_armed": False,
            "flight_mode": "UNKNOWN",
            "battery_voltage_v": 0.0,
            "gps_fix_type": 0,
            "gps_lat_deg": 0.0,
            "gps_lon_deg": 0.0,
            "gps_alt_m": 0.0,
            "gps_eph_m": 0.0,
            "gps_epv_m": 0.0,
            "gps_satellites": 0,
            "rangefinder_dist_m": -1.0,
            "fc_boot_ms": 0,
        }

    @staticmethod
    def parse_connection_string(connection_string: str, default_baud: int) -> tuple[str, int]:
        """Split '<device>:<baud>' serial shorthand into (device_or_uri, baudrate)."""
        conn_str = str(connection_string).strip()
        baud = int(default_baud)
        if (conn_str.startswith("/dev/") or conn_str.upper().startswith("COM")) and ":" in conn_str:
            dev_part, baud_part = conn_str.rsplit(":", 1)
            if baud_part.isdigit():
                return dev_part, int(baud_part)
        return conn_str, baud

    def connect(self) -> bool:
        """Establish MAVLink connection and install safety chokepoint."""
        try:
            conn_str, baud = self.parse_connection_string(
                self.cfg.connection_string,
                self.cfg.baudrate,
            )
            self.mav = mavutil.mavlink_connection(
                conn_str,
                baud=baud,
                autoreconnect=True,
            )
            # Install Safety Chokepoint before senders are initialized
            self.chokepoint = ShadowMAVLinkSafetyChokepoint(
                mav_connection=self.mav,
                mode=self.mode,
            )
            self.distance_sender = DistanceSensorSender(
                mav_connection=self.mav,
                config=self.cfg,
                rate_hz=self.cfg.distance_sensor_rate_hz,
                chokepoint=self.chokepoint,
            )
            self.odometry_sender = OdometrySender(
                mav_connection=self.mav,
                config=self.cfg,
                rate_hz=self.cfg.odometry_rate_hz,
                chokepoint=self.chokepoint,
            )
            return True
        except Exception as e:
            print(f"[ERROR] Failed to establish MAVLink connection: {e}")
            return False

    def start(self) -> None:
        """Start receiver and transmitter threads."""
        if self._running or self.mav is None:
            return
        self._running = True
        self._recv_thread = threading.Thread(target=self._recv_worker, daemon=True)
        self._recv_thread.start()
        if self.distance_sender:
            self.distance_sender.start_streaming()
        if self.odometry_sender:
            self.odometry_sender.start()

    def request_streams(self, rate_hz: Optional[int] = None) -> None:
        """Request high-rate data streams from ArduPilot.

        In shadow mode, sending COMMAND_LONG to the FC is strictly prohibited;
        streams must be pre-configured on ArduPilot telemetry port.
        """
        if self.mav is None:
            return
        if self.chokepoint and self.chokepoint.is_shadow_mode:
            # Under shadow mode invariants, sending COMMAND_LONG to FC is suppressed
            return

        rate = rate_hz if rate_hz is not None else self.cfg.stream_rate_hz
        interval_us = int(1e6 / rate)

        # Message IDs to request at high rate:
        # HIGHRES_IMU (105), ATTITUDE (30), LOCAL_POSITION_NED (32), VFR_HUD (74)
        msg_ids = [
            mavutil.mavlink.MAVLINK_MSG_ID_HIGHRES_IMU,
            mavutil.mavlink.MAVLINK_MSG_ID_ATTITUDE,
            mavutil.mavlink.MAVLINK_MSG_ID_LOCAL_POSITION_NED,
            mavutil.mavlink.MAVLINK_MSG_ID_VFR_HUD,
        ]

        for mid in msg_ids:
            self.mav.mav.command_long_send(
                self.mav.target_system,
                self.mav.target_component,
                mavutil.mavlink.MAV_CMD_SET_MESSAGE_INTERVAL,
                0,
                mid,
                interval_us,
                0, 0, 0, 0, 0
            )

    def send_heartbeat(self) -> bool:
        """Send HEARTBEAT packet via safety chokepoint."""
        if self.chokepoint:
            return self.chokepoint.send_heartbeat()
        return False

    def send_named_value_float(self, name: str, value: float) -> bool:
        """Send diagnostic float to GCS and ArduPilot NVF DataFlash log via chokepoint."""
        if self.chokepoint:
            return self.chokepoint.send_named_value_float(name, value)
        return False

    def send_statustext(self, text: str, severity: int = mavutil.mavlink.MAV_SEVERITY_INFO) -> bool:
        """Send diagnostic text string to GCS via chokepoint."""
        if self.chokepoint:
            return self.chokepoint.send_statustext(text, severity)
        return False

    def get_fc_status(self) -> Dict[str, Any]:
        """Return a thread-safe snapshot of flight controller arming, GPS, and health status."""
        with self._state_lock:
            return dict(self._fc_status)

    def _recv_worker(self) -> None:
        """Inbound MAVLink parser loop."""
        while self._running and self.mav is not None:
            msg = self.mav.recv_match(blocking=True, timeout=0.1)
            if msg is None:
                continue

            t_now = time.monotonic()
            msg_type = msg.get_type()

            if msg_type == "HEARTBEAT":
                autopilot = getattr(msg, "autopilot", mavutil.mavlink.MAV_AUTOPILOT_INVALID)
                if autopilot != mavutil.mavlink.MAV_AUTOPILOT_INVALID:
                    armed = bool(msg.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED)
                    try:
                        mode_name = str(mavutil.mode_string_v10(msg))
                    except Exception:
                        mode_name = str(getattr(msg, "custom_mode", "UNKNOWN"))
                    with self._state_lock:
                        self._fc_status["heartbeat_seen"] = True
                        self._fc_status["is_armed"] = armed
                        self._fc_status["flight_mode"] = mode_name
                continue
            elif msg_type == "GPS_RAW_INT":
                eph = getattr(msg, "eph", 65535)
                epv = getattr(msg, "epv", 65535)
                with self._state_lock:
                    self._fc_status["gps_fix_type"] = int(getattr(msg, "fix_type", 0))
                    self._fc_status["gps_lat_deg"] = float(getattr(msg, "lat", 0)) * 1e-7
                    self._fc_status["gps_lon_deg"] = float(getattr(msg, "lon", 0)) * 1e-7
                    self._fc_status["gps_alt_m"] = float(getattr(msg, "alt", 0)) * 1e-3
                    self._fc_status["gps_eph_m"] = (float(eph) * 0.01) if eph < 65535 else 0.0
                    self._fc_status["gps_epv_m"] = (float(epv) * 0.01) if epv < 65535 else 0.0
                    self._fc_status["gps_satellites"] = int(getattr(msg, "satellites_visible", 0))
                continue
            elif msg_type == "SYS_STATUS":
                vbatt = getattr(msg, "voltage_battery", 65535)
                if vbatt < 65535:
                    with self._state_lock:
                        self._fc_status["battery_voltage_v"] = float(vbatt) * 0.001
                continue
            elif msg_type == "RANGEFINDER":
                with self._state_lock:
                    self._fc_status["rangefinder_dist_m"] = float(getattr(msg, "distance", -1.0))
                continue
            elif msg_type == "HIGHRES_IMU":
                self._latest_gyro = np.array([msg.xgyro, msg.ygyro, msg.zgyro], dtype=np.float64)
                self._latest_accel = np.array([msg.xacc, msg.yacc, msg.zacc], dtype=np.float64)
            elif msg_type == "RAW_IMU":
                # Convert raw to rad/s if gyro scale factor applies
                self._latest_gyro = np.array([msg.xgyro, msg.ygyro, msg.zgyro], dtype=np.float64) * 0.001
                self._latest_accel = np.array([msg.xacc, msg.yacc, msg.zacc], dtype=np.float64) * 9.80665 / 1000.0
            elif msg_type == "ATTITUDE":
                self._latest_att = np.array([msg.roll, msg.pitch, msg.yaw], dtype=np.float64)
                with self._state_lock:
                    self._fc_status["fc_boot_ms"] = int(getattr(msg, "time_boot_ms", 0))
            elif msg_type == "LOCAL_POSITION_NED":
                self._latest_pos = np.array([msg.x, msg.y, msg.z], dtype=np.float64)
                self._latest_vel = np.array([msg.vx, msg.vy, msg.vz], dtype=np.float64)
            elif msg_type == "VFR_HUD":
                self._latest_baro = float(msg.alt)
            else:
                continue

            # Insert consolidated sample into ring buffer
            sample = TelemetrySample(
                timestamp_s=t_now,
                gyro_rad_s=self._latest_gyro.copy(),
                attitude_rpy_rad=self._latest_att.copy(),
                vel_ned_mps=self._latest_vel.copy(),
                baro_alt_m=self._latest_baro,
                pos_ned_m=self._latest_pos.copy(),
                accel_mps2=self._latest_accel.copy(),
            )
            self.buffer.add_sample(sample)

    def stop(self) -> None:
        """Stop threads and close connection."""
        self._running = False
        if self.distance_sender:
            self.distance_sender.stop_streaming()
        if self.odometry_sender:
            self.odometry_sender.stop()
        if self._recv_thread and self._recv_thread.is_alive():
            self._recv_thread.join(timeout=0.5)
        if self.mav is not None:
            self.mav.close()
            self.mav = None

