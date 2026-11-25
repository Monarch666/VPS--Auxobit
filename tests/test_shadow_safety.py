"""Unit tests verifying Shadow Mode Safety Guards, Outbound Whitelisting, and Interlocks.

Hard Requirements Verified:
1. Strict Whitelist in Shadow Mode: Only HEARTBEAT (<= 1 Hz), NAMED_VALUE_FLOAT (<= 2 Hz),
   and STATUSTEXT are permitted.
2. Complete suppression of DISTANCE_SENSOR, ODOMETRY, VISION_*, GPS_INPUT, COMMAND_*, PARAM_SET.
3. Proxy interception prevents bypassing the chokepoint via direct mav.mav.send() calls.
4. Configuration conflict raises ConfigurationConflictError when feed_fc_ekf=True in shadow mode.
5. Senders (DistanceSensorSender, OdometrySender, MAVLinkBridge) strictly honor the chokepoint.
6. ShadowSupervisor resource degradation and failure isolation.
"""

import time
import pytest
import numpy as np
from unittest.mock import MagicMock

from vps.config import VPSConfig, MAVLinkConfig, ConfigurationConflictError
from vps.mavlink.safety_chokepoint import ShadowMAVLinkSafetyChokepoint
from vps.mavlink.connection import MAVLinkBridge
from vps.mavlink.distance_sender import DistanceSensorSender
from vps.mavlink.odometry_sender import OdometrySender, OdometrySnapshot
from vps.health.watchdog import ShadowSupervisor
from vps.health.health_monitor import HealthStatus


class DummyMAVMessage:
    """Mock MAVLink message object for testing raw send() interception."""
    def __init__(self, msg_type: str):
        self._type = msg_type

    def get_type(self) -> str:
        return self._type


class DummyMAVMav:
    """Mock mav.mav object with transmission methods."""
    def __init__(self):
        self.sent_packets = []
        self.heartbeat_send = MagicMock(side_effect=lambda *a, **k: self.sent_packets.append(("HEARTBEAT", a, k)))
        self.named_value_float_send = MagicMock(side_effect=lambda *a, **k: self.sent_packets.append(("NAMED_VALUE_FLOAT", a, k)))
        self.statustext_send = MagicMock(side_effect=lambda *a, **k: self.sent_packets.append(("STATUSTEXT", a, k)))
        self.distance_sensor_send = MagicMock(side_effect=lambda *a, **k: self.sent_packets.append(("DISTANCE_SENSOR", a, k)))
        self.odometry_send = MagicMock(side_effect=lambda *a, **k: self.sent_packets.append(("ODOMETRY", a, k)))
        self.vision_position_estimate_send = MagicMock(side_effect=lambda *a, **k: self.sent_packets.append(("VISION_POSITION_ESTIMATE", a, k)))
        self.vision_speed_estimate_send = MagicMock(side_effect=lambda *a, **k: self.sent_packets.append(("VISION_SPEED_ESTIMATE", a, k)))
        self.gps_input_send = MagicMock(side_effect=lambda *a, **k: self.sent_packets.append(("GPS_INPUT", a, k)))
        self.command_long_send = MagicMock(side_effect=lambda *a, **k: self.sent_packets.append(("COMMAND_LONG", a, k)))
        self.param_set_send = MagicMock(side_effect=lambda *a, **k: self.sent_packets.append(("PARAM_SET", a, k)))

    def send(self, msg, *args, **kwargs):
        self.sent_packets.append((msg.get_type(), args, kwargs))


class DummyConnection:
    """Mock MAVLink connection."""
    def __init__(self):
        self.mav = DummyMAVMav()
        self.target_system = 1
        self.target_component = 1


# ==============================================================================
# 1. Whitelist & Blacklist Chokepoint Tests
# ==============================================================================

def test_shadow_chokepoint_permits_whitelisted_messages():
    """Verify HEARTBEAT, NAMED_VALUE_FLOAT, and STATUSTEXT pass through."""
    conn = DummyConnection()
    chokepoint = ShadowMAVLinkSafetyChokepoint(mav_connection=conn, mode="shadow")

    # HEARTBEAT
    assert chokepoint.send_heartbeat() is True
    # NAMED_VALUE_FLOAT
    assert chokepoint.send_named_value_float("VPS_H", 12.34) is True
    assert chokepoint.send_named_value_float("VPS_CONF", 0.95) is True
    # STATUSTEXT
    assert chokepoint.send_statustext("VPS Shadow Mode Active") is True

    assert chokepoint.total_allowed_packets == 4
    assert chokepoint.total_blocked_packets == 0


def test_shadow_chokepoint_strictly_blocks_all_navigation_and_command_packets():
    """Verify DISTANCE_SENSOR, ODOMETRY, VISION, GPS_INPUT, COMMAND, PARAM are hard-blocked."""
    conn = DummyConnection()
    chokepoint = ShadowMAVLinkSafetyChokepoint(mav_connection=conn, mode="shadow")

    blocked_types = [
        "DISTANCE_SENSOR",
        "ODOMETRY",
        "VISION_POSITION_ESTIMATE",
        "VISION_SPEED_ESTIMATE",
        "GPS_INPUT",
        "COMMAND_LONG",
        "COMMAND_INT",
        "PARAM_SET",
        "PARAM_REQUEST_READ",
        "SET_POSITION_TARGET_LOCAL_NED",
        "SET_POSITION_TARGET_GLOBAL_INT",
    ]

    for b_type in blocked_types:
        dummy_fn = MagicMock()
        result = chokepoint.send_intercepted(b_type, dummy_fn, 1, 2, key="val")
        assert result is False, f"Message type {b_type} should have been blocked in shadow mode!"
        assert dummy_fn.call_count == 0, f"Callable for {b_type} must never be invoked!"

    assert chokepoint.total_blocked_packets == len(blocked_types)
    assert chokepoint.total_allowed_packets == 0
    assert len(conn.mav.sent_packets) == 0


def test_shadow_chokepoint_proxy_blocks_direct_method_calls():
    """Verify direct calls to conn.mav.*_send are intercepted and blocked in shadow mode."""
    conn = DummyConnection()
    chokepoint = ShadowMAVLinkSafetyChokepoint(mav_connection=conn, mode="shadow")

    # Direct calls that would normally transmit
    assert conn.mav.distance_sensor_send(100) is False
    assert conn.mav.odometry_send(0, 0, 0, 0, [0]*4, 0, 0, 0, 0, 0, 0, [0]*21, [0]*21, 0, 0) is False
    assert conn.mav.command_long_send(1, 1, 511, 0, 0, 0, 0, 0, 0, 0, 0) is False
    assert conn.mav.param_set_send("PARAM_XYZ", 1.0, 0) is False

    assert chokepoint.total_blocked_packets == 4
    assert len(conn.mav.sent_packets) == 0


def test_shadow_chokepoint_raw_send_interception():
    """Verify calling raw mav.send(msg) is caught and dropped for blocked message types."""
    conn = DummyConnection()
    chokepoint = ShadowMAVLinkSafetyChokepoint(mav_connection=conn, mode="shadow")

    # Attempt to bypass using raw message
    msg = DummyMAVMessage("DISTANCE_SENSOR")
    result = conn.mav.send(msg)
    assert result is False
    assert len(conn.mav.sent_packets) == 0

    # Whitelisted raw message
    hb_msg = DummyMAVMessage("HEARTBEAT")
    result_hb = conn.mav.send(hb_msg)
    assert result_hb is not False
    assert len(conn.mav.sent_packets) == 1


def test_shadow_chokepoint_rate_limiting():
    """Verify NAMED_VALUE_FLOAT is rate limited to <= 2 Hz and HEARTBEAT <= 1 Hz."""
    conn = DummyConnection()
    chokepoint = ShadowMAVLinkSafetyChokepoint(
        mav_connection=conn,
        mode="shadow",
        max_named_value_rate_hz=2.0,
        max_heartbeat_rate_hz=1.0,
    )

    # First send succeeds
    assert chokepoint.send_named_value_float("VPS_H", 10.0) is True

    # Immediate second send of same variable should be suppressed
    assert chokepoint.send_named_value_float("VPS_H", 10.1) is False

    # Different variable has its own rate limiter
    assert chokepoint.send_named_value_float("VPS_CONF", 0.9) is True

    # HEARTBEAT rate limit
    assert chokepoint.send_heartbeat() is True
    assert chokepoint.send_heartbeat() is False  # Suppressed


def test_flight_mode_allows_all_navigation_transmissions():
    """Verify that when mode == 'flight', navigation packets are NOT blocked."""
    conn = DummyConnection()
    chokepoint = ShadowMAVLinkSafetyChokepoint(mav_connection=conn, mode="flight")

    assert chokepoint.is_shadow_mode is False
    assert chokepoint.is_message_allowed("DISTANCE_SENSOR") is True
    assert chokepoint.is_message_allowed("ODOMETRY") is True
    assert chokepoint.is_message_allowed("COMMAND_LONG") is True

    dummy_fn = MagicMock()
    assert chokepoint.send_intercepted("DISTANCE_SENSOR", dummy_fn) is True
    assert dummy_fn.call_count == 1


# ==============================================================================
# 2. Config Invariants & Safety Interlock Tests
# ==============================================================================

def test_config_raises_on_shadow_mode_with_feed_fc_ekf():
    """Violating the safety invariant (shadow mode + feed_fc_ekf=True) MUST fail fast."""
    with pytest.raises(ConfigurationConflictError) as exc_info:
        VPSConfig(
            mode="shadow",
            mavlink=MAVLinkConfig(feed_fc_ekf=True),
        )
    assert "Safety Violation" in str(exc_info.value)
    assert "feed_fc_ekf" in str(exc_info.value)


def test_config_loads_valid_shadow_configuration():
    """Valid shadow configuration (feed_fc_ekf=False) initializes cleanly."""
    cfg = VPSConfig(
        mode="shadow",
        mavlink=MAVLinkConfig(feed_fc_ekf=False),
    )
    assert cfg.mode == "shadow"
    assert cfg.mavlink.feed_fc_ekf is False


# ==============================================================================
# 3. Sender Integration Tests (DistanceSensorSender, OdometrySender, Bridge)
# ==============================================================================

def test_distance_sensor_sender_blocked_in_shadow_mode():
    """Verify DistanceSensorSender transmits nothing when wired to shadow chokepoint."""
    conn = DummyConnection()
    chokepoint = ShadowMAVLinkSafetyChokepoint(mav_connection=conn, mode="shadow")
    sender = DistanceSensorSender(
        mav_connection=conn,
        rate_hz=15.0,
        chokepoint=chokepoint,
    )

    sender.update_measurement(height_agl_m=15.0, confidence=0.9, covariance_m2=0.01)
    result = sender.send_single_packet()

    assert result is False
    assert sender.packets_sent == 0
    assert len(conn.mav.sent_packets) == 0


def test_odometry_sender_blocked_in_shadow_mode():
    """Verify OdometrySender transmits nothing when wired to shadow chokepoint."""
    conn = DummyConnection()
    chokepoint = ShadowMAVLinkSafetyChokepoint(mav_connection=conn, mode="shadow")
    sender = OdometrySender(
        mav_connection=conn,
        rate_hz=25.0,
        chokepoint=chokepoint,
    )

    # Populate a valid snapshot
    health = sender.health_mon.evaluate_health(
        confidence=0.95,
        innovation_chi2=1.0,
        observation_timestamp_s=time.monotonic(),
    )
    sender._current_snapshot = OdometrySnapshot(
        timestamp_usec=int(time.monotonic() * 1e6),
        pos_ned_m=np.array([1.0, 2.0, -10.0]),
        vel_ned_mps=np.array([0.5, 0.0, 0.0]),
        quat_wb=np.array([1.0, 0.0, 0.0, 0.0]),
        gyro_rad_s=np.zeros(3),
        pose_cov_21=[0.01] * 21,
        vel_cov_21=[0.01] * 21,
        confidence=0.95,
        health=health,
    )

    result = sender.send_current_packet()
    assert result is False
    assert sender.packets_sent == 0
    assert len(conn.mav.sent_packets) == 0


def test_mavlink_bridge_request_streams_suppressed_in_shadow_mode():
    """Verify request_streams does not issue COMMAND_LONG when in shadow mode."""
    bridge = MAVLinkBridge(mode="shadow")
    bridge.mav = DummyConnection()
    bridge.chokepoint = ShadowMAVLinkSafetyChokepoint(mav_connection=bridge.mav, mode="shadow")

    bridge.request_streams(rate_hz=100)
    assert len(bridge.mav.mav.sent_packets) == 0


# ==============================================================================
# 4. ShadowSupervisor Degradation & Failure Isolation Tests
# ==============================================================================

def test_shadow_supervisor_health_and_degradation():
    """Verify ShadowSupervisor monitors resources and triggers degradation callbacks."""
    supervisor = ShadowSupervisor(min_fps=15.0, max_cpu_temp_c=80.0, min_disk_free_mb=500.0)

    degrade_events = []
    supervisor.register_degradation_callback(lambda evt: degrade_events.append(evt))

    # Normal health check
    health = supervisor.evaluate_shadow_health()
    assert "disk_free_mb" in health
    assert "is_degraded" in health

    # Simulate thermal spike (e.g. 79 C)
    supervisor.read_cpu_temperature = MagicMock(return_value=79.5)
    health_hot = supervisor.evaluate_shadow_health()
    assert health_hot["is_degraded"] is True
    assert "DEGRADE_RESOURCE_PRESSURE" in degrade_events

    # Restore normal temperature
    supervisor.read_cpu_temperature = MagicMock(return_value=55.0)
    health_normal = supervisor.evaluate_shadow_health()
    assert health_normal["is_degraded"] is False
    assert "RESTORE_NORMAL_OPERATION" in degrade_events


def test_mavlink_bridge_parse_serial_connection_string():
    """Verify serial device:baud shorthand parses cleanly without breaking UDP URIs."""
    assert MAVLinkBridge.parse_connection_string("/dev/ttyAMA0:921600", 57600) == ("/dev/ttyAMA0", 921600)
    assert MAVLinkBridge.parse_connection_string("/dev/ttyAMA0", 921600) == ("/dev/ttyAMA0", 921600)
    assert MAVLinkBridge.parse_connection_string("COM4:115200", 57600) == ("COM4", 115200)
    assert MAVLinkBridge.parse_connection_string("udpin:127.0.0.1:14550", 921600) == ("udpin:127.0.0.1:14550", 921600)
