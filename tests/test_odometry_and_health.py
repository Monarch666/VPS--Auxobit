"""Unit tests for Milestone 10: MAVLink ODOMETRY Dispatcher and Health Monitor."""

from pathlib import Path
import threading
import time
import numpy as np
import pytest
from pymavlink import mavutil

from vps.config import MAVLinkConfig
from vps.health.health_monitor import HealthMonitor, HealthStatus
from vps.mavlink.odometry_sender import OdometrySender, cov6x6_to_upper_triangle, OdometrySnapshot
from vps.nav_ekf.es_ekf import NavState


class MockMAVLinkSink:
    """Mock in-memory MAVLink sink capturing dispatched messages."""

    def __init__(self) -> None:
        self.mav = self
        self.sent_odometry: list = []
        self.sent_vision_pos: list = []
        self.sent_vision_spd: list = []
        self.sent_gps_input: list = []

    def odometry_send(self, **kwargs) -> None:
        self.sent_odometry.append(kwargs)

    def vision_position_estimate_send(self, **kwargs) -> None:
        self.sent_vision_pos.append(kwargs)

    def vision_speed_estimate_send(self, **kwargs) -> None:
        self.sent_vision_spd.append(kwargs)

    def gps_input_send(self, **kwargs) -> None:
        self.sent_gps_input.append(kwargs)


class TestCovarianceUpperTriangle:
    """Test suite for covariance upper triangle extraction."""

    def test_cov6x6_to_upper_triangle_indices(self) -> None:
        """Verify 6x6 matrix converts to exactly 21 upper-triangular elements."""
        M = np.zeros((6, 6), dtype=np.float64)
        val = 1.0
        for i in range(6):
            for j in range(i, 6):
                M[i, j] = val
                M[j, i] = val
                val += 1.0

        tri = cov6x6_to_upper_triangle(M)
        assert len(tri) == 21

        # Check diagonal values
        # Diagonal indices in 21-element list: 0, 6, 11, 15, 18, 20
        diag_expected = [M[i, i] for i in range(6)]
        diag_extracted = [tri[0], tri[6], tri[11], tri[15], tri[18], tri[20]]
        assert diag_extracted == diag_expected


class TestHealthMonitor:
    """Test suite for HealthMonitor safety interlocks and covariance inflation."""

    def test_nominal_health_passes(self) -> None:
        """Verify nominal inputs pass all health gates."""
        cfg = MAVLinkConfig(feed_fc_ekf=True, min_confidence_to_feed=0.40)
        monitor = HealthMonitor(config=cfg)

        t_now = time.monotonic()
        status = monitor.evaluate_health(
            confidence=0.85,
            innovation_chi2=2.5,
            observation_timestamp_s=t_now - 0.03,  # 30 ms latency
            current_timestamp_s=t_now,
        )

        assert status.is_healthy is True
        assert status.can_feed_fc is True
        assert status.confidence_ok is True
        assert status.innovation_ok is True
        assert status.freshness_ok is True
        assert status.covariance_scale <= 2.5
        assert "HEALTH_NOMINAL" in status.status_message

    def test_advisory_mode_blocks_feeding(self) -> None:
        """Verify ADVISORY_LOG_ONLY mode suppresses active feeding even if healthy."""
        cfg = MAVLinkConfig(feed_fc_ekf=False)  # default safe mode
        monitor = HealthMonitor(config=cfg)

        t_now = time.monotonic()
        status = monitor.evaluate_health(
            confidence=0.90,
            innovation_chi2=1.0,
            observation_timestamp_s=t_now - 0.02,
            current_timestamp_s=t_now,
        )

        assert status.is_healthy is True
        assert status.can_feed_fc is False  # Safety interlock active
        assert status.mode == "ADVISORY_LOG_ONLY"

    def test_low_confidence_inflates_covariance(self) -> None:
        """Verify low confidence triggers covariance inflation and fails gate."""
        cfg = MAVLinkConfig(feed_fc_ekf=True, min_confidence_to_feed=0.50)
        monitor = HealthMonitor(config=cfg)

        t_now = time.monotonic()
        status = monitor.evaluate_health(
            confidence=0.25,  # below 0.50
            innovation_chi2=1.0,
            observation_timestamp_s=t_now - 0.02,
            current_timestamp_s=t_now,
        )

        assert status.is_healthy is False
        assert status.can_feed_fc is False
        assert status.confidence_ok is False
        assert status.covariance_scale >= 5.0
        assert "LOW_CONF" in status.status_message

    def test_stale_data_fails_freshness_gate(self) -> None:
        """Verify observation older than max_data_age_ms fails freshness gate."""
        monitor = HealthMonitor(max_data_age_ms=100.0)

        t_now = time.monotonic()
        status = monitor.evaluate_health(
            confidence=0.90,
            innovation_chi2=1.0,
            observation_timestamp_s=t_now - 0.25,  # 250 ms old
            current_timestamp_s=t_now,
        )

        assert status.freshness_ok is False
        assert status.is_healthy is False
        assert "STALE_DATA" in status.status_message


class TestOdometrySender:
    """Test suite for MAVLink ODOMETRY and VISION message streaming."""

    def test_odometry_message_packing(self) -> None:
        """Verify ODOMETRY packet contains correct frame IDs, pose, and twist."""
        sink = MockMAVLinkSink()
        cfg = MAVLinkConfig(feed_fc_ekf=True, odometry_mode="ODOMETRY")
        sender = OdometrySender(mav_connection=sink, config=cfg)  # type: ignore

        # Construct dummy state
        state = NavState(
            timestamp_s=10.0,
            pos_ned_m=np.array([125.5, -45.2, -50.0]),
            vel_ned_mps=np.array([15.0, 2.5, -0.5]),
            quat_wb=np.array([1.0, 0.0, 0.0, 0.0]),
            accel_bias_mps2=np.zeros(3),
            gyro_bias_rad_s=np.array([0.01, -0.02, 0.05]),
        )
        P = np.eye(15) * 0.1

        sender.update_from_ekf(
            nav_state=state,
            P_15x15=P,
            confidence=0.95,
            frame_timestamp_s=time.monotonic(),
        )

        sent = sender.send_current_packet()
        assert sent is True
        assert len(sink.sent_odometry) == 1

        pkt = sink.sent_odometry[0]
        assert pkt["frame_id"] == mavutil.mavlink.MAV_FRAME_LOCAL_NED
        assert pkt["child_frame_id"] == mavutil.mavlink.MAV_FRAME_BODY_NED
        assert pkt["x"] == pytest.approx(125.5)
        assert pkt["y"] == pytest.approx(-45.2)
        assert pkt["z"] == pytest.approx(-50.0)
        assert pkt["vx"] == pytest.approx(15.0)
        assert pkt["vy"] == pytest.approx(2.5)
        assert pkt["vz"] == pytest.approx(-0.5)
        assert pkt["q"] == [1.0, 0.0, 0.0, 0.0]
        assert len(pkt["pose_covariance"]) == 21
        assert len(pkt["velocity_covariance"]) == 21
        assert pkt["estimator_type"] == mavutil.mavlink.MAV_ESTIMATOR_TYPE_VISION

    def test_vision_position_estimate_fallback(self) -> None:
        """Verify VISION_POSITION_ESTIMATE mode sends legacy messages."""
        sink = MockMAVLinkSink()
        cfg = MAVLinkConfig(feed_fc_ekf=True, odometry_mode="VISION_POSITION_ESTIMATE")
        sender = OdometrySender(mav_connection=sink, config=cfg)  # type: ignore

        state = NavState(
            timestamp_s=1.0,
            pos_ned_m=np.array([10.0, 20.0, -30.0]),
            vel_ned_mps=np.array([1.0, 2.0, 0.0]),
            quat_wb=np.array([1.0, 0.0, 0.0, 0.0]),
            accel_bias_mps2=np.zeros(3),
            gyro_bias_rad_s=np.zeros(3),
        )
        sender.update_from_ekf(state, np.eye(15) * 0.2, confidence=0.9)
        sent = sender.send_current_packet()

        assert sent is True
        assert len(sink.sent_vision_pos) == 1
        pkt = sink.sent_vision_pos[0]
        assert pkt["x"] == pytest.approx(10.0)
        assert pkt["y"] == pytest.approx(20.0)
        assert pkt["z"] == pytest.approx(-30.0)

    def test_gps_input_emulation(self) -> None:
        """Verify GPS_INPUT mode emits valid virtual GNSS messages."""
        sink = MockMAVLinkSink()
        cfg = MAVLinkConfig(feed_fc_ekf=True, odometry_mode="GPS_INPUT")
        sender = OdometrySender(mav_connection=sink, config=cfg)  # type: ignore

        state = NavState(
            timestamp_s=1.0,
            pos_ned_m=np.array([500.0, -200.0, -100.0]),
            vel_ned_mps=np.array([10.0, 0.0, 0.0]),
            quat_wb=np.array([1.0, 0.0, 0.0, 0.0]),
            accel_bias_mps2=np.zeros(3),
            gyro_bias_rad_s=np.zeros(3),
        )
        sender.update_from_ekf(state, np.eye(15) * 0.5, confidence=0.85)
        sent = sender.send_current_packet()

        assert sent is True
        assert len(sink.sent_gps_input) == 1
        pkt = sink.sent_gps_input[0]
        assert pkt["fix_type"] == 3  # 3D Fix
        assert pkt["satellites_visible"] >= 10
        assert pkt["lat"] > 0
        assert pkt["lon"] > 0

    def test_background_thread_streaming_rate(self) -> None:
        """Verify background streaming thread dispatches at ~25 Hz."""
        sink = MockMAVLinkSink()
        cfg = MAVLinkConfig(feed_fc_ekf=True, odometry_mode="ODOMETRY")
        sender = OdometrySender(mav_connection=sink, config=cfg, rate_hz=25.0)  # type: ignore

        state = NavState(
            timestamp_s=1.0,
            pos_ned_m=np.array([0.0, 0.0, -10.0]),
            vel_ned_mps=np.zeros(3),
            quat_wb=np.array([1.0, 0.0, 0.0, 0.0]),
            accel_bias_mps2=np.zeros(3),
            gyro_bias_rad_s=np.zeros(3),
        )
        sender.update_from_ekf(state, np.eye(15) * 0.1, confidence=0.9)

        sender.start()
        time.sleep(0.30)  # 300 ms -> should send ~7-8 packets at 25 Hz
        sender.stop()

        assert sender.packets_sent >= 5
        assert len(sink.sent_odometry) >= 5


class TestArduPilotParameters:
    """Verify validity of ArduPilot 4.5.7 configuration file."""

    PARAM_FILE = Path("config/ardupilot_vps_params.param")

    def test_parameter_file_exists_and_valid(self) -> None:
        """Verify parameter file contains required ExternalNav parameters."""
        assert self.PARAM_FILE.is_file()
        content = self.PARAM_FILE.read_text(encoding="utf-8")

        required_params = [
            "SERIAL2_PROTOCOL",
            "SERIAL2_BAUD",
            "EK3_ENABLE",
            "EK3_SRC1_POSXY",
            "EK3_SRC1_VELXY",
            "EK3_SRC1_POSZ",
            "VISO_TYPE",
            "RNGFND1_TYPE",
            "RNGFND1_MAX_CM",
        ]

        for p in required_params:
            assert p in content, f"Missing required parameter {p} in {self.PARAM_FILE}"

