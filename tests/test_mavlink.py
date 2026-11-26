"""Unit tests for MAVLink telemetry ring buffer, DISTANCE_SENSOR sender, and SITL integration."""

import time
import numpy as np
import pytest

from vps.config import MAVLinkConfig
from vps.mavlink.telemetry_buffer import TelemetryRingBuffer, TelemetrySample
from vps.tools.sitl_harness import MockSITLServer, run_sitl_integration_test


class TestTelemetryBuffer:
    def test_buffer_interpolation_linear(self):
        buf = TelemetryRingBuffer(max_duration_s=5.0, time_offset_s=0.0)

        # Insert sample at t=1.0 and t=2.0
        s0 = TelemetrySample(
            timestamp_s=1.0,
            gyro_rad_s=np.array([0.1, 0.2, 0.3]),
            attitude_rpy_rad=np.array([0.0, 0.0, 0.0]),
            vel_ned_mps=np.array([10.0, 0.0, 0.0]),
            baro_alt_m=50.0,
            pos_ned_m=np.array([0.0, 0.0, -50.0]),
        )
        s1 = TelemetrySample(
            timestamp_s=2.0,
            gyro_rad_s=np.array([0.3, 0.4, 0.5]),
            attitude_rpy_rad=np.array([0.2, -0.1, 0.4]),
            vel_ned_mps=np.array([12.0, 2.0, 0.0]),
            baro_alt_m=60.0,
            pos_ned_m=np.array([11.0, 1.0, -60.0]),
        )
        buf.add_sample(s0)
        buf.add_sample(s1)

        # Interpolate at midpoint t=1.5
        interp = buf.interpolate(1.5)
        assert interp is not None
        assert abs(interp.timestamp_s - 1.5) < 1e-6
        assert abs(interp.baro_alt_m - 55.0) < 1e-5
        np.testing.assert_allclose(interp.gyro_rad_s, [0.2, 0.3, 0.4], atol=1e-5)
        np.testing.assert_allclose(interp.vel_ned_mps, [11.0, 1.0, 0.0], atol=1e-5)
        np.testing.assert_allclose(interp.attitude_rpy_rad, [0.1, -0.05, 0.2], atol=1e-5)

    def test_buffer_yaw_wrap_around(self):
        """Attitude interpolation should handle -pi to +pi wrap-around cleanly."""
        buf = TelemetryRingBuffer(max_duration_s=5.0)
        # s0 yaw is +3.10 rad (~177 deg), s1 yaw is -3.10 rad (~-177 deg)
        s0 = TelemetrySample(
            timestamp_s=0.0,
            gyro_rad_s=np.zeros(3),
            attitude_rpy_rad=np.array([0.0, 0.0, 3.10]),
            vel_ned_mps=np.zeros(3),
            baro_alt_m=10.0,
        )
        s1 = TelemetrySample(
            timestamp_s=1.0,
            gyro_rad_s=np.zeros(3),
            attitude_rpy_rad=np.array([0.0, 0.0, -3.10]),
            vel_ned_mps=np.zeros(3),
            baro_alt_m=10.0,
        )
        buf.add_sample(s0)
        buf.add_sample(s1)

        interp = buf.interpolate(0.5)
        assert interp is not None
        # Shortest angular path wraps through pi, so midpoint should be close to pi or -pi
        assert abs(abs(interp.attitude_rpy_rad[2]) - np.pi) < 0.15

    def test_buffer_pruning(self):
        buf = TelemetryRingBuffer(max_duration_s=1.0)
        # Add samples spanning 2.5 seconds
        for i in range(25):
            t = i * 0.1
            buf.add_sample(TelemetrySample(
                timestamp_s=t,
                gyro_rad_s=np.zeros(3),
                attitude_rpy_rad=np.zeros(3),
                vel_ned_mps=np.zeros(3),
                baro_alt_m=10.0,
            ))
        # Total duration in buffer should not exceed ~1.0 s
        with buf._lock:
            span = buf._samples[-1].timestamp_s - buf._samples[0].timestamp_s
            assert span <= 1.1


def _can_bind_loopback() -> bool:
    """Test if local UDP loopback is permitted in current sandbox/OS environment."""
    import socket
    try:
        s1 = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s1.bind(("127.0.0.1", 0))
        s1.settimeout(0.2)
        port = s1.getsockname()[1]
        s2 = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s2.sendto(b"ping", ("127.0.0.1", port))
        data, _ = s1.recvfrom(32)
        s1.close()
        s2.close()
        return data == b"ping"
    except Exception:
        return False


class TestSITLIntegration:
    @pytest.mark.skipif(not _can_bind_loopback(), reason="UDP loopback restricted by sandbox environment")
    def test_sitl_stream_reception(self, tmp_path):
        plot_path = tmp_path / "test_sitl_plot.png"
        res = run_sitl_integration_test(
            duration_s=2.0,
            output_plot_path=plot_path,
            client_port=14552,
            server_port=14561,
        )

        assert res["is_valid"]
        # Must stream at >= 10 Hz
        assert res["rate_hz"] >= 10.0
        assert res["packets_received"] >= 15
        assert res["ring_buffer_samples"] >= 50
        assert plot_path.is_file()


def test_mavlink_bridge_vfr_hud_updates_baro_even_with_local_pos_ned():
    """Verify VFR_HUD baro stays live when LOCAL_POSITION_NED is active without GLOBAL_POSITION_INT."""
    from types import SimpleNamespace
    from vps.mavlink.connection import MAVLinkBridge

    bridge = MAVLinkBridge(mode="shadow")

    class FakeMav:
        def __init__(self, msgs):
            self._msgs = list(msgs)

        def recv_match(self, blocking=True, timeout=0.1):
            if self._msgs:
                return self._msgs.pop(0)
            bridge._running = False
            return None

    class Msg:
        def __init__(self, mtype, **kw):
            self._t = mtype
            for k, v in kw.items():
                setattr(self, k, v)

        def get_type(self):
            return self._t

    bridge.mav = FakeMav([
        Msg("LOCAL_POSITION_NED", x=1.0, y=2.0, z=-5.0, vx=0.5, vy=0.0, vz=0.0),
        Msg("VFR_HUD", alt=540.0),
        Msg("VFR_HUD", alt=546.5),
    ])
    bridge._running = True
    bridge._recv_worker()

    assert bridge._has_local_pos_ned is True
    assert abs(bridge._latest_baro - 6.5) < 1e-6

