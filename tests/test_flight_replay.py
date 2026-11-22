"""Unit tests for real ArduPilot DataFlash (.bin) flight log parser and replayer."""

from pathlib import Path
import numpy as np
import pytest

from vps.tools.flight_log_replayer import FlightLogParser, FlightLogReplayer


class TestFlightLogParser:
    LOG_PATH = Path("data/flight_logs/log_19_2026-11-11-18-02-08.bin")

    def test_parse_real_bin_log(self):
        """Verify real binary log parsing extracts valid 6-DOF kinematics."""
        if not self.LOG_PATH.is_file():
            pytest.skip(f"Flight log {self.LOG_PATH} not found")

        parser = FlightLogParser()
        traj = parser.parse_bin_log(self.LOG_PATH, start_time_s=10.0, duration_s=15.0)

        assert len(traj.times_s) > 50
        assert traj.times_s[0] == 0.0
        assert traj.times_s[-1] > 10.0

        # Check shapes
        n = len(traj.times_s)
        assert traj.pos_ned_m.shape == (n, 3)
        assert traj.vel_ned_mps.shape == (n, 3)
        assert traj.attitude_rpy_rad.shape == (n, 3)
        assert traj.gyro_rad_s.shape == (n, 3)
        assert len(traj.baro_alt_m) == n
        assert len(traj.rel_alt_m) == n

        # Check values are realistic
        assert not np.isnan(traj.vel_ned_mps).any()
        assert not np.isnan(traj.attitude_rpy_rad).any()
        assert np.max(np.abs(traj.attitude_rpy_rad[:, :2])) < np.radians(45.0)

    def test_time_slice_filtering(self):
        """Verify time slice offset and duration boundaries."""
        if not self.LOG_PATH.is_file():
            pytest.skip(f"Flight log {self.LOG_PATH} not found")

        parser = FlightLogParser()
        traj = parser.parse_bin_log(self.LOG_PATH, start_time_s=20.0, duration_s=10.0)

        span = traj.times_s[-1] - traj.times_s[0]
        assert 8.0 <= span <= 12.0


class TestFlightLogReplayer:
    LOG_PATH = Path("data/flight_logs/log_19_2026-11-11-18-02-08.bin")

    def test_replayer_on_flight_slice(self):
        """Verify end-to-end replay on real flight slice meets accuracy requirements."""
        if not self.LOG_PATH.is_file():
            pytest.skip(f"Flight log {self.LOG_PATH} not found")

        parser = FlightLogParser()
        traj = parser.parse_bin_log(self.LOG_PATH, start_time_s=15.0, duration_s=15.0)

        replayer = FlightLogReplayer()
        res = replayer.replay_trajectory(trajectory=traj, target_fps=10.0, max_samples=20)

        assert len(res["times_s"]) == 20
        assert not np.isnan(res["kalman_agl_m"]).any()
        assert (res["sigma_agl_m"] > 0).all()

        # Mean Absolute Error on textured flight slice should be < 0.5 m
        assert res["mae_m"] < 1.0, f"MAE {res['mae_m']:.2f}m exceeds 1.0m limit!"
        assert res["mean_error_pct"] < 5.0, f"Error {res['mean_error_pct']:.2f}% exceeds 5% limit!"
