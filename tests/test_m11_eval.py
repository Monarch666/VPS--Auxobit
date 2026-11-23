"""Unit tests for Milestone 11 Real-Flight Log Replay and GPS/DEM Ground-Truth Benchmarking."""

from pathlib import Path
import numpy as np
import pytest

from vps.tools.aruco_validator import ArUcoValidator
from vps.map_db.runtime_lookup import RuntimeMapDB
from vps.tools.m11_flight_report_generator import (
    evaluate_aruco_low_altitude,
    replay_real_flight_log,
    evaluate_cruise_map_matching,
    run_m11_evaluation,
)


class TestMilestone11Evaluation:
    @pytest.fixture
    def map_db(self):
        db_path = Path("data/map_db/aurangabad_s2")
        if not db_path.is_dir() or not (db_path / "manifest.json").is_file():
            db_path = Path("data/map_db/aurangabad_full")
        return RuntimeMapDB(db_directory=db_path, max_cached_tiles=4)

    def test_aruco_low_altitude_precision(self):
        """Verify ArUco achieves sub-3.5 cm MAE at <= 5.0m and exact touchdown (0.00 m)."""
        validator = ArUcoValidator()
        res = evaluate_aruco_low_altitude(validator, altitudes_m=[0.15, 0.50, 1.0, 2.0, 3.0, 5.0])

        assert res["detection_rate_pct"] == 100.0
        assert res["mae_low_alt_m"] < 0.035, f"Low altitude MAE {res['mae_low_alt_m']*100:.2f} cm exceeds 3.5 cm limit!"
        assert res["touchdown_err_m"] < 0.005, f"Touchdown error {res['touchdown_err_m']*100:.2f} cm exceeds 0.5 cm limit!"
        assert res["passed"] is True

    def test_real_flight_log_replay_short(self):
        """Verify real ArduPilot flight log replay produces valid kinematic tracking with zero NaNs."""
        log_path = Path("data/flight_logs/log_19_2026-11-11-18-02-08.bin")
        if not log_path.is_file():
            pytest.skip(f"Flight log {log_path} not found.")

        res = replay_real_flight_log(log_path, start_time_s=60.0, duration_s=15.0)

        assert res["passed"] is True
        assert len(res["times_s"]) > 100
        assert not np.isnan(res["vio_pos_ned"]).any()
        assert not np.isnan(res["vio_vel_ned"]).any()
        assert res["rmse_vel_ned"][0] < 1.0
        assert res["rmse_vel_ned"][1] < 1.0
        assert res["rmse_alt_m"] < 2.5
        assert res["healthy_rate_pct"] > 90.0

    def test_cruise_map_matching_delayed_ekf(self, map_db):
        """Verify cruise navigation map matching bounds drift within 2x GSD and latency < 100 ms."""
        res = evaluate_cruise_map_matching(map_db, duration_s=10.0, dt_sim=0.01)

        gsd = map_db.manifest.gsd_meters_per_px
        assert res["passed"] is True
        assert len(res["map_fixes"]) >= 2
        assert res["rmse_pos_horiz"] <= (2.0 * gsd)
        assert res["mean_latency_ms"] < 100.0
        assert res["drift_pct_vps"] < res["drift_pct_imu"]
