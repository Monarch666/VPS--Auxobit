"""Unit tests for 2-state Kalman Filter and sensor fusion engine."""

from pathlib import Path
import numpy as np
import pytest

from vps.config import FusionConfig
from vps.fusion.confidence import ConfidenceEvaluator
from vps.fusion.kalman_filter import TerrainRelativeKalmanFilter
from vps.tools.m6_fusion_eval import run_m6_stress_tests


class TestKalmanFilter:
    def test_predict_kinematics(self):
        kf = TerrainRelativeKalmanFilter(initial_height_m=50.0, initial_terrain_m=0.0)
        # Climb rate: vel_d = -2.0 m/s (climbing at 2 m/s for 1.0 s)
        kf.predict(dt_s=1.0, vel_d_mps=-2.0, is_hover=False)

        assert abs(kf.x[0] - 52.0) < 1e-5  # height increased to 52.0 m
        assert abs(kf.x[1] - 0.0) < 1e-5   # terrain held at 0.0 m
        # Covariance increased due to process noise
        assert kf.P[0, 0] > 1.0

    def test_baro_update(self):
        kf = TerrainRelativeKalmanFilter(initial_height_m=50.0, initial_terrain_m=0.0)
        # Baro reports 52.0 m
        kf.update_barometer(baro_alt_m=52.0)
        # Both h_agl and z_terrain should update
        assert kf.x[0] > 50.0
        assert kf.x[1] > 0.0

    def test_vision_update_with_confidence_scaling(self):
        kf = TerrainRelativeKalmanFilter(initial_height_m=50.0, initial_terrain_m=0.0)

        # High confidence update (0.95)
        P_before = kf.P[0, 0]
        accepted = kf.update_vision_height(vision_height_m=50.5, confidence=0.95)
        assert accepted
        assert kf.P[0, 0] < P_before  # covariance reduced

        # Crazy outlier rejection (e.g. 500m when current estimate is 50m)
        rejected = kf.update_vision_height(vision_height_m=500.0, confidence=0.95)
        assert not rejected
        assert "GATED_OUTLIER" in kf._mode_status

    def test_hover_terrain_hold_and_covariance_inflation(self):
        kf_hover = TerrainRelativeKalmanFilter(initial_height_m=50.0, initial_terrain_m=10.0)
        kf_cruise = TerrainRelativeKalmanFilter(initial_height_m=50.0, initial_terrain_m=10.0)

        # In hover without vision: terrain is held
        terrain_init = kf_hover.x[1]

        for _ in range(10):
            kf_hover.predict(dt_s=0.1, vel_d_mps=0.0, is_hover=True)
            kf_cruise.predict(dt_s=0.1, vel_d_mps=0.0, is_hover=False)

        assert abs(kf_hover.x[1] - terrain_init) < 1e-5
        # Terrain covariance inflated significantly more in hover than cruise (10x process noise)
        delta_p_hover = kf_hover.P[1, 1] - 1.0
        delta_p_cruise = kf_cruise.P[1, 1] - 1.0
        assert delta_p_hover > delta_p_cruise * 8.0


class TestConfidenceEvaluator:
    def test_hover_speed_penalization(self):
        evaluator = ConfidenceEvaluator(hover_speed_thresh_mps=0.5)
        res = evaluator.evaluate(
            num_inliers=100,
            inlier_ratio=0.85,
            median_flow_px=12.0,
            ground_speed_mps=0.2,  # hover
            angular_rate_rad_s=0.05,
        )
        assert not res.is_acceptable
        assert res.c_ground_speed == 0.0
        assert res.status_code == "HOVER_DEGENERATE"

    def test_textureless_surface_penalization(self):
        evaluator = ConfidenceEvaluator()
        # Uniform image with near-zero contrast
        blank_patch = np.full((100, 100), 128, dtype=np.uint8)
        res = evaluator.evaluate(
            num_inliers=5,
            inlier_ratio=0.20,
            median_flow_px=1.0,
            ground_speed_mps=5.0,
            angular_rate_rad_s=0.05,
            image_gray=blank_patch,
        )
        assert not res.is_acceptable
        assert res.c_texture_contrast == 0.0


class TestFusionEngineSimulation:
    def test_m6_stress_tests_execution(self, tmp_path):
        plot_path = tmp_path / "test_m6_stress.png"
        res = run_m6_stress_tests(output_plot_path=plot_path)

        assert res["is_valid"]
        assert res["mean_err_pct"] <= 5.0
        assert res["hover_detected"]
        assert res["textureless_fallback"]
        assert plot_path.is_file()


def test_kalman_ground_reset_and_terrain_bound_on_baro_drift():
    """Verify ground reset prevents pre-arm baro drift from poisoning z_terrain at takeoff."""
    kf = TerrainRelativeKalmanFilter(initial_height_m=0.06, initial_terrain_m=0.0, baseline_ground_offset_m=0.06)

    # Simulate 100s on ground with -1.6m baro drift
    for _ in range(100):
        kf.predict(dt_s=1.0, vel_d_mps=0.0, is_hover=True)
        kf.update_barometer(baro_alt_m=-1.6)
        assert kf.x[0] >= kf.baseline_offset
        assert kf.P[1, 1] <= 2.0 + 1e-9

    # Ground reset on pad cleanly restores initial ground state & tight covariance
    kf.reset_ground_state()
    assert abs(kf.x[0] - 0.06) < 1e-6
    assert abs(kf.x[1]) < 1e-6
    assert kf.P[1, 1] <= 0.25 + 1e-9
