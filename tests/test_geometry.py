"""Unit tests for geometry, pinhole model, de-rotation, and scale expansion."""

import math
import shutil
from pathlib import Path
import numpy as np
import pytest

from vps.config import CameraConfig, VPSConfig
from vps.geometry.pinhole import PinholeModel
from vps.geometry.derotation import GyroDerotationEngine
from vps.geometry.height_flow import HeightFromFlowEngine, FlowHeightResult
from vps.geometry.scale_expansion import ScaleFromExpansionEngine, ScaleExpansionResult
from vps.tools.logger import TelemetryLogger, TelemetryRecord
from vps.tools.replayer import LogReplayer


@pytest.fixture
def default_pinhole():
    K = np.array([
        [1207.2, 0.0, 728.0],
        [0.0, 1199.1, 544.0],
        [0.0, 0.0, 1.0]
    ], dtype=np.float64)
    D = np.array([-0.042, 0.015, 0.0008, -0.0004, 0.0], dtype=np.float64)
    return PinholeModel(K=K, D=D, width=1456, height=1088)


class TestPinholeModel:
    def test_pinhole_properties(self, default_pinhole):
        model = default_pinhole
        assert math.isclose(model.fx, 1207.2, rel_tol=1e-5)
        assert math.isclose(model.fy, 1199.1, rel_tol=1e-5)
        assert math.isclose(model.cx, 728.0, rel_tol=1e-5)
        assert math.isclose(model.cy, 544.0, rel_tol=1e-5)
        assert 60.0 < model.hfov_deg < 65.0
        assert 46.0 < model.vfov_deg < 52.0

    def test_gsd_calculation(self, default_pinhole):
        model = default_pinhole
        gsd_x, gsd_y = model.compute_gsd(altitude_m=300.0)
        # At 300 m, fx ~ 1207.2 -> GSD ~ 300 / 1207.2 ~ 0.2485 m/px (~25 cm/px)
        assert math.isclose(gsd_x, 300.0 / 1207.2, rel_tol=1e-4)
        assert math.isclose(gsd_y, 300.0 / 1199.1, rel_tol=1e-4)
        assert 0.24 < gsd_x < 0.26

    def test_project_and_backproject_invariance(self, default_pinhole):
        model = default_pinhole
        altitude = 150.0
        # Create a grid of ground points in camera frame at depth Z = altitude
        x_coords = np.linspace(-30.0, 30.0, 5)
        y_coords = np.linspace(-20.0, 20.0, 5)
        xx, yy = np.meshgrid(x_coords, y_coords)
        pts_3d = np.column_stack([xx.ravel(), yy.ravel(), np.full(xx.size, altitude)])

        pixels = model.project_3d_to_pixel(pts_3d)
        assert pixels.shape == (25, 2)

        # Backproject should exactly recover original 3D points
        recovered_3d = model.backproject_pixel_to_ground(pixels, altitude_m=altitude)
        np.testing.assert_allclose(recovered_3d, pts_3d, atol=1e-8)


class TestGyroDerotation:
    def test_pure_rotation_residual_zero(self, default_pinhole):
        """When drone only rotates, de-rotation must cancel all flow, yielding 0 residual."""
        engine = GyroDerotationEngine(K=default_pinhole.K)

        # 50 random features on image plane
        rng = np.random.default_rng(123)
        pts_k = rng.uniform(low=[100, 100], high=[1350, 980], size=(50, 2))

        # Rotate by 2.0 degrees pitch, -1.5 degrees roll, 3.0 degrees yaw
        R_cam = engine.rotation_matrix_from_rpy_delta(
            delta_roll_rad=np.radians(-1.5),
            delta_pitch_rad=np.radians(2.0),
            delta_yaw_rad=np.radians(3.0)
        )

        # Under pure rotation, pts_current is given by rotational homography
        pts_c = engine.derotate_points(pts_k, R_cam)

        # Compute translational flow with de-rotation
        _, flow_trans = engine.compute_translational_flow(pts_k, pts_c, R_cam)

        # All translational flow vectors must be exactly zero
        np.testing.assert_allclose(flow_trans, np.zeros_like(flow_trans), atol=1e-7)

    def test_gyro_integration_matches_euler(self, default_pinhole):
        engine = GyroDerotationEngine(K=default_pinhole.K)
        # Constant pitch rate of 0.1 rad/s over 0.2 seconds -> 0.02 rad pitch
        dt = 0.01
        timestamps = np.arange(0.0, 0.21, dt)
        gyro_samples = np.zeros((len(timestamps), 3))
        gyro_samples[:, 1] = 0.1  # wy = pitch rate

        R_integrated = engine.integrate_gyro_to_rotation(gyro_samples, timestamps)
        R_euler = engine.rotation_matrix_from_rpy_delta(0.0, 0.1 * 0.2, 0.0)

        np.testing.assert_allclose(R_integrated, R_euler, atol=1e-5)


class TestHeightFromFlow:
    def test_pure_translation_height_recovery(self):
        engine = HeightFromFlowEngine(fx=1200.0, fy=1200.0)
        true_h = 100.0
        # Forward translation: dx = 5.0 m, dy = 0.0 m (at 10 m/s for 0.5 s)
        ground_disp = np.array([5.0, 0.0])

        # Theoretical flow: du = fx * dx / H = 1200 * 5 / 100 = 60 px
        n_pts = 40
        rng = np.random.default_rng(42)
        noise = rng.normal(0.0, 0.2, size=(n_pts, 2))  # small 0.2 px noise
        flow_trans = np.column_stack([np.full(n_pts, 60.0), np.zeros(n_pts)]) + noise

        res = engine.estimate_height(flow_trans, ground_disp)
        assert res.is_valid
        assert math.isclose(res.height_m, true_h, rel_tol=0.01)
        assert res.confidence > 0.8
        assert res.inlier_ratio > 0.9

    def test_coupled_rotation_and_translation(self, default_pinhole):
        """Drone both translates and pitches. De-rotation isolates translation to recover height."""
        K = default_pinhole.K
        derot_engine = GyroDerotationEngine(K=K)
        flow_engine = HeightFromFlowEngine(fx=default_pinhole.fx, fy=default_pinhole.fy)

        true_h = 150.0
        ground_disp = np.array([4.5, 1.5])  # dx = 4.5m, dy = 1.5m

        # 3D points on ground
        rng = np.random.default_rng(999)
        pts_k = rng.uniform(low=[200, 200], high=[1200, 800], size=(60, 2))
        pts_3d_k = default_pinhole.backproject_pixel_to_ground(pts_k, altitude_m=true_h)

        # Motion: Camera pitches by 1.0 degree and translates
        R_cam = derot_engine.rotation_matrix_from_rpy_delta(0.0, np.radians(1.0), 0.0)
        # In camera frame: P_c = R_cam * (P_k - T)
        T_cam = np.array([ground_disp[0], ground_disp[1], 0.0])
        pts_3d_c = (R_cam @ (pts_3d_k - T_cam).T).T

        # Project to current frame
        pts_c = default_pinhole.project_3d_to_pixel(pts_3d_c)

        # Step 1: De-rotate
        _, flow_trans = derot_engine.compute_translational_flow(pts_k, pts_c, R_cam)

        # Step 2: Estimate height from isolated translational flow
        res = flow_engine.estimate_height(flow_trans, ground_disp)

        assert res.is_valid
        assert math.isclose(res.height_m, true_h, rel_tol=0.02)
        assert res.confidence > 0.8

    def test_hover_degenerate_case_handled(self):
        engine = HeightFromFlowEngine(fx=1200.0, fy=1200.0)
        # Near-zero ground displacement during hover (0.01 m)
        ground_disp = np.array([0.01, 0.005])
        flow = np.zeros((30, 2))
        res = engine.estimate_height(flow, ground_disp)
        assert not res.is_valid
        assert res.confidence == 0.0
        assert "Ground displacement too small" in res.status_message


class TestScaleFromExpansion:
    def test_descent_scale_expansion(self):
        engine = ScaleFromExpansionEngine()
        h1_true = 120.0
        h2_true = 100.0
        delta_baro = h1_true - h2_true  # 20.0 m (>0 for descent)

        # Features at h1 and h2
        rng = np.random.default_rng(77)
        pts1 = rng.uniform(low=[100, 100], high=[1300, 900], size=(60, 2))
        # Centroid
        c = np.array([728.0, 544.0])
        # Scaling factor: at h2 (lower), features expand: r2 = r1 * (h1 / h2) -> s1 / s2 = h2 / h1
        scale = h1_true / h2_true
        pts2 = c + (pts1 - c) * scale

        res = engine.estimate_height_from_expansion(pts1, pts2, delta_baro_m=delta_baro)
        assert res.is_valid
        assert math.isclose(res.height_h1_m, h1_true, rel_tol=0.01)
        assert math.isclose(res.height_h2_m, h2_true, rel_tol=0.01)
        assert math.isclose(res.scale_ratio, h2_true / h1_true, rel_tol=0.01)

    def test_climb_scale_contraction(self):
        engine = ScaleFromExpansionEngine()
        h1_true = 80.0
        h2_true = 110.0
        delta_baro = h1_true - h2_true  # -30.0 m (<0 for climb)

        rng = np.random.default_rng(88)
        pts1 = rng.uniform(low=[100, 100], high=[1300, 900], size=(60, 2))
        c = np.array([728.0, 544.0])
        # At h2 (higher), features contract: r2 = r1 * (h1 / h2)
        scale = h1_true / h2_true
        pts2 = c + (pts1 - c) * scale

        res = engine.estimate_height_from_expansion(pts1, pts2, delta_baro_m=delta_baro)
        assert res.is_valid
        assert math.isclose(res.height_h1_m, h1_true, rel_tol=0.01)
        assert math.isclose(res.height_h2_m, h2_true, rel_tol=0.01)

    def test_zero_baro_delta_handled(self):
        engine = ScaleFromExpansionEngine()
        pts = np.ones((40, 2)) * 100.0
        res = engine.estimate_height_from_expansion(pts, pts, delta_baro_m=0.1)
        assert not res.is_valid
        assert res.confidence == 0.0


class TestLoggerAndReplayer:
    def test_log_and_replay_cycle(self, tmp_path):
        session_name = "test_run_01"
        logger = TelemetryLogger(output_dir=tmp_path, session_name=session_name, save_images=False)

        # Write 5 records
        for i in range(5):
            rec = TelemetryRecord(
                timestamp_s=1000.0 + i * 0.033,
                frame_idx=i,
                roll_rad=0.01 * i,
                pitch_rad=-0.02 * i,
                yaw_rad=1.57,
                gyro_x_rad_s=0.05,
                gyro_y_rad_s=-0.03,
                gyro_z_rad_s=0.01,
                vel_n_mps=10.0,
                vel_e_mps=1.0,
                vel_d_mps=0.0,
                baro_alt_m=50.0 + i * 0.1,
                estimated_h_m=49.9 + i * 0.1,
                confidence=0.95,
                groundtruth_h_m=50.0 + i * 0.1,
            )
            logger.log(rec)
        logger.close()

        # Replay
        session_dir = tmp_path / session_name
        replayer = LogReplayer(session_dir=session_dir, load_images=False, playback_rate=0.0)
        assert len(replayer) == 5

        packets = list(replayer.iterate())
        assert len(packets) == 5
        assert packets[0].record.frame_idx == 0
        assert math.isclose(packets[4].record.baro_alt_m, 50.4, rel_tol=1e-5)
        assert math.isclose(packets[2].record.pitch_rad, -0.04, rel_tol=1e-5)
