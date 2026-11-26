"""Unit tests for Milestone 7: VIO Velocity Estimator and 15-State Error-State EKF."""

import numpy as np
import pytest

from vps.config import CameraConfig
from vps.vio.velocity_estimator import VIOVelocityEstimator, VIOVelocityResult
from vps.nav_ekf.es_ekf import ErrorStateEKF, ESEKFConfig, NavState


class TestVIOVelocityEstimator:
    def setup_method(self):
        self.cam_cfg = CameraConfig()
        self.estimator = VIOVelocityEstimator(camera_config=self.cam_cfg)

    def test_forward_flight_metric_velocity(self):
        """Forward flight at 10 m/s North at 50 m AGL should produce ~10 m/s Vn."""
        height = 50.0
        dt = 0.033333  # ~30 fps
        speed_north = 10.0  # m/s

        # Moving North (+X_w) means in camera frame moving along -Y_c
        # Ground features shift in opposite direction (+Y_c / +dv):
        # dv = fy * ground_disp / height = fy * (speed_north * dt) / height
        expected_dv = (self.cam_cfg.fy * (speed_north * dt)) / height

        # Synthesize 50 feature points
        np.random.seed(42)
        p0 = np.random.uniform([200, 200], [1200, 800], size=(50, 2))
        p1 = p0.copy()
        p1[:, 1] += expected_dv  # Shift along +v

        inliers = np.ones(50, dtype=bool)
        H_eye = np.eye(3)
        att = np.array([0.0, 0.0, 0.0])  # zero roll/pitch/yaw

        res = self.estimator.estimate_velocity_from_flow(
            pts0=p0,
            pts1=p1,
            inlier_mask=inliers,
            H_derot=H_eye,
            height_agl_m=height,
            dt_s=dt,
            attitude_rpy_rad=att,
            timestamp_s=1.0,
            compute_homography_alternative=False,
        )

        assert abs(res.vel_ned_mps[0] - speed_north) < 0.05
        assert abs(res.vel_ned_mps[1]) < 0.05
        assert abs(res.vel_ned_mps[2]) < 0.05
        assert abs(res.ground_speed_mps - speed_north) < 0.05
        assert res.confidence > 0.8

    def test_rotation_invariance_with_derotation(self):
        """Attitude perturbation compensated by H_derot should leave velocity unchanged."""
        height = 100.0
        dt = 0.033333
        speed_east = 15.0  # 15 m/s East (+Y_w -> +X_c in camera frame)

        # Expected feature shift along -u
        expected_du = -(self.cam_cfg.fx * (speed_east * dt)) / height

        np.random.seed(101)
        p0 = np.random.uniform([300, 300], [1100, 700], size=(60, 2))

        # Add pure translation
        p1_trans = p0.copy()
        p1_trans[:, 0] += expected_du

        # Add pure camera pitch rotation of 2.0 degrees
        pitch_rad = np.radians(2.0)
        cp, sp = np.cos(pitch_rad), np.sin(pitch_rad)
        R_pitch = np.array([
            [1.0, 0.0, 0.0],
            [0.0,  cp,  sp],
            [0.0, -sp,  cp]
        ])
        H_rot = self.cam_cfg.K @ R_pitch @ np.linalg.inv(self.cam_cfg.K)

        # Rotate points to simulate perturbed frame
        p0_hom = np.column_stack([p1_trans, np.ones(60)])
        p1_hom = (H_rot @ p0_hom.T).T
        p1 = p1_hom[:, :2] / p1_hom[:, 2:3]

        inliers = np.ones(60, dtype=bool)
        att = np.array([0.0, pitch_rad, 0.0])

        res = self.estimator.estimate_velocity_from_flow(
            pts0=p0,
            pts1=p1,
            inlier_mask=inliers,
            H_derot=H_rot,
            height_agl_m=height,
            dt_s=dt,
            attitude_rpy_rad=att,
            timestamp_s=2.0,
            compute_homography_alternative=False,
        )

        assert abs(res.vel_ned_mps[1] - speed_east) < 0.25

    def test_radial_climb_decoupling_on_offcenter_features(self):
        """Pure vertical climb with off-center features must not hallucinate horizontal velocity."""
        height = 10.0
        dt = 0.1
        scale = 10.0 / 10.8  # Climbing cleanly upward

        u, v = np.meshgrid(np.linspace(150, 1250, 12), np.linspace(450, 980, 8))
        p0 = np.column_stack([u.ravel(), v.ravel()])
        center = np.array([self.cam_cfg.cx, self.cam_cfg.cy])
        p1 = center + scale * (p0 - center)

        res = self.estimator.estimate_velocity_from_flow(
            pts0=p0,
            pts1=p1,
            inlier_mask=np.ones(len(p0), dtype=bool),
            H_derot=np.eye(3),
            height_agl_m=height,
            dt_s=dt,
            attitude_rpy_rad=np.zeros(3),
            timestamp_s=3.0,
            compute_homography_alternative=False,
        )
        assert float(np.hypot(res.vel_ned_mps[0], res.vel_ned_mps[1])) < 0.15


class TestESEKF:
    def setup_method(self):
        self.cfg = ESEKFConfig()
        self.ekf = ErrorStateEKF(config=self.cfg, initial_pos_ned=np.zeros(3), initial_vel_ned=np.zeros(3))

    def test_skew_symmetric(self):
        v = np.array([1.0, 2.0, 3.0])
        w = np.array([4.0, 5.0, 6.0])
        cross_expected = np.cross(v, w)
        cross_skew = ErrorStateEKF.skew_symmetric(v) @ w
        np.testing.assert_allclose(cross_expected, cross_skew, atol=1e-12)

    def test_imu_kinematic_propagation(self):
        """Constant 2 m/s^2 forward acceleration in body frame should propagate velocity and position."""
        dt = 0.01  # 100 Hz
        a_fwd = 2.0
        g = 9.80665

        # Accelerometer measures specific force: f_b = [a_fwd, 0, -g]
        accel_meas = np.array([a_fwd, 0.0, -g])
        gyro_meas = np.zeros(3)

        # Propagate for 100 steps (1.0 second)
        for i in range(100):
            t = (i + 1) * dt
            state = self.ekf.predict_imu(accel_meas, gyro_meas, timestamp_s=t)

        # Expected velocity = a * t = 2.0 m/s North
        assert abs(state.vel_ned_mps[0] - 2.0) < 0.05
        # Expected position = 0.5 * a * t^2 = 1.0 m North
        assert abs(state.pos_ned_m[0] - 1.0) < 0.05
        # Vertical and East should be zero
        assert abs(state.pos_ned_m[1]) < 0.01
        assert abs(state.pos_ned_m[2]) < 0.01

    def test_compass_yaw_correction(self):
        """Compass update should correct heading misalignment."""
        # Initialize with zero heading
        ekf = ErrorStateEKF(initial_yaw_rad=0.0)
        assert abs(ekf.state.roll_pitch_yaw_rad[2]) < 0.01

        # Compass reports 45 deg (pi/4 rad)
        true_yaw = np.pi / 4.0
        # Multiple updates simulate 10 Hz compass stream
        for _ in range(5):
            accepted = ekf.update_compass_yaw(yaw_compass_rad=true_yaw)
            assert accepted

        # Estimated heading should converge towards true_yaw
        est_yaw = ekf.state.roll_pitch_yaw_rad[2]
        assert abs(est_yaw - true_yaw) < 0.05

    def test_vio_velocity_fusion_and_drift_bounding(self):
        """Fusing VIO velocity should correct velocity error and reduce covariance."""
        ekf = ErrorStateEKF(initial_pos_ned=np.zeros(3), initial_vel_ned=np.array([10.0, 0.0, 0.0]))

        # Initial velocity variance
        init_vel_cov = np.trace(ekf.P[3:6, 3:6])

        # True velocity is 12 m/s North
        v_vio = np.array([12.0, 0.0, 0.0])
        accepted = ekf.update_vio_velocity(vel_vio_ned_mps=v_vio, confidence=0.95)
        assert accepted

        # Velocity state should move towards 12 m/s
        assert ekf.state.vel_ned_mps[0] > 11.5
        # Covariance should have decreased
        post_vel_cov = np.trace(ekf.P[3:6, 3:6])
        assert post_vel_cov < init_vel_cov

    def test_outlier_rejection_via_chi2_gating(self):
        """Outlier velocity (> 100 m/s jump) should be rejected by chi-squared gate."""
        ekf = ErrorStateEKF(initial_pos_ned=np.zeros(3), initial_vel_ned=np.zeros(3))
        v_outlier = np.array([150.0, -80.0, 50.0])

        accepted = ekf.update_vio_velocity(vel_vio_ned_mps=v_outlier, confidence=1.0)
        assert not accepted
        # State should not be corrupted
        assert np.linalg.norm(ekf.state.vel_ned_mps) < 0.1

    def test_esekf_substep_low_fps_and_ground_reset_and_ahrs(self):
        """Low-rate predict_imu sub-steps cleanly; ground reset & AHRS leveling bound drift."""
        ekf = ErrorStateEKF(initial_pos_ned=np.zeros(3), initial_vel_ned=np.array([4.0, 0.0, 0.0]))
        g = 9.80665
        # Propagate 1.0 s at 10 Hz (0.10 s steps > old 0.05 s clip)
        ekf.predict_imu(np.array([0.0, 0.0, -g]), np.zeros(3), timestamp_s=0.0)
        for i in range(1, 11):
            ekf.predict_imu(np.array([0.0, 0.0, -g]), np.zeros(3), timestamp_s=i * 0.1)
        assert abs(ekf.state.pos_ned_m[0] - 4.0) < 0.2

        # AHRS attitude update brings tilted state back toward level
        ekf.align_attitude(0.15, -0.12, 0.0, sigma_att_rad=0.1)
        for _ in range(10):
            assert ekf.update_ahrs_attitude(0.0, 0.0, sigma_rp_rad=0.05)
        rpy = ekf.state.roll_pitch_yaw_rad
        assert abs(rpy[0]) < 0.03 and abs(rpy[1]) < 0.03

        # Ground reset restores clean position, zero velocity, and clears cross-covariance
        ekf.reset_ground_state(np.array([0.0, 0.0, -0.06]), attitude_rpy_rad=np.zeros(3))
        assert np.allclose(ekf.state.vel_ned_mps, 0.0)
        assert np.allclose(ekf.state.pos_ned_m, [0.0, 0.0, -0.06])
        assert np.allclose(ekf.P[0:3, 3:6], 0.0)

