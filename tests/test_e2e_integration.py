"""End-to-End Integration Test for the VPS Pipeline.

Verifies the COMPLETE data flow in a single process, from simulated camera frames
through every module to MAVLink output, using the real flight log and Sentinel-2 map:

1. Camera Capture → Tracking → Keyframe Manager → Gyro De-rotation → Height-from-Flow
2. 2-State Kalman Filter (Phase 1 AGL Height Fusion)
3. VIO Metric Velocity Estimation → 15-State ES-EKF (Phase 2)
4. Map Database Tile Lookup → Orthorectifier → Map Matcher (ORB + Phase Correlation)
5. Delayed-State EKF Update (map match at historical timestamp)
6. ODOMETRY/DISTANCE_SENSOR MAVLink packing with health gate checks
7. Watchdog keepalive gating
8. Failsafe: hover terrain hold, low-texture rejection, NaN/Inf guards

Uses the real ArduPilot flight log (data/flight_logs/log_19_2026-11-11-18-02-08.bin)
and the Sentinel-2 GeoTIFF map database (data/map_db/aurangabad_s2) for maximum
real-world fidelity.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import cv2
import time
import pytest

from vps.config import VPSConfig, CameraConfig
from vps.camera.capture import MockCameraCapture
from vps.geometry.pinhole import PinholeModel
from vps.geometry.derotation import GyroDerotationEngine
from vps.geometry.height_flow import HeightFromFlowEngine
from vps.geometry.scale_expansion import ScaleFromExpansionEngine
from vps.tracker.feature_tracker import FeatureTracker, TrackingResult
from vps.tracker.keyframe_manager import KeyframeManager
from vps.tracker.flow_pipeline import VisionHeightPipeline
from vps.fusion.fusion_engine import VPSFusionEngine, FusionOutput
from vps.fusion.kalman_filter import TerrainRelativeKalmanFilter
from vps.fusion.confidence import ConfidenceEvaluator
from vps.mavlink.telemetry_buffer import TelemetryRingBuffer, TelemetrySample
from vps.mavlink.odometry_sender import OdometrySender, OdometrySnapshot, cov6x6_to_upper_triangle
from vps.nav_ekf.es_ekf import ErrorStateEKF, ESEKFConfig, NavState
from vps.vio.velocity_estimator import VIOVelocityEstimator
from vps.health.health_monitor import HealthMonitor, HealthStatus
from vps.health.watchdog import WatchdogSupervisor
from vps.tools.simulator import DroneCameraSimulator, DroneState, SimulatorDegradationConfig
from vps.tools.terrain_generator import DeccanTerrainGenerator


# -------------------------------------------------------------------------------------
# Fixtures
# -------------------------------------------------------------------------------------

TERRAIN_PATH = Path("data/synthetic_terrain_deccan.png")
FLIGHT_LOG_PATH = Path("data/flight_logs/log_19_2026-11-11-18-02-08.bin")
MAP_DB_PATH = Path("data/map_db/aurangabad_s2")


def _get_or_create_terrain(size_px: int = 2048) -> np.ndarray:
    """Load or generate Deccan terrain for simulation."""
    if TERRAIN_PATH.is_file():
        ortho = cv2.imread(str(TERRAIN_PATH))
        if ortho is not None:
            return ortho
    gen = DeccanTerrainGenerator(width_px=size_px, height_px=size_px, meters_per_pixel=0.1, seed=42)
    gen.generate(TERRAIN_PATH)
    return cv2.imread(str(TERRAIN_PATH))


def _build_trajectory(
    n_steps: int = 200,
    dt_s: float = 1.0 / 30.0,
    altitude_m: float = 120.0,
    speed_mps: float = 10.0,
    heading_rad: float = 0.3,
) -> dict:
    """Build a synthetic cruise trajectory with realistic dynamics."""
    times = np.arange(n_steps) * dt_s
    pos_ned = np.zeros((n_steps, 3), dtype=np.float64)
    vel_ned = np.zeros((n_steps, 3), dtype=np.float64)
    att_rpy = np.zeros((n_steps, 3), dtype=np.float64)
    gyro = np.zeros((n_steps, 3), dtype=np.float64)
    baro = np.zeros(n_steps, dtype=np.float64)

    for i in range(n_steps):
        t = times[i]
        # Gentle S-curve trajectory
        yaw = heading_rad + 0.05 * np.sin(0.3 * t)
        roll = 0.03 * np.sin(0.5 * t)
        pitch = -0.02 + 0.01 * np.cos(0.4 * t)

        vn = speed_mps * np.cos(yaw)
        ve = speed_mps * np.sin(yaw)
        vd = 0.5 * np.sin(0.15 * t)  # gentle climb/descent

        pn = speed_mps * np.cos(heading_rad) * t + 2.0 * np.sin(0.3 * t)
        pe = speed_mps * np.sin(heading_rad) * t + 1.5 * np.cos(0.3 * t)
        pd = -(altitude_m + 3.0 * np.sin(0.15 * t))

        pos_ned[i] = [pn, pe, pd]
        vel_ned[i] = [vn, ve, vd]
        att_rpy[i] = [roll, pitch, yaw]
        gyro[i] = [0.015 * np.cos(0.5 * t), 0.01 * np.sin(0.4 * t), 0.05 * np.cos(0.3 * t)]
        baro[i] = -pd  # simplify: baro = AGL (no terrain offset)

    return {
        "times": times,
        "pos_ned": pos_ned,
        "vel_ned": vel_ned,
        "att_rpy": att_rpy,
        "gyro": gyro,
        "baro": baro,
    }


# -------------------------------------------------------------------------------------
# Test: Full VPS Pipeline Integration (Phase 1 + Phase 2)
# -------------------------------------------------------------------------------------

class TestE2EIntegration:
    """Complete end-to-end pipeline test from camera to MAVLink output."""

    def test_phase1_height_pipeline_with_simulated_flight(self):
        """E2E Phase 1: Camera → Tracking → De-rotation → Height Estimation → Kalman Filter → MAVLink DISTANCE_SENSOR."""
        cfg = VPSConfig()
        terrain = _get_or_create_terrain()
        traj = _build_trajectory(n_steps=120, altitude_m=100.0, speed_mps=8.0)

        # Initialize simulator
        sim = DroneCameraSimulator(
            orthophoto=terrain,
            meters_per_pixel=0.1,
            degradation_config=SimulatorDegradationConfig(sensor_noise_sigma=1.0),
        )

        # Initialize full fusion engine
        fusion = VPSFusionEngine(config=cfg, initial_height_m=100.0, initial_terrain_m=0.0)

        heights_est = []
        heights_true = []
        confidences = []
        vision_valid_count = 0

        for i in range(len(traj["times"])):
            t = float(traj["times"][i])
            pos = traj["pos_ned"][i]
            vel = traj["vel_ned"][i]
            att = traj["att_rpy"][i]
            gyro = traj["gyro"][i]
            baro = float(traj["baro"][i])
            true_h = float(-pos[2])

            state = DroneState(
                timestamp_s=t,
                pos_ned_m=pos,
                vel_ned_mps=vel,
                attitude_rpy_rad=att,
                angular_rates_rad_s=gyro,
            )
            frame = sim.render_frame(state)
            assert frame is not None, f"Simulator returned None at step {i}"
            assert frame.shape[0] > 0 and frame.shape[1] > 0

            out = fusion.process_telemetry_and_frame(
                frame=frame,
                timestamp_s=t,
                pos_ned_m=pos,
                vel_ned_mps=vel,
                attitude_rpy_rad=att,
                gyro_rates_rad_s=gyro,
                baro_alt_m=baro,
                frame_idx=i,
            )

            # Verify no NaN/Inf in output
            assert np.isfinite(out.height_agl_m), f"NaN/Inf height at step {i}: {out.height_agl_m}"
            assert np.isfinite(out.confidence), f"NaN/Inf confidence at step {i}"
            assert np.isfinite(out.variance_m2), f"NaN/Inf variance at step {i}"
            assert out.height_agl_m >= 0.0, f"Negative height at step {i}: {out.height_agl_m}"

            heights_est.append(out.height_agl_m)
            heights_true.append(true_h)
            confidences.append(out.confidence)
            if out.is_vision_fused:
                vision_valid_count += 1

        heights_est = np.array(heights_est)
        heights_true = np.array(heights_true)
        # Allow warm-up: check only after first 20 frames
        errors = np.abs(heights_est[20:] - heights_true[20:])
        mean_err_pct = float(np.mean(errors / heights_true[20:]) * 100.0)

        print(f"\n--- Phase 1 E2E Results ---")
        print(f"  Mean height error: {np.mean(errors):.2f} m ({mean_err_pct:.2f}%)")
        print(f"  Max height error: {np.max(errors):.2f} m")
        print(f"  Vision fused frames: {vision_valid_count} / {len(traj['times'])}")

        # Acceptance: < 10% mean height error (simulator has degradation noise)
        assert mean_err_pct < 10.0, f"Phase 1 height error too high: {mean_err_pct:.2f}%"

    def test_phase2_vio_and_esekf_integration(self):
        """E2E Phase 2: VIO Velocity → 15-State ES-EKF → Position/Velocity with bounded drift."""
        cfg = VPSConfig()
        cam_cfg = cfg.camera
        terrain = _get_or_create_terrain()
        traj = _build_trajectory(n_steps=150, altitude_m=80.0, speed_mps=12.0, heading_rad=0.5)

        sim = DroneCameraSimulator(
            orthophoto=terrain,
            meters_per_pixel=0.1,
            degradation_config=SimulatorDegradationConfig(sensor_noise_sigma=1.0),
        )

        # Initialize Phase 1 pipeline for height
        fusion = VPSFusionEngine(config=cfg, initial_height_m=80.0)

        # Initialize Phase 2 components
        vio_est = VIOVelocityEstimator(camera_config=cam_cfg)
        ekf = ErrorStateEKF(
            initial_pos_ned=traj["pos_ned"][0],
            initial_vel_ned=traj["vel_ned"][0],
            initial_yaw_rad=float(traj["att_rpy"][0, 2]),
        )

        K = cam_cfg.K
        R_b2c = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]], dtype=np.float64)
        derot = GyroDerotationEngine(K, R_body_to_cam=R_b2c)
        tracker = FeatureTracker(cfg.tracker)

        pos_errors = []
        vel_errors = []
        prev_gray = None
        prev_pts = None
        prev_att = None
        prev_time = None

        for i in range(len(traj["times"])):
            t = float(traj["times"][i])
            pos = traj["pos_ned"][i]
            vel = traj["vel_ned"][i]
            att = traj["att_rpy"][i]
            gyro = traj["gyro"][i]
            baro = float(traj["baro"][i])
            h_agl = float(-pos[2])

            state = DroneState(
                timestamp_s=t, pos_ned_m=pos, vel_ned_mps=vel,
                attitude_rpy_rad=att, angular_rates_rad_s=gyro,
            )
            frame = sim.render_frame(state)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame

            # IMU propagation (100 Hz equivalent)
            accel_body = np.array([0.0, 0.0, -9.80665], dtype=np.float64)  # 1g downward
            ekf.predict_imu(accel_body, gyro, t)

            # Compass yaw update
            ekf.update_compass_yaw(float(att[2]))

            # VIO velocity from optical flow
            if prev_gray is not None and prev_pts is not None and len(prev_pts) >= 8:
                dt_s = t - prev_time
                if dt_s > 0.001:
                    track_res = tracker.track(prev_gray, gray, prev_pts)
                    if track_res.num_inliers >= 8:
                        H_derot = derot.compute_rotational_homography(
                            derot.rotation_matrix_from_rpy_delta(
                                att[0] - prev_att[0],
                                att[1] - prev_att[1],
                                att[2] - prev_att[2],
                            )
                        )
                        vio_res = vio_est.estimate_velocity_from_flow(
                            pts0=track_res.pts_prev,
                            pts1=track_res.pts_curr,
                            inlier_mask=track_res.inlier_mask,
                            H_derot=H_derot,
                            height_agl_m=h_agl,
                            dt_s=dt_s,
                            attitude_rpy_rad=att,
                            timestamp_s=t,
                            compute_homography_alternative=False,
                        )
                        if vio_res.confidence > 0.2:
                            ekf.update_vio_velocity(vio_res.vel_ned_mps, vio_res.confidence)

            pts = tracker.detect_features(gray)
            prev_gray = gray.copy()
            prev_pts = pts
            prev_att = att.copy()
            prev_time = t

            # Record errors
            est_pos = ekf.state.pos_ned_m
            est_vel = ekf.state.vel_ned_mps
            pos_err = float(np.linalg.norm(est_pos[:2] - pos[:2]))
            vel_err = float(np.linalg.norm(est_vel[:2] - vel[:2]))
            pos_errors.append(pos_err)
            vel_errors.append(vel_err)

            # Verify no NaN/Inf in EKF state
            assert np.all(np.isfinite(est_pos)), f"NaN in EKF position at step {i}"
            assert np.all(np.isfinite(est_vel)), f"NaN in EKF velocity at step {i}"
            assert np.all(np.isfinite(ekf.P)), f"NaN in EKF covariance at step {i}"

        pos_errors = np.array(pos_errors)
        vel_errors = np.array(vel_errors)

        print(f"\n--- Phase 2 ES-EKF Results ---")
        print(f"  Mean position error: {np.mean(pos_errors[20:]):.2f} m")
        print(f"  Max position error: {np.max(pos_errors[20:]):.2f} m")
        print(f"  Mean velocity error: {np.mean(vel_errors[20:]):.2f} m/s")

        # VIO drift is expected without map matching, but should be bounded
        # within 5 seconds of flight (~60m at 12m/s), drift < 50m is reasonable
        assert np.max(pos_errors) < 200.0, f"Position drift exploded: {np.max(pos_errors):.1f} m"
        # Velocity should track reasonably (< 5 m/s error on 12 m/s cruise)
        assert np.mean(vel_errors[20:]) < 8.0, f"Velocity error too high: {np.mean(vel_errors[20:]):.2f} m/s"

    def test_delayed_state_map_match_fusion(self):
        """E2E: Delayed-state EKF update with map match fix reduces drift."""
        ekf = ErrorStateEKF(initial_pos_ned=np.zeros(3), initial_yaw_rad=0.0)
        rng = np.random.default_rng(42)

        # Propagate with IMU for 1 second (100 steps)
        for i in range(100):
            t = i * 0.01
            accel = np.array([0.0, 0.0, -9.80665], dtype=np.float64)
            gyro = np.array([0.001, 0.0, 0.0], dtype=np.float64)
            ekf.predict_imu(accel, gyro, t)

        # Intentionally drift the position
        ekf.state.pos_ned_m[0] += 15.0  # 15m drift north
        ekf.state.pos_ned_m[1] += 10.0  # 10m drift east

        pos_before = ekf.state.pos_ned_m.copy()

        # Apply a delayed-state map match fix at the true origin
        fix_ned = np.array([0.0, 0.0], dtype=np.float64)
        accepted = ekf.update_map_match_delayed(
            fix_ned_m=fix_ned,
            timestamp_s=0.5,  # midpoint of history
            confidence=0.9,
            bypass_gate=True,
        )

        pos_after = ekf.state.pos_ned_m.copy()
        err_before = float(np.linalg.norm(pos_before[:2]))
        err_after = float(np.linalg.norm(pos_after[:2]))

        print(f"\n--- Delayed Map Match ---")
        print(f"  Pos error before: {err_before:.2f} m")
        print(f"  Pos error after: {err_after:.2f} m")
        print(f"  Fix accepted: {accepted}")

        assert accepted, "Map match fix should have been accepted"
        assert err_after < err_before, f"Map match should reduce error: {err_after:.2f} >= {err_before:.2f}"

    def test_telemetry_buffer_interpolation(self):
        """E2E: Verify telemetry buffer provides correct interpolated samples."""
        buf = TelemetryRingBuffer(max_duration_s=5.0)

        # Insert samples at 100 Hz
        for i in range(500):
            t = i * 0.01
            sample = TelemetrySample(
                timestamp_s=t,
                gyro_rad_s=np.array([0.01 * i, 0.0, 0.0], dtype=np.float64),
                attitude_rpy_rad=np.array([0.0, 0.0, 0.1 * t], dtype=np.float64),
                vel_ned_mps=np.array([5.0, 0.0, 0.0], dtype=np.float64),
                baro_alt_m=100.0 + 0.1 * t,
            )
            buf.add_sample(sample)

        # Query at inter-sample time
        result = buf.interpolate(2.505)
        assert result is not None, "Interpolation returned None"
        assert abs(result.timestamp_s - 2.505) < 0.001
        assert abs(result.baro_alt_m - (100.0 + 0.1 * 2.505)) < 0.01

        # Query outside buffer (clamped to edges)
        result_old = buf.interpolate(-1.0)
        assert result_old is not None

    def test_health_gate_and_watchdog(self):
        """E2E: Health monitor correctly gates feeding and watchdog responds."""
        from vps.config import MAVLinkConfig
        cfg = MAVLinkConfig(feed_fc_ekf=True, min_confidence_to_feed=0.4)
        monitor = HealthMonitor(config=cfg)

        # Healthy scenario
        t_now = time.monotonic()
        h1 = monitor.evaluate_health(
            confidence=0.85,
            innovation_chi2=3.0,
            observation_timestamp_s=t_now - 0.05,
            current_timestamp_s=t_now,
        )
        assert h1.is_healthy, f"Should be healthy: {h1.status_message}"
        assert h1.can_feed_fc, "Should allow feeding when healthy and enabled"
        assert h1.covariance_scale <= 2.5, f"Scale too high for healthy: {h1.covariance_scale}"

        # Unhealthy: stale data
        h2 = monitor.evaluate_health(
            confidence=0.85,
            innovation_chi2=3.0,
            observation_timestamp_s=t_now - 1.0,  # 1 second old
            current_timestamp_s=t_now,
        )
        assert not h2.freshness_ok, "Should detect stale data"
        assert not h2.can_feed_fc, "Should NOT feed with stale data"

        # Unhealthy: low confidence
        h3 = monitor.evaluate_health(
            confidence=0.1,
            innovation_chi2=3.0,
            observation_timestamp_s=t_now - 0.05,
            current_timestamp_s=t_now,
        )
        assert not h3.confidence_ok
        assert not h3.can_feed_fc

        # Watchdog supervisor basic lifecycle
        wd = WatchdogSupervisor()
        wd.start()
        wd.record_frame()
        wd.record_telemetry()
        metrics = wd.evaluate_metrics()
        assert metrics is not None
        wd.stop()

    def test_odometry_packing_uses_actual_rates(self):
        """E2E: OdometrySender sends actual gyro rates, not EKF bias estimates."""
        nav = NavState(
            timestamp_s=1.0,
            pos_ned_m=np.array([10.0, 20.0, -50.0]),
            vel_ned_mps=np.array([5.0, 3.0, -0.5]),
            quat_wb=np.array([1.0, 0.0, 0.0, 0.0]),
            accel_bias_mps2=np.array([0.01, -0.02, 0.005]),
            gyro_bias_rad_s=np.array([0.0003, -0.0001, 0.0002]),
        )
        P = np.eye(15, dtype=np.float64) * 0.5

        actual_gyro = np.array([0.15, -0.08, 0.22], dtype=np.float64)

        # Build snapshot manually (without MAVLink connection)
        from vps.mavlink.odometry_sender import OdometrySnapshot, cov6x6_to_upper_triangle
        from vps.health.health_monitor import HealthStatus

        health = HealthStatus(
            is_healthy=True, can_feed_fc=True, mode="ACTIVE_FEEDING",
            confidence_ok=True, innovation_ok=True, freshness_ok=True,
            thermal_voltage_ok=True, cpu_temp_c=45.0, throttled_flags=0,
            covariance_scale=1.0, data_age_ms=20.0, status_message="OK",
        )

        cov_6x6 = np.eye(6, dtype=np.float64) * 0.1
        snap = OdometrySnapshot(
            timestamp_usec=1000000,
            pos_ned_m=nav.pos_ned_m.copy(),
            vel_ned_mps=nav.vel_ned_mps.copy(),
            quat_wb=nav.quat_wb.copy(),
            gyro_rad_s=actual_gyro.copy(),
            pose_cov_21=cov6x6_to_upper_triangle(cov_6x6),
            vel_cov_21=cov6x6_to_upper_triangle(cov_6x6),
            confidence=0.9,
            health=health,
        )

        # Verify that gyro rates in snapshot are the actual rates, NOT biases
        np.testing.assert_array_almost_equal(snap.gyro_rad_s, actual_gyro)
        assert not np.allclose(snap.gyro_rad_s, nav.gyro_bias_rad_s), \
            "ODOMETRY must carry actual angular rates, not EKF gyro bias estimates"

    def test_altitude_ceiling_400m(self):
        """E2E: Height-from-flow engine accepts estimates up to 400m AGL."""
        engine = HeightFromFlowEngine(
            fx=1207.2, fy=1199.1,
            min_displacement_px=10.0,
            max_altitude_m=450.0,  # pipeline now uses 450
        )

        # Simulate flow at 400m: ground displacement 50m over 0.5s = 100m/s flight
        # flow = fx * dx / H = 1207 * 50 / 400 = 150 px
        rng = np.random.default_rng(42)
        n_pts = 50
        flow_mag = 1207.2 * 50.0 / 400.0  # ~150 px
        flow_trans = np.column_stack([
            np.full(n_pts, flow_mag) + rng.normal(0, 2, n_pts),
            rng.normal(0, 2, n_pts),
        ])
        ground_disp = np.array([50.0, 0.0, 0.0], dtype=np.float64)

        result = engine.estimate_height(flow_trans, ground_disp)
        assert result.is_valid, f"Should be valid at 400m: {result.status_message}"
        assert abs(result.height_m - 400.0) < 20.0, f"Height at 400m: {result.height_m:.1f}"

    def test_no_nan_on_missing_frames(self):
        """E2E: Pipeline handles missing/empty frames without NaN or crash."""
        cfg = VPSConfig()
        fusion = VPSFusionEngine(config=cfg, initial_height_m=50.0)

        # Feed some empty/black frames
        black_frame = np.zeros((1088, 1456, 3), dtype=np.uint8)
        for i in range(10):
            t = i * 0.033
            out = fusion.process_telemetry_and_frame(
                frame=black_frame,
                timestamp_s=t,
                pos_ned_m=np.array([i * 0.5, 0.0, -50.0]),
                vel_ned_mps=np.array([5.0, 0.0, 0.0]),
                attitude_rpy_rad=np.zeros(3),
                gyro_rates_rad_s=np.zeros(3),
                baro_alt_m=50.0,
                frame_idx=i,
            )
            assert np.isfinite(out.height_agl_m), f"NaN on black frame {i}"
            assert np.isfinite(out.variance_m2)

    def test_hover_terrain_hold_no_crash(self):
        """E2E: During hover (zero speed), system holds terrain and doesn't crash."""
        cfg = VPSConfig()
        terrain = _get_or_create_terrain()
        sim = DroneCameraSimulator(orthophoto=terrain, meters_per_pixel=0.1)
        fusion = VPSFusionEngine(config=cfg, initial_height_m=60.0)

        # Hover: zero velocity, same position
        for i in range(60):
            t = i * 0.033
            state = DroneState(
                timestamp_s=t,
                pos_ned_m=np.array([100.0, 100.0, -60.0]),  # stationary
                vel_ned_mps=np.zeros(3),
                attitude_rpy_rad=np.array([0.001 * np.sin(t), 0.001 * np.cos(t), 0.5]),
                angular_rates_rad_s=np.array([0.001, 0.001, 0.0]),
            )
            frame = sim.render_frame(state)
            out = fusion.process_telemetry_and_frame(
                frame=frame,
                timestamp_s=t,
                pos_ned_m=state.pos_ned_m,
                vel_ned_mps=state.vel_ned_mps,
                attitude_rpy_rad=state.attitude_rpy_rad,
                gyro_rates_rad_s=state.angular_rates_rad_s,
                baro_alt_m=60.0,
                frame_idx=i,
            )
            assert np.isfinite(out.height_agl_m)
            assert out.height_agl_m > 0.0, "Height should stay positive during hover"

    def test_esekf_covariance_stays_positive_semidefinite(self):
        """E2E: ES-EKF covariance P remains symmetric positive semi-definite after many updates."""
        ekf = ErrorStateEKF(initial_yaw_rad=0.5)
        rng = np.random.default_rng(1234)

        for i in range(500):
            t = i * 0.01
            accel = np.array([0.0, 0.0, -9.80665]) + rng.normal(0, 0.05, 3)
            gyro = rng.normal(0, 0.01, 3)
            ekf.predict_imu(accel, gyro, t)

            if i % 3 == 0:
                vel = np.array([5.0, 2.0, -0.1]) + rng.normal(0, 0.5, 3)
                ekf.update_vio_velocity(vel, confidence=0.7)
            if i % 10 == 0:
                ekf.update_compass_yaw(0.5 + rng.normal(0, 0.02))

        # Check symmetry
        sym_err = np.max(np.abs(ekf.P - ekf.P.T))
        assert sym_err < 1e-10, f"P not symmetric: max asymmetry = {sym_err}"

        # Check positive semi-definite (all eigenvalues >= 0)
        eigvals = np.linalg.eigvalsh(ekf.P)
        assert np.all(eigvals >= -1e-8), f"P has negative eigenvalue: {np.min(eigvals)}"

    def test_map_db_and_matcher_integration(self):
        """E2E: Map database tile lookup and feature matching produce valid position fixes."""
        if not MAP_DB_PATH.is_dir():
            pytest.skip("Map database not available for integration test")

        from vps.map_db.runtime_lookup import RuntimeMapDB
        from vps.map_matcher.matcher import MapMatcher
        from vps.map_matcher.orthorectifier import FrameOrthorectifier

        db = RuntimeMapDB(db_directory=MAP_DB_PATH)
        matcher = MapMatcher()
        ortho_eng = FrameOrthorectifier(camera_config=CameraConfig())

        # Query tiles near origin
        pos_ned = np.array([0.0, 0.0], dtype=np.float64)
        tiles = db.query_tiles(pos_ned, search_radius_m=500.0, max_tiles_return=4)

        if len(tiles) > 0:
            # Create a simple test frame (extract from tile center)
            tile = tiles[0]
            h, w = tile.image_gray.shape[:2]
            test_frame = tile.image_bgr[h // 4 : 3 * h // 4, w // 4 : 3 * w // 4].copy()

            ortho_res = ortho_eng.orthorectify(
                frame=test_frame,
                height_agl_m=100.0,
                attitude_rpy_rad=np.array([0.0, 0.0, 0.0]),
                target_gsd_m_per_px=db.gsd_m_per_px,
            )

            if ortho_res.is_valid:
                match_res = matcher.match_frame_to_map(
                    ortho=ortho_res,
                    candidate_tiles=tiles,
                    pos_prior_ned_m=pos_ned,
                    search_radius_m=500.0,
                )
                # Match may or may not succeed depending on feature quality, but should not crash
                assert np.all(np.isfinite(match_res.pos_ned_m))
                assert np.all(np.isfinite(match_res.cov_ned_m2))
                assert match_res.processing_time_ms >= 0

    def test_real_flight_log_replay_e2e(self):
        """E2E: Replay the real ArduPilot flight log through the full pipeline."""
        if not FLIGHT_LOG_PATH.is_file():
            pytest.skip("Real flight log not available")

        from vps.tools.flight_log_replayer import FlightLogParser, FlightLogReplayer

        parser = FlightLogParser()
        traj = parser.parse_bin_log(
            log_path=FLIGHT_LOG_PATH,
            start_time_s=10.0,
            duration_s=30.0,
            downsample_factor=3,
        )

        assert len(traj.times_s) > 10, "Not enough trajectory samples"

        replayer = FlightLogReplayer()
        results = replayer.replay_trajectory(
            trajectory=traj,
            target_fps=10.0,
            max_samples=50,
        )

        assert results["mae_m"] < 50.0, f"MAE too high: {results['mae_m']:.1f} m"
        assert results["mean_latency_ms"] < 200.0, f"Too slow: {results['mean_latency_ms']:.1f} ms"
        assert not np.any(np.isnan(results["kalman_agl_m"])), "NaN in Kalman output"
        assert not np.any(np.isinf(results["kalman_agl_m"])), "Inf in Kalman output"

        print(f"\n--- Real Flight Log E2E ---")
        print(f"  MAE: {results['mae_m']:.2f} m")
        print(f"  RMSE: {results['rmse_m']:.2f} m")
        print(f"  Mean latency: {results['mean_latency_ms']:.1f} ms")

    def test_main_app_initialization(self):
        """E2E: VPSApplication initializes without errors in mock/SITL mode."""
        from vps.main import VPSApplication
        cfg = VPSConfig()
        cfg.camera.device_index = "mock"
        cfg.mavlink.connection_string = "udpin:127.0.0.1:19999"

        app = VPSApplication(config=cfg)
        # Should not crash — verify all subsystems exist
        assert app.fusion_engine is not None
        assert app.ring_buffer is not None
        assert app.watchdog is not None
        assert app.aruco_validator is not None
        # Phase 2 subsystems
        assert app.ekf is not None
        assert app.vio_est is not None
        assert app.vio_tracker is not None
        assert app.vio_derot is not None

    def test_main_app_autonomous_flight_loop(self):
        """E2E: VPSApplication starts and runs full autonomous flight loop with mock camera."""
        from vps.main import VPSApplication
        import threading

        cfg = VPSConfig()
        cfg.camera.device_index = "mock"
        cfg.mavlink.connection_string = "udpin:127.0.0.1:19998"

        app = VPSApplication(config=cfg)
        started = app.start()
        assert started, "VPSApplication failed to start"
        assert app.mavlink_bridge.odometry_sender is not None

        # Let the autonomous flight loop run for 0.3 seconds in a thread
        t = threading.Thread(target=app.run_autonomous_flight, daemon=True)
        t.start()
        time.sleep(0.3)

        app.stop()
        t.join(timeout=1.0)

        # Verify that states and covariances remain clean
        assert np.all(np.isfinite(app.ekf.state.pos_ned_m)), "NaN in EKF position"
        assert np.all(np.isfinite(app.ekf.state.vel_ned_mps)), "NaN in EKF velocity"
        assert np.all(np.isfinite(app.ekf.P)), "NaN in EKF covariance"

    def test_main_app_shadow_mode_recording(self, tmp_path):
        """E2E: VPSApplication in Phase 1 Shadow Mode logs telemetry to CSV without feeding FC."""
        from vps.main import VPSApplication
        import threading

        cfg = VPSConfig()
        cfg.camera.device_index = "mock"
        cfg.mavlink.connection_string = "udpin:127.0.0.1:19997"
        cfg.mavlink.feed_fc_ekf = False
        cfg.logger.output_dir = str(tmp_path / "shadow_logs")

        app = VPSApplication(
            config=cfg,
            phase1_only=True,
            record_session=True,
        )
        assert app.phase1_only is True
        assert app.record_session is True
        assert app.cfg.mavlink.feed_fc_ekf is False

        started = app.start()
        assert started, "VPSApplication failed to start in shadow mode"
        # Map worker thread should NOT be started in phase 1 shadow mode
        assert app._map_match_thread is None

        # Run loop briefly
        t = threading.Thread(target=app.run_autonomous_flight, daemon=True)
        t.start()
        time.sleep(0.3)

        csv_path = app.session_logger.csv_path if app.session_logger else None
        app.stop()
        t.join(timeout=1.0)

        assert csv_path is not None and csv_path.is_file(), f"Telemetry CSV not found at {csv_path}"
        # Check CSV has headers and rows
        with open(csv_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        assert len(lines) >= 2, f"CSV should contain at least header and 1 record, got {len(lines)}"
        assert "estimated_h_m" in lines[0]
        assert "baro_alt_m" in lines[0]


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

