"""Unit tests for adaptive keyframe management and flow tracking pipeline."""

from pathlib import Path
import numpy as np
import pytest

from vps.config import TrackerConfig, CameraConfig
from vps.tracker.keyframe_manager import KeyframeManager, KeyframeStatus
from vps.tracker.flow_pipeline import VisionHeightPipeline
from vps.tools.terrain_generator import DeccanTerrainGenerator
from vps.tools.simulator import DroneCameraSimulator, DroneState


class TestKeyframeManager:
    def test_keyframe_initialization(self):
        mgr = KeyframeManager(fx=1200.0, default_initial_altitude_m=50.0)
        img = np.zeros((100, 100), dtype=np.uint8)
        pts = np.array([[10, 10], [20, 20]], dtype=np.float32)
        pos = np.array([0.0, 0.0, -50.0])
        att = np.zeros(3)

        kf = mgr.initialize_keyframe(img, pts, timestamp_s=10.0, pos_ned_m=pos, attitude_rpy_rad=att, frame_idx=0)
        assert kf is not None
        assert kf.timestamp_s == 10.0
        assert kf.prior_altitude_m == 50.0

    def test_baseline_trigger_condition(self):
        mgr = KeyframeManager(fx=1200.0, default_initial_altitude_m=100.0)
        mgr.cfg.min_keyframe_displacement_px = 10.0

        img = np.zeros((100, 100), dtype=np.uint8)
        pts = np.ones((50, 2), dtype=np.float32)
        pos0 = np.array([0.0, 0.0, -100.0])
        att = np.zeros(3)
        mgr.initialize_keyframe(img, pts, timestamp_s=0.0, pos_ned_m=pos0, attitude_rpy_rad=att, frame_idx=0)

        # Move 0.4m -> pred flow = 1200 * 0.4 / 100 = 4.8 px (< 10 px)
        pos1 = np.array([0.4, 0.0, -100.0])
        vel = np.array([10.0, 0.0, 0.0])
        status1 = mgr.check_keyframe_status(current_timestamp_s=0.04, current_pos_ned_m=pos1, current_vel_ned_mps=vel, num_tracked_features=45)
        assert not status1.should_estimate_height
        assert not status1.should_reset_keyframe

        # Move 1.0m -> pred flow = 1200 * 1.0 / 100 = 12.0 px (>= 10 px)
        pos2 = np.array([1.0, 0.0, -100.0])
        status2 = mgr.check_keyframe_status(current_timestamp_s=0.10, current_pos_ned_m=pos2, current_vel_ned_mps=vel, num_tracked_features=42)
        assert status2.should_estimate_height
        assert status2.should_reset_keyframe
        assert status2.predicted_flow_px >= 10.0

    def test_hover_condition(self):
        mgr = KeyframeManager(fx=1200.0, hover_speed_thresh_mps=0.5)
        img = np.zeros((100, 100), dtype=np.uint8)
        pts = np.ones((30, 2), dtype=np.float32)
        mgr.initialize_keyframe(img, pts, timestamp_s=0.0, pos_ned_m=np.zeros(3), attitude_rpy_rad=np.zeros(3), frame_idx=0)

        # Velocity 0.2 m/s (< 0.5 m/s) -> hover
        pos = np.array([0.02, 0.0, 0.0])
        vel = np.array([0.2, 0.0, 0.0])
        status = mgr.check_keyframe_status(current_timestamp_s=0.1, current_pos_ned_m=pos, current_vel_ned_mps=vel, num_tracked_features=30)
        assert status.is_hovering

    def test_feature_loss_forces_keyframe_reset(self):
        mgr = KeyframeManager(fx=1200.0)
        mgr.cfg.min_features_threshold = 15
        img = np.zeros((100, 100), dtype=np.uint8)
        pts = np.ones((30, 2), dtype=np.float32)
        mgr.initialize_keyframe(img, pts, timestamp_s=0.0, pos_ned_m=np.zeros(3), attitude_rpy_rad=np.zeros(3), frame_idx=0)

        # Tracked features drop to 10 (< 15)
        pos = np.array([0.1, 0.0, 0.0])
        vel = np.array([5.0, 0.0, 0.0])
        status = mgr.check_keyframe_status(current_timestamp_s=0.02, current_pos_ned_m=pos, current_vel_ned_mps=vel, num_tracked_features=10)
        assert status.should_reset_keyframe
        assert not status.should_estimate_height
        assert "Feature attrition" in status.reason


class TestVisionHeightPipeline:
    def test_pipeline_on_simulated_sequence(self, tmp_path):
        # Generate small terrain
        terrain_file = tmp_path / "pipe_terrain.png"
        gen = DeccanTerrainGenerator(width_px=1024, height_px=1024, meters_per_pixel=0.1, seed=55)
        ortho = gen.generate(terrain_file)

        cam_cfg = CameraConfig()
        sim = DroneCameraSimulator(orthophoto=ortho, meters_per_pixel=0.1, camera_config=cam_cfg)
        pipeline = VisionHeightPipeline(camera_config=cam_cfg)

        alt = 80.0
        speed = 10.0
        dt = 1.0 / 30.0

        pos = np.array([50.0, 50.0, -alt])
        vel = np.array([speed, 0.0, 0.0])
        att = np.array([0.0, np.radians(2.0), 0.0])
        omega = np.zeros(3)

        estimates = []
        proc_times = []
        # Run 20 frames
        for f_idx in range(20):
            t = f_idx * dt
            pos += vel * dt
            state = DroneState(timestamp_s=t, pos_ned_m=pos.copy(), vel_ned_mps=vel, attitude_rpy_rad=att, angular_rates_rad_s=omega)
            frame = sim.render_frame(state)

            res = pipeline.process_frame(
                frame=frame,
                timestamp_s=t,
                pos_ned_m=pos,
                vel_ned_mps=vel,
                attitude_rpy_rad=att,
                frame_idx=f_idx,
            )

            proc_times.append(res.processing_time_ms)
            if res.is_valid and res.height_agl_m is not None:
                estimates.append(res.height_agl_m)

        # Average latency must strictly meet the < 100 ms requirement
        assert np.mean(proc_times) < 100.0
        assert len(estimates) > 0
        median_est = float(np.median(estimates))
        # Within 5% error
        assert abs(median_est - alt) / alt < 0.05

