"""Unit tests for camera capture, calibration, focus inspection, and ground baseline offset."""

from pathlib import Path
import numpy as np
import pytest
import cv2

from vps.camera.capture import MockCameraCapture, OpenCVCameraCapture
from vps.camera.calibration import CheckerboardCalibrator, render_synthetic_checkerboard
from vps.tools.calibrate_camera import generate_synthetic_calibration_dataset
from vps.tools.focus_inspector import InfinityFocusInspector
from vps.tools.baseline_calibrator import GroundBaselineCalibrator
from vps.config import VPSConfig


class TestCameraCapture:
    def test_mock_camera_capture(self):
        # Create 5 synthetic dummy frames
        frames = [np.full((100, 100, 3), i * 50, dtype=np.uint8) for i in range(5)]
        cap = MockCameraCapture(frames=frames, fps=30.0, loop=False)

        assert cap.open()
        read_frames = []
        timestamps = []
        for _ in range(5):
            ret, frame, t = cap.read_frame()
            assert ret
            assert frame is not None
            read_frames.append(frame)
            timestamps.append(t)

        # 6th read should return False when loop=False
        ret, frame, _ = cap.read_frame()
        assert not ret
        assert frame is None

        # Timestamps should be strictly monotonic with ~33.3ms intervals
        dts = np.diff(timestamps)
        assert np.allclose(dts, 1.0 / 30.0, atol=1e-3)
        cap.release()

    def test_opencv_camera_fallback(self):
        # Index 99 should gracefully return False if no device exists
        cap = OpenCVCameraCapture(device_index=99, use_threading=False)
        assert not cap.open()
        ret, frame, _ = cap.read_frame()
        assert not ret
        assert frame is None
        cap.release()


class TestCheckerboardCalibration:
    def test_synthetic_checkerboard_calibration(self, tmp_path):
        views_dir = tmp_path / "calib_views"
        views = generate_synthetic_calibration_dataset(views_dir, num_views=8)
        assert len(views) == 8

        images = [cv2.imread(str(p)) for p in views]
        calibrator = CheckerboardCalibrator(pattern_size=(9, 6), square_size_m=0.025)

        res = calibrator.calibrate_from_images(images)

        assert res.is_valid
        assert res.num_images_used >= 6
        assert res.mean_reprojection_error_px < 0.5  # sub-pixel error
        # True fx was 1207.2, fy was 1199.1
        assert 1150.0 < res.K[0, 0] < 1260.0
        assert 1140.0 < res.K[1, 1] < 1250.0

        out_yaml = tmp_path / "test_calib.yaml"
        calibrator.save_to_yaml(res, out_yaml)
        assert out_yaml.is_file()


class TestFocusInspector:
    def test_sharp_vs_blurred_discrimination(self):
        inspector = InfinityFocusInspector(min_laplacian_var_optimal=200.0, min_laplacian_var_acceptable=80.0)

        # High-contrast sharp image
        sharp_img = render_synthetic_checkerboard((9, 6), square_size_px=30)
        # Add high-frequency noise
        noise = np.random.default_rng(42).normal(0, 10, size=sharp_img.shape)
        sharp_img = np.clip(sharp_img.astype(np.float32) + noise, 0, 255).astype(np.uint8)

        # Defocused / blurred image
        blurred_img = cv2.GaussianBlur(sharp_img, (25, 25), 8.0)

        metrics_sharp = inspector.evaluate_sharpness(sharp_img)
        metrics_blurred = inspector.evaluate_sharpness(blurred_img)

        # Sharp image should have drastically higher Laplacian variance & Tenengrad
        assert metrics_sharp.laplacian_variance > metrics_blurred.laplacian_variance * 5.0
        assert metrics_sharp.tenengrad_energy > metrics_blurred.tenengrad_energy * 3.0
        assert metrics_sharp.fft_high_frequency_ratio > metrics_blurred.fft_high_frequency_ratio

        assert metrics_sharp.is_focused
        assert not metrics_blurred.is_focused
        assert metrics_blurred.focus_status == "DEFOCUSED"


class TestGroundBaselineCalibrator:
    def test_baseline_offset_calculation(self):
        calibrator = GroundBaselineCalibrator()
        # Simulated camera ground readings when sitting on landing gear (mean 0.152 m)
        samples = [0.151, 0.153, 0.152, 0.150, 0.154, 0.152]
        res = calibrator.calibrate_from_static_measurements(samples)

        assert res.is_valid
        assert abs(res.baseline_offset_m - 0.152) < 1e-4
        assert res.offset_std_dev_m < 0.005

    def test_baseline_correction_hand_held(self):
        calibrator = GroundBaselineCalibrator()
        offset = 0.15  # 15 cm landing gear offset

        # Drone held at 1.0 m AGL (camera measures 1.15 m from ground)
        agl = calibrator.apply_baseline_correction(raw_camera_height_m=1.15, baseline_offset_m=offset)
        assert abs(agl - 1.0) < 1e-5

        # Drone resting on ground (camera measures 0.15 m) -> AGL is exactly 0.0 m
        resting_agl = calibrator.apply_baseline_correction(raw_camera_height_m=0.15, baseline_offset_m=offset)
        assert resting_agl == 0.0

        # Noise below baseline clamps cleanly to 0.0 m
        clamped = calibrator.apply_baseline_correction(raw_camera_height_m=0.12, baseline_offset_m=offset)
        assert clamped == 0.0
