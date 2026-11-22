"""Unit tests for ArUco metric low-altitude validation and hand-held testing."""

from pathlib import Path
import numpy as np
import pytest
import cv2

from vps.config import CameraConfig
from vps.tools.aruco_validator import ArUcoValidator, ArUcoMarkerConfig, ArUcoMeasurement


class TestArUcoValidator:
    @pytest.fixture
    def validator(self):
        cam_cfg = CameraConfig(width=1456, height=1088, fx=1207.2, fy=1199.1, cx=728.0, cy=544.0)
        marker_cfg = ArUcoMarkerConfig(
            dictionary_id=cv2.aruco.DICT_4X4_50,
            marker_length_m=0.20,
            baseline_offset_m=0.15,
        )
        return ArUcoValidator(camera_config=cam_cfg, marker_config=marker_cfg)

    def test_marker_detection_and_pose(self, validator):
        """Verify synthetic marker detection and metric pose estimation at 1.0 m."""
        true_alt_m = 1.0
        frame = validator.render_synthetic_marker_frame(altitude_m=true_alt_m)
        meas = validator.detect_and_estimate(frame)

        assert meas is not None
        assert meas.detected
        assert meas.marker_id == 0

        # Optical Z should be true_alt + baseline (1.0 + 0.15 = 1.15 m)
        assert abs(meas.optical_height_m - 1.15) < 0.01

        # Net AGL should be 1.0 m within 5 mm
        assert abs(meas.net_agl_m - true_alt_m) < 0.005

        # Tilt angle should be small for level rendering (within 2.5 deg)
        assert meas.tilt_angle_deg < 2.5

        # Reprojection error should be sub-pixel
        assert meas.reprojection_error_px < 0.5

    def test_baseline_offset_subtraction(self):
        """Verify ground baseline offset subtraction (0.15 m offset yields net AGL = 0.00 m on ground)."""
        # Compact 6 cm marker that fits inside FOV at ground contact distance of 15 cm
        cam_cfg = CameraConfig(width=1456, height=1088, fx=1207.2, fy=1199.1, cx=728.0, cy=544.0)
        marker_cfg = ArUcoMarkerConfig(
            dictionary_id=cv2.aruco.DICT_4X4_50,
            marker_length_m=0.06,  # 6 cm compact ground target
            baseline_offset_m=0.15,
        )
        val = ArUcoValidator(camera_config=cam_cfg, marker_config=marker_cfg)

        ground_frame = val.render_synthetic_marker_frame(altitude_m=0.0)
        meas = val.detect_and_estimate(ground_frame)

        assert meas is not None
        # Camera is 15 cm above ground marker
        assert abs(meas.optical_height_m - 0.15) < 0.005
        # Net AGL must be zero when resting on ground (within 1 mm tolerance)
        assert abs(meas.net_agl_m) < 0.001

    def test_hand_held_altitude_recovery(self, validator):
        """Verify hand-held heights (0.5 m, 1.0 m, 1.5 m, 2.0 m) achieve millimeter accuracy."""
        for h in [0.5, 1.0, 1.5, 2.0]:
            frame = validator.render_synthetic_marker_frame(altitude_m=h)
            meas = validator.detect_and_estimate(frame)
            assert meas is not None
            err_mm = abs(meas.net_agl_m - h) * 1000.0
            # Must be within 5 mm at hand-held distances
            assert err_mm < 5.0, f"Height {h}m error {err_mm:.1f}mm exceeds 5mm limit!"

    def test_tilt_angle_invariance(self, validator):
        """Verify perpendicular height recovery is invariant to drone roll and pitch."""
        true_h = 1.2
        frame = validator.render_synthetic_marker_frame(
            altitude_m=true_h,
            roll_deg=8.0,
            pitch_deg=-10.0,
            yaw_deg=15.0,
        )
        meas = validator.detect_and_estimate(frame)
        assert meas is not None
        assert meas.detected

        # Measured tilt angle should be close to composite tilt angle (~12.7 deg)
        assert meas.tilt_angle_deg > 5.0

        # Perpendicular height should still match within 10 mm
        err_mm = abs(meas.net_agl_m - true_h) * 1000.0
        assert err_mm < 10.0, f"Tilted error {err_mm:.1f}mm exceeds 10mm limit!"

    def test_hud_overlay_drawing(self, validator):
        """Verify HUD overlay renders without errors."""
        frame = validator.render_synthetic_marker_frame(altitude_m=1.0)
        meas = validator.detect_and_estimate(frame)
        overlay = validator.draw_overlay(frame, meas, vps_agl_m=1.002)

        assert overlay.shape == frame.shape
        assert overlay.dtype == np.uint8
        # Overlay should have drawn colored elements (diff from original)
        assert not np.array_equal(overlay, frame)
