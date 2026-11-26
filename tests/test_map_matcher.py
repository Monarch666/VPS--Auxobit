"""Unit tests for Milestone 9: Map Matcher, Orthorectifier, and Delayed-State EKF Fusion."""

import numpy as np
import cv2
import pytest

from vps.config import CameraConfig
from vps.map_matcher.orthorectifier import FrameOrthorectifier, OrthoResult
from vps.map_matcher.matcher import MapMatcher, MapMatchResult
from vps.map_db.tiler import MapTileMetadata
from vps.map_db.runtime_lookup import TileLookupResult
from vps.nav_ekf.es_ekf import ErrorStateEKF, ESEKFConfig, NavState


class TestFrameOrthorectifier:
    """Test suite for live camera frame orthorectification."""

    def test_nadir_orthorectification(self) -> None:
        """Verify nadir attitude orthorectifies to expected ground footprint."""
        cam_cfg = CameraConfig(width=728, height=544, fx=603.5, fy=599.5)
        rectifier = FrameOrthorectifier(camera_config=cam_cfg)

        # Synthetic textured frame
        img = np.zeros((544, 728, 3), dtype=np.uint8)
        cv2.rectangle(img, (200, 150), (400, 350), (255, 255, 255), -1)

        height_m = 100.0
        rpy = np.array([0.0, 0.0, 0.0])  # Nadir, North-aligned
        target_gsd = 1.0  # 1 m/px

        res = rectifier.orthorectify(img, height_m, rpy, target_gsd)

        assert res.is_valid is True
        assert res.gsd_m_per_px == pytest.approx(1.0)
        assert res.image_bgr.shape[0] > 0
        assert res.image_bgr.shape[1] > 0
        assert res.grad_mag.shape == res.image_gray.shape
        assert res.grad_angle.shape == res.image_gray.shape
        assert res.ground_width_m > 50.0  # FOV at 100m covers > 50m

    def test_tilted_attitude_orthorectification(self) -> None:
        """Verify non-zero roll and pitch generate valid warped views and gradients."""
        rectifier = FrameOrthorectifier()
        img = (np.random.rand(544, 728) * 255).astype(np.uint8)

        # 10 deg pitch, 5 deg roll, 45 deg yaw
        rpy = np.radians([5.0, 10.0, 45.0])
        res = rectifier.orthorectify(img, 150.0, rpy, 2.0)

        assert res.is_valid is True
        assert res.grad_mag.dtype == np.uint8
        assert np.max(res.grad_mag) > 0  # gradients computed

    def test_degenerate_height_safety(self) -> None:
        """Verify heights <= 0.5m return invalid result safely without crashing."""
        rectifier = FrameOrthorectifier()
        img = np.zeros((544, 728, 3), dtype=np.uint8)

        res = rectifier.orthorectify(img, 0.2, np.zeros(3), 1.0)
        assert res.is_valid is False
        assert "Height below minimum" in res.status_message


class TestMapMatcher:
    """Test suite for feature and phase correlation map matching."""

    @staticmethod
    def _create_synthetic_tile_and_view() -> tuple[TileLookupResult, OrthoResult, np.ndarray]:
        """Create synthetic tile with rich features and a shifted live ortho view."""
        tile_size = 512
        gsd = 1.0  # 1 m/px

        # Reference tile with grid of identifiable circles and rectangles
        tile_img = np.zeros((tile_size, tile_size), dtype=np.uint8)
        np.random.seed(42)
        for _ in range(80):
            x = np.random.randint(50, tile_size - 50)
            y = np.random.randint(50, tile_size - 50)
            r = np.random.randint(6, 18)
            cv2.circle(tile_img, (x, y), r, int(np.random.randint(120, 255)), -1)

        # Precompute ORB descriptors for tile
        orb = cv2.ORB_create(nfeatures=500)
        kp_tile, desc_tile = orb.detectAndCompute(tile_img, None)
        kp_pts = np.array([k.pt for k in kp_tile], dtype=np.float32)

        # Tile metadata centered at Local NED (1000.0 m North, 2000.0 m East)
        center_n = 1000.0
        center_e = 2000.0
        w_m = tile_size * gsd
        h_m = tile_size * gsd
        meta = MapTileMetadata(
            tile_id="tile_test_001",
            grid_row=0,
            grid_col=0,
            pixel_bounds=(0, 0, tile_size, tile_size),
            ned_bounds_m=(center_n - h_m / 2, center_e - w_m / 2, center_n + h_m / 2, center_e + w_m / 2),
            wgs84_bounds_deg=(19.8, 75.3, 19.85, 75.35),
            width_m=w_m,
            height_m=h_m,
            gsd_m_per_px=gsd,
            image_file="mock.png",
            feature_file="mock.npz",
            num_orb_features=len(kp_pts),
            num_akaze_features=0,
        )

        tile_res = TileLookupResult(
            metadata=meta,
            image_bgr=cv2.cvtColor(tile_img, cv2.COLOR_GRAY2BGR),
            image_gray=tile_img,
            kp_orb=kp_pts,
            desc_orb=desc_tile,
            kp_akaze=np.empty((0, 2), dtype=np.float32),
            desc_akaze=np.empty((0, 61), dtype=np.uint8),
            grad_mag=cv2.Sobel(tile_img, cv2.CV_8U, 1, 1),
            grad_angle=np.zeros_like(tile_img),
        )

        # Crop a live ortho view centered at (u=220, v=240) on the tile
        u_true_tile, v_true_tile = 220.0, 240.0
        ortho_size = 180
        u1 = int(round(u_true_tile - ortho_size / 2.0))
        v1 = int(round(v_true_tile - ortho_size / 2.0))
        live_crop = tile_img[v1:v1 + ortho_size, u1:u1 + ortho_size].copy()

        # Compute true NED coordinates of the live crop center
        true_north, true_east = tile_res.pixel_to_ned(u_true_tile, v_true_tile)

        gx = cv2.Sobel(live_crop, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(live_crop, cv2.CV_32F, 0, 1, ksize=3)
        mag = cv2.magnitude(gx, gy)
        mag_norm = np.clip(mag, 0, 255).astype(np.uint8)

        ortho_res = OrthoResult(
            image_bgr=cv2.cvtColor(live_crop, cv2.COLOR_GRAY2BGR),
            image_gray=live_crop,
            grad_mag=mag_norm,
            grad_angle=np.zeros_like(live_crop),
            width_px=ortho_size,
            height_px=ortho_size,
            gsd_m_per_px=gsd,
            ground_width_m=float(ortho_size * gsd),
            ground_height_m=float(ortho_size * gsd),
            center_u_px=ortho_size / 2.0,
            center_v_px=ortho_size / 2.0,
            H_cam_to_ortho=np.eye(3),
            is_valid=True,
            status_message="SUCCESS",
        )

        return tile_res, ortho_res, np.array([true_north, true_east])

    def test_feature_matching_accuracy(self) -> None:
        """Verify ORB matching accurately recovers true georeferenced position."""
        tile, ortho, true_pos_ned = self._create_synthetic_tile_and_view()
        matcher = MapMatcher(min_inliers_feature=8)

        # Prior is within 20m of truth
        prior_pos = true_pos_ned + np.array([10.0, -10.0])

        res = matcher.match_frame_to_map(
            ortho=ortho,
            candidate_tiles=[tile],
            pos_prior_ned_m=prior_pos,
            search_radius_m=100.0,
            timestamp_s=10.5,
        )

        assert res.is_valid is True
        assert res.match_method == "ORB_RANSAC"
        assert res.num_inliers >= 8
        assert res.confidence > 0.4
        # Absolute position error must be within 1 pixel GSD (< 1.5 m)
        pos_err = np.linalg.norm(res.pos_ned_m - true_pos_ned)
        assert pos_err < 1.5

    def test_outlier_rejection_via_search_radius(self) -> None:
        """Verify matches outside search radius are gated out."""
        tile, ortho, true_pos_ned = self._create_synthetic_tile_and_view()
        matcher = MapMatcher(min_inliers_feature=8)

        # Prior is 500m away, while search_radius is 50m
        far_prior = true_pos_ned + np.array([500.0, 500.0])

        res = matcher.match_frame_to_map(
            ortho=ortho,
            candidate_tiles=[tile],
            pos_prior_ned_m=far_prior,
            search_radius_m=50.0,
            timestamp_s=10.5,
        )

        # Should either be invalid or gated out
        if res.is_valid:
            assert np.linalg.norm(res.pos_ned_m - far_prior) <= 50.0
        else:
            assert res.is_valid is False

    def test_phase_correlation_fallback(self) -> None:
        """Verify phase correlation fallback operates when feature count is zero."""
        tile, ortho, true_pos_ned = self._create_synthetic_tile_and_view()
        matcher = MapMatcher(min_inliers_feature=999, min_phase_correlation_peak=0.10)

        # Setting min_inliers_feature=999 forces fallback to phase correlation
        prior_pos = true_pos_ned + np.array([4.0, -3.0])

        res = matcher.match_frame_to_map(
            ortho=ortho,
            candidate_tiles=[tile],
            pos_prior_ned_m=prior_pos,
            search_radius_m=50.0,
            timestamp_s=12.0,
        )

        assert res.is_valid is True
        assert res.match_method == "PHASE_CORRELATION"
        pos_err = np.linalg.norm(res.pos_ned_m - true_pos_ned)
        assert pos_err < 2.0


class TestDelayedStateEKF:
    """Test suite for Delayed-State Map Matching EKF updates."""

    def test_delayed_state_update_reduces_drift(self) -> None:
        """Verify a delayed position fix corrects accumulated dead-reckoning drift."""
        # Drone initialized with a 15m horizontal error relative to true origin (0, 0)
        ekf = ErrorStateEKF(initial_pos_ned=[15.0, -15.0, -50.0])

        # Step IMU for 100 steps (1.0 second)
        t = 0.0
        for _ in range(100):
            t += 0.01
            ekf.predict_imu(
                accel_meas_mps2=np.array([0.0, 0.0, -9.80665]),
                gyro_meas_rad_s=np.zeros(3),
                timestamp_s=t,
            )

        initial_err = float(np.linalg.norm(ekf.state.pos_ned_m[:2]))
        assert initial_err > 20.0  # sqrt(15^2 + 15^2) ~ 21.2 m

        # A map fix arrived indicating true position at t = 0.5s was (0.0, 0.0)
        t_delayed = 0.5
        true_fix_delayed = np.array([0.0, 0.0])
        cov_fix = np.eye(2) * (0.5 ** 2)

        accepted = ekf.update_map_match_delayed(
            fix_ned_m=true_fix_delayed,
            cov_ned_m2=cov_fix,
            timestamp_s=t_delayed,
            confidence=0.95,
            bypass_gate=True,
        )

        assert accepted is True
        # Re-propagated current position should have its error drastically reduced (>75% error reduction)
        corrected_err = float(np.linalg.norm(ekf.state.pos_ned_m[:2]))
        assert corrected_err < 5.0
        assert corrected_err < (initial_err * 0.25)

    def test_map_match_chi2_outlier_rejection(self) -> None:
        """Verify wild map match fixes are rejected by chi-squared innovation gate."""
        ekf = ErrorStateEKF(initial_pos_ned=[100.0, 100.0, -100.0])
        # Run 5 steps to populate covariance
        for i in range(5):
            ekf.predict_imu(np.array([0.0, 0.0, -9.80665]), np.zeros(3), timestamp_s=0.01 * (i + 1))

        # Wild 1000m jump
        wild_fix = np.array([1100.0, 1100.0])
        accepted = ekf.update_map_match_delayed(
            fix_ned_m=wild_fix,
            timestamp_s=0.05,
            confidence=0.9,
            bypass_gate=False,
        )

        assert accepted is False
        # State must remain untouched
        assert ekf.state.pos_ned_m[0] == pytest.approx(100.0, abs=1.0)


def test_ortho_border_mask_prevents_false_phase_correlation() -> None:
    """Verify orthorectifier masks padding borders and matcher rejects tiny low-altitude views."""
    rectifier = FrameOrthorectifier()
    # Flat uniform bright frame whose only strong contrast would be its 0-padded outer border
    uniform = np.full((1088, 1456, 3), 180, dtype=np.uint8)
    ortho = rectifier.orthorectify(uniform, height_agl_m=18.0, attitude_rpy_rad=np.zeros(3), target_gsd_m_per_px=0.56)
    assert ortho.valid_mask is not None
    # Exterior gradient pixels must remain zero (no artificial border rectangle)
    assert np.all(ortho.grad_mag[ortho.valid_mask == 0] == 0)

    tile_img = np.random.default_rng(7).integers(30, 220, size=(512, 512), dtype=np.uint8)
    meta = MapTileMetadata(
        tile_id="t0", grid_row=0, grid_col=0, pixel_bounds=(0, 0, 512, 512),
        ned_bounds_m=(-140.0, -140.0, 140.0, 140.0), wgs84_bounds_deg=(19.8, 75.28, 19.81, 75.29),
        width_m=280.0, height_m=280.0, gsd_m_per_px=0.56,
        image_file="m.png", feature_file="f.npz", num_orb_features=0, num_akaze_features=0,
    )
    tile = TileLookupResult(
        metadata=meta, image_bgr=cv2.cvtColor(tile_img, cv2.COLOR_GRAY2BGR), image_gray=tile_img,
        kp_orb=np.empty((0, 2), dtype=np.float32), desc_orb=np.empty((0, 32), dtype=np.uint8),
        kp_akaze=np.empty((0, 2), dtype=np.float32), desc_akaze=np.empty((0, 61), dtype=np.uint8),
        grad_mag=cv2.Sobel(tile_img, cv2.CV_8U, 1, 1), grad_angle=np.zeros_like(tile_img),
    )
    matcher = MapMatcher(min_inliers_feature=10)
    res = matcher.match_frame_to_map(ortho, [tile], np.zeros(2), search_radius_m=200.0)
    assert res.is_valid is False
    assert res.status_message in ("FOOTPRINT_TOO_SMALL_FOR_GSD", "LOW_GRADIENT_TEXTURE", "NO_VALID_MATCH_FOUND")
