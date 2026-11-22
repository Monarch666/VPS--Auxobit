"""ArUco metric ground-truth validation tool for low-altitude (<= 30 m) and hand-held testing.

Functions:
1. Detects planar ArUco fiducial markers of known physical dimension (e.g. DICT_4X4_50, 0.20 m).
2. Computes 6-DOF camera pose via solvePnP (SOLVEPNP_IPPE_SQUARE).
3. Recovers mathematically exact perpendicular ground-truth altitude invariant to tilt:
   h_optical = |(-R^T * t)_z|
4. Subtracts calibrated landing-gear baseline offset (15 cm):
   h_net_agl = h_optical - h_baseline
   When resting on the ground, h_net_agl = 0.00 m.
5. Provides synthetic validation generator across 0.15 m to 30.0 m for automated benchmarking.
6. Renders live HUD overlay displaying detected marker, 3D coordinate axes, raw optical Z,
   baseline offset, net AGL height, VPS estimated height, error delta, and tilt angle.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import sys
import time
import numpy as np
import cv2

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from vps.config import CameraConfig
from vps.camera.calibration import load_calibration_yaml


@dataclass
class ArUcoMarkerConfig:
    """Configuration for ArUco ground-truth marker validation."""
    dictionary_id: int = cv2.aruco.DICT_4X4_50
    marker_length_m: float = 0.20  # 20 cm physical side length
    target_id: Optional[int] = None  # None = accept any detected marker
    baseline_offset_m: float = 0.15  # 15 cm camera-to-landing-gear baseline
    max_valid_altitude_m: float = 30.0  # Requirement: validation at <= 30 m


@dataclass
class ArUcoMeasurement:
    """Metric ArUco detection and ground-truth measurement."""
    timestamp_s: float
    marker_id: int
    detected: bool
    corners_px: np.ndarray  # 4x2 corner coordinates
    rvec: np.ndarray  # (3, 1) rotation vector
    tvec: np.ndarray  # (3, 1) translation vector in camera frame
    optical_height_m: float  # Perpendicular normal distance to marker plane
    net_agl_m: float  # Baseline-subtracted height (landing gear AGL)
    tilt_angle_deg: float  # Tilt angle between camera optical axis and marker normal
    pixel_width_px: float  # Apparent side width in image
    reprojection_error_px: float = 0.0


class ArUcoValidator:
    """Precision metric validator using planar fiducial markers."""

    def __init__(
        self,
        camera_config: Optional[CameraConfig] = None,
        marker_config: Optional[ArUcoMarkerConfig] = None,
        calibration_path: Optional[Path | str] = None,
    ) -> None:
        """Initialize ArUco validator with camera intrinsics and marker configuration."""
        self.cam_cfg = camera_config if camera_config is not None else CameraConfig()
        self.marker_cfg = marker_config if marker_config is not None else ArUcoMarkerConfig()

        # Load calibrated intrinsics if available, otherwise use config defaults
        calib_file = Path(calibration_path) if calibration_path else Path("config/imx296_calibrated.yaml")
        if calib_file.is_file():
            self.K, self.dist_coeffs = load_calibration_yaml(calib_file)
        else:
            self.K = np.array([
                [self.cam_cfg.fx, 0.0, self.cam_cfg.cx],
                [0.0, self.cam_cfg.fy, self.cam_cfg.cy],
                [0.0, 0.0, 1.0],
            ], dtype=np.float64)
            self.dist_coeffs = np.array(self.cam_cfg.dist_coeffs, dtype=np.float64)

        # Initialize OpenCV ArUco detector
        self.dictionary = cv2.aruco.getPredefinedDictionary(self.marker_cfg.dictionary_id)
        self.detector_params = cv2.aruco.DetectorParameters()
        self.detector_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        self.detector = cv2.aruco.ArucoDetector(self.dictionary, self.detector_params)

        # Object 3D points in marker local frame (centered at origin, z=0)
        # Order: Top-Left, Top-Right, Bottom-Right, Bottom-Left
        L = self.marker_cfg.marker_length_m
        self.obj_points = np.array([
            [-L / 2.0,  L / 2.0, 0.0],
            [ L / 2.0,  L / 2.0, 0.0],
            [ L / 2.0, -L / 2.0, 0.0],
            [-L / 2.0, -L / 2.0, 0.0],
        ], dtype=np.float32)

    def detect_and_estimate(
        self,
        frame: np.ndarray,
        timestamp_s: float = 0.0,
    ) -> Optional[ArUcoMeasurement]:
        """Detect ArUco marker in frame and compute metric 3D pose and perpendicular height.

        Args:
            frame: Grayscale or BGR image (1456x1088).
            timestamp_s: Timestamp of frame capture.

        Returns:
            ArUcoMeasurement if marker detected and valid, else None.
        """
        if frame.ndim == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame

        corners, ids, _ = self.detector.detectMarkers(gray)
        if ids is None or len(ids) == 0:
            return None

        # Select target marker or the largest detected marker
        selected_idx = 0
        if self.marker_cfg.target_id is not None:
            matches = np.where(ids.ravel() == self.marker_cfg.target_id)[0]
            if len(matches) == 0:
                return None
            selected_idx = int(matches[0])
        else:
            # Pick marker with largest apparent pixel perimeter
            perims = [cv2.arcLength(c[0], True) for c in corners]
            selected_idx = int(np.argmax(perims))

        c_pts = corners[selected_idx][0]  # (4, 2)
        marker_id = int(ids.ravel()[selected_idx])

        # Solve PnP using IPPE_SQUARE (planar square fiducial solver)
        success, rvec, tvec = cv2.solvePnP(
            self.obj_points,
            c_pts.astype(np.float32),
            self.K,
            self.dist_coeffs,
            flags=cv2.SOLVEPNP_IPPE_SQUARE,
        )
        if not success:
            return None

        # Compute rotation matrix R (from marker to camera coordinates)
        R_cam_marker, _ = cv2.Rodrigues(rvec)

        # Invert pose to get camera position in marker frame:
        # C_marker = -R^T * tvec
        # In marker frame, z is perpendicular to the ground plane!
        cam_in_marker = -R_cam_marker.T @ tvec
        optical_height_m = float(abs(cam_in_marker.ravel()[2]))

        # Net AGL height (subtracted landing-gear baseline offset)
        net_agl_m = max(0.0, optical_height_m - self.marker_cfg.baseline_offset_m)

        # Tilt angle between camera optical axis (+Z in camera frame) and marker normal
        # Marker normal in camera frame is R_cam_marker[:, 2]
        cam_z = np.array([0.0, 0.0, 1.0])
        normal_cam = R_cam_marker[:, 2]
        cos_tilt = np.clip(np.dot(cam_z, normal_cam), -1.0, 1.0)
        tilt_angle_deg = float(np.degrees(np.arccos(abs(cos_tilt))))

        # Measure pixel width
        p_w = float(np.mean([
            np.linalg.norm(c_pts[0] - c_pts[1]),
            np.linalg.norm(c_pts[1] - c_pts[2]),
            np.linalg.norm(c_pts[2] - c_pts[3]),
            np.linalg.norm(c_pts[3] - c_pts[0]),
        ]))

        # Reprojection error
        proj_pts, _ = cv2.projectPoints(self.obj_points, rvec, tvec, self.K, self.dist_coeffs)
        reproj_err = float(np.mean(np.linalg.norm(c_pts - proj_pts.reshape(-1, 2), axis=1)))

        return ArUcoMeasurement(
            timestamp_s=timestamp_s,
            marker_id=marker_id,
            detected=True,
            corners_px=c_pts,
            rvec=rvec,
            tvec=tvec,
            optical_height_m=optical_height_m,
            net_agl_m=net_agl_m,
            tilt_angle_deg=tilt_angle_deg,
            pixel_width_px=p_w,
            reprojection_error_px=reproj_err,
        )

    def draw_overlay(
        self,
        frame: np.ndarray,
        meas: Optional[ArUcoMeasurement],
        vps_agl_m: Optional[float] = None,
    ) -> np.ndarray:
        """Render diagnostic HUD overlay on camera frame."""
        vis = frame.copy()
        if vis.ndim == 2:
            vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)

        h, w = vis.shape[:2]

        if meas is not None and meas.detected:
            # Draw marker polygon
            pts = meas.corners_px.astype(np.int32).reshape((-1, 1, 2))
            cv2.polylines(vis, [pts], isClosed=True, color=(0, 255, 0), thickness=2)

            # Draw 3D coordinate axes at marker origin (length = 0.1 m)
            axis_len = self.marker_cfg.marker_length_m * 0.75
            cv2.drawFrameAxes(vis, self.K, self.dist_coeffs, meas.rvec, meas.tvec, axis_len, thickness=2)

            # Draw corner 0 indicator
            c0 = tuple(meas.corners_px[0].astype(int))
            cv2.circle(vis, c0, 4, (0, 0, 255), -1)

            # HUD Panel
            panel_w, panel_h = 380, 160
            cv2.rectangle(vis, (15, 15), (15 + panel_w, 15 + panel_h), (20, 20, 20), -1)
            cv2.rectangle(vis, (15, 15), (15 + panel_w, 15 + panel_h), (0, 200, 0), 2)

            lines = [
                f"ARUCO VALIDATOR (ID: {meas.marker_id})",
                f"Raw Optical Z:    {meas.optical_height_m:6.3f} m",
                f"Baseline Offset: -{self.marker_cfg.baseline_offset_m:6.3f} m",
                f"Net AGL Height:   {meas.net_agl_m:6.3f} m",
                f"Tilt Angle:       {meas.tilt_angle_deg:5.1f} deg",
            ]
            if vps_agl_m is not None:
                err = vps_agl_m - meas.net_agl_m
                err_pct = (abs(err) / max(0.1, meas.net_agl_m)) * 100.0
                lines.append(f"VPS AGL: {vps_agl_m:6.3f} m (Err: {err:+.3f}m, {err_pct:.1f}%)")

            for i, line in enumerate(lines):
                color = (0, 255, 0) if i == 0 or i == 3 else (220, 220, 220)
                if i == 5:
                    color = (0, 220, 255)
                cv2.putText(vis, line, (25, 40 + i * 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)
        else:
            # Not detected banner
            cv2.rectangle(vis, (15, 15), (320, 75), (20, 20, 20), -1)
            cv2.rectangle(vis, (15, 15), (320, 75), (0, 0, 220), 2)
            cv2.putText(vis, "ARUCO: NO MARKER DETECTED", (25, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 100, 255), 2)

        return vis

    def render_synthetic_marker_frame(
        self,
        altitude_m: float,
        roll_deg: float = 0.0,
        pitch_deg: float = 0.0,
        yaw_deg: float = 0.0,
        offset_xy_m: Tuple[float, float] = (0.0, 0.0),
        background_img: Optional[np.ndarray] = None,
        sensor_noise_sigma: float = 2.0,
    ) -> np.ndarray:
        """Render a synthetic camera frame observing an ArUco marker on the ground.

        Args:
            altitude_m: Drone true AGL height in meters.
            roll_deg: Drone roll angle in degrees.
            pitch_deg: Drone pitch angle in degrees.
            yaw_deg: Drone yaw angle in degrees.
            offset_xy_m: Horizontal offset of marker center relative to camera optical axis (m).
            background_img: Optional background image (1456x1088).
            sensor_noise_sigma: Gaussian noise standard deviation.

        Returns:
            Rendered synthetic camera image with marker.
        """
        w = self.cam_cfg.width
        h = self.cam_cfg.height

        if background_img is not None:
            canvas = background_img.copy()
            if canvas.shape[:2] != (h, w):
                canvas = cv2.resize(canvas, (w, h))
        else:
            # Textured ground background
            canvas = np.full((h, w, 3), 140, dtype=np.uint8)
            # Add subtle texture
            rng = np.random.default_rng(42)
            noise = rng.integers(-15, 15, size=(h, w, 3), dtype=np.int16)
            canvas = np.clip(canvas.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        # Generate ArUco marker with border
        marker_id = self.marker_cfg.target_id if self.marker_cfg.target_id is not None else 0
        marker_px = 400
        marker_base = cv2.aruco.generateImageMarker(self.dictionary, marker_id, marker_px)
        # Add white quiet zone around marker (required for standard ArUco detection)
        border_px = marker_px // 5
        marker_full = cv2.copyMakeBorder(
            marker_base, border_px, border_px, border_px, border_px,
            cv2.BORDER_CONSTANT, value=255
        )
        total_marker_px = marker_full.shape[0]

        # Camera total height above marker plane:
        # Camera is mounted at landing gear + baseline offset:
        cam_height_m = altitude_m + self.marker_cfg.baseline_offset_m

        # Marker 3D corners in world frame (ground plane, Z=0)
        # Marker center is at (offset_xy_m[0], offset_xy_m[1], 0)
        # Size of full marker including white border:
        L_full = self.marker_cfg.marker_length_m * (total_marker_px / marker_px)
        half_L = L_full / 2.0
        mx, my = offset_xy_m

        # Marker 3D corners in world frame (ground plane, Z=0, where +Z points UP)
        # Order: Top-Left (x-, y+), Top-Right (x+, y+), Bottom-Right (x+, y-), Bottom-Left (x-, y-)
        pts_world = np.array([
            [mx - half_L, my + half_L, 0.0],
            [mx + half_L, my + half_L, 0.0],
            [mx + half_L, my - half_L, 0.0],
            [mx - half_L, my - half_L, 0.0],
        ], dtype=np.float64)

        # Drone attitude rotation (roll, pitch, yaw)
        phi = np.radians(roll_deg)
        theta = np.radians(pitch_deg)
        psi = np.radians(yaw_deg)

        Rx = np.array([[1, 0, 0], [0, np.cos(phi), -np.sin(phi)], [0, np.sin(phi), np.cos(phi)]])
        Ry = np.array([[np.cos(theta), 0, np.sin(theta)], [0, 1, 0], [-np.sin(theta), 0, np.cos(theta)]])
        Rz = np.array([[np.cos(psi), -np.sin(psi), 0], [np.sin(psi), np.cos(psi), 0], [0, 0, 1]])
        R_body = Rz @ Ry @ Rx

        # Level downward camera in world frame (+X right/east, +Y north, +Z up)
        # Optical frame: +X right, +Y down (south), +Z forward (downwards)
        R_cam0_world = np.array([
            [1.0,  0.0,  0.0],
            [0.0, -1.0,  0.0],
            [0.0,  0.0, -1.0],
        ], dtype=np.float64)
        R_cam_world = R_body @ R_cam0_world

        # Camera center in world coordinates
        C_world = np.array([0.0, 0.0, cam_height_m], dtype=np.float64)

        # Transformation from world to camera frame:
        # P_cam = R_cam_world.T @ (P_world - C_world) = R_w2c @ P_world + t_w2c
        R_w2c = R_cam_world.T
        t_w2c = -R_w2c @ C_world
        rvec_cam, _ = cv2.Rodrigues(R_w2c)

        # Project 3D marker corners into camera image
        img_pts, _ = cv2.projectPoints(pts_world, rvec_cam, t_w2c, self.K, self.dist_coeffs)
        img_pts = img_pts.reshape(-1, 2)

        # Source coordinates in marker image
        src_pts = np.array([
            [0, 0],
            [total_marker_px, 0],
            [total_marker_px, total_marker_px],
            [0, total_marker_px],
        ], dtype=np.float32)

        # Compute homography and warp marker into canvas
        H, _ = cv2.findHomography(src_pts, img_pts.astype(np.float32))
        if H is not None:
            marker_bgr = cv2.cvtColor(marker_full, cv2.COLOR_GRAY2BGR)
            warped = cv2.warpPerspective(marker_bgr, H, (w, h), borderValue=(128, 128, 128))
            mask = cv2.warpPerspective(np.ones_like(marker_full) * 255, H, (w, h), borderValue=0)

            # Blend marker into canvas
            mask_bool = mask > 128
            canvas[mask_bool] = warped[mask_bool]

        # Add sensor noise
        if sensor_noise_sigma > 0:
            noise = np.random.normal(0, sensor_noise_sigma, canvas.shape).astype(np.int16)
            canvas = np.clip(canvas.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        return canvas


def run_aruco_altitude_benchmark(
    altitudes: Optional[List[float]] = None,
    output_plot_path: Path = Path("reports/m7_aruco_benchmark.png"),
) -> Dict[str, Any]:
    """Evaluate ArUco ground-truth validation accuracy across 0.2 m to 30 m."""
    if altitudes is None:
        altitudes = [0.2, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0]

    validator = ArUcoValidator()
    results = []

    print(f"\n--- Running ArUco Metric Altitude Benchmark (0.2 m to 30.0 m) ---")
    print(f"{'True AGL':>10} | {'Optical Z':>10} | {'Net AGL':>10} | {'Error (mm)':>11} | {'Err %':>8} | {'Px Width':>9} | {'Status':>8}")
    print("-" * 76)

    for alt in altitudes:
        # Render synthetic marker frame
        frame = validator.render_synthetic_marker_frame(altitude_m=alt)
        meas = validator.detect_and_estimate(frame)

        if meas is not None:
            err_m = meas.net_agl_m - alt
            err_mm = err_m * 1000.0
            err_pct = (abs(err_m) / alt) * 100.0
            status = "PASS" if err_pct <= 2.0 else "WARN"

            print(
                f"{alt:9.2f}m | {meas.optical_height_m:9.3f}m | {meas.net_agl_m:9.3f}m | "
                f"{err_mm:+10.1f}mm | {err_pct:7.2f}% | {meas.pixel_width_px:8.1f}px | {status:>8}"
            )
            results.append({
                "true_agl_m": alt,
                "optical_z_m": meas.optical_height_m,
                "net_agl_m": meas.net_agl_m,
                "error_m": err_m,
                "error_pct": err_pct,
                "pixel_width_px": meas.pixel_width_px,
                "detected": True,
            })
        else:
            print(f"{alt:9.2f}m | {'NOT DETECTED':>45} | {'FAIL':>8}")
            results.append({
                "true_agl_m": alt,
                "detected": False,
                "error_pct": 100.0,
            })

    output_plot_path.parent.mkdir(parents=True, exist_ok=True)
    _plot_aruco_benchmark(results, output_plot_path)

    return {"results": results, "plot_path": str(output_plot_path)}


def _plot_aruco_benchmark(results: List[Dict[str, Any]], output_path: Path) -> None:
    """Plot ArUco benchmark diagnostic curves."""
    valid_pts = [r for r in results if r["detected"]]
    if not valid_pts:
        return

    alts = [r["true_agl_m"] for r in valid_pts]
    meas_agl = [r["net_agl_m"] for r in valid_pts]
    err_pct = [r["error_pct"] for r in valid_pts]
    px_w = [r["pixel_width_px"] for r in valid_pts]

    fig, axs = plt.subplots(1, 3, figsize=(15, 4.5))

    # 1. Measured vs True AGL
    axs[0].plot(alts, meas_agl, "ro-", label="ArUco Net AGL (m)")
    axs[0].plot(alts, alts, "k--", label="1:1 Ideal Reference")
    axs[0].set_title("ArUco Ground-Truth Height Recovery", fontweight="bold")
    axs[0].set_xlabel("True AGL Height (m)")
    axs[0].set_ylabel("Measured AGL (m)")
    axs[0].grid(True, alpha=0.3)
    axs[0].legend()

    # 2. Altitude Percentage Error
    axs[1].plot(alts, err_pct, "b^-", label="ArUco Height Error (%)")
    axs[1].axhline(2.0, color="red", linestyle="--", label="2% Accuracy Target")
    axs[1].set_title("Pose Estimation Percentage Error vs Altitude", fontweight="bold")
    axs[1].set_xlabel("True AGL Height (m)")
    axs[1].set_ylabel("Relative Error (%)")
    axs[1].set_ylim(0, max(3.0, max(err_pct) * 1.2))
    axs[1].grid(True, alpha=0.3)
    axs[1].legend()

    # 3. Marker Pixel Width vs Altitude
    axs[2].plot(alts, px_w, "g.-", label="Marker Apparent Width (px)")
    axs[2].axhline(10.0, color="orange", linestyle="--", label="Detection Resolution Limit (~10 px)")
    axs[2].set_title("Fiducial Resolution vs Altitude (f=1200px, L=0.2m)", fontweight="bold")
    axs[2].set_xlabel("True AGL Height (m)")
    axs[2].set_ylabel("Apparent Width (px)")
    axs[2].set_yscale("log")
    axs[2].grid(True, alpha=0.3)
    axs[2].legend()

    plt.tight_layout()
    plt.savefig(str(output_path), dpi=200)
    plt.close()
    print(f"ArUco benchmark plot saved to {output_path}")


if __name__ == "__main__":
    run_aruco_altitude_benchmark()
