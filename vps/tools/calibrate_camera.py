"""CLI tool and automated runner for camera calibration and focus inspection."""

from pathlib import Path
from typing import Optional, List
import matplotlib.pyplot as plt
import numpy as np
import cv2

from vps.camera.calibration import CheckerboardCalibrator, CalibrationResult, render_synthetic_checkerboard
from vps.tools.focus_inspector import InfinityFocusInspector
from vps.tools.baseline_calibrator import GroundBaselineCalibrator


def generate_synthetic_calibration_dataset(
    output_dir: Path = Path("data/calibration_samples"),
    num_views: int = 12,
    img_w: int = 1456,
    img_h: int = 1088,
) -> List[Path]:
    """Generate synthetic distorted checkerboard views for testing and verification."""
    output_dir.mkdir(parents=True, exist_ok=True)
    board = render_synthetic_checkerboard(pattern_size=(9, 6), square_size_px=45)
    bh, bw = board.shape

    # Nominal IMX296 camera matrix
    K_true = np.array([
        [1207.2, 0.0, 728.0],
        [0.0, 1199.1, 544.0],
        [0.0, 0.0, 1.0]
    ], dtype=np.float64)

    rng = np.random.default_rng(202)
    saved_paths = []

    for idx in range(num_views):
        # Create random 3D pose for target board in front of camera
        # Board corners in object space
        obj_corners = np.array([
            [0, 0, 0],
            [bw, 0, 0],
            [bw, bh, 0],
            [0, bh, 0]
        ], dtype=np.float32)

        # Distance: 0.6m to 1.2m
        tz = rng.uniform(0.65, 1.1)
        tx = rng.uniform(-0.15, 0.15)
        ty = rng.uniform(-0.10, 0.10)
        tvec = np.array([tx, ty, tz], dtype=np.float64)

        # Orientations: roll, pitch, yaw up to 25 degrees
        r_roll = np.radians(rng.uniform(-20, 20))
        r_pitch = np.radians(rng.uniform(-20, 20))
        r_yaw = np.radians(rng.uniform(-15, 15))
        rvec = np.array([r_roll, r_pitch, r_yaw], dtype=np.float64)

        # Scale board coordinates from pixels to meters
        scale_m = 0.025 / 45.0  # 25mm per square
        obj_pts_m = obj_corners * scale_m

        # Project 4 board corners to image plane
        img_corners, _ = cv2.projectPoints(obj_pts_m, rvec, tvec, K_true, None)
        img_corners = img_corners.reshape(-1, 2)

        # Warp checkerboard image to camera frame
        src_pts = np.array([[0, 0], [bw, 0], [bw, bh], [0, bh]], dtype=np.float32)
        H_warp = cv2.getPerspectiveTransform(src_pts, img_corners.astype(np.float32))

        view = cv2.warpPerspective(board, H_warp, (img_w, img_h), borderValue=200)
        # Add slight sensor noise
        noise = rng.normal(0, 1.5, size=view.shape)
        view = np.clip(view.astype(np.float32) + noise, 0, 255).astype(np.uint8)

        fn = output_dir / f"calib_{idx:02d}.png"
        cv2.imwrite(str(fn), view)
        saved_paths.append(fn)

    return saved_paths


def run_full_calibration(
    images_dir: Path = Path("data/calibration_samples"),
    output_yaml: Path = Path("config/imx296_calibrated.yaml"),
    output_plot: Path = Path("reports/m3_calibration_report.png"),
) -> CalibrationResult:
    """Run full calibration pipeline and generate report plot."""
    images_paths = sorted(list(images_dir.glob("*.png")) + list(images_dir.glob("*.jpg")))
    if len(images_paths) < 5:
        print(f"Generating synthetic calibration views in {images_dir}...")
        images_paths = generate_synthetic_calibration_dataset(images_dir, num_views=10)

    images = [cv2.imread(str(p)) for p in images_paths]
    calibrator = CheckerboardCalibrator(pattern_size=(9, 6), square_size_m=0.025)

    print(f"Running camera calibration on {len(images)} frames...")
    res = calibrator.calibrate_from_images(images)

    if res.is_valid:
        print(f"Calibration successful! Mean Reprojection Error: {res.mean_reprojection_error_px:.4f} px")
        print(f"Recovered fx: {res.K[0,0]:.2f}, fy: {res.K[1,1]:.2f}, cx: {res.K[0,2]:.2f}, cy: {res.K[1,2]:.2f}")
        calibrator.save_to_yaml(res, output_yaml)
        print(f"Saved calibrated intrinsics to {output_yaml}")
    else:
        print(f"[WARN] Calibration failed or high error ({res.mean_reprojection_error_px:.3f} px)")

    # Generate M3 Validation & Focus Plot
    _plot_m3_results(res, images[0], output_plot)
    return res


def _plot_m3_results(res: CalibrationResult, sample_img: np.ndarray, output_plot: Path) -> None:
    """Generate diagnostic plot of calibration and focus analysis."""
    output_plot.parent.mkdir(parents=True, exist_ok=True)
    fig, axs = plt.subplots(1, 3, figsize=(16, 5))

    # 1. Reprojection error per view
    axs[0].bar(range(1, len(res.per_image_errors_px) + 1), res.per_image_errors_px, color='royalblue', edgecolor='black')
    axs[0].axhline(res.mean_reprojection_error_px, color='crimson', linestyle='--', label=f'Mean Error ({res.mean_reprojection_error_px:.3f} px)')
    axs[0].set_title("Per-View Reprojection Error (px)", fontsize=11, fontweight='bold')
    axs[0].set_xlabel("Calibration Image Index")
    axs[0].set_ylabel("Error (px)")
    axs[0].grid(True, alpha=0.3)
    axs[0].legend()

    # 2. Focus analysis on sample frame
    inspector = InfinityFocusInspector()
    focus_metrics = inspector.evaluate_sharpness(sample_img)
    overlay = inspector.draw_focus_overlay(sample_img, focus_metrics)
    axs[1].imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
    axs[1].set_title(f"Focus Quality: {focus_metrics.focus_status}\n(LapVar: {focus_metrics.laplacian_variance:.1f})", fontsize=11, fontweight='bold')
    axs[1].axis('off')

    # 3. Spatial Sharpness Map (9-Zones)
    zones = focus_metrics.zone_sharpness
    grid_vals = np.array([
        [zones["top_left"], zones["top_center"], zones["top_right"]],
        [zones["mid_left"], zones["center"],     zones["mid_right"]],
        [zones["bot_left"], zones["bot_center"], zones["bot_right"]],
    ])
    im = axs[2].imshow(grid_vals, cmap='viridis')
    axs[2].set_title("3x3 Spatial Sharpness Heatmap\n(Lens Optical Flatness)", fontsize=11, fontweight='bold')
    axs[2].set_xticks([0, 1, 2], ["Left", "Center", "Right"])
    axs[2].set_yticks([0, 1, 2], ["Top", "Middle", "Bottom"])
    fig.colorbar(im, ax=axs[2], fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig(str(output_plot), dpi=200)
    plt.close()
    print(f"Calibration & focus report saved to {output_plot}")


if __name__ == "__main__":
    run_full_calibration()
