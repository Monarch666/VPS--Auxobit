"""Automated simulator evaluation of height-from-flow from 30 m to 300 m."""

from pathlib import Path
from typing import List, Dict, Any, Tuple
import matplotlib.pyplot as plt
import numpy as np
import cv2

from vps.config import VPSConfig, CameraConfig, TrackerConfig
from vps.geometry.pinhole import PinholeModel
from vps.geometry.derotation import GyroDerotationEngine
from vps.geometry.height_flow import HeightFromFlowEngine
from vps.tracker.feature_tracker import FeatureTracker
from vps.tools.terrain_generator import DeccanTerrainGenerator
from vps.tools.simulator import DroneCameraSimulator, DroneState, SimulatorDegradationConfig


def run_altitude_benchmark(
    terrain_path: Path = Path("data/synthetic_terrain_deccan.png"),
    output_plot_path: Path = Path("reports/m2_simulator_altitude_benchmark.png"),
    altitudes: Optional[List[float]] = None,
) -> Dict[str, Any]:
    """Run end-to-end height-from-flow benchmark across 30 m to 300 m AGL.

    Args:
        terrain_path: Path to cached or output terrain orthophoto.
        output_plot_path: Destination path for visualization plot.
        altitudes: List of altitudes to test (defaults to 30, 50, 75, 100, 150, 200, 250, 300 m).

    Returns:
        Dictionary of results containing errors, inliers, and validation statuses.
    """
    if altitudes is None:
        altitudes = [30.0, 50.0, 75.0, 100.0, 150.0, 200.0, 250.0, 300.0]

    # 1. Load or Generate Terrain
    if terrain_path.is_file():
        ortho = cv2.imread(str(terrain_path))
    else:
        print(f"Generating synthetic Deccan Plateau terrain at {terrain_path}...")
        gen = DeccanTerrainGenerator(width_px=3072, height_px=3072, meters_per_pixel=0.1)
        ortho = gen.generate(terrain_path)

    # 2. Setup Simulator & Algorithms
    cam_cfg = CameraConfig()
    deg_cfg = SimulatorDegradationConfig(
        exposure_time_s=0.002,
        haze_beta=0.0012,
        sensor_noise_sigma=1.5,
    )
    sim = DroneCameraSimulator(
        orthophoto=ortho,
        meters_per_pixel=0.1,
        camera_config=cam_cfg,
        degradation_config=deg_cfg,
        origin_ned_m=(0.0, 0.0),
    )

    pinhole = PinholeModel(cam_cfg.K, cam_cfg.D, cam_cfg.width, cam_cfg.height)
    tracker = FeatureTracker(TrackerConfig(max_features=300, min_keyframe_displacement_px=10.0))
    derot_engine = GyroDerotationEngine(cam_cfg.K)
    flow_engine = HeightFromFlowEngine(fx=cam_cfg.fx, fy=cam_cfg.fy, min_displacement_px=10.0)

    results = []
    sample_frames = {}

    nominal_speed = 10.0  # m/s forward (North)
    # Camera mounts: optical axis Z_c down, X_c right (East), Y_c down across image (South / -North)
    # Moving North (+X_w) means in camera coordinates moving along -Y_c!

    print(f"Running Height-from-Flow Altitude Benchmark (30 m to 300 m)...")

    for alt in altitudes:
        # Compute baseline distance guaranteeing >= 10 px flow at this altitude:
        # flow_px = fx * ground_disp / alt >= 10 px -> ground_disp >= 10 * alt / fx
        min_disp = 10.0 * alt / cam_cfg.fx
        # Add 50% safety margin: target ~15 px flow
        target_disp = max(min_disp * 1.5, 0.4)
        dt = target_disp / nominal_speed  # baseline time interval

        # Drone trajectory: forward flight with pitch/roll wind gust oscillation
        t0 = 0.0
        pos0 = np.array([150.0, 150.0, -alt], dtype=np.float64)
        vel = np.array([nominal_speed, 0.0, 0.0], dtype=np.float64)  # 10 m/s North
        att0 = np.array([np.radians(0.5), np.radians(1.0), 0.0], dtype=np.float64)
        omega0 = np.array([0.01, -0.02, 0.0], dtype=np.float64)

        state0 = DroneState(timestamp_s=t0, pos_ned_m=pos0, vel_ned_mps=vel, attitude_rpy_rad=att0, angular_rates_rad_s=omega0)
        frame0 = sim.render_frame(state0)
        gray0 = cv2.cvtColor(frame0, cv2.COLOR_BGR2GRAY)

        # Current frame after dt
        t1 = t0 + dt
        pos1 = pos0 + vel * dt
        # Wind perturbation on attitude
        att1 = np.array([
            att0[0] + np.radians(1.2 * np.sin(2.0 * np.pi * 0.5 * dt)),
            att0[1] + np.radians(0.8 * np.cos(2.0 * np.pi * 0.4 * dt)),
            att0[2] + np.radians(0.3 * dt)
        ], dtype=np.float64)
        omega1 = (att1 - att0) / dt

        state1 = DroneState(timestamp_s=t1, pos_ned_m=pos1, vel_ned_mps=vel, attitude_rpy_rad=att1, angular_rates_rad_s=omega1)
        frame1 = sim.render_frame(state1)
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)

        if int(alt) in [30, 100, 300]:
            sample_frames[int(alt)] = frame1

        # 3. Detect and Track Features
        pts0 = tracker.detect_features(gray0)
        track_res = tracker.track(gray0, gray1, pts0)

        # 4. De-rotate features using inter-frame rotation
        # Relative camera rotation R_k2c
        R_w_b0 = sim._euler_to_rotation_matrix(*att0)
        R_w_b1 = sim._euler_to_rotation_matrix(*att1)
        # R_c0_to_c1 = (R_w_c1)^T * R_w_c0 = R_c2b.T * R_w_b1.T * R_w_b0 * R_c2b
        R_cam_rel = sim.R_b2c @ (R_w_b1.T @ R_w_b0) @ sim.R_c2b

        inlier_pts0 = track_res.pts_prev[track_res.inlier_mask]
        inlier_pts1 = track_res.pts_curr[track_res.inlier_mask]

        if len(inlier_pts0) < 8:
            print(f"  [WARN] Alt {alt:.0f}m: Insufficient inliers ({len(inlier_pts0)})")
            results.append({
                "alt_true": alt,
                "alt_est": 0.0,
                "error_pct": 100.0,
                "inliers": len(inlier_pts0),
                "confidence": 0.0,
                "is_valid": False,
            })
            continue

        _, flow_trans = derot_engine.compute_translational_flow(inlier_pts0, inlier_pts1, R_cam_rel)

        # 5. Ground displacement in camera frame
        # World displacement: delta_pos_w = pos1 - pos0
        delta_pos_w = pos1 - pos0
        # Average body frame displacement: delta_pos_b = R_w_b0.T * delta_pos_w
        delta_pos_b = R_w_b0.T @ delta_pos_w
        # Camera frame displacement: delta_pos_c = R_b2c * delta_pos_b
        delta_pos_c = sim.R_b2c @ delta_pos_b

        # 6. Estimate Height
        est_res = flow_engine.estimate_height(flow_trans, delta_pos_c)

        err_m = abs(est_res.height_m - alt)
        err_pct = (err_m / alt) * 100.0 if est_res.is_valid else 100.0

        print(f"  Alt {alt:5.1f}m -> Est: {est_res.height_m:5.1f}m | Error: {err_pct:4.2f}% | "
              f"Flow: {est_res.median_flow_px:4.1f}px | Inliers: {est_res.num_inliers:3d} | Conf: {est_res.confidence:.2f}")

        results.append({
            "alt_true": alt,
            "alt_est": est_res.height_m,
            "err_m": err_m,
            "error_pct": err_pct,
            "inliers": est_res.num_inliers,
            "median_flow_px": est_res.median_flow_px,
            "confidence": est_res.confidence,
            "is_valid": est_res.is_valid,
        })

    # 7. Generate Performance Visualization Plot
    _plot_altitude_benchmark(results, sample_frames, output_plot_path)

    return {"results": results, "plot_path": str(output_plot_path)}


def _plot_altitude_benchmark(
    results: List[Dict[str, Any]],
    sample_frames: Dict[int, np.ndarray],
    output_path: Path,
) -> None:
    """Generate comprehensive multi-panel evaluation plot."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(15, 11))
    gs = fig.add_gridspec(2, 3)

    # Panel 1: Rendered frames at 30m, 100m, 300m
    for idx, alt in enumerate([30, 100, 300]):
        if alt in sample_frames:
            ax = fig.add_subplot(gs[0, idx])
            img_rgb = cv2.cvtColor(sample_frames[alt], cv2.COLOR_BGR2RGB)
            ax.imshow(img_rgb)
            ax.set_title(f"Simulated Frame @ {alt}m AGL\n(Haze & Sensor Noise)")
            ax.axis('off')

    # Panel 2: Altitude Estimation vs Ground Truth
    ax2 = fig.add_subplot(gs[1, :2])
    alts_true = [r["alt_true"] for r in results]
    alts_est = [r["alt_est"] for r in results]

    ax2.plot(alts_true, alts_true, 'k--', label='Ideal Ground Truth (1:1)', linewidth=2)
    # 5% Acceptance Bounds
    upper_5 = [a * 1.05 for a in alts_true]
    lower_5 = [a * 0.95 for a in alts_true]
    ax2.fill_between(alts_true, lower_5, upper_5, color='green', alpha=0.15, label='±5% Acceptance Corridor')

    ax2.plot(alts_true, alts_est, 'o-', color='navy', markersize=8, linewidth=2.5, label='VPS Height-from-Flow')
    ax2.set_title("Above Ground Level (AGL) Estimation vs Ground Truth", fontsize=12, fontweight='bold')
    ax2.set_xlabel("True Altitude AGL (m)", fontsize=11)
    ax2.set_ylabel("Estimated Altitude AGL (m)", fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=10)

    # Panel 3: Relative Error (%) vs Altitude
    ax3 = fig.add_subplot(gs[1, 2])
    errors_pct = [r["error_pct"] for r in results]
    bars = ax3.bar(alts_true, errors_pct, width=16, color='steelblue', edgecolor='black', alpha=0.85)

    # Highlight <= 5% threshold
    ax3.axhline(5.0, color='crimson', linestyle='--', linewidth=2, label='5% Acceptance Limit')
    ax3.set_title("Relative Error (%) vs Altitude", fontsize=12, fontweight='bold')
    ax3.set_xlabel("Altitude AGL (m)", fontsize=11)
    ax3.set_ylabel("Absolute Error (%)", fontsize=11)
    ax3.set_ylim(0, max(6.0, max(errors_pct) * 1.2))
    ax3.grid(True, alpha=0.3)
    ax3.legend(fontsize=10)

    for bar, err in zip(bars, errors_pct):
        yval = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width() / 2.0, yval + 0.15, f"{err:.1f}%", ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(str(output_path), dpi=200)
    plt.close()
    print(f"Benchmark plot successfully saved to {output_path}")


if __name__ == "__main__":
    run_altitude_benchmark()

