"""Script to generate validation plots for Milestone 1."""

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

from vps.geometry.pinhole import PinholeModel
from vps.geometry.derotation import GyroDerotationEngine
from vps.geometry.height_flow import HeightFromFlowEngine
from vps.geometry.scale_expansion import ScaleFromExpansionEngine


def generate_m1_report_plots(output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: GSD vs Altitude (30 to 300 m)
    K = np.array([[1207.2, 0.0, 728.0], [0.0, 1199.1, 544.0], [0.0, 0.0, 1.0]])
    pinhole = PinholeModel(K=K, width=1456, height=1088)
    alts = np.linspace(30.0, 300.0, 100)
    gsd_x = [pinhole.compute_gsd(h)[0] * 100 for h in alts]  # cm/px
    gsd_y = [pinhole.compute_gsd(h)[1] * 100 for h in alts]

    axs[0, 0].plot(alts, gsd_x, 'b-', label='GSD X (horizontal)', linewidth=2)
    axs[0, 0].plot(alts, gsd_y, 'r--', label='GSD Y (vertical)', linewidth=2)
    axs[0, 0].axhline(24.85, color='gray', linestyle=':', label='300m reference (~25 cm/px)')
    axs[0, 0].set_title("Ground Sample Distance (GSD) vs Altitude AGL")
    axs[0, 0].set_xlabel("Altitude AGL (m)")
    axs[0, 0].set_ylabel("GSD (cm / pixel)")
    axs[0, 0].grid(True, alpha=0.3)
    axs[0, 0].legend()

    # Plot 2: De-rotation residual flow across rotation angles
    derot_engine = GyroDerotationEngine(K=K)
    angles_deg = np.linspace(0.1, 10.0, 50)
    residuals = []
    rng = np.random.default_rng(10)
    pts_k = rng.uniform(low=[200, 200], high=[1200, 800], size=(40, 2))

    for deg in angles_deg:
        R_cam = derot_engine.rotation_matrix_from_rpy_delta(np.radians(deg), np.radians(deg * 0.7), np.radians(deg * 0.3))
        pts_c = derot_engine.derotate_points(pts_k, R_cam)
        _, flow_trans = derot_engine.compute_translational_flow(pts_k, pts_c, R_cam)
        max_err = np.max(np.hypot(flow_trans[:, 0], flow_trans[:, 1]))
        residuals.append(max_err)

    axs[0, 1].plot(angles_deg, residuals, 'g-', linewidth=2)
    axs[0, 1].set_title("Pure-Rotation Cancellation Residual (Homography Derotation)")
    axs[0, 1].set_xlabel("Attitude Perturbation Magnitude (deg)")
    axs[0, 1].set_ylabel("Translational Flow Residual (px)")
    axs[0, 1].set_ylim(-1e-6, 1e-6)
    axs[0, 1].grid(True, alpha=0.3)

    # Plot 3: Height from Flow estimation across 30m to 300m
    flow_engine = HeightFromFlowEngine(fx=pinhole.fx, fy=pinhole.fy)
    test_altitudes = np.array([30, 50, 75, 100, 150, 200, 250, 300])
    est_altitudes = []
    disp = np.array([4.0, 0.0])  # 4m ground displacement (at 10 m/s for 0.4s)

    for h in test_altitudes:
        # Expected flow: du = fx * dx / h
        du = pinhole.fx * disp[0] / h
        flow = np.column_stack([np.full(50, du), np.zeros(50)])
        # Add 0.3 px tracking noise
        flow += rng.normal(0.0, 0.3, size=flow.shape)
        res = flow_engine.estimate_height(flow, disp)
        est_altitudes.append(res.height_m)

    est_altitudes = np.array(est_altitudes)
    errors_pct = (est_altitudes - test_altitudes) / test_altitudes * 100

    axs[1, 0].bar(test_altitudes, errors_pct, width=12, color='teal', alpha=0.75, edgecolor='black')
    axs[1, 0].axhline(0, color='black', linewidth=1)
    axs[1, 0].set_title("Height-from-Flow Relative Error (with 0.3px noise)")
    axs[1, 0].set_xlabel("True Altitude AGL (m)")
    axs[1, 0].set_ylabel("Error (%)")
    axs[1, 0].set_ylim(-2.0, 2.0)
    axs[1, 0].grid(True, alpha=0.3)

    # Plot 4: Scale Expansion Recovery
    scale_engine = ScaleFromExpansionEngine()
    climb_descent_alts = [(50, 40), (80, 60), (120, 100), (200, 170), (300, 260)]
    recovered_h1 = []

    for h1, h2 in climb_descent_alts:
        pts1 = rng.uniform(low=[100, 100], high=[1300, 900], size=(60, 2))
        c = np.array([728.0, 544.0])
        scale = h1 / h2
        pts2 = c + (pts1 - c) * scale
        delta_baro = h1 - h2
        res = scale_engine.estimate_height_from_expansion(pts1, pts2, delta_baro)
        recovered_h1.append(res.height_h1_m)

    true_h1 = [p[0] for p in climb_descent_alts]
    axs[1, 1].plot(true_h1, true_h1, 'k--', label='Ideal 1:1 line', linewidth=1.5)
    axs[1, 1].scatter(true_h1, recovered_h1, color='darkorange', s=80, zorder=3, label='Estimated H1')
    axs[1, 1].set_title("Scale-from-Expansion Absolute Height Recovery")
    axs[1, 1].set_xlabel("True Initial Altitude H1 (m)")
    axs[1, 1].set_ylabel("Recovered Initial Altitude H1 (m)")
    axs[1, 1].legend()
    axs[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(str(output_path), dpi=200)
    plt.close()
    print(f"Validation plots successfully written to {output_path}")


if __name__ == "__main__":
    generate_m1_report_plots(Path("reports/m1_geometry_validation.png"))
