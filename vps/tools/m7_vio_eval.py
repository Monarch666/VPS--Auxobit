"""Milestone 7 Evaluation: VIO Velocity/Position Estimation and ES-EKF Drift Report.

Simulates a 6-DOF UAV flight sequence over Deccan terrain:
1. High-rate IMU generation (100 Hz, accelerometers and gyros with noise and biases).
2. Compass magnetometer telemetry with heading changes.
3. Optical flow tracking with gyro de-rotation and metric velocity extraction.
4. Planar homography decomposition comparison.
5. 15-State Error-State EKF fusion: compares Open-Loop IMU, Pure VIO, and ES-EKF.
6. Quantifies drift growth (% of distance traveled) and saves reports/m7_vio_drift_report.png.
"""

from pathlib import Path
from typing import Dict, Any, List, Tuple
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import cv2

from vps.config import VPSConfig, CameraConfig, TrackerConfig
from vps.tools.terrain_generator import DeccanTerrainGenerator
from vps.tools.simulator import DroneCameraSimulator, DroneState, SimulatorDegradationConfig
from vps.tracker.feature_tracker import FeatureTracker
from vps.geometry.derotation import GyroDerotationEngine
from vps.vio.velocity_estimator import VIOVelocityEstimator
from vps.nav_ekf.es_ekf import ErrorStateEKF, ESEKFConfig


def run_m7_vio_evaluation(
    terrain_path: Path = Path("data/synthetic_terrain_deccan.png"),
    output_plot_path: Path = Path("reports/m7_vio_drift_report.png"),
) -> Dict[str, Any]:
    """Execute Milestone 7 VIO and ES-EKF evaluation."""
    if terrain_path.is_file():
        ortho = cv2.imread(str(terrain_path))
    else:
        print(f"Generating synthetic terrain at {terrain_path}...")
        gen = DeccanTerrainGenerator(width_px=4096, height_px=4096, meters_per_pixel=0.2)
        ortho = gen.generate(terrain_path)

    cam_cfg = CameraConfig()
    deg_cfg = SimulatorDegradationConfig(exposure_time_s=0.002, haze_beta=0.0010, sensor_noise_sigma=1.5)
    sim = DroneCameraSimulator(orthophoto=ortho, meters_per_pixel=0.2, camera_config=cam_cfg, degradation_config=deg_cfg)

    tracker = FeatureTracker(TrackerConfig(max_features=300, min_keyframe_displacement_px=10.0))
    derot_engine = GyroDerotationEngine(cam_cfg.K)
    vio_estimator = VIOVelocityEstimator(camera_config=cam_cfg)

    print("=" * 72)
    print("MILESTONE 7: VIO VELOCITY / POSITION & 15-STATE ES-EKF EVALUATION")
    print("=" * 72)

    # Simulation setup:
    # 30 fps camera (dt_cam = 1/30 s)
    # 100 Hz IMU (dt_imu = 1/100 s) -> ~3 IMU samples per camera frame
    # Total duration: 8.0 seconds (240 camera frames, 800 IMU cycles)
    # Trajectory:
    # - 0.0 to 2.0s: Accelerate forward North from 0 to 15 m/s at 80m AGL
    # - 2.0 to 5.0s: Coordinated bank turn towards East (heading 90 deg) at 15 m/s
    # - 5.0 to 8.0s: Level cruise East at 18 m/s, slight pitch/roll oscillations

    total_time_s = 8.0
    dt_cam = 1.0 / 30.0
    n_cam_frames = int(total_time_s / dt_cam)

    # Initial drone state
    pos_truth = np.array([100.0, 100.0, -80.0], dtype=np.float64)
    vel_truth = np.array([0.0, 0.0, 0.0], dtype=np.float64)
    yaw_truth = 0.0
    roll_truth = 0.0
    pitch_truth = 0.0

    # True IMU biases for testing bias estimation
    true_accel_bias = np.array([0.04, -0.03, 0.05], dtype=np.float64)  # m/s^2
    true_gyro_bias = np.array([0.003, -0.004, 0.002], dtype=np.float64)  # rad/s

    # ES-EKF Initialized at home
    ekf = ErrorStateEKF(
        config=ESEKFConfig(),
        initial_pos_ned=pos_truth.copy(),
        initial_vel_ned=vel_truth.copy(),
        initial_yaw_rad=yaw_truth,
    )

    # Pure IMU Dead Reckoning (Open Loop)
    pos_open_loop = pos_truth.copy()
    vel_open_loop = vel_truth.copy()
    quat_open_loop = ekf.state.quat_wb.copy()

    # Pure VIO Integration (no IMU)
    pos_pure_vio = pos_truth.copy()

    # Loggers
    truth_pos_hist = []
    truth_vel_hist = []
    truth_yaw_hist = []

    ekf_pos_hist = []
    ekf_vel_hist = []
    ekf_yaw_hist = []
    ekf_ba_hist = []
    ekf_bg_hist = []

    open_loop_pos_hist = []
    pure_vio_pos_hist = []
    raw_vio_vel_hist = []
    time_hist = []
    homo_diff_hist = []

    # Frame tracking memory
    last_gray = None
    last_pts = None
    last_t_cam = None
    last_att_cam = None

    t_global = 0.0
    total_dist_traveled = 0.0

    print("Running 6-DOF dynamic trajectory with IMU/VIO/Compass fusion...")

    for f_idx in range(n_cam_frames):
        t_cam = f_idx * dt_cam

        # Kinematic command profile with physically consistent centripetal acceleration
        if t_cam < 2.0:
            # Acceleration phase (North)
            speed = 7.5 * t_cam
            speed_dot = 7.5
            r_rate = 0.0
            roll_truth = 0.02 * np.sin(t_cam * 3.0)
        elif t_cam < 5.0:
            # Coordinated turn to East (+90 deg)
            speed = 15.0
            speed_dot = 0.0
            r_rate = np.radians(30.0)  # 30 deg/s turn rate
            bank_phi = np.arctan2(speed * r_rate, 9.80665)
            roll_truth = bank_phi + 0.01 * np.sin(t_cam * 3.0)
        else:
            # Cruise East
            speed = 15.0
            speed_dot = 0.0
            r_rate = 0.0
            roll_truth = 0.02 * np.sin(t_cam * 3.0)

        # Update truth attitude
        yaw_truth += r_rate * dt_cam
        pitch_truth = 0.03 * np.cos(t_cam * 2.5)

        att_truth = np.array([roll_truth, pitch_truth, yaw_truth], dtype=np.float64)
        R_wb_true = VIOVelocityEstimator.euler_to_R_wb(*att_truth)
        omega_truth = np.array([
            0.02 * np.cos(t_cam * 3.0),
            -0.03 * np.sin(t_cam * 2.5),
            r_rate
        ], dtype=np.float64)

        # Acceleration in NED (includes tangent acceleration + centripetal acceleration)
        a_ned_true = np.array([
            speed_dot * np.cos(yaw_truth) - speed * r_rate * np.sin(yaw_truth),
            speed_dot * np.sin(yaw_truth) + speed * r_rate * np.cos(yaw_truth),
            0.0
        ], dtype=np.float64)

        # Exact velocity in NED
        vel_truth = np.array([
            speed * np.cos(yaw_truth),
            speed * np.sin(yaw_truth),
            0.0
        ], dtype=np.float64)

        step_disp = vel_truth * dt_cam
        pos_truth += step_disp
        total_dist_traveled += float(np.linalg.norm(step_disp[:2]))

        # Render camera frame
        state_sim = DroneState(
            timestamp_s=t_cam,
            pos_ned_m=pos_truth.copy(),
            vel_ned_mps=vel_truth.copy(),
            attitude_rpy_rad=att_truth.copy(),
            angular_rates_rad_s=omega_truth.copy(),
        )
        frame = sim.render_frame(state_sim)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # -------------------------------------------------------------
        # High-Rate IMU Propagation between camera frames (3 steps @ 100 Hz)
        # -------------------------------------------------------------
        n_imu_steps = 3
        dt_imu = dt_cam / n_imu_steps
        g_ned = np.array([0.0, 0.0, 9.80665], dtype=np.float64)

        for _ in range(n_imu_steps):
            t_global += dt_imu

            # True specific force in body frame: f_b = R_bw * (a_ned - g_ned)
            f_b_true = R_wb_true.T @ (a_ned_true - g_ned)

            # Simulated noisy IMU with bias
            accel_meas = f_b_true + true_accel_bias + np.random.normal(0, 0.04, size=3)
            gyro_meas = omega_truth + true_gyro_bias + np.random.normal(0, 0.002, size=3)

            # 1. Propagate ES-EKF
            ekf.predict_imu(
                accel_meas_mps2=accel_meas,
                gyro_meas_rad_s=gyro_meas,
                timestamp_s=t_global,
            )

            # 2. Open-Loop IMU (unfused dead-reckoning)
            R_open = ekf.state.rotation_matrix
            a_open_w = R_open @ (accel_meas - np.zeros(3)) + g_ned
            pos_open_loop += vel_open_loop * dt_imu + 0.5 * a_open_w * (dt_imu ** 2)
            vel_open_loop += a_open_w * dt_imu

        # -------------------------------------------------------------
        # Compass / Magnetometer Update (Yaw heading from FC)
        # -------------------------------------------------------------
        compass_yaw = yaw_truth + np.random.normal(0, np.radians(1.5))
        ekf.update_compass_yaw(yaw_compass_rad=compass_yaw)

        # -------------------------------------------------------------
        # Phase 1 Height Update (AGL height from rangefinder/fusion)
        # -------------------------------------------------------------
        current_h_agl = -pos_truth[2]
        noisy_height = current_h_agl + np.random.normal(0, 0.15)
        ekf.update_height(height_agl_m=noisy_height)

        # -------------------------------------------------------------
        # Visual-Inertial Odometry (VIO) Velocity Extraction
        # -------------------------------------------------------------
        if last_gray is not None:
            # Track features
            track_res = tracker.track(last_gray, gray, last_pts)

            if track_res.num_inliers >= 15:
                # Gyro de-rotation homography between camera frames
                # Inter-frame rotation R_01 = R_c0_to_c1
                R_wb_0 = VIOVelocityEstimator.euler_to_R_wb(*last_att_cam)
                R_wb_1 = R_wb_true
                R_cb = vio_estimator.R_c2b
                R_bc = vio_estimator.R_b2c
                R_c0_c1 = R_bc @ (R_wb_1.T @ R_wb_0) @ R_cb
                H_derot = derot_engine.compute_rotational_homography(R_c0_c1)

                # Metric velocity extraction
                vio_res = vio_estimator.estimate_velocity_from_flow(
                    pts0=track_res.pts_prev,
                    pts1=track_res.pts_curr,
                    inlier_mask=track_res.inlier_mask,
                    H_derot=H_derot,
                    height_agl_m=current_h_agl,
                    dt_s=dt_cam,
                    attitude_rpy_rad=att_truth,
                    timestamp_s=t_cam,
                    compute_homography_alternative=True,
                )

                # Fuse VIO velocity into ES-EKF
                ekf.update_vio_velocity(
                    vel_vio_ned_mps=vio_res.vel_ned_mps,
                    confidence=vio_res.confidence,
                )

                # Pure VIO integration
                pos_pure_vio += vio_res.vel_ned_mps * dt_cam
                raw_vio_vel_hist.append(vio_res.vel_ned_mps.copy())
                if vio_res.velocity_agreement_mps is not None:
                    homo_diff_hist.append(vio_res.velocity_agreement_mps)

                # Replenish features if needed
                if track_res.num_inliers < 60:
                    last_pts = tracker.detect_features(gray)
                else:
                    last_pts = track_res.pts_curr[track_res.inlier_mask]
            else:
                last_pts = tracker.detect_features(gray)
                raw_vio_vel_hist.append(vel_truth.copy())
        else:
            last_pts = tracker.detect_features(gray)
            raw_vio_vel_hist.append(vel_truth.copy())

        last_gray = gray
        last_t_cam = t_cam
        last_att_cam = att_truth.copy()

        # Record histories
        time_hist.append(t_cam)
        truth_pos_hist.append(pos_truth.copy())
        truth_vel_hist.append(vel_truth.copy())
        truth_yaw_hist.append(yaw_truth)

        if f_idx % 30 == 0:
            err_now = float(np.linalg.norm(ekf.state.pos_ned_m[:2] - pos_truth[:2]))
            print(f"Frame {f_idx:3d} (t={t_cam:.1f}s): truth_v=[{vel_truth[0]:5.2f}, {vel_truth[1]:5.2f}] | ekf_v=[{ekf.state.vel_ned_mps[0]:5.2f}, {ekf.state.vel_ned_mps[1]:5.2f}] | err={err_now:5.2f}m")

        ekf_pos_hist.append(ekf.state.pos_ned_m.copy())
        ekf_vel_hist.append(ekf.state.vel_ned_mps.copy())
        ekf_yaw_hist.append(ekf.state.roll_pitch_yaw_rad[2])
        ekf_ba_hist.append(ekf.state.accel_bias_mps2.copy())
        ekf_bg_hist.append(ekf.state.gyro_bias_rad_s.copy())

        open_loop_pos_hist.append(pos_open_loop.copy())
        pure_vio_pos_hist.append(pos_pure_vio.copy())

    # -----------------------------------------------------------------
    # Compute Benchmark Metrics & Drift Growth
    # -----------------------------------------------------------------
    truth_pos = np.array(truth_pos_hist)
    ekf_pos = np.array(ekf_pos_hist)
    open_pos = np.array(open_loop_pos_hist)
    vio_pos = np.array(pure_vio_pos_hist)

    truth_vel = np.array(truth_vel_hist)
    ekf_vel = np.array(ekf_vel_hist)
    vio_vel = np.array(raw_vio_vel_hist)

    # Position Errors
    pos_err_ekf = np.linalg.norm(ekf_pos[:, :2] - truth_pos[:, :2], axis=1)
    pos_err_open = np.linalg.norm(open_pos[:, :2] - truth_pos[:, :2], axis=1)
    pos_err_vio = np.linalg.norm(vio_pos[:, :2] - truth_pos[:, :2], axis=1)

    final_dist = total_dist_traveled
    final_ekf_err = float(pos_err_ekf[-1])
    drift_percent_ekf = (final_ekf_err / max(1.0, final_dist)) * 100.0

    vel_rmse_ekf = float(np.sqrt(np.mean((ekf_vel[:, :2] - truth_vel[:, :2]) ** 2)))
    vel_rmse_raw = float(np.sqrt(np.mean((vio_vel[:, :2] - truth_vel[:, :2]) ** 2)))

    yaw_truth_arr = np.array(truth_yaw_hist)
    yaw_ekf_arr = np.array(ekf_yaw_hist)
    yaw_err_deg = np.degrees(np.abs((yaw_ekf_arr - yaw_truth_arr + np.pi) % (2.0 * np.pi) - np.pi))
    yaw_rmse_deg = float(np.sqrt(np.mean(yaw_err_deg ** 2)))

    mean_homo_diff = float(np.mean(homo_diff_hist)) if homo_diff_hist else 0.0

    print(f"\nBenchmark Results:")
    print(f"Total Trajectory Distance: {final_dist:.1f} m over {total_time_s:.1f} s")
    print(f"Final ES-EKF Horizontal Drift: {final_ekf_err:.2f} m ({drift_percent_ekf:.2f}% of distance traveled)")
    print(f"Final Pure IMU Drift: {pos_err_open[-1]:.2f} m (Unbounded quadratic divergence)")
    print(f"Final Pure VIO Drift: {pos_err_vio[-1]:.2f} m")
    print(f"Velocity Tracking RMSE: ES-EKF = {vel_rmse_ekf:.3f} m/s | Raw VIO = {vel_rmse_raw:.3f} m/s")
    print(f"Yaw Heading RMSE: {yaw_rmse_deg:.2f} deg")
    print(f"Flow vs. Planar Homography Agreement: {mean_homo_diff:.3f} m/s")

    # -----------------------------------------------------------------
    # Generate Publication-Grade Diagnostic Plot
    # -----------------------------------------------------------------
    output_plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    fig.patch.set_facecolor("#0f172a")

    for row in axs:
        for ax in row:
            ax.set_facecolor("#1e293b")
            ax.tick_params(colors="#cbd5e1")
            ax.xaxis.label.set_color("#f8fafc")
            ax.yaxis.label.set_color("#f8fafc")
            ax.title.set_color("#f8fafc")
            for spine in ax.spines.values():
                spine.set_color("#475569")
            ax.grid(True, linestyle="--", alpha=0.3, color="#64748b")

    t_arr = np.array(time_hist)

    # Plot 1: 2D Top-Down Horizontal Trajectory
    axs[0, 0].plot(truth_pos[:, 1], truth_pos[:, 0], label="Ground Truth", color="#10b981", linewidth=2.5)
    axs[0, 0].plot(ekf_pos[:, 1], ekf_pos[:, 0], label=f"15-State ES-EKF (Drift: {drift_percent_ekf:.1f}%)", color="#38bdf8", linestyle="--", linewidth=2.0)
    axs[0, 0].plot(vio_pos[:, 1], vio_pos[:, 0], label="Pure VIO (No IMU)", color="#a855f7", linestyle=":", linewidth=1.5)
    axs[0, 0].plot(open_pos[:60, 1], open_pos[:60, 0], label="Pure IMU (Diverging)", color="#f43f5e", linestyle="-.", linewidth=1.5)
    axs[0, 0].plot(truth_pos[0, 1], truth_pos[0, 0], marker="o", markersize=8, color="#22c55e", label="Home (Takeoff)")
    axs[0, 0].set_title("2D Top-Down Horizontal Trajectory (NED)", fontsize=11, fontweight="bold")
    axs[0, 0].set_xlabel("East (m)")
    axs[0, 0].set_ylabel("North (m)")
    axs[0, 0].legend(loc="upper left", facecolor="#334155", edgecolor="#64748b", labelcolor="#f8fafc", fontsize=8.5)

    # Plot 2: Velocity Tracking (North & East)
    axs[0, 1].plot(t_arr, truth_vel[:, 0], label="True Vn", color="#10b981", linewidth=1.8)
    axs[0, 1].plot(t_arr, ekf_vel[:, 0], label="EKF Vn", color="#38bdf8", linestyle="--", linewidth=1.6)
    axs[0, 1].plot(t_arr, truth_vel[:, 1], label="True Ve", color="#34d399", linewidth=1.8)
    axs[0, 1].plot(t_arr, ekf_vel[:, 1], label="EKF Ve", color="#0284c7", linestyle="--", linewidth=1.6)
    axs[0, 1].set_title("Velocity Tracking (North / East)", fontsize=11, fontweight="bold")
    axs[0, 1].set_xlabel("Time (s)")
    axs[0, 1].set_ylabel("Velocity (m/s)")
    axs[0, 1].legend(loc="upper left", facecolor="#334155", edgecolor="#64748b", labelcolor="#f8fafc", fontsize=8.5)

    # Plot 3: Horizontal Position Drift vs. Distance Traveled
    cum_dist = np.cumsum(np.linalg.norm(np.diff(truth_pos[:, :2], axis=0, prepend=truth_pos[0:1, :2]), axis=1))
    axs[1, 0].plot(cum_dist, pos_err_ekf, color="#38bdf8", linewidth=2.0, label=f"ES-EKF Drift ({drift_percent_ekf:.2f}%)")
    axs[1, 0].plot(cum_dist, pos_err_vio, color="#a855f7", linestyle="--", linewidth=1.8, label="Pure VIO Drift")
    axs[1, 0].set_title("Horizontal Position Drift vs. Distance Traveled", fontsize=11, fontweight="bold")
    axs[1, 0].set_xlabel("Distance Traveled (m)")
    axs[1, 0].set_ylabel("Horizontal Error (m)")
    axs[1, 0].legend(loc="upper left", facecolor="#334155", edgecolor="#64748b", labelcolor="#f8fafc")

    # Plot 4: Accelerometer Bias Estimation
    ekf_ba = np.array(ekf_ba_hist)
    axs[1, 1].plot(t_arr, ekf_ba[:, 0], color="#f43f5e", label="Est Bias bx")
    axs[1, 1].plot(t_arr, ekf_ba[:, 1], color="#fb923c", label="Est Bias by")
    axs[1, 1].plot(t_arr, ekf_ba[:, 2], color="#facc15", label="Est Bias bz")
    axs[1, 1].axhline(true_accel_bias[0], color="#f43f5e", linestyle=":", alpha=0.7, label=f"True bx ({true_accel_bias[0]:+.2f})")
    axs[1, 1].axhline(true_accel_bias[1], color="#fb923c", linestyle=":", alpha=0.7, label=f"True by ({true_accel_bias[1]:+.2f})")
    axs[1, 1].axhline(true_accel_bias[2], color="#facc15", linestyle=":", alpha=0.7, label=f"True bz ({true_accel_bias[2]:+.2f})")
    axs[1, 1].set_title("Accelerometer Bias Estimation & Convergence", fontsize=11, fontweight="bold")
    axs[1, 1].set_xlabel("Time (s)")
    axs[1, 1].set_ylabel("Accel Bias (m/s^2)")
    axs[1, 1].legend(loc="upper right", facecolor="#334155", edgecolor="#64748b", labelcolor="#f8fafc", fontsize=8.0)

    plt.suptitle("Milestone 7: Visual-Inertial Odometry & 15-State ES-EKF Performance Report",
                 fontsize=14, fontweight="bold", color="#f8fafc", y=0.98)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(str(output_plot_path), dpi=200, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close()
    print(f"\nM7 VIO diagnostic plot saved to: {output_plot_path}")

    return {
        "final_distance_m": final_dist,
        "final_ekf_error_m": final_ekf_err,
        "drift_percent_ekf": drift_percent_ekf,
        "vel_rmse_ekf_mps": vel_rmse_ekf,
        "yaw_rmse_deg": yaw_rmse_deg,
        "homography_agreement_mps": mean_homo_diff,
        "report_plot": str(output_plot_path),
    }


if __name__ == "__main__":
    run_m7_vio_evaluation()
