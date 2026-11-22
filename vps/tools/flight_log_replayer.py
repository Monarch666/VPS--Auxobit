"""Real-Flight ArDuPilot DataFlash (.bin) replay tool for VPS.

Ingests binary flight logs from real drone flights over the Deccan Plateau:
1. Extracts synchronized 6-DOF trajectory (time, attitude, gyro rates, NED velocities, baro altitude).
2. Simulates camera frames along the exact real-world flight path over Deccan terrain.
3. Feeds vision + telemetry through the complete VPS pipeline:
   - Keyframe Optical Flow + Gyro De-Rotation
   - 2-State Kalman Filter (State: [h_agl, z_terrain]^T)
   - Hover / Stationary Loiter Detection & Terrain Hold
4. Logs and analyzes estimation errors, covariance bounds, confidence, and execution latency.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import sys
import time
import numpy as np
import cv2
from pymavlink import mavutil

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from vps.config import VPSConfig, CameraConfig, TrackerConfig, FusionConfig
from vps.mavlink.telemetry_buffer import TelemetryRingBuffer, TelemetrySample
from vps.tracker.flow_pipeline import VisionHeightPipeline
from vps.fusion.fusion_engine import VPSFusionEngine
from vps.tools.simulator import DroneCameraSimulator, DroneState, SimulatorDegradationConfig
from vps.tools.terrain_generator import DeccanTerrainGenerator


@dataclass
class FlightTrajectorySlice:
    """Synchronized 6-DOF trajectory extracted from ArduPilot DataFlash log."""
    times_s: np.ndarray  # (N,) timestamps starting at 0.0
    pos_ned_m: np.ndarray  # (N, 3) North, East, Down positions
    vel_ned_mps: np.ndarray  # (N, 3) North, East, Down velocities
    attitude_rpy_rad: np.ndarray  # (N, 3) Roll, Pitch, Yaw
    gyro_rad_s: np.ndarray  # (N, 3) Gyro angular rates
    baro_alt_m: np.ndarray  # (N,) Barometric altitude
    rel_alt_m: np.ndarray  # (N,) Relative altitude above home
    flight_modes: List[Tuple[float, str]] = field(default_factory=list)
    filename: str = ""


class FlightLogParser:
    """Parses ArduPilot DataFlash binary (.bin) log into synchronized trajectory."""

    @staticmethod
    def parse_bin_log(
        log_path: Path | str,
        start_time_s: Optional[float] = None,
        duration_s: Optional[float] = None,
        downsample_factor: int = 1,
    ) -> FlightTrajectorySlice:
        """Parse DataFlash .bin log and extract synchronized 6-DOF state timeseries.

        Args:
            log_path: Path to ArduPilot .bin log.
            start_time_s: Optional offset from flight start (seconds).
            duration_s: Optional duration slice to extract (seconds).
            downsample_factor: Downsample factor (1 = keep all samples).

        Returns:
            FlightTrajectorySlice containing synchronized kinematic timeseries.
        """
        path = Path(log_path)
        if not path.is_file():
            raise FileNotFoundError(f"Flight log not found: {path}")

        mlog = mavutil.mavlink_connection(str(path))

        times_list = []
        pos_list = []
        vel_list = []
        att_list = []
        gyro_list = []
        baro_list = []
        rel_alt_list = []
        modes = []

        # State tracking
        current_baro_alt = 0.0
        start_us = None

        while True:
            msg = mlog.recv_msg()
            if msg is None:
                break

            mtype = msg.get_type()

            if mtype == "MODE":
                m_name = getattr(msg, "Mode", str(getattr(msg, "ModeNum", "UNKNOWN")))
                t_m = getattr(msg, "TimeUS", 0) * 1e-6
                modes.append((t_m, m_name))

            elif mtype == "BARO":
                current_baro_alt = float(getattr(msg, "Alt", 0.0))

            elif mtype == "XKF1":
                t_us = getattr(msg, "TimeUS", 0)
                if start_us is None:
                    start_us = t_us

                t_rel = (t_us - start_us) * 1e-6

                # Filter by time slice if requested
                if start_time_s is not None and t_rel < start_time_s:
                    continue
                if duration_s is not None and start_time_s is not None:
                    if t_rel > start_time_s + duration_s:
                        break
                elif duration_s is not None and t_rel > duration_s:
                    break

                # Positions (NED: North, East, Down)
                pn = float(getattr(msg, "PN", 0.0))
                pe = float(getattr(msg, "PE", 0.0))
                pd = float(getattr(msg, "PD", 0.0))

                # Velocities (NED)
                vn = float(getattr(msg, "VN", 0.0))
                ve = float(getattr(msg, "VE", 0.0))
                vd = float(getattr(msg, "VD", 0.0))

                # Attitude (degrees to radians)
                roll = np.radians(float(getattr(msg, "Roll", 0.0)))
                pitch = np.radians(float(getattr(msg, "Pitch", 0.0)))
                yaw = np.radians(float(getattr(msg, "Yaw", 0.0)))

                # Gyro rates (GX, GY, GZ in deg/s to rad/s)
                gx = np.radians(float(getattr(msg, "GX", 0.0)))
                gy = np.radians(float(getattr(msg, "GY", 0.0)))
                gz = np.radians(float(getattr(msg, "GZ", 0.0)))

                # Relative altitude: In NED frame, altitude = -PD
                rel_h = -pd

                times_list.append(t_rel)
                pos_list.append([pn, pe, pd])
                vel_list.append([vn, ve, vd])
                att_list.append([roll, pitch, yaw])
                gyro_list.append([gx, gy, gz])
                baro_list.append(current_baro_alt)
                rel_alt_list.append(rel_h)

        if not times_list:
            raise ValueError(f"No XKF1 telemetry messages found in {path}")

        # Normalize time to start at 0.0
        times_arr = np.array(times_list, dtype=np.float64)
        times_arr -= times_arr[0]

        pos_arr = np.array(pos_list, dtype=np.float64)[::downsample_factor]
        vel_arr = np.array(vel_list, dtype=np.float64)[::downsample_factor]
        att_arr = np.array(att_list, dtype=np.float64)[::downsample_factor]
        gyro_arr = np.array(gyro_list, dtype=np.float64)[::downsample_factor]
        baro_arr = np.array(baro_list, dtype=np.float64)[::downsample_factor]
        rel_alt_arr = np.array(rel_alt_list, dtype=np.float64)[::downsample_factor]
        times_arr = times_arr[::downsample_factor]

        return FlightTrajectorySlice(
            times_s=times_arr,
            pos_ned_m=pos_arr,
            vel_ned_mps=vel_arr,
            attitude_rpy_rad=att_arr,
            gyro_rad_s=gyro_arr,
            baro_alt_m=baro_arr,
            rel_alt_m=rel_alt_arr,
            flight_modes=modes,
            filename=path.name,
        )


class FlightLogReplayer:
    """End-to-end replay engine for real flight logs."""

    def __init__(
        self,
        config: Optional[VPSConfig] = None,
        terrain_path: Optional[Path | str] = None,
    ) -> None:
        """Initialize replayer with VPS config and terrain model."""
        self.cfg = config if config is not None else VPSConfig()

        # Initialize or generate Deccan terrain
        t_path = Path(terrain_path) if terrain_path else Path("data/synthetic_terrain_deccan.png")
        if not t_path.is_file():
            print(f"Generating Deccan terrain orthophoto at {t_path}...")
            gen = DeccanTerrainGenerator(width_px=3072, height_px=3072, meters_per_pixel=0.1, seed=42)
            gen.generate(t_path)

        ortho = cv2.imread(str(t_path))
        if ortho is None:
            raise FileNotFoundError(f"Failed to load terrain image: {t_path}")

        # Initialize simulator
        deg_cfg = SimulatorDegradationConfig(sensor_noise_sigma=1.5, exposure_time_s=0.002, haze_beta=0.002)
        self.sim = DroneCameraSimulator(orthophoto=ortho, meters_per_pixel=0.1, degradation_config=deg_cfg)

        # Initialize VPS Fusion Engine
        self.fusion_engine = VPSFusionEngine(config=self.cfg, initial_height_m=20.0)

    def replay_trajectory(
        self,
        trajectory: FlightTrajectorySlice,
        target_fps: float = 15.0,
        max_samples: Optional[int] = 120,
    ) -> Dict[str, Any]:
        """Replay flight trajectory through vision + telemetry fusion pipeline.

        Args:
            trajectory: Flight trajectory slice from real flight log.
            target_fps: Target frame rate for camera processing.
            max_samples: Maximum number of visual frames to process (None = all).

        Returns:
            Dictionary of logged outputs, ground truth, and performance statistics.
        """
        n_total = len(trajectory.times_s)
        total_time_s = trajectory.times_s[-1] - trajectory.times_s[0]

        # Determine frame indices matching target_fps
        frame_interval_s = 1.0 / target_fps
        next_frame_time = trajectory.times_s[0]
        selected_indices = []

        for i, t in enumerate(trajectory.times_s):
            if t >= next_frame_time:
                selected_indices.append(i)
                next_frame_time += frame_interval_s
                if max_samples and len(selected_indices) >= max_samples:
                    break

        print(f"\n--- Replaying Real Flight: {trajectory.filename} ---")
        print(f"Total Trajectory Samples: {n_total} ({total_time_s:.1f} s)")
        print(f"Vision Replay Frames: {len(selected_indices)} @ {target_fps:.1f} FPS")

        # Set initial Kalman filter state matching initial trajectory height
        init_h = max(1.5, float(trajectory.rel_alt_m[selected_indices[0]]))
        self.fusion_engine.filter.x[0] = init_h
        self.fusion_engine.filter.x[1] = 0.0

        # Logging containers
        log_times = []
        log_true_agl = []
        log_baro_alt = []
        log_vision_agl = []
        log_kalman_agl = []
        log_terrain_elev = []
        log_sigma_agl = []
        log_confidence = []
        log_speeds = []
        log_pitches = []
        log_roll_rates = []
        log_pitch_rates = []
        latencies_ms = []

        # Baro initial ground offset
        baro_ground_ref = trajectory.baro_alt_m[0]

        for step, idx in enumerate(selected_indices):
            t_s = float(trajectory.times_s[idx])
            pos = trajectory.pos_ned_m[idx].copy()
            vel = trajectory.vel_ned_mps[idx].copy()
            att = trajectory.attitude_rpy_rad[idx].copy()
            gyro = trajectory.gyro_rad_s[idx].copy()
            baro = float(trajectory.baro_alt_m[idx])
            true_agl = float(trajectory.rel_alt_m[idx])

            # Ensure positive AGL altitude
            true_agl = max(1.5, true_agl)
            pos[2] = -true_agl  # NED frame: Z is negative altitude

            # 1. Render camera view from simulated terrain along actual flight path
            drone_state = DroneState(
                timestamp_s=t_s,
                pos_ned_m=pos,
                vel_ned_mps=vel,
                attitude_rpy_rad=att,
                angular_rates_rad_s=gyro,
            )
            frame = self.sim.render_frame(drone_state)

            # 2. Process frame & telemetry through master fusion engine
            fusion_out = self.fusion_engine.process_telemetry_and_frame(
                frame=frame,
                timestamp_s=t_s,
                pos_ned_m=pos,
                vel_ned_mps=vel,
                attitude_rpy_rad=att,
                gyro_rates_rad_s=gyro,
                baro_alt_m=baro,
                frame_idx=step,
            )

            # Record logs
            log_times.append(t_s)
            log_true_agl.append(true_agl)
            log_baro_alt.append(baro - baro_ground_ref)
            log_vision_agl.append(fusion_out.vision_height_m if fusion_out.vision_height_m is not None else np.nan)
            log_kalman_agl.append(fusion_out.height_agl_m)
            log_terrain_elev.append(fusion_out.terrain_elevation_m)
            log_sigma_agl.append(np.sqrt(fusion_out.variance_m2))
            log_confidence.append(fusion_out.confidence)
            log_speeds.append(float(np.linalg.norm(vel[:2])))
            log_pitches.append(float(np.degrees(att[1])))
            log_roll_rates.append(float(np.degrees(gyro[0])))
            log_pitch_rates.append(float(np.degrees(gyro[1])))
            latencies_ms.append(fusion_out.processing_time_ms)

            if (step + 1) % 25 == 0 or step == len(selected_indices) - 1:
                vis_str = f"{fusion_out.vision_height_m:5.1f}m" if fusion_out.vision_height_m is not None else "   ---"
                print(
                    f"  [{step+1:3d}/{len(selected_indices)}] t={t_s:5.1f}s | "
                    f"True AGL: {true_agl:5.1f}m | Baro: {baro-baro_ground_ref:5.1f}m | "
                    f"Vision: {vis_str} | KF AGL: {fusion_out.height_agl_m:5.1f}m | "
                    f"Err: {abs(fusion_out.height_agl_m - true_agl):4.2f}m | "
                    f"Spd: {log_speeds[-1]:4.1f}m/s | Conf: {fusion_out.confidence:.2f} | Lat: {fusion_out.processing_time_ms:4.1f}ms"
                )

        # Performance analytics
        arr_true = np.array(log_true_agl)
        arr_kf = np.array(log_kalman_agl)
        errors = np.abs(arr_kf - arr_true)
        errors_pct = (errors / arr_true) * 100.0

        mae_m = float(np.mean(errors))
        rmse_m = float(np.sqrt(np.mean(errors ** 2)))
        max_err_m = float(np.max(errors))
        mean_err_pct = float(np.mean(errors_pct))
        mean_latency_ms = float(np.mean(latencies_ms))

        print(f"\n--- Replay Performance Summary ---")
        print(f"  Mean Absolute Error (MAE): {mae_m:.2f} m ({mean_err_pct:.2f}%)")
        print(f"  Root Mean Square Error:   {rmse_m:.2f} m")
        print(f"  Max Absolute Error:       {max_err_m:.2f} m")
        print(f"  Mean Pipeline Latency:    {mean_latency_ms:.1f} ms ({1000.0/mean_latency_ms:.1f} FPS)")

        return {
            "times_s": np.array(log_times),
            "true_agl_m": arr_true,
            "baro_rel_m": np.array(log_baro_alt),
            "vision_agl_m": np.array(log_vision_agl),
            "kalman_agl_m": arr_kf,
            "terrain_elev_m": np.array(log_terrain_elev),
            "sigma_agl_m": np.array(log_sigma_agl),
            "confidence": np.array(log_confidence),
            "speeds_mps": np.array(log_speeds),
            "pitches_deg": np.array(log_pitches),
            "gyro_pitch_rates_deg_s": np.array(log_pitch_rates),
            "latencies_ms": np.array(latencies_ms),
            "mae_m": mae_m,
            "rmse_m": rmse_m,
            "max_error_m": max_err_m,
            "mean_error_pct": mean_err_pct,
            "mean_latency_ms": mean_latency_ms,
            "filename": trajectory.filename,
        }


def run_flight_log_replay(
    log_path: Path | str = Path("data/flight_logs/log_19_2026-11-11-18-02-08.bin"),
    duration_s: float = 60.0,
    target_fps: float = 15.0,
    output_plot_path: Path = Path("reports/m7_flight_log_replay.png"),
) -> Dict[str, Any]:
    """Execute flight log ingestion and replay benchmark."""
    parser = FlightLogParser()
    # Extract flight slice where drone is active in guided/loiter
    trajectory = parser.parse_bin_log(
        log_path=log_path,
        start_time_s=10.0,
        duration_s=duration_s,
    )

    replayer = FlightLogReplayer()
    results = replayer.replay_trajectory(trajectory=trajectory, target_fps=target_fps)

    output_plot_path.parent.mkdir(parents=True, exist_ok=True)
    _plot_flight_replay(results, output_plot_path)

    return results


def _plot_flight_replay(res: Dict[str, Any], output_path: Path) -> None:
    """Generate diagnostic multi-panel figure for flight log replay."""
    t = res["times_s"]
    fig, axs = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

    # 1. Height Tracking
    axs[0].plot(t, res["true_agl_m"], "k-", linewidth=2.5, label="True Flight AGL (Ground Truth)")
    axs[0].plot(t, res["baro_rel_m"], "g--", alpha=0.7, label="Raw Baro Relative Alt")
    valid_vis = ~np.isnan(res["vision_agl_m"])
    if np.any(valid_vis):
        axs[0].scatter(t[valid_vis], res["vision_agl_m"][valid_vis], color="red", s=30, zorder=4, label="Keyframe Vision AGL")
    axs[0].plot(t, res["kalman_agl_m"], "b-", linewidth=2, label="2-State Kalman Filter Output")
    axs[0].fill_between(
        t,
        res["kalman_agl_m"] - 2.0 * res["sigma_agl_m"],
        res["kalman_agl_m"] + 2.0 * res["sigma_agl_m"],
        color="blue", alpha=0.15, label="Kalman ±2σ Confidence Envelope"
    )
    axs[0].set_title(f"Real Flight Log Replay ({res['filename']}) - Height Estimation vs Ground Truth", fontweight="bold")
    axs[0].set_ylabel("Height AGL (m)")
    axs[0].grid(True, alpha=0.3)
    axs[0].legend(loc="upper right")

    # 2. Deccan Terrain Elevation & Confidence Score
    ax1_twin = axs[1].twinx()
    axs[1].plot(t, res["terrain_elev_m"], color="sienna", linewidth=2, label="Estimated Terrain Elevation (m)")
    ax1_twin.plot(t, res["confidence"], color="purple", linestyle="-.", linewidth=1.5, label="Confidence Metric C")
    axs[1].set_ylabel("Terrain Elevation (m)", color="sienna")
    ax1_twin.set_ylabel("Confidence [0, 1]", color="purple")
    ax1_twin.set_ylim(-0.05, 1.05)
    axs[1].grid(True, alpha=0.3)
    axs[1].set_title("Deccan Terrain State Tracking & Dynamic Confidence", fontweight="bold")

    # 3. Flight Kinematics & Latency Profile
    ax2_twin = axs[2].twinx()
    axs[2].plot(t, res["speeds_mps"], "navy", label="Ground Speed (m/s)")
    axs[2].plot(t, res["pitches_deg"], "teal", linestyle="--", label="Pitch Angle (deg)")
    ax2_twin.plot(t, res["latencies_ms"], "darkorange", alpha=0.8, label="Processing Latency (ms)")
    ax2_twin.axhline(100.0, color="red", linestyle=":", label="100 ms Real-Time Deadline")
    axs[2].set_ylabel("Kinematics (m/s, deg)")
    ax2_twin.set_ylabel("Pipeline Latency (ms)", color="darkorange")
    axs[2].set_xlabel("Flight Time (s)")
    axs[2].grid(True, alpha=0.3)
    axs[2].legend(loc="upper left")
    ax2_twin.legend(loc="upper right")
    axs[2].set_title("Drone Flight Dynamics & Real-Time Processing Profile", fontweight="bold")

    plt.tight_layout()
    plt.savefig(str(output_path), dpi=200)
    plt.close()
    print(f"Flight replay diagnostic plot saved to {output_path}")


if __name__ == "__main__":
    run_flight_log_replay()
