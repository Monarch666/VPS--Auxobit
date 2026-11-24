"""Visual Positioning System (VPS) - Master Production Daemon.

Full-stack production daemon for Raspberry Pi 5 companion computer and host PC:
1. Camera capture: Sony IMX296 global shutter / OpenCV V4L2 / Mock simulator.
2. ArduPilot bidirectional MAVLink: UART serial or UDP loopback.
3. Phase 1 Height (AGL):
   - Pyramidal Lucas-Kanade optical flow + Shi-Tomasi feature tracking
   - High-rate gyro de-rotation (H = K * R * K^-1)
   - 2-State Kalman Filter ([h_agl, z_terrain]^T)
   - Hover / stationary loiter terrain hold
   - MAVLink DISTANCE_SENSOR packets at >= 15 Hz
4. Phase 2 GNSS-Denied Horizontal Navigation:
   - Metric VIO horizontal velocity estimation (v = flow * h / f_px)
   - 15-State Error-State EKF (Position, Velocity, Attitude, Accel/Gyro Bias)
   - Offline Sentinel-2 GeoTIFF tiled map database with sub-2ms LRU lookup
   - Asynchronous Nadir Frame Orthorectifier & Multi-Scale Map Matcher (1-2 Hz)
   - Delayed-State EKF fusion with latency compensation & chi^2 innovation gating
   - MAVLink ODOMETRY / VISION_POSITION_ESTIMATE output at 25 Hz
5. System Safety & Reliability:
   - HealthMonitor safety interlocks (covariance inflation & advisory gating)
   - Hardware Watchdog Supervisor
"""

from pathlib import Path
from typing import Optional, Dict, Any, List
import argparse
import logging
import signal
import sys
import threading
import time
import numpy as np
import cv2

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vps.config import VPSConfig, load_config
from vps.camera.capture import OpenCVCameraCapture, MockCameraCapture
from vps.camera.calibration import load_calibration_yaml
from vps.mavlink.telemetry_buffer import TelemetryRingBuffer, TelemetrySample
from vps.mavlink.connection import MAVLinkBridge
from vps.fusion.fusion_engine import VPSFusionEngine
from vps.tracker.feature_tracker import FeatureTracker
from vps.geometry.derotation import GyroDerotationEngine
from vps.nav_ekf.es_ekf import ErrorStateEKF, ESEKFConfig, NavState
from vps.vio.velocity_estimator import VIOVelocityEstimator
from vps.map_db.runtime_lookup import RuntimeMapDB
from vps.map_matcher.orthorectifier import FrameOrthorectifier
from vps.map_matcher.matcher import MapMatcher
from vps.tools.aruco_validator import ArUcoValidator, ArUcoMarkerConfig
from vps.tools.logger import TelemetryLogger, TelemetryRecord
from vps.tools.session_recorder import ShadowSessionRecorder, ShadowTelemetryRecord
from vps.health.watchdog import WatchdogSupervisor, ShadowSupervisor
from vps.health.health_monitor import HealthMonitor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("VPS")


class VPSApplication:
    """Master runtime coordinator for VPS autonomous height estimation and horizontal navigation."""

    def __init__(
        self,
        config: VPSConfig,
        connection_override: Optional[str] = None,
        phase1_only: bool = False,
        record_session: bool = False,
        save_images: bool = False,
    ) -> None:
        self.cfg = config
        if connection_override:
            self.cfg.mavlink.connection_string = connection_override

        self.phase1_only = phase1_only
        self.record_session = record_session
        self.save_images = save_images
        self.session_logger: Optional[TelemetryLogger] = None

        if self.record_session:
            self.session_logger = ShadowSessionRecorder(
                output_dir=self.cfg.logger.output_dir,
                save_images=self.save_images,
                config_snapshot={
                    "mode": self.cfg.mode,
                    "camera": self.cfg.camera.__dict__,
                    "fusion": self.cfg.fusion.__dict__,
                    "mavlink": self.cfg.mavlink.__dict__,
                    "map": self.cfg.map.__dict__ if hasattr(self.cfg, "map") else {},
                },
            )
            logger.info("Session telemetry recorder active: %s", self.session_logger.session_dir)

        self._running = False
        self.ring_buffer = TelemetryRingBuffer(max_duration_s=10.0)

        # 1. Initialize MAVLink Bridge
        self.mavlink_bridge = MAVLinkBridge(
            config=self.cfg.mavlink,
            ring_buffer=self.ring_buffer,
            mode=self.cfg.mode,
        )

        # 2. Initialize Camera Capture
        if str(self.cfg.camera.device_index).lower() == "mock":
            self.camera = MockCameraCapture(fps=self.cfg.camera.fps)
        else:
            try:
                dev_idx = int(self.cfg.camera.device_index)
            except ValueError:
                dev_idx = self.cfg.camera.device_index
            self.camera = OpenCVCameraCapture(
                device_index=dev_idx,
                config=self.cfg.camera,
            )

        # 3. Load Calibrated Intrinsics
        calib_path = Path("config/imx296_calibrated.yaml")
        if calib_path.is_file():
            K, D = load_calibration_yaml(calib_path)
            self.cfg.camera.fx = float(K[0, 0])
            self.cfg.camera.fy = float(K[1, 1])
            self.cfg.camera.cx = float(K[0, 2])
            self.cfg.camera.cy = float(K[1, 2])
            self.cfg.camera.dist_coeffs = [float(x) for x in D.ravel()[:5]]
            logger.info("Loaded calibrated camera intrinsics from %s", calib_path)

        # 4. Initialize Phase 1 Fusion Engine
        self.fusion_engine = VPSFusionEngine(
            config=self.cfg,
            initial_height_m=self.cfg.camera.baseline_ground_offset_m,
            initial_terrain_m=0.0,
        )

        # 5. Hand-held ArUco Validator
        self.aruco_validator = ArUcoValidator(
            camera_config=self.cfg.camera,
            marker_config=ArUcoMarkerConfig(
                baseline_offset_m=self.cfg.camera.baseline_ground_offset_m,
            ),
        )

        # 6. Hardware Watchdog and Supervisor
        if self.cfg.mode == "shadow":
            self.watchdog = ShadowSupervisor(storage_path=self.cfg.logger.output_dir)
        else:
            self.watchdog = WatchdogSupervisor()

        # 7. Phase 2: Error-State EKF (15-State)
        self.ekf = ErrorStateEKF(
            initial_pos_ned=np.array([0.0, 0.0, -self.cfg.camera.baseline_ground_offset_m], dtype=np.float64),
            initial_vel_ned=np.zeros(3, dtype=np.float64),
            initial_yaw_rad=0.0,
        )

        # 8. Phase 2: VIO Velocity Estimator & Frame Tracker
        self.vio_est = VIOVelocityEstimator(camera_config=self.cfg.camera)
        self.vio_tracker = FeatureTracker(self.cfg.tracker)
        R_b2c = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]], dtype=np.float64)
        self.vio_derot = GyroDerotationEngine(self.cfg.camera.K, R_body_to_cam=R_b2c)
        self._prev_vio_gray: Optional[np.ndarray] = None
        self._prev_vio_pts: Optional[np.ndarray] = None
        self._prev_vio_att: Optional[np.ndarray] = None
        self._prev_vio_time: Optional[float] = None

        # 9. Phase 2: Offline Map DB & Absolute Map Matcher (Background Async Worker)
        self.map_db: Optional[RuntimeMapDB] = None
        self.orthorectifier: Optional[FrameOrthorectifier] = None
        self.map_matcher: Optional[MapMatcher] = None
        self._map_match_thread: Optional[threading.Thread] = None
        self._map_match_running = False
        self._map_match_lock = threading.Lock()
        self._pending_map_job: Optional[Dict[str, Any]] = None
        self._last_map_match_time: float = 0.0
        self._last_map_match_status: str = "STANDBY"
        self._reset_counter: int = 0

        if getattr(self.cfg, "map", None) and self.cfg.map.enabled:
            map_path = Path(self.cfg.map.db_path)
            if (map_path / "manifest.json").is_file():
                try:
                    self.map_db = RuntimeMapDB(db_directory=map_path)
                    self.orthorectifier = FrameOrthorectifier(camera_config=self.cfg.camera)
                    self.map_matcher = MapMatcher(
                        min_inliers_feature=self.cfg.map.min_inliers,
                    )
                    logger.info("Initialized Phase 2 Map Database from %s (GSD: %.2f m/px)", map_path, self.map_db.gsd_m_per_px)
                except Exception as e:
                    logger.warning("Could not load Map DB from %s: %s", map_path, e)
            else:
                logger.info("No Map DB found at %s. Operating in VIO-only dead-reckoning mode.", map_path)

    def start(self) -> bool:
        """Start hardware devices, telemetry streams, and background workers."""
        if self.phase1_only:
            logger.info("Starting Visual Positioning System in PHASE 1 SHADOW MODE...")
        else:
            logger.info("Starting Visual Positioning System (Phase 1 + Phase 2)...")
        logger.info("Camera: %s (%dx%d @ %d fps)", self.cfg.camera.name, self.cfg.camera.width, self.cfg.camera.height, self.cfg.camera.fps)
        logger.info("Baseline Landing-Gear Offset: %.3f m", self.cfg.camera.baseline_ground_offset_m)
        logger.info("MAVLink Endpoint: %s", self.cfg.mavlink.connection_string)
        logger.info("Feed FC EKF: %s (Mode: %s)", self.cfg.mavlink.feed_fc_ekf, self.cfg.mavlink.odometry_mode)

        # Connect MAVLink
        if not self.mavlink_bridge.connect():
            logger.warning("Failed to connect to MAVLink endpoint: %s (Continuing in sensor-only mode)", self.cfg.mavlink.connection_string)
        else:
            self.mavlink_bridge.start()
            self.mavlink_bridge.request_streams(rate_hz=self.cfg.mavlink.stream_rate_hz)
            if self.phase1_only:
                logger.info("MAVLink bridge connected and streaming DISTANCE_SENSOR (%.1f Hz) [Passive Advisory]",
                            self.cfg.mavlink.distance_sensor_rate_hz)
            else:
                logger.info("MAVLink bridge connected and streaming DISTANCE_SENSOR (%.1f Hz) & ODOMETRY (%.1f Hz)",
                            self.cfg.mavlink.distance_sensor_rate_hz, self.cfg.mavlink.odometry_rate_hz)

        # Start Camera
        if not self.camera.start():
            logger.error("Failed to open camera device %s", self.cfg.camera.device_index)
            return False

        # Start Watchdog Supervisor
        self.watchdog.start()

        # Start Map Matcher Background Thread (Phase 2 only)
        if not self.phase1_only:
            self._start_map_worker()

        self._running = True
        return True

    def _start_map_worker(self) -> None:
        """Start background worker thread for asynchronous map matching."""
        if self.map_db is None or self.map_matcher is None or self.orthorectifier is None:
            return
        self._map_match_running = True
        self._map_match_thread = threading.Thread(
            target=self._map_match_worker_loop,
            name="VPS_MapMatcherWorker",
            daemon=True,
        )
        self._map_match_thread.start()
        logger.info("Started background Map Matcher worker thread (target rate: %.1f Hz)", self.cfg.map.match_rate_hz)

    def _map_match_worker_loop(self) -> None:
        """Asynchronous worker loop executing orthorectification and tile matching."""
        while self._map_match_running:
            job = None
            with self._map_match_lock:
                if self._pending_map_job is not None:
                    job = self._pending_map_job
                    self._pending_map_job = None

            if job is None:
                time.sleep(0.01)
                continue

            frame = job["frame"]
            timestamp_s = job["timestamp_s"]
            rpy = job["attitude_rpy_rad"]
            height_agl_m = job["height_agl_m"]
            pos_prior_ned = job["pos_prior_ned"]

            try:
                # 1. Orthorectify live frame to nadir North-up image matching map GSD
                target_gsd = self.map_db.gsd_m_per_px
                ortho = self.orthorectifier.orthorectify(
                    frame=frame,
                    height_agl_m=height_agl_m,
                    attitude_rpy_rad=rpy,
                    target_gsd_m_per_px=target_gsd,
                )
                if not ortho.is_valid:
                    self._last_map_match_status = f"ORTHO_INVALID: {ortho.status_message}"
                    continue

                # 2. Query spatial candidate tiles from runtime DB
                candidate_tiles = self.map_db.query_tiles(
                    pos_ned_m=pos_prior_ned[:2],
                    search_radius_m=self.cfg.map.search_radius_m,
                    max_tiles_return=4,
                )
                if not candidate_tiles:
                    self._last_map_match_status = "NO_TILES_IN_SEARCH_WINDOW"
                    continue

                # 3. Match frame against reference tiles
                match_res = self.map_matcher.match_frame_to_map(
                    ortho=ortho,
                    candidate_tiles=candidate_tiles,
                    pos_prior_ned_m=pos_prior_ned[:2],
                    search_radius_m=self.cfg.map.search_radius_m,
                    timestamp_s=timestamp_s,
                )

                if match_res.is_valid and match_res.confidence >= self.cfg.map.min_confidence:
                    # 4. Inject delayed-state correction into ES-EKF
                    fused = self.ekf.update_map_match_delayed(
                        fix_ned_m=match_res.pos_ned_m,
                        cov_ned_m2=match_res.cov_ned_m2,
                        timestamp_s=timestamp_s,
                        confidence=match_res.confidence,
                    )
                    if fused:
                        self._reset_counter += 1
                    self._last_map_match_status = (
                        f"MATCHED({match_res.match_method}, inliers={match_res.num_inliers}, "
                        f"conf={match_res.confidence:.2f}, fused={fused})"
                    )
                else:
                    self._last_map_match_status = f"MATCH_FAILED: {match_res.status_message or 'low confidence'}"
            except Exception as e:
                self._last_map_match_status = f"ERROR: {e}"

    def run_autonomous_flight(self) -> None:
        """Main autonomous vision-in-the-loop processing loop (Phase 1 + Phase 2)."""
        frame_idx = 0
        t_last_log = time.monotonic()
        fps_counter = 0

        if self.phase1_only:
            logger.info("Entering Phase 1 autonomous height estimation loop (SHADOW MODE)...")
            logger.info("FC EKF Active Feeding: %s (ESTIMATOR PASSIVE / SHADOW ONLY)", self.cfg.mavlink.feed_fc_ekf)
        else:
            logger.info("Entering autonomous flight navigation loop (Phase 1 Height + Phase 2 VIO & Map Matching)...")

        while self._running:
            ok, frame, t_capture = self.camera.read_frame()
            if not ok or frame is None:
                time.sleep(0.002)
                continue

            self.watchdog.record_frame()
            frame_idx += 1
            fps_counter += 1

            # 1. Interpolate synchronized telemetry matching frame exposure midpoint
            telem = self.ring_buffer.interpolate(t_capture)
            if telem is not None:
                self.watchdog.record_telemetry()
            else:
                telem = TelemetrySample(
                    timestamp_s=t_capture,
                    gyro_rad_s=np.zeros(3),
                    attitude_rpy_rad=np.zeros(3),
                    vel_ned_mps=np.zeros(3),
                    baro_alt_m=self.cfg.camera.baseline_ground_offset_m,
                    pos_ned_m=np.zeros(3),
                    accel_mps2=np.array([0.0, 0.0, -9.80665], dtype=np.float64),
                )

            # 4. Phase 1: Height from De-Rotated Flow & Barometer Fusion
            fusion_out = self.fusion_engine.process_telemetry_and_frame(
                frame=frame,
                timestamp_s=t_capture,
                pos_ned_m=telem.pos_ned_m,
                vel_ned_mps=telem.vel_ned_mps,
                attitude_rpy_rad=telem.attitude_rpy_rad,
                gyro_rates_rad_s=telem.gyro_rad_s,
                baro_alt_m=telem.baro_alt_m,
                frame_idx=frame_idx,
            )

            h_agl = max(self.cfg.camera.baseline_ground_offset_m, fusion_out.height_agl_m)

            if not self.phase1_only:
                # 2. ES-EKF High-Rate IMU Propagation
                self.ekf.predict_imu(
                    accel_meas_mps2=telem.accel_mps2,
                    gyro_meas_rad_s=telem.gyro_rad_s,
                    timestamp_s=t_capture,
                )

                # 3. ES-EKF Compass Heading Update
                self.ekf.update_compass_yaw(yaw_compass_rad=float(telem.attitude_rpy_rad[2]))

                # 5. ES-EKF Vertical Position Update (Height AGL)
                self.ekf.update_height(height_agl_m=h_agl)

                # 6. Phase 2: Frame-to-Frame Metric VIO Velocity Estimation
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame

                if h_agl > 0.3 and self._prev_vio_gray is not None and self._prev_vio_pts is not None and len(self._prev_vio_pts) >= 8:
                    dt_vio = t_capture - (self._prev_vio_time if self._prev_vio_time is not None else (t_capture - 1.0 / 30.0))
                    if dt_vio > 0.001:
                        track_res = self.vio_tracker.track(self._prev_vio_gray, gray, self._prev_vio_pts)
                        if track_res.num_inliers >= 8 and self._prev_vio_att is not None:
                            delta_rpy = telem.attitude_rpy_rad - self._prev_vio_att
                            R_rel = self.vio_derot.rotation_matrix_from_rpy_delta(delta_rpy[0], delta_rpy[1], delta_rpy[2])
                            H_derot = self.vio_derot.compute_rotational_homography(R_rel)

                            try:
                                vio_res = self.vio_est.estimate_velocity_from_flow(
                                    pts0=track_res.pts_prev,
                                    pts1=track_res.pts_curr,
                                    inlier_mask=track_res.inlier_mask,
                                    H_derot=H_derot,
                                    height_agl_m=h_agl,
                                    dt_s=dt_vio,
                                    attitude_rpy_rad=telem.attitude_rpy_rad,
                                    timestamp_s=t_capture,
                                    compute_homography_alternative=False,
                                )
                                if vio_res.confidence > 0.2:
                                    self.ekf.update_vio_velocity(
                                        vel_ned_mps=vio_res.vel_ned_mps,
                                        confidence=vio_res.confidence,
                                    )
                            except Exception as e:
                                logger.debug("VIO estimation skipped: %s", e)

                # Refresh tracking features for next frame
                self._prev_vio_pts = self.vio_tracker.detect_features(gray)
                self._prev_vio_gray = gray.copy()
                self._prev_vio_att = telem.attitude_rpy_rad.copy()
                self._prev_vio_time = t_capture

                # 7. Asynchronous Map Matcher Job Queue (1-2 Hz)
                if self.map_db is not None and (t_capture - self._last_map_match_time) >= (1.0 / self.cfg.map.match_rate_hz):
                    with self._map_match_lock:
                        if self._pending_map_job is None:
                            self._pending_map_job = {
                                "frame": frame.copy(),
                                "timestamp_s": t_capture,
                                "attitude_rpy_rad": telem.attitude_rpy_rad.copy(),
                                "height_agl_m": h_agl,
                                "pos_prior_ned": self.ekf.state.pos_ned_m.copy(),
                            }
                            self._last_map_match_time = t_capture

                # 9. MAVLink Output 2: ODOMETRY / VISION_POSITION_ESTIMATE (Phase 2)
                if self.mavlink_bridge.odometry_sender:
                    self.mavlink_bridge.odometry_sender.update_from_ekf(
                        nav_state=self.ekf.state,
                        P_15x15=self.ekf.P,
                        confidence=fusion_out.confidence,
                        frame_timestamp_s=t_capture,
                        gyro_rad_s=telem.gyro_rad_s,
                        reset_counter=self._reset_counter,
                    )

            # Record session log if active
            if self.session_logger is not None:
                if isinstance(self.session_logger, ShadowSessionRecorder):
                    shadow_rec = ShadowTelemetryRecord(
                        timestamp_s=t_capture,
                        frame_idx=frame_idx,
                        is_armed=True,
                        roll_rad=float(telem.attitude_rpy_rad[0]),
                        pitch_rad=float(telem.attitude_rpy_rad[1]),
                        yaw_rad=float(telem.attitude_rpy_rad[2]),
                        gyro_x_rad_s=float(telem.gyro_rad_s[0]),
                        gyro_y_rad_s=float(telem.gyro_rad_s[1]),
                        gyro_z_rad_s=float(telem.gyro_rad_s[2]),
                        fc_vel_n_mps=float(telem.vel_ned_mps[0]),
                        fc_vel_e_mps=float(telem.vel_ned_mps[1]),
                        fc_vel_d_mps=float(telem.vel_ned_mps[2]),
                        fc_pos_n_m=float(telem.pos_ned_m[0]),
                        fc_pos_e_m=float(telem.pos_ned_m[1]),
                        fc_pos_d_m=float(telem.pos_ned_m[2]),
                        fc_baro_alt_m=float(telem.baro_alt_m),
                        baro_alt_m=float(telem.baro_alt_m),
                        vps_height_agl_m=float(fusion_out.height_agl_m),
                        estimated_h_m=float(fusion_out.height_agl_m),
                        vps_confidence=float(fusion_out.confidence),
                        vps_status=fusion_out.mode_status,
                    )
                    if not self.phase1_only:
                        shadow_rec.vps_pos_n_m = float(self.ekf.state.pos_ned_m[0])
                        shadow_rec.vps_pos_e_m = float(self.ekf.state.pos_ned_m[1])
                        shadow_rec.vps_pos_d_m = float(self.ekf.state.pos_ned_m[2])
                        shadow_rec.vps_vel_n_mps = float(self.ekf.state.vel_ned_mps[0])
                        shadow_rec.vps_vel_e_mps = float(self.ekf.state.vel_ned_mps[1])
                        shadow_rec.vps_vel_d_mps = float(self.ekf.state.vel_ned_mps[2])
                    self.session_logger.log_record_and_frame(shadow_rec, frame if self.save_images else None)
                else:
                    rec = TelemetryRecord(
                        timestamp_s=t_capture,
                        frame_idx=frame_idx,
                        roll_rad=float(telem.attitude_rpy_rad[0]),
                        pitch_rad=float(telem.attitude_rpy_rad[1]),
                        yaw_rad=float(telem.attitude_rpy_rad[2]),
                        gyro_x_rad_s=float(telem.gyro_rad_s[0]),
                        gyro_y_rad_s=float(telem.gyro_rad_s[1]),
                        gyro_z_rad_s=float(telem.gyro_rad_s[2]),
                        vel_n_mps=float(telem.vel_ned_mps[0]),
                        vel_e_mps=float(telem.vel_ned_mps[1]),
                        vel_d_mps=float(telem.vel_ned_mps[2]),
                        baro_alt_m=float(telem.baro_alt_m),
                        estimated_h_m=float(fusion_out.height_agl_m),
                        confidence=float(fusion_out.confidence),
                    )
                    self.session_logger.log(rec, frame=frame if self.save_images else None)

            # 8. MAVLink Output 1: DISTANCE_SENSOR (Phase 1)
            # Sends passive distance sensor packet for pilot display in Mission Planner HUD
            if self.mavlink_bridge.distance_sender:
                self.mavlink_bridge.distance_sender.update_measurement(
                    height_agl_m=fusion_out.height_agl_m,
                    confidence=fusion_out.confidence,
                    covariance_m2=fusion_out.variance_m2,
                )

            # 9. Shadow Mode HUD: Broadcast VPS_H and VPS_CONF via NAMED_VALUE_FLOAT
            # These are whitelisted through the safety chokepoint at <= 2 Hz
            if self.cfg.mode == "shadow" and self.mavlink_bridge.chokepoint:
                self.mavlink_bridge.send_named_value_float("VPS_H", float(fusion_out.height_agl_m))
                self.mavlink_bridge.send_named_value_float("VPS_CONF", float(fusion_out.confidence))

            # 10. Periodic Status Logging (1 Hz)
            t_now = time.monotonic()
            if t_now - t_last_log >= 1.0:
                fps = fps_counter / (t_now - t_last_log)
                fps_counter = 0
                t_last_log = t_now

                vis_str = f"{fusion_out.vision_height_m:5.2f}m" if fusion_out.vision_height_m is not None else "  --- "
                if self.phase1_only:
                    logger.info(
                        "[SHADOW P1] AGL: %5.2fm | Baro: %5.2fm | Vision: %s | Conf: %.2f | Mode: %-18s | Rate: %4.1f fps",
                        fusion_out.height_agl_m,
                        telem.baro_alt_m,
                        vis_str,
                        fusion_out.confidence,
                        fusion_out.mode_status,
                        fps,
                    )
                else:
                    pos_n = self.ekf.state.pos_ned_m[0]
                    pos_e = self.ekf.state.pos_ned_m[1]
                    vel_n = self.ekf.state.vel_ned_mps[0]
                    vel_e = self.ekf.state.vel_ned_mps[1]

                    logger.info(
                        "AGL: %5.2fm | Pos: [%6.1f, %6.1f]m | Vel: [%5.1f, %5.1f]m/s | Map: %-22s | Conf: %.2f | Rate: %4.1f fps",
                        fusion_out.height_agl_m,
                        pos_n,
                        pos_e,
                        vel_n,
                        vel_e,
                        self._last_map_match_status[:22],
                        fusion_out.confidence,
                        fps,
                    )

    def run_handheld_testing(self) -> None:
        """Interactive hand-held testing mode with live ArUco & optical flow HUD."""
        logger.info("Entering interactive hand-held testing mode (press 'q' or ESC in HUD window to exit)...")
        frame_idx = 0

        while self._running:
            ok, frame, t_capture = self.camera.read_frame()
            if not ok or frame is None:
                time.sleep(0.005)
                continue

            self.watchdog.record_frame()
            frame_idx += 1

            # 1. Check for ArUco ground marker
            meas = self.aruco_validator.detect_and_estimate(frame, timestamp_s=t_capture)

            # 2. Query fusion state
            telem = self.ring_buffer.interpolate(t_capture)
            if telem is not None:
                self.watchdog.record_telemetry()
            else:
                telem = TelemetrySample(timestamp_s=t_capture, gyro_rad_s=np.zeros(3), attitude_rpy_rad=np.zeros(3), vel_ned_mps=np.zeros(3), baro_alt_m=0.0)

            fusion_out = self.fusion_engine.process_telemetry_and_frame(
                frame=frame,
                timestamp_s=t_capture,
                pos_ned_m=telem.pos_ned_m,
                vel_ned_mps=telem.vel_ned_mps,
                attitude_rpy_rad=telem.attitude_rpy_rad,
                gyro_rates_rad_s=telem.gyro_rad_s,
                baro_alt_m=telem.baro_alt_m,
                frame_idx=frame_idx,
            )

            # Dispatch rangefinder message
            if self.mavlink_bridge.distance_sender:
                # In hand-held mode over ArUco marker, use optical ArUco height if available
                h_report = meas.net_agl_m if meas is not None else fusion_out.height_agl_m
                conf_report = 0.99 if meas is not None else fusion_out.confidence
                self.mavlink_bridge.distance_sender.update_measurement(
                    height_agl_m=h_report,
                    confidence=conf_report,
                    covariance_m2=0.01 if meas is not None else fusion_out.variance_m2,
                )

            # Draw HUD overlay
            hud_frame = self.aruco_validator.draw_overlay(
                frame=frame,
                meas=meas,
                vps_agl_m=fusion_out.height_agl_m,
            )

            # Display window (if GUI supported)
            try:
                cv2.imshow("VPS Hand-Held Validation HUD", hud_frame)
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q"), ord("Q")):
                    logger.info("Exit requested by user keypress.")
                    break
            except cv2.error:
                # Headless environment
                pass

        cv2.destroyAllWindows()

    def stop(self) -> None:
        """Gracefully stop hardware, connections and background threads."""
        if not self._running:
            return
        logger.info("Shutting down VPS...")
        self._running = False
        self._map_match_running = False
        if self._map_match_thread is not None:
            self._map_match_thread.join(timeout=0.5)
            self._map_match_thread = None
        if self.session_logger is not None:
            self.session_logger.close()
            logger.info("Session telemetry log saved to: %s", self.session_logger.csv_path)
            self.session_logger = None
        self.watchdog.stop()
        self.camera.stop()
        self.mavlink_bridge.stop()
        logger.info("VPS shutdown complete.")


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Visual Positioning System (VPS) Phase 1 (Height) & Phase 2 (Horizontal Navigation)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", "-c", type=str, default="config/default_config.yaml", help="Path to YAML config file")
    parser.add_argument("--connection", "-m", type=str, default=None, help="MAVLink connection string (e.g. /dev/ttyAMA0:921600 or udpin:127.0.0.1:14550)")
    parser.add_argument("--camera", type=str, default=None, help="Camera device index, path, or 'mock'")
    parser.add_argument("--mode", choices=["flight", "handheld", "sitl"], default="flight", help="Operational mode")
    parser.add_argument("--baseline", type=float, default=None, help="Camera-to-landing-gear baseline ground offset in meters")
    parser.add_argument("--map-db", type=str, default=None, help="Path to offline tiled GeoTIFF map database")
    parser.add_argument("--no-map", action="store_true", help="Disable absolute map matching (run in VIO dead-reckoning mode)")
    parser.add_argument("--feed-ekf", action="store_true", help="Enable active control feeding into ArduPilot FC EKF")
    parser.add_argument("--shadow", action="store_true", help="Run Phase 1 in Shadow Mode: logs telemetry + estimates to CSV, disables FC feeding, skips Phase 2")
    parser.add_argument("--phase1-only", action="store_true", help="Run exclusively Phase 1 (Vision-based AGL height estimation)")
    parser.add_argument("--record", action="store_true", help="Record synchronized telemetry and height estimates to a CSV session log")
    parser.add_argument("--save-images", action="store_true", help="Save captured camera frames to disk alongside CSV telemetry")
    return parser.parse_args()


def main() -> int:
    """CLI application entrypoint."""
    args = parse_arguments()

    # Load configuration
    cfg_file = Path(args.config)
    if cfg_file.is_file():
        cfg = load_config(cfg_file)
    else:
        cfg = VPSConfig()

    if args.camera is not None:
        cfg.camera.device_index = args.camera
    if args.baseline is not None:
        cfg.camera.baseline_ground_offset_m = float(args.baseline)
    if args.map_db is not None:
        cfg.map.db_path = args.map_db
    if args.no_map:
        cfg.map.enabled = False
    if args.feed_ekf:
        cfg.mavlink.feed_fc_ekf = True

    is_shadow = bool(args.shadow)
    phase1_only = bool(args.phase1_only or is_shadow)
    record_session = bool(args.record or is_shadow)

    if is_shadow:
        cfg.mode = "shadow"
        cfg.mavlink.feed_fc_ekf = False

    if args.mode == "sitl":
        if args.connection is None:
            args.connection = "udpin:127.0.0.1:14550"
        cfg.camera.device_index = "mock"

    app = VPSApplication(
        config=cfg,
        connection_override=args.connection,
        phase1_only=phase1_only,
        record_session=record_session,
        save_images=bool(args.save_images),
    )

    # Signal handlers for clean termination
    def handle_signal(sig, frame):
        logger.info("Signal %d received, stopping...", sig)
        app.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    if not app.start():
        return 1

    try:
        if args.mode == "handheld":
            app.run_handheld_testing()
        else:
            app.run_autonomous_flight()
    except KeyboardInterrupt:
        logger.info("Interrupted by user.")
    finally:
        app.stop()

    return 0


if __name__ == "__main__":
    sys.exit(main())
