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
from vps.camera.capture import create_camera_capture
from vps.camera.calibration import load_calibration_yaml
from vps.mavlink.telemetry_buffer import TelemetryRingBuffer, TelemetrySample
from vps.mavlink.connection import MAVLinkBridge
from vps.fusion.fusion_engine import VPSFusionEngine
from vps.tracker.feature_tracker import FeatureTracker
from vps.geometry.derotation import GyroDerotationEngine
from vps.nav_ekf.es_ekf import ErrorStateEKF, ESEKFConfig, NavState
from vps.vio.velocity_estimator import VIOVelocityEstimator
from vps.map_db.geotiff_parser import GeoTIFFParser
from vps.map_db.runtime_lookup import RuntimeMapDB
from vps.map_matcher.orthorectifier import FrameOrthorectifier
from vps.map_matcher.matcher import MapMatcher
from vps.tools.aruco_validator import ArUcoValidator, ArUcoMarkerConfig
from vps.tools.logger import TelemetryLogger, TelemetryRecord
from vps.tools.session_recorder import ShadowSessionRecorder, ShadowTelemetryRecord
from vps.tools.vtx_hud import VTXDisplayServer, VTXHudData
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

        # Prevent OpenCV thread pools from starving libcamera / MAVLink on 4-core Pi
        try:
            cv2.setNumThreads(2)
        except Exception:
            pass

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

        # 2. Initialize Camera Capture (auto-selects Mock / OpenCV / Picamera2 CSI)
        self.camera = create_camera_capture(
            device_index=self.cfg.camera.device_index,
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
        self._last_ekf_imu_t: Optional[float] = None

        # 9. Phase 2: Offline Map DB & Absolute Map Matcher (Background Async Worker)
        self.map_db: Optional[RuntimeMapDB] = None
        self.orthorectifier: Optional[FrameOrthorectifier] = None
        self.map_matcher: Optional[MapMatcher] = None
        self._map_match_thread: Optional[threading.Thread] = None
        self._map_match_running = False
        self._map_match_lock = threading.Lock()
        self._ekf_lock = threading.RLock()
        self._pending_map_job: Optional[Dict[str, Any]] = None
        self._last_map_match_time: float = 0.0
        self._last_map_match_status: str = "STANDBY"
        self._reset_counter: int = 0
        self._map_geo_parser: Optional[GeoTIFFParser] = None
        self._ekf_geo_aligned: bool = False
        self._ekf_yaw_aligned: bool = False
        self._map_origin_offset_ne: np.ndarray = np.zeros(2, dtype=np.float64)
        self._last_map_diag: Dict[str, Any] = {
            "attempted": False,
            "accepted": False,
            "lat_deg": 0.0,
            "lon_deg": 0.0,
            "error_m": 0.0,
            "reason": "",
        }

        if getattr(self.cfg, "map", None) and self.cfg.map.enabled:
            map_path = Path(self.cfg.map.db_path)
            if (map_path / "manifest.json").is_file():
                try:
                    self.map_db = RuntimeMapDB(db_directory=map_path)
                    h_lat, h_lon = self.map_db.home_origin_wgs84
                    self._map_geo_parser = GeoTIFFParser(home_lat_deg=h_lat, home_lon_deg=h_lon)
                    self.orthorectifier = FrameOrthorectifier(camera_config=self.cfg.camera)
                    self.map_matcher = MapMatcher(
                        min_inliers_feature=self.cfg.map.min_inliers,
                    )
                    logger.info("Initialized Phase 2 Map Database from %s (GSD: %.2f m/px)", map_path, self.map_db.gsd_m_per_px)
                except Exception as e:
                    logger.warning("Could not load Map DB from %s: %s", map_path, e)
            else:
                logger.info("No Map DB found at %s. Operating in VIO-only dead-reckoning mode.", map_path)

        # 10. Optional Minimalist VTX OSD Video Feed
        self.vtx_hud: Optional[VTXDisplayServer] = (
            VTXDisplayServer(self.cfg.vtx) if self.cfg.vtx.enabled else None
        )

    def start(self) -> bool:
        """Start hardware devices, telemetry streams, and background workers."""
        if self.phase1_only:
            logger.info("Starting Visual Positioning System in PHASE 1 SHADOW MODE...")
        elif self.cfg.mode == "shadow":
            logger.info("Starting Visual Positioning System in PHASE 2 SHADOW MODE (Observe-Only VIO + Map Matching)...")
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

        # Start VTX OSD Output Server if enabled
        if self.vtx_hud is not None:
            self.vtx_hud.start()
            logger.info("VTX OSD interface enabled (sink=%s, %dx%d)", self.cfg.vtx.sink, self.cfg.vtx.output_width, self.cfg.vtx.output_height)

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
            map_offset_ne = job["map_offset_ne"]

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
                    with self._map_match_lock:
                        self._last_map_diag = {
                            "attempted": False,
                            "accepted": False,
                            "lat_deg": 0.0,
                            "lon_deg": 0.0,
                            "error_m": 0.0,
                            "reason": self._last_map_match_status,
                        }
                    continue

                # 2. Query spatial candidate tiles from runtime DB (in Map DB NED)
                pos_prior_map_ne = pos_prior_ned[:2] + map_offset_ne
                candidate_tiles = self.map_db.query_tiles(
                    pos_ned_m=pos_prior_map_ne,
                    search_radius_m=self.cfg.map.search_radius_m,
                    max_tiles_return=4,
                )
                if not candidate_tiles:
                    self._last_map_match_status = "NO_TILES_IN_SEARCH_WINDOW"
                    with self._map_match_lock:
                        self._last_map_diag = {
                            "attempted": False,
                            "accepted": False,
                            "lat_deg": 0.0,
                            "lon_deg": 0.0,
                            "error_m": 0.0,
                            "reason": "NO_TILES_IN_SEARCH_WINDOW",
                        }
                    continue

                # 3. Match frame against reference tiles
                match_res = self.map_matcher.match_frame_to_map(
                    ortho=ortho,
                    candidate_tiles=candidate_tiles,
                    pos_prior_ned_m=pos_prior_map_ne,
                    search_radius_m=self.cfg.map.search_radius_m,
                    timestamp_s=timestamp_s,
                )

                m_lat, m_lon = 0.0, 0.0
                err_m = float(np.linalg.norm(match_res.pos_ned_m - pos_prior_map_ne)) if match_res.is_valid else 0.0
                if match_res.is_valid and self._map_geo_parser is not None:
                    m_lat, m_lon = self._map_geo_parser.local_ned_to_wgs84(
                        float(match_res.pos_ned_m[0]),
                        float(match_res.pos_ned_m[1]),
                    )

                if match_res.is_valid and match_res.confidence >= self.cfg.map.min_confidence:
                    # 4. Convert Map DB fix back into EKF Local NED frame and inject delayed-state update
                    fix_ekf_ne = match_res.pos_ned_m - map_offset_ne
                    with self._ekf_lock:
                        fused = self.ekf.update_map_match_delayed(
                            fix_ned_m=fix_ekf_ne,
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
                    reason = f"{match_res.match_method}:inliers={match_res.num_inliers}:fused={fused}"
                    accepted = bool(fused)
                else:
                    reason = match_res.status_message or "low confidence"
                    self._last_map_match_status = f"MATCH_FAILED: {reason}"
                    accepted = False

                with self._map_match_lock:
                    self._last_map_diag = {
                        "attempted": True,
                        "accepted": accepted,
                        "lat_deg": float(m_lat),
                        "lon_deg": float(m_lon),
                        "error_m": float(err_m),
                        "reason": str(reason)[:64],
                    }
            except Exception as e:
                self._last_map_match_status = f"ERROR: {e}"
                with self._map_match_lock:
                    self._last_map_diag = {
                        "attempted": False,
                        "accepted": False,
                        "lat_deg": 0.0,
                        "lon_deg": 0.0,
                        "error_m": 0.0,
                        "reason": str(e)[:64],
                    }

    def run_autonomous_flight(self) -> None:
        """Main autonomous vision-in-the-loop processing loop (Phase 1 + Phase 2)."""
        frame_idx = 0
        t_last_log = time.monotonic()
        fps_counter = 0
        last_fps = 0.0

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
            has_live_telem = telem is not None
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
            vio_conf = float(fusion_out.confidence)
            vio_inliers = 0
            vio_flow_px = 0.0

            need_fc_status = (
                (not self.phase1_only and not self._ekf_geo_aligned and self._map_geo_parser is not None)
                or (self.session_logger is not None and isinstance(self.session_logger, ShadowSessionRecorder))
                or (self.vtx_hud is not None)
                or True
            )
            fc_status: Optional[Dict[str, Any]] = self.mavlink_bridge.get_fc_status()

            # Reset ground height/terrain baseline continuously while FC confirms disarmed & grounded on pad
            is_grounded_pad = bool(
                fc_status.get("heartbeat_seen")
                and not fc_status.get("is_armed")
                and abs(float(telem.baro_alt_m)) < 3.0
                and abs(float(telem.vel_ned_mps[2])) < 0.25
                and float(np.hypot(telem.vel_ned_mps[0], telem.vel_ned_mps[1])) < 0.4
            )
            if is_grounded_pad:
                self.fusion_engine.reset_ground_state()
                h_agl = self.cfg.camera.baseline_ground_offset_m
                fusion_out.height_agl_m = float(h_agl)
                fusion_out.terrain_elevation_m = float(self.fusion_engine.filter.x[1])
                fusion_out.variance_m2 = float(self.fusion_engine.filter.P[0, 0])
                if not self.phase1_only:
                    with self._ekf_lock:
                        self.ekf.reset_ground_state(
                            pos_ned_m=np.array(
                                [float(telem.pos_ned_m[0]), float(telem.pos_ned_m[1]), -h_agl],
                                dtype=np.float64,
                            ),
                            attitude_rpy_rad=telem.attitude_rpy_rad,
                        )
                        self._ekf_yaw_aligned = True
                        self._last_ekf_imu_t = t_capture

            if not self.phase1_only:
                if (not self._ekf_geo_aligned or is_grounded_pad) and self._map_geo_parser is not None and fc_status is not None:
                    gps_fix = int(fc_status.get("gps_fix_type", 0))
                    gps_eph = float(fc_status.get("gps_eph_m", 99.0))
                    gps_lat = float(fc_status.get("gps_lat_deg", 0.0))
                    gps_lon = float(fc_status.get("gps_lon_deg", 0.0))
                    if gps_fix >= 3 and (abs(gps_lat) > 1e-5 or abs(gps_lon) > 1e-5) and (not self._ekf_geo_aligned or gps_eph <= 2.5):
                        n_off, e_off = self._map_geo_parser.wgs84_to_local_ned(gps_lat, gps_lon)
                        if float(np.hypot(n_off, e_off)) <= 5000.0:
                            # Preserve FC Local NED frame inside EKF while recording offset to Map DB NED
                            self._map_origin_offset_ne = np.array(
                                [n_off - float(telem.pos_ned_m[0]), e_off - float(telem.pos_ned_m[1])],
                                dtype=np.float64,
                            )
                            if not self._ekf_geo_aligned:
                                self._ekf_geo_aligned = True
                                logger.info(
                                    "Aligned Map DB offset: N=%.2fm, E=%.2fm (GPS %.6f, %.6f)",
                                    self._map_origin_offset_ne[0], self._map_origin_offset_ne[1], gps_lat, gps_lon,
                                )

                with self._ekf_lock:
                    # Align EKF attitude and altitude before pushing first live history snapshot
                    if has_live_telem and not self._ekf_yaw_aligned:
                        self.ekf.align_attitude(
                            roll_rad=float(telem.attitude_rpy_rad[0]),
                            pitch_rad=float(telem.attitude_rpy_rad[1]),
                            yaw_rad=float(telem.attitude_rpy_rad[2]),
                        )
                        self.ekf.update_height(height_agl_m=h_agl, bypass_gate=True)
                        self._ekf_yaw_aligned = True

                    if not is_grounded_pad:
                        # 2. ES-EKF High-Rate IMU Propagation using all buffered telemetry samples since last frame
                        if self._last_ekf_imu_t is not None and t_capture > self._last_ekf_imu_t:
                            inter_samples = self.ring_buffer.get_samples_between(self._last_ekf_imu_t, t_capture)
                            for s in inter_samples:
                                if s.timestamp_s < t_capture - 1e-4:
                                    self.ekf.predict_imu(
                                        accel_meas_mps2=s.accel_mps2,
                                        gyro_meas_rad_s=s.gyro_rad_s,
                                        timestamp_s=s.timestamp_s,
                                    )
                        self.ekf.predict_imu(
                            accel_meas_mps2=telem.accel_mps2,
                            gyro_meas_rad_s=telem.gyro_rad_s,
                            timestamp_s=t_capture,
                        )
                        self._last_ekf_imu_t = t_capture

                        # 3. ES-EKF Attitude Stabilization (Compass Yaw + AHRS Roll/Pitch)
                        self.ekf.update_compass_yaw(yaw_compass_rad=float(telem.attitude_rpy_rad[2]))
                        self.ekf.update_ahrs_attitude(
                            roll_rad=float(telem.attitude_rpy_rad[0]),
                            pitch_rad=float(telem.attitude_rpy_rad[1]),
                        )

                        # 5. ES-EKF Vertical Position Update (Height AGL)
                        self.ekf.update_height(height_agl_m=h_agl)

                # 6. Phase 2: Frame-to-Frame Metric VIO Velocity Estimation (0.5x resolution on high-res sensors)
                gray_full = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
                vio_scale = 0.5 if gray_full.shape[1] >= 1000 else 1.0
                inv_vio_scale = 1.0 / vio_scale
                if vio_scale != 1.0:
                    gray = cv2.resize(gray_full, (0, 0), fx=vio_scale, fy=vio_scale, interpolation=cv2.INTER_AREA)
                else:
                    gray = gray_full

                if h_agl > 0.3 and self._prev_vio_gray is not None and self._prev_vio_pts is not None and len(self._prev_vio_pts) >= 8:
                    dt_vio = t_capture - (self._prev_vio_time if self._prev_vio_time is not None else (t_capture - 1.0 / 30.0))
                    if dt_vio > 0.001:
                        track_res = self.vio_tracker.track(self._prev_vio_gray, gray, self._prev_vio_pts)
                        vio_inliers = int(track_res.num_inliers)
                        vio_flow_px = float(track_res.median_flow_px * inv_vio_scale)
                        if track_res.num_inliers >= 8 and self._prev_vio_att is not None:
                            R_w_b0 = self.vio_est.euler_to_R_wb(*self._prev_vio_att)
                            R_w_b1 = self.vio_est.euler_to_R_wb(*telem.attitude_rpy_rad)
                            R_cam_rel = self.vio_est.R_b2c @ (R_w_b1.T @ R_w_b0) @ self.vio_est.R_c2b
                            H_derot = self.vio_derot.compute_rotational_homography(R_cam_rel)

                            try:
                                vio_res = self.vio_est.estimate_velocity_from_flow(
                                    pts0=track_res.pts_prev * inv_vio_scale,
                                    pts1=track_res.pts_curr * inv_vio_scale,
                                    inlier_mask=track_res.inlier_mask,
                                    H_derot=H_derot,
                                    height_agl_m=h_agl,
                                    dt_s=dt_vio,
                                    attitude_rpy_rad=telem.attitude_rpy_rad,
                                    timestamp_s=t_capture,
                                    compute_homography_alternative=False,
                                )
                                vio_conf = float(vio_res.confidence)
                                if vio_res.confidence > 0.2:
                                    with self._ekf_lock:
                                        self.ekf.update_vio_velocity(
                                            vel_vio_ned_mps=vio_res.vel_ned_mps,
                                            confidence=vio_res.confidence,
                                        )
                            except Exception as e:
                                logger.debug("VIO estimation skipped: %s", e)

                # Refresh tracking features for next frame
                self._prev_vio_pts = self.vio_tracker.detect_features(gray)
                self._prev_vio_gray = gray.copy() if vio_scale == 1.0 else gray
                self._prev_vio_att = telem.attitude_rpy_rad.copy()
                self._prev_vio_time = t_capture

                # 7. Asynchronous Map Matcher Job Queue (1-2 Hz, above min_altitude_m)
                min_map_alt = float(getattr(self.cfg.map, "min_altitude_m", 12.0))
                if (
                    self.map_db is not None
                    and h_agl >= min_map_alt
                    and (t_capture - self._last_map_match_time) >= (1.0 / self.cfg.map.match_rate_hz)
                ):
                    with self._ekf_lock:
                        pos_prior_copy = self.ekf.state.pos_ned_m.copy()
                        reset_cnt_copy = self._reset_counter
                    with self._map_match_lock:
                        if self._pending_map_job is None:
                            self._pending_map_job = {
                                "frame": frame.copy(),
                                "timestamp_s": t_capture,
                                "attitude_rpy_rad": telem.attitude_rpy_rad.copy(),
                                "height_agl_m": h_agl,
                                "pos_prior_ned": pos_prior_copy,
                                "map_offset_ne": self._map_origin_offset_ne.copy(),
                            }
                            self._last_map_match_time = t_capture
                else:
                    with self._ekf_lock:
                        reset_cnt_copy = self._reset_counter

                # 9. MAVLink Output 2: ODOMETRY / VISION_POSITION_ESTIMATE (Phase 2)
                if self.mavlink_bridge.odometry_sender and self.cfg.mode != "shadow":
                    with self._ekf_lock:
                        self.mavlink_bridge.odometry_sender.update_from_ekf(
                            nav_state=self.ekf.state,
                            P_15x15=self.ekf.P,
                            confidence=fusion_out.confidence,
                            frame_timestamp_s=t_capture,
                            gyro_rad_s=telem.gyro_rad_s,
                            reset_counter=reset_cnt_copy,
                        )

            if not self.phase1_only:
                with self._ekf_lock:
                    ekf_pos = self.ekf.state.pos_ned_m.copy()
                    ekf_vel = self.ekf.state.vel_ned_mps.copy()
                    ekf_pos_std = float(np.sqrt(max(0.0, 0.5 * (self.ekf.P[0, 0] + self.ekf.P[1, 1]))))
                    ekf_vel_std = float(np.sqrt(max(0.0, 0.5 * (self.ekf.P[3, 3] + self.ekf.P[4, 4]))))
            else:
                ekf_pos = np.zeros(3, dtype=np.float64)
                ekf_vel = np.zeros(3, dtype=np.float64)
                ekf_pos_std = 0.0
                ekf_vel_std = 0.0

            # Record session log if active
            if self.session_logger is not None:
                if isinstance(self.session_logger, ShadowSessionRecorder):
                    assert fc_status is not None
                    shadow_rec = ShadowTelemetryRecord(
                        timestamp_s=t_capture,
                        fc_boot_ms=int(fc_status["fc_boot_ms"]),
                        frame_idx=frame_idx,
                        is_armed=bool(fc_status["is_armed"]),
                        flight_mode=str(fc_status["flight_mode"]),
                        battery_voltage_v=float(fc_status["battery_voltage_v"]),
                        roll_rad=float(telem.attitude_rpy_rad[0]),
                        pitch_rad=float(telem.attitude_rpy_rad[1]),
                        yaw_rad=float(telem.attitude_rpy_rad[2]),
                        gyro_x_rad_s=float(telem.gyro_rad_s[0]),
                        gyro_y_rad_s=float(telem.gyro_rad_s[1]),
                        gyro_z_rad_s=float(telem.gyro_rad_s[2]),
                        accel_x_mps2=float(telem.accel_mps2[0]),
                        accel_y_mps2=float(telem.accel_mps2[1]),
                        accel_z_mps2=float(telem.accel_mps2[2]),
                        fc_vel_n_mps=float(telem.vel_ned_mps[0]),
                        fc_vel_e_mps=float(telem.vel_ned_mps[1]),
                        fc_vel_d_mps=float(telem.vel_ned_mps[2]),
                        fc_pos_n_m=float(telem.pos_ned_m[0]),
                        fc_pos_e_m=float(telem.pos_ned_m[1]),
                        fc_pos_d_m=float(telem.pos_ned_m[2]),
                        fc_baro_alt_m=float(telem.baro_alt_m),
                        baro_alt_m=float(telem.baro_alt_m),
                        gps_fix_type=int(fc_status["gps_fix_type"]),
                        gps_lat_deg=float(fc_status["gps_lat_deg"]),
                        gps_lon_deg=float(fc_status["gps_lon_deg"]),
                        gps_alt_m=float(fc_status["gps_alt_m"]),
                        gps_eph_m=float(fc_status["gps_eph_m"]),
                        gps_epv_m=float(fc_status["gps_epv_m"]),
                        gps_satellites=int(fc_status["gps_satellites"]),
                        rangefinder_dist_m=float(fc_status["rangefinder_dist_m"]),
                        vps_height_agl_m=float(fusion_out.height_agl_m),
                        estimated_h_m=float(fusion_out.height_agl_m),
                        vps_height_std_m=float(np.sqrt(max(0.0, fusion_out.variance_m2))),
                        vps_terrain_alt_m=float(fusion_out.terrain_elevation_m),
                        vps_confidence=float(fusion_out.confidence),
                        vps_status=fusion_out.mode_status,
                        features_detected=int(fusion_out.num_inliers),
                        features_tracked=int(fusion_out.num_inliers),
                        inlier_count=int(fusion_out.num_inliers),
                        optical_flow_mean_px=float(fusion_out.median_flow_px),
                    )
                    if not self.phase1_only:
                        shadow_rec.vps_pos_n_m = float(ekf_pos[0])
                        shadow_rec.vps_pos_e_m = float(ekf_pos[1])
                        shadow_rec.vps_pos_d_m = float(ekf_pos[2])
                        shadow_rec.vps_pos_std_m = ekf_pos_std
                        shadow_rec.vps_vel_n_mps = float(ekf_vel[0])
                        shadow_rec.vps_vel_e_mps = float(ekf_vel[1])
                        shadow_rec.vps_vel_d_mps = float(ekf_vel[2])
                        shadow_rec.vps_vel_std_mps = ekf_vel_std
                        shadow_rec.vps_nav_confidence = float(vio_conf)
                        if vio_inliers > 0:
                            shadow_rec.features_tracked = vio_inliers
                            shadow_rec.inlier_count = vio_inliers
                            shadow_rec.optical_flow_mean_px = vio_flow_px
                        with self._map_match_lock:
                            map_diag = dict(self._last_map_diag)
                        shadow_rec.map_match_attempted = bool(map_diag["attempted"])
                        shadow_rec.map_match_accepted = bool(map_diag["accepted"])
                        shadow_rec.map_match_lat_deg = float(map_diag["lat_deg"])
                        shadow_rec.map_match_lon_deg = float(map_diag["lon_deg"])
                        shadow_rec.map_match_error_m = float(map_diag["error_m"])
                        shadow_rec.map_match_reason = str(map_diag["reason"])
                    self.session_logger.log_record_and_frame(
                        shadow_rec,
                        frame if self.save_images else None,
                        force_recording=(not fc_status["heartbeat_seen"]),
                    )
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
            if self.mavlink_bridge.distance_sender and self.cfg.mode != "shadow":
                self.mavlink_bridge.distance_sender.update_measurement(
                    height_agl_m=fusion_out.height_agl_m,
                    confidence=fusion_out.confidence,
                    covariance_m2=fusion_out.variance_m2,
                )

            # 9. Shadow Mode HUD: Broadcast VPS_H, VPS_CONF, and companion HEARTBEAT
            # Whitelisted through the safety chokepoint (HEARTBEAT <= 1 Hz, NVF <= 2 Hz)
            if self.cfg.mode == "shadow" and self.mavlink_bridge.chokepoint:
                self.mavlink_bridge.send_heartbeat()
                self.mavlink_bridge.send_named_value_float("VPS_H", float(fusion_out.height_agl_m))
                self.mavlink_bridge.send_named_value_float("VPS_CONF", float(fusion_out.confidence))
                if not self.phase1_only:
                    self.mavlink_bridge.send_named_value_float("VPS_PN", float(ekf_pos[0]))
                    self.mavlink_bridge.send_named_value_float("VPS_PE", float(ekf_pos[1]))

            # 10. Periodic Status Logging (1 Hz) & Non-blocking VTX OSD submission
            t_now = time.monotonic()
            if t_now - t_last_log >= 1.0:
                last_fps = fps_counter / (t_now - t_last_log)
                fps = last_fps
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
                    logger.info(
                        "AGL: %5.2fm | Pos: [%6.1f, %6.1f]m | Vel: [%5.1f, %5.1f]m/s | Map: %-22s | Conf: %.2f | Rate: %4.1f fps",
                        fusion_out.height_agl_m,
                        ekf_pos[0],
                        ekf_pos[1],
                        ekf_vel[0],
                        ekf_vel[1],
                        self._last_map_match_status[:22],
                        fusion_out.confidence,
                        fps,
                    )

            # 11. Submit latest frame to non-blocking VTX OSD worker
            if self.vtx_hud is not None:
                fc_info = fc_status or {}
                hud_data = VTXHudData(
                    height_agl_m=float(fusion_out.height_agl_m),
                    baro_alt_m=float(telem.baro_alt_m),
                    confidence=float(fusion_out.confidence),
                    mode_status=fusion_out.mode_status,
                    is_armed=bool(fc_info.get("is_armed", False)),
                    flight_mode=str(fc_info.get("flight_mode", "UNKNOWN")),
                    battery_voltage_v=float(fc_info.get("battery_voltage_v", 0.0)),
                    gps_satellites=int(fc_info.get("gps_satellites", 0)),
                    fps=float(last_fps),
                    num_inliers=int(fusion_out.num_inliers),
                    pos_n_m=float(ekf_pos[0]),
                    pos_e_m=float(ekf_pos[1]),
                    vel_n_mps=float(ekf_vel[0]),
                    vel_e_mps=float(ekf_vel[1]),
                    map_status=self._last_map_match_status,
                    phase1_only=self.phase1_only,
                    is_shadow=(self.cfg.mode == "shadow"),
                    tracked_points=fusion_out.tracked_points,
                )
                self.vtx_hud.submit(frame, hud_data)

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
        if self.vtx_hud is not None:
            self.vtx_hud.stop()
            self.vtx_hud = None
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
    parser.add_argument("--shadow", action="store_true", help="Run Phase 1 + Phase 2 (VIO + Map Matching) in strictly observe-only Shadow Mode")
    parser.add_argument("--shadow-p2", action="store_true", help="Alias for --shadow (Phase 1 + Phase 2 observe-only Shadow Mode)")
    parser.add_argument("--phase1-only", action="store_true", help="Run exclusively Phase 1 (Vision-based AGL height estimation)")
    parser.add_argument("--record", action="store_true", help="Record synchronized telemetry and height estimates to a CSV session log")
    parser.add_argument("--save-images", action="store_true", help="Save captured camera frames to disk alongside CSV telemetry")
    parser.add_argument("--vtx", action="store_true", help="Enable minimalist VTX OSD video feed output")
    parser.add_argument(
        "--vtx-sink",
        choices=["auto", "fb", "window", "mjpeg", "fb+mjpeg"],
        default=None,
        help="VTX video output backend (fb=/dev/fb0 analog/HDMI, mjpeg=HTTP stream, window=desktop)",
    )
    parser.add_argument("--vtx-port", type=int, default=None, help="HTTP port for MJPEG VTX preview stream (default 8080)")
    return parser.parse_args()


def _resolve_operational_flags(args: argparse.Namespace) -> tuple[bool, bool, bool]:
    """Resolve (is_shadow, phase1_only, record_session) from CLI arguments."""
    is_shadow = bool(args.shadow or args.shadow_p2)
    phase1_only = bool(args.phase1_only)
    record_session = bool(args.record or is_shadow)
    return is_shadow, phase1_only, record_session


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
    if args.vtx:
        cfg.vtx.enabled = True
    if args.vtx_sink is not None:
        cfg.vtx.enabled = True
        cfg.vtx.sink = args.vtx_sink
    if args.vtx_port is not None:
        cfg.vtx.mjpeg_port = int(args.vtx_port)

    is_shadow, phase1_only, record_session = _resolve_operational_flags(args)

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
