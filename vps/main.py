"""Visual Positioning System (VPS) - Master Production Daemon.

Entrypoint for standalone operation on Raspberry Pi 5 companion computer or host PC:
1. Initializes camera capture (OpenCV V4L2 / Sony IMX296 / Mock).
2. Connects to ArduPilot via MAVLink (UART serial or UDP loopback).
3. Executes real-time vision-based height estimation:
   - Pyramidal LK optical flow + Shi-Tomasi features
   - High-rate gyro de-rotation (H = K * R * K^-1)
   - 2-State Kalman Filter ([h_agl, z_terrain]^T)
   - Hover / stationary loiter terrain hold
4. Dispatches MAVLink DISTANCE_SENSOR packets to ArduPilot EKF at >= 15 Hz.
5. Supports interactive hand-held testing HUD mode with landing-gear baseline zeroing.
"""

from pathlib import Path
from typing import Optional
import argparse
import logging
import signal
import sys
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
from vps.tools.aruco_validator import ArUcoValidator, ArUcoMarkerConfig
from vps.health.watchdog import WatchdogSupervisor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("VPS")


class VPSApplication:
    """Master runtime coordinator for VPS autonomous height estimation."""

    def __init__(self, config: VPSConfig, connection_override: Optional[str] = None) -> None:
        self.cfg = config
        if connection_override:
            self.cfg.mavlink.connection_string = connection_override

        self._running = False
        self.ring_buffer = TelemetryRingBuffer(max_duration_s=10.0)

        # 1. Initialize MAVLink Bridge
        self.mavlink_bridge = MAVLinkBridge(
            config=self.cfg.mavlink,
            ring_buffer=self.ring_buffer,
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
                width=self.cfg.camera.width,
                height=self.cfg.camera.height,
                fps=self.cfg.camera.fps,
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

        # 4. Initialize Fusion Engine
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
        self.watchdog = WatchdogSupervisor()

    def start(self) -> bool:
        """Start hardware devices and telemetry streams."""
        logger.info("Starting Visual Positioning System...")
        logger.info("Camera: %s (%dx%d @ %d fps)", self.cfg.camera.name, self.cfg.camera.width, self.cfg.camera.height, self.cfg.camera.fps)
        logger.info("Baseline Landing-Gear Offset: %.3f m", self.cfg.camera.baseline_ground_offset_m)
        logger.info("MAVLink Endpoint: %s", self.cfg.mavlink.connection_string)

        # Connect MAVLink
        if not self.mavlink_bridge.connect():
            logger.warning("Failed to connect to MAVLink endpoint: %s (Continuing in sensor-only mode)", self.cfg.mavlink.connection_string)
        else:
            self.mavlink_bridge.start()
            self.mavlink_bridge.request_streams(rate_hz=self.cfg.mavlink.stream_rate_hz)
            logger.info("MAVLink bridge connected and streaming DISTANCE_SENSOR at %.1f Hz", self.cfg.mavlink.distance_sensor_rate_hz)

        # Start Camera
        if not self.camera.start():
            logger.error("Failed to open camera device %s", self.cfg.camera.device_index)
            return False

        # Start Watchdog Supervisor
        self.watchdog.start()
        self._running = True
        return True

    def run_autonomous_flight(self) -> None:
        """Main autonomous vision-in-the-loop processing loop."""
        frame_idx = 0
        t_last_log = time.monotonic()
        fps_counter = 0

        logger.info("Entering autonomous flight height estimation loop...")
        while self._running:
            ok, frame, t_capture = self.camera.read_frame()
            if not ok or frame is None:
                time.sleep(0.002)
                continue

            self.watchdog.record_frame()
            frame_idx += 1
            fps_counter += 1

            # Interpolate synchronized telemetry matching frame exposure time
            telem = self.ring_buffer.get_interpolated(t_capture)
            if telem is not None:
                self.watchdog.record_telemetry()
            else:
                # Default kinematic state if telemetry hasn't arrived yet
                telem = TelemetrySample(
                    timestamp_s=t_capture,
                    gyro_rad_s=np.zeros(3),
                    attitude_rpy_rad=np.zeros(3),
                    vel_ned_mps=np.zeros(3),
                    baro_alt_m=self.cfg.camera.baseline_ground_offset_m,
                )

            # Process through fusion engine
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

            # Update outbound MAVLink rangefinder packet
            if self.mavlink_bridge.distance_sender:
                self.mavlink_bridge.distance_sender.update_measurement(
                    height_agl_m=fusion_out.height_agl_m,
                    confidence=fusion_out.confidence,
                    covariance_m2=fusion_out.variance_m2,
                )

            # Periodic status logging (1 Hz)
            t_now = time.monotonic()
            if t_now - t_last_log >= 1.0:
                fps = fps_counter / (t_now - t_last_log)
                fps_counter = 0
                t_last_log = t_now

                vis_str = f"{fusion_out.vision_height_m:5.2f}m" if fusion_out.vision_height_m is not None else "  --- "
                logger.info(
                    "AGL: %5.2fm | Baro: %5.2fm | Vision: %s | Conf: %.2f | Mode: %-18s | Rate: %4.1f fps",
                    fusion_out.height_agl_m,
                    telem.baro_alt_m,
                    vis_str,
                    fusion_out.confidence,
                    fusion_out.mode_status,
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
            telem = self.ring_buffer.get_interpolated(t_capture)
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
        self.watchdog.stop()
        self.camera.stop()
        self.mavlink_bridge.stop()
        logger.info("VPS shutdown complete.")


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Visual Positioning System (VPS) Phase 1: Vision-Based Height Estimator",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", "-c", type=str, default="config/default_config.yaml", help="Path to YAML config file")
    parser.add_argument("--connection", "-m", type=str, default=None, help="MAVLink connection string (e.g. /dev/ttyAMA0:921600 or udpin:127.0.0.1:14550)")
    parser.add_argument("--camera", type=str, default=None, help="Camera device index, path, or 'mock'")
    parser.add_argument("--mode", choices=["flight", "handheld", "sitl"], default="flight", help="Operational mode")
    parser.add_argument("--baseline", type=float, default=None, help="Camera-to-landing-gear baseline ground offset in meters")
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

    if args.mode == "sitl":
        if args.connection is None:
            args.connection = "udpin:127.0.0.1:14550"
        cfg.camera.device_index = "mock"

    app = VPSApplication(config=cfg, connection_override=args.connection)

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
