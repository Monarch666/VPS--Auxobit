"""End-to-end vision-based height-from-flow tracking pipeline.

Combines:
- Shi-Tomasi corner detection & pyramidal Lucas-Kanade tracking
- Adaptive keyframe selection guaranteeing >= 10 px de-rotated flow
- Gyro inter-frame homography de-rotation (H = K * R * K^-1)
- Metric optical flow height estimation and confidence evaluation
- Sub-40 ms runtime profiling (target >= 25 fps, latency < 100 ms)
"""

from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any
import time
import numpy as np
import cv2

from vps.config import VPSConfig, CameraConfig, TrackerConfig
from vps.geometry.pinhole import PinholeModel
from vps.geometry.derotation import GyroDerotationEngine
from vps.geometry.height_flow import HeightFromFlowEngine, FlowHeightResult
from .feature_tracker import FeatureTracker, TrackingResult
from .keyframe_manager import KeyframeManager, KeyframeStatus


@dataclass
class FlowPipelineResult:
    """Output telemetry generated per processed frame."""
    frame_idx: int
    timestamp_s: float
    height_agl_m: Optional[float]
    confidence: float
    is_valid: bool
    median_flow_px: float
    num_inliers: int
    processing_time_ms: float
    rolling_fps: float
    status_code: str


class VisionHeightPipeline:
    """End-to-end vision height estimation engine."""

    def __init__(
        self,
        config: Optional[VPSConfig] = None,
        camera_config: Optional[CameraConfig] = None,
        tracker_config: Optional[TrackerConfig] = None,
        R_cam_to_body: Optional[np.ndarray] = None,
    ) -> None:
        """Initialize pipeline with configuration and geometric engines."""
        self.vps_cfg = config if config is not None else VPSConfig()
        self.cam_cfg = camera_config if camera_config is not None else self.vps_cfg.camera
        self.trk_cfg = tracker_config if tracker_config is not None else self.vps_cfg.tracker

        # Extrinsics: camera to body
        if R_cam_to_body is not None:
            self.R_c2b = np.array(R_cam_to_body, dtype=np.float64).reshape((3, 3))
        else:
            # Standard downward mount:
            # X_c right (Body Y), Y_c down across image (Body -X), Z_c down (Body Z)
            self.R_c2b = np.array([
                [0.0, -1.0, 0.0],
                [1.0,  0.0, 0.0],
                [0.0,  0.0, 1.0]
            ], dtype=np.float64)
        self.R_b2c = self.R_c2b.T

        # Core Engines
        self.pinhole = PinholeModel(self.cam_cfg.K, self.cam_cfg.D, self.cam_cfg.width, self.cam_cfg.height)
        self.tracker = FeatureTracker(self.trk_cfg)
        self.derot_engine = GyroDerotationEngine(self.cam_cfg.K, R_body_to_cam=self.R_b2c)
        self.flow_engine = HeightFromFlowEngine(
            fx=self.cam_cfg.fx,
            fy=self.cam_cfg.fy,
            min_displacement_px=self.trk_cfg.min_keyframe_displacement_px,
            min_altitude_m=0.1,
            max_altitude_m=450.0,  # operational ceiling 400m + margin
        )
        self.keyframe_mgr = KeyframeManager(
            fx=self.cam_cfg.fx,
            config=self.trk_cfg,
            hover_speed_thresh_mps=self.vps_cfg.fusion.hover_speed_threshold_mps,
            max_keyframe_age_s=2.0,
            default_initial_altitude_m=30.0,
        )

        # Performance statistics
        self._proc_times_ms: List[float] = []
        self._max_proc_history = 30

    def _euler_to_rotation_matrix(self, roll: float, pitch: float, yaw: float) -> np.ndarray:
        """Compute R_w_b (body to world rotation matrix)."""
        cr, sr = np.cos(roll), np.sin(roll)
        cp, sp = np.cos(pitch), np.sin(pitch)
        cy, sy = np.cos(yaw), np.sin(yaw)
        return np.array([
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp,     cp * sr,                cp * cr]
        ], dtype=np.float64)

    def process_frame(
        self,
        frame: np.ndarray,
        timestamp_s: float,
        pos_ned_m: np.ndarray,
        vel_ned_mps: np.ndarray,
        attitude_rpy_rad: np.ndarray,
        frame_idx: int,
    ) -> FlowPipelineResult:
        """Process incoming camera frame and telemetry.

        Args:
            frame: Camera frame (BGR or grayscale).
            timestamp_s: Exact frame exposure timestamp (seconds).
            pos_ned_m: Drone position in Local NED (meters) [X, Y, Z].
            vel_ned_mps: Drone velocity in Local NED (m/s) [Vn, Ve, Vd].
            attitude_rpy_rad: Drone Euler angles in radians [roll, pitch, yaw].
            frame_idx: Monotonic frame counter.

        Returns:
            FlowPipelineResult with height estimate, confidence, and runtime profile.
        """
        t_start = time.perf_counter()

        if frame.ndim == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame

        # Step 1: If no active keyframe, initialize one
        if self.keyframe_mgr.active_keyframe is None:
            pts = self.tracker.detect_features(gray)
            self.keyframe_mgr.initialize_keyframe(
                image_gray=gray,
                features=pts,
                timestamp_s=timestamp_s,
                pos_ned_m=pos_ned_m,
                attitude_rpy_rad=attitude_rpy_rad,
                frame_idx=frame_idx,
            )
            t_elapsed = (time.perf_counter() - t_start) * 1000.0
            self._record_latency(t_elapsed)

            return FlowPipelineResult(
                frame_idx=frame_idx,
                timestamp_s=timestamp_s,
                height_agl_m=None,
                confidence=0.0,
                is_valid=False,
                median_flow_px=0.0,
                num_inliers=len(pts),
                processing_time_ms=t_elapsed,
                rolling_fps=self.rolling_fps,
                status_code="KEYFRAME_INITIALIZED",
            )

        kf = self.keyframe_mgr.active_keyframe

        # Step 2: Track features from keyframe to current frame
        track_res = self.tracker.track(kf.image_gray, gray, kf.features)

        # Step 3: Evaluate keyframe status
        status = self.keyframe_mgr.check_keyframe_status(
            current_timestamp_s=timestamp_s,
            current_pos_ned_m=pos_ned_m,
            current_vel_ned_mps=vel_ned_mps,
            num_tracked_features=track_res.num_inliers,
        )

        height_est: Optional[float] = None
        confidence = 0.0
        is_valid = False
        median_flow_px = track_res.median_flow_px
        status_msg = status.reason

        # Step 4: If baseline threshold reached, compute height-from-flow
        if status.should_estimate_height and track_res.num_inliers >= 8:
            # Inter-frame camera rotation R_k2c
            R_w_b0 = self._euler_to_rotation_matrix(*kf.attitude_rpy_rad)
            R_w_b1 = self._euler_to_rotation_matrix(*attitude_rpy_rad)
            R_cam_rel = self.R_b2c @ (R_w_b1.T @ R_w_b0) @ self.R_c2b

            inlier_pts_k = track_res.pts_prev[track_res.inlier_mask]
            inlier_pts_c = track_res.pts_curr[track_res.inlier_mask]

            # De-rotate optical flow
            _, flow_trans = self.derot_engine.compute_translational_flow(
                inlier_pts_k, inlier_pts_c, R_cam_rel
            )

            # Ground displacement in camera coordinates
            delta_pos_w = pos_ned_m - kf.pos_ned_m
            delta_pos_b = R_w_b0.T @ delta_pos_w
            delta_pos_c = self.R_b2c @ delta_pos_b

            # Estimate metric height
            flow_res = self.flow_engine.estimate_height(flow_trans, delta_pos_c)

            if flow_res.is_valid:
                height_est = flow_res.height_m
                confidence = flow_res.confidence
                is_valid = True
                status_msg = "FLOW_HEIGHT_VALID"
                self.keyframe_mgr.update_altitude_prior(flow_res.height_m)
            else:
                status_msg = f"FLOW_REJECTED: {flow_res.status_message}"

        # Step 5: Check if keyframe should be reset or replenished
        if status.should_reset_keyframe:
            new_features = self.tracker.detect_features(gray)
            self.keyframe_mgr.initialize_keyframe(
                image_gray=gray,
                features=new_features,
                timestamp_s=timestamp_s,
                pos_ned_m=pos_ned_m,
                attitude_rpy_rad=attitude_rpy_rad,
                frame_idx=frame_idx,
            )

        t_elapsed = (time.perf_counter() - t_start) * 1000.0
        self._record_latency(t_elapsed)

        return FlowPipelineResult(
            frame_idx=frame_idx,
            timestamp_s=timestamp_s,
            height_agl_m=height_est,
            confidence=confidence,
            is_valid=is_valid,
            median_flow_px=median_flow_px,
            num_inliers=track_res.num_inliers,
            processing_time_ms=t_elapsed,
            rolling_fps=self.rolling_fps,
            status_code=status_msg,
        )

    def _record_latency(self, elapsed_ms: float) -> None:
        """Record rolling latency."""
        self._proc_times_ms.append(elapsed_ms)
        if len(self._proc_times_ms) > self._max_proc_history:
            self._proc_times_ms.pop(0)

    @property
    def rolling_fps(self) -> float:
        """Estimate current processing throughput (FPS)."""
        if not self._proc_times_ms:
            return 30.0
        avg_ms = float(np.mean(self._proc_times_ms))
        if avg_ms <= 0:
            return 30.0
        return 1000.0 / avg_ms

