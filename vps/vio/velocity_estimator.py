"""VIO Metric Velocity Estimator.

Calculates metric 3D horizontal velocity (Body and NED frames) using:
1. Primary method: De-rotated optical flow scaled by Phase 1 AGL height:
   v_c = (h_AGL / (f_px * dt)) * delta_p_trans
2. Secondary alternative: Planar homography decomposition via cv2.decomposeHomographyMat.
3. Cross-validation: Compares both estimators for planar surface consistency.
"""

from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict, Any
import numpy as np
import cv2

from vps.config import CameraConfig


@dataclass
class HomographyDecompositionResult:
    """Output of planar homography decomposition."""
    is_valid: bool
    velocity_cam_mps: np.ndarray  # shape (3,)
    normal_cam: np.ndarray  # shape (3,) normal vector to ground plane
    rotation_matrix: np.ndarray  # shape (3, 3)
    num_solutions: int
    error_message: str = ""


@dataclass
class VIOVelocityResult:
    """Consolidated VIO velocity estimation result."""
    timestamp_s: float
    dt_s: float
    height_agl_m: float
    # Metric velocities
    vel_cam_mps: np.ndarray  # shape (3,) [Vx, Vy, Vz] in Camera frame
    vel_body_mps: np.ndarray  # shape (3,) [Vx, Vy, Vz] in Body frame (X forward, Y right, Z down)
    vel_ned_mps: np.ndarray  # shape (3,) [Vn, Ve, Vd] in Local NED frame
    ground_speed_mps: float
    # Quality metrics
    num_inliers: int
    median_flow_px: float
    translational_flow_px: np.ndarray  # (2,) [du, dv]
    confidence: float
    # Homography alternative comparison
    homography_result: Optional[HomographyDecompositionResult] = None
    velocity_agreement_mps: Optional[float] = None


class VIOVelocityEstimator:
    """Recovers metric horizontal velocity from de-rotated optical flow and Phase 1 AGL height."""

    def __init__(
        self,
        camera_config: Optional[CameraConfig] = None,
        R_cam_to_body: Optional[np.ndarray] = None,
    ) -> None:
        """Initialize VIO velocity estimator.

        Args:
            camera_config: Camera intrinsic parameters.
            R_cam_to_body: 3x3 extrinsic rotation matrix mapping camera to body coordinates.
        """
        self.cam_cfg = camera_config if camera_config is not None else CameraConfig()

        # Default downward camera mount:
        # Camera Z_c is optical axis (pointing down || Body Z_b)
        # Camera X_c is image right (Body Y_b)
        # Camera Y_c is image down (Body -X_b, so top of image is forward)
        if R_cam_to_body is not None:
            self.R_c2b = np.array(R_cam_to_body, dtype=np.float64).reshape((3, 3))
        else:
            self.R_c2b = np.array([
                [0.0, -1.0, 0.0],
                [1.0,  0.0, 0.0],
                [0.0,  0.0, 1.0]
            ], dtype=np.float64)

        self.R_b2c = self.R_c2b.T

    @staticmethod
    def euler_to_R_wb(roll: float, pitch: float, yaw: float) -> np.ndarray:
        """Body to World NED rotation matrix (Z-Y-X sequence)."""
        cr, sr = np.cos(roll), np.sin(roll)
        cp, sp = np.cos(pitch), np.sin(pitch)
        cy, sy = np.cos(yaw), np.sin(yaw)

        return np.array([
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp,     cp * sr,                cp * cr]
        ], dtype=np.float64)

    def estimate_velocity_from_flow(
        self,
        pts0: np.ndarray,
        pts1: np.ndarray,
        inlier_mask: np.ndarray,
        H_derot: np.ndarray,
        height_agl_m: float,
        dt_s: float,
        attitude_rpy_rad: np.ndarray,
        timestamp_s: float,
        compute_homography_alternative: bool = True,
    ) -> VIOVelocityResult:
        """Calculate metric velocity from de-rotated optical flow correspondences.

        Args:
            pts0: Keyframe pixel coordinates shape (N, 2).
            pts1: Current frame pixel coordinates shape (N, 2).
            inlier_mask: Boolean or uint8 mask of valid tracked points shape (N,).
            H_derot: 3x3 gyro de-rotation homography H_rot = K R_{0->1} K^-1.
            height_agl_m: Current estimated AGL height from Phase 1.
            dt_s: Time elapsed between keyframe and current frame (seconds).
            attitude_rpy_rad: Current drone body attitude [roll, pitch, yaw] in radians.
            timestamp_s: Current frame timestamp.
            compute_homography_alternative: Whether to run planar homography decomposition.

        Returns:
            VIOVelocityResult with metric velocities in Camera, Body, and NED frames.
        """
        if dt_s <= 1e-4:
            raise ValueError(f"Time step too small for velocity differentiation: {dt_s:.6f} s")
        if height_agl_m <= 0.1:
            raise ValueError(f"AGL height too small for metric projection: {height_agl_m:.2f} m")

        # Extract inlier points
        mask = inlier_mask.astype(bool).ravel()
        p0_in = pts0[mask]
        p1_in = pts1[mask]
        num_inliers = int(len(p0_in))

        if num_inliers < 4:
            # Degenerate tracking: zero velocity with zero confidence
            return VIOVelocityResult(
                timestamp_s=timestamp_s,
                dt_s=dt_s,
                height_agl_m=height_agl_m,
                vel_cam_mps=np.zeros(3),
                vel_body_mps=np.zeros(3),
                vel_ned_mps=np.zeros(3),
                ground_speed_mps=0.0,
                num_inliers=num_inliers,
                median_flow_px=0.0,
                translational_flow_px=np.zeros(2),
                confidence=0.0,
            )

        # 1. De-rotate keyframe points to current frame attitude
        p0_hom = np.column_stack([p0_in, np.ones(num_inliers, dtype=np.float64)])
        p0_derot_hom = (H_derot @ p0_hom.T).T
        p0_derot = p0_derot_hom[:, :2] / np.maximum(p0_derot_hom[:, 2:3], 1e-8)

        # 2. Compute translational displacement vector: delta_p = p1 - p0_derot
        delta_p = p1_in - p0_derot  # shape (N, 2) [du, dv]
        median_du = float(np.median(delta_p[:, 0]))
        median_dv = float(np.median(delta_p[:, 1]))
        trans_flow = np.array([median_du, median_dv], dtype=np.float64)
        median_flow_norm = float(np.hypot(median_du, median_dv))

        # 3. Metric velocity in Camera frame:
        # Note: If camera moves forward (+X_w / -Y_c), ground features shift in the opposite direction (+Y_c / +dv).
        # Therefore, drone velocity in camera frame is opposite to image flow:
        # v_c_x = - (du * h) / (fx * dt)
        # v_c_y = - (dv * h) / (fy * dt)
        # v_c_z = 0.0 (horizontal optical flow)
        v_cam_x = -(median_du * height_agl_m) / (self.cam_cfg.fx * dt_s)
        v_cam_y = -(median_dv * height_agl_m) / (self.cam_cfg.fy * dt_s)
        vel_cam = np.array([v_cam_x, v_cam_y, 0.0], dtype=np.float64)

        # 4. Transform to Body Frame: V_body = R_c2b * V_cam
        vel_body = self.R_c2b @ vel_cam

        # 5. Transform to Local NED Frame: V_ned = R_w_b * V_body
        R_wb = self.euler_to_R_wb(*attitude_rpy_rad)
        vel_ned = R_wb @ vel_body

        ground_speed = float(np.hypot(vel_ned[0], vel_ned[1]))

        # Confidence based on inlier count and flow magnitude
        inlier_score = min(1.0, num_inliers / 30.0)
        flow_score = min(1.0, median_flow_norm / 3.0) if median_flow_norm > 0.5 else 0.5
        conf = float(np.clip(inlier_score * flow_score, 0.0, 1.0))

        # 6. Optional Planar Homography Decomposition Alternative
        homo_res: Optional[HomographyDecompositionResult] = None
        agreement_mps: Optional[float] = None

        if compute_homography_alternative and num_inliers >= 8:
            homo_res = self.decompose_planar_homography(
                pts0=p0_in,
                pts1=p1_in,
                height_agl_m=height_agl_m,
                dt_s=dt_s,
            )
            if homo_res.is_valid:
                diff = np.linalg.norm(vel_cam[:2] - homo_res.velocity_cam_mps[:2])
                agreement_mps = float(diff)

        return VIOVelocityResult(
            timestamp_s=timestamp_s,
            dt_s=dt_s,
            height_agl_m=height_agl_m,
            vel_cam_mps=vel_cam,
            vel_body_mps=vel_body,
            vel_ned_mps=vel_ned,
            ground_speed_mps=ground_speed,
            num_inliers=num_inliers,
            median_flow_px=median_flow_norm,
            translational_flow_px=trans_flow,
            confidence=conf,
            homography_result=homo_res,
            velocity_agreement_mps=agreement_mps,
        )

    def decompose_planar_homography(
        self,
        pts0: np.ndarray,
        pts1: np.ndarray,
        height_agl_m: float,
        dt_s: float,
    ) -> HomographyDecompositionResult:
        """Decompose planar homography matrix into rotation, normal, and metric translation.

        Uses cv2.findHomography and cv2.decomposeHomographyMat.
        """
        try:
            H, mask = cv2.findHomography(pts0, pts1, cv2.RANSAC, 2.0)
            if H is None:
                return HomographyDecompositionResult(
                    is_valid=False,
                    velocity_cam_mps=np.zeros(3),
                    normal_cam=np.zeros(3),
                    rotation_matrix=np.eye(3),
                    num_solutions=0,
                    error_message="Homography estimation failed",
                )

            K = self.cam_cfg.K
            num_solutions, rotations, translations, normals = cv2.decomposeHomographyMat(H, K)

            if num_solutions <= 0:
                return HomographyDecompositionResult(
                    is_valid=False,
                    velocity_cam_mps=np.zeros(3),
                    normal_cam=np.zeros(3),
                    rotation_matrix=np.eye(3),
                    num_solutions=0,
                    error_message="Decomposition yielded zero solutions",
                )

            # Ground plane selection:
            # For a downward camera, the ground plane normal in camera frame points primarily along +Z_c (optical axis):
            # n_cam ~ [0, 0, 1]^T
            best_idx = -1
            best_score = -1.0

            for idx in range(num_solutions):
                normal = normals[idx].ravel()
                # Ensure normal points towards ground (positive Z_c)
                if normal[2] > 0.0:
                    # Score by proximity to pure nadir normal [0, 0, 1]
                    score = float(normal[2])
                    if score > best_score:
                        best_score = score
                        best_idx = idx

            if best_idx == -1:
                # Fallback to solution 0 if none has positive Z
                best_idx = 0

            best_normal = normals[best_idx].ravel()
            n_norm = np.linalg.norm(best_normal)
            if n_norm > 1e-8:
                best_normal = best_normal / n_norm

            # Translation from decomposition is normalized by distance d (t/d)
            # True translation in camera frame is: t_metric = - height_agl * (t / d)
            t_unit = translations[best_idx].ravel()
            t_metric = -t_unit * height_agl_m
            vel_cam_homo = t_metric / dt_s

            return HomographyDecompositionResult(
                is_valid=True,
                velocity_cam_mps=vel_cam_homo,
                normal_cam=best_normal,
                rotation_matrix=rotations[best_idx],
                num_solutions=num_solutions,
            )

        except Exception as ex:
            return HomographyDecompositionResult(
                is_valid=False,
                velocity_cam_mps=np.zeros(3),
                normal_cam=np.zeros(3),
                rotation_matrix=np.eye(3),
                num_solutions=0,
                error_message=str(ex),
            )
