"""Live Camera Frame Orthorectifier and Gradient Engine.

Transforms oblique, tilted live camera frames into nadir, North-aligned, metric-scaled
orthophotos matching reference map coordinates:
1. De-tilts camera view using current pitch and roll (removes perspective keystoning).
2. Rotates image to True North using compass yaw heading (North is always UP).
3. Scales image to exact map GSD using Phase 1 AGL height: GSD_map meters/pixel.
4. Computes normalized Sobel gradient orientation and magnitude maps to bridge the
   seasonal appearance gap between live video and satellite reference imagery.
"""

from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np
import cv2

from vps.config import CameraConfig


@dataclass
class OrthoResult:
    """Output of live frame orthorectification."""
    # Metric orthorectified image (BGR uint8)
    image_bgr: np.ndarray
    image_gray: np.ndarray
    # Normalized Sobel gradient maps (for seasonal appearance gap matching)
    grad_mag: np.ndarray  # shape (H, W) uint8 [0, 255]
    grad_angle: np.ndarray  # shape (H, W) uint8 [0, 180]
    # Spatial dimensions and scale
    width_px: int
    height_px: int
    gsd_m_per_px: float
    # Metric ground footprint covered by the rectified view (meters)
    ground_width_m: float
    ground_height_m: float
    # Center pixel is the drone's nadir ground projection (0, 0 in relative NED)
    center_u_px: float
    center_v_px: float
    # Homography used for rectification
    H_cam_to_ortho: np.ndarray  # (3, 3)
    is_valid: bool
    status_message: str = ""
    valid_mask: Optional[np.ndarray] = None  # shape (H, W) uint8 (255 inside warped FOV, 0 on border)


class FrameOrthorectifier:
    """Orthorectifies live camera frames to match georeferenced satellite/aerial maps."""

    def __init__(
        self,
        camera_config: Optional[CameraConfig] = None,
        R_cam_to_body: Optional[np.ndarray] = None,
    ) -> None:
        """Initialize frame orthorectifier.

        Args:
            camera_config: Intrinsics and distortion of camera.
            R_cam_to_body: 3x3 extrinsic rotation matrix mapping camera to body coordinates.
        """
        self.cam_cfg = camera_config if camera_config is not None else CameraConfig()

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

    def compute_gradient_maps(
        self,
        gray_image: np.ndarray,
        valid_mask: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute normalized Sobel gradient magnitude and orientation maps.

        Bridges the appearance gap between live monsoon/green vegetation and dry reference maps.
        """
        gx = cv2.Sobel(gray_image, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray_image, cv2.CV_32F, 0, 1, ksize=3)

        mag = cv2.magnitude(gx, gy)
        if valid_mask is not None:
            inner = mag[valid_mask > 0]
            p98 = float(np.percentile(inner, 98)) if inner.size > 0 else 0.0
        else:
            p98 = float(np.percentile(mag, 98))
        mag_norm = np.clip((mag / max(1e-4, p98)) * 255.0, 0, 255).astype(np.uint8)

        angle_rad = np.arctan2(gy, gx)
        angle_deg = np.mod(np.degrees(angle_rad), 180.0).astype(np.uint8)
        if valid_mask is not None:
            mag_norm[valid_mask == 0] = 0
            angle_deg[valid_mask == 0] = 0
        return mag_norm, angle_deg

    def orthorectify(
        self,
        frame: np.ndarray,
        height_agl_m: float,
        attitude_rpy_rad: np.ndarray,
        target_gsd_m_per_px: float,
        output_size_px: Optional[Tuple[int, int]] = None,
    ) -> OrthoResult:
        """Orthorectify a live camera frame to a nadir, North-up metric image.

        Args:
            frame: Input camera image (BGR or grayscale).
            height_agl_m: Current estimated AGL height from Phase 1.
            attitude_rpy_rad: Drone Euler angles [roll, pitch, yaw] in radians.
            target_gsd_m_per_px: Metric scale of the target map (e.g. 15.62 m/px for Sentinel-2).
            output_size_px: Optional (width, height) of output rectified image.
                           If None, dynamically sized to cover camera FOV.

        Returns:
            OrthoResult containing rectified image, gradients, and spatial mappings.
        """
        if height_agl_m <= 0.5:
            # Degenerate ground contact
            h_out, w_out = (256, 256) if output_size_px is None else output_size_px
            zero_img = np.zeros((h_out, w_out, 3), dtype=np.uint8)
            return OrthoResult(
                image_bgr=zero_img,
                image_gray=np.zeros((h_out, w_out), dtype=np.uint8),
                grad_mag=np.zeros((h_out, w_out), dtype=np.uint8),
                grad_angle=np.zeros((h_out, w_out), dtype=np.uint8),
                width_px=w_out,
                height_px=h_out,
                gsd_m_per_px=target_gsd_m_per_px,
                ground_width_m=0.0,
                ground_height_m=0.0,
                center_u_px=w_out / 2.0,
                center_v_px=h_out / 2.0,
                H_cam_to_ortho=np.eye(3),
                is_valid=False,
                status_message="Height below minimum threshold for aerial orthorectification",
            )

        roll, pitch, yaw = attitude_rpy_rad
        target_gsd = max(0.01, float(target_gsd_m_per_px))

        # 1. Compute ground footprint dimensions from camera FOV at height_agl_m
        # Nominal HFOV ~62.2 deg, VFOV ~48.8 deg
        fov_x_half = np.arctan(self.cam_cfg.width / (2.0 * self.cam_cfg.fx))
        fov_y_half = np.arctan(self.cam_cfg.height / (2.0 * self.cam_cfg.fy))

        ground_extent_x = 2.0 * height_agl_m * np.tan(fov_x_half)
        ground_extent_y = 2.0 * height_agl_m * np.tan(fov_y_half)
        max_ground_dim = max(ground_extent_x, ground_extent_y) * 1.5  # include tilt & rotation margin

        if output_size_px is not None:
            w_out, h_out = output_size_px
        else:
            # Determine pixel dimensions at target map GSD
            dim_px = int(np.clip(round(max_ground_dim / target_gsd), 64, 1024))
            w_out, h_out = dim_px, dim_px

        c_u = w_out / 2.0
        c_v = h_out / 2.0

        # 2. Coordinate transformations:
        # Ground plane point (x_N, y_E, 0) relative to drone nadir:
        # In rectified image:
        # u_rect = c_u + y_E / target_gsd
        # v_rect = c_v - x_N / target_gsd  (North is UP, so decreasing v is North)
        # In matrix form: [u_rect, v_rect, 1]^T = T_g2rect * [x_N, y_E, 1]^T
        T_g2rect = np.array([
            [0.0,             1.0 / target_gsd, c_u],
            [-1.0 / target_gsd, 0.0,            c_v],
            [0.0,             0.0,              1.0]
        ], dtype=np.float64)

        T_rect2g = np.linalg.inv(T_g2rect)

        # 3. Ground plane to Camera homography:
        # Drone position in World NED relative to nadir: C_w = [0, 0, -height_agl_m]^T
        R_wb = self.euler_to_R_wb(roll, pitch, yaw)
        R_cw = self.R_b2c @ R_wb.T

        # Camera translation in camera frame: t_c = -R_cw * C_w = R_cw * [0, 0, h]^T
        t_c = R_cw @ np.array([0.0, 0.0, height_agl_m], dtype=np.float64)

        # H_g2cam maps (x_N, y_E, 1) -> (u_cam, v_cam, 1)
        H_g2cam = self.cam_cfg.K @ np.column_stack([R_cw[:, 0], R_cw[:, 1], t_c])

        # H_rect2cam maps (u_rect, v_rect, 1) -> (u_cam, v_cam, 1)
        H_rect2cam = H_g2cam @ T_rect2g

        try:
            H_cam2rect = np.linalg.inv(H_rect2cam)
        except np.linalg.LinAlgError:
            H_cam2rect = np.eye(3)

        # 4. Warp live camera frame to rectified projection
        # cv2.warpPerspective with WARP_INVERSE_MAP uses H_rect2cam
        if frame.ndim == 3:
            img_rect_bgr = cv2.warpPerspective(
                frame,
                H_rect2cam,
                (w_out, h_out),
                flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=(0, 0, 0),
            )
            img_rect_gray = cv2.cvtColor(img_rect_bgr, cv2.COLOR_BGR2GRAY)
        else:
            img_rect_gray = cv2.warpPerspective(
                frame,
                H_rect2cam,
                (w_out, h_out),
                flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0,
            )
            img_rect_bgr = cv2.cvtColor(img_rect_gray, cv2.COLOR_GRAY2BGR)

        # 5. Build eroded valid-FOV mask so black padding edges never enter gradient or ORB maps
        src_h, src_w = frame.shape[:2]
        raw_mask = cv2.warpPerspective(
            np.full((src_h, src_w), 255, dtype=np.uint8),
            H_rect2cam,
            (w_out, h_out),
            flags=cv2.INTER_NEAREST | cv2.WARP_INVERSE_MAP,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        valid_mask = cv2.erode(raw_mask, np.ones((3, 3), dtype=np.uint8), iterations=2)

        # 6. Compute Sobel gradient orientation and magnitude maps (interior only)
        grad_mag, grad_angle = self.compute_gradient_maps(img_rect_gray, valid_mask=valid_mask)

        return OrthoResult(
            image_bgr=img_rect_bgr,
            image_gray=img_rect_gray,
            grad_mag=grad_mag,
            grad_angle=grad_angle,
            width_px=w_out,
            height_px=h_out,
            gsd_m_per_px=target_gsd,
            ground_width_m=float(w_out * target_gsd),
            ground_height_m=float(h_out * target_gsd),
            center_u_px=c_u,
            center_v_px=c_v,
            H_cam_to_ortho=H_cam2rect,
            is_valid=True,
            status_message="ORTHORECTIFICATION_SUCCESS",
            valid_mask=valid_mask,
        )

