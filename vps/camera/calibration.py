"""Checkerboard camera calibration for Sony IMX296.

Implements:
- Sub-pixel corner detection
- Monocular camera calibration (K matrix, distortion coeffs D)
- Per-view and overall reprojection error estimation
- YAML export and synthetic calibration target generation for testing
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
import numpy as np
import cv2
import yaml

from vps.config import CameraConfig


@dataclass
class CalibrationResult:
    """Output of camera calibration."""
    K: np.ndarray  # 3x3
    D: np.ndarray  # 1x5 or 5,
    mean_reprojection_error_px: float
    per_image_errors_px: List[float]
    image_width: int
    image_height: int
    num_images_used: int
    is_valid: bool


class CheckerboardCalibrator:
    """Calibrates camera intrinsics using planar checkerboard images."""

    def __init__(
        self,
        pattern_size: Tuple[int, int] = (9, 6),  # interior corners (cols, rows)
        square_size_m: float = 0.025,  # 25 mm square size
    ) -> None:
        """Initialize calibrator.

        Args:
            pattern_size: Tuple of (columns, rows) of interior checkerboard corners.
            square_size_m: Physical width/height of a checkerboard square in meters.
        """
        self.pattern_cols, self.pattern_rows = pattern_size
        self.square_size = float(square_size_m)

        # 3D object points in target coordinate frame (Z = 0)
        self.objp = np.zeros((self.pattern_rows * self.pattern_cols, 3), np.float32)
        self.objp[:, :2] = np.mgrid[0:self.pattern_cols, 0:self.pattern_rows].T.reshape(-1, 2) * self.square_size

    def find_corners(self, image: np.ndarray) -> Tuple[bool, Optional[np.ndarray]]:
        """Detect and refine checkerboard corners with sub-pixel precision.

        Args:
            image: Input image (BGR or grayscale).

        Returns:
            Tuple of (found_bool, refined_corners_array).
        """
        if image.ndim == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE + cv2.CALIB_CB_FAST_CHECK
        ret, corners = cv2.findChessboardCorners(gray, (self.pattern_cols, self.pattern_rows), flags)

        if ret and corners is not None:
            # Sub-pixel refinement
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 0.001)
            refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            return True, refined

        return False, None

    def calibrate_from_images(
        self,
        images: List[np.ndarray],
    ) -> CalibrationResult:
        """Calibrate camera intrinsics from a list of images.

        Args:
            images: List of image frames (BGR or grayscale).

        Returns:
            CalibrationResult dataclass.
        """
        if len(images) == 0:
            raise ValueError("No images provided for calibration")

        h, w = images[0].shape[:2]
        objpoints = []
        imgpoints = []

        for img in images:
            found, corners = self.find_corners(img)
            if found and corners is not None:
                objpoints.append(self.objp)
                imgpoints.append(corners)

        n_valid = len(objpoints)
        if n_valid < 3:
            return CalibrationResult(
                K=np.eye(3),
                D=np.zeros(5),
                mean_reprojection_error_px=999.0,
                per_image_errors_px=[],
                image_width=w,
                image_height=h,
                num_images_used=n_valid,
                is_valid=False,
            )

        # Run calibration
        ret, K, D, rvecs, tvecs = cv2.calibrateCamera(
            objpoints,
            imgpoints,
            (w, h),
            None,
            None,
            flags=cv2.CALIB_RATIONAL_MODEL if False else 0,
        )

        # Compute per-image reprojection errors
        per_img_errs = []
        total_error = 0.0
        total_points = 0
        for i in range(len(objpoints)):
            projected, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], K, D)
            p_meas = imgpoints[i].reshape(-1, 2).astype(np.float64)
            p_proj = projected.reshape(-1, 2).astype(np.float64)
            err = float(np.linalg.norm(p_meas - p_proj))
            num_pts = len(objpoints[i])
            mre = err / np.sqrt(num_pts)
            per_img_errs.append(float(mre))
            total_error += err * err
            total_points += num_pts

        mean_error = float(np.sqrt(total_error / total_points))

        return CalibrationResult(
            K=K,
            D=D.ravel()[:5],
            mean_reprojection_error_px=mean_error,
            per_image_errors_px=per_img_errs,
            image_width=w,
            image_height=h,
            num_images_used=n_valid,
            is_valid=(mean_error < 1.0),
        )

    def save_to_yaml(self, result: CalibrationResult, output_path: Path | str, camera_name: str = "Sony_IMX296") -> None:
        """Export calibration results to YAML config format."""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "camera": {
                "name": camera_name,
                "width": result.image_width,
                "height": result.image_height,
                "fx": float(result.K[0, 0]),
                "fy": float(result.K[1, 1]),
                "cx": float(result.K[0, 2]),
                "cy": float(result.K[1, 2]),
                "dist_coeffs": [float(x) for x in result.D[:5]],
                "mean_reprojection_error_px": float(result.mean_reprojection_error_px),
                "num_images_used": int(result.num_images_used),
            }
        }
        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)


def render_synthetic_checkerboard(
    pattern_size: Tuple[int, int] = (9, 6),
    square_size_px: int = 40,
    margin_px: int = 50,
) -> np.ndarray:
    """Render a clean 2D checkerboard pattern image for synthetic testing."""
    cols, rows = pattern_size
    # Total board squares = (cols + 1) x (rows + 1)
    n_sq_x = cols + 1
    n_sq_y = rows + 1
    w = n_sq_x * square_size_px + 2 * margin_px
    h = n_sq_y * square_size_px + 2 * margin_px

    board = np.full((h, w), 255, dtype=np.uint8)
    for y in range(n_sq_y):
        for x in range(n_sq_x):
            if (x + y) % 2 == 1:
                x0 = margin_px + x * square_size_px
                y0 = margin_px + y * square_size_px
                board[y0:y0 + square_size_px, x0:x0 + square_size_px] = 0

    return board
