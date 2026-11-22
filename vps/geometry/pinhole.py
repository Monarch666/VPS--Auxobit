"""Pinhole camera model with lens undistortion and ray projection.

Coordinate convention:
- Camera frame C: X right, Y down, Z forward (pointing vertically down to ground).
- Image plane: u horizontal (0 to width-1), v vertical (0 to height-1).
"""

from typing import Tuple, Optional
import numpy as np
import cv2


class PinholeModel:
    """Pinhole camera model with OpenCV radial-tangential lens distortion."""

    def __init__(
        self,
        K: np.ndarray,
        D: Optional[np.ndarray] = None,
        width: int = 1456,
        height: int = 1088,
    ) -> None:
        """Initialize pinhole model.

        Args:
            K: 3x3 intrinsic calibration matrix [[fx, 0, cx], [0, fy, cy], [0, 0, 1]].
            D: Distortion coefficients [k1, k2, p1, p2, k3] or None for ideal lens.
            width: Image width in pixels.
            height: Image height in pixels.
        """
        self.K = np.array(K, dtype=np.float64).reshape((3, 3))
        self.K_inv = np.linalg.inv(self.K)
        if D is not None:
            self.D = np.array(D, dtype=np.float64).ravel()
        else:
            self.D = np.zeros(5, dtype=np.float64)
        self.width = width
        self.height = height

    @property
    def fx(self) -> float:
        return float(self.K[0, 0])

    @property
    def fy(self) -> float:
        return float(self.K[1, 1])

    @property
    def cx(self) -> float:
        return float(self.K[0, 2])

    @property
    def cy(self) -> float:
        return float(self.K[1, 2])

    @property
    def f_mean(self) -> float:
        return (self.fx + self.fy) / 2.0

    @property
    def hfov_deg(self) -> float:
        """Horizontal field of view in degrees."""
        return float(np.degrees(2.0 * np.arctan(self.width / (2.0 * self.fx))))

    @property
    def vfov_deg(self) -> float:
        """Vertical field of view in degrees."""
        return float(np.degrees(2.0 * np.arctan(self.height / (2.0 * self.fy))))

    def compute_gsd(self, altitude_m: float) -> Tuple[float, float]:
        """Compute ground sample distance (GSD) in meters/pixel at given altitude.

        Args:
            altitude_m: Height above ground level in meters (H > 0).

        Returns:
            Tuple of (gsd_x, gsd_y) in meters/pixel.
        """
        if altitude_m <= 0:
            raise ValueError(f"Altitude must be strictly positive, got {altitude_m}")
        gsd_x = altitude_m / self.fx
        gsd_y = altitude_m / self.fy
        return gsd_x, gsd_y

    def project_3d_to_pixel(self, points_c: np.ndarray) -> np.ndarray:
        """Project 3D points in camera frame to 2D pixel coordinates.

        Assumes ideal pinhole projection without distortion (or pre-rectified).

        Args:
            points_c: Array of shape (N, 3) or (3,) representing points in camera coordinates.

        Returns:
            Array of shape (N, 2) of pixel coordinates (u, v).
        """
        pts = np.atleast_2d(points_c).astype(np.float64)
        z = pts[:, 2:3]
        if np.any(z <= 0):
            raise ValueError("All points must have positive depth (Z > 0) in camera frame")
        norm_pts = pts[:, :2] / z
        u = self.fx * norm_pts[:, 0] + self.cx
        v = self.fy * norm_pts[:, 1] + self.cy
        return np.column_stack([u, v])

    def backproject_pixel_to_ground(self, pixels: np.ndarray, altitude_m: float) -> np.ndarray:
        """Backproject pixel coordinates to 3D metric ground plane assuming flat ground at depth Z = altitude.

        Args:
            pixels: Array of shape (N, 2) in pixel coordinates (u, v).
            altitude_m: Height above ground level in meters (Z = altitude).

        Returns:
            Array of shape (N, 3) in camera frame (X, Y, Z).
        """
        if altitude_m <= 0:
            raise ValueError(f"Altitude must be strictly positive, got {altitude_m}")
        pts = np.atleast_2d(pixels).astype(np.float64)
        n = pts.shape[0]
        homogeneous = np.column_stack([pts, np.ones(n)])
        # Ray direction in camera frame
        rays = (self.K_inv @ homogeneous.T).T
        # Scale rays to depth Z = altitude
        scale = altitude_m / rays[:, 2:3]
        points_c = rays * scale
        return points_c

    def undistort_points(self, points: np.ndarray) -> np.ndarray:
        """Undistort 2D pixel coordinates using camera distortion coefficients.

        Args:
            points: Array of shape (N, 2) or (N, 1, 2) of distorted pixel coordinates.

        Returns:
            Array of shape (N, 2) of undistorted pixel coordinates.
        """
        pts = np.ascontiguousarray(points, dtype=np.float32)
        if pts.ndim == 2:
            pts = pts[:, np.newaxis, :]
        undistorted = cv2.undistortPoints(pts, self.K, self.D, P=self.K)
        return undistorted.reshape(-1, 2).astype(np.float64)

    def undistort_image(self, image: np.ndarray) -> np.ndarray:
        """Undistort a full image frame.

        Args:
            image: Input image (grayscale or BGR).

        Returns:
            Undistorted image.
        """
        return cv2.undistort(image, self.K, self.D, None, self.K)
