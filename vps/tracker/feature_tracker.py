"""Feature detector and pyramidal Lucas-Kanade tracker with forward-backward check.

Implements:
- Shi-Tomasi corner detection with minimum distance suppression
- Multi-scale pyramidal Lucas-Kanade optical flow
- Forward-backward consistency check to reject occlusions and tracking drift
- RANSAC affine/homography geometric outlier pruning
"""

from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np
import cv2

from vps.config import TrackerConfig


@dataclass
class TrackingResult:
    """Result of inter-frame feature tracking."""
    pts_prev: np.ndarray  # (N, 2)
    pts_curr: np.ndarray  # (N, 2)
    inlier_mask: np.ndarray  # (N,) bool
    num_tracked: int
    num_inliers: int
    median_flow_px: float


class FeatureTracker:
    """Pyramidal Lucas-Kanade feature tracker with consistency checks."""

    def __init__(self, config: Optional[TrackerConfig] = None) -> None:
        """Initialize feature tracker.

        Args:
            config: Tracker configuration parameters.
        """
        self.cfg = config if config is not None else TrackerConfig()
        self.lk_params = dict(
            winSize=self.cfg.lk_win_size,
            maxLevel=self.cfg.lk_max_level,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
        )

    def detect_features(
        self,
        image_gray: np.ndarray,
        mask: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Detect strong Shi-Tomasi corner features.

        Args:
            image_gray: Grayscale input image (uint8).
            mask: Optional binary mask of valid detection regions.

        Returns:
            Array of shape (N, 2) of detected (u, v) pixel coordinates.
        """
        corners = cv2.goodFeaturesToTrack(
            image_gray,
            maxCorners=self.cfg.max_features,
            qualityLevel=self.cfg.quality_level,
            minDistance=self.cfg.min_distance_px,
            mask=mask,
            blockSize=7,
        )
        if corners is None or len(corners) == 0:
            return np.empty((0, 2), dtype=np.float32)
        return corners.reshape(-1, 2).astype(np.float32)

    def track(
        self,
        prev_gray: np.ndarray,
        curr_gray: np.ndarray,
        prev_pts: np.ndarray,
    ) -> TrackingResult:
        """Track features from previous frame to current frame with forward-backward verification.

        Args:
            prev_gray: Grayscale image of previous / key frame.
            curr_gray: Grayscale image of current frame.
            prev_pts: Feature points in previous frame (N, 2).

        Returns:
            TrackingResult containing matched points and verified inlier mask.
        """
        pts_p = np.ascontiguousarray(prev_pts, dtype=np.float32)
        n_pts = len(pts_p)

        if n_pts == 0:
            return TrackingResult(
                pts_prev=np.empty((0, 2), dtype=np.float32),
                pts_curr=np.empty((0, 2), dtype=np.float32),
                inlier_mask=np.zeros(0, dtype=bool),
                num_tracked=0,
                num_inliers=0,
                median_flow_px=0.0,
            )

        # 1. Forward Optical Flow: prev -> curr
        curr_pts_fwd, status_fwd, _ = cv2.calcOpticalFlowPyrLK(
            prev_gray, curr_gray, pts_p, None, **self.lk_params
        )

        valid_fwd = (status_fwd.ravel() == 1)

        # 2. Backward Optical Flow: curr -> prev
        prev_pts_bwd, status_bwd, _ = cv2.calcOpticalFlowPyrLK(
            curr_gray, prev_gray, curr_pts_fwd, None, **self.lk_params
        )

        valid_bwd = (status_bwd.ravel() == 1)

        # 3. Forward-Backward Consistency Check: ||prev_pts - prev_pts_bwd|| < threshold
        fb_dist = np.hypot(pts_p[:, 0] - prev_pts_bwd[:, 0], pts_p[:, 1] - prev_pts_bwd[:, 1])
        valid_fb = valid_fwd & valid_bwd & (fb_dist <= self.cfg.forward_backward_thresh_px)

        # 4. Image Boundary Check
        h, w = curr_gray.shape[:2]
        in_bounds = (
            (curr_pts_fwd[:, 0] >= 5) & (curr_pts_fwd[:, 0] < w - 5) &
            (curr_pts_fwd[:, 1] >= 5) & (curr_pts_fwd[:, 1] < h - 5)
        )
        initial_mask = valid_fb & in_bounds
        n_valid = int(np.sum(initial_mask))

        if n_valid < 4:
            return TrackingResult(
                pts_prev=pts_p,
                pts_curr=curr_pts_fwd,
                inlier_mask=initial_mask,
                num_tracked=n_valid,
                num_inliers=n_valid,
                median_flow_px=0.0,
            )

        # 5. RANSAC Homography Geometric Filtering
        ransac_mask = np.zeros(n_pts, dtype=bool)
        p_prev_valid = pts_p[initial_mask]
        p_curr_valid = curr_pts_fwd[initial_mask]

        _, r_mask = cv2.findHomography(
            p_prev_valid,
            p_curr_valid,
            method=cv2.RANSAC,
            ransacReprojThreshold=self.cfg.ransac_reproj_thresh_px,
            maxIters=2000,
        )

        if r_mask is not None:
            ransac_submask = (r_mask.ravel() == 1)
            valid_indices = np.where(initial_mask)[0]
            ransac_mask[valid_indices[ransac_submask]] = True
        else:
            ransac_mask = initial_mask

        n_inliers = int(np.sum(ransac_mask))
        if n_inliers > 0:
            flow_vectors = curr_pts_fwd[ransac_mask] - pts_p[ransac_mask]
            median_flow = float(np.median(np.hypot(flow_vectors[:, 0], flow_vectors[:, 1])))
        else:
            median_flow = 0.0

        return TrackingResult(
            pts_prev=pts_p,
            pts_curr=curr_pts_fwd,
            inlier_mask=ransac_mask,
            num_tracked=n_valid,
            num_inliers=n_inliers,
            median_flow_px=median_flow,
        )
