"""Multi-Scale Feature and Phase Correlation Map Matcher.

Matches orthorectified live camera frames against offline georeferenced map tiles:
1. Primary matching: Rotation-invariant ORB / SIFT feature matching with Lowe's ratio
   test and RANSAC affine/homography geometric verification.
2. Fallback matching: FFT-based Phase Correlation on normalized Sobel gradient maps
   for textureless, repetitive, or severe seasonal appearance gap conditions.
3. Produces absolute metric Local NED position fixes (x_N, y_E) with 2x2 covariance
   matrices and inlier-based confidence scores.
"""

from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Dict, Any
import time
import numpy as np
import cv2

from vps.map_db.runtime_lookup import TileLookupResult
from .orthorectifier import OrthoResult


@dataclass
class MapMatchResult:
    """Absolute georeferenced position fix produced by map matching."""
    timestamp_s: float
    is_valid: bool
    # Estimated absolute position in Local NED (meters)
    pos_ned_m: np.ndarray  # shape (2,) [North, East]
    # 2x2 horizontal position error covariance matrix (m^2)
    cov_ned_m2: np.ndarray  # shape (2, 2)
    confidence: float  # [0.0, 1.0]
    match_method: str  # "ORB_RANSAC", "SIFT_RANSAC", "PHASE_CORRELATION", "NONE"
    tile_id: str
    num_matches: int
    num_inliers: int
    inlier_ratio: float
    reprojection_rmse_px: float
    correlation_peak: float
    processing_time_ms: float
    status_message: str = ""


class MapMatcher:
    """Matches live orthorectified imagery against pre-tiled satellite map database."""

    def __init__(
        self,
        min_inliers_feature: int = 10,
        min_phase_correlation_peak: float = 0.32,
        ransac_reproj_thresh_px: float = 3.0,
        base_feature_sigma_m: float = 8.0,
        base_phase_sigma_m: float = 16.0,
    ) -> None:
        """Initialize map matcher.

        Args:
            min_inliers_feature: Minimum verified inliers required for feature match.
            min_phase_correlation_peak: Minimum FFT Dirac peak for phase correlation.
            ransac_reproj_thresh_px: RANSAC inlier threshold in pixels.
            base_feature_sigma_m: Baseline position standard deviation for feature fixes.
            base_phase_sigma_m: Baseline position standard deviation for phase fixes.
        """
        self.min_inliers = int(min_inliers_feature)
        self.min_peak = float(min_phase_correlation_peak)
        self.ransac_thresh = float(ransac_reproj_thresh_px)
        self.base_sigma_feat = float(base_feature_sigma_m)
        self.base_sigma_phase = float(base_phase_sigma_m)

        # Feature detectors
        self.orb = cv2.ORB_create(
            nfeatures=500,
            scaleFactor=1.2,
            nlevels=4,
            edgeThreshold=15,
            fastThreshold=10,
        )
        self.bf_orb = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    def match_frame_to_map(
        self,
        ortho: OrthoResult,
        candidate_tiles: List[TileLookupResult],
        pos_prior_ned_m: np.ndarray,
        search_radius_m: float = 400.0,
        timestamp_s: float = 0.0,
    ) -> MapMatchResult:
        """Execute multi-strategy matching of orthorectified live frame against candidate tiles.

        Args:
            ortho: OrthoResult from FrameOrthorectifier.
            candidate_tiles: List of TileLookupResult candidates from RuntimeMapDB.
            pos_prior_ned_m: Prior drone position [North, East] (meters).
            search_radius_m: Maximum allowable innovation distance for outlier rejection.
            timestamp_s: Monotonic frame capture timestamp.

        Returns:
            MapMatchResult containing absolute fix or invalid status.
        """
        t_start = time.perf_counter()

        if not ortho.is_valid or not candidate_tiles:
            return MapMatchResult(
                timestamp_s=timestamp_s,
                is_valid=False,
                pos_ned_m=np.array(pos_prior_ned_m[:2], dtype=np.float64),
                cov_ned_m2=np.eye(2) * (100.0 ** 2),
                confidence=0.0,
                match_method="NONE",
                tile_id="",
                num_matches=0,
                num_inliers=0,
                inlier_ratio=0.0,
                reprojection_rmse_px=0.0,
                correlation_peak=0.0,
                processing_time_ms=0.0,
                status_message="EMPTY_INPUT_OR_NO_TILES",
            )

        # Extract features from live rectified frame (mask out artificial border padding)
        kp_live, desc_live = self.orb.detectAndCompute(ortho.image_gray, ortho.valid_mask)
        if desc_live is None or len(kp_live) < self.min_inliers:
            # Low feature count -> attempt phase correlation fallback immediately
            return self._fallback_phase_correlation(
                ortho, candidate_tiles, pos_prior_ned_m, search_radius_m, timestamp_s, t_start
            )

        best_result: Optional[MapMatchResult] = None
        best_score = -1.0

        # Try feature matching on each candidate tile
        for tile in candidate_tiles:
            if tile.desc_orb is None or len(tile.desc_orb) < self.min_inliers:
                continue

            # Match ORB descriptors using k-NN with k=2
            try:
                knn_matches = self.bf_orb.knnMatch(desc_live, tile.desc_orb, k=2)
            except cv2.error:
                continue

            # Lowe's ratio test (0.75)
            good_matches = []
            for m_pair in knn_matches:
                if len(m_pair) == 2:
                    m, n = m_pair
                    if m.distance < 0.75 * n.distance:
                        good_matches.append(m)

            if len(good_matches) < self.min_inliers:
                continue

            # Extract source (live ortho) and destination (tile) coordinates
            pts_live = np.float32([kp_live[m.queryIdx].pt for m in good_matches]).reshape(-1, 2)
            pts_tile = np.float32([tile.kp_orb[m.trainIdx] for m in good_matches]).reshape(-1, 2)

            # Geometric verification using RANSAC affine partial (translation + uniform scale + small rotation)
            M, inlier_mask = cv2.estimateAffinePartial2D(
                pts_live,
                pts_tile,
                method=cv2.RANSAC,
                ransacReprojThreshold=self.ransac_thresh,
                maxIters=1000,
                confidence=0.99,
            )

            if M is None or inlier_mask is None:
                continue

            inliers = inlier_mask.ravel().astype(bool)
            num_inliers = int(np.sum(inliers))
            if num_inliers < self.min_inliers:
                continue

            inlier_ratio = float(num_inliers / max(1, len(good_matches)))

            # Reprojection RMSE
            pts_live_in = pts_live[inliers]
            pts_tile_in = pts_tile[inliers]
            pts_proj = (M[:2, :2] @ pts_live_in.T).T + M[:2, 2]
            reproj_err = np.linalg.norm(pts_tile_in - pts_proj, axis=1)
            reproj_rmse = float(np.mean(reproj_err))

            # Transform live frame center (drone nadir) to tile coordinates
            center_live = np.array([ortho.center_u_px, ortho.center_v_px], dtype=np.float64)
            center_on_tile = M[:2, :2] @ center_live + M[:2, 2]
            u_tile, v_tile = float(center_on_tile[0]), float(center_on_tile[1])

            # Convert tile pixel to metric Local NED
            pos_north, pos_east = tile.pixel_to_ned(u_tile, v_tile)
            fix_ned = np.array([pos_north, pos_east], dtype=np.float64)

            # Innovation check against prior position
            innovation_dist = float(np.linalg.norm(fix_ned - pos_prior_ned_m[:2]))
            if innovation_dist > search_radius_m:
                # Gated out: too far from prior
                continue

            # Confidence score
            inlier_score = min(1.0, num_inliers / 30.0)
            ratio_score = inlier_ratio
            rmse_score = max(0.0, 1.0 - (reproj_rmse / self.ransac_thresh))
            conf = float(np.clip(inlier_score * 0.5 + ratio_score * 0.3 + rmse_score * 0.2, 0.0, 1.0))

            score = conf * num_inliers
            if score > best_score:
                best_score = score
                # Covariance: scales inversely with inliers and confidence
                eff_sigma = (self.base_sigma_feat / max(0.3, conf)) / np.sqrt(max(1, num_inliers / 10.0))
                cov = np.eye(2, dtype=np.float64) * (eff_sigma ** 2)

                t_elapsed_ms = (time.perf_counter() - t_start) * 1000.0
                best_result = MapMatchResult(
                    timestamp_s=timestamp_s,
                    is_valid=True,
                    pos_ned_m=fix_ned,
                    cov_ned_m2=cov,
                    confidence=conf,
                    match_method="ORB_RANSAC",
                    tile_id=tile.metadata.tile_id,
                    num_matches=len(good_matches),
                    num_inliers=num_inliers,
                    inlier_ratio=inlier_ratio,
                    reprojection_rmse_px=reproj_rmse,
                    correlation_peak=0.0,
                    processing_time_ms=t_elapsed_ms,
                    status_message="ORB_RANSAC_FIX_ACCEPTED",
                )

        if best_result is not None:
            return best_result

        # If feature matching found no acceptable inliers, try phase correlation fallback
        return self._fallback_phase_correlation(
            ortho, candidate_tiles, pos_prior_ned_m, search_radius_m, timestamp_s, t_start
        )

    def _fallback_phase_correlation(
        self,
        ortho: OrthoResult,
        candidate_tiles: List[TileLookupResult],
        pos_prior_ned_m: np.ndarray,
        search_radius_m: float,
        timestamp_s: float,
        t_start: float,
    ) -> MapMatchResult:
        """Phase correlation fallback using normalized gradient magnitude maps."""
        live_grad = ortho.grad_mag
        h_live, w_live = live_grad.shape[:2]
        if h_live < 32 or w_live < 32:
            return self._empty_result(timestamp_s, pos_prior_ned_m, t_start, "LIVE_VIEW_TOO_SMALL")

        valid_px_count = int(h_live * w_live)
        if ortho.valid_mask is not None:
            ys, xs = np.where(ortho.valid_mask > 0)
            if ys.size == 0:
                return self._empty_result(timestamp_s, pos_prior_ned_m, t_start, "NO_VALID_ORTHO_PIXELS")
            valid_px_count = int(ys.size)
            valid_frac = float(valid_px_count / max(1, h_live * w_live))
            span_w = int(xs.max() - xs.min() + 1)
            span_h = int(ys.max() - ys.min() + 1)
            # Enforce large pixel span on high-resolution maps (<= 2 m/px) where low-altitude
            # crops otherwise degenerate into small padded rectangles inside the 64x64 grid.
            if ortho.gsd_m_per_px <= 2.0 and (valid_frac < 0.18 or span_w < 48 or span_h < 40):
                return self._empty_result(timestamp_s, pos_prior_ned_m, t_start, "FOOTPRINT_TOO_SMALL_FOR_GSD")
            if valid_px_count < 16:
                return self._empty_result(timestamp_s, pos_prior_ned_m, t_start, "FOOTPRINT_TOO_SMALL_FOR_GSD")

        min_grad_px = 80 if ortho.gsd_m_per_px <= 2.0 else max(16, int(0.25 * valid_px_count))
        if int(np.count_nonzero(live_grad > 20)) < min_grad_px:
            return self._empty_result(timestamp_s, pos_prior_ned_m, t_start, "LOW_GRADIENT_TEXTURE")

        hann = cv2.createHanningWindow((w_live, h_live), cv2.CV_32F)
        live_float = np.float32(live_grad)

        best_fix: Optional[MapMatchResult] = None
        best_peak = -1.0

        for tile in candidate_tiles:
            tile_grad = tile.grad_mag
            h_tile, w_tile = tile_grad.shape[:2]

            # Crop region of tile around prior position projection
            u_prior, v_prior = tile.ned_to_pixel(pos_prior_ned_m[0], pos_prior_ned_m[1])

            # Crop window around prior
            pad_w = w_live
            pad_h = h_live
            u1 = int(round(u_prior - pad_w / 2.0))
            v1 = int(round(v_prior - pad_h / 2.0))
            u2 = u1 + pad_w
            v2 = v1 + pad_h

            if u1 < 0 or v1 < 0 or u2 > w_tile or v2 > h_tile:
                # Tile seam near prior: extract overlapping intersection padded to live shape
                c_u1, c_v1 = max(0, u1), max(0, v1)
                c_u2, c_v2 = min(w_tile, u2), min(h_tile, v2)
                if (c_u2 - c_u1) < int(0.5 * pad_w) or (c_v2 - c_v1) < int(0.5 * pad_h):
                    continue
                tile_sub = np.zeros((pad_h, pad_w), dtype=np.float32)
                dx1, dy1 = c_u1 - u1, c_v1 - v1
                tile_sub[dy1 : dy1 + (c_v2 - c_v1), dx1 : dx1 + (c_u2 - c_u1)] = tile_grad[c_v1:c_v2, c_u1:c_u2]
            else:
                tile_sub = np.float32(tile_grad[v1:v2, u1:u2])
                if tile_sub.shape != live_float.shape:
                    continue

            # Run FFT Phase Correlation
            shift, response = cv2.phaseCorrelate(tile_sub, live_float, hann)
            peak = float(response)

            if peak >= self.min_peak and peak > best_peak:
                best_peak = peak
                du, dv = shift

                # True nadir pixel on tile
                u_fix_tile = u1 + (pad_w / 2.0) - du
                v_fix_tile = v1 + (pad_h / 2.0) - dv

                pos_north, pos_east = tile.pixel_to_ned(u_fix_tile, v_fix_tile)
                fix_ned = np.array([pos_north, pos_east], dtype=np.float64)

                innov = float(np.linalg.norm(fix_ned - pos_prior_ned_m[:2]))
                if innov <= search_radius_m:
                    conf = float(np.clip((peak - self.min_peak) / (0.5 - self.min_peak), 0.2, 0.9))
                    eff_sigma = self.base_sigma_phase / max(0.2, conf)
                    cov = np.eye(2, dtype=np.float64) * (eff_sigma ** 2)

                    t_elapsed_ms = (time.perf_counter() - t_start) * 1000.0
                    best_fix = MapMatchResult(
                        timestamp_s=timestamp_s,
                        is_valid=True,
                        pos_ned_m=fix_ned,
                        cov_ned_m2=cov,
                        confidence=conf,
                        match_method="PHASE_CORRELATION",
                        tile_id=tile.metadata.tile_id,
                        num_matches=1,
                        num_inliers=1,
                        inlier_ratio=1.0,
                        reprojection_rmse_px=0.0,
                        correlation_peak=peak,
                        processing_time_ms=t_elapsed_ms,
                        status_message="PHASE_CORRELATION_FIX_ACCEPTED",
                    )

        if best_fix is not None:
            return best_fix

        return self._empty_result(timestamp_s, pos_prior_ned_m, t_start, "NO_VALID_MATCH_FOUND")

    def _empty_result(
        self,
        timestamp_s: float,
        pos_prior_ned_m: np.ndarray,
        t_start: float,
        reason: str,
    ) -> MapMatchResult:
        """Construct invalid fallback result."""
        t_elapsed_ms = (time.perf_counter() - t_start) * 1000.0
        return MapMatchResult(
            timestamp_s=timestamp_s,
            is_valid=False,
            pos_ned_m=np.array(pos_prior_ned_m[:2], dtype=np.float64),
            cov_ned_m2=np.eye(2) * (150.0 ** 2),
            confidence=0.0,
            match_method="NONE",
            tile_id="",
            num_matches=0,
            num_inliers=0,
            inlier_ratio=0.0,
            reprojection_rmse_px=0.0,
            correlation_peak=0.0,
            processing_time_ms=t_elapsed_ms,
            status_message=reason,
        )

