"""Scale change estimation during climb/descent.

Mathematical foundation:
    During vertical velocity (climb or descent), image features expand or contract.
    For two features with ground distance D:
        d1 = f * D / H1
        d2 = f * D / H2
    Ratio of feature spacing:
        s = s1 / s2 = H2 / H1
    With barometric height change dH = H1 - H2:
        1 - s1 / s2 = 1 - H2 / H1 = (H1 - H2) / H1 = dH / H1
    Therefore:
        H1 = dH / (1 - s1 / s2)
        H2 = H1 - dH
"""

from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np


@dataclass
class ScaleExpansionResult:
    """Result of height estimation from scale expansion."""
    height_h1_m: float
    height_h2_m: float
    scale_ratio: float  # s1 / s2
    delta_baro_m: float  # H1 - H2
    confidence: float
    num_pairs: int
    is_valid: bool
    status_message: str


class ScaleFromExpansionEngine:
    """Estimates absolute height from image scale ratio and barometric delta during vertical motion."""

    def __init__(
        self,
        min_delta_baro_m: float = 0.5,
        min_scale_change: float = 0.02,  # at least 2% scale change
        max_pairs_sampled: int = 500,
        min_features: int = 15,
        min_altitude_m: float = 0.5,
        max_altitude_m: float = 350.0,
    ) -> None:
        """Initialize scale-from-expansion engine.

        Args:
            min_delta_baro_m: Minimum vertical baro displacement magnitude required (|dH| >= min_delta_baro_m).
            min_scale_change: Minimum |1 - s1/s2| ratio to prevent division by near-zero.
            max_pairs_sampled: Maximum number of feature pairs to sample for pairwise distance ratio.
            min_features: Minimum tracked features required.
            min_altitude_m: Lower altitude bound (m).
            max_altitude_m: Upper altitude bound (m).
        """
        self.min_delta_baro_m = float(min_delta_baro_m)
        self.min_scale_change = float(min_scale_change)
        self.max_pairs_sampled = int(max_pairs_sampled)
        self.min_features = int(min_features)
        self.min_altitude_m = float(min_altitude_m)
        self.max_altitude_m = float(max_altitude_m)

    def estimate_scale_ratio(
        self,
        pts1: np.ndarray,
        pts2: np.ndarray,
    ) -> Tuple[float, int]:
        """Compute the robust scale ratio s1 / s2 from matched point sets.

        Args:
            pts1: Matched pixel positions in frame 1 (N, 2).
            pts2: Matched pixel positions in frame 2 (N, 2).

        Returns:
            Tuple of (scale_ratio, num_valid_pairs).
        """
        p1 = np.atleast_2d(pts1).astype(np.float64)
        p2 = np.atleast_2d(pts2).astype(np.float64)
        n = p1.shape[0]

        if n < self.min_features:
            return 1.0, 0

        # Method 1: Centroid-relative radial distances
        c1 = np.median(p1, axis=0)
        c2 = np.median(p2, axis=0)
        r1 = np.hypot(p1[:, 0] - c1[0], p1[:, 1] - c1[1])
        r2 = np.hypot(p2[:, 0] - c2[0], p2[:, 1] - c2[1])

        # Filter points too close to centroid (< 15 px)
        valid_rad = (r1 > 15.0) & (r2 > 15.0)
        if np.sum(valid_rad) >= self.min_features:
            radial_ratios = r1[valid_rad] / r2[valid_rad]
            scale_from_centroid = float(np.median(radial_ratios))
        else:
            scale_from_centroid = 1.0

        # Method 2: Sampled pairwise distance ratios
        rng = np.random.default_rng(seed=42)
        total_possible_pairs = n * (n - 1) // 2
        num_samples = min(self.max_pairs_sampled, total_possible_pairs)

        idx_i = rng.integers(0, n, size=num_samples * 2)
        idx_j = rng.integers(0, n, size=num_samples * 2)
        distinct = idx_i != idx_j
        idx_i, idx_j = idx_i[distinct][:num_samples], idx_j[distinct][:num_samples]

        d1 = np.hypot(p1[idx_i, 0] - p1[idx_j, 0], p1[idx_i, 1] - p1[idx_j, 1])
        d2 = np.hypot(p2[idx_i, 0] - p2[idx_j, 0], p2[idx_i, 1] - p2[idx_j, 1])

        valid_pairs = (d1 > 20.0) & (d2 > 20.0)
        if np.sum(valid_pairs) < self.min_features:
            return scale_from_centroid, int(np.sum(valid_rad))

        pairwise_ratios = d1[valid_pairs] / d2[valid_pairs]
        scale_from_pairs = float(np.median(pairwise_ratios))

        # Combined consensus: mean of centroid and pairwise medians
        combined_scale = 0.5 * (scale_from_centroid + scale_from_pairs)
        return combined_scale, int(np.sum(valid_pairs))

    def estimate_height_from_expansion(
        self,
        pts1: np.ndarray,
        pts2: np.ndarray,
        delta_baro_m: float,
    ) -> ScaleExpansionResult:
        """Estimate heights H1 and H2 from feature scale change and barometric altitude delta.

        Args:
            pts1: Matched feature coordinates in frame 1 (N, 2).
            pts2: Matched feature coordinates in frame 2 (N, 2).
            delta_baro_m: Vertical altitude difference dH = H1 - H2 (positive if descending, negative if climbing).

        Returns:
            ScaleExpansionResult with estimated H1, H2, scale ratio, and confidence.
        """
        pts1 = np.atleast_2d(pts1).astype(np.float64)
        pts2 = np.atleast_2d(pts2).astype(np.float64)
        n = pts1.shape[0]

        if n < self.min_features:
            return ScaleExpansionResult(
                height_h1_m=0.0,
                height_h2_m=0.0,
                scale_ratio=1.0,
                delta_baro_m=delta_baro_m,
                confidence=0.0,
                num_pairs=0,
                is_valid=False,
                status_message=f"Insufficient features: {n} < {self.min_features}"
            )

        if abs(delta_baro_m) < self.min_delta_baro_m:
            return ScaleExpansionResult(
                height_h1_m=0.0,
                height_h2_m=0.0,
                scale_ratio=1.0,
                delta_baro_m=delta_baro_m,
                confidence=0.0,
                num_pairs=0,
                is_valid=False,
                status_message=f"Baro delta too small: {abs(delta_baro_m):.2f} m < {self.min_delta_baro_m:.2f} m"
            )

        scale_ratio, num_pairs = self.estimate_scale_ratio(pts1, pts2)
        scale_change = 1.0 - scale_ratio

        if abs(scale_change) < self.min_scale_change:
            return ScaleExpansionResult(
                height_h1_m=0.0,
                height_h2_m=0.0,
                scale_ratio=scale_ratio,
                delta_baro_m=delta_baro_m,
                confidence=0.0,
                num_pairs=num_pairs,
                is_valid=False,
                status_message=f"Scale change too small: |1 - s| = {abs(scale_change):.4f} < {self.min_scale_change}"
            )

        # Check physical consistency: signs of delta_baro and scale_change must match
        # Climbing: H2 > H1 -> delta_baro < 0, s1/s2 > 1 -> scale_change < 0
        # Descending: H2 < H1 -> delta_baro > 0, s1/s2 < 1 -> scale_change > 0
        if (delta_baro_m * scale_change) <= 0:
            return ScaleExpansionResult(
                height_h1_m=0.0,
                height_h2_m=0.0,
                scale_ratio=scale_ratio,
                delta_baro_m=delta_baro_m,
                confidence=0.0,
                num_pairs=num_pairs,
                is_valid=False,
                status_message="Sign mismatch between barometric change and visual scale expansion"
            )

        h1 = delta_baro_m / scale_change
        h2 = h1 - delta_baro_m

        if not (self.min_altitude_m <= h1 <= self.max_altitude_m and self.min_altitude_m <= h2 <= self.max_altitude_m):
            return ScaleExpansionResult(
                height_h1_m=h1,
                height_h2_m=h2,
                scale_ratio=scale_ratio,
                delta_baro_m=delta_baro_m,
                confidence=0.0,
                num_pairs=num_pairs,
                is_valid=False,
                status_message=f"Estimated heights out of bounds: H1={h1:.1f}m, H2={h2:.1f}m"
            )

        # Confidence based on magnitude of baro delta and number of pairs
        baro_score = min(1.0, abs(delta_baro_m) / 3.0)
        pairs_score = min(1.0, num_pairs / 100.0)
        confidence = float(np.clip(baro_score * 0.6 + pairs_score * 0.4, 0.0, 1.0))

        return ScaleExpansionResult(
            height_h1_m=h1,
            height_h2_m=h2,
            scale_ratio=scale_ratio,
            delta_baro_m=delta_baro_m,
            confidence=confidence,
            num_pairs=num_pairs,
            is_valid=True,
            status_message="Valid scale expansion height estimate"
        )

