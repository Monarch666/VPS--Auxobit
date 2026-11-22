"""Infinity-focus and optical sharpness inspection tool for Sony IMX296.

Assesses whether a downward-pointing drone camera lens is set to the correct
hyperfocal/infinity focus distance before flight.

Sharpness Metrics:
1. Modified Laplacian Variance: Var(Laplacian(I))
2. Tenengrad Gradient Energy: Mean(Sobel_x(I)^2 + Sobel_y(I)^2)
3. High-Frequency Fourier Spectral Ratio: Energy(f > 0.25 * f_nyquist) / Energy_total
4. 9-Zone Spatial Flatness: Checks focus across center, edges, and corners to detect lens tilt.
"""

from dataclasses import dataclass
from typing import Dict, Any, Tuple, Optional
import numpy as np
import cv2


@dataclass
class FocusQualityMetrics:
    """Optical sharpness and focus metrics for an image."""
    laplacian_variance: float
    tenengrad_energy: float
    fft_high_frequency_ratio: float
    zone_sharpness: Dict[str, float]  # 3x3 spatial grid sharpness
    focus_status: str  # "OPTIMAL_INFINITY_FOCUS", "ACCEPTABLE_FOCUS", "DEFOCUSED"
    is_focused: bool


class InfinityFocusInspector:
    """Inspects camera focus sharpness across image plane."""

    def __init__(
        self,
        min_laplacian_var_optimal: float = 300.0,
        min_laplacian_var_acceptable: float = 120.0,
    ) -> None:
        """Initialize focus inspector.

        Args:
            min_laplacian_var_optimal: Threshold for crisp infinity focus.
            min_laplacian_var_acceptable: Lower bound for acceptable focus.
        """
        self.thresh_optimal = float(min_laplacian_var_optimal)
        self.thresh_acceptable = float(min_laplacian_var_acceptable)

    def evaluate_sharpness(self, frame: np.ndarray) -> FocusQualityMetrics:
        """Evaluate focus and sharpness metrics on an image frame.

        Args:
            frame: Input image (BGR or grayscale).

        Returns:
            FocusQualityMetrics dataclass.
        """
        if frame.ndim == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame

        h, w = gray.shape

        # 1. Laplacian Variance
        lap = cv2.Laplacian(gray, cv2.CV_64F, ksize=3)
        lap_var = float(lap.var())

        # 2. Tenengrad Energy
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        tenengrad = float(np.mean(sobelx * sobelx + sobely * sobely))

        # 3. High-Frequency FFT Power
        f = np.fft.fft2(gray.astype(np.float64))
        fshift = np.fft.fftshift(f)
        magnitude_spectrum = np.abs(fshift)

        # Create high-pass mask: radius > 0.25 of half-dimension
        cy, cx = h // 2, w // 2
        y, x = np.ogrid[:h, :w]
        r = np.hypot(x - cx, y - cy)
        max_r = min(cx, cy)
        high_pass_mask = (r > (0.25 * max_r))

        total_power = float(np.sum(magnitude_spectrum))
        high_power = float(np.sum(magnitude_spectrum[high_pass_mask]))
        fft_ratio = float(high_power / max(total_power, 1e-6))

        # 4. 3x3 Spatial Grid Sharpness
        zones = {}
        zone_names = [
            ["top_left", "top_center", "top_right"],
            ["mid_left", "center",     "mid_right"],
            ["bot_left", "bot_center", "bot_right"],
        ]
        zh, zw = h // 3, w // 3
        for r_idx in range(3):
            for c_idx in range(3):
                y0, y1 = r_idx * zh, (r_idx + 1) * zh
                x0, x1 = c_idx * zw, (c_idx + 1) * zw
                patch = gray[y0:y1, x0:x1]
                z_lap = cv2.Laplacian(patch, cv2.CV_64F).var()
                zones[zone_names[r_idx][c_idx]] = float(z_lap)

        # 5. Status Determination
        if lap_var >= self.thresh_optimal:
            status = "OPTIMAL_INFINITY_FOCUS"
            is_focused = True
        elif lap_var >= self.thresh_acceptable:
            status = "ACCEPTABLE_FOCUS"
            is_focused = True
        else:
            status = "DEFOCUSED"
            is_focused = False

        return FocusQualityMetrics(
            laplacian_variance=lap_var,
            tenengrad_energy=tenengrad,
            fft_high_frequency_ratio=fft_ratio,
            zone_sharpness=zones,
            focus_status=status,
            is_focused=is_focused,
        )

    def draw_focus_overlay(self, frame: np.ndarray, metrics: FocusQualityMetrics) -> np.ndarray:
        """Render focus metrics HUD overlay onto frame for live inspection."""
        overlay = frame.copy()
        color = (0, 255, 0) if metrics.is_focused else (0, 0, 255)
        text1 = f"Focus Status: {metrics.focus_status}"
        text2 = f"Laplacian Var: {metrics.laplacian_variance:.1f} (Opt: >{self.thresh_optimal:.0f})"
        text3 = f"Tenengrad: {metrics.tenengrad_energy:.1f} | FFT Ratio: {metrics.fft_high_frequency_ratio:.3f}"

        cv2.putText(overlay, text1, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        cv2.putText(overlay, text2, (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(overlay, text3, (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        # 3x3 Grid indicator
        h, w = frame.shape[:2]
        cv2.line(overlay, (w // 3, 0), (w // 3, h), (100, 100, 100), 1)
        cv2.line(overlay, (2 * w // 3, 0), (2 * w // 3, h), (100, 100, 100), 1)
        cv2.line(overlay, (0, h // 3), (w, h // 3), (100, 100, 100), 1)
        cv2.line(overlay, (0, 2 * h // 3), (w, 2 * h // 3), (100, 100, 100), 1)

        return overlay
