"""Ground baseline height offset calibrator for hand-held and landing gear calibration.

For hand-held drone testing:
    When the drone sits on the ground on its landing gear/skids, the downward
    camera lens is positioned at a fixed physical vertical offset (h_baseline_offset)
    above the ground plane (typically 10 to 20 cm).

    When holding the drone by hand and raising or lowering it, the true Above
    Ground Level (AGL) height of the drone frame is:
        h_AGL = h_camera_measured - h_baseline_offset

    This calibrator measures or records the baseline ground offset so that
    when resting on the ground, h_AGL is exactly 0.0 m.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Any
import numpy as np
import yaml

from vps.config import VPSConfig


@dataclass
class BaselineCalibrationResult:
    """Result of ground baseline height calibration."""
    baseline_offset_m: float
    resting_ground_agl_m: float
    num_samples_averaged: int
    offset_std_dev_m: float
    is_valid: bool


class GroundBaselineCalibrator:
    """Manages ground baseline offset calibration for drone landing gear and hand-held testing."""

    def __init__(self, config_path: Optional[Path | str] = None) -> None:
        """Initialize calibrator.

        Args:
            config_path: Path to configuration YAML to update with calibrated baseline.
        """
        self.config_path = Path(config_path) if config_path is not None else Path("config/default_config.yaml")

    def calibrate_from_static_measurements(
        self,
        ground_distance_samples_m: list[float] | np.ndarray,
    ) -> BaselineCalibrationResult:
        """Calibrate baseline offset by averaging camera distance measurements while resting on ground.

        Args:
            ground_distance_samples_m: Array of measured distances from camera to ground while on landing gear.

        Returns:
            BaselineCalibrationResult with calibrated offset.
        """
        samples = np.asarray(ground_distance_samples_m, dtype=np.float64)
        if len(samples) == 0:
            raise ValueError("No measurement samples provided")

        mean_offset = float(np.mean(samples))
        std_offset = float(np.std(samples))

        # Sanity check: drone landing gear height is typically 0.05m to 0.50m
        is_valid = (0.02 <= mean_offset <= 0.60) and (std_offset < 0.03)

        return BaselineCalibrationResult(
            baseline_offset_m=mean_offset,
            resting_ground_agl_m=0.0,
            num_samples_averaged=len(samples),
            offset_std_dev_m=std_offset,
            is_valid=is_valid,
        )

    def apply_baseline_correction(
        self,
        raw_camera_height_m: float,
        baseline_offset_m: float,
    ) -> float:
        """Transform raw optical center height to drone landing gear AGL height.

        Args:
            raw_camera_height_m: Height measured from camera optical center to ground.
            baseline_offset_m: Calibrated vertical distance from camera to landing gear.

        Returns:
            Corrected drone AGL height (meters), clamped to >= 0.0.
        """
        corrected = raw_camera_height_m - baseline_offset_m
        return float(max(0.0, corrected))

    def save_baseline_to_config(self, baseline_offset_m: float) -> None:
        """Save calibrated baseline offset into the configuration YAML."""
        if not self.config_path.is_file():
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")

        cfg = VPSConfig.from_yaml(self.config_path)
        cfg.camera.baseline_ground_offset_m = float(baseline_offset_m)
        cfg.to_yaml(self.config_path)
