"""Telemetry and frame logger for VPS.

Records synchronized camera frames and MAVLink telemetry to disk for offline
replay, algorithm benchmarking, and flight diagnostics.
"""

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, Dict, Any
import csv
import json
import time
import cv2
import numpy as np


@dataclass
class TelemetryRecord:
    """Synchronized record of telemetry at camera exposure time."""
    timestamp_s: float
    frame_idx: int
    image_rel_path: str = ""
    # Attitude in radians
    roll_rad: float = 0.0
    pitch_rad: float = 0.0
    yaw_rad: float = 0.0
    # Gyro angular rates in rad/s (body frame)
    gyro_x_rad_s: float = 0.0
    gyro_y_rad_s: float = 0.0
    gyro_z_rad_s: float = 0.0
    # Velocity in NED (m/s)
    vel_n_mps: float = 0.0
    vel_e_mps: float = 0.0
    vel_d_mps: float = 0.0
    # Barometer
    baro_alt_m: float = 0.0
    # Estimator outputs
    estimated_h_m: Optional[float] = None
    confidence: Optional[float] = None
    # Ground truth (if from simulator or ArUco validation)
    groundtruth_h_m: Optional[float] = None


class TelemetryLogger:
    """Logs synchronized video frames and telemetry packets to disk."""

    def __init__(
        self,
        output_dir: str | Path,
        session_name: Optional[str] = None,
        save_images: bool = True,
        image_format: str = "jpg",
    ) -> None:
        """Initialize logger.

        Args:
            output_dir: Root output directory.
            session_name: Subfolder name for session. Defaults to timestamp.
            save_images: Whether to save individual image frames to disk.
            image_format: Image extension ("jpg" or "png").
        """
        if session_name is None:
            session_name = f"session_{time.strftime('%Y%m%d_%H%M%S')}"

        self.session_dir = Path(output_dir) / session_name
        self.images_dir = self.session_dir / "images"
        self.save_images = save_images
        self.image_format = image_format

        self.session_dir.mkdir(parents=True, exist_ok=True)
        if self.save_images:
            self.images_dir.mkdir(parents=True, exist_ok=True)

        self.csv_path = self.session_dir / "telemetry.csv"
        self._csv_file = open(self.csv_path, "w", newline="", encoding="utf-8")
        self._csv_writer = csv.DictWriter(
            self._csv_file,
            fieldnames=list(TelemetryRecord.__annotations__.keys())
        )
        self._csv_writer.writeheader()
        self._frame_counter = 0

    def log(
        self,
        record: TelemetryRecord,
        frame: Optional[np.ndarray] = None,
    ) -> None:
        """Log a single telemetry record and optional frame.

        Args:
            record: Populated TelemetryRecord dataclass.
            frame: Optional camera frame (grayscale or BGR).
        """
        if self.save_images and frame is not None:
            image_name = f"frame_{record.frame_idx:06d}.{self.image_format}"
            full_img_path = self.images_dir / image_name
            cv2.imwrite(str(full_img_path), frame)
            record.image_rel_path = f"images/{image_name}"

        self._csv_writer.writerow(asdict(record))
        self._csv_file.flush()
        self._frame_counter += 1

    def close(self) -> None:
        """Flush and close open log files."""
        if hasattr(self, "_csv_file") and not self._csv_file.closed:
            self._csv_file.flush()
            self._csv_file.close()

    def __enter__(self) -> "TelemetryLogger":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()
