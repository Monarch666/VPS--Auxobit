"""Telemetry and video log replayer.

Loads logged sessions and iterates through frames and telemetry packets with
deterministic or wall-clock timing.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional, Dict, Any
import csv
import time
import cv2
import numpy as np

from .logger import TelemetryRecord


@dataclass
class ReplayPacket:
    """Single packet yielded during log replay."""
    record: TelemetryRecord
    frame: Optional[np.ndarray] = None


class LogReplayer:
    """Replays logged sessions consisting of telemetry CSV and stored frame images."""

    def __init__(
        self,
        session_dir: str | Path,
        load_images: bool = True,
        playback_rate: float = 0.0,  # 0.0 means as-fast-as-possible (generator)
    ) -> None:
        """Initialize replayer.

        Args:
            session_dir: Path to directory containing telemetry.csv and images/.
            load_images: Whether to load frame images from disk.
            playback_rate: Speed multiplier (1.0 = real-time, 2.0 = 2x speed, 0.0 = full-speed no delay).
        """
        self.session_dir = Path(session_dir)
        self.csv_path = self.session_dir / "telemetry.csv"
        self.load_images = load_images
        self.playback_rate = float(playback_rate)

        if not self.csv_path.is_file():
            raise FileNotFoundError(f"Telemetry CSV not found in {self.session_dir}")

    def __len__(self) -> int:
        """Count total records in telemetry CSV."""
        with open(self.csv_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader, None)
            return sum(1 for _ in reader)

    def iterate(self) -> Iterator[ReplayPacket]:
        """Yield ReplayPackets sequentially from the log."""
        with open(self.csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            prev_sim_time: Optional[float] = None

            for row in reader:
                # Convert types
                record = TelemetryRecord(
                    timestamp_s=float(row["timestamp_s"]),
                    frame_idx=int(row["frame_idx"]),
                    image_rel_path=row.get("image_rel_path", ""),
                    roll_rad=float(row.get("roll_rad", 0.0)),
                    pitch_rad=float(row.get("pitch_rad", 0.0)),
                    yaw_rad=float(row.get("yaw_rad", 0.0)),
                    gyro_x_rad_s=float(row.get("gyro_x_rad_s", 0.0)),
                    gyro_y_rad_s=float(row.get("gyro_y_rad_s", 0.0)),
                    gyro_z_rad_s=float(row.get("gyro_z_rad_s", 0.0)),
                    vel_n_mps=float(row.get("vel_n_mps", 0.0)),
                    vel_e_mps=float(row.get("vel_e_mps", 0.0)),
                    vel_d_mps=float(row.get("vel_d_mps", 0.0)),
                    baro_alt_m=float(row.get("baro_alt_m", 0.0)),
                    estimated_h_m=float(row["estimated_h_m"]) if row.get("estimated_h_m") else None,
                    confidence=float(row["confidence"]) if row.get("confidence") else None,
                    groundtruth_h_m=float(row["groundtruth_h_m"]) if row.get("groundtruth_h_m") else None,
                )

                # Optional wall-clock delay
                if self.playback_rate > 0.0 and prev_sim_time is not None:
                    dt_sim = record.timestamp_s - prev_sim_time
                    if dt_sim > 0:
                        time.sleep(dt_sim / self.playback_rate)
                prev_sim_time = record.timestamp_s

                # Optional image loading
                frame: Optional[np.ndarray] = None
                if self.load_images and record.image_rel_path:
                    img_full_path = self.session_dir / record.image_rel_path
                    if img_full_path.is_file():
                        frame = cv2.imread(str(img_full_path))

                yield ReplayPacket(record=record, frame=frame)

