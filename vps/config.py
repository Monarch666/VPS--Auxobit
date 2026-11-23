"""Configuration module for VPS.

Loads and validates parameters from YAML for camera, tracker, MAVLink,
fusion, and logging.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple, Any, Dict, Optional
import yaml
import numpy as np


@dataclass
class CameraConfig:
    name: str = "Sony_IMX296"
    width: int = 1456
    height: int = 1088
    fps: int = 30
    fx: float = 1207.2
    fy: float = 1199.1
    cx: float = 728.0
    cy: float = 544.0
    dist_coeffs: List[float] = field(default_factory=lambda: [-0.042, 0.015, 0.0008, -0.0004, 0.0])
    baseline_ground_offset_m: float = 0.06
    device_index: Any = 0

    @property
    def K(self) -> np.ndarray:
        """Intrinsic calibration matrix 3x3."""
        return np.array([
            [self.fx, 0.0, self.cx],
            [0.0, self.fy, self.cy],
            [0.0, 0.0, 1.0]
        ], dtype=np.float64)

    @property
    def D(self) -> np.ndarray:
        """Distortion coefficients array [k1, k2, p1, p2, k3]."""
        return np.array(self.dist_coeffs, dtype=np.float64)


@dataclass
class TrackerConfig:
    max_features: int = 250
    quality_level: float = 0.01
    min_distance_px: float = 15.0
    lk_win_size: Tuple[int, int] = (21, 21)
    lk_max_level: int = 3
    forward_backward_thresh_px: float = 1.0
    ransac_reproj_thresh_px: float = 2.0
    min_keyframe_displacement_px: float = 10.0
    min_features_threshold: int = 15


@dataclass
class MAVLinkConfig:
    connection_string: str = "udpin:localhost:14550"
    baudrate: int = 921600
    stream_rate_hz: int = 100
    distance_sensor_rate_hz: int = 15
    odometry_rate_hz: int = 25
    odometry_mode: str = "ODOMETRY"  # "ODOMETRY", "VISION_POSITION_ESTIMATE", "GPS_INPUT"
    feed_fc_ekf: bool = False  # Safety interlock: default False (ADVISORY_LOG_ONLY)
    min_confidence_to_feed: float = 0.40
    max_data_age_ms: float = 150.0
    emulate_gps: bool = False
    sensor_orientation: str = "PITCH_270"
    min_distance_cm: int = 5
    max_distance_cm: int = 40000


@dataclass
class FusionConfig:
    q_height: float = 0.1
    q_terrain: float = 0.02
    r_baro: float = 0.5
    r_vision_base: float = 0.2
    min_confidence: float = 0.3
    hover_speed_threshold_mps: float = 0.5


@dataclass
class LoggerConfig:
    enabled: bool = True
    output_dir: str = "logs"


@dataclass
class MapConfig:
    enabled: bool = True
    db_path: str = "data/map_db/aurangabad_s2"
    search_radius_m: float = 400.0
    match_rate_hz: float = 1.0
    min_inliers: int = 10
    min_confidence: float = 0.25


@dataclass
class VPSConfig:
    camera: CameraConfig = field(default_factory=CameraConfig)
    tracker: TrackerConfig = field(default_factory=TrackerConfig)
    mavlink: MAVLinkConfig = field(default_factory=MAVLinkConfig)
    fusion: FusionConfig = field(default_factory=FusionConfig)
    logger: LoggerConfig = field(default_factory=LoggerConfig)
    map: MapConfig = field(default_factory=MapConfig)

    @classmethod
    def from_yaml(cls, yaml_path: str | Path) -> "VPSConfig":
        """Load configuration from a YAML file.
        
        Args:
            yaml_path: Path to YAML configuration file.
            
        Returns:
            Validated VPSConfig instance.
        """
        path = Path(yaml_path)
        if not path.is_file():
            raise FileNotFoundError(f"Configuration file not found: {path}")

        with open(path, "r", encoding="utf-8") as f:
            data: Dict[str, Any] = yaml.safe_load(f) or {}

        camera_data = data.get("camera", {})
        tracker_data = data.get("tracker", {})
        if "lk_win_size" in tracker_data and isinstance(tracker_data["lk_win_size"], list):
            tracker_data["lk_win_size"] = tuple(tracker_data["lk_win_size"])
        mavlink_data = data.get("mavlink", {})
        fusion_data = data.get("fusion", {})
        logger_data = data.get("logger", {})
        map_data = data.get("map", {})

        return cls(
            camera=CameraConfig(**camera_data),
            tracker=TrackerConfig(**tracker_data),
            mavlink=MAVLinkConfig(**mavlink_data),
            fusion=FusionConfig(**fusion_data),
            logger=LoggerConfig(**logger_data),
            map=MapConfig(**map_data),
        )

    def to_yaml(self, yaml_path: str | Path) -> None:
        """Save configuration to a YAML file."""
        path = Path(yaml_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "camera": self.camera.__dict__,
            "tracker": {
                **self.tracker.__dict__,
                "lk_win_size": list(self.tracker.lk_win_size)
            },
            "mavlink": self.mavlink.__dict__,
            "fusion": self.fusion.__dict__,
            "logger": self.logger.__dict__,
            "map": self.map.__dict__,
        }
        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)


def load_config(yaml_path: str | Path = Path("config/default_config.yaml")) -> VPSConfig:
    """Helper function to load VPS configuration from YAML file."""
    return VPSConfig.from_yaml(yaml_path)


