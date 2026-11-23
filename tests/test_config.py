"""Unit tests for configuration loading and validation."""

import pytest
from pathlib import Path
from vps.config import VPSConfig, CameraConfig, TrackerConfig, MAVLinkConfig, FusionConfig


def test_default_config_loading():
    config_path = Path("config/default_config.yaml")
    assert config_path.is_file()
    cfg = VPSConfig.from_yaml(config_path)

    assert cfg.camera.name == "Sony_IMX296"
    assert cfg.camera.width == 1456
    assert cfg.camera.height == 1088
    assert cfg.camera.fps == 30
    assert cfg.camera.fx == 1207.2
    assert cfg.camera.fy == 1199.1
    assert cfg.camera.K.shape == (3, 3)
    assert len(cfg.camera.D) == 5
    assert cfg.camera.baseline_ground_offset_m == 0.06

    assert cfg.tracker.max_features == 250
    assert cfg.tracker.min_keyframe_displacement_px == 10.0
    assert cfg.tracker.lk_win_size == (21, 21)

    assert cfg.mavlink.stream_rate_hz == 100
    assert cfg.mavlink.distance_sensor_rate_hz == 15
    assert cfg.mavlink.sensor_orientation == "PITCH_270"

    assert cfg.fusion.min_confidence == 0.3


def test_config_roundtrip(tmp_path):
    out_yaml = tmp_path / "custom_config.yaml"
    cfg = VPSConfig()
    cfg.camera.fx = 1250.0
    cfg.tracker.max_features = 300
    cfg.to_yaml(out_yaml)

    loaded = VPSConfig.from_yaml(out_yaml)
    assert loaded.camera.fx == 1250.0
    assert loaded.tracker.max_features == 300
    assert loaded.tracker.lk_win_size == (21, 21)

