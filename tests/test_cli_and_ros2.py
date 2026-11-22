"""Unit tests for standalone CLI daemon and optional ROS 2 wrapper."""

from pathlib import Path
import pytest

from vps.config import VPSConfig
from vps.main import VPSApplication, parse_arguments
from vps.ros2 import VPSRos2Node, ROS2_AVAILABLE


class TestStandaloneCLI:
    def test_cli_argument_parsing(self, monkeypatch):
        """Verify command line flags parse correctly."""
        test_args = [
            "vps/main.py",
            "--config", "config/default_config.yaml",
            "--connection", "udpin:127.0.0.1:14550",
            "--camera", "mock",
            "--mode", "sitl",
            "--baseline", "0.18",
        ]
        monkeypatch.setattr("sys.argv", test_args)
        args = parse_arguments()

        assert args.config == "config/default_config.yaml"
        assert args.connection == "udpin:127.0.0.1:14550"
        assert args.camera == "mock"
        assert args.mode == "sitl"
        assert args.baseline == 0.18

    def test_vps_application_lifecycle(self):
        """Verify startup and shutdown cycle of VPSApplication in mock mode."""
        cfg = VPSConfig()
        cfg.camera.device_index = "mock"
        cfg.mavlink.connection_string = "udpin:127.0.0.1:14588"

        app = VPSApplication(config=cfg)
        assert app.start()
        assert app._running
        app.stop()
        assert not app._running


class TestROS2Wrapper:
    def test_ros2_import_safety(self):
        """Verify ROS 2 module can be imported without throwing uncaught import errors."""
        import vps.ros2.vps_node as r2
        assert hasattr(r2, "ROS2_AVAILABLE")
        assert hasattr(r2, "VPSRos2Node")

    def test_ros2_runtime_check(self):
        """Verify node raises informative RuntimeError if initialized without ROS 2."""
        if not ROS2_AVAILABLE:
            with pytest.raises(RuntimeError, match="ROS 2 \\(rclpy\\) is not installed"):
                VPSRos2Node()
