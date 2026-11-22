"""Optional ROS 2 wrapper package for VPS (Visual Positioning System)."""

from .vps_node import VPSRos2Node, ROS2_AVAILABLE

__all__ = ["VPSRos2Node", "ROS2_AVAILABLE"]
