"""Geometry module for VPS.

Includes pinhole camera model, radial-tangential undistortion, gyro-based
inter-frame de-rotation, metric height estimation from optical flow, and
vertical scale expansion.
"""

from .pinhole import PinholeModel
from .derotation import GyroDerotationEngine
from .height_flow import HeightFromFlowEngine, FlowHeightResult
from .scale_expansion import ScaleFromExpansionEngine, ScaleExpansionResult

__all__ = [
    "PinholeModel",
    "GyroDerotationEngine",
    "HeightFromFlowEngine",
    "FlowHeightResult",
    "ScaleFromExpansionEngine",
    "ScaleExpansionResult",
]
