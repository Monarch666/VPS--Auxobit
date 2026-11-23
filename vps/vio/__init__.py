"""Visual-Inertial Odometry (VIO) module for VPS.

Provides:
- Metric velocity recovery from de-rotated optical flow and AGL height
- Planar homography decomposition alternative
- Relative frame-to-frame displacement tracking
"""

from .velocity_estimator import (
    VIOVelocityEstimator,
    VIOVelocityResult,
    HomographyDecompositionResult,
)

__all__ = [
    "VIOVelocityEstimator",
    "VIOVelocityResult",
    "HomographyDecompositionResult",
]

