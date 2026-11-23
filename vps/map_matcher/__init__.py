"""Map Matching and Visual Relocalization module for VPS.

Provides:
- Frame orthorectification (de-tilt, North-alignment, scale-to-map-GSD)
- Multi-scale ORB/SIFT feature matching with RANSAC homography
- Phase correlation fallback on normalized Sobel gradient orientation maps
- Delayed-state ES-EKF fusion with chi-squared innovation gating
- Loss-of-fix relocalization mode
"""

from .orthorectifier import FrameOrthorectifier, OrthoResult
from .matcher import MapMatcher, MapMatchResult

__all__ = [
    "FrameOrthorectifier",
    "OrthoResult",
    "MapMatcher",
    "MapMatchResult",
]

