"""Map Database and Spatial Indexing module for VPS.

Provides:
- GeoTIFF orthomosaic ingestion and WGS 84 -> Local NED coordinate projection
- Multi-scale spatial tiling and feature precomputation (ORB, AKAZE, Gradients)
- High-speed runtime spatial tile lookup with LRU memory caching for Raspberry Pi 5
"""

from .geotiff_parser import GeoTIFFParser, GeoTIFFMetadata, GeoReference
from .tiler import MapTiler, MapTileMetadata, MapDatabaseManifest
from .runtime_lookup import RuntimeMapDB, TileLookupResult

__all__ = [
    "GeoTIFFParser",
    "GeoTIFFMetadata",
    "GeoReference",
    "MapTiler",
    "MapTileMetadata",
    "MapDatabaseManifest",
    "RuntimeMapDB",
    "TileLookupResult",
]

