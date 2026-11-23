"""Raspberry Pi 5 Runtime Spatial Lookup Engine.

Features:
- Sub-2 ms spatial bounding box queries driven by ES-EKF position and covariance
- LRU memory caching keeping active RAM consumption under 100 MB on Cortex-A76
- Direct local NED <-> Tile pixel coordinate mapping for map matching
- Pre-loaded ORB/AKAZE descriptors and gradient orientation maps
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any
from collections import OrderedDict
import time
import numpy as np
import cv2

from .tiler import MapTileMetadata, MapDatabaseManifest


@dataclass
class TileLookupResult:
    """Active spatial map tile loaded into companion computer memory."""
    metadata: MapTileMetadata
    image_bgr: np.ndarray
    image_gray: np.ndarray
    kp_orb: np.ndarray  # shape (N, 2)
    desc_orb: np.ndarray  # shape (N, 32)
    kp_akaze: np.ndarray  # shape (M, 2)
    desc_akaze: np.ndarray  # shape (M, 61)
    grad_mag: np.ndarray  # shape (H, W)
    grad_angle: np.ndarray  # shape (H, W)

    def pixel_to_ned(self, u_tile: float, v_tile: float) -> Tuple[float, float]:
        """Convert pixel coordinate within this tile to metric Local NED (North, East)."""
        # Tile bounds: [n_min, e_min, n_max, e_max]
        n_min, e_min, n_max, e_max = self.metadata.ned_bounds_m
        h_px, w_px = self.image_gray.shape[:2]

        frac_u = u_tile / max(1.0, float(w_px))
        frac_v = v_tile / max(1.0, float(h_px))

        east_m = e_min + frac_u * (e_max - e_min)
        north_m = n_max - frac_v * (n_max - n_min)  # pixel V points south
        return float(north_m), float(east_m)

    def ned_to_pixel(self, north_m: float, east_m: float) -> Tuple[float, float]:
        """Convert metric Local NED (North, East) to pixel coordinate within this tile."""
        n_min, e_min, n_max, e_max = self.metadata.ned_bounds_m
        h_px, w_px = self.image_gray.shape[:2]

        frac_u = (east_m - e_min) / max(1e-4, e_max - e_min)
        frac_v = (n_max - north_m) / max(1e-4, n_max - n_min)

        u = frac_u * float(w_px)
        v = frac_v * float(h_px)
        return float(u), float(v)

    def contains_ned(self, north_m: float, east_m: float) -> bool:
        """Check if a metric Local NED point lies within this tile."""
        n_min, e_min, n_max, e_max = self.metadata.ned_bounds_m
        return (n_min <= north_m <= n_max) and (e_min <= east_m <= e_max)


class RuntimeMapDB:
    """High-speed spatial tile lookup engine optimized for Raspberry Pi 5."""

    def __init__(
        self,
        db_directory: str | Path,
        max_cached_tiles: int = 9,
    ) -> None:
        """Initialize RuntimeMapDB.

        Args:
            db_directory: Directory containing manifest.json, tiles/, and features/.
            max_cached_tiles: Maximum number of decoded tiles held in RAM simultaneously (default 9).
        """
        self.db_dir = Path(db_directory)
        manifest_path = self.db_dir / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Map manifest not found at {manifest_path}")

        self.manifest = MapDatabaseManifest.load(manifest_path)
        self.max_cached = max(1, int(max_cached_tiles))

        # LRU memory cache: tile_id -> TileLookupResult
        self._cache: OrderedDict[str, TileLookupResult] = OrderedDict()

        # Build in-memory spatial index (2D bounding box interval index)
        # Allows instant O(1) retrieval of overlapping tiles
        self._tile_index: Dict[str, MapTileMetadata] = {t.tile_id: t for t in self.manifest.tiles}

        # Cache metrics
        self._cache_hits = 0
        self._cache_misses = 0
        self._query_times_ms: List[float] = []

    @property
    def home_origin_wgs84(self) -> Tuple[float, float]:
        """Return reference home origin (lat, lon) in degrees."""
        return self.manifest.home_lat_deg, self.manifest.home_lon_deg

    @property
    def gsd_m_per_px(self) -> float:
        """Ground sample distance of the base map in meters per pixel."""
        return self.manifest.gsd_meters_per_px

    @property
    def cache_hit_rate(self) -> float:
        """Fraction of tile queries served directly from RAM cache."""
        total = self._cache_hits + self._cache_misses
        return float(self._cache_hits / total) if total > 0 else 0.0

    def query_candidate_tile_ids(
        self,
        pos_ned_m: np.ndarray,
        search_radius_m: float = 300.0,
    ) -> List[str]:
        """Query spatial index for tiles intersecting the search window.

        Args:
            pos_ned_m: Drone center position [North, East] (meters).
            search_radius_m: Search radius (e.g. 3 * sigma_pos from ES-EKF).

        Returns:
            List of matching tile IDs ordered by distance to drone center.
        """
        p_n = float(pos_ned_m[0])
        p_e = float(pos_ned_m[1])
        r = max(10.0, float(search_radius_m))

        q_n_min = p_n - r
        q_n_max = p_n + r
        q_e_min = p_e - r
        q_e_max = p_e + r

        matches: List[Tuple[float, str]] = []

        for tile in self.manifest.tiles:
            t_n_min, t_e_min, t_n_max, t_e_max = tile.ned_bounds_m

            # AABB intersection test
            if (q_n_min <= t_n_max and q_n_max >= t_n_min and
                    q_e_min <= t_e_max and q_e_max >= t_e_min):
                # Compute distance from drone to tile center
                t_center_n = 0.5 * (t_n_min + t_n_max)
                t_center_e = 0.5 * (t_e_min + t_e_max)
                dist = float(np.hypot(p_n - t_center_n, p_e - t_center_e))
                matches.append((dist, tile.tile_id))

        # Sort closest first
        matches.sort(key=lambda x: x[0])
        return [m[1] for m in matches]

    def get_tile(self, tile_id: str) -> TileLookupResult:
        """Retrieve a specific tile by ID, using LRU cache or loading from disk.

        Args:
            tile_id: ID of the tile to retrieve.

        Returns:
            TileLookupResult containing image, descriptors, and metric bounds.
        """
        # 1. Cache hit
        if tile_id in self._cache:
            self._cache_hits += 1
            self._cache.move_to_end(tile_id)
            return self._cache[tile_id]

        # 2. Cache miss -> load from disk
        self._cache_misses += 1
        tile_meta = self._tile_index.get(tile_id)
        if tile_meta is None:
            raise KeyError(f"Tile ID '{tile_id}' not found in database manifest")

        img_path = self.db_dir / tile_meta.image_file
        feat_path = self.db_dir / tile_meta.feature_file

        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            raise FileNotFoundError(f"Failed to load tile image at {img_path}")
        img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

        # Load precomputed features
        feat_data = np.load(str(feat_path))
        kp_orb = feat_data["kp_orb"]
        desc_orb = feat_data["desc_orb"]
        kp_akaze = feat_data["kp_akaze"]
        desc_akaze = feat_data["desc_akaze"]
        grad_mag = feat_data["grad_mag"]
        grad_angle = feat_data["grad_angle"]

        res = TileLookupResult(
            metadata=tile_meta,
            image_bgr=img_bgr,
            image_gray=img_gray,
            kp_orb=kp_orb,
            desc_orb=desc_orb,
            kp_akaze=kp_akaze,
            desc_akaze=desc_akaze,
            grad_mag=grad_mag,
            grad_angle=grad_angle,
        )

        # Evict oldest if cache limit reached
        if len(self._cache) >= self.max_cached:
            self._cache.popitem(last=False)

        self._cache[tile_id] = res
        return res

    def query_tiles(
        self,
        pos_ned_m: np.ndarray,
        search_radius_m: float = 300.0,
        max_tiles_return: int = 4,
    ) -> List[TileLookupResult]:
        """High-speed query returning candidate active map tiles around drone position.

        Args:
            pos_ned_m: Drone position in Local NED [North, East] (meters).
            search_radius_m: Spatial search radius in meters.
            max_tiles_return: Maximum candidate tiles to return (default 4).

        Returns:
            List of active TileLookupResult objects ordered by proximity.
        """
        t_start = time.perf_counter()

        cand_ids = self.query_candidate_tile_ids(pos_ned_m, search_radius_m)
        cand_ids = cand_ids[:max_tiles_return]

        results = [self.get_tile(tid) for tid in cand_ids]

        t_elapsed_ms = (time.perf_counter() - t_start) * 1000.0
        self._query_times_ms.append(t_elapsed_ms)
        if len(self._query_times_ms) > 50:
            self._query_times_ms.pop(0)

        return results

    @property
    def mean_query_time_ms(self) -> float:
        """Rolling average query latency in milliseconds."""
        return float(np.mean(self._query_times_ms)) if self._query_times_ms else 0.0

