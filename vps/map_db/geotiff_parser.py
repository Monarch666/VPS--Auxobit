"""GeoTIFF Parser and Coordinate Projection Engine.

Parses georeferenced GeoTIFF orthomosaic files (e.g. Sentinel-2 L2A or drone orthophotos)
without requiring GDAL/rasterio by extracting standard TIFF tags via tifffile/PIL:
- Tag 33550 (ModelPixelScaleTag): [scale_x, scale_y, scale_z] in degrees or meters
- Tag 33922 (ModelTiepointTag): [i, j, k, x, y, z] tiepoint mapping
- Tag 34735 (GeoKeyDirectoryTag): projection keys (e.g. EPSG:4326 WGS 84)

Converts WGS 84 (Latitude, Longitude) to Local Cartesian NED (North, East, Down)
relative to a mission home / EKF origin (lat0, lon0).
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any
import numpy as np
import tifffile
import cv2


@dataclass
class GeoReference:
    """Georeferencing parameters for a raster map."""
    # Top-left tiepoint in world coordinates (lon, lat for EPSG:4326)
    origin_lon_deg: float
    origin_lat_deg: float
    # Pixel scale in degrees per pixel
    dlon_deg_per_px: float
    dlat_deg_per_px: float
    # Coordinate Reference System
    crs_epsg: int = 4326  # default WGS 84
    # Image dimensions
    width_px: int = 0
    height_px: int = 0

    @property
    def bottom_right_lon_deg(self) -> float:
        """Maximum longitude in degrees."""
        return self.origin_lon_deg + (self.width_px * self.dlon_deg_per_px)

    @property
    def bottom_right_lat_deg(self) -> float:
        """Minimum latitude in degrees (since raster Y points south)."""
        return self.origin_lat_deg - (self.height_px * self.dlat_deg_per_px)

    def pixel_to_lon_lat(self, px: float, py: float) -> Tuple[float, float]:
        """Convert pixel coordinates (u, v) to geographic (longitude, latitude)."""
        lon = self.origin_lon_deg + (px * self.dlon_deg_per_px)
        lat = self.origin_lat_deg - (py * self.dlat_deg_per_px)
        return float(lon), float(lat)

    def lon_lat_to_pixel(self, lon_deg: float, lat_deg: float) -> Tuple[float, float]:
        """Convert geographic (longitude, latitude) to pixel coordinates (u, v)."""
        px = (lon_deg - self.origin_lon_deg) / self.dlon_deg_per_px
        py = (self.origin_lat_deg - lat_deg) / self.dlat_deg_per_px
        return float(px), float(py)


@dataclass
class GeoTIFFMetadata:
    """Comprehensive GeoTIFF raster metadata."""
    file_path: Path
    width_px: int
    height_px: int
    num_channels: int
    dtype: str
    georef: GeoReference
    # Approximate metric ground resolution
    gsd_meters_per_px: float
    total_area_km2: float
    # Bounds in WGS 84
    lat_min_deg: float
    lat_max_deg: float
    lon_min_deg: float
    lon_max_deg: float


class GeoTIFFParser:
    """Parses GeoTIFF metadata and converts between Geographic and Local NED frames."""

    WGS84_A = 6378137.0  # Earth equatorial radius in meters
    WGS84_F = 1.0 / 298.257223563  # Flatting

    def __init__(
        self,
        home_lat_deg: Optional[float] = None,
        home_lon_deg: Optional[float] = None,
    ) -> None:
        """Initialize GeoTIFF parser with mission home origin.

        Args:
            home_lat_deg: Reference home latitude (degrees) for Local NED frame.
            home_lon_deg: Reference home longitude (degrees) for Local NED frame.
        """
        self.home_lat = home_lat_deg
        self.home_lon = home_lon_deg

    def set_home_origin(self, lat_deg: float, lon_deg: float) -> None:
        """Set or update mission home origin."""
        self.home_lat = float(lat_deg)
        self.home_lon = float(lon_deg)

    @classmethod
    def extract_metadata(cls, file_path: str | Path) -> GeoTIFFMetadata:
        """Extract spatial georeferencing metadata from a GeoTIFF file.

        Args:
            file_path: Path to GeoTIFF file.

        Returns:
            GeoTIFFMetadata instance.
        """
        path = Path(file_path)
        if not path.is_file():
            raise FileNotFoundError(f"GeoTIFF file not found: {path}")

        with tifffile.TiffFile(str(path)) as tif:
            page = tif.pages[0]
            shape = page.shape
            dtype_str = str(page.dtype)

            if len(shape) == 2:
                height, width = shape
                channels = 1
            else:
                height, width, channels = shape[0], shape[1], shape[2]

            # Standard GeoTIFF tags
            # Tag 33550: ModelPixelScaleTag
            pixel_scale = None
            # Tag 33922: ModelTiepointTag
            tiepoint = None
            # Tag 34735: GeoKeyDirectoryTag
            geokey = None

            for tag in page.tags:
                code = tag.code
                if code == 33550:
                    pixel_scale = tag.value
                elif code == 33922:
                    tiepoint = tag.value
                elif code == 34735:
                    geokey = tag.value

            if pixel_scale is None or tiepoint is None:
                raise ValueError(f"File {path.name} is missing GeoTIFF spatial tags (33550 / 33922).")

            dlon = float(pixel_scale[0])
            dlat = float(pixel_scale[1])

            # ModelTiepointTag format: [i, j, k, x, y, z]
            # (i, j) is pixel, (x, y) is geographic coordinate (lon, lat)
            origin_lon = float(tiepoint[3])
            origin_lat = float(tiepoint[4])

            # Parse EPSG code if available in GeoKeyDirectoryTag
            epsg = 4326  # default WGS 84
            if geokey is not None and len(geokey) > 0:
                # Key 2048 is GeographicTypeGeoKey (e.g. 4326)
                # Key 3072 is ProjectedCSTypeGeoKey
                for k_idx in range(4, len(geokey) - 3, 4):
                    k_id = geokey[k_idx]
                    k_val = geokey[k_idx + 3]
                    if k_id == 2048 and k_val > 0:
                        epsg = int(k_val)
                    elif k_id == 3072 and k_val > 0:
                        epsg = int(k_val)

        georef = GeoReference(
            origin_lon_deg=origin_lon,
            origin_lat_deg=origin_lat,
            dlon_deg_per_px=dlon,
            dlat_deg_per_px=dlat,
            crs_epsg=epsg,
            width_px=width,
            height_px=height,
        )

        lat_min = georef.bottom_right_lat_deg
        lat_max = origin_lat
        lon_min = origin_lon
        lon_max = georef.bottom_right_lon_deg

        # Approximate metric GSD at center latitude
        center_lat_rad = np.radians(0.5 * (lat_min + lat_max))
        m_per_deg_lat = 111132.954  # meters per degree latitude
        m_per_deg_lon = 111412.84 * np.cos(center_lat_rad)

        gsd_y = dlat * m_per_deg_lat
        gsd_x = dlon * m_per_deg_lon
        gsd_m = float(0.5 * (gsd_x + gsd_y))

        total_width_km = (width * gsd_x) / 1000.0
        total_height_km = (height * gsd_y) / 1000.0
        area_km2 = float(total_width_km * total_height_km)

        return GeoTIFFMetadata(
            file_path=path,
            width_px=width,
            height_px=height,
            num_channels=channels,
            dtype=dtype_str,
            georef=georef,
            gsd_meters_per_px=gsd_m,
            total_area_km2=area_km2,
            lat_min_deg=lat_min,
            lat_max_deg=lat_max,
            lon_min_deg=lon_min,
            lon_max_deg=lon_max,
        )

    def wgs84_to_local_ned(self, lat_deg: float, lon_deg: float) -> Tuple[float, float]:
        """Convert WGS 84 geographic coordinates to metric Local NED relative to home.

        Args:
            lat_deg: Target latitude in degrees.
            lon_deg: Target longitude in degrees.

        Returns:
            Tuple of (North_meters, East_meters).
        """
        if self.home_lat is None or self.home_lon is None:
            raise ValueError("Home origin (home_lat, home_lon) must be set before NED conversion.")

        lat_rad = np.radians(lat_deg)
        lon_rad = np.radians(lon_deg)
        lat0_rad = np.radians(self.home_lat)
        lon0_rad = np.radians(self.home_lon)

        dlat = lat_rad - lat0_rad
        dlon = lon_rad - lon0_rad

        # Standard flat-earth tangent projection for local aviation areas (< 100 km)
        north_m = dlat * self.WGS84_A
        east_m = dlon * self.WGS84_A * np.cos(lat0_rad)

        return float(north_m), float(east_m)

    def local_ned_to_wgs84(self, north_m: float, east_m: float) -> Tuple[float, float]:
        """Convert metric Local NED coordinates relative to home into WGS 84 (lat, lon).

        Args:
            north_m: North coordinate in meters.
            east_m: East coordinate in meters.

        Returns:
            Tuple of (lat_degrees, lon_degrees).
        """
        if self.home_lat is None or self.home_lon is None:
            raise ValueError("Home origin (home_lat, home_lon) must be set before WGS84 conversion.")

        lat0_rad = np.radians(self.home_lat)
        lon0_rad = np.radians(self.home_lon)

        dlat = north_m / self.WGS84_A
        dlon = east_m / (self.WGS84_A * np.cos(lat0_rad))

        lat_deg = np.degrees(lat0_rad + dlat)
        lon_deg = np.degrees(lon0_rad + dlon)

        return float(lat_deg), float(lon_deg)

    def pixel_to_local_ned(self, georef: GeoReference, px: float, py: float) -> Tuple[float, float]:
        """Directly map pixel coordinate (u, v) on GeoTIFF to metric Local NED (North, East)."""
        lon, lat = georef.pixel_to_lon_lat(px, py)
        return self.wgs84_to_local_ned(lat, lon)

    def local_ned_to_pixel(self, georef: GeoReference, north_m: float, east_m: float) -> Tuple[float, float]:
        """Directly map metric Local NED (North, East) to pixel coordinate (u, v) on GeoTIFF."""
        lat, lon = self.local_ned_to_wgs84(north_m, east_m)
        return georef.lon_lat_to_pixel(lon, lat)

    @classmethod
    def load_raster_as_bgr_uint8(cls, file_path: str | Path) -> np.ndarray:
        """Load GeoTIFF raster as standard 8-bit BGR image for OpenCV processing.

        Handles float32 [0.0, 1.0] (Sentinel-2 L2A), uint16, and uint8.
        """
        path = Path(file_path)
        data = tifffile.imread(str(path))

        if data.dtype == np.float32 or data.dtype == np.float64:
            # Check range
            dmin = np.nanmin(data)
            dmax = np.nanmax(data)
            if dmax <= 1.05:
                # Normalized [0, 1] reflectance
                img_uint8 = np.clip(data * 255.0, 0, 255).astype(np.uint8)
            else:
                img_uint8 = np.clip((data - dmin) / max(1e-6, dmax - dmin) * 255.0, 0, 255).astype(np.uint8)
        elif data.dtype == np.uint16:
            img_uint8 = (data / 256.0).astype(np.uint8)
        elif data.dtype == np.uint8:
            img_uint8 = data
        else:
            img_uint8 = data.astype(np.uint8)

        # Ensure 3-channel BGR for OpenCV
        if img_uint8.ndim == 2:
            bgr = cv2.cvtColor(img_uint8, cv2.COLOR_GRAY2BGR)
        elif img_uint8.ndim == 3 and img_uint8.shape[2] == 3:
            # Tifffile loads RGB, OpenCV expects BGR
            bgr = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2BGR)
        elif img_uint8.ndim == 3 and img_uint8.shape[2] >= 4:
            # RGBA or multi-band -> extract RGB and convert to BGR
            bgr = cv2.cvtColor(img_uint8[:, :, :3], cv2.COLOR_RGB2BGR)
        else:
            raise ValueError(f"Unsupported raster image shape: {img_uint8.shape}")

        return bgr

