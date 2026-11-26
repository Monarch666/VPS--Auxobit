"""CLI Tool: Fetch High-Resolution Satellite GeoTIFF from Esri World Imagery and Build Phase 2 Map DB.

Downloads public Web Mercator XYZ satellite tiles centered on a flight site,
resamples them onto a uniform WGS 84 (EPSG:4326) raster grid, writes a standard
GeoTIFF compatible with GeoTIFFParser, and compiles the tiled Phase 2 Map DB.

Usage:
    python -m vps.tools.fetch_highres_map --lat 19.80955 --lon 75.28280 --size-km 1.0 --zoom 18
"""

import argparse
import concurrent.futures
import io
import math
import time
import urllib.request
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
import tifffile
from PIL import Image

from vps.map_db.geotiff_parser import GeoTIFFParser
from vps.map_db.runtime_lookup import RuntimeMapDB
from vps.map_db.tiler import MapTiler

ESRI_WORLD_IMAGERY_URL = (
    "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
)
TILE_SIZE_PX = 256


def latlon_to_tile_float(lat_deg: float, lon_deg: float, zoom: int) -> Tuple[float, float]:
    """Convert WGS 84 (lat, lon) to continuous Web Mercator tile coordinates (x, y)."""
    lat_rad = math.radians(lat_deg)
    n = 2.0 ** zoom
    x = (lon_deg + 180.0) / 360.0 * n
    y = (1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n
    return x, y


def tile_to_latlon(x_tile: float, y_tile: float, zoom: int) -> Tuple[float, float]:
    """Convert Web Mercator tile coordinate (x, y) to WGS 84 (lat, lon)."""
    n = 2.0 ** zoom
    lon_deg = x_tile / n * 360.0 - 180.0
    lat_rad = math.atan(math.sinh(math.pi * (1.0 - 2.0 * y_tile / n)))
    return math.degrees(lat_rad), lon_deg


def _download_single_tile(args: Tuple[int, int, int, str]) -> Tuple[int, int, np.ndarray]:
    """Download one 256x256 XYZ satellite tile as an RGB uint8 array."""
    z, x, y, url_template = args
    url = url_template.format(z=z, x=x, y=y)
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (VPS MapBuilder)"})
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                img = Image.open(io.BytesIO(resp.read())).convert("RGB")
                return x, y, np.asarray(img, dtype=np.uint8)
        except Exception:
            if attempt == 2:
                raise
            time.sleep(0.4 * (attempt + 1))
    raise RuntimeError(f"Failed downloading tile ({z}/{x}/{y})")


def fetch_geotiff_from_esri(
    center_lat: float,
    center_lon: float,
    size_km: float = 1.0,
    zoom: int = 18,
    output_geotiff: Path = Path("data/raw_satellite/aurangabad_esri_z18.tif"),
) -> Path:
    """Download satellite tiles covering size_km x size_km and save a WGS 84 GeoTIFF."""
    half_m = (size_km * 1000.0) / 2.0
    dlat = half_m / 111132.954
    dlon = half_m / (111412.84 * math.cos(math.radians(center_lat)))

    lat_max, lat_min = center_lat + dlat, center_lat - dlat
    lon_min, lon_max = center_lon - dlon, center_lon + dlon

    x_min_f, y_min_f = latlon_to_tile_float(lat_max, lon_min, zoom)
    x_max_f, y_max_f = latlon_to_tile_float(lat_min, lon_max, zoom)

    x_start, x_end = int(math.floor(x_min_f)), int(math.floor(x_max_f))
    y_start, y_end = int(math.floor(y_min_f)), int(math.floor(y_max_f))

    nx = x_end - x_start + 1
    ny = y_end - y_start + 1
    total_tiles = nx * ny
    print(f"Fetching {nx}x{ny} = {total_tiles} tiles at Zoom {zoom} around ({center_lat:.5f}, {center_lon:.5f})...")

    tasks = [
        (zoom, x, y, ESRI_WORLD_IMAGERY_URL)
        for y in range(y_start, y_end + 1)
        for x in range(x_start, x_end + 1)
    ]

    merc_rgb = np.zeros((ny * TILE_SIZE_PX, nx * TILE_SIZE_PX, 3), dtype=np.uint8)
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        for x, y, tile_rgb in pool.map(_download_single_tile, tasks):
            px = (x - x_start) * TILE_SIZE_PX
            py = (y - y_start) * TILE_SIZE_PX
            merc_rgb[py : py + TILE_SIZE_PX, px : px + TILE_SIZE_PX] = tile_rgb

    # Compute geographic extent of the full stitched tile block
    top_lat, left_lon = tile_to_latlon(x_start, y_start, zoom)
    bot_lat, right_lon = tile_to_latlon(x_end + 1, y_end + 1, zoom)

    # Resample Web Mercator vertical axis onto a uniform WGS 84 (EPSG:4326) latitude grid
    out_h, out_w = merc_rgb.shape[:2]
    dlat_px = (top_lat - bot_lat) / out_h
    dlon_px = (right_lon - left_lon) / out_w

    rows = np.arange(out_h, dtype=np.float64)
    target_lats = top_lat - rows * dlat_px
    n_zoom = 2.0 ** zoom
    merc_y_tile = (1.0 - np.arcsinh(np.tan(np.radians(target_lats))) / math.pi) / 2.0 * n_zoom
    src_y = ((merc_y_tile - y_start) * TILE_SIZE_PX).astype(np.float32)

    map_x = np.broadcast_to(np.arange(out_w, dtype=np.float32)[None, :], (out_h, out_w))
    map_y = np.broadcast_to(src_y[:, None], (out_h, out_w))
    wgs84_rgb = cv2.remap(merc_rgb, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

    output_geotiff.parent.mkdir(parents=True, exist_ok=True)
    # Standard GeoTIFF spatial tags read by GeoTIFFParser
    extratags = [
        (33550, "d", 3, (dlon_px, dlat_px, 0.0), True),
        (33922, "d", 6, (0.0, 0.0, 0.0, left_lon, top_lat, 0.0), True),
        (34735, "H", 16, (1, 1, 0, 3, 1024, 0, 1, 2, 1025, 0, 1, 1, 2048, 0, 1, 4326), True),
    ]
    tifffile.imwrite(str(output_geotiff), wgs84_rgb, photometric="rgb", extratags=extratags)
    print(f"Saved WGS 84 GeoTIFF: {output_geotiff} ({out_w}x{out_h} px)")
    return output_geotiff


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch high-res Esri imagery GeoTIFF and build Phase 2 Map DB.")
    parser.add_argument("--lat", type=float, default=19.80955, help="Center latitude in degrees.")
    parser.add_argument("--lon", type=float, default=75.28280, help="Center longitude in degrees.")
    parser.add_argument("--size-km", type=float, default=1.0, help="Square region width/height in km (default 1.0).")
    parser.add_argument("--zoom", type=int, default=18, help="XYZ tile zoom level (18 ~ 0.56 m/px, 19 ~ 0.28 m/px).")
    parser.add_argument(
        "--geotiff-out",
        type=str,
        default="data/raw_satellite/aurangabad_esri_z18_1km.tif",
        help="Output path for generated GeoTIFF.",
    )
    parser.add_argument(
        "--db-out",
        type=str,
        default="data/map_db/aurangabad_highres",
        help="Output directory for tiled Phase 2 Map DB.",
    )
    parser.add_argument("--tile-size", type=int, default=512, help="Map DB tile size in pixels.")
    parser.add_argument("--overlap", type=float, default=0.15, help="Map DB tile overlap ratio.")
    args = parser.parse_args()

    geotiff_path = fetch_geotiff_from_esri(
        center_lat=args.lat,
        center_lon=args.lon,
        size_km=args.size_km,
        zoom=args.zoom,
        output_geotiff=Path(args.geotiff_out),
    )

    meta = GeoTIFFParser.extract_metadata(geotiff_path)
    print(f"Verified GeoTIFF Metadata: {meta.width_px}x{meta.height_px} px, GSD={meta.gsd_meters_per_px:.3f} m/px, Area={meta.total_area_km2:.3f} km^2")

    tiler = MapTiler(tile_size_px=args.tile_size, overlap_ratio=args.overlap)
    manifest = tiler.build_database(
        geotiff_path=geotiff_path,
        output_dir=args.db_out,
        map_name=f"aurangabad_esri_z{args.zoom}",
        home_origin_deg=(args.lat, args.lon),
    )

    db = RuntimeMapDB(db_directory=args.db_out, max_cached_tiles=9)
    tiles = db.query_tiles(pos_ned_m=np.array([0.0, 0.0]), search_radius_m=150.0)
    print(f"RuntimeMapDB check: loaded {manifest.num_tiles} total tiles; found {len(tiles)} tiles within 150m of origin.")
    if tiles:
        t0 = tiles[0]
        print(
            f"  Center tile '{t0.metadata.tile_id}': {len(t0.kp_orb)} ORB features, "
            f"{len(t0.kp_akaze)} AKAZE features, GSD={t0.metadata.gsd_m_per_px:.3f} m/px"
        )


if __name__ == "__main__":
    main()
