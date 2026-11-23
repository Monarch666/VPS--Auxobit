"""CLI Tool: Build Multi-Scale Map Database from User GeoTIFF for Raspberry Pi 5.

Usage:
    python -m vps.tools.build_map_db --input <path_to_geotiff> --output data/map_db/aurangabad_s2
"""

import argparse
from pathlib import Path
from typing import Optional, Tuple
import numpy as np

from vps.map_db.geotiff_parser import GeoTIFFParser
from vps.map_db.tiler import MapTiler
from vps.map_db.runtime_lookup import RuntimeMapDB


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Multi-Scale Map Database from GeoTIFF for Raspberry Pi 5.")
    parser.add_argument(
        "--input",
        type=str,
        default="2026-11-14-00_00_2026-11-14-23_59_Sentinel-2_L2A_True_color.tiff",
        help="Path to input georeferenced GeoTIFF file.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/map_db/aurangabad_s2",
        help="Destination directory for tiled map database.",
    )
    parser.add_argument(
        "--map-name",
        type=str,
        default="aurangabad_sentinel2",
        help="Identifier name for this map database.",
    )
    parser.add_argument(
        "--tile-size",
        type=int,
        default=512,
        help="Tile dimension in pixels (default 512).",
    )
    parser.add_argument(
        "--overlap",
        type=float,
        default=0.15,
        help="Tile overlap fraction (default 0.15 = 15%%).",
    )
    parser.add_argument(
        "--origin",
        type=str,
        default=None,
        help="Optional home origin 'lat,lon' in degrees (e.g. '19.876,75.343').",
    )
    parser.add_argument(
        "--crop-km",
        type=float,
        default=None,
        help="Optional trial flight square crop size in km (e.g. 5.0 for 5x5 km area).",
    )

    args = parser.parse_args()

    input_file = Path(args.input)
    if not input_file.is_file():
        # Try finding in root or data
        alt_file = Path("data") / input_file.name
        if alt_file.is_file():
            input_file = alt_file
        else:
            raise FileNotFoundError(f"GeoTIFF file not found at: {input_file}")

    print("=" * 72)
    print("VPS OFFLINE MAP DATABASE BUILDER")
    print(f"Source GeoTIFF: {input_file}")
    print(f"Target Output:  {args.output}")
    print("=" * 72)

    # Parse origin if supplied
    home_origin: Optional[Tuple[float, float]] = None
    if args.origin is not None:
        parts = [float(p.strip()) for p in args.origin.split(",")]
        home_origin = (parts[0], parts[1])

    # Parse crop if supplied
    crop_bounds_wgs84 = None
    if args.crop_km is not None and args.crop_km > 0:
        meta = GeoTIFFParser.extract_metadata(input_file)
        center_lat = home_origin[0] if home_origin else 0.5 * (meta.lat_min_deg + meta.lat_max_deg)
        center_lon = home_origin[1] if home_origin else 0.5 * (meta.lon_min_deg + meta.lon_max_deg)

        half_side_m = (args.crop_km * 1000.0) / 2.0
        dlat_deg = half_side_m / 111132.954
        dlon_deg = half_side_m / (111412.84 * np.cos(np.radians(center_lat)))

        crop_bounds_wgs84 = (
            center_lat - dlat_deg,
            center_lon - dlon_deg,
            center_lat + dlat_deg,
            center_lon + dlon_deg,
        )
        print(f"Applying {args.crop_km:.1f}x{args.crop_km:.1f} km Trial Flight Crop Box around ({center_lat:.4f}, {center_lon:.4f})")

    tiler = MapTiler(
        tile_size_px=args.tile_size,
        overlap_ratio=args.overlap,
    )

    manifest = tiler.build_database(
        geotiff_path=input_file,
        output_dir=args.output,
        map_name=args.map_name,
        home_origin_deg=home_origin,
        crop_bounds_wgs84=crop_bounds_wgs84,
    )

    print("\nVerifying Runtime Spatial Lookup Performance on Companion Computer...")
    db = RuntimeMapDB(db_directory=args.output, max_cached_tiles=9)

    # Perform sample spatial queries
    sample_queries = [
        np.array([0.0, 0.0]),
        np.array([500.0, -300.0]),
        np.array([-800.0, 600.0]),
    ]

    for q in sample_queries:
        res = db.query_tiles(pos_ned_m=q, search_radius_m=300.0)
        tile_ids = [t.metadata.tile_id for t in res]
        print(f"Query pos=[{q[0]:+7.1f}, {q[1]:+7.1f}] -> {len(res)} tiles found: {tile_ids}")

    print(f"\nMean Spatial Query Latency: {db.mean_query_time_ms:.3f} ms (Target < 2.0 ms on Pi 5)")
    print(f"Cache Hit Rate: {db.cache_hit_rate * 100.0:.1f}%")
    print(f"Map Database '{args.map_name}' is fully ready for Milestone 9 (Delayed-State Map Matcher).")


if __name__ == "__main__":
    main()

