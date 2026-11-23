"""Unit and Integration tests for Milestone 8: GeoTIFF Ingestion, Tiling & Runtime Lookup."""

from pathlib import Path
import numpy as np
import pytest

from vps.map_db.geotiff_parser import GeoTIFFParser, GeoReference
from vps.map_db.tiler import MapTiler, MapDatabaseManifest
from vps.map_db.runtime_lookup import RuntimeMapDB


TIFF_PATH = Path("2026-11-14-00_00_2026-11-14-23_59_Sentinel-2_L2A_True_color.tiff")


class TestGeoTIFFParser:
    @pytest.mark.skipif(not TIFF_PATH.is_file(), reason="Sentinel-2 GeoTIFF not found in root")
    def test_sentinel2_metadata_extraction(self):
        meta = GeoTIFFParser.extract_metadata(TIFF_PATH)

        assert meta.width_px == 2500
        assert meta.height_px == 1615
        assert meta.num_channels == 3
        assert "float32" in meta.dtype

        # Geographic bounds in WGS 84
        assert abs(meta.georef.origin_lon_deg - 75.1118) < 0.01
        assert abs(meta.georef.origin_lat_deg - 19.9014) < 0.01

        # Pixel scale: ~15.5 m/px GSD
        assert 14.0 < meta.gsd_meters_per_px < 17.0
        # Total area: ~900 to 1000 km^2
        assert 800.0 < meta.total_area_km2 < 1100.0

    def test_wgs84_to_local_ned_roundtrip(self):
        home_lat, home_lon = 19.876165, 75.343314
        parser = GeoTIFFParser(home_lat_deg=home_lat, home_lon_deg=home_lon)

        # Home origin should map to (0, 0)
        n0, e0 = parser.wgs84_to_local_ned(home_lat, home_lon)
        assert abs(n0) < 1e-4
        assert abs(e0) < 1e-4

        # Test point 2500 m North, -1800 m East
        target_n, target_e = 2500.0, -1800.0
        lat_pt, lon_pt = parser.local_ned_to_wgs84(target_n, target_e)

        n_back, e_back = parser.wgs84_to_local_ned(lat_pt, lon_pt)
        assert abs(n_back - target_n) < 1e-4
        assert abs(e_back - target_e) < 1e-4

    @pytest.mark.skipif(not TIFF_PATH.is_file(), reason="Sentinel-2 GeoTIFF not found in root")
    def test_raster_conversion_to_bgr_uint8(self):
        bgr = GeoTIFFParser.load_raster_as_bgr_uint8(TIFF_PATH)

        assert bgr.dtype == np.uint8
        assert bgr.shape == (1615, 2500, 3)
        assert bgr.min() >= 0
        assert bgr.max() <= 255
        # Sentinel-2 reflectance in Aurangabad should average between 80 and 160
        assert 80 < np.mean(bgr) < 160


class TestMapTilerAndRuntimeDB:
    def test_synthetic_tiling_and_runtime_query(self, tmp_path):
        db_dir = tmp_path / "test_map_db"

        # Create a synthetic georeferenced GeoTIFF in tmp_path
        synth_tiff = tmp_path / "synth_ortho.tiff"
        import tifffile
        synth_data = np.random.uniform(0.1, 0.9, size=(600, 600, 3)).astype(np.float32)

        # Save with standard GeoTIFF tags (0.0001 deg/px ~ 10m/px, origin lat 20.0, lon 75.0)
        tifffile.imwrite(
            str(synth_tiff),
            synth_data,
            extratags=[
                (33550, "d", 3, (0.0001, 0.0001, 0.0), False),
                (33922, "d", 6, (0.0, 0.0, 0.0, 75.0, 20.0, 0.0), False),
            ],
        )

        tiler = MapTiler(tile_size_px=256, overlap_ratio=0.15, max_orb_features_per_tile=100)
        manifest = tiler.build_database(
            geotiff_path=synth_tiff,
            output_dir=db_dir,
            map_name="synthetic_test_db",
            home_origin_deg=(19.97, 75.03),
        )

        assert manifest.num_tiles > 1
        assert (db_dir / "manifest.json").is_file()
        assert len(list((db_dir / "tiles").glob("*.jpg"))) == manifest.num_tiles
        assert len(list((db_dir / "features").glob("*.npz"))) == manifest.num_tiles

        # Test RuntimeMapDB
        rdb = RuntimeMapDB(db_directory=db_dir, max_cached_tiles=4)
        assert rdb.manifest.num_tiles == manifest.num_tiles

        # Query tiles around origin
        tiles_found = rdb.query_tiles(pos_ned_m=np.array([0.0, 0.0]), search_radius_m=500.0)
        assert len(tiles_found) >= 1

        tile0 = tiles_found[0]
        assert tile0.image_bgr.shape[0] == 256
        assert tile0.image_bgr.shape[1] == 256
        assert len(tile0.kp_orb) <= 100
        assert tile0.desc_orb.shape[1] == 32
        assert tile0.grad_mag.shape == (256, 256)

        # Test pixel-to-NED mapping
        n_mid, e_mid = tile0.pixel_to_ned(128.0, 128.0)
        u_back, v_back = tile0.ned_to_pixel(n_mid, e_mid)
        assert abs(u_back - 128.0) < 0.5
        assert abs(v_back - 128.0) < 0.5

        # Query latency check on cached tiles (< 2.0 ms)
        rdb.query_tiles(pos_ned_m=np.array([100.0, -100.0]), search_radius_m=300.0)
        rdb._query_times_ms.clear()
        for _ in range(10):
            rdb.query_tiles(pos_ned_m=np.array([100.0, -100.0]), search_radius_m=300.0)
        assert rdb.mean_query_time_ms < 2.0
        assert rdb.cache_hit_rate > 0.0
