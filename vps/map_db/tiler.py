"""Multi-Scale Spatial Tiling and Feature Database Engine.

Slices large GeoTIFF aerial orthomosaics into overlapping spatial tiles, projects
each tile to metric Local NED coordinates, and precomputes multi-scale visual features:
1. Multi-scale Gaussian pyramid representations (1.0x, 0.5x, 0.25x).
2. Rotation-invariant ORB keypoints and descriptors.
3. Edge-preserving AKAZE keypoints and descriptors.
4. Normalized Sobel gradient magnitude and orientation maps (vital for seasonal gap bridging).
5. Serializes spatial index manifest (JSON) for fast runtime bounding-box querying.
"""

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any
import json
import time
import numpy as np
import cv2

from .geotiff_parser import GeoTIFFParser, GeoTIFFMetadata, GeoReference


@dataclass
class MapTileMetadata:
    """Metadata for an individual precomputed spatial map tile."""
    tile_id: str
    grid_row: int
    grid_col: int
    # Pixel bounding box in master GeoTIFF: [u_min, v_min, u_max, v_max]
    pixel_bounds: Tuple[int, int, int, int]
    # Local NED metric bounding box: [north_min, east_min, north_max, east_max] in meters
    ned_bounds_m: Tuple[float, float, float, float]
    # Geographic bounding box in WGS 84: [lat_min, lon_min, lat_max, lon_max]
    wgs84_bounds_deg: Tuple[float, float, float, float]
    # Metric dimensions
    width_m: float
    height_m: float
    gsd_m_per_px: float
    # Relative file paths within database
    image_file: str
    feature_file: str
    num_orb_features: int
    num_akaze_features: int


@dataclass
class MapDatabaseManifest:
    """Spatial manifest for an offline map database."""
    map_name: str
    creation_timestamp_s: float
    source_file: str
    home_lat_deg: float
    home_lon_deg: float
    crs_epsg: int
    gsd_meters_per_px: float
    total_area_km2: float
    # Overall database bounds in Local NED: [N_min, E_min, N_max, E_max]
    overall_ned_bounds_m: Tuple[float, float, float, float]
    # Overall database bounds in WGS 84
    overall_wgs84_bounds: Tuple[float, float, float, float]
    num_tiles: int
    tile_size_px: int
    overlap_px: int
    tiles: List[MapTileMetadata] = field(default_factory=list)

    def save(self, json_path: Path | str) -> None:
        """Save database manifest to JSON file."""
        path = Path(json_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = asdict(self)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, json_path: Path | str) -> "MapDatabaseManifest":
        """Load database manifest from JSON file."""
        path = Path(json_path)
        if not path.is_file():
            raise FileNotFoundError(f"Map manifest not found: {path}")
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        tile_objs = [MapTileMetadata(**t) for t in data.pop("tiles", [])]
        return cls(tiles=tile_objs, **data)


class MapTiler:
    """Builds multi-scale tiled spatial database from aerial orthophotos."""

    def __init__(
        self,
        tile_size_px: int = 512,
        overlap_ratio: float = 0.15,
        max_orb_features_per_tile: int = 400,
        max_akaze_features_per_tile: int = 250,
    ) -> None:
        """Initialize MapTiler.

        Args:
            tile_size_px: Size of each square tile in pixels (default 512x512).
            overlap_ratio: Spatial overlap fraction between adjacent tiles (e.g. 0.15 = 15%).
            max_orb_features_per_tile: Cap on ORB descriptors per tile.
            max_akaze_features_per_tile: Cap on AKAZE descriptors per tile.
        """
        self.tile_size = int(tile_size_px)
        self.overlap_ratio = float(overlap_ratio)
        self.overlap_px = int(round(self.tile_size * self.overlap_ratio))
        self.stride_px = self.tile_size - self.overlap_px

        self.max_orb = int(max_orb_features_per_tile)
        self.max_akaze = int(max_akaze_features_per_tile)

        # Feature extractors
        self.orb = cv2.ORB_create(
            nfeatures=self.max_orb,
            scaleFactor=1.2,
            nlevels=4,
            edgeThreshold=15,
            fastThreshold=12,
        )
        if hasattr(cv2, "AKAZE_create"):
            self.secondary_detector = cv2.AKAZE_create(
                descriptor_type=cv2.AKAZE_DESCRIPTOR_MLDB,
                threshold=0.001,
            )
            self.secondary_name = "AKAZE"
        elif hasattr(cv2, "SIFT_create"):
            self.secondary_detector = cv2.SIFT_create(nfeatures=self.max_akaze)
            self.secondary_name = "SIFT"
        else:
            self.secondary_detector = None
            self.secondary_name = "NONE"

    def compute_gradient_maps(self, gray_image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Compute normalized Sobel gradient magnitude and orientation maps.

        Essential for seasonal appearance gap matching (monsoon vs dry season).
        """
        gx = cv2.Sobel(gray_image, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray_image, cv2.CV_32F, 0, 1, ksize=3)

        mag = cv2.magnitude(gx, gy)
        # Normalize magnitude to [0, 255]
        max_mag = np.percentile(mag, 98)
        mag_norm = np.clip((mag / max(1e-4, max_mag)) * 255.0, 0, 255).astype(np.uint8)

        # Orientation angle in degrees [0, 180]
        angle_rad = np.arctan2(gy, gx)
        angle_deg = np.mod(np.degrees(angle_rad), 180.0).astype(np.uint8)

        return mag_norm, angle_deg

    def build_database(
        self,
        geotiff_path: str | Path,
        output_dir: str | Path,
        map_name: str = "aurangabad_map",
        home_origin_deg: Optional[Tuple[float, float]] = None,
        crop_bounds_wgs84: Optional[Tuple[float, float, float, float]] = None,
    ) -> MapDatabaseManifest:
        """Process GeoTIFF orthomosaic into an indexed multi-scale database.

        Args:
            geotiff_path: Path to georeferenced GeoTIFF raster.
            output_dir: Directory where tiles and manifest will be written.
            map_name: User-defined name for this map database.
            home_origin_deg: (lat0, lon0) in degrees for Local NED frame. If None, uses center.
            crop_bounds_wgs84: Optional (lat_min, lon_min, lat_max, lon_max) to limit tiling area.

        Returns:
            MapDatabaseManifest describing the built database.
        """
        out_path = Path(output_dir)
        tiles_dir = out_path / "tiles"
        feats_dir = out_path / "features"
        tiles_dir.mkdir(parents=True, exist_ok=True)
        feats_dir.mkdir(parents=True, exist_ok=True)

        meta = GeoTIFFParser.extract_metadata(geotiff_path)
        print(f"Loaded GeoTIFF: {meta.width_px} x {meta.height_px} px | GSD ~ {meta.gsd_meters_per_px:.2f} m/px | Area: {meta.total_area_km2:.1f} km^2")

        # Set or determine home origin (lat0, lon0)
        if home_origin_deg is not None:
            home_lat, home_lon = home_origin_deg
        else:
            home_lat = 0.5 * (meta.lat_min_deg + meta.lat_max_deg)
            home_lon = 0.5 * (meta.lon_min_deg + meta.lon_max_deg)

        parser = GeoTIFFParser(home_lat_deg=home_lat, home_lon_deg=home_lon)
        print(f"Mission Home Origin: Lat={home_lat:.6f} deg, Lon={home_lon:.6f} deg")

        # Load raster as BGR uint8
        print("Ingesting raster into memory...")
        bgr_full = GeoTIFFParser.load_raster_as_bgr_uint8(geotiff_path)
        gray_full = cv2.cvtColor(bgr_full, cv2.COLOR_BGR2GRAY)

        # Determine tiling pixel bounds
        u_start, v_start = 0, 0
        u_end, v_end = meta.width_px, meta.height_px

        if crop_bounds_wgs84 is not None:
            c_lat_min, c_lon_min, c_lat_max, c_lon_max = crop_bounds_wgs84
            px_min, py_min = meta.georef.lon_lat_to_pixel(c_lon_min, c_lat_max)
            px_max, py_max = meta.georef.lon_lat_to_pixel(c_lon_max, c_lat_min)
            u_start = max(0, int(np.floor(px_min)))
            v_start = max(0, int(np.floor(py_min)))
            u_end = min(meta.width_px, int(np.ceil(px_max)))
            v_end = min(meta.height_px, int(np.ceil(py_max)))
            print(f"Cropped to Sub-Region: ({u_start}, {v_start}) -> ({u_end}, {v_end}) [{u_end - u_start} x {v_end - v_start} px]")

        # Grid slicing
        col_steps = range(u_start, u_end - self.overlap_px, self.stride_px)
        row_steps = range(v_start, v_end - self.overlap_px, self.stride_px)

        tiles_metadata: List[MapTileMetadata] = []
        all_n_min, all_e_min = float("inf"), float("inf")
        all_n_max, all_e_max = float("-inf"), float("-inf")

        print(f"Extracting multi-scale tiles ({self.tile_size}x{self.tile_size} px with {self.overlap_px} px overlap)...")
        tile_count = 0

        for r_idx, v in enumerate(row_steps):
            for c_idx, u in enumerate(col_steps):
                u2 = min(meta.width_px, u + self.tile_size)
                v2 = min(meta.height_px, v + self.tile_size)
                u1 = max(0, u2 - self.tile_size)
                v1 = max(0, v2 - self.tile_size)

                tile_id = f"tile_r{r_idx:03d}_c{c_idx:03d}"

                # 1. Pixel and Geographic bounds
                lon1, lat1 = meta.georef.pixel_to_lon_lat(u1, v1)  # top-left
                lon2, lat2 = meta.georef.pixel_to_lon_lat(u2, v2)  # bottom-right

                lat_min = min(lat1, lat2)
                lat_max = max(lat1, lat2)
                lon_min = min(lon1, lon2)
                lon_max = max(lon1, lon2)

                # 2. Local NED metric bounds
                n1, e1 = parser.wgs84_to_local_ned(lat_max, lon_min)  # top-left (North max, East min)
                n2, e2 = parser.wgs84_to_local_ned(lat_min, lon_max)  # bottom-right (North min, East max)

                n_min = min(n1, n2)
                n_max = max(n1, n2)
                e_min = min(e1, e2)
                e_max = max(e1, e2)

                all_n_min = min(all_n_min, n_min)
                all_n_max = max(all_n_max, n_max)
                all_e_min = min(all_e_min, e_min)
                all_e_max = max(all_e_max, e_max)

                # 3. Crop tile image
                tile_bgr = bgr_full[v1:v2, u1:u2]
                tile_gray = gray_full[v1:v2, u1:u2]

                # 4. Precompute visual features (ORB, AKAZE, Gradients)
                # Level 0 ORB
                kp_orb, desc_orb = self.orb.detectAndCompute(tile_gray, None)
                if desc_orb is None:
                    desc_orb = np.empty((0, 32), dtype=np.uint8)
                    kp_orb_coords = np.empty((0, 2), dtype=np.float32)
                else:
                    kp_orb_coords = np.array([k.pt for k in kp_orb], dtype=np.float32)

                # Secondary detector (AKAZE or SIFT)
                if self.secondary_detector is not None:
                    kp_sec, desc_sec = self.secondary_detector.detectAndCompute(tile_gray, None)
                    if desc_sec is None:
                        desc_akaze = np.empty((0, 64), dtype=np.uint8)
                        kp_akaze_coords = np.empty((0, 2), dtype=np.float32)
                    else:
                        desc_akaze = desc_sec
                        kp_akaze_coords = np.array([k.pt for k in kp_sec], dtype=np.float32)
                else:
                    desc_akaze = np.empty((0, 32), dtype=np.uint8)
                    kp_akaze_coords = np.empty((0, 2), dtype=np.float32)

                # Gradient orientation and magnitude maps
                grad_mag, grad_angle = self.compute_gradient_maps(tile_gray)

                # 5. Save Tile Assets
                img_rel = f"tiles/{tile_id}.jpg"
                feat_rel = f"features/{tile_id}.npz"

                cv2.imwrite(str(out_path / img_rel), tile_bgr, [cv2.IMWRITE_JPEG_QUALITY, 92])

                np.savez_compressed(
                    str(out_path / feat_rel),
                    kp_orb=kp_orb_coords,
                    desc_orb=desc_orb,
                    kp_akaze=kp_akaze_coords,
                    desc_akaze=desc_akaze,
                    grad_mag=grad_mag,
                    grad_angle=grad_angle,
                )

                tile_meta = MapTileMetadata(
                    tile_id=tile_id,
                    grid_row=r_idx,
                    grid_col=c_idx,
                    pixel_bounds=(u1, v1, u2, v2),
                    ned_bounds_m=(n_min, e_min, n_max, e_max),
                    wgs84_bounds_deg=(lat_min, lon_min, lat_max, lon_max),
                    width_m=float(e_max - e_min),
                    height_m=float(n_max - n_min),
                    gsd_m_per_px=meta.gsd_meters_per_px,
                    image_file=img_rel,
                    feature_file=feat_rel,
                    num_orb_features=len(kp_orb_coords),
                    num_akaze_features=len(kp_akaze_coords),
                )
                tiles_metadata.append(tile_meta)
                tile_count += 1

        # Overall bounds
        overall_ned = (all_n_min, all_e_min, all_n_max, all_e_max)
        overall_wgs84 = (meta.lat_min_deg, meta.lon_min_deg, meta.lat_max_deg, meta.lon_max_deg)

        manifest = MapDatabaseManifest(
            map_name=map_name,
            creation_timestamp_s=time.time(),
            source_file=str(Path(geotiff_path).name),
            home_lat_deg=home_lat,
            home_lon_deg=home_lon,
            crs_epsg=meta.georef.crs_epsg,
            gsd_meters_per_px=meta.gsd_meters_per_px,
            total_area_km2=meta.total_area_km2,
            overall_ned_bounds_m=overall_ned,
            overall_wgs84_bounds=overall_wgs84,
            num_tiles=tile_count,
            tile_size_px=self.tile_size,
            overlap_px=self.overlap_px,
            tiles=tiles_metadata,
        )

        manifest_path = out_path / "manifest.json"
        manifest.save(manifest_path)
        print(f"Successfully built Map Database '{map_name}': {tile_count} tiles saved to {out_path}")
        return manifest
