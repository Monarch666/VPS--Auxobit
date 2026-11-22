"""Procedural aerial terrain generator for Deccan Plateau simulation.

Generates realistic high-resolution orthophoto imagery simulating the Aurangabad
and Deccan plateau landscape:
- Diverse agricultural parcel geometries (black cotton soil, laterite soil, green crops, harvested stubble)
- Scrubland, rocky basalt terrain patches, and dry earth
- Field boundaries, rural dirt paths, paved roads, and irrigation channels
- Built-up farm structures and cluster rooftops
- High-frequency soil/foliage texture synthesis
"""

from pathlib import Path
from typing import Tuple, Optional
import numpy as np
import cv2


class DeccanTerrainGenerator:
    """Generates synthetic aerial orthophoto terrain representing the Deccan plateau."""

    def __init__(
        self,
        width_px: int = 4096,
        height_px: int = 4096,
        meters_per_pixel: float = 0.1,  # 0.1 m/px -> 409.6 m x 409.6 m coverage
        seed: int = 42,
    ) -> None:
        """Initialize terrain generator.

        Args:
            width_px: Image width in pixels.
            height_px: Image height in pixels.
            meters_per_pixel: Metric scale of the generated orthophoto.
            seed: Random seed for deterministic procedural generation.
        """
        self.width = width_px
        self.height = height_px
        self.m_per_px = float(meters_per_pixel)
        self.rng = np.random.default_rng(seed)

    def generate(self, output_path: Optional[Path | str] = None) -> np.ndarray:
        """Generate high-resolution RGB orthophoto.

        Args:
            output_path: Optional path to save the generated image.

        Returns:
            RGB image array of shape (height, width, 3) with uint8 values.
        """
        # Base background: warm Deccan clay/soil (BGR for OpenCV)
        # Base soil tones (BGR): [55, 75, 110] (dark brown laterite)
        img = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        img[:] = [60, 80, 115]

        # 1. Agricultural Fields Partitioning (Voronoi-like field parcel grid)
        n_fields = 60
        field_centers_x = self.rng.integers(100, self.width - 100, size=n_fields)
        field_centers_y = self.rng.integers(100, self.height - 100, size=n_fields)

        # Palette of Deccan agricultural parcels (in BGR)
        field_palettes = [
            [35, 45, 55],     # Dark Deccan black cotton soil (regur)
            [50, 70, 105],    # Brownish dry laterite soil
            [30, 95, 45],     # Lush irrigated sorghum/sugarcane crop
            [45, 110, 75],    # Cotton/soybean green crop
            [70, 130, 160],   # Harvested golden wheat/millet stubble
            [85, 120, 145],   # Dry fallow field
            [40, 50, 60],     # Moist plowed soil
            [55, 85, 65],     # Scattered scrub / pulses crop
        ]

        # Create low-res field index map for fast assignment
        downscale = 8
        w_small = self.width // downscale
        h_small = self.height // downscale

        xs = np.arange(w_small) * downscale
        ys = np.arange(h_small) * downscale
        grid_x, grid_y = np.meshgrid(xs, ys)

        # Compute nearest field center for each downscaled pixel
        coords = np.column_stack([grid_x.ravel(), grid_y.ravel()])
        centers = np.column_stack([field_centers_x, field_centers_y])

        # Distance matrix (downsampled)
        # Process in chunks to maintain memory efficiency
        chunk_size = 50000
        field_ids_flat = np.zeros(coords.shape[0], dtype=np.int32)
        for i in range(0, coords.shape[0], chunk_size):
            chunk = coords[i:i + chunk_size]
            # dist shape: (chunk_len, n_fields)
            dists = np.hypot(chunk[:, 0:1] - centers[:, 0], chunk[:, 1:2] - centers[:, 1])
            field_ids_flat[i:i + chunk_size] = np.argmin(dists, axis=1)

        field_map_small = field_ids_flat.reshape((h_small, w_small))
        field_map = cv2.resize(field_map_small.astype(np.int32), (self.width, self.height), interpolation=cv2.INTER_NEAREST)

        # Assign colors to each field with internal color variations
        field_colors = np.array([
            field_palettes[self.rng.integers(0, len(field_palettes))]
            for _ in range(n_fields)
        ], dtype=np.uint8)

        img = field_colors[field_map]

        # 2. Add Crop Rows / Furrows inside fields
        # Create subtle periodic grating along various orientations
        y_indices, x_indices = np.indices((self.height, self.width), dtype=np.float32)
        freq = 0.25  # furrow frequency
        grating = (np.sin(x_indices * freq + y_indices * freq * 0.5) * 15).astype(np.int16)

        img_int = img.astype(np.int16) + grating[:, :, np.newaxis]
        img = np.clip(img_int, 0, 255).astype(np.uint8)

        # 3. Roads, Canal Network & Boundaries
        # Main paved highway (gray)
        cv2.line(img, (0, int(self.height * 0.35)), (self.width, int(self.height * 0.4)), (120, 120, 120), thickness=int(6.0 / self.m_per_px))
        # Dirt roads (tan/sand)
        cv2.line(img, (int(self.width * 0.4), 0), (int(self.width * 0.45), self.height), (90, 120, 150), thickness=int(3.5 / self.m_per_px))
        cv2.line(img, (0, int(self.height * 0.7)), (self.width, int(self.height * 0.72)), (90, 120, 150), thickness=int(3.0 / self.m_per_px))
        # Meandering irrigation canal / dry stream (nullah)
        pts_canal = []
        for x in range(0, self.width, 100):
            y = int(self.height * 0.15 + 40 * np.sin(x * 0.005) + 20 * np.cos(x * 0.015))
            pts_canal.append([x, y])
        cv2.polylines(img, [np.array(pts_canal, dtype=np.int32)], isClosed=False, color=(70, 95, 75), thickness=int(4.0 / self.m_per_px))

        # 4. Field Boundary Vegetation / Hedgerows (Dark green trees/bushes along borders)
        field_borders = cv2.Canny(field_map.astype(np.uint8), 0, 1)
        # Dilate borders slightly
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        field_borders = cv2.dilate(field_borders, kernel, iterations=1)
        border_mask = field_borders > 0
        img[border_mask] = (img[border_mask].astype(np.int16) * 0.65).astype(np.uint8)

        # Add scattered trees (small dark green circles) along roads and borders
        n_trees = 350
        tree_x = self.rng.integers(50, self.width - 50, size=n_trees)
        tree_y = self.rng.integers(50, self.height - 50, size=n_trees)
        tree_r = self.rng.integers(int(1.5 / self.m_per_px), int(3.5 / self.m_per_px), size=n_trees)
        for tx, ty, tr in zip(tree_x, tree_y, tree_r):
            # Tree foliage (dark green with shadow)
            cv2.circle(img, (tx + int(tr * 0.4), ty + int(tr * 0.4)), tr, (15, 20, 20), -1)  # shadow
            cv2.circle(img, (tx, ty), tr, (25, 65, 30), -1)  # canopy

        # 5. Built-up Settlements / Farm Structures (Roofs: terracotta red, tin gray)
        n_clusters = 5
        for _ in range(n_clusters):
            cx = self.rng.integers(400, self.width - 400)
            cy = self.rng.integers(400, self.height - 400)
            n_buildings = self.rng.integers(4, 10)
            for _ in range(n_buildings):
                bx = cx + self.rng.integers(-80, 80)
                by = cy + self.rng.integers(-80, 80)
                bw = self.rng.integers(int(6.0 / self.m_per_px), int(14.0 / self.m_per_px))
                bh = self.rng.integers(int(5.0 / self.m_per_px), int(10.0 / self.m_per_px))
                # Roof color: terracotta tile (BGR: [30, 45, 140]) or tin (BGR: [160, 160, 170])
                roof_color = (30, 45, 150) if self.rng.random() > 0.4 else (160, 165, 175)
                # Building shadow
                cv2.rectangle(img, (bx + 4, by + 4), (bx + bw + 4, by + bh + 4), (20, 20, 20), -1)
                # Building roof
                cv2.rectangle(img, (bx, by), (bx + bw, by + bh), roof_color, -1)

        # 6. High-Frequency Soil/Grain Texture
        noise = self.rng.normal(0, 7.0, size=(self.height, self.width, 3)).astype(np.int16)
        img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        if output_path is not None:
            path = Path(output_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(path), img)

        return img
