"""Unit tests for the minimalist VTX On-Screen Display (OSD) renderer and output sinks."""

import time
import urllib.request
import numpy as np

from vps.config import VTXConfig, VPSConfig
from vps.tools.vtx_hud import (
    MJPEGStreamSink,
    VTXDisplayServer,
    VTXHudData,
    VTXHudRenderer,
    bgr_to_rgb565_bytes,
)


def test_vtx_hud_keeps_central_fov_uncluttered() -> None:
    """Verify top/bottom OSD strips stay compact and >96% of the center FOV is untouched."""
    cfg = VTXConfig(
        enabled=True,
        output_width=720,
        output_height=540,
        show_features=True,
        max_drawn_features=30,
        show_velocity_vector=True,
    )
    renderer = VTXHudRenderer(cfg)

    # Constant mid-gray 1456x1088 frame (like IMX296 monochrome capture)
    raw = np.full((1088, 1456), 120, dtype=np.uint8)
    pts = np.column_stack([
        np.linspace(100, 1300, 80, dtype=np.float32),
        np.linspace(150, 950, 80, dtype=np.float32),
    ])

    data = VTXHudData(
        height_agl_m=6.72,
        baro_alt_m=6.65,
        confidence=0.88,
        mode_status="HOVER_TERRAIN_HOLD",
        is_armed=True,
        flight_mode="LOITER",
        battery_voltage_v=15.8,
        gps_satellites=12,
        fps=14.2,
        num_inliers=80,
        pos_n_m=12.4,
        pos_e_m=-3.1,
        vel_n_mps=1.2,
        vel_e_mps=-0.4,
        map_status="LOCKED",
        phase1_only=False,
        is_shadow=True,
        tracked_points=pts,
    )

    out = renderer.render(raw, data)
    assert out.shape == (540, 720, 3)
    assert out.dtype == np.uint8

    # Central 80% vertical band must remain >97% identical to original 120 gray
    center_band = out[54:486, :, :]
    unchanged_frac = float(np.mean(np.all(center_band == 120, axis=2)))
    assert unchanged_frac > 0.97, f"Center FOV cluttered: only {unchanged_frac:.3f} untouched"


def test_rgb565_framebuffer_packing() -> None:
    """Verify RGB565 conversion packs 2 bytes per pixel with correct color channels."""
    # 1x3 BGR pixels: Blue, Green, Red
    bgr = np.array([[[255, 0, 0], [0, 255, 0], [0, 0, 255]]], dtype=np.uint8)
    packed = bgr_to_rgb565_bytes(bgr)
    assert len(packed) == 6
    vals = np.frombuffer(packed, dtype="<u2")
    assert vals[0] == 0x001F  # Pure Blue (lower 5 bits)
    assert vals[1] == 0x07E0  # Pure Green (middle 6 bits)
    assert vals[2] == 0xF800  # Pure Red (top 5 bits)


def test_vtx_mjpeg_and_display_server_lifecycle() -> None:
    """Verify non-blocking background server starts, encodes MJPEG frames, and stops cleanly."""
    sink = MJPEGStreamSink(host="127.0.0.1", port=0)
    raw = np.zeros((480, 640, 3), dtype=np.uint8)
    sink.write(raw)
    assert sink._latest_jpeg.startswith(b"\xff\xd8")

    cfg = VTXConfig(
        enabled=True,
        sink="auto",
        output_width=640,
        output_height=480,
        target_fps=30.0,
    )
    server = VTXDisplayServer(cfg)
    assert server.start()

    try:
        server.submit(
            raw,
            VTXHudData(height_agl_m=4.5, confidence=0.75, phase1_only=True, is_shadow=True),
        )
        time.sleep(0.08)
    finally:
        server.stop()


def test_vps_config_loads_vtx_section() -> None:
    """Verify VPSConfig exposes VTXConfig cleanly."""
    cfg = VPSConfig()
    assert cfg.vtx.enabled is False
    assert cfg.vtx.sink == "auto"
    assert cfg.vtx.output_width == 720


def test_vtx_hud_handles_small_resolution_empty_and_bgra_frames() -> None:
    """Verify renderer handles compact resolutions, empty frames, and 4-channel BGRA safely."""
    import cv2

    cfg = VTXConfig(enabled=True, output_width=320, output_height=240, show_features=True)
    renderer = VTXHudRenderer(cfg)
    data = VTXHudData(
        height_agl_m=12.3,
        baro_alt_m=12.1,
        confidence=0.92,
        is_armed=True,
        flight_mode="GUIDED",
        battery_voltage_v=16.2,
        gps_satellites=14,
        vel_n_mps=2.5,
        vel_e_mps=-1.2,
        is_shadow=True,
    )

    out_empty = renderer.render(np.empty((0, 0, 3), dtype=np.uint8), data)
    assert out_empty.shape == (240, 320, 3)

    bgra = np.full((480, 640, 4), 100, dtype=np.uint8)
    out_bgra = renderer.render(bgra, data)
    assert out_bgra.shape == (240, 320, 3)

    sink = MJPEGStreamSink(host="127.0.0.1", port=0)
    sink.write(out_bgra)
    seq1 = sink._frame_seq
    sink.write(cv2.cvtColor(bgra, cv2.COLOR_BGRA2BGR))
    assert sink._frame_seq == seq1 + 1
