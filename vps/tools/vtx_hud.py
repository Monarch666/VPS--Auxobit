"""Lightweight VTX On-Screen Display (OSD) Renderer and Multi-Sink Output.

Provides a clean, aviation-style HUD overlaid onto the downward camera stream
while keeping >85% of the central field of view completely clear. Designed for
low-CPU asynchronous rendering on Raspberry Pi 4 over:
- Direct Linux Framebuffer (/dev/fb0) for composite RCA analog VTX or HDMI
- Lightweight HTTP MJPEG preview stream (port 8080) for wireless bench/GCS view
- OpenCV desktop window for bench/laptop testing
"""

from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, List, Optional, Tuple
import logging
import os
import threading
import time

import cv2
import numpy as np

from vps.config import VTXConfig

logger = logging.getLogger("VPS.VTX")


@dataclass
class VTXHudData:
    """Snapshot of telemetry and perception metrics shown on the VTX OSD."""

    height_agl_m: float = 0.0
    baro_alt_m: float = 0.0
    confidence: float = 0.0
    mode_status: str = "INITIALIZING"
    is_armed: bool = False
    flight_mode: str = "UNKNOWN"
    battery_voltage_v: float = 0.0
    gps_satellites: int = 0
    fps: float = 0.0
    num_inliers: int = 0
    pos_n_m: float = 0.0
    pos_e_m: float = 0.0
    vel_n_mps: float = 0.0
    vel_e_mps: float = 0.0
    map_status: str = "STANDBY"
    phase1_only: bool = False
    is_shadow: bool = False
    tracked_points: Optional[np.ndarray] = None
    source_width: int = 1456
    source_height: int = 1088


def _confidence_color_bgr(conf: float) -> Tuple[int, int, int]:
    """Return traffic-light BGR color based on estimator confidence."""
    if conf >= 0.60:
        return (80, 220, 80)  # Green
    if conf >= 0.30:
        return (0, 195, 255)  # Amber
    return (70, 70, 240)  # Red


def _draw_outlined_text(
    img: np.ndarray,
    text: str,
    org: Tuple[int, int],
    scale: float = 0.46,
    color: Tuple[int, int, int] = (245, 245, 245),
    thickness: int = 1,
) -> None:
    """Render crisp text with a 1px drop-shadow using identical thickness so kerning aligns."""
    sx, sy = org
    cv2.putText(img, text, (sx + 1, sy + 1), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thickness, cv2.LINE_AA)
    cv2.putText(img, text, (sx, sy), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def _measure_text_width(text: str, scale: float, thickness: int = 1) -> int:
    """Return rendered pixel width of a string for clean dynamic alignment."""
    (w, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    return int(w)


class VTXHudRenderer:
    """Minimalist aviation OSD renderer keeping the central FOV unobstructed."""

    def __init__(self, config: Optional[VTXConfig] = None) -> None:
        self.cfg = config if config is not None else VTXConfig()

    def render(self, frame: np.ndarray, data: VTXHudData) -> np.ndarray:
        """Render clean edge-anchored HUD onto a resized BGR copy of the camera frame."""
        out_w = max(320, int(self.cfg.output_width))
        out_h = max(240, int(self.cfg.output_height))

        if frame is None or frame.size == 0:
            bgr = np.zeros((out_h, out_w, 3), dtype=np.uint8)
        elif frame.ndim == 2:
            bgr = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        elif frame.ndim == 3 and frame.shape[2] == 4:
            bgr = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        else:
            bgr = frame

        src_h, src_w = bgr.shape[:2]
        if (src_w, src_h) != (out_w, out_h):
            interp = cv2.INTER_AREA if src_w > out_w else cv2.INTER_LINEAR
            canvas = cv2.resize(bgr, (out_w, out_h), interpolation=interp)
        else:
            canvas = bgr.copy()

        top_bar_h = max(24, int(round(out_h * 0.056)))
        bot_bar_h = max(24, int(round(out_h * 0.056)))
        conf_bar_h = 3

        # Blend thin translucent dark strips along top and bottom edges only
        overlay = canvas.copy()
        cv2.rectangle(overlay, (0, 0), (out_w, top_bar_h), (12, 15, 18), -1)
        cv2.rectangle(overlay, (0, out_h - bot_bar_h - conf_bar_h), (out_w, out_h - conf_bar_h), (12, 15, 18), -1)
        cv2.addWeighted(overlay, 0.68, canvas, 0.32, 0.0, dst=canvas)

        # Subtle center crosshair (non-intrusive 10px arms)
        cx, cy = out_w // 2, out_h // 2
        cv2.line(canvas, (cx - 10, cy), (cx + 10, cy), (210, 230, 210), 1, cv2.LINE_AA)
        cv2.line(canvas, (cx, cy - 10), (cx, cy + 10), (210, 230, 210), 1, cv2.LINE_AA)

        # Optional sparse tracked feature dots (small 2px points, no cluttered mesh)
        if self.cfg.show_features and data.tracked_points is not None and len(data.tracked_points) > 0:
            pts = np.asarray(data.tracked_points, dtype=np.float32).reshape(-1, 2)
            max_pts = max(1, int(self.cfg.max_drawn_features))
            step = max(1, len(pts) // max_pts)
            sx = out_w / max(1.0, float(src_w))
            sy = out_h / max(1.0, float(src_h))
            for pt in pts[::step][:max_pts]:
                px = int(round(pt[0] * sx))
                py = int(round(pt[1] * sy))
                if top_bar_h + 4 <= py <= out_h - bot_bar_h - conf_bar_h - 4 and 4 <= px <= out_w - 4:
                    cv2.circle(canvas, (px, py), 2, (0, 0, 0), -1, cv2.LINE_AA)
                    cv2.circle(canvas, (px, py), 1, (90, 240, 120), -1, cv2.LINE_AA)

        # Optional subtle Phase 2 horizontal velocity vector anchored at center
        if self.cfg.show_velocity_vector and not data.phase1_only:
            speed = float(np.hypot(data.vel_n_mps, data.vel_e_mps))
            if speed >= 0.15:
                scale_px = min(45.0, speed * 12.0) / max(speed, 1e-6)
                vx_px = int(round(data.vel_e_mps * scale_px))
                vy_px = int(round(-data.vel_n_mps * scale_px))
                cv2.arrowedLine(
                    canvas,
                    (cx, cy),
                    (cx + vx_px, cy + vy_px),
                    (0, 220, 255),
                    1,
                    cv2.LINE_AA,
                    tipLength=0.28,
                )

        # Top Strip Text
        font_scale = max(0.36, min(0.54, out_w / 1450.0))
        top_y = int(round(top_bar_h * 0.72))

        if data.is_shadow:
            badge = "SHADOW"
            badge_col = (0, 200, 255)
        elif data.phase1_only:
            badge = "PHASE1"
            badge_col = (240, 200, 80)
        else:
            badge = "VPS"
            badge_col = (80, 230, 110)

        arm_str = "ARM" if data.is_armed else "DIS"
        arm_col = (80, 240, 90) if data.is_armed else (175, 180, 185)
        mode_short = (data.flight_mode or "UNK")[:8]

        badge_text = f"[{badge}]"
        arm_text = f"{arm_str} {mode_short}"
        _draw_outlined_text(canvas, badge_text, (8, top_y), font_scale, badge_col)
        arm_x = 14 + _measure_text_width(badge_text, font_scale)
        _draw_outlined_text(canvas, arm_text, (arm_x, top_y), font_scale, arm_col)
        left_end_x = arm_x + _measure_text_width(arm_text, font_scale) + 10

        alt_str = f"AGL:{data.height_agl_m:5.2f}m  BARO:{data.baro_alt_m:5.2f}m"
        alt_w = _measure_text_width(alt_str, font_scale)
        right_top = f"{data.battery_voltage_v:4.1f}V  SAT:{data.gps_satellites:2d}"
        rt_w = _measure_text_width(right_top, font_scale)
        alt_x = min(max(left_end_x, (out_w - alt_w) // 2), max(left_end_x, out_w - rt_w - alt_w - 18))
        _draw_outlined_text(canvas, alt_str, (alt_x, top_y), font_scale, (255, 255, 255))

        _draw_outlined_text(canvas, right_top, (max(8, out_w - rt_w - 10), top_y), font_scale, (220, 230, 240))

        # Bottom Strip Text
        bot_y = out_h - conf_bar_h - int(round(bot_bar_h * 0.28))
        conf_col = _confidence_color_bgr(data.confidence)

        if data.phase1_only:
            status_short = (data.mode_status or "READY")[:18]
            left_bot = f"INL:{data.num_inliers:3d}  {status_short}"
        else:
            speed = float(np.hypot(data.vel_n_mps, data.vel_e_mps))
            left_bot = f"POS[{data.pos_n_m:+5.1f},{data.pos_e_m:+5.1f}]m  VEL:{speed:4.1f}m/s"
        _draw_outlined_text(canvas, left_bot, (8, bot_y), font_scale, (235, 240, 245))

        map_tag = "P1" if data.phase1_only else (data.map_status or "VIO")[:10]
        conf_pct = int(round(np.clip(data.confidence, 0.0, 1.0) * 100.0))
        right_bot = f"{map_tag} | {conf_pct:3d}% | {data.fps:4.1f}FPS"
        rb_w = _measure_text_width(right_bot, font_scale)
        _draw_outlined_text(canvas, right_bot, (max(8, out_w - rb_w - 10), bot_y), font_scale, conf_col)

        # Bottom-edge peripheral confidence bar (3px tall)
        fill_w = int(round(out_w * np.clip(data.confidence, 0.0, 1.0)))
        cv2.rectangle(canvas, (0, out_h - conf_bar_h), (out_w, out_h), (35, 38, 42), -1)
        if fill_w > 0:
            cv2.rectangle(canvas, (0, out_h - conf_bar_h), (fill_w, out_h), conf_col, -1)

        return canvas


def bgr_to_rgb565_bytes(bgr: np.ndarray) -> bytes:
    """Convert BGR uint8 image to packed little-endian RGB565 bytes for 16-bit /dev/fb0."""
    b = (bgr[:, :, 0].astype(np.uint16) >> 3)
    g = (bgr[:, :, 1].astype(np.uint16) >> 2) << 5
    r = (bgr[:, :, 2].astype(np.uint16) >> 3) << 11
    rgb565 = (r | g | b).astype("<u2")
    return rgb565.tobytes()


class FramebufferSink:
    """Direct writer to Linux Framebuffer (/dev/fb0) for composite analog VTX or HDMI console."""

    def __init__(self, fb_device: str = "/dev/fb0") -> None:
        self.fb_device = fb_device
        self.width = 720
        self.height = 576
        self.bpp = 32
        self._fd: Optional[int] = None

    def open(self) -> bool:
        """Probe framebuffer geometry via sysfs and open device for writing."""
        fb_path = Path(self.fb_device)
        if not fb_path.exists():
            return False

        sys_dir = Path("/sys/class/graphics") / fb_path.name
        try:
            vsize = (sys_dir / "virtual_size").read_text().strip().split(",")
            self.width, self.height = int(vsize[0]), int(vsize[1])
            self.bpp = int((sys_dir / "bits_per_pixel").read_text().strip())
        except Exception:
            pass

        try:
            self._fd = os.open(self.fb_device, os.O_WRONLY)
            logger.info("Opened VTX framebuffer %s (%dx%d @ %dbpp)", self.fb_device, self.width, self.height, self.bpp)
            return True
        except OSError as exc:
            logger.warning("Cannot open framebuffer %s: %s", self.fb_device, exc)
            self._fd = None
            return False

    def write(self, bgr: np.ndarray) -> None:
        """Scale frame if needed and write packed pixels directly to /dev/fb0."""
        if self._fd is None:
            return
        h, w = bgr.shape[:2]
        if (w, h) != (self.width, self.height):
            img = cv2.resize(bgr, (self.width, self.height), interpolation=cv2.INTER_NEAREST)
        else:
            img = bgr

        if self.bpp == 16:
            payload = bgr_to_rgb565_bytes(img)
        else:
            bgra = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
            payload = bgra.tobytes()

        try:
            os.lseek(self._fd, 0, os.SEEK_SET)
            os.write(self._fd, payload)
        except OSError:
            pass

    def close(self) -> None:
        if self._fd is not None:
            try:
                os.close(self._fd)
            except OSError:
                pass
            self._fd = None


class MJPEGStreamSink:
    """Lightweight HTTP MJPEG stream server for viewing the VTX OSD over Wi-Fi/Ethernet."""

    def __init__(self, host: str = "0.0.0.0", port: int = 8080) -> None:
        self.host = host
        self.port = int(port)
        self._cond = threading.Condition()
        self._latest_jpeg: bytes = b""
        self._frame_seq: int = 0
        self._server: Optional[ThreadingHTTPServer] = None
        self._thread: Optional[threading.Thread] = None
        self._running = False

    def start(self) -> bool:
        """Start the background HTTP MJPEG server."""
        sink_ref = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:
                if self.path in ("/", "/index.html"):
                    html = (
                        b"<html><body style='margin:0;background:#000;display:flex;"
                        b"justify-content:center;align-items:center;height:100vh'>"
                        b"<img src='/stream.mjpg' style='max-width:100%;max-height:100%'/>"
                        b"</body></html>"
                    )
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html")
                    self.send_header("Content-Length", str(len(html)))
                    self.end_headers()
                    self.wfile.write(html)
                    return

                if self.path == "/stream.mjpg":
                    self.send_response(200)
                    self.send_header("Age", "0")
                    self.send_header("Cache-Control", "no-cache, private")
                    self.send_header("Pragma", "no-cache")
                    self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=FRAME")
                    self.end_headers()
                    last_seq = -1
                    while sink_ref._running:
                        with sink_ref._cond:
                            sink_ref._cond.wait_for(
                                lambda: (not sink_ref._running) or (sink_ref._frame_seq != last_seq),
                                timeout=0.5,
                            )
                            if not sink_ref._running:
                                break
                            jpg = sink_ref._latest_jpeg
                            last_seq = sink_ref._frame_seq
                        if not jpg:
                            continue
                        try:
                            self.wfile.write(b"--FRAME\r\n")
                            self.wfile.write(b"Content-Type: image/jpeg\r\n")
                            self.wfile.write(f"Content-Length: {len(jpg)}\r\n\r\n".encode("ascii"))
                            self.wfile.write(jpg)
                            self.wfile.write(b"\r\n")
                        except (BrokenPipeError, ConnectionResetError, OSError):
                            break
                    return

                self.send_error(404)

            def log_message(self, format: str, *args: Any) -> None:
                pass

        try:
            self._server = ThreadingHTTPServer((self.host, self.port), Handler)
            self._server.daemon_threads = True
            self._running = True
            self._thread = threading.Thread(target=self._server.serve_forever, name="VTX-MJPEG", daemon=True)
            self._thread.start()
            logger.info("VTX MJPEG live preview streaming at http://%s:%d/", self.host, self.port)
            return True
        except OSError as exc:
            logger.warning("Failed to bind VTX MJPEG server on port %d: %s", self.port, exc)
            return False

    def write(self, bgr: np.ndarray) -> None:
        ok, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 75])
        if ok:
            with self._cond:
                self._latest_jpeg = buf.tobytes()
                self._frame_seq += 1
                self._cond.notify_all()

    def close(self) -> None:
        with self._cond:
            self._running = False
            self._cond.notify_all()
        if self._server is not None:
            try:
                self._server.shutdown()
                self._server.server_close()
            except Exception:
                pass
            self._server = None
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None


class VTXDisplayServer:
    """Non-blocking background worker managing VTX OSD rendering and output sinks."""

    def __init__(self, config: Optional[VTXConfig] = None) -> None:
        self.cfg = config if config is not None else VTXConfig()
        self.renderer = VTXHudRenderer(self.cfg)
        self._fb_sink: Optional[FramebufferSink] = None
        self._mjpeg_sink: Optional[MJPEGStreamSink] = None
        self._use_window = False

        self._slot_lock = threading.Lock()
        self._slot_event = threading.Event()
        self._pending: Optional[Tuple[np.ndarray, VTXHudData]] = None
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._min_interval_s = 1.0 / max(1.0, float(self.cfg.target_fps))
        self._last_render_t = 0.0

    def start(self) -> bool:
        """Initialize configured VTX output sinks and launch background render thread."""
        sink_mode = (self.cfg.sink or "auto").lower()

        if "fb" in sink_mode or sink_mode == "auto":
            fb = FramebufferSink(self.cfg.fb_device)
            if fb.open():
                self._fb_sink = fb
            elif sink_mode == "fb":
                logger.warning("Framebuffer %s unavailable; VTX fb sink disabled.", self.cfg.fb_device)

        if "mjpeg" in sink_mode:
            mjpeg = MJPEGStreamSink(port=self.cfg.mjpeg_port)
            if mjpeg.start():
                self._mjpeg_sink = mjpeg

        if sink_mode == "window":
            self._use_window = True
        elif sink_mode == "auto" and self._fb_sink is None and os.environ.get("DISPLAY"):
            self._use_window = True

        self._running = True
        self._thread = threading.Thread(target=self._worker_loop, name="VTX-HUD-Worker", daemon=True)
        self._thread.start()
        return True

    def submit(self, frame: np.ndarray, hud_data: VTXHudData) -> None:
        """Queue latest frame and telemetry for VTX OSD (latest-wins, zero blocking)."""
        if not self._running:
            return
        now = time.monotonic()
        if now - self._last_render_t < self._min_interval_s:
            return
        with self._slot_lock:
            self._pending = (frame.copy(), hud_data)
        self._slot_event.set()

    def _worker_loop(self) -> None:
        while self._running:
            self._slot_event.wait(timeout=0.25)
            self._slot_event.clear()
            with self._slot_lock:
                item = self._pending
                self._pending = None
            if item is None:
                continue

            frame, data = item
            self._last_render_t = time.monotonic()
            osd_bgr = self.renderer.render(frame, data)

            if self._fb_sink is not None:
                self._fb_sink.write(osd_bgr)
            if self._mjpeg_sink is not None:
                self._mjpeg_sink.write(osd_bgr)
            if self._use_window:
                try:
                    cv2.imshow("VPS VTX HUD", osd_bgr)
                    cv2.waitKey(1)
                except cv2.error:
                    self._use_window = False

    def stop(self) -> None:
        """Stop render thread and release framebuffer/HTTP sinks cleanly."""
        self._running = False
        self._slot_event.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None
        if self._fb_sink is not None:
            self._fb_sink.close()
            self._fb_sink = None
        if self._mjpeg_sink is not None:
            self._mjpeg_sink.close()
            self._mjpeg_sink = None
        if self._use_window:
            try:
                cv2.destroyWindow("VPS VTX HUD")
            except cv2.error:
                pass
            self._use_window = False
