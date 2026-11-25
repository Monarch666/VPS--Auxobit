"""Camera capture abstraction for VPS.

Supports:
- OpenCV VideoCapture (V4L2 / USB / CSI on Linux/Windows)
- Picamera2 adapter for Raspberry Pi 4 / Pi 5 (libcamera / Sony IMX296)
- Mock camera capture (replaying image directory or synthetic simulator)
- Threaded non-blocking capture buffer to ensure lowest latency (< 100 ms)
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Tuple, Optional, List
import logging
import threading
import time
import cv2
import numpy as np

from vps.config import CameraConfig

logger = logging.getLogger("VPS.Camera")


class CameraCaptureBase(ABC):
    """Abstract base class for camera capture."""

    @abstractmethod
    def open(self) -> bool:
        """Open camera stream."""
        pass

    def start(self) -> bool:
        """Alias for open()."""
        return self.open()

    @abstractmethod
    def read_frame(self) -> Tuple[bool, Optional[np.ndarray], float]:
        """Read latest frame.

        Returns:
            Tuple of (success, frame_bgr_or_gray, monotonic_timestamp_s).
        """
        pass

    @abstractmethod
    def release(self) -> None:
        """Release camera resources."""
        pass

    def stop(self) -> None:
        """Alias for release()."""
        self.release()


class OpenCVCameraCapture(CameraCaptureBase):
    """Camera capture using OpenCV VideoCapture backend (V4L2 / DirectShow / MSMF)."""

    def __init__(
        self,
        device_index: int | str = 0,
        config: Optional[CameraConfig] = None,
        use_threading: bool = True,
    ) -> None:
        """Initialize OpenCV capture.

        Args:
            device_index: Integer index (0, 1) or device path (e.g. "/dev/video0").
            config: Camera configuration with width, height, and fps.
            use_threading: If True, uses background reader thread to eliminate buffer lag.
        """
        self.device = device_index
        self.cfg = config if config is not None else CameraConfig()
        self.use_threading = use_threading
        self.cap: Optional[cv2.VideoCapture] = None

        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._lock = threading.Lock()
        self._latest_frame: Optional[np.ndarray] = None
        self._latest_timestamp: float = 0.0
        self._frame_ready = False

    def open(self) -> bool:
        """Open video capture device and configure format."""
        if isinstance(self.device, str) and self.device.isdigit():
            idx = int(self.device)
        else:
            idx = self.device

        self.cap = cv2.VideoCapture(idx)
        if not self.cap.isOpened():
            return False

        # Request resolution and framerate
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.cfg.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.cfg.height)
        self.cap.set(cv2.CAP_PROP_FPS, self.cfg.fps)

        if self.use_threading:
            self._running = True
            self._thread = threading.Thread(target=self._worker, daemon=True)
            self._thread.start()

        return True

    def _worker(self) -> None:
        """Background frame polling loop to keep video buffer fresh."""
        while self._running and self.cap is not None and self.cap.isOpened():
            ret, frame = self.cap.read()
            t_now = time.monotonic()
            if ret and frame is not None:
                with self._lock:
                    self._latest_frame = frame
                    self._latest_timestamp = t_now
                    self._frame_ready = True
            else:
                time.sleep(0.005)

    def read_frame(self) -> Tuple[bool, Optional[np.ndarray], float]:
        """Get latest frame without latency buffer queueing."""
        if not self.cap or not self.cap.isOpened():
            return False, None, 0.0

        if self.use_threading:
            with self._lock:
                if self._frame_ready and self._latest_frame is not None:
                    return True, self._latest_frame.copy(), self._latest_timestamp
                return False, None, 0.0
        else:
            ret, frame = self.cap.read()
            t_now = time.monotonic()
            return ret, frame, t_now

    def release(self) -> None:
        """Stop background worker and release capture device."""
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=0.5)
        if self.cap is not None:
            self.cap.release()
            self.cap = None


class MockCameraCapture(CameraCaptureBase):
    """Mock capture replaying frames from a directory, video file, or synthetic generator."""

    def __init__(
        self,
        frames: Optional[List[np.ndarray]] = None,
        image_dir: Optional[Path | str] = None,
        fps: float = 30.0,
        loop: bool = True,
    ) -> None:
        """Initialize mock capture.

        Args:
            frames: Pre-loaded list of numpy image arrays.
            image_dir: Directory containing ordered image files.
            fps: Playback framerate.
            loop: Whether to loop back to start after reaching the end.
        """
        self.frames: List[np.ndarray] = []
        if frames is not None:
            self.frames = list(frames)
        elif image_dir is not None:
            p = Path(image_dir)
            files = sorted(list(p.glob("*.jpg")) + list(p.glob("*.png")))
            self.frames = [cv2.imread(str(f)) for f in files if cv2.imread(str(f)) is not None]

        if len(self.frames) == 0:
            # Generate default synthetic frames if none provided
            rng = np.random.default_rng(42)
            base = np.full((1088, 1456, 3), 130, dtype=np.uint8)
            self.frames = [
                np.clip(base.astype(np.int16) + rng.integers(-25, 25, base.shape, dtype=np.int16), 0, 255).astype(np.uint8)
                for _ in range(5)
            ]

        self.fps = float(fps)
        self.loop = loop
        self.idx = 0
        self._is_open = False
        self._start_time = 0.0

    def open(self) -> bool:
        if len(self.frames) == 0:
            return False
        self._is_open = True
        self.idx = 0
        self._start_time = time.monotonic()
        return True

    def read_frame(self) -> Tuple[bool, Optional[np.ndarray], float]:
        if not self._is_open or len(self.frames) == 0:
            return False, None, 0.0

        if self.idx >= len(self.frames):
            if self.loop:
                self.idx = 0
            else:
                return False, None, 0.0

        frame = self.frames[self.idx]
        t = self._start_time + (self.idx / self.fps)
        self.idx += 1
        return True, frame.copy(), t

    def release(self) -> None:
        self._is_open = False


class Picamera2CameraCapture(CameraCaptureBase):
    """Camera capture using Raspberry Pi Picamera2/libcamera backend (Sony IMX296)."""

    def __init__(
        self,
        config: Optional[CameraConfig] = None,
        use_threading: bool = True,
    ) -> None:
        self.cfg = config if config is not None else CameraConfig()
        self.use_threading = use_threading
        self._picam2 = None
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._is_open = False
        self._lock = threading.Lock()
        self._latest_frame: Optional[np.ndarray] = None
        self._latest_timestamp: float = 0.0
        self._frame_ready = False

    @staticmethod
    def is_available() -> bool:
        """Return True if picamera2 is importable in the current environment."""
        try:
            import picamera2  # noqa: F401
            return True
        except ImportError:
            return False

    def open(self) -> bool:
        """Open IMX296 / CSI camera via Picamera2 at configured resolution and FPS."""
        try:
            from picamera2 import Picamera2
        except ImportError as e:
            logger.warning("Picamera2 not available: %s", e)
            return False

        try:
            self._picam2 = Picamera2()
            video_cfg = self._picam2.create_video_configuration(
                main={
                    "size": (int(self.cfg.width), int(self.cfg.height)),
                    "format": "YUV420",
                },
                buffer_count=4,
                queue=False,
                controls={"FrameRate": float(self.cfg.fps)},
            )
            self._picam2.configure(video_cfg)
            self._picam2.start()
            self._is_open = True

            if self.use_threading:
                self._running = True
                self._thread = threading.Thread(target=self._worker, daemon=True, name="VPS_Picamera2")
                self._thread.start()
            return True
        except Exception as e:
            logger.error("Failed to initialize Picamera2: %s", e)
            self.release()
            return False

    def _extract_gray(self, raw: np.ndarray) -> np.ndarray:
        """Extract exact (H, W) luminance grayscale plane from YUV420 buffer."""
        h, w = int(self.cfg.height), int(self.cfg.width)
        if raw.ndim == 2 and raw.shape[0] >= h and raw.shape[1] >= w:
            return raw[:h, :w].copy()
        return raw

    def _worker(self) -> None:
        """Background capture loop grabbing frames with monotonic timestamps."""
        while self._running and self._picam2 is not None:
            try:
                raw = self._picam2.capture_array()
                t_now = time.monotonic()
                if raw is not None:
                    frame = self._extract_gray(raw)
                    with self._lock:
                        self._latest_frame = frame
                        self._latest_timestamp = max(t_now, self._latest_timestamp + 1e-6)
                        self._frame_ready = True
            except Exception:
                time.sleep(0.005)

    def read_frame(self) -> Tuple[bool, Optional[np.ndarray], float]:
        """Return the newest unconsumed frame and its monotonic timestamp."""
        if not self._is_open or self._picam2 is None:
            return False, None, 0.0

        if self.use_threading:
            with self._lock:
                if self._frame_ready and self._latest_frame is not None:
                    self._frame_ready = False
                    return True, self._latest_frame.copy(), self._latest_timestamp
                return False, None, 0.0
        else:
            try:
                raw = self._picam2.capture_array()
                t_now = time.monotonic()
                if raw is not None:
                    return True, self._extract_gray(raw), t_now
            except Exception:
                pass
            return False, None, 0.0

    def release(self) -> None:
        """Stop background capture and release Picamera2 resources."""
        self._running = False
        self._is_open = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)
            self._thread = None
        if self._picam2 is not None:
            try:
                self._picam2.stop()
            except Exception:
                pass
            try:
                self._picam2.close()
            except Exception:
                pass
            self._picam2 = None


class AutoCameraCapture(CameraCaptureBase):
    """Selects OpenCV V4L2/USB capture or falls back to Picamera2 on Raspberry Pi CSI."""

    def __init__(
        self,
        device_index: int | str = 0,
        config: Optional[CameraConfig] = None,
    ) -> None:
        self.device_index = device_index
        self.cfg = config if config is not None else CameraConfig()
        self._backend: Optional[CameraCaptureBase] = None

    def open(self) -> bool:
        dev_str = str(self.device_index).lower().strip()
        if dev_str == "mock":
            self._backend = MockCameraCapture(fps=self.cfg.fps)
            return self._backend.open()

        if dev_str in ("picamera2", "libcamera", "csi"):
            self._backend = Picamera2CameraCapture(config=self.cfg)
            return self._backend.open()

        # On Raspberry Pi with libcamera/CSI (e.g., IMX296), raw V4L2 (/dev/video0)
        # opens in OpenCV without failing open() but yields zero frames on read().
        # Prefer Picamera2 automatically when a libcamera sensor is present for device 0.
        if dev_str in ("0", "/dev/video0") and Picamera2CameraCapture.is_available():
            try:
                from picamera2 import Picamera2
                if len(Picamera2.global_camera_info()) > 0:
                    logger.info("Detected Raspberry Pi CSI camera via libcamera; using Picamera2 backend for %s", self.device_index)
                    pi_cam = Picamera2CameraCapture(config=self.cfg)
                    if pi_cam.open():
                        self._backend = pi_cam
                        return True
            except Exception as e:
                logger.debug("Picamera2 probe skipped: %s", e)

        try:
            dev_arg: int | str = int(self.device_index)
        except (ValueError, TypeError):
            dev_arg = self.device_index

        cv_cam = OpenCVCameraCapture(device_index=dev_arg, config=self.cfg)
        if cv_cam.open():
            self._backend = cv_cam
            return True

        return False

    def read_frame(self) -> Tuple[bool, Optional[np.ndarray], float]:
        if self._backend is None:
            return False, None, 0.0
        return self._backend.read_frame()

    def release(self) -> None:
        if self._backend is not None:
            self._backend.release()
            self._backend = None


def create_camera_capture(
    device_index: int | str = 0,
    config: Optional[CameraConfig] = None,
) -> CameraCaptureBase:
    """Create the appropriate camera capture backend for the given device specifier."""
    cfg = config if config is not None else CameraConfig()
    if str(device_index).lower().strip() == "mock":
        return MockCameraCapture(fps=cfg.fps)
    return AutoCameraCapture(device_index=device_index, config=cfg)

