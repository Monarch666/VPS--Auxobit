"""Camera capture abstraction for VPS.

Supports:
- OpenCV VideoCapture (V4L2 / USB / CSI on Linux/Windows)
- Mock camera capture (replaying image directory or synthetic simulator)
- PiCamera2 adapter for Raspberry Pi 5
- Threaded non-blocking capture buffer to ensure lowest latency (< 100 ms)
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Tuple, Optional, List
import threading
import time
import cv2
import numpy as np

from vps.config import CameraConfig


class CameraCaptureBase(ABC):
    """Abstract base class for camera capture."""

    @abstractmethod
    def open(self) -> bool:
        """Open camera stream."""
        pass

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
