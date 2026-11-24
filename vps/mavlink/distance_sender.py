"""MAVLink DISTANCE_SENSOR publisher for ArduPilot rangefinder integration.

Constructs and transmits DISTANCE_SENSOR messages:
- Downward orientation: MAV_SENSOR_ROTATION_PITCH_270 (value 25)
- Detection envelope: 10 cm to 30,000 cm (300 m)
- Confidence mapping: signal_quality = int(confidence * 100) (0 to 100%)
- Continuous streaming at >= 10 Hz (default 15 Hz)
"""

import os
os.environ["MAVLINK20"] = "1"

from dataclasses import dataclass
from typing import Optional
import inspect
import threading
import time
import numpy as np
from pymavlink import mavutil

from vps.config import MAVLinkConfig


class DistanceSensorSender:
    """Streams DISTANCE_SENSOR packets over MAVLink connection at fixed rate."""

    def __init__(
        self,
        mav_connection: mavutil.mavlink_connection,
        config: Optional[MAVLinkConfig] = None,
        rate_hz: float = 15.0,
        chokepoint: Optional[Any] = None,
    ) -> None:
        """Initialize distance sensor sender.

        Args:
            mav_connection: Active pymavlink connection.
            config: MAVLink configuration parameters.
            rate_hz: Transmission rate (Hz), must be >= 10 Hz.
            chokepoint: Optional ShadowMAVLinkSafetyChokepoint instance.
        """
        self.mav = mav_connection
        self.cfg = config if config is not None else MAVLinkConfig()
        self.rate_hz = max(10.0, float(rate_hz))
        self.period_s = 1.0 / self.rate_hz
        self.chokepoint = chokepoint

        self._lock = threading.Lock()
        self._current_distance_cm: int = 100
        self._signal_quality: int = 0
        self._covariance: int = 0
        self._is_running = False
        self._thread: Optional[threading.Thread] = None
        self._sent_count = 0
        self._boot_time_s = time.monotonic()

        # Inspect signature to support both MAVLink 1 and MAVLink 2 dialects
        sig = inspect.signature(self.mav.mav.distance_sensor_send)
        self._supports_mavlink2 = "signal_quality" in sig.parameters

    def update_measurement(
        self,
        height_agl_m: float,
        confidence: float,
        covariance_m2: float = 0.05,
    ) -> None:
        """Update current height estimate to be dispatched.

        Args:
            height_agl_m: Estimated Above Ground Level height in meters.
            confidence: Estimated confidence score [0.0, 1.0].
            covariance_m2: Estimated height error variance (m^2).
        """
        dist_cm = int(round(height_agl_m * 100.0))
        dist_clamped = max(self.cfg.min_distance_cm, min(self.cfg.max_distance_cm, dist_cm))
        qual = int(np.clip(round(confidence * 100.0), 0, 100))
        cov = int(min(255, max(0, int(round(covariance_m2 * 1000.0)))))

        with self._lock:
            self._current_distance_cm = dist_clamped
            self._signal_quality = qual
            self._covariance = cov

    def send_single_packet(self) -> bool:
        """Construct and send one DISTANCE_SENSOR packet immediately."""
        if self.chokepoint and not self.chokepoint.is_message_allowed("DISTANCE_SENSOR"):
            return False

        with self._lock:
            dist_cm = self._current_distance_cm
            qual = self._signal_quality
            cov = self._covariance

        time_boot_ms = int((time.monotonic() - self._boot_time_s) * 1000) & 0xFFFFFFFF
        orientation = mavutil.mavlink.MAV_SENSOR_ROTATION_PITCH_270

        sensor_type = getattr(mavutil.mavlink, "MAV_DISTANCE_SENSOR_LASER", 0)

        try:
            if self._supports_mavlink2:
                self.mav.mav.distance_sensor_send(
                    time_boot_ms=time_boot_ms,
                    min_distance=self.cfg.min_distance_cm,
                    max_distance=self.cfg.max_distance_cm,
                    current_distance=dist_cm,
                    type=sensor_type,
                    id=0,
                    orientation=orientation,
                    covariance=cov,
                    horizontal_fov=0.0,
                    vertical_fov=0.0,
                    quaternion=[1.0, 0.0, 0.0, 0.0],
                    signal_quality=qual,
                )
            else:
                self.mav.mav.distance_sensor_send(
                    time_boot_ms=time_boot_ms,
                    min_distance=self.cfg.min_distance_cm,
                    max_distance=self.cfg.max_distance_cm,
                    current_distance=dist_cm,
                    type=sensor_type,
                    id=0,
                    orientation=orientation,
                    covariance=cov,
                )
            self._sent_count += 1
            return True
        except Exception as e:
            return False

    def start_streaming(self) -> None:
        """Start background transmission thread."""
        if self._is_running:
            return
        self._is_running = True
        self._thread = threading.Thread(target=self._streaming_worker, daemon=True)
        self._thread.start()

    def _streaming_worker(self) -> None:
        """Periodic transmission loop maintaining exact rate."""
        next_send_time = time.monotonic()
        while self._is_running:
            self.send_single_packet()
            next_send_time += self.period_s
            sleep_dt = next_send_time - time.monotonic()
            if sleep_dt > 0:
                time.sleep(sleep_dt)
            else:
                next_send_time = time.monotonic()

    def stop_streaming(self) -> None:
        """Stop background transmission thread."""
        self._is_running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=0.5)

    @property
    def packets_sent(self) -> int:
        return self._sent_count
