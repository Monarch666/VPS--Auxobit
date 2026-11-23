"""Telemetry interpolation ring buffer for exact timestamp synchronization.

Stores high-rate telemetry packets (gyro, attitude, velocity, baro) and interpolates
them to the exact camera exposure midpoint timestamp:
    t_interp = t_exposure + time_offset_s
"""

from dataclasses import dataclass, field
from typing import Optional, List, Tuple
import threading
import numpy as np


@dataclass
class TelemetrySample:
    """Individual telemetry sample with monotonic timestamp."""
    timestamp_s: float
    # High-rate Gyro in body frame (rad/s) [p, q, r]
    gyro_rad_s: np.ndarray  # shape (3,)
    # Attitude Euler angles in body frame (rad) [roll, pitch, yaw]
    attitude_rpy_rad: np.ndarray  # shape (3,)
    # Velocity in Local NED frame (m/s) [Vn, Ve, Vd]
    vel_ned_mps: np.ndarray  # shape (3,)
    # Barometer altitude (m)
    baro_alt_m: float
    # 3D Position in Local NED (m) [X, Y, Z]
    pos_ned_m: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))
    # High-rate Acceleration in body frame (m/s^2) [ax, ay, az] (default 1g downward)
    accel_mps2: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, -9.80665], dtype=np.float64))


class TelemetryRingBuffer:
    """Thread-safe circular telemetry buffer supporting sub-millisecond interpolation."""

    def __init__(
        self,
        max_duration_s: float = 5.0,
        time_offset_s: float = 0.0,  # camera exposure midpoint to telemetry clock offset
    ) -> None:
        """Initialize telemetry ring buffer.

        Args:
            max_duration_s: Maximum history to preserve in seconds.
            time_offset_s: Configurable calibration offset added to camera timestamps.
        """
        self.max_duration_s = float(max_duration_s)
        self.time_offset_s = float(time_offset_s)
        self._samples: List[TelemetrySample] = []
        self._lock = threading.Lock()

    def add_sample(self, sample: TelemetrySample) -> None:
        """Insert a new telemetry sample into the buffer."""
        with self._lock:
            # Maintain monotonic order
            if not self._samples or sample.timestamp_s >= self._samples[-1].timestamp_s:
                self._samples.append(sample)
            else:
                # Out-of-order insertion (binary search insertion)
                idx = 0
                for i, s in enumerate(self._samples):
                    if sample.timestamp_s < s.timestamp_s:
                        idx = i
                        break
                self._samples.insert(idx, sample)

            # Prune old samples
            t_cutoff = self._samples[-1].timestamp_s - self.max_duration_s
            while self._samples and self._samples[0].timestamp_s < t_cutoff:
                self._samples.pop(0)

    def __len__(self) -> int:
        with self._lock:
            return len(self._samples)

    def interpolate(self, target_timestamp_s: float) -> Optional[TelemetrySample]:
        """Interpolate telemetry state at target timestamp with configured time offset.

        Args:
            target_timestamp_s: Frame exposure timestamp (s).

        Returns:
            Interpolated TelemetrySample or None if target is outside available buffer window.
        """
        query_t = target_timestamp_s + self.time_offset_s

        with self._lock:
            if len(self._samples) < 2:
                return self._samples[0] if self._samples else None

            # Check bounds
            if query_t < self._samples[0].timestamp_s:
                # Query is older than oldest sample: clamp to oldest
                return self._samples[0]
            if query_t > self._samples[-1].timestamp_s:
                # Query is newer than newest sample: clamp to newest
                return self._samples[-1]

            # Find bounding bracket [t0, t1]
            s0 = self._samples[0]
            s1 = self._samples[1]
            for i in range(len(self._samples) - 1):
                if self._samples[i].timestamp_s <= query_t <= self._samples[i + 1].timestamp_s:
                    s0 = self._samples[i]
                    s1 = self._samples[i + 1]
                    break

            dt = s1.timestamp_s - s0.timestamp_s
            if dt <= 1e-7:
                return s0

            alpha = (query_t - s0.timestamp_s) / dt

            # Linear interpolation for vectors
            gyro_interp = (1.0 - alpha) * s0.gyro_rad_s + alpha * s1.gyro_rad_s
            vel_interp = (1.0 - alpha) * s0.vel_ned_mps + alpha * s1.vel_ned_mps
            pos_interp = (1.0 - alpha) * s0.pos_ned_m + alpha * s1.pos_ned_m
            accel_interp = (1.0 - alpha) * s0.accel_mps2 + alpha * s1.accel_mps2
            baro_interp = float((1.0 - alpha) * s0.baro_alt_m + alpha * s1.baro_alt_m)

            # Attitude interpolation: handle yaw wrap-around (-pi to +pi)
            att_diff = s1.attitude_rpy_rad - s0.attitude_rpy_rad
            # Wrap angles to [-pi, pi]
            att_diff = (att_diff + np.pi) % (2.0 * np.pi) - np.pi
            att_interp = s0.attitude_rpy_rad + alpha * att_diff

            return TelemetrySample(
                timestamp_s=query_t,
                gyro_rad_s=gyro_interp,
                attitude_rpy_rad=att_interp,
                vel_ned_mps=vel_interp,
                baro_alt_m=baro_interp,
                pos_ned_m=pos_interp,
                accel_mps2=accel_interp,
            )

    def get_gyro_samples_between(
        self,
        t_start_s: float,
        t_end_s: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Extract all high-rate gyro samples between two timestamps for de-rotation integration.

        Args:
            t_start_s: Start timestamp (s).
            t_end_s: End timestamp (s).

        Returns:
            Tuple of (gyro_samples_array (N, 3), timestamps_array (N,)).
        """
        t0 = min(t_start_s, t_end_s) + self.time_offset_s
        t1 = max(t_start_s, t_end_s) + self.time_offset_s

        with self._lock:
            selected_gyros = []
            selected_times = []
            for s in self._samples:
                if t0 <= s.timestamp_s <= t1:
                    selected_gyros.append(s.gyro_rad_s)
                    selected_times.append(s.timestamp_s)

            if not selected_gyros:
                return np.zeros((0, 3), dtype=np.float64), np.zeros(0, dtype=np.float64)

            return np.array(selected_gyros, dtype=np.float64), np.array(selected_times, dtype=np.float64)

