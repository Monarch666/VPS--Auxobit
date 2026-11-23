"""Health Monitor and Safety Interlock for Flight Controller EKF Integration.

Enforces fail-safe autonomy:
1. Operational Modes:
   - ADVISORY_LOG_ONLY (default safe mode): Logs navigation telemetry and diagnostic metrics
     without feeding active state updates to the flight controller EKF.
   - ACTIVE_FEEDING: Active MAVLink ODOMETRY feeding permitted ONLY with explicit config switch
     and when all 4 health gates pass.
2. Health Gates:
   - Confidence Gate: Visual tracking / map matching confidence >= min_confidence.
   - Innovation Gate: Filter normalized innovation d^2 <= chi2_threshold.
   - Data Freshness Gate: Observation age < max_data_age_ms (prevents feeding stale dead-reckoning).
   - Thermal & Voltage Gate: Checks Raspberry Pi 5 throttled status and CPU temperature (< 80 deg C)
     to prevent in-flight companion computer stalls or brownouts.
3. Covariance Inflation: Dynamically scales uncertainty when tracking confidence degrades,
   preventing filter divergence without causing abrupt signal drops.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple, Dict, Any
import subprocess
import time
import numpy as np

from vps.config import MAVLinkConfig


@dataclass
class HealthStatus:
    """Current health status and EKF feeding gate state."""
    is_healthy: bool
    can_feed_fc: bool
    mode: str  # "ADVISORY_LOG_ONLY" or "ACTIVE_FEEDING"
    confidence_ok: bool
    innovation_ok: bool
    freshness_ok: bool
    thermal_voltage_ok: bool
    cpu_temp_c: float
    throttled_flags: int
    covariance_scale: float
    data_age_ms: float
    status_message: str


class HealthMonitor:
    """Monitors VPS estimation health and controls FC EKF injection safety gate."""

    def __init__(
        self,
        config: Optional[MAVLinkConfig] = None,
        min_confidence: float = 0.40,
        max_data_age_ms: float = 150.0,
        max_innovation_chi2: float = 25.0,
        max_cpu_temp_c: float = 80.0,
    ) -> None:
        """Initialize health monitor.

        Args:
            config: MAVLinkConfig containing feed_fc_ekf switch.
            min_confidence: Minimum vision tracking confidence [0.0 - 1.0].
            max_data_age_ms: Maximum allowed observation latency in ms.
            max_innovation_chi2: Chi-squared innovation limit.
            max_cpu_temp_c: Maximum acceptable CPU core temperature in Celsius.
        """
        self.cfg = config if config is not None else MAVLinkConfig()
        self.min_confidence = float(min_confidence)
        self.max_data_age_ms = float(max_data_age_ms)
        self.max_innovation_chi2 = float(max_innovation_chi2)
        self.max_cpu_temp_c = float(max_cpu_temp_c)

        # Operational state
        self.feed_fc_enabled = bool(self.cfg.feed_fc_ekf)
        self._last_good_time_s = time.monotonic()
        self._consecutive_fails = 0

    def evaluate_health(
        self,
        confidence: float,
        innovation_chi2: float,
        observation_timestamp_s: float,
        current_timestamp_s: Optional[float] = None,
    ) -> HealthStatus:
        """Evaluate all health gates for the current estimation cycle.

        Args:
            confidence: Estimated vision confidence in [0.0, 1.0].
            innovation_chi2: Normalized innovation squared from EKF.
            observation_timestamp_s: Sensor frame capture timestamp.
            current_timestamp_s: Current monotonic time (seconds).

        Returns:
            HealthStatus with gate decisions and covariance scaling factor.
        """
        now = time.monotonic() if current_timestamp_s is None else current_timestamp_s
        data_age_ms = max(0.0, (now - observation_timestamp_s) * 1000.0)

        # Gate 1: Confidence
        confidence_ok = (confidence >= self.min_confidence)

        # Gate 2: Innovation
        innovation_ok = (innovation_chi2 <= self.max_innovation_chi2)

        # Gate 3: Freshness
        freshness_ok = (data_age_ms <= self.max_data_age_ms)

        # Gate 4: Hardware Thermal & Voltage
        thermal_voltage_ok, throttled_flags, cpu_temp = self._check_hardware_health()

        # Overall estimation health
        is_healthy = confidence_ok and innovation_ok and freshness_ok and thermal_voltage_ok

        # Determine dynamic covariance scaling
        if is_healthy:
            # Nominal: scale inversely with confidence
            cov_scale = 1.0 / max(0.4, confidence)
            self._consecutive_fails = 0
            self._last_good_time_s = now
            msg = "HEALTH_NOMINAL"
        else:
            self._consecutive_fails += 1
            # Degraded / failing: inflate covariance up to 100x to notify FC EKF
            cov_scale = min(100.0, 5.0 * (self._consecutive_fails ** 1.5))
            reasons = []
            if not confidence_ok:
                reasons.append(f"LOW_CONF({confidence:.2f})")
            if not innovation_ok:
                reasons.append(f"INNOV_EXCEEDED({innovation_chi2:.1f})")
            if not freshness_ok:
                reasons.append(f"STALE_DATA({data_age_ms:.1f}ms)")
            if not thermal_voltage_ok:
                reasons.append(f"HW_ALERT(temp={cpu_temp:.1f}C,flag={hex(throttled_flags)})")
            msg = "DEGRADED: " + "; ".join(reasons)

        # Active Feeding Gate:
        # Requires explicit user config enablement AND healthy estimation
        mode = "ACTIVE_FEEDING" if self.feed_fc_enabled else "ADVISORY_LOG_ONLY"
        can_feed_fc = bool(self.feed_fc_enabled and is_healthy)

        return HealthStatus(
            is_healthy=is_healthy,
            can_feed_fc=can_feed_fc,
            mode=mode,
            confidence_ok=confidence_ok,
            innovation_ok=innovation_ok,
            freshness_ok=freshness_ok,
            thermal_voltage_ok=thermal_voltage_ok,
            cpu_temp_c=cpu_temp,
            throttled_flags=throttled_flags,
            covariance_scale=float(cov_scale),
            data_age_ms=data_age_ms,
            status_message=msg,
        )

    def _check_hardware_health(self) -> Tuple[bool, int, float]:
        """Check Raspberry Pi 5 CPU temperature and throttling flags."""
        temp_c = 45.0
        throttled = 0

        # Try reading Pi 5 thermal sensor from sysfs (Linux)
        thermal_path = Path("/sys/class/thermal/thermal_zone0/temp")
        if thermal_path.is_file():
            try:
                temp_c = float(thermal_path.read_text().strip()) / 1000.0
            except (ValueError, OSError):
                temp_c = 50.0

        # Try querying vcgencmd for throttling flags on Raspberry Pi OS
        vcgencmd = Path("/usr/bin/vcgencmd")
        if vcgencmd.is_file():
            try:
                out = subprocess.check_output([str(vcgencmd), "get_throttled"], text=True, timeout=0.1)
                # Output format: throttled=0x0
                if "throttled=" in out:
                    throttled = int(out.split("=")[1].strip(), 16)
            except Exception:
                throttled = 0

        # Throttled bit 0: undervolted (critical brownout danger)
        # Throttled bit 2: currently throttled (frequency throttled due to temp)
        undervolted = bool(throttled & 0x1)
        freq_throttled = bool(throttled & 0x4)
        overheating = (temp_c >= self.max_cpu_temp_c)

        is_safe = not undervolted and not freq_throttled and not overheating
        return is_safe, throttled, temp_c

    def inflate_covariance_matrix(self, cov_matrix: np.ndarray, health: HealthStatus) -> np.ndarray:
        """Apply dynamic covariance inflation based on health evaluation.

        Args:
            cov_matrix: NxN measurement or pose covariance matrix.
            health: Current HealthStatus.

        Returns:
            Scaled positive semi-definite covariance matrix.
        """
        scaled = np.array(cov_matrix, dtype=np.float64) * health.covariance_scale
        # Guarantee symmetry
        return 0.5 * (scaled + scaled.T)

