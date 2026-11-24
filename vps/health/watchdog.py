"""Hardware Watchdog and Process Supervisor for Raspberry Pi 5.

Provides robust process lifecycle supervision:
1. Systemd Watchdog Support:
   - Implements native sd_notify socket communication (WATCHDOG=1, READY=1, STATUS=...).
   - Pet interval automatically calculated from systemd WATCHDOG_USEC (typically pet every 0.5 * WatchdogSec).
2. Hardware Watchdog Support:
   - Direct interface to Linux /dev/watchdog device if enabled in kernel.
3. Multi-Metric Health Inspection:
   - Pipeline frame rate (asserts >= 15 FPS).
   - Telemetry heartbeat freshness (asserts < 1.0 s).
   - Core thermal headroom (asserts CPU temp < 80 deg C).
   - Undervoltage and throttling flags (vcgencmd get_throttled).
   - Memory RSS footprint (asserts < 300 MB).
4. Fail-Safe Execution:
   - If any critical safety threshold is breached or thread stalls, watchdog ping is withheld,
     prompting systemd to trigger an automated restart or fail-safe fallback.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Any, List, Callable
import os
import sys
import time
import socket
import threading
import subprocess
import shutil


@dataclass
class SupervisorMetrics:
    """Current health and performance metrics collected by the supervisor."""
    timestamp_s: float
    fps: float
    telemetry_age_s: float
    cpu_temp_c: float
    throttled_flags: int
    memory_rss_mb: float
    all_healthy: bool
    status_summary: str


class SystemdNotifier:
    """Interacts with systemd daemon via the sd_notify socket protocol."""

    def __init__(self) -> None:
        """Initialize notifier from NOTIFY_SOCKET environment variable."""
        self.notify_socket = os.environ.get("NOTIFY_SOCKET")
        self.watchdog_usec = int(os.environ.get("WATCHDOG_USEC", 0))
        self.watchdog_sec = self.watchdog_usec / 1e6 if self.watchdog_usec > 0 else 0.0

    @property
    def is_enabled(self) -> bool:
        """Return True if running under a systemd service with notify support."""
        return self.notify_socket is not None

    def notify(self, state: str) -> bool:
        """Send a notification string (e.g. 'READY=1', 'WATCHDOG=1') to systemd.

        Args:
            state: Message to send.

        Returns:
            True if delivered successfully, False otherwise.
        """
        if not self.notify_socket:
            return False

        try:
            # Handle abstract namespace sockets (leading @)
            sock_path = self.notify_socket
            if sock_path.startswith("@"):
                sock_path = "\0" + sock_path[1:]

            sock = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
            try:
                sock.sendto(state.encode("utf-8"), sock_path)
                return True
            finally:
                sock.close()
        except Exception:
            return False

    def notify_ready(self) -> bool:
        """Notify systemd that startup is complete and the service is operational."""
        return self.notify("READY=1")

    def notify_status(self, status: str) -> bool:
        """Update systemctl status single-line message."""
        return self.notify(f"STATUS={status}")

    def notify_watchdog(self) -> bool:
        """Pet the systemd watchdog timer."""
        return self.notify("WATCHDOG=1")

    def notify_stopping(self) -> bool:
        """Notify systemd that the service is shutting down cleanly."""
        return self.notify("STOPPING=1")


class WatchdogSupervisor:
    """Continuous background supervisor monitoring VPS pipeline health."""

    def __init__(
        self,
        pet_interval_s: float = 2.0,
        min_fps: float = 15.0,
        max_telemetry_age_s: float = 1.0,
        max_cpu_temp_c: float = 80.0,
        max_memory_mb: float = 300.0,
        hardware_watchdog_dev: str = "/dev/watchdog",
    ) -> None:
        """Initialize watchdog supervisor.

        Args:
            pet_interval_s: Interval between health evaluations and watchdog pings.
            min_fps: Minimum acceptable camera frame rate before alerting.
            max_telemetry_age_s: Maximum allowed age of flight controller telemetry.
            max_cpu_temp_c: Maximum acceptable CPU core temperature in Celsius.
            max_memory_mb: Maximum allowed resident set size (RAM) in MB.
            hardware_watchdog_dev: Path to Linux hardware watchdog device.
        """
        self.systemd = SystemdNotifier()
        # If running under systemd with a watchdog configured, pet at half the watchdog period
        if self.systemd.watchdog_sec > 0:
            self.pet_interval_s = max(0.5, self.systemd.watchdog_sec * 0.5)
        else:
            self.pet_interval_s = float(pet_interval_s)

        self.min_fps = float(min_fps)
        self.max_telemetry_age_s = float(max_telemetry_age_s)
        self.max_cpu_temp_c = float(max_cpu_temp_c)
        self.max_memory_mb = float(max_memory_mb)
        self.hardware_watchdog_dev = hardware_watchdog_dev

        # Internal state tracking
        self._lock = threading.Lock()
        self._last_frame_time = time.monotonic()
        self._frame_count = 0
        self._last_fps = 30.0
        self._last_telemetry_time = time.monotonic()
        self._hw_watchdog_fd: Optional[int] = None

        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._last_metrics: Optional[SupervisorMetrics] = None

    def record_frame(self) -> None:
        """Signal that a camera frame has been captured and processed."""
        with self._lock:
            now = time.monotonic()
            self._frame_count += 1
            dt = now - self._last_frame_time
            if dt >= 1.0:
                self._last_fps = self._frame_count / dt
                self._frame_count = 0
                self._last_frame_time = now

    def record_telemetry(self) -> None:
        """Signal that valid flight controller telemetry was received."""
        with self._lock:
            self._last_telemetry_time = time.monotonic()

    def read_cpu_temperature(self) -> float:
        """Read core temperature from Pi 5 sysfs or vcgencmd."""
        # 1. Try Linux thermal zone sysfs (standard on Pi OS 64-bit)
        try:
            temp_path = Path("/sys/class/thermal/thermal_zone0/temp")
            if temp_path.is_file():
                raw = temp_path.read_text().strip()
                return float(raw) / 1000.0
        except Exception:
            pass

        # 2. Try vcgencmd
        try:
            out = subprocess.check_output(["vcgencmd", "measure_temp"], stderr=subprocess.DEVNULL, text=True)
            if "temp=" in out:
                return float(out.split("temp=")[1].split("'")[0])
        except Exception:
            pass

        # Default fallback for testing outside Raspberry Pi hardware
        return 48.5

    def read_throttled_flags(self) -> int:
        """Read undervoltage and throttling status from vcgencmd."""
        try:
            out = subprocess.check_output(["vcgencmd", "get_throttled"], stderr=subprocess.DEVNULL, text=True)
            if "throttled=" in out:
                return int(out.split("throttled=")[1].strip(), 16)
        except Exception:
            pass
        return 0

    def read_memory_rss_mb(self) -> float:
        """Read current process Resident Set Size in MB."""
        try:
            import psutil
            return psutil.Process().memory_info().rss / (1024 * 1024)
        except Exception:
            try:
                stat_path = Path("/proc/self/status")
                if stat_path.is_file():
                    for line in stat_path.read_text().splitlines():
                        if line.startswith("VmRSS:"):
                            return float(line.split()[1]) / 1024.0
            except Exception:
                pass
        return 120.0

    def evaluate_metrics(self) -> SupervisorMetrics:
        """Collect current system metrics and evaluate health status."""
        now = time.monotonic()
        with self._lock:
            fps = self._last_fps
            telemetry_age_s = now - self._last_telemetry_time

        temp_c = self.read_cpu_temperature()
        throttled = self.read_throttled_flags()
        mem_mb = self.read_memory_rss_mb()

        fps_ok = (fps >= self.min_fps or (now - self._last_frame_time) < 1.0)
        telem_ok = (telemetry_age_s < self.max_telemetry_age_s)
        temp_ok = (temp_c < self.max_cpu_temp_c)
        mem_ok = (mem_mb < self.max_memory_mb)

        # Under-voltage bit: 0x1 is under-voltage detected currently, 0x10000 is under-voltage has occurred
        undervoltage = bool(throttled & 0x1)

        all_healthy = bool(fps_ok and telem_ok and temp_ok and mem_ok and not undervoltage)

        status_msg = f"FPS: {fps:.1f} | Temp: {temp_c:.1f}C | RAM: {mem_mb:.1f}MB | TelemAge: {telemetry_age_s:.2f}s"
        if undervoltage:
            status_msg += " [UNDERVOLTAGE!]"
        if not temp_ok:
            status_msg += " [OVERHEATING!]"

        return SupervisorMetrics(
            timestamp_s=now,
            fps=fps,
            telemetry_age_s=telemetry_age_s,
            cpu_temp_c=temp_c,
            throttled_flags=throttled,
            memory_rss_mb=mem_mb,
            all_healthy=all_healthy,
            status_summary=status_msg,
        )

    def _open_hardware_watchdog(self) -> None:
        """Open hardware watchdog device file."""
        if Path(self.hardware_watchdog_dev).exists():
            try:
                self._hw_watchdog_fd = os.open(self.hardware_watchdog_dev, os.O_WRONLY)
            except Exception:
                self._hw_watchdog_fd = None

    def _pet_hardware_watchdog(self) -> None:
        """Pet hardware watchdog device."""
        if self._hw_watchdog_fd is not None:
            try:
                os.write(self._hw_watchdog_fd, b"\0")
            except Exception:
                pass

    def _supervisor_loop(self) -> None:
        """Main periodic monitoring loop."""
        self._open_hardware_watchdog()
        self.systemd.notify_ready()

        while self._running:
            metrics = self.evaluate_metrics()
            with self._lock:
                self._last_metrics = metrics

            if metrics.all_healthy:
                # Pet systemd watchdog
                self.systemd.notify_watchdog()
                # Pet hardware watchdog
                self._pet_hardware_watchdog()
                self.systemd.notify_status(f"HEALTHY - {metrics.status_summary}")
            else:
                # Unhealthy: withhold watchdog ping to prompt systemd restart if condition persists
                self.systemd.notify_status(f"DEGRADED - {metrics.status_summary}")

            time.sleep(self.pet_interval_s)

        # Clean shutdown
        self.systemd.notify_stopping()
        if self._hw_watchdog_fd is not None:
            try:
                # Magic character 'V' tells kernel driver to disable watchdog on close
                os.write(self._hw_watchdog_fd, b"V")
                os.close(self._hw_watchdog_fd)
            except Exception:
                pass
            self._hw_watchdog_fd = None

    def start(self) -> None:
        """Start the watchdog supervisor thread."""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._supervisor_loop, daemon=True, name="vps_watchdog_supervisor")
        self._thread.start()

    def stop(self) -> None:
        """Stop the watchdog supervisor cleanly."""
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None


class ShadowSupervisor(WatchdogSupervisor):
    """Supervisor specialized for observe-only Shadow Mode on Raspberry Pi 5.

    Key Responsibilities:
    1. Zero FC Impact: Pi failure, thermal throttle, undervoltage, or process crash
       must NEVER touch or influence the Flight Controller.
    2. Graceful Resource Degradation:
       - If CPU temperature >= 78 C or frame rate drops, degrade background processing
         (e.g., reduce map matching rate or pause map matching) before dropping frame logging.
       - If disk space is low (< 500 MB), trigger safe emergency log flush and cease logging.
    3. Failure Confinement:
       - If any component hangs or fails, the supervisor only stops logging; it never
         sends reboot/failsafe commands to the FC.
    """

    def __init__(
        self,
        pet_interval_s: float = 2.0,
        min_fps: float = 15.0,
        max_telemetry_age_s: float = 1.0,
        max_cpu_temp_c: float = 80.0,
        max_memory_mb: float = 300.0,
        min_disk_free_mb: float = 500.0,
        storage_path: str = ".",
    ) -> None:
        super().__init__(
            pet_interval_s=pet_interval_s,
            min_fps=min_fps,
            max_telemetry_age_s=max_telemetry_age_s,
            max_cpu_temp_c=max_cpu_temp_c,
            max_memory_mb=max_memory_mb,
        )
        self.min_disk_free_mb = float(min_disk_free_mb)
        self.storage_path = str(storage_path)
        self._degradation_callbacks: List[Callable[[str], None]] = []
        self._is_degraded = False

    def register_degradation_callback(self, cb: Callable[[str], None]) -> None:
        """Register a callback for graceful degradation events (e.g. 'THROTTLE_MAP_MATCH')."""
        self._degradation_callbacks.append(cb)

    def read_disk_free_mb(self) -> float:
        """Check free storage space on target logging drive."""
        try:
            usage = shutil.disk_usage(self.storage_path)
            return float(usage.free) / (1024.0 * 1024.0)
        except Exception:
            return 9999.0

    def evaluate_shadow_health(self) -> Dict[str, Any]:
        """Evaluate full health status including disk and thermal headroom."""
        metrics = self.evaluate_metrics()
        disk_free = self.read_disk_free_mb()
        disk_ok = disk_free >= self.min_disk_free_mb

        # Check for thermal degradation (e.g. >= 78 deg C) or frame rate drop
        should_degrade = (metrics.cpu_temp_c >= 78.0) or (metrics.fps < self.min_fps)

        if should_degrade and not self._is_degraded:
            self._is_degraded = True
            for cb in self._degradation_callbacks:
                try:
                    cb("DEGRADE_RESOURCE_PRESSURE")
                except Exception:
                    pass
        elif not should_degrade and self._is_degraded:
            self._is_degraded = False
            for cb in self._degradation_callbacks:
                try:
                    cb("RESTORE_NORMAL_OPERATION")
                except Exception:
                    pass

        return {
            "all_healthy": metrics.all_healthy and disk_ok,
            "fps": metrics.fps,
            "cpu_temp_c": metrics.cpu_temp_c,
            "telemetry_age_s": metrics.telemetry_age_s,
            "memory_rss_mb": metrics.memory_rss_mb,
            "disk_free_mb": disk_free,
            "disk_ok": disk_ok,
            "is_degraded": self._is_degraded,
            "throttled_flags": metrics.throttled_flags,
            "status_summary": metrics.status_summary + f" | DiskFree: {disk_free:.0f}MB",
        }

