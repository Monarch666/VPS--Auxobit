"""Unit tests for Milestone 12: Raspberry Pi 5 Optimization, Watchdog Supervisor, and Packaging."""

from pathlib import Path
import time
import pytest

from vps.health.watchdog import WatchdogSupervisor, SystemdNotifier, SupervisorMetrics
from vps.benchmarks.pi5_profiler import Pi5Profiler, StageBenchmark


class TestMilestone12Packaging:
    def test_systemd_notifier_interface(self):
        """Verify SystemdNotifier functions gracefully in non-systemd and mock socket environments."""
        notifier = SystemdNotifier()
        # In test environment, NOTIFY_SOCKET is usually unset
        assert notifier.is_enabled == (notifier.notify_socket is not None)

        # Notification methods must never raise unhandled exceptions
        assert notifier.notify_ready() in (True, False)
        assert notifier.notify_status("TESTING") in (True, False)
        assert notifier.notify_watchdog() in (True, False)
        assert notifier.notify_stopping() in (True, False)

    def test_watchdog_supervisor_nominal_health(self):
        """Verify WatchdogSupervisor monitors frames, telemetry, and evaluates nominal health."""
        supervisor = WatchdogSupervisor(pet_interval_s=0.5, min_fps=10.0, max_telemetry_age_s=2.0)
        supervisor.start()

        try:
            # Simulate active pipeline
            for _ in range(15):
                supervisor.record_frame()
                supervisor.record_telemetry()
                time.sleep(0.02)

            metrics: SupervisorMetrics = supervisor.evaluate_metrics()
            assert metrics.telemetry_age_s < 1.0
            assert metrics.cpu_temp_c > 0.0
            assert metrics.memory_rss_mb > 0.0
            assert metrics.all_healthy is True
            assert "FPS:" in metrics.status_summary
        finally:
            supervisor.stop()

    def test_watchdog_stale_telemetry_detection(self):
        """Verify supervisor detects stalled telemetry and flags degraded status."""
        supervisor = WatchdogSupervisor(pet_interval_s=0.2, min_fps=5.0, max_telemetry_age_s=0.05)
        supervisor.record_frame()
        supervisor.record_telemetry()

        # Sleep longer than max telemetry age
        time.sleep(0.15)
        metrics = supervisor.evaluate_metrics()

        assert metrics.telemetry_age_s >= 0.10
        # Telemetry is stale, so all_healthy must be False
        assert metrics.all_healthy is False

    def test_pi5_profiler_harness_budget(self):
        """Verify Pi 5 profiling harness executes and validates against performance budget."""
        profiler = Pi5Profiler()

        # Run quick iterations
        t_ingest = profiler.benchmark_frame_ingest_and_resize(n_iterations=5)
        t_derot = profiler.benchmark_gyro_derotation(n_iterations=10)
        t_ekf = profiler.benchmark_es_ekf_strapdown(n_iterations=20)
        t_mav = profiler.benchmark_mavlink_serialization(n_iterations=20)

        assert len(t_ingest) == 5
        assert len(t_derot) == 10
        assert len(t_ekf) == 20
        assert len(t_mav) == 20
        assert sum(t_ekf) / len(t_ekf) < 5.0  # EKF step < 5 ms

    def test_systemd_unit_file_configuration(self):
        """Verify production vps.service unit file adheres to systemd specifications."""
        service_file = Path("packaging/vps.service")
        assert service_file.is_file(), f"Missing systemd service file at {service_file}"

        content = service_file.read_text(encoding="utf-8")
        assert "[Unit]" in content
        assert "[Service]" in content
        assert "[Install]" in content
        assert "Type=notify" in content
        assert "WatchdogSec=" in content
        assert "CPUAffinity=1,2,3" in content
        assert "Nice=-10" in content
        assert "Restart=on-failure" in content
        assert "WantedBy=multi-user.target" in content

    def test_install_script_validity(self):
        """Verify packaging/install.sh exists and contains required Raspberry Pi OS instructions."""
        script_file = Path("packaging/install.sh")
        assert script_file.is_file(), f"Missing installer script at {script_file}"

        content = script_file.read_text(encoding="utf-8")
        assert "#!/usr/bin/env bash" in content
        assert "dtoverlay=imx296" in content
        assert "enable_uart=1" in content
        assert "vps.service" in content
        assert "systemctl enable vps.service" in content

