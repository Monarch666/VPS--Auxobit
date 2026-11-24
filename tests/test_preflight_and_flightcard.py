"""Unit tests for preflight checklist and flight card generator."""

from pathlib import Path
import json
import pytest

from vps.config import VPSConfig
from vps.tools.preflight import PreflightChecker
from vps.tools.flightcard import generate_flight_cards, render_markdown


def test_preflight_checker_mock(tmp_path):
    """Verify preflight checker passes under nominal mock setup."""
    cfg = VPSConfig(mode="shadow")
    cfg.camera.device_index = "mock"
    cfg.mavlink.feed_fc_ekf = False
    cfg.camera.baseline_ground_offset_m = 0.06

    checker = PreflightChecker(config=cfg)
    passed = checker.run_all_checks()
    assert passed is True

    report_path = tmp_path / "preflight_test.json"
    checker.export_report(report_path)
    assert report_path.is_file()

    data = json.loads(report_path.read_text(encoding="utf-8"))
    assert data["all_critical_passed"] is True
    check_names = [c["name"] for c in data["checks"]]
    assert "Safety Mode & Interlocks" in check_names
    assert "Camera Intrinsics Calibration" in check_names
    assert "Camera Stream & Monotonicity" in check_names


def test_preflight_fails_on_illegal_config():
    """Verify preflight fails if mode is not shadow."""
    cfg = VPSConfig(mode="flight")
    checker = PreflightChecker(config=cfg)
    checker._check_safety_invariants()
    result = [r for r in checker.results if r.name == "Safety Mode & Interlocks"][0]
    assert result.passed is False


def test_flightcard_generation(tmp_path):
    """Verify flightcard generator creates valid cards and markdown."""
    cards = generate_flight_cards()
    assert "1" in cards
    assert "2" in cards
    assert "3" in cards

    card1 = cards["1"]
    assert card1.flight_id == "FLIGHT-01"
    assert card1.max_altitude_m == 15.0
    assert len(card1.steps) == 5

    md = render_markdown(card1)
    assert "FLIGHT-01" in md
    assert "0.06m" in md
    assert "19.808798 N, 75.283380 E" in md

