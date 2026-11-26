"""Unit tests for Phase 1 & Phase 2 shadow evaluation and HTML report generator."""

from pathlib import Path
import json
import pytest

from vps.tools.session_recorder import ShadowSessionRecorder, ShadowTelemetryRecord
from vps.tools.evaluate import ShadowEvaluator


def test_shadow_evaluator_metrics_and_html(tmp_path):
    """Verify evaluation metrics and HTML report generation on recorded flight session."""
    session_dir = tmp_path / "session_eval_test"
    recorder = ShadowSessionRecorder(
        output_dir=tmp_path,
        session_name="session_eval_test",
        save_images=False,
    )

    # Generate 50 frames of climb and hover
    for i in range(50):
        t = 1000.0 + i * 0.1
        h_true = 5.0 + i * 0.2  # 5m to 15m
        vps_h = h_true + 0.05   # 5 cm bias error
        rec = ShadowTelemetryRecord(
            timestamp_s=t,
            is_armed=True,
            vps_height_agl_m=vps_h,
            vps_confidence=0.92,
            fc_pos_d_m=-(h_true - 0.06),
            fc_baro_alt_m=h_true - 0.06,
            baro_alt_m=h_true - 0.06,
            gps_alt_m=h_true + 500.0,
            fc_pos_n_m=float(i * 0.5),
            fc_pos_e_m=0.0,
            vps_pos_n_m=float(i * 0.5 + 0.1),
            vps_pos_e_m=0.05,
            fc_vel_n_mps=5.0,
            vps_vel_n_mps=4.95,
        )
        recorder.log_record_and_frame(rec)

    recorder.close()

    evaluator = ShadowEvaluator(session_dir=session_dir, baseline_offset_m=0.06)

    # 1. Phase 1 Evaluation
    p1 = evaluator.evaluate_phase1()
    assert p1.total_samples == 50
    assert p1.overall_mean_abs_error_m < 0.2  # Expected ~0.05m
    assert p1.overall_mean_rel_pct < 5.0     # Below 8% threshold
    assert p1.pass_status is True
    assert len(p1.bands) >= 2

    # 2. Phase 2 Evaluation
    p2 = evaluator.evaluate_phase2()
    assert p2.total_samples == 50
    assert p2.pos_2d_rmse_m < 0.5
    assert p2.cep_95_m < 1.0
    assert p2.map_match_attempts == 0
    assert p2.map_match_accepted == 0
    assert p2.pass_status is True

    # 3. Virtual Denial
    denial_runs = evaluator.run_virtual_gnss_denial(denial_durations_s=[2.0])
    assert len(denial_runs) >= 1
    assert denial_runs[0].final_drift_m >= 0.0

    # 4. HTML Report
    html_path = session_dir / "report.html"
    res_path = evaluator.generate_html_report(html_path, p1, p2)
    assert res_path.is_file()
    content = html_path.read_text(encoding="utf-8")
    assert "VPS Shadow Mode - Flight Evaluation Report" in content
    assert "Phase 1 Height RMSE" in content
    assert "Phase 2: Post-Flight Virtual GNSS Denial Test" in content
    assert "data:image/png;base64," in content
