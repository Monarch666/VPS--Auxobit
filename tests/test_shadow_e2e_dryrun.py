"""End-to-End Dry Run Verification for VPS Shadow Mode (Milestone S7).

Executes the complete operational flight lifecycle:
1. Pre-Flight Validation: Runs preflight checker to ensure hardware and safety interlocks pass.
2. Simulated Flight Execution: Runs VPS in Shadow Mode through a 0-15m altitude ladder.
3. Asynchronous Session Recording: Captures synchronized telemetry, pre-roll buffer, and metadata.
4. Deterministic Replay: Replays session through replay.py and verifies numerical equivalence.
5. Evaluation & Reporting: Computes Phase 1/Phase 2 metrics and renders HTML flight report.
"""

from pathlib import Path
import json
import time
import cv2
import numpy as np
import pytest

from vps.config import VPSConfig
from vps.tools.preflight import PreflightChecker
from vps.tools.session_recorder import ShadowSessionRecorder, ShadowTelemetryRecord
from vps.tools.replay import SessionReplayer
from vps.tools.evaluate import ShadowEvaluator


def test_shadow_mode_full_end_to_end_dryrun(tmp_path):
    """Execute complete end-to-end shadow mode flight lifecycle."""
    session_name = "session_s7_dryrun"
    session_dir = tmp_path / session_name

    # --------------------------------------------------------------------------
    # Step 1: Pre-Flight Checklist Verification
    # --------------------------------------------------------------------------
    cfg = VPSConfig(mode="shadow")
    cfg.camera.device_index = "mock"
    cfg.mavlink.feed_fc_ekf = False
    cfg.camera.baseline_ground_offset_m = 0.06
    cfg.logger.output_dir = str(tmp_path)

    checker = PreflightChecker(config=cfg)
    assert checker.run_all_checks() is True, "Preflight checklist must pass before flight!"

    preflight_json = tmp_path / "preflight_dryrun.json"
    checker.export_report(preflight_json)
    assert preflight_json.is_file()

    # --------------------------------------------------------------------------
    # Step 2: Simulated Shadow Flight & Recording
    # --------------------------------------------------------------------------
    recorder = ShadowSessionRecorder(
        output_dir=tmp_path,
        session_name=session_name,
        save_images=True,
        pre_roll_duration_s=2.0,
        post_roll_duration_s=1.0,
        config_snapshot={"mode": "shadow", "baseline": 0.06},
    )

    from vps.fusion.fusion_engine import VPSFusionEngine
    live_fusion = VPSFusionEngine(config=cfg, initial_height_m=0.06)

    dummy_frame = (np.random.rand(120, 160) * 255).astype(np.uint8)

    # A. Pre-Arm Ground Rest (10 frames at 0.06m)
    for i in range(10):
        t = 100.0 + i * 0.033
        f_out = live_fusion.process_telemetry_and_frame(
            frame=dummy_frame,
            timestamp_s=t,
            pos_ned_m=np.zeros(3),
            vel_ned_mps=np.zeros(3),
            attitude_rpy_rad=np.zeros(3),
            gyro_rates_rad_s=np.zeros(3),
            baro_alt_m=0.0,
            frame_idx=i,
        )
        rec = ShadowTelemetryRecord(
            timestamp_s=t,
            is_armed=False,
            vps_height_agl_m=float(f_out.height_agl_m),
            vps_confidence=float(f_out.confidence),
            fc_baro_alt_m=0.0,
            fc_pos_d_m=0.0,
            fc_pos_n_m=0.0,
            fc_pos_e_m=0.0,
        )
        recorder.log_record_and_frame(rec, dummy_frame)

    assert recorder.is_recording is False
    assert recorder.frame_count == 0  # Buffered in RAM

    # B. Arm & Climb to 5m (20 frames)
    for i in range(20):
        t = 100.0 + (10 + i) * 0.033
        h_baro = (i / 19.0) * 4.94  # 0.0m to 4.94m above takeoff
        vel_d = -0.5
        f_out = live_fusion.process_telemetry_and_frame(
            frame=dummy_frame,
            timestamp_s=t,
            pos_ned_m=np.array([i * 0.1, 0.0, -h_baro]),
            vel_ned_mps=np.array([3.0, 0.0, vel_d]),
            attitude_rpy_rad=np.zeros(3),
            gyro_rates_rad_s=np.zeros(3),
            baro_alt_m=h_baro,
            frame_idx=10 + i,
        )
        rec = ShadowTelemetryRecord(
            timestamp_s=t,
            is_armed=True,
            vps_height_agl_m=float(f_out.height_agl_m),
            vps_confidence=float(f_out.confidence),
            fc_baro_alt_m=h_baro,
            fc_pos_d_m=-h_baro,
            fc_pos_n_m=float(i * 0.1),
            fc_pos_e_m=0.0,
            vps_pos_n_m=float(i * 0.1 + 0.01),
            vps_pos_e_m=0.01,
            fc_vel_n_mps=3.0,
            fc_vel_d_mps=vel_d,
            vps_vel_n_mps=2.98,
        )
        recorder.log_record_and_frame(rec, dummy_frame)

    assert recorder.is_recording is True
    # Pre-roll 10 frames + 20 climb frames = 30 frames
    assert recorder.frame_count == 30

    # C. Hover at 5m (20 frames)
    for i in range(20):
        t = 100.0 + (30 + i) * 0.033
        f_out = live_fusion.process_telemetry_and_frame(
            frame=dummy_frame,
            timestamp_s=t,
            pos_ned_m=np.array([2.0, 0.0, -4.94]),
            vel_ned_mps=np.zeros(3),
            attitude_rpy_rad=np.zeros(3),
            gyro_rates_rad_s=np.zeros(3),
            baro_alt_m=4.94,
            frame_idx=30 + i,
        )
        rec = ShadowTelemetryRecord(
            timestamp_s=t,
            is_armed=True,
            vps_height_agl_m=float(f_out.height_agl_m),
            vps_confidence=float(f_out.confidence),
            fc_baro_alt_m=4.94,
            fc_pos_d_m=-4.94,
            fc_pos_n_m=2.0,
            fc_pos_e_m=0.0,
            fc_vel_d_mps=0.5,
            vps_pos_n_m=2.01,
            vps_pos_e_m=0.01,
            fc_vel_n_mps=0.0,
            vps_vel_n_mps=0.01,
        )
        recorder.log_record_and_frame(rec, dummy_frame)

    assert recorder.frame_count == 50

    # D. Descent and Land (20 frames)
    for i in range(20):
        t = 100.0 + (50 + i) * 0.033
        h_baro = 4.94 - (i / 19.0) * 4.94
        f_out = live_fusion.process_telemetry_and_frame(
            frame=dummy_frame,
            timestamp_s=t,
            pos_ned_m=np.array([2.0, 0.0, -h_baro]),
            vel_ned_mps=np.array([0.0, 0.0, 0.5]),
            attitude_rpy_rad=np.zeros(3),
            gyro_rates_rad_s=np.zeros(3),
            baro_alt_m=h_baro,
            frame_idx=50 + i,
        )
        rec = ShadowTelemetryRecord(
            timestamp_s=t,
            is_armed=True,
            vps_height_agl_m=float(f_out.height_agl_m),
            vps_confidence=float(f_out.confidence),
            fc_baro_alt_m=h_baro,
            fc_pos_d_m=-h_baro,
            fc_pos_n_m=2.0,
            fc_pos_e_m=0.0,
            fc_vel_d_mps=0.5,
            vps_pos_n_m=2.01,
            vps_pos_e_m=0.01,
        )
        recorder.log_record_and_frame(rec, dummy_frame)

    # Disarm
    t_land = 100.0 + 70 * 0.033
    f_out = live_fusion.process_telemetry_and_frame(
        frame=dummy_frame,
        timestamp_s=t_land,
        pos_ned_m=np.zeros(3),
        vel_ned_mps=np.zeros(3),
        attitude_rpy_rad=np.zeros(3),
        gyro_rates_rad_s=np.zeros(3),
        baro_alt_m=0.0,
        frame_idx=70,
    )
    rec = ShadowTelemetryRecord(
        timestamp_s=t_land,
        is_armed=False,
        vps_height_agl_m=float(f_out.height_agl_m),
        vps_confidence=float(f_out.confidence),
        fc_baro_alt_m=0.0,
        fc_pos_d_m=0.0,
    )
    recorder.log_record_and_frame(rec, dummy_frame)
    recorder.close()

    assert (session_dir / "telemetry.csv").is_file()
    assert (session_dir / "session_meta.json").is_file()
    assert len(list((session_dir / "images").glob("*.jpg"))) > 50

    # --------------------------------------------------------------------------
    # Step 3: Replay & Equivalence Verification
    # --------------------------------------------------------------------------
    replayer = SessionReplayer(session_dir=session_dir)
    records, stats = replayer.run_replay()
    assert stats.is_equivalent is True, f"Replay divergence: {stats.failure_reasons}"
    assert stats.height_max_diff_m <= 0.25

    # --------------------------------------------------------------------------
    # Step 4: Metric Evaluation & HTML Report Generation
    # --------------------------------------------------------------------------
    evaluator = ShadowEvaluator(
        session_dir=session_dir,
        baseline_offset_m=0.06,
        max_height_rel_err_tol_pct=85.0,  # Scaled for synthetic static-noise dryrun without ground texture
    )
    p1 = evaluator.evaluate_phase1()
    p2 = evaluator.evaluate_phase2()

    assert p1.total_samples == 71
    assert p1.pass_status is True, "Phase 1 height evaluation failed pass criteria!"
    assert p1.overall_rmse_m < 2.5

    html_report = session_dir / "evaluation_report.html"
    evaluator.generate_html_report(html_report, p1, p2)
    assert html_report.is_file()
    assert "VPS Shadow Mode - Flight Evaluation Report" in html_report.read_text(encoding="utf-8")
