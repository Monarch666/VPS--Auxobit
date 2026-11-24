"""Unit tests for offline session replayer and numerical equivalence verifier."""

import json
from pathlib import Path
import numpy as np
import pytest

from vps.tools.session_recorder import ShadowSessionRecorder, ShadowTelemetryRecord
from vps.tools.replay import SessionReplayer, ReplayEquivalenceStats


def test_session_replayer_equivalence(tmp_path):
    """Verify session replay achieves numerical equivalence on recorded telemetry."""
    from vps.config import VPSConfig
    from vps.fusion.fusion_engine import VPSFusionEngine

    session_dir = tmp_path / "session_test_replay"
    cfg = VPSConfig(mode="shadow")
    cfg.camera.baseline_ground_offset_m = 0.06

    recorder = ShadowSessionRecorder(
        output_dir=tmp_path,
        session_name="session_test_replay",
        save_images=False,
        config_snapshot={"mode": "shadow", "baseline": 0.06},
    )

    live_fusion = VPSFusionEngine(config=cfg, initial_height_m=0.06)
    dummy_frame = np.full((cfg.camera.height, cfg.camera.width), 128, dtype=np.uint8)

    # Record 10 sequential frames
    for i in range(10):
        t = 100.0 + i * 0.033
        h_baro = 5.0 + i * 0.1
        f_out = live_fusion.process_telemetry_and_frame(
            frame=dummy_frame,
            timestamp_s=t,
            pos_ned_m=np.array([0.0, 0.0, -h_baro]),
            vel_ned_mps=np.array([0.0, 0.0, -0.1]),
            attitude_rpy_rad=np.zeros(3),
            gyro_rates_rad_s=np.zeros(3),
            baro_alt_m=h_baro,
            frame_idx=i,
        )
        rec = ShadowTelemetryRecord(
            timestamp_s=t,
            is_armed=True,
            vps_height_agl_m=float(f_out.height_agl_m),
            vps_confidence=float(f_out.confidence),
            fc_baro_alt_m=h_baro,
            baro_alt_m=h_baro,
            fc_pos_d_m=-h_baro,
            fc_vel_d_mps=-0.1,
            roll_rad=0.0,
            pitch_rad=0.0,
            yaw_rad=0.0,
        )
        recorder.log_record_and_frame(rec)

    recorder.close()

    assert (session_dir / "telemetry.csv").is_file()

    # Run Replayer
    replayer = SessionReplayer(session_dir=session_dir, config=cfg)
    records, stats = replayer.run_replay()

    assert stats.total_frames_evaluated == 10
    # Equivalence should pass
    assert stats.is_equivalent is True
    assert stats.height_max_diff_m <= 0.05

