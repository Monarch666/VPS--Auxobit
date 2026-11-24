"""Unit tests for offline session replayer and numerical equivalence verifier."""

import json
from pathlib import Path
import numpy as np
import pytest

from vps.tools.session_recorder import ShadowSessionRecorder, ShadowTelemetryRecord
from vps.tools.replay import SessionReplayer, ReplayEquivalenceStats


def test_session_replayer_equivalence(tmp_path):
    """Verify session replay achieves numerical equivalence on recorded telemetry."""
    session_dir = tmp_path / "session_test_replay"
    recorder = ShadowSessionRecorder(
        output_dir=tmp_path,
        session_name="session_test_replay",
        save_images=False,
    )

    # Record 10 sequential frames
    for i in range(10):
        t = 100.0 + i * 0.033
        rec = ShadowTelemetryRecord(
            timestamp_s=t,
            is_armed=True,
            vps_height_agl_m=5.0 + i * 0.1,
            vps_confidence=0.95,
            fc_baro_alt_m=5.0 + i * 0.1,
            baro_alt_m=5.0 + i * 0.1,
            roll_rad=0.0,
            pitch_rad=0.0,
            yaw_rad=0.0,
        )
        recorder.log_record_and_frame(rec)

    recorder.close()

    assert (session_dir / "telemetry.csv").is_file()

    # Run Replayer
    replayer = SessionReplayer(session_dir=session_dir)
    records, stats = replayer.run_replay()

    assert stats.total_frames_evaluated == 10
    # Equivalence should pass
    assert stats.is_equivalent is True
    assert stats.height_max_diff_m <= 0.05
