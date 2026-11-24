"""Unit tests for ShadowSessionRecorder, pre-roll buffer, and async frame writer."""

import json
import time
from pathlib import Path
import numpy as np
import pytest

from vps.tools.session_recorder import (
    ShadowSessionRecorder,
    ShadowTelemetryRecord,
    AsynchronousFrameWriter,
)


def test_async_frame_writer(tmp_path):
    """Verify asynchronous frame writer encodes and saves JPEGs."""
    images_dir = tmp_path / "images"
    writer = AsynchronousFrameWriter(images_dir=images_dir, max_queue_size=10, jpeg_quality=80)
    writer.start()

    dummy_frame = (np.random.rand(100, 100) * 255).astype(np.uint8)

    for i in range(5):
        rel_path = writer.submit_frame(frame_idx=i, timestamp_s=time.monotonic(), frame=dummy_frame)
        assert rel_path is not None
        assert "images/frame_" in rel_path

    writer.stop()
    assert writer.written_frames == 5
    assert writer.dropped_frames == 0
    saved_files = list(images_dir.glob("*.jpg"))
    assert len(saved_files) == 5


def test_async_frame_writer_drops_when_queue_full(tmp_path):
    """Verify queue overflow reports dropped frames and doesn't block caller."""
    images_dir = tmp_path / "images"
    # Small queue size
    writer = AsynchronousFrameWriter(images_dir=images_dir, max_queue_size=2)
    # Don't start worker thread immediately to simulate extreme disk stall

    dummy_frame = np.zeros((50, 50), dtype=np.uint8)
    res1 = writer.submit_frame(1, 1.0, dummy_frame)
    res2 = writer.submit_frame(2, 2.0, dummy_frame)
    res3 = writer.submit_frame(3, 3.0, dummy_frame)  # Should overflow

    assert res1 is not None
    assert res2 is not None
    assert res3 is None
    assert writer.dropped_frames >= 1

    # Now drain
    writer.start()
    writer.stop()


def test_session_recorder_preroll_and_arming_trigger(tmp_path):
    """Verify 10s pre-roll buffer buffers when disarmed and drains when armed."""
    recorder = ShadowSessionRecorder(
        output_dir=tmp_path,
        session_name="test_flight",
        save_images=True,
        pre_roll_duration_s=2.0,
        post_roll_duration_s=1.0,
    )

    dummy_frame = np.ones((80, 80), dtype=np.uint8) * 128

    # 1. Feed 5 frames while DISARMED
    for i in range(5):
        rec = ShadowTelemetryRecord(
            timestamp_s=time.monotonic(),
            is_armed=False,
            vps_height_agl_m=0.06,
        )
        recorder.log_record_and_frame(rec, dummy_frame)

    # In disarmed state, no frames should have been committed to CSV yet
    assert recorder.is_recording is False
    assert recorder.frame_count == 0

    # 2. Feed 1 frame with ARMED = True
    rec_armed = ShadowTelemetryRecord(
        timestamp_s=time.monotonic(),
        is_armed=True,
        vps_height_agl_m=1.5,
    )
    recorder.log_record_and_frame(rec_armed, dummy_frame)

    # Now recording must be active, and all 5 pre-roll frames + 1 armed frame drained (total 6)
    assert recorder.is_recording is True
    assert recorder.frame_count == 6

    # 3. Close and verify CSV and metadata
    recorder.close()

    csv_path = tmp_path / "test_flight" / "telemetry.csv"
    assert csv_path.is_file()
    lines = csv_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 7  # header + 6 records

    meta_path = tmp_path / "test_flight" / "session_meta.json"
    assert meta_path.is_file()
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    assert meta["session_id"] == "test_flight"
    assert meta["status"] == "COMPLETED"
    assert meta["ground_baseline_offset_m"] == 0.06
    assert meta["storage"]["total_frames_recorded"] == 6


def test_session_recorder_disarm_postroll(tmp_path):
    """Verify disarming keeps recording for post_roll_duration_s before closing."""
    recorder = ShadowSessionRecorder(
        output_dir=tmp_path,
        session_name="test_disarm",
        save_images=False,
        post_roll_duration_s=0.2,
    )

    t0 = time.monotonic()
    # Arm
    recorder.log_record_and_frame(ShadowTelemetryRecord(timestamp_s=t0, is_armed=True))
    assert recorder.is_recording is True

    # Disarm immediately
    recorder.log_record_and_frame(ShadowTelemetryRecord(timestamp_s=t0 + 0.05, is_armed=False))
    assert recorder.is_recording is True  # Still recording during post-roll

    # Exceed post-roll duration
    recorder.log_record_and_frame(ShadowTelemetryRecord(timestamp_s=t0 + 0.30, is_armed=False))
    assert recorder.is_recording is False  # Stopped

    recorder.close()

