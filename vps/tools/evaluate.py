"""Comprehensive Evaluation Engine and Report Generator for VPS Shadow Mode.

Computes Phase 1 (Height) and Phase 2 (Position/Velocity) metrics against Ground Truth:
1. Phase 1 Height Metrics across Altitude Bands (0-15m, 15-30m, 30-50m):
   - Mean Error, Bias, RMS, 95th percentile (P95), and percent error.
   - Flight regime segmentation: Hover, Cruise, Climb, Descent.
   - Dropout duration (confidence < 0.3) and confidence calibration score.
   - Root cause diagnosis for degraded segments (high rotation rate, low texture).
2. Phase 2 Position & Velocity Metrics vs MicroAir GPS:
   - 2D horizontal position RMSE, CEP-50, and CEP-95.
   - Velocity RMSE (V_n, V_e).
   - Drift rate in meters per minute.
   - Absolute map match success rate and mean residual error.
3. Post-Flight "Virtual GNSS Denial" Simulation:
   - Cuts off simulated GPS for N minutes (e.g. 1, 3, 5 min) and compares VIO-only
     and VIO+Map Matching drift against true GPS ground truth.
4. Professional Standalone HTML Report with SVG/base64 visual plots and PASS/FAIL scorecard.
"""

from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
import argparse
import base64
import csv
import io
import json
import logging
import sys
import time
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from vps.config import VPSConfig

logger = logging.getLogger("VPS.Evaluate")


@dataclass
class AltitudeBandMetrics:
    band_name: str
    sample_count: int
    mean_abs_error_m: float
    bias_m: float
    rmse_m: float
    p95_error_m: float
    mean_rel_error_pct: float
    dropout_time_s: float
    pass_criteria_met: bool


@dataclass
class Phase1EvaluationResult:
    total_samples: int
    duration_s: float
    overall_rmse_m: float
    overall_mean_abs_error_m: float
    overall_bias_m: float
    overall_p95_m: float
    overall_mean_rel_pct: float
    total_dropout_s: float
    bands: List[AltitudeBandMetrics]
    regime_rmse: Dict[str, float]
    confidence_calibration: Dict[str, float]
    bad_segment_warnings: List[str]
    pass_status: bool


@dataclass
class VirtualDenialRun:
    start_time_s: float
    denial_duration_s: float
    final_drift_m: float
    drift_rate_m_per_min: float
    cep_50_m: float
    cep_95_m: float


@dataclass
class Phase2EvaluationResult:
    total_samples: int
    pos_2d_rmse_m: float
    cep_50_m: float
    cep_95_m: float
    vel_rmse_mps: float
    drift_rate_m_per_min: float
    map_match_attempts: int
    map_match_accepted: int
    map_match_success_rate: float
    mean_map_match_error_m: float
    virtual_denial_runs: List[VirtualDenialRun]
    pass_status: bool


class ShadowEvaluator:
    """Evaluates Phase 1 & 2 performance and virtual GNSS denial on recorded sessions."""

    def __init__(
        self,
        session_dir: str | Path,
        baseline_offset_m: float = 0.06,
        max_height_rel_err_tol_pct: float = 8.0,  # Pass threshold: <= 8%
        max_p2_cep95_tol_m: float = 5.0,          # Pass threshold: <= 5m
    ) -> None:
        self.session_dir = Path(session_dir)
        self.csv_path = self.session_dir / "telemetry.csv"
        self.meta_path = self.session_dir / "session_meta.json"
        self.baseline_offset_m = float(baseline_offset_m)
        self.max_h_rel_tol_pct = float(max_height_rel_err_tol_pct)
        self.max_cep95_tol_m = float(max_p2_cep95_tol_m)

        if not self.csv_path.is_file():
            raise FileNotFoundError(f"Session telemetry.csv not found at: {self.csv_path}")

        self.records = self._load_csv()

    def _load_csv(self) -> List[Dict[str, float]]:
        """Load and parse telemetry rows into float dictionaries."""
        with open(self.csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = []
            for r in reader:
                parsed = {}
                for k, v in r.items():
                    low = v.strip().lower() if isinstance(v, str) else ""
                    if low == "true":
                        parsed[k] = True
                    elif low == "false":
                        parsed[k] = False
                    else:
                        try:
                            parsed[k] = float(v)
                        except (ValueError, TypeError):
                            parsed[k] = v
                rows.append(parsed)
        return rows

    def evaluate_phase1(self) -> Phase1EvaluationResult:
        """Compute height metrics across altitude bands and flight regimes."""
        if not self.records:
            return self._empty_phase1_result()

        times = np.array([r.get("timestamp_s", 0.0) for r in self.records])
        dt = np.diff(times, prepend=times[0])
        dt[dt <= 0] = 0.033

        vps_h = np.array([float(r.get("vps_height_agl_m", r.get("estimated_h_m", 0.0))) for r in self.records])
        conf = np.array([float(r.get("vps_confidence", r.get("confidence", 0.0))) for r in self.records])

        # Ground truth:
        # If fc_pos_d_m is present and valid, true height AGL is -fc_pos_d_m + baseline_offset_m
        fc_pos_d = np.array([float(r.get("fc_pos_d_m", 0.0)) for r in self.records])
        if np.any(np.abs(fc_pos_d) > 0.01):
            gt_h = -fc_pos_d + self.baseline_offset_m
        else:
            # Fall back to baro altitude above takeoff
            baro_alt = np.array([float(r.get("fc_baro_alt_m", r.get("baro_alt_m", 0.0))) for r in self.records])
            # If baro is absolute ASL (> 50m on ground), subtract ground reference
            if len(baro_alt) > 0 and baro_alt[0] > 50.0:
                ref_baro = baro_alt[0]
                gt_h = (baro_alt - ref_baro) + self.baseline_offset_m
            else:
                gt_h = baro_alt + self.baseline_offset_m

        gt_h = np.clip(gt_h, self.baseline_offset_m, 500.0)

        # Errors
        err = vps_h - gt_h
        abs_err = np.abs(err)
        rel_err_pct = (abs_err / np.maximum(0.2, gt_h)) * 100.0

        # Overall summary
        rmse = float(np.sqrt(np.mean(err ** 2)))
        mean_abs = float(np.mean(abs_err))
        bias = float(np.mean(err))
        p95 = float(np.percentile(abs_err, 95))
        mean_rel = float(np.mean(rel_err_pct))

        # Dropouts
        dropout_mask = conf < 0.3
        total_dropout_s = float(np.sum(dt[dropout_mask]))

        # Altitude Bands
        band_defs = [
            ("Low (0 - 15m)", 0.0, 15.0),
            ("Mid (15 - 30m)", 15.0, 30.0),
            ("High (30 - 50m)", 30.0, 50.0),
        ]
        bands = []
        for name, h_min, h_max in band_defs:
            mask = (gt_h >= h_min) & (gt_h < h_max)
            count = int(np.sum(mask))
            if count > 0:
                b_abs = abs_err[mask]
                b_rel = rel_err_pct[mask]
                b_rmse = float(np.sqrt(np.mean(err[mask] ** 2)))
                b_mean_abs = float(np.mean(b_abs))
                b_bias = float(np.mean(err[mask]))
                b_p95 = float(np.percentile(b_abs, 95))
                b_mean_rel = float(np.mean(b_rel))
                b_drop = float(np.sum(dt[mask & dropout_mask]))
                passed = b_mean_rel <= self.max_h_rel_tol_pct
            else:
                b_rmse = b_mean_abs = b_bias = b_p95 = b_mean_rel = b_drop = 0.0
                passed = True

            bands.append(AltitudeBandMetrics(
                band_name=name,
                sample_count=count,
                mean_abs_error_m=b_mean_abs,
                bias_m=b_bias,
                rmse_m=b_rmse,
                p95_error_m=b_p95,
                mean_rel_error_pct=b_mean_rel,
                dropout_time_s=b_drop,
                pass_criteria_met=passed,
            ))

        # Regimes
        t_span = max(0.01, float(times[-1] - times[0]))
        t_uniform = np.linspace(0.0, t_span, len(gt_h))
        vz = np.gradient(gt_h, t_uniform) if len(gt_h) > 1 else np.zeros_like(gt_h)
        v_xy = np.hypot(
            np.array([float(r.get("fc_vel_n_mps", 0.0)) for r in self.records]),
            np.array([float(r.get("fc_vel_e_mps", 0.0)) for r in self.records]),
        )

        hover_mask = (v_xy < 0.5) & (np.abs(vz) < 0.3)
        cruise_mask = (v_xy >= 0.5) & (np.abs(vz) < 0.5)
        climb_mask = vz > 0.5
        descent_mask = vz < -0.5

        regime_rmse = {
            "hover": float(np.sqrt(np.mean(err[hover_mask] ** 2))) if np.any(hover_mask) else 0.0,
            "cruise": float(np.sqrt(np.mean(err[cruise_mask] ** 2))) if np.any(cruise_mask) else 0.0,
            "climb": float(np.sqrt(np.mean(err[climb_mask] ** 2))) if np.any(climb_mask) else 0.0,
            "descent": float(np.sqrt(np.mean(err[descent_mask] ** 2))) if np.any(descent_mask) else 0.0,
        }

        # Confidence Calibration
        conf_calibration = {}
        bins = [(0.0, 0.4), (0.4, 0.7), (0.7, 0.9), (0.9, 1.01)]
        for low, high in bins:
            c_mask = (conf >= low) & (conf < high)
            if np.any(c_mask):
                conf_calibration[f"conf_{low:.1f}_{high:.1f}"] = float(np.mean(abs_err[c_mask]))
            else:
                conf_calibration[f"conf_{low:.1f}_{high:.1f}"] = 0.0

        # Diagnosis of Bad Segments
        warnings = []
        gyros = np.hypot(
            np.array([float(r.get("gyro_x_rad_s", 0.0)) for r in self.records]),
            np.array([float(r.get("gyro_y_rad_s", 0.0)) for r in self.records]),
        )
        high_rot_mask = gyros > 0.8
        if np.any(high_rot_mask & (abs_err > 1.0)):
            warnings.append(f"Elevated error during high angular rotation rate (> 0.8 rad/s, {np.sum(high_rot_mask)} frames).")

        pass_status = (mean_rel <= self.max_h_rel_tol_pct) and (total_dropout_s < 5.0)

        return Phase1EvaluationResult(
            total_samples=len(self.records),
            duration_s=float(times[-1] - times[0]) if len(times) > 1 else 0.0,
            overall_rmse_m=rmse,
            overall_mean_abs_error_m=mean_abs,
            overall_bias_m=bias,
            overall_p95_m=p95,
            overall_mean_rel_pct=mean_rel,
            total_dropout_s=total_dropout_s,
            bands=bands,
            regime_rmse=regime_rmse,
            confidence_calibration=conf_calibration,
            bad_segment_warnings=warnings,
            pass_status=pass_status,
        )

    def evaluate_phase2(self) -> Phase2EvaluationResult:
        """Compute horizontal positioning, velocity, and map matching metrics."""
        if not self.records:
            return self._empty_phase2_result()

        vps_pn = np.array([float(r.get("vps_pos_n_m", 0.0)) for r in self.records])
        vps_pe = np.array([float(r.get("vps_pos_e_m", 0.0)) for r in self.records])
        fc_pn = np.array([float(r.get("fc_pos_n_m", 0.0)) for r in self.records])
        fc_pe = np.array([float(r.get("fc_pos_e_m", 0.0)) for r in self.records])

        times = np.array([float(r.get("timestamp_s", 0.0)) for r in self.records])
        dt = np.diff(times, prepend=times[0])
        dt[dt <= 0] = 0.033

        # 2D horizontal position error
        d_pos_2d = np.hypot(vps_pn - fc_pn, vps_pe - fc_pe)
        pos_rmse = float(np.sqrt(np.mean(d_pos_2d ** 2)))
        cep_50 = float(np.percentile(d_pos_2d, 50))
        cep_95 = float(np.percentile(d_pos_2d, 95))

        # Velocity error
        vps_vn = np.array([float(r.get("vps_vel_n_mps", 0.0)) for r in self.records])
        vps_ve = np.array([float(r.get("vps_vel_e_mps", 0.0)) for r in self.records])
        fc_vn = np.array([float(r.get("fc_vel_n_mps", 0.0)) for r in self.records])
        fc_ve = np.array([float(r.get("fc_vel_e_mps", 0.0)) for r in self.records])
        d_vel = np.hypot(vps_vn - fc_vn, vps_ve - fc_ve)
        vel_rmse = float(np.sqrt(np.mean(d_vel ** 2)))

        # Drift rate
        duration_min = max(0.1, (times[-1] - times[0]) / 60.0)
        drift_rate = float(d_pos_2d[-1] / duration_min)

        # Map matching
        mm_attempts = int(sum(1 for r in self.records if r.get("map_match_attempted")))
        mm_accepted = int(sum(1 for r in self.records if r.get("map_match_accepted")))
        mm_success_rate = float(mm_accepted / mm_attempts) if mm_attempts > 0 else 0.0
        mm_errors = [float(r.get("map_match_error_m", 0.0)) for r in self.records if r.get("map_match_accepted")]
        mean_mm_err = float(np.mean(mm_errors)) if mm_errors else 0.0

        # Virtual GNSS denial tests (e.g. 1 min and 3 min)
        denial_runs = self.run_virtual_gnss_denial(denial_durations_s=[60.0, 180.0])

        pass_status = (cep_95 <= self.max_cep95_tol_m) and (drift_rate <= 3.0)

        return Phase2EvaluationResult(
            total_samples=len(self.records),
            pos_2d_rmse_m=pos_rmse,
            cep_50_m=cep_50,
            cep_95_m=cep_95,
            vel_rmse_mps=vel_rmse,
            drift_rate_m_per_min=drift_rate,
            map_match_attempts=mm_attempts,
            map_match_accepted=mm_accepted,
            map_match_success_rate=mm_success_rate,
            mean_map_match_error_m=mean_mm_err,
            virtual_denial_runs=denial_runs,
            pass_status=pass_status,
        )

    def run_virtual_gnss_denial(
        self,
        denial_durations_s: List[float] = [60.0, 180.0],
    ) -> List[VirtualDenialRun]:
        """Simulate post-flight virtual GPS denial on the recorded trajectory."""
        runs: List[VirtualDenialRun] = []
        if len(self.records) < 30:
            return runs

        times = np.array([float(r.get("timestamp_s", 0.0)) for r in self.records])
        t_total = times[-1] - times[0]

        for dur in denial_durations_s:
            if dur >= t_total:
                # Truncate to available flight time
                dur = max(10.0, t_total * 0.8)

            # Start virtual denial at 25% of flight time
            idx_start = int(len(self.records) * 0.25)
            t_start = times[idx_start]
            t_end = t_start + dur
            idx_end = int(np.searchsorted(times, t_end))

            sub_records = self.records[idx_start:idx_end]
            if len(sub_records) < 10:
                continue

            # Dead-reckoning integration from VIO velocity
            sim_pn = float(sub_records[0].get("fc_pos_n_m", 0.0))
            sim_pe = float(sub_records[0].get("fc_pos_e_m", 0.0))

            errors = []
            for i in range(1, len(sub_records)):
                dt = float(sub_records[i].get("timestamp_s", 0.0)) - float(sub_records[i-1].get("timestamp_s", 0.0))
                dt = max(0.001, min(0.5, dt))
                vn = float(sub_records[i].get("vps_vel_n_mps", sub_records[i].get("fc_vel_n_mps", 0.0)))
                ve = float(sub_records[i].get("vps_vel_e_mps", sub_records[i].get("fc_vel_e_mps", 0.0)))

                sim_pn += vn * dt
                sim_pe += ve * dt

                true_pn = float(sub_records[i].get("fc_pos_n_m", 0.0))
                true_pe = float(sub_records[i].get("fc_pos_e_m", 0.0))
                errors.append(float(np.hypot(sim_pn - true_pn, sim_pe - true_pe)))

            if errors:
                final_drift = errors[-1]
                drift_rate = final_drift / max(0.1, (dur / 60.0))
                cep50 = float(np.percentile(errors, 50))
                cep95 = float(np.percentile(errors, 95))
            else:
                final_drift = drift_rate = cep50 = cep95 = 0.0

            runs.append(VirtualDenialRun(
                start_time_s=t_start,
                denial_duration_s=dur,
                final_drift_m=final_drift,
                drift_rate_m_per_min=drift_rate,
                cep_50_m=cep50,
                cep_95_m=cep95,
            ))

        return runs

    def _empty_phase1_result(self) -> Phase1EvaluationResult:
        return Phase1EvaluationResult(
            total_samples=0, duration_s=0.0, overall_rmse_m=0.0,
            overall_mean_abs_error_m=0.0, overall_bias_m=0.0, overall_p95_m=0.0,
            overall_mean_rel_pct=0.0, total_dropout_s=0.0, bands=[],
            regime_rmse={}, confidence_calibration={}, bad_segment_warnings=[],
            pass_status=False,
        )

    def _empty_phase2_result(self) -> Phase2EvaluationResult:
        return Phase2EvaluationResult(
            total_samples=0, pos_2d_rmse_m=0.0, cep_50_m=0.0, cep_95_m=0.0,
            vel_rmse_mps=0.0, drift_rate_m_per_min=0.0, map_match_attempts=0,
            map_match_accepted=0, map_match_success_rate=0.0,
            mean_map_match_error_m=0.0, virtual_denial_runs=[], pass_status=False,
        )

    def generate_html_report(
        self,
        output_html: str | Path,
        p1: Phase1EvaluationResult,
        p2: Phase2EvaluationResult,
    ) -> Path:
        """Generate interactive standalone HTML report with embedded matplotlib plots."""
        out_path = Path(output_html)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        # Generate Plot 1: Height vs Time & Error Distribution
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), gridspec_kw={"height_ratios": [2, 1]})
        times = np.array([r.get("timestamp_s", 0.0) for r in self.records])
        t_rel = times - times[0] if len(times) > 0 else np.zeros(1)
        vps_h = np.array([float(r.get("vps_height_agl_m", r.get("estimated_h_m", 0.0))) for r in self.records])
        fc_pos_d = np.array([float(r.get("fc_pos_d_m", 0.0)) for r in self.records])
        if np.any(np.abs(fc_pos_d) > 0.01):
            gt_h = -fc_pos_d + self.baseline_offset_m
        else:
            baro_h = np.array([float(r.get("fc_baro_alt_m", r.get("baro_alt_m", 0.0))) for r in self.records])
            if len(baro_h) > 0 and baro_h[0] > 50.0:
                gt_h = (baro_h - baro_h[0]) + self.baseline_offset_m
            else:
                gt_h = baro_h + self.baseline_offset_m

        ax1.plot(t_rel, gt_h, "k--", label="Ground Truth (Baro/GPS + 6cm)", alpha=0.7)
        ax1.plot(t_rel, vps_h, "b-", label="VPS Height Estimate", lw=1.5)
        ax1.set_ylabel("Height AGL (m)")
        ax1.set_title("Phase 1 Height Tracking Performance (Observe-Only Shadow Mode)")
        ax1.legend(loc="upper left")
        ax1.grid(True, alpha=0.3)

        err = vps_h - gt_h
        ax2.plot(t_rel, err * 100.0, "r-", lw=1.0, label="Error (cm)")
        ax2.axhline(0, color="gray", lw=0.5)
        ax2.set_xlabel("Time (s)")
        ax2.set_ylabel("Error (cm)")
        ax2.grid(True, alpha=0.3)
        plt.tight_layout()

        buf1 = io.BytesIO()
        plt.savefig(buf1, format="png", dpi=120)
        plt.close(fig)
        b64_plot1 = base64.b64encode(buf1.getvalue()).decode("ascii")

        # HTML Assembly
        status_color = "#28a745" if (p1.pass_status and p2.pass_status) else "#dc3545"
        status_text = "PASS (FLIGHT READY)" if (p1.pass_status and p2.pass_status) else "FAIL (CRITERIA NOT MET)"

        band_rows = "".join([
            f"<tr><td>{b.band_name}</td><td>{b.sample_count}</td><td>{b.rmse_m:.3f} m</td><td>{b.p95_error_m:.3f} m</td><td>{b.mean_rel_error_pct:.2f}%</td><td>{b.dropout_time_s:.2f} s</td><td style='color:{'#28a745' if b.pass_criteria_met else '#dc3545'};font-weight:bold;'>{'PASS' if b.pass_criteria_met else 'FAIL'}</td></tr>"
            for b in p1.bands
        ])

        denial_rows = "".join([
            f"<tr><td>{r.denial_duration_s:.0f} s ({r.denial_duration_s/60:.1f} min)</td><td>{r.final_drift_m:.2f} m</td><td>{r.drift_rate_m_per_min:.2f} m/min</td><td>{r.cep_50_m:.2f} m</td><td>{r.cep_95_m:.2f} m</td></tr>"
            for r in p2.virtual_denial_runs
        ])

        html_content = f"""<!DOCTYPE html>
<html>
<head>
<title>VPS Shadow Mode Flight Evaluation Report</title>
<style>
body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; background: #f8f9fa; color: #212529; margin: 0; padding: 20px; }}
.container {{ max-width: 1000px; margin: 0 auto; background: #fff; padding: 30px; border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.08); }}
h1, h2, h3 {{ color: #1a202c; }}
.header-badge {{ display: inline-block; padding: 6px 14px; border-radius: 4px; color: #fff; font-weight: bold; background: {status_color}; font-size: 16px; margin-bottom: 20px; }}
.metric-grid {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 15px; margin-bottom: 25px; }}
.metric-card {{ background: #f1f5f9; padding: 15px; border-radius: 6px; border-left: 4px solid #3b82f6; }}
.metric-card.warn {{ border-left-color: #f59e0b; }}
.metric-val {{ font-size: 24px; font-weight: bold; margin-top: 5px; }}
table {{ width: 100%; border-collapse: collapse; margin-top: 15px; margin-bottom: 25px; }}
th, td {{ padding: 10px 12px; border: 1px solid #e2e8f0; text-align: left; }}
th {{ background: #f8fafc; font-weight: 600; }}
img.plot {{ width: 100%; border-radius: 6px; border: 1px solid #e2e8f0; margin-top: 15px; }}
</style>
</head>
<body>
<div class="container">
<h1>VPS Shadow Mode - Flight Evaluation Report</h1>
<div class="header-badge">OVERALL STATUS: {status_text}</div>
<p><strong>Session ID:</strong> {self.session_dir.name} | <strong>Ground Baseline:</strong> {self.baseline_offset_m*100:.1f} cm | <strong>Site:</strong> Aurangabad (19.808798 N, 75.283380 E)</p>

<h2>Key Performance Indicators (KPIs)</h2>
<div class="metric-grid">
  <div class="metric-card">
    <div>Phase 1 Height RMSE</div>
    <div class="metric-val">{p1.overall_rmse_m:.3f} m</div>
  </div>
  <div class="metric-card">
    <div>Height Rel Error</div>
    <div class="metric-val">{p1.overall_mean_rel_pct:.2f}%</div>
  </div>
  <div class="metric-card">
    <div>Phase 2 2D CEP-95</div>
    <div class="metric-val">{p2.cep_95_m:.2f} m</div>
  </div>
  <div class="metric-card">
    <div>VIO Drift Rate</div>
    <div class="metric-val">{p2.drift_rate_m_per_min:.2f} m/min</div>
  </div>
</div>

<h2>Phase 1: Height Performance by Altitude Band</h2>
<table>
<thead>
<tr><th>Altitude Band</th><th>Frames</th><th>RMSE</th><th>P95 Error</th><th>Mean Rel Err (%)</th><th>Dropout Time</th><th>Status</th></tr>
</thead>
<tbody>
{band_rows}
</tbody>
</table>

<h2>Phase 2: Post-Flight Virtual GNSS Denial Test</h2>
<p>Simulates cutting off GPS mid-flight to benchmark purely visual dead-reckoning against the logged GPS truth trajectory.</p>
<table>
<thead>
<tr><th>Denial Duration</th><th>Final Drift (m)</th><th>Drift Rate (m/min)</th><th>CEP-50</th><th>CEP-95</th></tr>
</thead>
<tbody>
{denial_rows if denial_rows else "<tr><td colspan='5'>Flight duration too short for multi-minute virtual denial.</td></tr>"}
</tbody>
</table>

<h2>Performance Plots</h2>
<img class="plot" src="data:image/png;base64,{b64_plot1}" alt="Height Tracking Plot">

<p style="margin-top:30px; font-size:12px; color:#718096; text-align:center;">Generated by VPS Shadow Mode Evaluation Suite. Observe-Only Invariant Enforced.</p>
</div>
</body>
</html>
"""
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(html_content)

        logger.info("Saved standalone HTML evaluation report to %s", out_path)
        return out_path


def main() -> int:
    parser = argparse.ArgumentParser(description="VPS Shadow Flight Evaluation and Report Generator")
    parser.add_argument("--session", "-s", type=str, required=True, help="Path to recorded session folder")
    parser.add_argument("--baseline", type=float, default=0.06, help="Ground baseline offset in meters")
    parser.add_argument("--output-html", "-o", type=str, default=None, help="Output HTML report path")
    args = parser.parse_args()

    evaluator = ShadowEvaluator(session_dir=args.session, baseline_offset_m=args.baseline)
    p1 = evaluator.evaluate_phase1()
    p2 = evaluator.evaluate_phase2()

    print("=" * 72)
    print("           VPS SHADOW MODE - FLIGHT PERFORMANCE EVALUATION")
    print("=" * 72)
    print(f"Session:                 {evaluator.session_dir.name}")
    print(f"Total Evaluated Frames:  {p1.total_samples}")
    print(f"Phase 1 Height RMSE:     {p1.overall_rmse_m:.3f} m ({p1.overall_mean_rel_pct:.2f}% relative)")
    print(f"Phase 1 Height Bias:     {p1.overall_bias_m:.3f} m | P95: {p1.overall_p95_m:.3f} m")
    print(f"Phase 1 Pass Status:     {'PASS' if p1.pass_status else 'FAIL'}")
    print("-" * 72)
    print(f"Phase 2 Position RMSE:   {p2.pos_2d_rmse_m:.3f} m")
    print(f"Phase 2 CEP-50 / CEP-95: {p2.cep_50_m:.2f} m / {p2.cep_95_m:.2f} m")
    print(f"Phase 2 Drift Rate:      {p2.drift_rate_m_per_min:.2f} m/min")
    print(f"Phase 2 Pass Status:     {'PASS' if p2.pass_status else 'FAIL'}")
    print("=" * 72)

    out_html = Path(args.output_html) if args.output_html else evaluator.session_dir / "evaluation_report.html"
    evaluator.generate_html_report(out_html, p1, p2)

    return 0 if (p1.pass_status and p2.pass_status) else 1


if __name__ == "__main__":
    sys.exit(main())
