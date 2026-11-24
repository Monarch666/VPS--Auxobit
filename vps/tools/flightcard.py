"""Flight Test Card Generator for VPS Shadow Mode Operations.

Generates structured, stage-by-stage test cards for the pilot and flight team:
- Altitude ladder (hover durations, climb/descent rates).
- Ground truth reference setup (6 cm physical baseline offset + MicroAir GNSS).
- Pre-arm, in-flight HUD, and post-arm checklists.
- Clear abort thresholds (battery, drift, GPS loss).
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any, Optional
import argparse
import sys
import time


@dataclass
class FlightStep:
    step_num: int
    name: str
    target_alt_agl_m: float
    duration_s: float
    maneuver: str
    pilot_action: str
    expected_vps_behavior: str


@dataclass
class FlightCard:
    flight_id: str
    stage_name: str
    objective: str
    target_site: str
    coordinates: str
    max_altitude_m: float
    ground_truth_setup: str
    steps: List[FlightStep]
    abort_criteria: List[str]
    pilot_checklist_pre: List[str]
    pilot_checklist_post: List[str]


def generate_flight_cards() -> Dict[str, FlightCard]:
    """Define standard staged test plan flight cards."""
    cards = {}

    # Flight 1: Ground Baseline & Low Altitude Ladder
    cards["1"] = FlightCard(
        flight_id="FLIGHT-01",
        stage_name="Stage 1A: Ground Baseline & Low Hover (0 - 15m)",
        objective="Verify 6 cm ground baseline subtraction, optical flow lock, and initial height estimate vs MicroAir GPS.",
        target_site="Aurangabad Operational Flight Area (1 km radius)",
        coordinates="19.808798 N, 75.283380 E",
        max_altitude_m=15.0,
        ground_truth_setup="Physical 6.0 cm lens-to-skid offset + MicroAir GNSS altitude above takeoff point.",
        steps=[
            FlightStep(
                step_num=1,
                name="Ground Pre-Arm Baseline",
                target_alt_agl_m=0.06,
                duration_s=30.0,
                maneuver="Resting flat on takeoff pad before arming",
                pilot_action="Power on Pi 5. Verify preflight.py passes. Confirm GCS HUD shows VPS_H = 0.06m (+-0.02m).",
                expected_vps_behavior="Pre-roll buffer active in RAM. Estimator stationary lock at 6.0 cm.",
            ),
            FlightStep(
                step_num=2,
                name="Arming & Takeoff Transient",
                target_alt_agl_m=2.0,
                duration_s=15.0,
                maneuver="Vertical climb at 0.5 m/s to 2m",
                pilot_action="Arm drone in Loiter/GPS mode and climb gently to 2m.",
                expected_vps_behavior="Auto-trigger recording: drains 10s pre-roll buffer to disk. Height tracks takeoff transient.",
            ),
            FlightStep(
                step_num=3,
                name="Low Hover Station-Keeping",
                target_alt_agl_m=5.0,
                duration_s=45.0,
                maneuver="Loiter hover at 5m AGL",
                pilot_action="Hold position over natural ground texture. Observe GCS HUD VPS_H.",
                expected_vps_behavior="Visual height variance < 0.05m. Feature tracker tracks >= 150 points.",
            ),
            FlightStep(
                step_num=4,
                name="Mid-Altitude Hover Ladder",
                target_alt_agl_m=15.0,
                duration_s=45.0,
                maneuver="Climb at 1.0 m/s to 15m and hover",
                pilot_action="Ascend to 15m, loiter for 45s, verify telemetry streams.",
                expected_vps_behavior="Height estimation maintains high confidence (> 0.85). Zero FC interference.",
            ),
            FlightStep(
                step_num=5,
                name="Descent & Touchdown",
                target_alt_agl_m=0.06,
                duration_s=30.0,
                maneuver="Vertical descent at 0.8 m/s to landing",
                pilot_action="Land smoothly and disarm immediately on touchdown.",
                expected_vps_behavior="Landing detected: height recovers ground baseline (0.06m). Enters 10s post-roll logging.",
            ),
        ],
        abort_criteria=[
            "MicroAir GPS satellites drop below 10 or HDOP > 1.5 in flight.",
            "Visual Positioning System CPU temperature exceeds 82 deg C.",
            "Battery cell voltage drops below 3.65V per cell.",
            "Wind gusts exceed 10 m/s.",
        ],
        pilot_checklist_pre=[
            "[ ] MicroSD card inserted with >= 2 GB free space.",
            "[ ] IMX296 lens clean, protective cap removed, aperture locked.",
            "[ ] Run 'python vps/tools/preflight.py' on Pi 5 -> Confirm ALL PASS.",
            "[ ] Mission Planner / QGroundControl HUD shows VPS_H (0.06m) and VPS_CONF (1.0).",
            "[ ] Confirm VPS operating in SHADOW MODE (Zero output injection to EKF).",
        ],
        pilot_checklist_post=[
            "[ ] DO NOT cut power immediately after landing.",
            "[ ] Wait >= 15 seconds after disarm to allow post-roll buffer flush to MicroSD.",
            "[ ] Verify LED indicators on Pi 5 stop rapid disk write activity.",
            "[ ] Disconnect flight battery.",
        ],
    )

    # Flight 2: Full Phase 1 Altitude Ladder (0 - 50m)
    cards["2"] = FlightCard(
        flight_id="FLIGHT-02",
        stage_name="Stage 1B: Full Height Altitude Ladder (0 - 50m)",
        objective="Characterize height estimator accuracy and scale across full 0 - 50m operational envelope.",
        target_site="Aurangabad Operational Flight Area (1 km radius)",
        coordinates="19.808798 N, 75.283380 E",
        max_altitude_m=50.0,
        ground_truth_setup="Physical 6.0 cm baseline + Baro / MicroAir GNSS height ladder.",
        steps=[
            FlightStep(1, "Pre-Arm Baseline", 0.06, 30.0, "Ground rest", "Preflight checklist.", "Stationary hold 0.06m."),
            FlightStep(2, "Hover Band 1", 10.0, 45.0, "Hover at 10m", "Hold 10m AGL.", "Track >= 180 features."),
            FlightStep(3, "Hover Band 2", 25.0, 45.0, "Hover at 25m", "Hold 25m AGL.", "Confidence >= 0.80."),
            FlightStep(4, "Hover Band 3 (Ceiling)", 50.0, 60.0, "Hover at 50m", "Hold 50m AGL (initial ceiling).", "Confidence >= 0.75."),
            FlightStep(5, "Descent Transect", 0.06, 45.0, "Continuous descent", "Descend at 1.0 m/s to landing.", "Tracks descent profile."),
        ],
        abort_criteria=[
            "Altitude ceiling breached (> 55m AGL).",
            "MicroAir GPS fix degraded.",
            "Companion computer telemetry loss > 3.0s.",
        ],
        pilot_checklist_pre=[
            "[ ] Stage 1A flight data reviewed and validated.",
            "[ ] Airspace clear up to 60m AGL.",
            "[ ] Preflight checklist verified PASS.",
        ],
        pilot_checklist_post=[
            "[ ] Wait 15s post-disarm before shutdown.",
            "[ ] Check logs directory for telemetry.csv and session_meta.json.",
        ],
    )

    # Flight 3: Horizontal Transect & Phase 2 VIO Speed Benchmark
    cards["3"] = FlightCard(
        flight_id="FLIGHT-03",
        stage_name="Stage 2A: Metric VIO Horizontal Navigation Benchmark",
        objective="Validate horizontal metric velocity estimation and position dead-reckoning against GPS.",
        target_site="Aurangabad Operational Flight Area (1 km radius)",
        coordinates="19.808798 N, 75.283380 E",
        max_altitude_m=30.0,
        ground_truth_setup="MicroAir GPS EKF velocity and position logged in telemetry.csv.",
        steps=[
            FlightStep(1, "Takeoff to 30m", 30.0, 30.0, "Climb to 30m", "Ascend to 30m AGL.", "Altitude locked."),
            FlightStep(2, "Slow Line Transect (3 m/s)", 30.0, 40.0, "100m straight line", "Fly forward 100m at 3 m/s.", "VIO vel tracks GPS vel."),
            FlightStep(3, "Medium Line Transect (6 m/s)", 30.0, 30.0, "Return 100m transect", "Fly return 100m at 6 m/s.", "Flow de-rotation active."),
            FlightStep(4, "Descent and Land", 0.06, 30.0, "Vertical descent", "Land on target pad.", "Flush logs."),
        ],
        abort_criteria=[
            "Drone position exceeds 200m from pilot.",
            "GPS velocity innovation alert.",
        ],
        pilot_checklist_pre=[
            "[ ] Optical flow de-rotation calibrated.",
            "[ ] Compass health verified in QGroundControl.",
        ],
        pilot_checklist_post=[
            "[ ] Confirm full 30 FPS frame sequence saved.",
        ],
    )

    return cards


def render_markdown(card: FlightCard) -> str:
    """Format FlightCard as printable Markdown."""
    md = []
    md.append(f"# {card.flight_id}: {card.stage_name}")
    md.append(f"**Target Site:** {card.target_site} (`{card.coordinates}`)")
    md.append(f"**Max Altitude:** {card.max_altitude_m} m AGL | **Ground Truth:** {card.ground_truth_setup}\n")
    md.append(f"### Objective\n{card.objective}\n")

    md.append("### Flight Sequence & Altitude Ladder\n")
    md.append("| Step | Name | Target Alt | Duration | Maneuver | Pilot Action | Expected VPS Output |")
    md.append("|---|---|---|---|---|---|---|")
    for s in card.steps:
        md.append(f"| {s.step_num} | {s.name} | **{s.target_alt_agl_m}m** | {s.duration_s}s | {s.maneuver} | {s.pilot_action} | {s.expected_vps_behavior} |")
    md.append("")

    md.append("### Pre-Flight Pilot Checklist")
    for item in card.pilot_checklist_pre:
        md.append(f"- {item}")
    md.append("")

    md.append("### Safety Invariants & Abort Thresholds")
    for item in card.abort_criteria:
        md.append(f"- **ABORT IF:** {item}")
    md.append("")

    md.append("### Post-Flight Data Preservation Protocol")
    for item in card.pilot_checklist_post:
        md.append(f"- {item}")
    md.append("")

    return "\n".join(md)


def main() -> int:
    parser = argparse.ArgumentParser(description="VPS Shadow Flight Card Generator")
    parser.add_argument("--flight", "-f", type=str, default="1", help="Flight ID ('1', '2', '3', or 'all')")
    parser.add_argument("--output-dir", "-o", type=str, default="flightcards", help="Directory to save generated flight cards")
    args = parser.parse_args()

    cards = generate_flight_cards()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    selected = list(cards.keys()) if args.flight.lower() == "all" else [args.flight]

    for fid in selected:
        if fid not in cards:
            print(f"[ERROR] Flight '{fid}' not found. Available: {list(cards.keys())}")
            return 1
        card = cards[fid]
        md_text = render_markdown(card)
        file_path = out_dir / f"flightcard_{card.flight_id.lower()}.md"
        file_path.write_text(md_text, encoding="utf-8")
        print(f"[OK] Generated {file_path}")
        print("-" * 72)
        print(md_text)
        print("=" * 72)

    return 0


if __name__ == "__main__":
    sys.exit(main())
