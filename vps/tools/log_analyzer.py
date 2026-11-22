"""ArduPilot DataFlash (.bin) flight log analyzer for VPS.

Parses flight logs from real missions to extract:
- Flight modes and mission duration
- Altitude profiles (takeoff, climb, cruise AGL, descent, landing)
- Horizontal velocity distributions (cruise speed, max speed)
- Vertical climb/descent rates
- Attitude envelopes (roll, pitch, yaw) and gyro angular rates
- Rangefinder / distance sensor configuration parameters
"""

from pathlib import Path
from typing import Dict, Any, List
import numpy as np
from pymavlink import mavutil


def analyze_flight_log(log_path: Path | str) -> Dict[str, Any]:
    """Parse ArduPilot DataFlash binary log and extract kinematic flight statistics."""
    path = Path(log_path)
    if not path.is_file():
        raise FileNotFoundError(f"Log file not found: {path}")

    mlog = mavutil.mavlink_connection(str(path))

    modes = []
    times_s = []
    rel_alts = []
    baro_alts = []
    speeds_mps = []
    vz_mps = []
    rolls_deg = []
    pitches_deg = []
    yaws_deg = []
    gyr_x_deg_s = []
    gyr_y_deg_s = []
    gyr_z_deg_s = []
    rfnd_dists_cm = []
    parameters = {}

    start_time_us = None

    while True:
        msg = mlog.recv_msg()
        if msg is None:
            break

        msg_type = msg.get_type()

        # Parameters
        if msg_type == "PARM":
            param_name = getattr(msg, "Name", None)
            param_val = getattr(msg, "Value", None)
            if param_name:
                parameters[param_name] = param_val

        # Flight Mode Changes
        elif msg_type == "MODE":
            mode_name = getattr(msg, "Mode", str(getattr(msg, "ModeNum", "UNKNOWN")))
            t_s = getattr(msg, "TimeUS", 0) * 1e-6
            modes.append((t_s, mode_name))

        # Attitude
        elif msg_type == "ATT":
            t_s = getattr(msg, "TimeUS", 0) * 1e-6
            rolls_deg.append(getattr(msg, "Roll", 0.0))
            pitches_deg.append(getattr(msg, "Pitch", 0.0))
            yaws_deg.append(getattr(msg, "Yaw", 0.0))

        # IMU Gyro
        elif msg_type == "IMU":
            gyr_x_deg_s.append(np.degrees(getattr(msg, "GyrX", 0.0)))
            gyr_y_deg_s.append(np.degrees(getattr(msg, "GyrY", 0.0)))
            gyr_z_deg_s.append(np.degrees(getattr(msg, "GyrZ", 0.0)))

        # Position & Altitude
        elif msg_type == "POS":
            t_s = getattr(msg, "TimeUS", 0) * 1e-6
            if start_time_us is None:
                start_time_us = getattr(msg, "TimeUS", 0)
            times_s.append(t_s)
            rel_alts.append(getattr(msg, "RelHomeAlt", 0.0))

        # Barometer
        elif msg_type == "BARO":
            baro_alts.append(getattr(msg, "Alt", 0.0))

        # GPS Velocity
        elif msg_type == "GPS":
            spd = getattr(msg, "Spd", 0.0)
            speeds_mps.append(spd)
            vz = getattr(msg, "VZ", 0.0)
            vz_mps.append(vz)

        # Rangefinder
        elif msg_type in ("RFND", "RNGFND"):
            dist = getattr(msg, "Dist", None) or getattr(msg, "Dist1", None)
            if dist is not None:
                rfnd_dists_cm.append(dist)

    total_duration_s = (times_s[-1] - times_s[0]) if len(times_s) > 1 else 0.0

    summary = {
        "filename": path.name,
        "duration_s": total_duration_s,
        "modes": modes,
        "rel_alt_max_m": float(np.max(rel_alts)) if rel_alts else 0.0,
        "rel_alt_median_m": float(np.median(rel_alts)) if rel_alts else 0.0,
        "speed_max_mps": float(np.max(speeds_mps)) if speeds_mps else 0.0,
        "speed_median_mps": float(np.median(speeds_mps)) if speeds_mps else 0.0,
        "roll_std_deg": float(np.std(rolls_deg)) if rolls_deg else 0.0,
        "pitch_std_deg": float(np.std(pitches_deg)) if pitches_deg else 0.0,
        "gyro_x_std_deg_s": float(np.std(gyr_x_deg_s)) if gyr_x_deg_s else 0.0,
        "gyro_y_std_deg_s": float(np.std(gyr_y_deg_s)) if gyr_y_deg_s else 0.0,
        "rngfnd_present": len(rfnd_dists_cm) > 0,
        "rngfnd_params": {k: v for k, v in parameters.items() if "RNGFND" in k},
    }
    return summary


if __name__ == "__main__":
    for f in Path("data/flight_logs").glob("*.bin"):
        print(f"\nAnalyzing {f.name}...")
        res = analyze_flight_log(f)
        print(f"  Duration: {res['duration_s']:.1f} s ({res['duration_s']/60:.1f} min)")
        print(f"  Modes: {res['modes']}")
        print(f"  Max Rel Alt: {res['rel_alt_max_m']:.1f} m | Median Rel Alt: {res['rel_alt_median_m']:.1f} m")
        print(f"  Max Speed: {res['speed_max_mps']:.1f} m/s | Median Speed: {res['speed_median_mps']:.1f} m/s")
        print(f"  Roll std: {res['roll_std_deg']:.2f} deg | Pitch std: {res['pitch_std_deg']:.2f} deg")
        print(f"  Gyro X std: {res['gyro_x_std_deg_s']:.2f} deg/s | Gyro Y std: {res['gyro_y_std_deg_s']:.2f} deg/s")
        print(f"  RNGFND params: {res['rngfnd_params']}")
