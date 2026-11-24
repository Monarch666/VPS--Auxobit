# FLIGHT-01: Stage 1A: Ground Baseline & Low Hover (0 - 15m)
**Target Site:** Aurangabad Operational Flight Area (1 km radius) (`19.808798 N, 75.283380 E`)
**Max Altitude:** 15.0 m AGL | **Ground Truth:** Physical 6.0 cm lens-to-skid offset + MicroAir GNSS altitude above takeoff point.

### Objective
Verify 6 cm ground baseline subtraction, optical flow lock, and initial height estimate vs MicroAir GPS.

### Flight Sequence & Altitude Ladder

| Step | Name | Target Alt | Duration | Maneuver | Pilot Action | Expected VPS Output |
|---|---|---|---|---|---|---|
| 1 | Ground Pre-Arm Baseline | **0.06m** | 30.0s | Resting flat on takeoff pad before arming | Power on Pi 5. Verify preflight.py passes. Confirm GCS HUD shows VPS_H = 0.06m (+-0.02m). | Pre-roll buffer active in RAM. Estimator stationary lock at 6.0 cm. |
| 2 | Arming & Takeoff Transient | **2.0m** | 15.0s | Vertical climb at 0.5 m/s to 2m | Arm drone in Loiter/GPS mode and climb gently to 2m. | Auto-trigger recording: drains 10s pre-roll buffer to disk. Height tracks takeoff transient. |
| 3 | Low Hover Station-Keeping | **5.0m** | 45.0s | Loiter hover at 5m AGL | Hold position over natural ground texture. Observe GCS HUD VPS_H. | Visual height variance < 0.05m. Feature tracker tracks >= 150 points. |
| 4 | Mid-Altitude Hover Ladder | **15.0m** | 45.0s | Climb at 1.0 m/s to 15m and hover | Ascend to 15m, loiter for 45s, verify telemetry streams. | Height estimation maintains high confidence (> 0.85). Zero FC interference. |
| 5 | Descent & Touchdown | **0.06m** | 30.0s | Vertical descent at 0.8 m/s to landing | Land smoothly and disarm immediately on touchdown. | Landing detected: height recovers ground baseline (0.06m). Enters 10s post-roll logging. |

### Pre-Flight Pilot Checklist
- [ ] MicroSD card inserted with >= 2 GB free space.
- [ ] IMX296 lens clean, protective cap removed, aperture locked.
- [ ] Run 'python vps/tools/preflight.py' on Pi 5 -> Confirm ALL PASS.
- [ ] Mission Planner / QGroundControl HUD shows VPS_H (0.06m) and VPS_CONF (1.0).
- [ ] Confirm VPS operating in SHADOW MODE (Zero output injection to EKF).

### Safety Invariants & Abort Thresholds
- **ABORT IF:** MicroAir GPS satellites drop below 10 or HDOP > 1.5 in flight.
- **ABORT IF:** Visual Positioning System CPU temperature exceeds 82 deg C.
- **ABORT IF:** Battery cell voltage drops below 3.65V per cell.
- **ABORT IF:** Wind gusts exceed 10 m/s.

### Post-Flight Data Preservation Protocol
- [ ] DO NOT cut power immediately after landing.
- [ ] Wait >= 15 seconds after disarm to allow post-roll buffer flush to MicroSD.
- [ ] Verify LED indicators on Pi 5 stop rapid disk write activity.
- [ ] Disconnect flight battery.
