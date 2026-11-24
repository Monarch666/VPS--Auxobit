# FLIGHT-03: Stage 2A: Metric VIO Horizontal Navigation Benchmark
**Target Site:** Aurangabad Operational Flight Area (1 km radius) (`19.808798 N, 75.283380 E`)
**Max Altitude:** 30.0 m AGL | **Ground Truth:** MicroAir GPS EKF velocity and position logged in telemetry.csv.

### Objective
Validate horizontal metric velocity estimation and position dead-reckoning against GPS.

### Flight Sequence & Altitude Ladder

| Step | Name | Target Alt | Duration | Maneuver | Pilot Action | Expected VPS Output |
|---|---|---|---|---|---|---|
| 1 | Takeoff to 30m | **30.0m** | 30.0s | Climb to 30m | Ascend to 30m AGL. | Altitude locked. |
| 2 | Slow Line Transect (3 m/s) | **30.0m** | 40.0s | 100m straight line | Fly forward 100m at 3 m/s. | VIO vel tracks GPS vel. |
| 3 | Medium Line Transect (6 m/s) | **30.0m** | 30.0s | Return 100m transect | Fly return 100m at 6 m/s. | Flow de-rotation active. |
| 4 | Descent and Land | **0.06m** | 30.0s | Vertical descent | Land on target pad. | Flush logs. |

### Pre-Flight Pilot Checklist
- [ ] Optical flow de-rotation calibrated.
- [ ] Compass health verified in QGroundControl.

### Safety Invariants & Abort Thresholds
- **ABORT IF:** Drone position exceeds 200m from pilot.
- **ABORT IF:** GPS velocity innovation alert.

### Post-Flight Data Preservation Protocol
- [ ] Confirm full 30 FPS frame sequence saved.
