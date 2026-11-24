# FLIGHT-02: Stage 1B: Full Height Altitude Ladder (0 - 50m)
**Target Site:** Aurangabad Operational Flight Area (1 km radius) (`19.808798 N, 75.283380 E`)
**Max Altitude:** 50.0 m AGL | **Ground Truth:** Physical 6.0 cm baseline + Baro / MicroAir GNSS height ladder.

### Objective
Characterize height estimator accuracy and scale across full 0 - 50m operational envelope.

### Flight Sequence & Altitude Ladder

| Step | Name | Target Alt | Duration | Maneuver | Pilot Action | Expected VPS Output |
|---|---|---|---|---|---|---|
| 1 | Pre-Arm Baseline | **0.06m** | 30.0s | Ground rest | Preflight checklist. | Stationary hold 0.06m. |
| 2 | Hover Band 1 | **10.0m** | 45.0s | Hover at 10m | Hold 10m AGL. | Track >= 180 features. |
| 3 | Hover Band 2 | **25.0m** | 45.0s | Hover at 25m | Hold 25m AGL. | Confidence >= 0.80. |
| 4 | Hover Band 3 (Ceiling) | **50.0m** | 60.0s | Hover at 50m | Hold 50m AGL (initial ceiling). | Confidence >= 0.75. |
| 5 | Descent Transect | **0.06m** | 45.0s | Continuous descent | Descend at 1.0 m/s to landing. | Tracks descent profile. |

### Pre-Flight Pilot Checklist
- [ ] Stage 1A flight data reviewed and validated.
- [ ] Airspace clear up to 60m AGL.
- [ ] Preflight checklist verified PASS.

### Safety Invariants & Abort Thresholds
- **ABORT IF:** Altitude ceiling breached (> 55m AGL).
- **ABORT IF:** MicroAir GPS fix degraded.
- **ABORT IF:** Companion computer telemetry loss > 3.0s.

### Post-Flight Data Preservation Protocol
- [ ] Wait 15s post-disarm before shutdown.
- [ ] Check logs directory for telemetry.csv and session_meta.json.
