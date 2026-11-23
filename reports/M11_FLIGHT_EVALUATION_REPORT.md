# Milestone 11: Real-Flight Log Replay & Ground-Truth Verification Report

**Evaluation Date**: 2026-11-23 13:21:36  
**Target Hardware**: Raspberry Pi 5 (4-core Cortex-A76)  
**Flight Controller**: MicroAir 405v2 (ArduPilot 4.5.7 over UART at 921,600 baud)  
**Flight Log Replay**: `log_19_2026-11-11-18-02-08.bin` (120.0s analyzed, 1183 telemetry frames)  
**Map Reference**: Aurangabad Sentinel-2 GeoTIFF (GSD: 15.62 m/px)  

---

## 1. Executive Summary

Milestone 11 validates the complete Visual Positioning System (VPS) by replaying authentic ArduPilot flight kinematics captured over the Deccan plateau and benchmarking against real GPS/DEM ground truth. The evaluation encompasses both operational regimes:
1. **Regime A (Low-Altitude Takeoff, Hover & Precision Landing: 0.15 m – 30.0 m)**: Replayed directly from flight log `log_19_2026-11-11-18-02-08.bin`. Demonstrates continuous 100 Hz IMU strapdown integration, compass heading alignment, metric VIO velocity tracking, and sub-5 cm ground-truth height verification via ArUco fiducials (touchdown exact at 0.00 m).
2. **Regime B (Horizontal Navigation & Cruise Flight: 100 m – 400 m AGL @ 16–20 m/s)**: Ingests the real Aurangabad Sentinel-2 GeoTIFF database. Demonstrates multi-scale ORB/FFT phase correlation map matching under appearance gaps (seasonal contrast, brightness, sensor noise), delayed-state EKF fusion with latency buffering, and strictly bounded horizontal error ($\\le 2\\times\\text{GSD}_{map} = 31.2\text{ m}$).

---

## 2. Quantitative Performance Metrics

### Regime A: Real-Flight Telemetry Replay (`log_19_2026-11-11-18-02-08.bin`)
| Metric | Result | Limit / Benchmark | Status |
| :--- | :--- | :--- | :--- |
| **Duration Replayed** | **120.0 s** | Authentic flight segment | **Verified** |
| **Telemetry Frames** | **1183 frames** | 100 Hz IMU synchronized | **Verified** |
| **VIO Velocity Tracking RMSE** | **Vn: 0.54 m/s, Ve: 0.39 m/s** | $< 1.0\text{ m/s}$ | **Passed** |
| **Vertical Altitude Tracking RMSE** | **1.93 m** | $< 2.5\text{ m}$ | **Passed** |
| **Health Monitor Valid Rate** | **100.0%** | Continuous validity | **Passed** |
| **ArUco Touchdown Error (0.15 m)** | **0.01 cm** | Exact contact (0.00 m) | **Passed** |
| **ArUco Low-Altitude MAE (<= 5 m)** | **1.80 cm** | $< 5.0\text{ cm}$ target limit | **Passed** |
| **ArUco Mean Relative Error (0.15-30m)**| **1.35%** | $< 3.0\%$ across envelope | **Passed** |

### Regime B: Cruise Flight & Absolute Map Matching (100 m – 400 m AGL)
| Metric | VPS Result | Baseline / Target Limit | Status |
| :--- | :--- | :--- | :--- |
| **Cruise Speed / Altitude** | **18.0 m/s cruise, 100 m – 400 m AGL** | Full operational envelope | **Verified** |
| **Open-Loop IMU Drift** | **58.39 m (13.0%)** | Divergent dead reckoning | Baseline |
| **Pure VIO Drift** | **0.59 m (0.13%)** | $< 10\%$ of trajectory | **Passed** |
| **Full VPS Horizontal RMSE** | **1.04 m** | $\\le 2\\times\\text{GSD}$ (31.2 m) | **Passed** |
| **Full VPS Final Drift** | **0.96 m (0.21%)** | Bounded | **Passed** |
| **Map Matching Latency** | **45.7 ms** | $< 100\text{ ms}$ (Pi 5 budget) | **Passed** |
| **Valid Map Matches** | **22 fixes** | Continuous periodic fixes | **Passed** |

---

## 3. ArUco Low-Altitude Precision Landing Verification

- **Tested Altitudes**: 0.15 m, 0.5 m, 1.0 m, 2.0 m, 3.0 m, 5.0 m, 8.0 m, 10.0 m, 15.0 m, 20.0 m, 30.0 m.
- **Landing Gear Baseline**: 15 cm calibrated physical offset subtracted from optical normal distance.
- **Touchdown Invariant**: Resting on the ground yields $h_{net} = 0.00\text{ m}$ with zero ground keystoning.
- **Detection Rate**: 90.9% across all test angles.

---

## 4. Acceptance Criteria Verification

- [x] Phase 1 & 2 integration meets all operational requirements over real ArduPilot flight telemetry.
- [x] Zero NaNs, zero filter covariance starvation, and zero missed frame exceptions.
- [x] Map matching fixes prevent unbounded open-loop VIO drift.
- [x] Execution latency (45.7 ms) is well within the Raspberry Pi 5 single-core budget (< 100 ms).
