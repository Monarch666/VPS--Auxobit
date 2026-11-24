# Visual Positioning System (VPS) - Shadow Mode Operational Guide

**Companion Platform:** Raspberry Pi 5 (8 GB) + Sony IMX296 Global Shutter (6mm lens)  
**Flight Controller:** MicroAir 405v2 running ArduPilot 4.5.7 over MAVLink 2 (UART 921,600 baud)  
**GNSS Module:** MicroAir GPS (Standard single-frequency GNSS, non-RTK)  
**Operational Flight Envelope:** $19.808798^\circ\text{ N}, 75.283380^\circ\text{ E}$ (1.0 km radius, Aurangabad, Maharashtra)  
**Physical Camera Baseline:** $0.060\text{ m}$ ($6.0\text{ cm}$) landing-gear-to-lens offset  

---

## 1. Executive Summary & Core Principle

**Shadow Mode is strictly observe-only.**  
During shadow flights, the drone operates under normal GPS control and pilot supervision. The VPS companion computer captures 30 FPS downward video frames, reads flight controller telemetry, computes Phase 1 AGL height and Phase 2 horizontal navigation states in real time, and logs everything to MicroSD storage.

### Safety Invariants Enforced in Code
- **Outbound Whitelist:** Only `HEARTBEAT` ($\le 1\text{ Hz}$), `NAMED_VALUE_FLOAT` ($\le 2\text{ Hz}$ for non-intrusive monitoring), and `STATUSTEXT` can ever cross the MAVLink interface.
- **Strict Blacklist:** `DISTANCE_SENSOR`, `ODOMETRY`, `VISION_*`, `GPS_INPUT`, `COMMAND_LONG/INT`, `PARAM_SET`, and `SET_POSITION_TARGET_*` are hard-blocked by `ShadowMAVLinkSafetyChokepoint`.
- **Software Interlock:** `vps/config.py` raises `ConfigurationConflictError` and fails startup if `mode == "shadow"` and `feed_fc_ekf == True`.
- **Companion Isolation:** Watchdog supervisor ensures any companion computer crash, freeze, thermal throttle, or brownout can only stop data recording; it is physically incapable of affecting the flight controller.

---

## 2. Flight-Day Step-by-Step Pilot SOP

### A. Pre-Flight Bench Verification
1. Power the Raspberry Pi 5 from the regulated 5V/5A flight power supply.
2. Connect via SSH or terminal and run the automated pre-flight tool:
   ```bash
   python vps/tools/preflight.py
   ```
3. Confirm that all critical checks report **`PASS`**:
   - `Safety Mode & Interlocks`: `PASS` (Mode=shadow, Feed=False, Base=6.0cm)
   - `Camera Intrinsics Calibration`: `PASS` (fx=1204.3, fy=1196.4)
   - `Camera Stream & Monotonicity`: `PASS` (30 FPS, dt > 0 strictly monotonic)
   - `Lens Focus & Contrast`: `PASS` (Laplacian Var >= 10.0)
   - `MicroSD Storage Headroom`: `PASS` (>= 1000 MB free)
   - `Companion CPU Core Temperature`: `PASS` (< 75°C pre-arm)
   - `Phase 2 Map DB Coverage`: `PASS` (covers site coordinates)

### B. Field Checklist & Flight Card
Print or consult the flight test card generated for the flight stage:
```bash
python vps/tools/flightcard.py --flight 1
```
- **Inspect Lens:** Verify lens is free of dust/oil, protective lens cap is removed, and aperture lock screw is tight.
- **Inspect Ground Clearance:** Ensure drone rests flat on the takeoff pad with $6.0\text{ cm}$ clearance between lens and ground.
- **Check GCS HUD (Mission Planner / QGroundControl):**
  - Verify `VPS_H` reads approximately `0.06 m` ($\pm 0.02\text{ m}$) while sitting on the ground.
  - Verify `VPS_CONF` reads `1.0`.

### C. Takeoff & Flight Execution
1. Arm the drone in **Loiter** or **Position Hold** mode.
   - *Automated Trigger:* The VPS detects the ArduPilot armed flag in `HEARTBEAT` and immediately drains the 10-second RAM pre-roll buffer to disk, capturing the takeoff transient.
2. Follow the staged altitude ladder on the Flight Card:
   - **FLIGHT-01:** Ground (30s) $\to$ 2m climb $\to$ 5m hover (45s) $\to$ 15m hover (45s) $\to$ Touchdown.
   - **FLIGHT-02:** 0 to 50m ladder (10m, 25m, 50m hover bands).
   - **FLIGHT-03:** 50m AGL horizontal transects (3 m/s, 6 m/s) to benchmark VIO velocity vs GPS.
3. Observe live diagnostic telemetry in Mission Planner (`VPS_H`, `VPS_CONF`, `VPS_ERR`) broadcast at $\le 2\text{ Hz}$.

### D. Touchdown & Data Preservation Protocol
> [!IMPORTANT]
> **DO NOT DISCONNECT BATTERY IMMEDIATELY UPON LANDING.**
1. Land and disarm the drone.
2. **Wait at least 15 seconds** after disarm before removing flight battery power.
   - The VPS recorder maintains a 10-second post-roll buffer to record the touchdown event, flush all I/O queues, update `session_meta.json`, and safely close file descriptors.
3. Power down and retrieve the MicroSD card (or connect via Wi-Fi/Ethernet).

---

## 3. Post-Flight Data Evaluation & Reporting

Every flight generates a self-contained session folder under `logs/session_YYYYMMDD_HHMMSS/`:
```
logs/session_20261124_130000/
├── session_meta.json      # Git commit, config hash, camera calibration, site coords
├── telemetry.csv          # Synchronized 30 FPS telemetry & estimator states
├── images/                # Time-stamped JPEG frames (frame_000001_*.jpg)
└── evaluation_report.html # Comprehensive HTML analysis report
```

### 1. Verify Numerical Equivalence
Run the offline deterministic replayer to ensure bit-level reproducibility:
```bash
python vps/tools/replay.py --session logs/session_20261124_130000
```
- Expects `Equivalence Status: PASS (EQUIVALENT)`.
- Verifies replayed estimates match live companion computer outputs within $\le 0.01\%$ or $< 0.25\text{ m}$ under lossy JPEG compression.

### 2. Generate Performance Scorecard & Visual Report
Run the evaluation engine against GPS/Baro ground truth:
```bash
python vps/tools/evaluate.py --session logs/session_20261124_130000
```
This generates `evaluation_report.html`, featuring:
- **Phase 1 Height Tracking:** Height RMSE, Bias, P95 Error, and Relative Error (%) across altitude bands ($0\text{–}15\text{ m}$, $15\text{–}30\text{ m}$, $30\text{–}50\text{ m}$).
- **Flight Regime Separation:** RMSE segregated by Hover ($< 0.5\text{ m/s}$), Climb, Cruise, and Descent.
- **Confidence Calibration:** Compares estimated confidence against real error magnitude.
- **Phase 2 Virtual GNSS Denial Simulation:** Evaluates dead-reckoning drift growth over $1\text{ min}$ and $3\text{ min}$ GPS denial windows against true logged GPS.
- **Visual Plots:** Time-series tracking vs truth, error distribution, and CEP-50/95 metrics.

---

## 4. Raspberry Pi 5 Systemd Autostart Service

To configure the VPS daemon to launch automatically upon companion computer boot:

1. Create `/etc/systemd/system/vps.service`:
   ```ini
   [Unit]
   Description=Visual Positioning System (VPS) Shadow Mode Daemon
   After=network.target

   [Service]
   Type=notify
   User=pi
   WorkingDirectory=/home/pi/VPS
   ExecStart=/usr/bin/python3 /home/pi/VPS/vps/main.py --shadow --record --save-images --connection /dev/ttyAMA0:921600
   Restart=on-failure
   RestartSec=3
   WatchdogSec=10

   [Install]
   WantedBy=multi-user.target
   ```
2. Enable and start service:
   ```bash
   sudo systemctl daemon-reload
   sudo systemctl enable vps.service
   sudo systemctl start vps.service
   ```

---

## 5. Pass Criteria & Operational Limits

| Milestone | Metric | Target Threshold | Rationale |
|---|---|---|---|
| **Phase 1 (Height)** | Mean Relative Error (0 - 50m) | $\le 8.0\%$ | Prevents vertical overshoot in terrain-relative flight |
| **Phase 1 (Height)** | Low Band Error (0 - 15m) | $\le 0.35\text{ m}$ RMS | Critical landing and low-altitude obstacle clearance |
| **Phase 1 (Height)** | Dropout Duration (Conf < 0.3) | $\le 3.0\text{ s}$ per flight | Guarantees continuous state estimation |
| **Phase 2 (Position)** | VIO Drift Rate (GNSS-Denied) | $\le 3.0\text{ m/min}$ | Bounded short-term inertial dead-reckoning |
| **Phase 2 (Position)** | 2D CEP-95 Radius (Map Match) | $\le 5.0\text{ m}$ | Absolute relocalization accuracy within Aurangabad site |
| **Hardware** | Pi 5 Core Temperature | $< 80^\circ\text{C}$ | Prevents thermal throttling on companion computer |
| **Storage** | Frame Drop Rate | $0\text{ drops}$ ($100\%$ recorded) | Protects replay reproducibility and offline ML training |

---

## 6. Known Limitations & Field Mitigations

1. **Non-RTK GPS Baseline Jitter:** MicroAir GPS has standard positioning variance ($\approx \pm 1.5\text{–}2.5\text{ m}$). On ground evaluation, height is anchored to the physical $6.0\text{ cm}$ landing gear offset and takeoff-relative baro/GPS delta.
2. **High-Altitude Optical Flow Saturation:** Above $50\text{ m}$ AGL, apparent pixel displacement decreases ($v_{px} \propto 1/h$). At higher altitudes, Phase 2 absolute map matching takes precedence over fine-scale flow.
3. **Monsoon / Sun Cloud Shadows:** Rapid cloud shadow transitions can generate non-translational optical flow. The high-rate gyro de-rotation engine and RANSAC geometric outlier filter prune these spurious vectors before EKF update.

