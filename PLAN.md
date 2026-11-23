# Visual Positioning System (VPS) - Phase 1 (Height AGL) & Phase 2 (Horizontal GNSS-Denied Navigation)
## Complete Project Implementation Plan (10 m to 400 m AGL)

---

### Executive Summary & Scope
This plan details the full engineering roadmap for a plug-and-play **Visual Positioning System (VPS)** module designed for an autonomous UAV operating over the Deccan Plateau (Aurangabad, Maharashtra). The system operates on a **Raspberry Pi 5** companion computer paired with a downward-facing **Sony IMX296 global-shutter camera** ($1456 \times 1088$ @ 30 fps) and communicates bidirectionally with an **ArduPilot flight controller** over MAVLink 2 (921,600 baud).

The project is structured into two sequential phases with a strict approval gate between them:
- **Phase 1: Height (AGL) Estimator (10 m to 400 m)**: Fuses keyframe optical flow, gyro de-rotation, scale expansion, and barometric altimetry in a 2-State Kalman Filter ($[h_{AGL}, z_{terrain}]^T$). Outputs MAVLink `DISTANCE_SENSOR` at $\ge 15\text{ Hz}$.
- **Phase 2: Horizontal Position & Velocity for GNSS-Denied Navigation**:
  - **2a. Relative Visual-Inertial Odometry (VIO)**: Metric velocity ($v_N, v_E$) from de-rotated flow and planar homography decomposition fused with FC IMU and compass in an Error-State EKF (ES-EKF).
  - **2b. Absolute Map Matching**: Offline GeoTIFF/orthophoto multi-scale feature database + runtime orthorectification, feature matching (ORB/AKAZE + RANSAC), phase correlation fallback, and delayed-state EKF fusion.
  - Outputs MAVLink `ODOMETRY` / `VISION_POSITION_ESTIMATE` at 20–30 Hz (or `GPS_INPUT`).

---

### Raspberry Pi 5 Target Architecture & Constraints
1. **Compute & Threading Budget**:
   - Total CPU load $< 80\%$ on 4-core Cortex-A76 (no GPU / no CUDA).
   - Vision tracking loop $\ge 25\text{ fps}$ (target $30\text{ fps}$), end-to-end latency $< 100\text{ ms}$.
   - Map matching loop $\ge 1\text{ Hz}$.
   - Multi-process architecture using shared memory (`multiprocessing.shared_memory`) with CPU core pinning:
     - Core 0: libcamera / Picamera2 zero-copy capture & V4L2 ingestion.
     - Core 1: Feature tracking & gyro de-rotation (OpenCV with ARM NEON optimization at $728 \times 544$ scaled resolution).
     - Core 2: Map matching engine (orthorectification & feature database lookup).
     - Core 3: MAVLink I/O, ES-EKF sensor fusion, and system health watchdog.
2. **Thermal & Power Monitoring**:
   - Continuous polling of CPU throttling, temperature (`vcgencmd measure_temp`), and undervoltage flags (`vcgencmd get_throttled`).
3. **Sub-Millisecond Time Synchronization**:
   - Monotonic hardware sensor timestamps from libcamera synced to FC clock via MAVLink `TIMESYNC` / `SYSTEM_TIME`.

---

### 12-Milestone Roadmap

```mermaid
flowchart TD
    subgraph PHASE 1: Height AGL Estimator (10 m to 400 m)
        M1["M1: Skeleton, Perf Harness & Geometry Tests (10-400m)"]
        M2["M2: Synthetic Simulator & Appearance Gap Engine"]
        M3["M3: IMX296 Capture, Checkerboard Calibration & Focus"]
        M4["M4: Keyframe Tracker, De-Rotation & 10-400m Benchmark"]
        M5["M5: MAVLink I/O & SITL Loopback (DISTANCE_SENSOR >= 15Hz)"]
        M6["M6: Scale Expansion, Kalman Fusion & Hover Fallback"]
        GATE{"PHASE 1 ACCEPTANCE GATE<br/>(Simulator <= 5% Error, User Approval)"}
    end

    subgraph PHASE 2: Horizontal Position & Velocity (GNSS-Denied)
        M7["M7: VIO Velocity/Position & Error-State EKF (IMU+Compass)"]
        M8["M8: GeoTIFF Map Database & Tiled Multi-Scale Lookup"]
        M9["M9: Map Matcher, Delayed-State Fusion & Relocalisation"]
        M10["M10: MAVLink ODOMETRY/VISION & SITL EKF Integration"]
        M11["M11: Real-Flight Replay, ArUco & DEM/GPS Ground-Truth"]
        M12["M12: Pi 5 Multi-Process Optimization, Systemd & ROS 2"]
    end

    M1 --> M2 --> M3 --> M4 --> M5 --> M6 --> GATE
    GATE -->|Approved| M7 --> M8 --> M9 --> M10 --> M11 --> M12
```

---

### Detailed Milestone Breakdown

#### Milestone 1 (M1): Architecture Skeleton, Multi-Process Harness, Geometry Engine & Pi 5 Budget
- Establish modular directory layout including `vps/vio`, `vps/map_db`, `vps/map_matcher`, `vps/nav_ekf`, `vps/health`.
- Mathematical validation of GSD and optical flow across the full **10 m to 400 m** envelope:
  - At $10\text{ m}$: $\text{GSD} \approx 0.83\text{ cm/px}$, $10\text{ px}$ baseline $= 8.3\text{ cm}$ ($\Delta t \approx 0.008\text{ s}$ @ $10\text{ m/s}$).
  - At $400\text{ m}$: $\text{GSD} \approx 33.1\text{ cm/px}$, $10\text{ px}$ baseline $= 3.31\text{ m}$ ($\Delta t \approx 0.33\text{ s}$ @ $10\text{ m/s}$).
- Raspberry Pi 5 benchmark harness measuring OpenCV NEON execution times and establishing the latency budget table.
- Comprehensive unit test suite for pinhole projection, gyro de-rotation, and scale expansion.

#### Milestone 2 (M2): Synthetic Aerial Simulator with Seasonal Appearance Gap
- Procedural multi-class Deccan Plateau orthophoto ($4096 \times 4096$) with farmland, rocky hills, and villages.
- Dual-appearance engine: renders live camera view (e.g. monsoon lush green, haze) and a degraded reference map (dry season, color shift, shadow changes, lower GSD).
- 6-DOF drone kinematics simulator covering altitudes $10\text{ m}$ to $400\text{ m}$, velocities $0$ to $20\text{ m/s}$, roll/pitch bank angles up to $25^\circ$, and synthetic noisy IMU (gyro + accel), compass (MAG), and baro.

#### Milestone 3 (M3): Camera Capture, Calibration & Infinity-Focus Inspector
- Sony IMX296 driver abstraction supporting Picamera2 / V4L2 zero-copy buffers.
- Checkerboard / ChArUco calibration tool exporting calibrated intrinsics $K$ and distortion $D$.
- Ground baseline height offset calibrator (landing-gear clearance $15\text{ cm}$).
- Pre-flight infinity focus verification tool (Laplacian variance, Tenengrad, and 2D FFT spectral ratio across a 9-zone grid).

#### Milestone 4 (M4): Pyramidal Feature Tracker, Keyframing & 10–400m Height Benchmark
- Shi-Tomasi corner detector with adaptive FAST/ORB replenishment capped at 300–500 features.
- Bidirectional pyramidal Lucas-Kanade optical flow with forward-backward error threshold ($\le 1.0\text{ px}$) and RANSAC homography.
- Adaptive Keyframe Manager dynamically scaling baseline distance $D_{ground} \ge \frac{10 H}{f_{px}}$ across $10\text{ m}$ to $400\text{ m}$.
- High-rate gyro de-rotation ($H_{rot} = K R K^{-1}$) eliminating roll/pitch rotational flow.
- Continuous 10–400 m altitude benchmark report.

#### Milestone 5 (M5): MAVLink I/O & ArduPilot SITL Integration
- Asynchronous MAVLink 2 bridge over serial UART (921,600 baud) and UDP loopback.
- Inbound telemetry ring buffer interpolating `ATTITUDE`, `HIGHRES_IMU`, `LOCAL_POSITION_NED`, `GLOBAL_POSITION_INT`, and `VFR_HUD`.
- Outbound `DISTANCE_SENSOR` dispatcher streaming downward rangefinder packets ($10\text{ cm} - 40,000\text{ cm}$) at $\ge 15\text{ Hz}$.
- Integration test against ArduPilot SITL verifying rangefinder parameter reception.

#### Milestone 6 (M6): Scale Expansion, 2-State Kalman Fusion & Fallback Logic
- Scale-from-expansion engine: feature distance expansion ratio ($s_1 / s_2$) fused with barometric delta $\Delta H$.
- 2-State Kalman Filter: state $\mathbf{x} = [h_{AGL}, z_{terrain}]^T$.
  - State prediction using vertical velocity $v_D$.
  - Barometric update ($a = h + z_{terrain}$).
  - Vision measurement update with dynamic measurement variance $R_{vis} = R_{base} / C^2$ and $\chi^2$ innovation gating.
  - Hover & textureless fallback: locks terrain elevation $z_{terrain}$, propagates $h = a - z_{terrain}$, inflates covariance by $10\times$.
- Simulator evaluation across cruise, climb, descent, hover, and baro drift.
- **PHASE 1 ACCEPTANCE GATE**: Must achieve $\le 5\%$ error from $60\text{ m}$ to $400\text{ m}$ on textured terrain. **Stop and wait for user approval before Phase 2.**

---

#### Milestone 7 (M7): Visual-Inertial Odometry (VIO) & Error-State EKF (Phase 2 Start)
- Metric horizontal velocity calculation: $\mathbf{v}_{body} = \frac{h_{AGL}}{f_{px}} \cdot \mathbf{u}_{trans\_flow}$.
- Planar homography decomposition alternative for velocity recovery.
- Error-State Extended Kalman Filter (ES-EKF) fusing:
  - Nominal state: $[\mathbf{p}_{NED}, \mathbf{v}_{NED}, \mathbf{q}_{body\_world}, \mathbf{b}_{accel}, \mathbf{b}_{gyro}]^T$.
  - Compass heading update from magnetometer / FC yaw.
  - VIO velocity measurement updates.
- Velocity drift characterization report.

#### Milestone 8 (M8): GeoTIFF Map Database & Tiled Multi-Scale Lookup
- Offline ingestion tool for georeferenced GeoTIFF orthomosaics.
- Multi-scale tiling engine (pyramidal image tiles with geographical bounds).
- Pre-computed spatial index (ORB/AKAZE descriptors + gradient maps per tile).
- Runtime bounding box query driven by ES-EKF position and covariance.

#### Milestone 9 (M9): Absolute Map Matcher & Delayed-State EKF Fusion
- Runtime orthorectification of camera frame to map coordinate frame:
  - De-tilts using camera attitude $(R_{roll}, R_{pitch})$.
  - Rotates to North using compass heading $(R_{yaw})$.
  - Scales to map GSD using Phase 1 AGL height ($h_{AGL}$).
- Robust matching: ORB/AKAZE with RANSAC homography, falling back to phase correlation.
- Appearance gap compensation (contrast normalization, gradient orientation maps).
- Delayed-State EKF update: buffers historical states, applies delayed absolute position fix at exact frame timestamp, re-propagates forward.
- Outlier innovation gating ($\chi^2$ test) and loss-of-fix relocalization mode.

#### Milestone 10 (M10): MAVLink ODOMETRY / VISION Output & Health Gate
- Dispatches MAVLink `ODOMETRY` (#331) or `VISION_POSITION_ESTIMATE` (#102) with full $6\times 6$ covariance matrices at 20–30 Hz.
- Optional `GPS_INPUT` (#232) emulation mode.
- System safety health gate: monitors fix confidence, data age, camera FPS, undervoltage, and CPU temperature. Defaults to advisory mode unless safety gate passes.
- SITL verification with ArduPilot EKF3 external navigation parameters (`EK3_SRC1_POSXY = 6`, `EK3_SRC1_VELXY = 6`).

#### Milestone 11 (M11): Real-Flight Log Replay, ArUco & DEM/GPS Ground-Truth Benchmark
- Ingests real ArduPilot flight logs (`log_19_*.bin`, `log_36_*.bin`).
- ArUco metric low-altitude validation ($\le 30\text{ m}$) with landing gear baseline subtraction.
- Trajectory ground-truth comparison against GPS position and DEM elevation.
- Comprehensive diagnostic report generation (`reports/m11_gnss_denied_navigation_report.png`).

#### Milestone 12 (M12): Raspberry Pi 5 Optimization, Packaging & ROS 2 Wrapper
- Multi-process pipeline profiling (`perf`, `cProfile`) ensuring total CPU $< 80\%$ and latency $< 100\text{ ms}$.
- `systemd` autonomous service setup (`vps.service`) with watchdog monitoring.
- Optional ROS 2 wrapper publishing `sensor_msgs/Range` and `nav_msgs/Odometry`.
- Production README and hardware deployment manual.

---

### Acceptance Criteria
1. **Phase 1 (Height)**:
   - $\le 5\%$ altitude error across $60\text{ m}$ to $400\text{ m}$ on textured imagery in simulator.
   - Smooth degradation with terrain elevation hold in hover ($< 0.5\text{ m/s}$) and textureless patches.
   - Continuous $\ge 15\text{ Hz}$ `DISTANCE_SENSOR` output with zero `NaN`s.
2. **Phase 2 (Horizontal GNSS-Denied Navigation)**:
   - Absolute position error bounded to $\le 2\times \text{GSD}_{map} + \text{VIO drift between fixes}$ with appearance gap active.
   - Robust outlier rejection (zero false fixes accepted during bad matches).
3. **Hardware Constraints**:
   - Tracking loop $\ge 25\text{ fps}$, latency $< 100\text{ ms}$, total Pi 5 CPU $< 80\%$.
