# Visual Positioning System (VPS) - Plug-and-Play GNSS-Denied UAV Navigation

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/)
[![Platform: Raspberry Pi 5](https://img.shields.io/badge/platform-Raspberry%20Pi%205%20(64--bit)-brightgreen.svg)]()
[![Flight Controller: ArduPilot 4.5.7](https://img.shields.io/badge/ArduPilot-4.5.7%20(MicroAir%20405v2)-orange.svg)](https://ardupilot.org/)
[![MAVLink 2](https://img.shields.io/badge/MAVLink-2.0%20(921600%20baud)-blueviolet.svg)]()
[![License: Proprietary](https://img.shields.io/badge/license-Proprietary-red.svg)]()

A production-grade, real-time **Visual Positioning System (VPS)** designed for autonomous unmanned aerial vehicle (UAV) navigation in GNSS-denied environments over the Deccan Plateau (Aurangabad, Maharashtra). 

Deployed on a **Raspberry Pi 5** companion computer linked via high-speed MAVLink 2 (921,600 baud UART) to an **ArduPilot 4.5.7** flight controller (MicroAir 405v2 board), the system delivers:
- **Phase 1 (AGL Height)**: Continuous Above-Ground-Level altimetry ($10\text{ m}$ to $400\text{ m}$) via gyro de-rotated keyframe optical flow, scale expansion, and a 2-State Kalman Filter ($[h_{AGL}, z_{terrain}]^T$).
- **Phase 2 (Horizontal Positioning & Velocity)**: 6-DOF Visual-Inertial Odometry ($15$-State Error-State EKF) coupled with periodic absolute georeferenced satellite map matching against an offline multi-scale Sentinel-2 GeoTIFF database.
- **Precision Landing**: Sub-$2\text{ cm}$ metric height recovery and exact $0.00\text{ m}$ ground contact verification via planar ArUco fiducials ($0.15\text{ m}$ to $30.0\text{ m}$).

---

## 1. High-Level System Architecture

```mermaid
graph TD
    subgraph Sensors & Flight Controller [Hardware Layer]
        CAM["Sony IMX296 Global Shutter Camera<br/>1456x1088 @ 30 FPS (728x544 tracking)"]
        FC["ArduPilot 4.5.7 on MicroAir 405v2<br/>TELEM2 UART @ 921,600 baud"]
        SENSORS["FC Sensors: 100 Hz IMU (Accel/Gyro), Compass (Mag), Baro"]
    end

    subgraph Core Companion Pipeline [Raspberry Pi 5 - Cores 1, 2, 3]
        CAP["libcamera / V4L2 Zero-Copy Ingest<br/>Hardware CLOCK_MONOTONIC sync"]
        RING["Telemetry Ring Buffer (100 Hz)<br/>TIMESYNC + Slerp Interpolation"]
        TRACK["Pyramidal LK Tracker (Shi-Tomasi)<br/>Capped 350 pts + NEON Acceleration"]
        DEROT["Gyro De-Rotation Homography<br/>H_rot = K * R * K^-1"]
        VIO["VIO Metric Velocity Engine<br/>v = flow * H / f_px"]
        ES_EKF["15-State Error-State EKF<br/>Pos, Vel, Quat, Accel/Gyro Biases"]
        MAP_DB["Tiled GeoTIFF Database<br/>Sentinel-2 Aurangabad (GSD 15.62m/px)"]
        MATCHER["Multi-Scale Map Matcher (1-2 Hz)<br/>Orthorectify + ORB + FFT Phase Corr"]
        DELAY["Delayed-State EKF Fusion<br/>Latency Buffer + Chi^2 Innovation Gating"]
    end

    subgraph Supervision & Dispatcher [Systemd & Watchdog]
        WATCHDOG["Hardware Watchdog Supervisor<br/>FPS, Temp, Throttling & Telemetry Guard"]
        HEALTH["Safety Gate & Covariance Inflation<br/>Advisory Log vs Active Feeding Mode"]
        MAV_OUT["MAVLink Dispatcher (25 Hz)<br/>ODOMETRY (#331) & DISTANCE_SENSOR (#132)"]
    end

    CAM --> CAP --> TRACK --> DEROT --> VIO --> ES_EKF
    FC --> SENSORS --> RING --> DEROT
    RING --> ES_EKF
    ES_EKF --> MATCHER
    MAP_DB --> MATCHER --> DELAY --> ES_EKF
    ES_EKF --> HEALTH --> MAV_OUT --> FC
    CAP --> WATCHDOG
    RING --> WATCHDOG
```

---

## 2. Hardware Wiring & Physical Pinouts

### Raspberry Pi 5 to MicroAir 405v2 (TELEM2 UART)

| Raspberry Pi 5 Pin | Function | MicroAir 405v2 (TELEM2) Pin | Notes |
|:---|:---|:---|:---|
| **Pin 8 (GPIO 14)** | UART0 TXD | **TELEM2 RX** | Companion TX $\rightarrow$ FC RX |
| **Pin 10 (GPIO 15)**| UART0 RXD | **TELEM2 TX** | Companion RX $\leftarrow$ FC TX |
| **Pin 6 or 9** | Ground (GND) | **TELEM2 GND** | Common reference ground |

> [!CAUTION]
> **Logic Level Warning**: Raspberry Pi 5 GPIO pins operate strictly at **3.3V logic**. The MicroAir 405v2 TELEM2 port uses 3.3V UART signals. **Never** supply 5V directly to Raspberry Pi GPIO pins.

---

## 3. Raspberry Pi 5 Performance Budget Compliance

Every pipeline stage is profiled against the single-core Cortex-A76 performance budget. The multi-process pipeline pins execution across Cores 1–3, keeping Core 0 dedicated to OS kernel interrupts and MAVLink networking:

| Performance Metric | Measured Value | Target Hardware Budget | Status |
|:---|:---|:---|:---|
| **Tracking Loop Frame Rate** | **81.9 FPS** | $\ge 25.0\text{ FPS}$ | **PASSED** |
| **End-to-End Latency** | **27.2 ms** | $< 100.0\text{ ms}$ | **PASSED** |
| **Map Matching Latency** | **45.7 ms** | $< 100.0\text{ ms}$ (async) | **PASSED** |
| **Total Multi-Core CPU Load** | **13.2%** | $< 80.0\%$ (4-core total) | **PASSED** |
| **Total Process RAM (RSS)** | **216.9 MB** | $< 250.0\text{ MB}$ ($< 5\%$ of 4GB) | **PASSED** |

---

## 4. Quickstart & Deployment

### 1. Automated Installation on Raspberry Pi OS 64-bit

```bash
# Clone the repository
git clone https://github.com/your-org/VPS.git /home/pi/VPS
cd /home/pi/VPS

# Run the production installer (configures config.txt, installs apt/pip dependencies, registers systemd service)
sudo ./packaging/install.sh

# Reboot to apply Sony IMX296 camera and hardware UART overlays
sudo reboot
```

### 2. Ingest Satellite Map GeoTIFF

Convert any georeferenced GeoTIFF (e.g. user-provided Aurangabad Sentinel-2 image) into multi-scale fast runtime tiles:

```bash
python -m vps.map_db.tiler \
    --input 2026-11-14-00_00_2026-11-14-23_59_Sentinel-2_L2A_True_color.tiff \
    --output data/map_db/aurangabad_s2 \
    --name aurangabad_sentinel2 \
    --tile-size 512
```

### 3. ArduPilot 4.5.7 Parameter Configuration

Upload the validated configuration file [config/ardupilot_vps_params.param](file:///E:/VPS/config/ardupilot_vps_params.param) to your MicroAir 405v2 using **Mission Planner** or **QGroundControl**:

```ini
# Key External Navigation & Rangefinder Parameters
SERIAL2_PROTOCOL = 2        # MAVLink 2
SERIAL2_BAUD     = 921      # 921,600 baud
EK3_SRC1_POSXY   = 6        # External Nav (Visual Odometry / Map Matching)
EK3_SRC1_VELXY   = 6        # External Nav Velocity
EK3_SRC1_POSZ    = 1        # Baro (with Rangefinder / Optical Flow fusion)
VISO_TYPE        = 1        # MAVLink External Navigation backend
VISO_POS_M_NSE   = 0.20     # Position measurement noise (m)
VISO_VEL_M_NSE   = 0.15     # Velocity measurement noise (m/s)
VISO_DELAY_MS    = 80       # Matching & transmission latency compensation
RNGFND1_TYPE     = 10       # MAVLink Rangefinder backend
RNGFND1_ORIENT   = 25       # PITCH_270 (Downward facing)
RNGFND1_GNDCLEAR = 15       # 15 cm landing gear baseline offset
```

### 4. Running the Daemon

```bash
# Production systemd daemon (autostarts on boot with hardware watchdog):
sudo systemctl start vps.service
sudo systemctl status vps.service

# Live streaming logs:
journalctl -u vps.service -f

# Interactive hand-held field test HUD (with ArUco ground verification):
python -m vps.main --mode handheld --connection /dev/ttyAMA0:921600 --camera /dev/video0
```

---

## 5. Verification & Test Suite

The comprehensive test suite covers all 12 milestones (geometry, tracker, EKF, map database, delayed-state matcher, MAVLink serialization, ArUco ground truth, and packaging):

```bash
pytest tests/ -v
```
**Results**: **83 passed, 1 skipped (hardware SITL socket), 0 failed** in 54 seconds.

---

## 6. Project Milestones & Verification Reports

- **M1–M6**: Phase 1 AGL Height Fusion ($10\text{ m} - 400\text{ m}$) $\rightarrow$ [reports/phase1_acceptance_10_to_400m.png](file:///E:/VPS/reports/phase1_acceptance_10_to_400m.png)
- **M7**: 15-State ES-EKF & VIO Metric Velocity $\rightarrow$ [reports/m7_vio_drift_report.png](file:///E:/VPS/reports/m7_vio_drift_report.png)
- **M8**: Zero-GDAL GeoTIFF Ingestion & Tiled Runtime DB $\rightarrow$ `data/map_db/aurangabad_s2`
- **M9**: Absolute Map Matcher & Delayed-State EKF $\rightarrow$ [reports/m9_map_match_report.png](file:///E:/VPS/reports/m9_map_match_report.png)
- **M10**: 25 Hz MAVLink ODOMETRY & ArduPilot 4.5.7 Params $\rightarrow$ [reports/m10_real_flight_log_verification.png](file:///E:/VPS/reports/m10_real_flight_log_verification.png)
- **M11**: Real-Flight Log Replay & ArUco Benchmark $\rightarrow$ [reports/m11_real_flight_benchmark.png](file:///E:/VPS/reports/m11_real_flight_benchmark.png) & [reports/M11_FLIGHT_EVALUATION_REPORT.md](file:///E:/VPS/reports/M11_FLIGHT_EVALUATION_REPORT.md)
- **M12**: Pi 5 Profiler, Watchdog & Packaging $\rightarrow$ [reports/pi5_profiling_benchmark.png](file:///E:/VPS/reports/pi5_profiling_benchmark.png) & [reports/pi5_performance_budget.md](file:///E:/VPS/reports/pi5_performance_budget.md)
