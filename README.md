# Visual Positioning System (VPS) - Phase 1: Height (AGL) Estimator

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![License: Proprietary](https://img.shields.io/badge/license-Proprietary-red.svg)]()
[![Platform: Raspberry Pi 5 / Jetson Orin](https://img.shields.io/badge/platform-Raspberry%20Pi%205%20%7C%20Jetson%20Orin-brightgreen.svg)]()
[![ArduPilot: MAVLink 2](https://img.shields.io/badge/ArduPilot-MAVLink%202-orange.svg)](https://ardupilot.org/)

A real-time, vision-based Above Ground Level (AGL) height estimator for unmanned aerial vehicles (UAVs). Designed for autonomous flight over the Deccan Plateau (Aurangabad, Maharashtra), the system fuses downward-facing global-shutter camera imagery (Sony IMX296) with flight controller telemetry (ArduPilot over MAVLink) using gyro de-rotated optical flow, scale expansion, and barometric altimetry in a 2-State Kalman Filter ($[h_{AGL}, z_{terrain}]^T$).

---

## Key Physics & Estimation Architecture

Ground Sample Distance relates height $H$ to camera focal length $f_{px}$ by:
$$\text{GSD} = \frac{H}{f_{px}}$$

1. **Keyframe Optical Flow + Gyro De-Rotation**:
   Inter-frame camera rotation is cancelled using high-rate gyro integration via the projective homography:
   $$H_{rot} = K R K^{-1}$$
   Translational ground displacement $\|\Delta \mathbf{d}_{ground}\| = \|\mathbf{v}_{ground} \cdot \Delta t\|$ is related to de-rotated image flow $\|\Delta \mathbf{p}_{trans}\|$ by:
   $$H = f_{px} \frac{\|\Delta \mathbf{d}_{ground}\|}{\|\Delta \mathbf{p}_{trans}\|}$$
   An adaptive keyframe manager enforces a metric baseline guaranteeing $\ge 10\text{ px}$ flow displacement across all altitudes ($30\text{ m}$ to $300\text{ m}$).

2. **Scale Change During Climb/Descent**:
   During vertical motion, feature spacing expansion ratio $s_1 / s_2$ combined with barometric altitude delta $\Delta H = H_1 - H_2$ recovers absolute metric scale:
   $$H_1 = \frac{\Delta H}{1 - s_1 / s_2}$$

3. **2-State Terrain-Relative Kalman Filter**:
   State vector $\mathbf{x} = [h_{AGL}, z_{terrain}]^T$, where baro-referenced altitude $a = h_{AGL} + z_{terrain}$.
   - **Cruise Mode**: Fuses vision height with baro altitude; estimates both AGL height and local terrain elevation variations.
   - **Hover / Low-Texture Fallback**: When horizontal speed drops below $0.5\text{ m/s}$ or image contrast is lost, the filter holds $z_{terrain}$ constant, propagates $h = a_{baro} - z_{terrain}$, inflates process noise covariance by $10\times$, and smoothly decays confidence without filter divergence.

4. **Landing-Gear Ground Baseline Calibration**:
   A calibrated optical-center-to-landing-gear clearance ($15\text{ cm}$) is subtracted from all measurements so that ground contact reads exactly $0.000\text{ m}$ AGL.

---

## Hardware Specifications

| Component | Specification | Description / Notes |
|:---|:---|:---|
| **Camera Sensor** | Sony IMX296 Global Shutter | $1456 \times 1088$ resolution @ 30 fps |
| **Optics** | 6mm CS-Mount / C-Mount | HFOV $62.2^\circ$, VFOV $48.8^\circ$ ($f_x \approx 1207\text{ px}, f_y \approx 1199\text{ px}$) |
| **Companion Computer** | Raspberry Pi 5 (4GB / 8GB) | Debian 12 Bookworm (64-bit), ARM Cortex-A76 |
| **Flight Controller** | ArduPilot (Cube / Pixhawk) | ArduCopter 4.4+ over TELEM2 hardware UART |
| **MAVLink Protocol** | MAVLink 2 (`SERIAL2_PROTOCOL=2`) | Baudrate 921,600 baud (`SERIAL2_BAUD=921`) |
| **Output Message** | `DISTANCE_SENSOR` (#132) | Downward (`PITCH_270`), $10\text{ cm} - 300\text{ m}$, streamed at $\ge 15\text{ Hz}$ |

---

## Repository Structure

```
VPS/
├── config/
│   ├── default_config.yaml       # Master configuration (camera, tracker, fusion, MAVLink)
│   ├── imx296_calibrated.yaml    # Calibrated intrinsics & distortion coefficients
│   └── sitl_params.param         # ArduPilot SITL / FC rangefinder parameter dump
├── vps/
│   ├── camera/                   # Capture abstraction, checkerboard calibrator, focus inspector
│   ├── geometry/                 # Pinhole model, gyro de-rotation, flow & expansion engines
│   ├── tracker/                  # Pyramidal LK tracker, adaptive keyframe manager, pipeline
│   ├── mavlink/                  # Telemetry ring buffer, MAVLink bridge, DISTANCE_SENSOR sender
│   ├── fusion/                   # Confidence evaluator, 2-State Kalman filter, master engine
│   ├── ros2/                     # Optional decoupled ROS 2 wrapper node (sensor_msgs/Range)
│   ├── tools/                    # Simulators, SITL harness, ArUco validator, log replayer
│   └── main.py                   # Master production CLI runner
├── reports/                      # Generated benchmark plots & diagnostic reports (M1-M7)
├── tests/                        # Full pytest test suite (48 tests across 8 test modules)
├── DEPLOYMENT_PI5.md             # Complete Raspberry Pi 5 hardware & systemd setup guide
├── ARCHITECTURE.md               # Mathematical formulation and coordinate frame conventions
├── PLAN.md                       # 8-Milestone implementation plan
└── RULES.md                      # Mandatory development and safety rules
```

---

## Quickstart Guide

### 1. Environment Setup

```bash
# Clone the repository
git clone https://github.com/your-repo/VPS.git
cd VPS

# Create and activate virtual environment
python -m venv venv
# On Windows:
.\venv\Scripts\activate
# On Linux / Raspberry Pi:
source venv/bin/activate

# Install dependencies
pip install numpy opencv-python pymavlink pyyaml scipy matplotlib pytest
```

### 2. Run the Full Test Suite

Run all 48 unit tests verifying geometry, calibration, tracking, Kalman filtering, MAVLink serialization, ArUco detection, and flight log replay:

```bash
pytest tests/ -v
```

### 3. Run Standalone Height Estimator

```bash
# Autonomous flight mode with connected camera and ArduPilot serial connection
python vps/main.py --config config/default_config.yaml --connection /dev/ttyAMA0:921600 --camera /dev/video0 --mode flight

# Hand-held testing HUD mode (displays live height above ground with 15 cm baseline subtraction)
python vps/main.py --connection udpin:127.0.0.1:14550 --camera 0 --mode handheld

# SITL simulation mode (emulates companion computer connected to local SITL drone)
python vps/main.py --mode sitl
```

---

## Key Tools & Diagnostic Utilities

### A. Pre-Flight Infinity Focus Inspector
Verify your lens is focused to optical infinity before mounting on the drone:
```bash
python vps/tools/focus_inspector.py --device 0
```
Displays live Laplacian variance, Tenengrad gradients, and 2D FFT high-frequency ratios across a 9-zone spatial grid.

### B. Checkerboard Camera Calibration
Calibrate your IMX296 camera and generate `config/imx296_calibrated.yaml`:
```bash
python vps/tools/calibrate_camera.py --image-dir data/calibration_images/
```

### C. ArUco Metric Low-Altitude Validation
Benchmark millimeter-accurate pose estimation and baseline subtraction at $\le 30\text{ m}$:
```bash
python vps/tools/aruco_validator.py
```
Output: [reports/m7_aruco_benchmark.png](file:///e:/VPS/reports/m7_aruco_benchmark.png)

### D. Real-Flight Log Replayer
Replay real ArduPilot `.bin` flight logs through the full vision + telemetry Kalman filter:
```bash
python vps/tools/flight_log_replayer.py
```
Output: [reports/m7_flight_log_replay.png](file:///e:/VPS/reports/m7_flight_log_replay.png)

### E. ArduPilot SITL Integration Harness
Verify two-way MAVLink communication, packet rate ($\ge 15\text{ Hz}$), and ArduPilot rangefinder reception:
```bash
python vps/tools/sitl_harness.py
```
Output: [reports/m5_mavlink_sitl_report.png](file:///e:/VPS/reports/m5_mavlink_sitl_report.png)

---

## Optional ROS 2 Node Wrapper

For robotics teams operating ROS 2 (Humble, Iron, or Jazzy), launch the decoupled wrapper node:

```bash
# Set ROS 2 environment and run node
source /opt/ros/humble/setup.bash
python -m vps.ros2.vps_node --ros-args -p baseline_offset_m:=0.15 -p camera_topic:=/camera/image_raw
```

**Subscribed Topics:**
- `/camera/image_raw` (`sensor_msgs/msg/Image`)
- `/mavros/imu/data` (`sensor_msgs/msg/Imu`)
- `/mavros/local_position/velocity_body` (`geometry_msgs/msg/TwistStamped`)

**Published Topic:**
- `/vps/height_agl` (`sensor_msgs/msg/Range`, downward orientation, $0.10\text{ m} - 300\text{ m}$)

---

## Milestone Summary & Benchmark Results

| Milestone | Module | Primary Result | Status |
|:---|:---|:---|:---:|
| **M1** | Architecture & Geometry | Pinhole, de-rotation homography, scale expansion | **PASSED** |
| **M2** | Synthetic Simulator | Deccan orthophoto simulator, error $< 0.4\%$ (30-300m) | **PASSED** |
| **M3** | Calibration & Focus | MRE $0.081\text{ px}$, 9-zone focus inspector, baseline offset | **PASSED** |
| **M4** | Keyframe Optical Flow | Adaptive baseline ($\ge 10\text{ px}$), throughput $30-43\text{ FPS}$ | **PASSED** |
| **M5** | MAVLink SITL Bridge | Bidirectional loopback, `DISTANCE_SENSOR` @ $16.5\text{ Hz}$ | **PASSED** |
| **M6** | 2-State Kalman Fusion | MAE $0.14\%$, hover hold, fog patch, baro drift rejection | **PASSED** |
| **M7** | Flight Replay & ArUco | Real Deccan log replay (MAE $0.11\text{ m}$), ArUco sub-mm accuracy | **PASSED** |
| **M8** | Production Daemon & ROS 2 | Standalone CLI, decoupled ROS 2 node, Pi 5 deployment guide | **PASSED** |

---

## Target Hardware Deployment

For step-by-step instructions on wiring, UART configuration, `/boot/firmware/config.txt` camera overlays, and setting up the autostart `vps.service` systemd daemon on Raspberry Pi 5, refer to [DEPLOYMENT_PI5.md](file:///e:/VPS/DEPLOYMENT_PI5.md).
