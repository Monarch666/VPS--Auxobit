# Visual Positioning System (VPS) - Phase 1: Height (AGL) Estimator
## Project Implementation Plan

---

### Executive Summary & Objectives
The goal of Phase 1 is to build a robust, production-grade vision-based Above Ground Level (AGL) height estimator for an autonomous unmanned aerial vehicle (UAV). The system fuses downward-facing global-shutter camera imagery (Sony IMX296) with flight controller telemetry (ArduPilot over MAVLink) using de-rotated keyframe optical flow, vertical scale expansion, and barometric altimetry in a 2-state Kalman filter ($[h_{AGL}, z_{terrain}]^T$).

The software must operate locally on the host laptop during development and testing, and deploy seamlessly to a Raspberry Pi 5 (and optionally Jetson Orin) connected to the flight controller. Phase 1 targets height only, leaving clean abstractions for Phase 2 horizontal positioning.

---

### Milestone Breakdown

#### Milestone 1 (M1): Architecture Skeleton, Configuration, Logger, Replay & Geometry Engine
- **M1.1 Project Structure & Environment**: Establish modular directory tree (`vps/camera`, `vps/tracker`, `vps/geometry`, `vps/mavlink`, `vps/fusion`, `vps/tools`, `tests/`, `config/`).
- **M1.2 Configuration System**: Central YAML configuration schema with strict typing (camera intrinsics, baseline physical offsets, tracker parameters, MAVLink endpoints, Kalman noise matrices).
- **M1.3 Core Geometry Engine**:
  - Pinhole camera model & radial-tangential undistortion.
  - Inter-frame de-rotation homography computation: $H_{derot} = K \cdot R \cdot K^{-1}$.
  - Ground-displacement height-from-flow: $H = f_{px} \frac{\|\mathbf{v}_{ground} \cdot \Delta t\|}{\|\Delta \mathbf{p}_{derot}\|}$.
  - Scale-from-expansion during climb/descent: $H_1 = \frac{\Delta H}{1 - s_1/s_2}$.
- **M1.4 Telemetry & Video Logger**: Lightweight binary/HDF5 or synchronized CSV/JSONL format logging frame timestamps, image paths, and interpolated telemetry packets.
- **M1.5 Replay Tooling**: Stream replayer to inject logged or simulated telemetry and imagery back into the processing pipeline deterministically.
- **M1.6 Rigorous Unit Test Suite**: Comprehensive pytest tests for geometric invariants, pure-rotation cancellation, pure-translation scaling, and numerical edge cases.

#### Milestone 2 (M2): Synthetic Aerial Simulation Harness (30 m to 300 m)
- **M2.1 Orthophoto Pinhole Renderer**: Load high-resolution aerial orthophoto / satellite imagery (Deccan plateau landscape: farmland, rocky terrain, built-up areas).
- **M2.2 6-DOF Drone Kinematics Simulator**: Synthetic trajectory generator supporting forward cruise, climb, descent, hover, yaw turns, and wind-induced roll/pitch oscillations.
- **M2.3 Degradation Models**:
  - Motion blur (velocity & exposure convolution).
  - Additive Gaussian & sensor noise.
  - Atmospheric haze / contrast attenuation simulating seasonal conditions in Aurangabad.
- **M2.4 Synthetic Telemetry Generator**: Synchronized MAVLink telemetry streams (gyro rates with bias and noise, NED velocities, barometric altitude with thermal drift).
- **M2.5 Automated Evaluation Suite**: Automated altitude benchmark evaluating raw height-from-flow error across 30 m to 300 m AGL.

#### Milestone 3 (M3): Camera Capture, Calibration & Focus Inspection
- **M3.1 Camera Abstraction Layer**: Support OpenCV V4L2 capture, mock video streamer, and PiCamera2/V4L2 interface for Raspberry Pi 5.
- **M3.2 Checkerboard / ChArUco Calibration Pipeline**: Camera calibration script computing exact focal lengths $(f_x, f_y)$, principal point $(c_x, c_y)$, and distortion coefficients $(k_1, k_2, p_1, p_2, k_3)$.
- **M3.3 Ground Baseline Height Offset Calibration**: Calibration tool to determine the physical optical center ground offset when the drone rests on its landing gear / hand baseline.
- **M3.4 Hyperfocal / Infinity Focus Inspection Tool**: Real-time Laplacian contrast and frequency-domain sharpness inspector to verify lens infinity focus before flight.

#### Milestone 4 (M4): Optical Flow Tracker, Keyframing & De-Rotated Height Benchmark
- **M4.1 Pyramidal Lucas-Kanade Tracker**: Multi-level iterative optical flow tracking Shi-Tomasi corners, with adaptive ORB/FAST feature replenishment.
- **M4.2 Outlier Rejection & Verification**: Forward-backward optical flow consistency check ($<0.5-1.0$ px error) and RANSAC homography/affine geometric pruning.
- **M4.3 Adaptive Keyframe Engine**: Dynamic keyframe selection guaranteeing metric baseline displacement $\ge 10$ px regardless of altitude (30–300 m) or forward velocity.
- **M4.4 Gyro Integration & Frame De-rotation**: Quaternion/Rodrigues integration of high-rate gyro samples synchronized between keyframe and current frame.
- **M4.5 Benchmark Report**: Error-vs-altitude analysis over synthetic and mock flight sequences.

#### Milestone 5 (M5): MAVLink I/O Engine & ArduPilot SITL Integration
- **M5.1 Asynchronous MAVLink I/O Engine**: High-performance thread-safe `pymavlink` bridge for serial/UDP connections.
- **M5.2 Inbound Telemetry Parser & Ring Buffer**:
  - `ATTITUDE`, `HIGHRES_IMU` / `RAW_IMU` (gyro rates at high rate).
  - `LOCAL_POSITION_NED` / `GLOBAL_POSITION_INT` (ground velocities).
  - `SCALED_PRESSURE` / `VFR_HUD` (baro altitude).
  - Telemetry interpolation ring-buffer matching exact frame exposure timestamps.
- **M5.3 Outbound Rangefinder Dispatch**: Formatter for `DISTANCE_SENSOR` messages (downward orientation `PITCH_270`, max 30000 cm, confidence-mapped `signal_quality`) streamed at $\ge 10$ Hz.
- **M5.4 ArduPilot SITL Testing**: Local testing against ArduPilot SITL with documented flight controller parameter set (`RNGFND1_TYPE`, `RNGFND1_MIN_CM`, `RNGFND1_MAX_CM`, `RNGFND1_ORIENT`).

#### Milestone 6 (M6): Scale-from-Expansion, 1D Kalman Fusion & Fallback Logic
- **M6.1 Scale-from-Expansion Engine**: Ratio of feature distance expansion ($s_1 / s_2$) fused with barometric delta $\Delta H$ during climb/descent.
- **M6.2 2-State Kalman Filter**: State vector $\mathbf{x} = [h_{AGL}, z_{terrain}]^T$.
  - State prediction driven by vertical velocity $v_D$ and baro altitude updates ($a = h_{AGL} + z_{terrain}$).
  - Vision measurement update when de-rotated flow or expansion height is valid.
- **M6.3 Confidence & Health Monitoring**: Multi-criteria confidence score based on feature count, inlier ratio, flow vector magnitude, rotation rate, and image contrast.
- **M6.4 Hover & Low-Texture Fallback**: Automatic terrain hold, covariance inflation, and fallback to baro-referenced altitude when visual flow is degenerate.
- **M6.5 Comprehensive Simulation Stress-Testing**: Climb, descent, high-altitude hover, low-texture patch traversals, and baro drift rejection.

#### Milestone 7 (M7): Hand-Held Validation, ArUco Metric Benchmark & Flight Log Replay
- **M7.1 Hand-Held Baseline Height Validation**: Tooling to measure height accurately when the drone is held by hand and moved vertically/laterally relative to calibrated ground.
- **M7.2 ArUco Ground-Truth Marker Estimator**: Metric ArUco / AprilTag detection for ground-truth validation at altitudes $\le 30$ m.
- **M7.3 Full Flight Log Replay & Verification**: Replay end-to-end flight logs, generating synchronized residual plots, covariance envelopes, and error metrics.

#### Milestone 8 (M8): Optional ROS 2 Wrapper & Deployment Documentation
- **M8.1 Thin ROS 2 Node**: Clean wrapper node publishing `sensor_msgs/Range` and custom AGL telemetry without contaminating the core library.
- **M8.2 Target Platform Deployment Guide**: Optimization and configuration instructions for Raspberry Pi 5 (V4L2/libcamera/PiCamera2, serial UART to FC) and Jetson Orin.
- **M8.3 Documentation & User Manual**: Full operational instructions, calibration steps, and parameter tuning guide.

---

### Acceptance Criteria & Verification Matrix
1. **Geometric Precision**: Core de-rotation and height-from-flow unit tests pass with zero error in idealized conditions.
2. **Synthetic Flight Performance**: $\le 5\%$ AGL error across $60-300$ m on textured terrain; graceful degradation with inflated covariance under degenerate conditions (hover, textureless).
3. **Robustness**: Zero `NaN` or unhandled exceptions; continuous $\ge 10$ Hz `DISTANCE_SENSOR` output.
4. **Latency & Throughput**: End-to-end latency $< 100$ ms, processing throughput $\ge 25$ fps on target hardware.

