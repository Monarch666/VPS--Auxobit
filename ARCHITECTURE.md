# Visual Positioning System (VPS) - Phase 1 (Height AGL) & Phase 2 (Horizontal GNSS-Denied Navigation)
## System Architecture & Mathematical Specification (10 m to 400 m AGL)

---

### 1. High-Level System Architecture

The VPS system provides autonomous navigation in GNSS-denied environments. It combines high-rate relative visual-inertial state estimation with periodic absolute georeferenced satellite map matching, operating on a Raspberry Pi 5 companion computer connected to an ArduPilot flight controller.

```mermaid
graph TD
    subgraph Sensors & Hardware [Physical Layer]
        CAM["Sony IMX296 Global Shutter Camera<br/>(1456x1088 @ 30 fps / 728x544 tracking)"]
        FC["ArduPilot Flight Controller<br/>(MAVLink 2 Serial @ 921,600 baud)"]
        IMU_COMPASS["Sensors: IMU (Gyro/Accel), Compass (Mag), Baro"]
    end

    subgraph Ingestion & Synchronization [Core 0 & Shared Memory]
        CAP["Camera Driver / Zero-Copy Ingestion<br/>(libcamera / Picamera2 / V4L2)"]
        SYNC["Hardware Clock Synchronization<br/>(CLOCK_MONOTONIC to FC via TIMESYNC)"]
        TS_BUF["Telemetry Ring Buffer (100 Hz)<br/>(ATTITUDE, HIGHRES_IMU, COMPASS, VFR_HUD)"]
    end

    subgraph Phase 1: Height (AGL) Estimation [Core 1]
        TRACK["Pyramidal LK Feature Tracker<br/>(Shi-Tomasi + NEON + Forward-Backward)"]
        KEY["Adaptive Keyframe Selector<br/>(Metric Baseline >= 10 px for 10-400m)"]
        DEROT["Gyro De-rotation Engine<br/>(H_rot = K * R * K^-1)"]
        SCALE_EXP["Scale-from-Expansion Engine<br/>(Vertical: s1/s2 + dH_baro)"]
        HEIGHT_KF["2-State Height Kalman Filter<br/>State: [h_AGL, z_terrain]^T"]
        HOVER_HOLD["Hover / Low-Texture Fallback<br/>(Hold z_terrain, inflate covariance)"]
    end

    subgraph Phase 2: Horizontal Navigation [Core 1, 2, 3]
        VIO_ENG["Visual Odometry Engine<br/>(Metric velocity v = flow * h / f_px)"]
        ES_EKF["Error-State EKF (15-State)<br/>Pos N/E, Vel N/E, Attitude, Accel/Gyro Bias"]
        MAP_DB["GeoTIFF Tiled Map Database<br/>(Multi-scale descriptors & gradients)"]
        MAP_MATCH["Absolute Map Matcher<br/>(Orthorectify + ORB/AKAZE + RANSAC + Phase Corr)"]
        DELAY_FUSION["Delayed-State Fusion Engine<br/>(State history buffer + Innovation Chi-Square Gating)"]
    end

    subgraph Safety & Output Dispatcher [Core 3]
        HEALTH["System Health Gate<br/>(Thermal, Undervoltage, Latency, Confidence)"]
        DIST_OUT["MAVLink DISTANCE_SENSOR (>= 15 Hz)"]
        ODOM_OUT["MAVLink ODOMETRY / VISION / GPS_INPUT (20-30 Hz)"]
        LOG["Synchronized Telemetry & Video Logger (NVMe/SSD)"]
        ROS2["Optional Decoupled ROS 2 Wrapper Node"]
    end

    CAM --> CAP --> SYNC --> TRACK
    FC --> IMU_COMPASS --> TS_BUF
    TS_BUF --> SYNC
    TS_BUF --> DEROT
    TS_BUF --> ES_EKF

    TRACK --> KEY --> DEROT --> HEIGHT_KF
    TRACK --> SCALE_EXP --> HEIGHT_KF
    HOVER_HOLD <--> HEIGHT_KF

    HEIGHT_KF --> VIO_ENG
    DEROT --> VIO_ENG
    VIO_ENG --> ES_EKF

    ES_EKF -.->|Search Window Window & Prior| MAP_MATCH
    HEIGHT_KF -.->|Scale Factor| MAP_MATCH
    TS_BUF -.->|Heading & Tilt| MAP_MATCH
    MAP_DB --> MAP_MATCH
    MAP_MATCH --> DELAY_FUSION --> ES_EKF

    HEIGHT_KF --> DIST_OUT --> FC
    ES_EKF --> HEALTH --> ODOM_OUT --> FC
    HEIGHT_KF & ES_EKF --> LOG
    HEIGHT_KF & ES_EKF --> ROS2
```

---

### 2. Mathematical Formulations & Coordinate Frames

#### 2.1 Coordinate Conventions
- **World Navigation Frame ($W$)**: Local NED (North-East-Down), origin at mission start or EKF origin.
- **Body Frame ($B$)**: Drone structural frame (X forward, Y right, Z down).
- **Camera Frame ($C$)**: Optical frame with Z pointing forward along optical axis (downward relative to drone), X pointing right, Y pointing down.
- **Image Plane**: Pixel coordinates $(u, v)$ with origin at top-left.

#### 2.2 Pinhole Optics & Ground Sample Distance (GSD) across 10 m to 400 m
For focal length $f_{px} \approx 1207\text{ px}$ (full resolution) or $603.5\text{ px}$ (half resolution $728 \times 544$):
$$\text{GSD} = \frac{H}{f_{px}}$$

| Altitude AGL ($H$) | Full GSD ($1456\times 1088$) | Scaled GSD ($728\times 544$) | Flow at 10 m/s (1 frame @ 30 fps) | $10\text{ px}$ Baseline Ground Distance |
|:---|:---|:---|:---|:---|
| **$10\text{ m}$** | $0.83\text{ cm/px}$ | $1.66\text{ cm/px}$ | $40.2\text{ px}$ | $0.083\text{ m}$ ($8.3\text{ cm}$) |
| **$30\text{ m}$** | $2.48\text{ cm/px}$ | $4.96\text{ cm/px}$ | $13.4\text{ px}$ | $0.25\text{ m}$ |
| **$60\text{ m}$** | $4.97\text{ cm/px}$ | $9.94\text{ cm/px}$ | $6.7\text{ px}$ | $0.50\text{ m}$ |
| **$100\text{ m}$** | $8.28\text{ cm/px}$ | $16.56\text{ cm/px}$ | $4.0\text{ px}$ | $0.83\text{ m}$ |
| **$200\text{ m}$** | $16.57\text{ cm/px}$ | $33.14\text{ cm/px}$ | $2.0\text{ px}$ | $1.66\text{ m}$ |
| **$300\text{ m}$** | $24.85\text{ cm/px}$ | $49.70\text{ cm/px}$ | $1.3\text{ px}$ | $2.49\text{ m}$ |
| **$400\text{ m}$** | $33.14\text{ cm/px}$ | $66.28\text{ cm/px}$ | $1.0\text{ px}$ | $3.31\text{ m}$ |

#### 2.3 Gyro-Based Frame De-Rotation Homography
Between keyframe timestamp $t_k$ and current frame $t_c$, relative camera rotation $R_{k \to c} \in SO(3)$ is obtained by integrating high-rate gyro angular velocities:
$$R_{k \to c} = \exp\left( \int_{t_k}^{t_c} [\boldsymbol{\omega}_C(t)]_\times dt \right)$$
The infinite homography mapping pixels from frame $t_k$ to $t_c$ induced solely by rotation is:
$$H_{rot} = K R_{k \to c} K^{-1}$$
For feature point $\mathbf{p}_k$, its purely rotated projection in the current frame is $\tilde{\mathbf{p}}_{k,derot} = H_{rot} \tilde{\mathbf{p}}_k$.
The residual vector is purely translational ground motion:
$$\Delta \mathbf{p}_{trans} = \mathbf{p}_c - \tilde{\mathbf{p}}_{k,derot}$$

#### 2.4 Height from Keyframe De-Rotated Flow (10 m to 400 m)
With ground translation $\|\Delta \mathbf{d}_{ground}\| = \left\|\int_{t_k}^{t_c} \mathbf{v}_{ground}(t) dt\right\|$:
$$H = f_{px} \cdot \frac{\|\Delta \mathbf{d}_{ground}\|}{\|\Delta \mathbf{p}_{trans}\|}$$
The Adaptive Keyframe Manager enforces:
$$\|\Delta \mathbf{d}_{ground}\| \ge \frac{10 \cdot H_{prior}}{f_{px}}$$
- At $10\text{ m}$: baseline $= 8.3\text{ cm}$ ($\Delta t \approx 8\text{ ms}$ @ $10\text{ m/s}$ $\rightarrow$ triggers almost every frame).
- At $400\text{ m}$: baseline $= 3.31\text{ m}$ ($\Delta t \approx 330\text{ ms}$ @ $10\text{ m/s}$ $\rightarrow$ triggers every ~10 frames).

#### 2.5 2-State Height Kalman Filter (State: $[h_{AGL}, z_{terrain}]^T$)
State vector:
$$\mathbf{x} = \begin{bmatrix} h_{AGL} \\ z_{terrain} \end{bmatrix}, \quad a_{baro} = h_{AGL} + z_{terrain}$$
- **Kinematic Prediction**: Driven by vertical velocity $v_D$:
  $$\mathbf{x}_{k|k-1} = \mathbf{F} \mathbf{x}_{k-1} + \mathbf{B} v_D, \quad \mathbf{F} = \mathbf{I}_2, \quad \mathbf{B} = \begin{bmatrix} -\Delta t \\ 0 \end{bmatrix}$$
- **Barometer Update**:
  $$z_{baro} = \begin{bmatrix} 1 & 1 \end{bmatrix} \mathbf{x} + v_{baro}$$
- **Vision Height Update**:
  $$z_{vis} = \begin{bmatrix} 1 & 0 \end{bmatrix} \mathbf{x} + v_{vis}, \quad R_{vis} = \frac{R_{base}}{C^2}$$
- **Hover & Textureless Fallback**: When horizontal speed $< 0.5\text{ m/s}$ or image contrast is lost:
  $$\dot{z}_{terrain} = 0, \quad q_{terrain} \leftarrow 10 \cdot q_{terrain}, \quad h_{AGL} = a_{baro} - z_{terrain}$$

---

### 3. Phase 2: Horizontal Navigation & Absolute Map Matching

#### 3.1 Relative Visual-Inertial Odometry (VIO)
Given estimated height $h_{AGL}$, metric horizontal velocity in the camera frame is:
$$\mathbf{v}_{C,xy} = \frac{h_{AGL}}{f_{px} \cdot \Delta t} \Delta \mathbf{p}_{trans}$$
Transforming from camera optical frame to drone body frame ($B$):
$$\mathbf{v}_{B} = R_{C}^B \mathbf{v}_C$$
Combined with compass heading $\psi_{compass}$ and pitch/roll attitude, velocity in local NED frame is:
$$\mathbf{v}_{NED} = R_{B}^W(\phi, \theta, \psi) \mathbf{v}_B$$

#### 3.2 Error-State Extended Kalman Filter (ES-EKF)
State vector (15 states):
$$\delta \mathbf{x} = [\delta \mathbf{p}_{NED}, \delta \mathbf{v}_{NED}, \delta \boldsymbol{\theta}, \delta \mathbf{b}_a, \delta \mathbf{b}_g]^T$$
- **IMU Propagation (100 Hz)**: Integrates high-rate accelerometer and gyro measurements.
- **Compass Update**: Magnetometer yaw innovation constrains heading drift.
- **VIO Velocity Update (30 Hz)**: Fuses metric visual velocity.

#### 3.3 Absolute Map Matching via Georeferenced Orthophoto
1. **Offline Preparation**: Ingests high-resolution GeoTIFF orthomosaic of the flight operational area (Aurangabad Deccan region), builds multi-scale image pyramids, and pre-extracts ORB/AKAZE descriptors.
2. **Runtime Orthorectification**: Live camera frame is transformed into a nadir, North-aligned, map-scaled image:
   $$H_{rect} = S_{GSD} \cdot R_{yaw}(\psi) \cdot K_{map} \cdot R_{tilt}(\phi, \theta) \cdot K^{-1}$$
3. **Feature Matching**: Matches live orthorectified image against local candidate map tiles.
4. **Delayed-State EKF Update**:
   Because map matching requires $50-150\text{ ms}$:
   - The filter maintains a sliding state history buffer.
   - When a valid match is confirmed at timestamp $t_{match}$, the measurement update is applied to the historical state at $t_{match}$.
   - Subsequent IMU and VIO measurements are re-propagated forward to the current time $t_{now}$.
5. **Outlier Rejection**:
   Normalized innovation squared test ($\chi^2$ gating):
   $$d^2 = \boldsymbol{\nu}^T (\mathbf{H} \mathbf{P} \mathbf{H}^T + \mathbf{R})^{-1} \boldsymbol{\nu} \le \chi^2_{p=0.01}$$
   Outliers are rejected and logged; covariance is inflated gracefully if fixes are unavailable.

---

### 4. Raspberry Pi 5 Multi-Process Architecture & Resource Budget

```
 +---------------------------------------------------------------------------------------------------+
 |                                RASPBERRY PI 5 COMPANION COMPUTER                                  |
 |                                                                                                   |
 |   CORE 0: Ingestion & TimeSync         CORE 1: Vision Tracking & Height AGL                       |
 |   - libcamera zero-copy capture        - Shi-Tomasi feature tracking (NEON)                       |
 |   - CLOCK_MONOTONIC timestamping       - Gyro de-rotation homography                              |
 |   - MAVLink TIMESYNC clock sync        - 2-State Kalman Filter ([h_agl, z_terrain]^T)             |
 |   - CPU load: ~10-15%                  - CPU load: ~25-30%                                        |
 |                          \                  /                                                     |
 |                           Shared Memory IPC                                                       |
 |                          /                  \                                                     |
 |   CORE 2: Absolute Map Matching        CORE 3: Sensor Fusion & MAVLink Dispatch                   |
 |   - GeoTIFF tile retrieval             - 15-State ES-EKF (VIO + IMU + Compass)                    |
 |   - Orthorectification & matching      - MAVLink ODOMETRY / DISTANCE_SENSOR dispatch              |
 |   - Delayed-state correction           - Thermal / undervoltage health monitor                    |
 |   - CPU load: ~15-20% (1-2 Hz)         - CPU load: ~10-15%                                        |
 +---------------------------------------------------------------------------------------------------+
```

| Pipeline Stage | Frequency | Target Latency | CPU Budget (Pi 5) | Optimization Strategy |
|:---|:---|:---|:---|:---|
| **Camera Capture** | 30 Hz | $< 5\text{ ms}$ | $< 10\%$ Core 0 | Zero-copy V4L2/libcamera ring buffer |
| **Optical Flow & De-Rotation** | 30 Hz | $< 25\text{ ms}$ | $< 30\%$ Core 1 | Capped features (300-500), $728\times 544$ resolution, ARM NEON |
| **Height Kalman Fusion** | 30 Hz | $< 2\text{ ms}$ | $< 5\%$ Core 1 | Analytical Joseph-form covariance update |
| **VIO Velocity Recovery** | 30 Hz | $< 3\text{ ms}$ | $< 5\%$ Core 1 | Direct projection from height & de-rotated flow |
| **Map Matching** | 1–2 Hz | $< 150\text{ ms}$ | $< 20\%$ Core 2 | Asynchronous worker process, tiled spatial indexing |
| **MAVLink I/O & Health** | 100 Hz | $< 2\text{ ms}$ | $< 10\%$ Core 3 | Thread-safe serial socket @ 921,600 baud |
| **Total System Budget** | — | **$< 100\text{ ms}$** | **$< 80\%$** across 4 cores | Multi-process core pinning (`os.sched_setaffinity`) |
