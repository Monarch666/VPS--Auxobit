# Raspberry Pi 5 Target Platform Deployment & Operating Manual

This document provides the definitive field deployment and operation manual for the **Visual Positioning System (VPS)** on a **Raspberry Pi 5** companion computer linked to an **ArduPilot 4.5.7** flight controller on a **MicroAir 405v2** board.

---

## 1. System Architecture & Physical Connections

```
 +-------------------------------------------------------------------------------+
 |                               DRONE PLATFORM                                  |
 |                                                                               |
 |   +-----------------------+              +--------------------------------+   |
 |   |  Sony IMX296 Camera   |              |     MicroAir 405v2 FC          |   |
 |   |  (Global Shutter)     |              |     (ArduPilot 4.5.7)          |   |
 |   |  1456x1088 @ 30 FPS   |              +--------------------------------+   |
 |   +-----------+-----------+                              ^                    |
 |               | MIPI-CSI (22-pin)                        | TELEM2 UART        |
 |               v                                          v (921,600 baud)     |
 |   +-----------+------------------------------------------+----------------+   |
 |   |                     Raspberry Pi 5 Companion Computer                 |   |
 |   |                                                                       |   |
 |   |   1. V4L2 Zero-Copy Ingest (728x544 tracking resolution)              |   |
 |   |   2. Pyramidal LK Optical Flow (Shi-Tomasi, capped 350 pts)           |   |
 |   |   3. Gyro De-Rotation Homography (H = K * R * K^-1)                   |   |
 |   |   4. 15-State Error-State EKF (Position, Velocity, Attitude, Biases)  |   |
 |   |   5. Offline Multi-Scale GeoTIFF Map Database (Sentinel-2 Aurangabad) |   |
 |   |   6. Frame Orthorectification & Multi-Scale Map Matcher (1-2 Hz)      |   |
 |   |   7. Delayed-State EKF Latency Compensation & Outlier Gating          |   |
 |   |   8. Hardware Watchdog & Systemd Process Supervision                  |   |
 |   |   9. MAVLink ODOMETRY / DISTANCE_SENSOR Dispatcher (25 Hz)            |   |
 |   +-----------------------------------------------------------------------+   |
 +-------------------------------------------------------------------------------+
```

### Physical Wiring (Raspberry Pi 5 to MicroAir 405v2 TELEM2)

| Raspberry Pi 5 Pin | Function | MicroAir 405v2 TELEM2 Pin | Notes |
|:---|:---|:---|:---|
| **Pin 8 (GPIO 14)** | UART0 TXD | **TELEM2 RX** | Companion TX $\rightarrow$ FC RX |
| **Pin 10 (GPIO 15)**| UART0 RXD | **TELEM2 TX** | Companion RX $\leftarrow$ FC TX |
| **Pin 6 or 9** | Ground (GND) | **TELEM2 GND** | Common reference ground |

> [!CAUTION]
> **Voltage Warning**: Raspberry Pi 5 GPIO pins operate strictly at **3.3V logic**. The MicroAir 405v2 TELEM2 port operates at 3.3V UART levels. **Never** connect 5V signals directly to Raspberry Pi GPIO pins.

---

## 2. Operating System & Hardware Overlay Configuration

1. Install **Raspberry Pi OS 64-bit (Debian 12 Bookworm)** on a high-speed NVMe base or Class 10 U3 microSD card.
2. Edit `/boot/firmware/config.txt`:
   ```bash
   sudo nano /boot/firmware/config.txt
   ```
   Add the following overlays at the end of the file:
   ```ini
   # Enable Sony IMX296 Global Shutter Camera on CAM 0
   dtoverlay=imx296,cam0

   # Enable Primary Hardware UART0 for MAVLink 2
   enable_uart=1
   dtoverlay=uart0

   # Enable hardware watchdog
   dtparam=watchdog=on
   ```
3. **Disable Linux Serial Console**:
   The Linux serial console must be detached from the UART to prevent login prompt characters from corrupting MAVLink packets:
   ```bash
   sudo raspi-config
   ```
   Navigate to: `Interface Options` $\rightarrow$ `Serial Port`:
   - *Would you like a login shell to be accessible over serial?* $\rightarrow$ **NO**
   - *Would you like the serial port hardware to be enabled?* $\rightarrow$ **YES**
4. Reboot the Raspberry Pi 5:
   ```bash
   sudo reboot
   ```
5. Verify camera recognition:
   ```bash
   v4l2-ctl --list-devices
   libcamera-hello --list-cameras
   ```
   Confirm `imx296 [1456x1088]` is listed under `/dev/video0`.

---

## 3. ArduPilot 4.5.7 Flight Controller Parameters

Load the pre-configured parameter file [config/ardupilot_vps_params.param](file:///E:/VPS/config/ardupilot_vps_params.param) into your MicroAir 405v2 using **Mission Planner** or **QGroundControl**:

| Parameter | Value | Description |
|:---|:---|:---|
| `SERIAL2_PROTOCOL` | `2` | MAVLink 2 protocol on TELEM2 |
| `SERIAL2_BAUD` | `921` | 921,600 baud serial rate |
| `EK3_SRC1_POSXY` | `6` | External Navigation (Visual Odometry / Map Matching) |
| `EK3_SRC1_VELXY` | `6` | External Navigation Linear Velocity |
| `EK3_SRC1_POSZ` | `1` | Barometer with Rangefinder / Optical Flow fusion |
| `EK3_SRC1_YAW` | `1` | Compass (Magnetometer) heading |
| `VISO_TYPE` | `1` | MAVLink External Navigation backend |
| `VISO_POS_M_NSE` | `0.20` | Visual position measurement noise (meters) |
| `VISO_VEL_M_NSE` | `0.15` | Visual velocity measurement noise (m/s) |
| `VISO_DELAY_MS` | `80` | Matching & transmission latency compensation |
| `RNGFND1_TYPE` | `10` | MAVLink Rangefinder backend |
| `RNGFND1_ORIENT` | `25` | `PITCH_270` (Downward-facing sensor) |
| `RNGFND1_MIN_CM` | `10` | 10 cm minimum detection threshold |
| `RNGFND1_MAX_CM` | `40000` | 40,000 cm (400 m) operational maximum |
| `RNGFND1_GNDCLEAR` | `15` | 15 cm landing-gear ground clearance |

---

## 4. Automated Installation & Setup

1. Run the one-step installer script:
   ```bash
   cd /home/pi/VPS
   sudo ./packaging/install.sh
   ```
2. Build the runtime spatial map database from the user's Sentinel-2 GeoTIFF:
   ```bash
   python -m vps.map_db.tiler \
       --input 2026-11-14-00_00_2026-11-14-23_59_Sentinel-2_L2A_True_color.tiff \
       --output data/map_db/aurangabad_s2 \
       --name aurangabad_sentinel2 \
       --tile-size 512
   ```

---

## 5. Hand-Held Pre-Flight Field Verification

Before arming, perform the **Hand-Held Ground Baseline Test**:

```bash
python -m vps.main --connection /dev/ttyAMA0:921600 --camera /dev/video0 --mode handheld
```

1. **Resting on Ground**:
   - Place drone on landing gear over flat ground or ArUco target.
   - Verify reported **Net AGL** reads `0.00 m` (15 cm physical offset subtracted).
2. **Waist Height Lift Test**:
   - Raise drone to waist level ($1.00\text{ m}$).
   - Verify Mission Planner / QGroundControl `sonarrange` displays `1.00 m ± 0.02 m`.
   - Tilt drone $\pm 10^\circ$ pitch and roll; verify altitude remains steady due to gyro de-rotation.
3. **Mission Planner EKF Status**:
   - Confirm EKF3 lane switches to external navigation without errors.
   - Confirm `vis_pos` and `vis_vel` indicators in Mission Planner show green.

---

## 6. Autostart Systemd Service (`vps.service`)

The systemd service autostarts on boot with hardware watchdog monitoring:

```bash
# Start service
sudo systemctl start vps.service

# Check service status and health watchdog
sudo systemctl status vps.service

# Follow real-time streaming logs
journalctl -u vps.service -f
```

---

## 7. Diagnostics & Hardware Health Guard

The background supervisor continuously inspects:
- **Frame Rate**: Asserts $\ge 15\text{ FPS}$ (nominal $30\text{ FPS}$).
- **Telemetry Freshness**: Asserts MAVLink heartbeats arrive within $1.0\text{ s}$.
- **CPU Thermal Headroom**: Monitored via `/sys/class/thermal/thermal_zone0/temp`. If core temperature exceeds $80^\circ\text{C}$, the safety interlock inflates filter covariance. Under active cooling, operating temperature stays at $48^\circ\text{C} - 55^\circ\text{C}$.
- **Undervoltage Protection**: Monitored via `vcgencmd get_throttled`. Any voltage drop below $4.8\text{V}$ is immediately logged and alerts the operator.
