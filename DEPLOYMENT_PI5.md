# Raspberry Pi 5 Target Platform Deployment Guide

This guide provides end-to-end instructions for deploying the **Visual Positioning System (VPS) Phase 1 Height Estimator** on a **Raspberry Pi 5** companion computer connected to an ArduPilot flight controller.

---

## 1. System Architecture & Hardware Connections

```
 +-------------------------------------------------------------------------------+
 |                               DRONE PLATFORM                                  |
 |                                                                               |
 |   +-----------------------+              +--------------------------------+   |
 |   |  Sony IMX296 Camera   |              |     ArduPilot Flight           |   |
 |   |  (Global Shutter)     |              |     Controller (Cube/Pixhawk)  |   |
 |   |  1456x1088 @ 30 FPS   |              +--------------------------------+   |
 |   +-----------+-----------+                              ^                    |
 |               | MIPI-CSI (22-pin)                        | TELEM2 UART        |
 |               v                                          v (921,600 baud)     |
 |   +-----------+------------------------------------------+----------------+   |
 |   |                     Raspberry Pi 5 Companion Computer                 |   |
 |   |                                                                       |   |
 |   |   1. V4L2 Camera Capture (Zero-copy ring buffer)                      |   |
 |   |   2. Shi-Tomasi Pyramidal LK Optical Flow                             |   |
 |   |   3. High-Rate Gyro De-Rotation Homography (H = K * R * K^-1)         |   |
 |   |   4. 2-State Kalman Filter ([h_agl, z_terrain]^T)                     |   |
 |   |   5. MAVLink DISTANCE_SENSOR Dispatcher (>= 15 Hz to ArduPilot EKF)   |   |
 |   +-----------------------------------------------------------------------+   |
 +-------------------------------------------------------------------------------+
```

### Physical Wiring (Raspberry Pi 5 to Flight Controller TELEM2)

| Raspberry Pi 5 Pin | Function | ArduPilot TELEM2 Pin | Notes |
|:---|:---|:---|:---|
| **Pin 8 (GPIO 14)** | UART0 TXD | **TELEM2 RX** | Companion TX to FC RX |
| **Pin 10 (GPIO 15)**| UART0 RXD | **TELEM2 TX** | Companion RX to FC TX |
| **Pin 6 or 9** | Ground (GND) | **TELEM2 GND** | Common reference ground |

> [!CAUTION]
> **Voltage Warning**: Raspberry Pi 5 GPIO pins operate at **3.3V logic**. Ensure your flight controller TELEM port uses 3.3V UART signals (standard on Cube Orange, Pixhawk 4/6, Holybro). **Never** connect 5V directly to Raspberry Pi GPIO pins.

---

## 2. Operating System & Kernel Configuration

1. Install **Raspberry Pi OS 64-bit (Debian 12 Bookworm)** on a high-end microSD or NVMe base.
2. Edit `/boot/firmware/config.txt`:
   ```bash
   sudo nano /boot/firmware/config.txt
   ```
   Add the following overlays at the end of the file:
   ```ini
   # Enable Sony IMX296 Global Shutter Camera on CAM 0
   dtoverlay=imx296,cam0

   # Enable Primary Hardware UART for MAVLink
   enable_uart=1
   dtoverlay=uart0
   ```
3. **Disable Serial Console**:
   The Linux serial console must be detached from the hardware UART to avoid garbage characters corrupting MAVLink packets:
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
   You should see `imx296 [1456x1088]` listed under `/dev/video0`.

---

## 3. ArduPilot Flight Controller Parameters

Connect to the flight controller using **Mission Planner** or **QGroundControl** and configure the following parameters:

| Parameter | Recommended Value | Description |
|:---|:---|:---|
| `SERIAL2_PROTOCOL` | `2` | MAVLink 2 protocol on TELEM2 |
| `SERIAL2_BAUD` | `921` | 921,600 baud serial rate |
| `RNGFND1_TYPE` | `10` | MAVLink Rangefinder backend |
| `RNGFND1_ORIENT` | `25` | `PITCH_270` (Downward-facing sensor) |
| `RNGFND1_MIN_CM` | `10` | 10 cm minimum detection threshold |
| `RNGFND1_MAX_CM` | `30000` | 30,000 cm (300 m) survey maximum |
| `RNGFND1_GNDCLEAR` | `15` | 15 cm landing-gear ground clearance |
| `EK3_SRC1_POSZ` | `2` or `1` | `2` = Rangefinder for low-altitude terrain following; `1` = Baro with Rangefinder fusion |

Save parameters to flash and reboot the flight controller.

---

## 4. Software Installation & Setup

1. Clone or copy the VPS repository to the Raspberry Pi 5:
   ```bash
   cd ~
   git clone https://github.com/your-repo/VPS.git
   cd VPS
   ```
2. Create and activate a Python virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```
3. Install required packages:
   ```bash
   pip install --upgrade pip
   pip install numpy opencv-python pymavlink pyyaml scipy matplotlib pytest
   ```
4. Verify the test suite passes locally on the Pi 5:
   ```bash
   pytest tests/ -v
   ```

---

## 5. Hand-Held Pre-Flight Field Verification

Before taking off, perform the **Hand-Held Ground Baseline Test**:

```bash
# Run with interactive HUD on connected monitor or VNC
python vps/main.py --connection /dev/ttyAMA0:921600 --camera /dev/video0 --mode handheld
```

1. **Resting on Ground**:
   - Place the drone on its landing gear over flat ground or an ArUco calibration target.
   - Verify reported **Net AGL** reads `0.00 m` (the $15\text{ cm}$ physical offset is automatically subtracted).
2. **Hand-Held Lift Test**:
   - Lift the drone by hand to waist height ($1.00\text{ m}$).
   - Verify Mission Planner / QGroundControl `sonarrange` displays `1.00 m ± 0.02 m`.
   - Tilt the drone $\pm 10^\circ$ pitch and roll; verify height remains steady due to gyro de-rotation.
3. **Mission Planner HUD**:
   - Confirm `sonarrange` increases smoothly as you raise the drone.
   - Confirm `signal_quality` displays `80% - 99%`.

---

## 6. Autostart Systemd Service (`vps.service`)

Configure VPS to start automatically upon drone battery connection:

1. Create systemd unit file:
   ```bash
   sudo nano /etc/systemd/system/vps.service
   ```
2. Paste the following configuration:
   ```ini
   [Unit]
   Description=Visual Positioning System (VPS) AGL Height Estimator
   After=network.target local-fs.target

   [Service]
   Type=simple
   User=pi
   WorkingDirectory=/home/pi/VPS
   ExecStart=/home/pi/VPS/venv/bin/python /home/pi/VPS/vps/main.py --config config/default_config.yaml --connection /dev/ttyAMA0:921600 --camera /dev/video0 --mode flight
   Restart=always
   RestartSec=3
   StandardOutput=journal
   StandardError=journal

   [Install]
   WantedBy=multi-user.target
   ```
3. Enable and start the service:
   ```bash
   sudo systemctl daemon-reload
   sudo systemctl enable vps.service
   sudo systemctl start vps.service
   ```
4. Check live service status and streaming logs:
   ```bash
   sudo systemctl status vps.service
   journalctl -u vps.service -f
   ```

---

## 7. Troubleshooting & Diagnostics

- **No MAVLink heartbeat from FC**:
  Verify `/dev/ttyAMA0` permissions: `sudo usermod -aG dialout pi`. Verify TX/RX wires are crossed (Pi TX to FC RX; Pi RX to FC TX).
- **Camera frame rate drops**:
  Check CPU temperature: `vcgencmd measure_temp`. On Pi 5 with active cooler, CPU stays $< 55^\circ\text{C}$ with full pipeline running at 30 FPS.
- **ArduPilot reports "Bad LiDAR Health"**:
  Ensure VPS is running and streaming `DISTANCE_SENSOR` packets at $\ge 10\text{ Hz}$. Check `journalctl -u vps.service -f` for active dispatch logs.
