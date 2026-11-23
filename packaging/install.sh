#!/usr/bin/env bash
# ==============================================================================
# Visual Positioning System (VPS) - Production Installer for Raspberry Pi 5
# Operating System: Raspberry Pi OS 64-bit (Debian 12 Bookworm)
# Hardware: Raspberry Pi 5 + Sony IMX296 Camera + MicroAir 405v2 (ArduPilot 4.5.7)
# ==============================================================================

set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}==================================================================${NC}"
echo -e "${BLUE}   VISUAL POSITIONING SYSTEM (VPS) - RASPBERRY PI 5 INSTALLER     ${NC}"
echo -e "${BLUE}==================================================================${NC}"

# Check for root privilege
if [ "$EUID" -ne 0 ]; then
    echo -e "${RED}[ERROR] This installer must be run as root (sudo ./install.sh)${NC}"
    exit 1
fi

TARGET_USER="${SUDO_USER:-pi}"
VPS_DIR="/home/${TARGET_USER}/VPS"
BOOT_CONFIG="/boot/firmware/config.txt"

# If Bookworm fallback path
if [ ! -f "$BOOT_CONFIG" ]; then
    BOOT_CONFIG="/boot/config.txt"
fi

echo -e "${GREEN}[1/6] Installing Debian system dependencies...${NC}"
apt-get update
apt-get install -y \
    python3 \
    python3-venv \
    python3-pip \
    python3-dev \
    v4l-utils \
    libcamera-dev \
    libcap-dev \
    i2c-tools \
    git

echo -e "${GREEN}[2/6] Configuring Raspberry Pi 5 Hardware Overlays in ${BOOT_CONFIG}...${NC}"

# Configure Sony IMX296 global shutter camera overlay
if ! grep -q "dtoverlay=imx296" "$BOOT_CONFIG"; then
    echo "" >> "$BOOT_CONFIG"
    echo "# VPS: Sony IMX296 Global Shutter Camera" >> "$BOOT_CONFIG"
    echo "dtoverlay=imx296,cam0" >> "$BOOT_CONFIG"
    echo -e "  -> Added IMX296 camera overlay"
else
    echo -e "  -> IMX296 overlay already configured"
fi

# Configure primary hardware UART0 for ArduPilot MAVLink
if ! grep -q "enable_uart=1" "$BOOT_CONFIG"; then
    echo "# VPS: Primary UART0 for ArduPilot MAVLink 2 (921600 baud)" >> "$BOOT_CONFIG"
    echo "enable_uart=1" >> "$BOOT_CONFIG"
    echo "dtoverlay=uart0" >> "$BOOT_CONFIG"
    echo -e "  -> Added UART0 overlay"
else
    echo -e "  -> UART0 overlay already configured"
fi

# Enable hardware watchdog
if ! grep -q "dtparam=watchdog=on" "$BOOT_CONFIG"; then
    echo "dtparam=watchdog=on" >> "$BOOT_CONFIG"
    echo -e "  -> Enabled hardware watchdog"
fi

echo -e "${GREEN}[3/6] Setting user group permissions for ${TARGET_USER}...${NC}"
usermod -aG dialout,video,gpio,i2c "${TARGET_USER}"

echo -e "${GREEN}[4/6] Setting up Python virtual environment at ${VPS_DIR}/venv...${NC}"
if [ ! -d "${VPS_DIR}/venv" ]; then
    sudo -u "${TARGET_USER}" python3 -m venv "${VPS_DIR}/venv"
fi

sudo -u "${TARGET_USER}" "${VPS_DIR}/venv/bin/pip" install --upgrade pip
sudo -u "${TARGET_USER}" "${VPS_DIR}/venv/bin/pip" install \
    numpy \
    opencv-python-headless \
    pymavlink \
    pyyaml \
    scipy \
    psutil \
    pytest \
    tifffile

echo -e "${GREEN}[5/6] Deploying systemd production service...${NC}"
cp "${VPS_DIR}/packaging/vps.service" /etc/systemd/system/vps.service
sed -i "s|/home/pi/VPS|${VPS_DIR}|g" /etc/systemd/system/vps.service
sed -i "s|User=pi|User=${TARGET_USER}|g" /etc/systemd/system/vps.service

systemctl daemon-reload
systemctl enable vps.service

echo -e "${GREEN}[6/6] Creating logging and map database directories...${NC}"
mkdir -p "${VPS_DIR}/data/logs" "${VPS_DIR}/data/map_db" "${VPS_DIR}/reports"
chown -R "${TARGET_USER}:${TARGET_USER}" "${VPS_DIR}/data" "${VPS_DIR}/reports"

echo -e "${BLUE}==================================================================${NC}"
echo -e "${GREEN}[SUCCESS] Installation Complete!${NC}"
echo -e "1. Please ${YELLOW}reboot your Raspberry Pi 5${NC} to apply hardware overlays:"
echo -e "   sudo reboot"
echo -e "2. After reboot, check camera detection:"
echo -e "   v4l2-ctl --list-devices"
echo -e "3. Verify live VPS status:"
echo -e "   sudo systemctl status vps.service"
echo -e "   journalctl -u vps.service -f"
echo -e "${BLUE}==================================================================${NC}"

