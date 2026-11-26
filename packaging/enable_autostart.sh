#!/usr/bin/env bash
# Enables automatic VPS + VTX OSD startup on Raspberry Pi boot via systemd.
set -euo pipefail

if [ "$EUID" -ne 0 ]; then
    echo "Please run as root: sudo ./packaging/enable_autostart.sh"
    exit 1
fi

TARGET_USER="${SUDO_USER:-admin}"
VPS_DIR="/home/${TARGET_USER}/VPS"

usermod -aG dialout,video,render "${TARGET_USER}" || true
cp "${VPS_DIR}/packaging/vps_pi4.service" /etc/systemd/system/vps.service
sed -i "s|/home/admin/VPS|${VPS_DIR}|g; s|User=admin|User=${TARGET_USER}|g" /etc/systemd/system/vps.service

systemctl daemon-reload
systemctl enable vps.service
systemctl restart vps.service
systemctl status vps.service --no-pager

