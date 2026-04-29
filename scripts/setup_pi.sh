#!/usr/bin/env bash
# One-time setup script for Raspberry Pi 5.
# Run once after cloning the repo:  bash scripts/setup_pi.sh
set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
SERVICE_NAME="horizon-hud"
CURRENT_USER="$(whoami)"

echo "==> Repo: $REPO_DIR"
echo "==> User: $CURRENT_USER"

# ── 1. System packages ──────────────────────────────────────────────────────
echo "==> Installing system packages..."
sudo apt-get update -qq
sudo apt-get install -y python3-venv python3-pip libcamera-dev v4l-utils

# ── 2. Python virtual environment ──────────────────────────────────────────
echo "==> Setting up Python venv..."
python3 -m venv "$REPO_DIR/.venv"
"$REPO_DIR/.venv/bin/pip" install --upgrade pip --quiet
"$REPO_DIR/.venv/bin/pip" install -r "$REPO_DIR/requirements.txt" --quiet
echo "    venv ready."

# ── 3. Log directory ───────────────────────────────────────────────────────
mkdir -p "$HOME/logs"
echo "==> Log directory: $HOME/logs"

# ── 4. Systemd service ─────────────────────────────────────────────────────
echo "==> Installing systemd service..."
# Patch the service file with the actual user and repo path
sed \
  -e "s|User=pi|User=$CURRENT_USER|g" \
  -e "s|/home/pi/Horizon-HUD|$REPO_DIR|g" \
  -e "s|/home/pi/logs|$HOME/logs|g" \
  "$REPO_DIR/scripts/horizon-hud.service" \
  | sudo tee "/etc/systemd/system/$SERVICE_NAME.service" > /dev/null

sudo systemctl daemon-reload
sudo systemctl enable "$SERVICE_NAME"
echo "    Service installed and enabled."

# ── 5. Sudoers rule (lets Actions runner restart the service w/o password) ─
SUDOERS_LINE="$CURRENT_USER ALL=(ALL) NOPASSWD: /usr/bin/systemctl restart $SERVICE_NAME, /usr/bin/systemctl start $SERVICE_NAME, /usr/bin/systemctl stop $SERVICE_NAME"
SUDOERS_FILE="/etc/sudoers.d/$SERVICE_NAME"

if sudo grep -qF "systemctl restart $SERVICE_NAME" "$SUDOERS_FILE" 2>/dev/null; then
  echo "==> Sudoers rule already present, skipping."
else
  echo "$SUDOERS_LINE" | sudo tee "$SUDOERS_FILE" > /dev/null
  sudo chmod 440 "$SUDOERS_FILE"
  echo "==> Sudoers rule added: $SUDOERS_FILE"
fi

echo ""
echo "==> Done. Next steps:"
echo "    1. Register a GitHub Actions self-hosted runner:"
echo "       https://github.com/Ioan-Vlad-Pantiru/Horizon-HUD/settings/actions/runners/new"
echo "    2. Start the service manually to verify:"
echo "       sudo systemctl start $SERVICE_NAME"
echo "       journalctl -u $SERVICE_NAME -f"
