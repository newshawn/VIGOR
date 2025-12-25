#!/usr/bin/env bash
set -euo pipefail

SERVICE_NAME="mihomo"
# The mihomo binary is located under bin/ in your setup.
BIN_PATH="${HOME}/tools/mihomo/bin/mihomo"
WORK_DIR="${HOME}/tools/mihomo"
SERVICE_FILE="${HOME}/.config/systemd/user/${SERVICE_NAME}.service"
CRON_TAG="# mihomo-autostart"

usage() {
  echo "Usage: $0 {install|uninstall|status|install-cron|uninstall-cron|status-cron}"
  exit 1
}

ensure_bin() {
  if [[ ! -x "${BIN_PATH}" ]]; then
    echo "Error: ${BIN_PATH} not found or not executable." >&2
    exit 1
  fi
}

ensure_crontab() {
  if ! command -v crontab >/dev/null 2>&1; then
    echo "Error: crontab command not found. Install cron (e.g., sudo apt-get install -y cron) and retry." >&2
    exit 1
  fi
}

install_service() {
  ensure_bin
  mkdir -p "$(dirname "${SERVICE_FILE}")"
  cat > "${SERVICE_FILE}" <<EOF
[Unit]
Description=Mihomo
After=network-online.target

[Service]
ExecStart=${BIN_PATH} -d ${WORK_DIR}
Restart=always
RestartSec=3
LimitNOFILE=1048576

[Install]
WantedBy=default.target
EOF
  systemctl --user daemon-reload
  systemctl --user enable --now "${SERVICE_NAME}.service"
  echo "Installed and started ${SERVICE_NAME}.service"
}

uninstall_service() {
  systemctl --user disable --now "${SERVICE_NAME}.service" || true
  rm -f "${SERVICE_FILE}"
  systemctl --user daemon-reload
  echo "Removed ${SERVICE_NAME}.service"
}

status_service() {
  systemctl --user status "${SERVICE_NAME}.service" --no-pager
}

install_cron() {
  ensure_crontab
  ensure_bin
  local tmp
  tmp="$(mktemp)"
  # Drop previous mihomo entries with our tag, then append fresh ones.
  crontab -l 2>/dev/null | grep -v "${CRON_TAG}" > "${tmp}" || true
  {
    cat "${tmp}"
    echo "@reboot ${BIN_PATH} -d ${WORK_DIR} >${WORK_DIR}/logs/auto.log 2>&1 ${CRON_TAG}"
    echo "* * * * * pgrep -x ${SERVICE_NAME} >/dev/null 2>&1 || ${BIN_PATH} -d ${WORK_DIR} >>${WORK_DIR}/logs/auto.log 2>&1 ${CRON_TAG}"
  } | crontab -
  rm -f "${tmp}"
  echo "Installed cron autostart/watch entries (${CRON_TAG})."
}

uninstall_cron() {
  ensure_crontab
  local tmp
  tmp="$(mktemp)"
  crontab -l 2>/dev/null | grep -v "${CRON_TAG}" > "${tmp}" || true
  crontab "${tmp}"
  rm -f "${tmp}"
  echo "Removed cron entries tagged ${CRON_TAG}."
}

status_cron() {
  ensure_crontab
  crontab -l 2>/dev/null | grep "${CRON_TAG}" || { echo "No mihomo cron entries found."; return 1; }
}

if [[ $# -ne 1 ]]; then
  usage
fi

case "$1" in
  install) install_service ;;
  uninstall) uninstall_service ;;
  status) status_service ;;
  install-cron) install_cron ;;
  uninstall-cron) uninstall_cron ;;
  status-cron) status_cron ;;
  *) usage ;;
esac
