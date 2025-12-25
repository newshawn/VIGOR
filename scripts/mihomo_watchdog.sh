#!/usr/bin/env bash
# Lightweight watchdog for mihomo when systemd/cron are unavailable.
# Starts mihomo if not running; loop until this process exits.

set -euo pipefail

BIN_PATH="${HOME}/tools/mihomo/bin/mihomo"
WORK_DIR="${HOME}/tools/mihomo"
LOG_FILE="${WORK_DIR}/logs/watchdog.log"
INTERVAL=60

if [[ ! -x "${BIN_PATH}" ]]; then
  echo "Error: ${BIN_PATH} not found or not executable." >&2
  exit 1
fi

mkdir -p "$(dirname "${LOG_FILE}")"

echo "Starting mihomo watchdog. Interval=${INTERVAL}s, log=${LOG_FILE}"
while true; do
  if ! pgrep -x mihomo >/dev/null 2>&1; then
    echo "$(date '+%F %T') mihomo not running; starting..." | tee -a "${LOG_FILE}"
    "${BIN_PATH}" -d "${WORK_DIR}" >>"${LOG_FILE}" 2>&1 &
  fi
  sleep "${INTERVAL}"
done
