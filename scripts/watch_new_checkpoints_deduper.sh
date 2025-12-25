#!/usr/bin/env bash
set -euo pipefail

INTERVAL_SECONDS=1200
DRY_RUN=0
RUN_ONCE=0
LOG_FILE=""
MATCH_RE='watch_new_checkpoints\.sh'

usage() {
  cat <<'EOF'
Usage: scripts/watch_new_checkpoints_deduper.sh [--interval SECONDS] [--log FILE] [--dry-run] [--once]

Detects duplicate running scripts/watch_new_checkpoints.sh processes that share the same --watch-dir,
keeps the oldest (largest elapsed time), and kill -9 the rest.

Examples:
  # Run once (dry-run)
  scripts/watch_new_checkpoints_deduper.sh --once --dry-run

  # Run in background, check every 20 minutes, log to a file
  nohup scripts/watch_new_checkpoints_deduper.sh --interval 1200 --log watch_deduper.log >/dev/null 2>&1 &
EOF
}

log() {
  local msg="$1"
  local line
  line="[$(date '+%F %T')] ${msg}"
  if [[ -n "${LOG_FILE}" ]]; then
    printf '%s\n' "${line}" >>"${LOG_FILE}"
  else
    printf '%s\n' "${line}"
  fi
}

extract_watch_dir() {
  local cmd="$1"
  # Expect: ... --watch-dir /path ... (no spaces inside watch-dir)
  sed -n 's/.*--watch-dir[[:space:]]\([^[:space:]]\+\).*/\1/p' <<<"${cmd}" | head -n1
}

elapsed_seconds() {
  local pid="$1"
  local etimes
  etimes="$(ps -o etimes= -p "${pid}" 2>/dev/null | awk '{print $1}' || true)"
  if [[ -z "${etimes}" ]]; then
    printf '0\n'
  else
    printf '%s\n' "${etimes}"
  fi
}

choose_keep_pid() {
  local keep_pid=""
  local keep_age=-1
  local pid age
  for pid in "$@"; do
    age="$(elapsed_seconds "${pid}")"
    if (( age > keep_age )); then
      keep_age="${age}"
      keep_pid="${pid}"
    fi
  done

  if [[ -n "${keep_pid}" ]]; then
    printf '%s\n' "${keep_pid}"
    return 0
  fi

  # Fallback: smallest PID
  printf '%s\n' "$@" | sort -n | head -n1
}

dedupe_once() {
  local lines
  lines="$(pgrep -af "${MATCH_RE}" || true)"
  if [[ -z "${lines}" ]]; then
    log "No matching processes found."
    return 0
  fi

  declare -A pids_by_dir=()

  while IFS= read -r line; do
    [[ -z "${line}" ]] && continue
    local pid cmd watch_dir
    pid="${line%% *}"
    cmd="${line#* }"
    watch_dir="$(extract_watch_dir "${cmd}")"
    [[ -z "${watch_dir}" ]] && continue
    pids_by_dir["${watch_dir}"]="${pids_by_dir["${watch_dir}"]-} ${pid}"
  done <<<"${lines}"

  local dir pid_list keep_pid
  local -a pids
  for dir in "${!pids_by_dir[@]}"; do
    # shellcheck disable=SC2206
    pids=(${pids_by_dir["${dir}"]})
    if (( ${#pids[@]} <= 1 )); then
      continue
    fi

    keep_pid="$(choose_keep_pid "${pids[@]}")"
    for pid in "${pids[@]}"; do
      [[ "${pid}" == "${keep_pid}" ]] && continue
      log "Duplicate --watch-dir=${dir} keep=${keep_pid} kill=${pid}"
      if (( DRY_RUN )); then
        continue
      fi
      kill -9 "${pid}" 2>/dev/null || true
    done
  done
}

while (( $# > 0 )); do
  case "$1" in
    --interval)
      INTERVAL_SECONDS="${2:-}"
      shift 2
      ;;
    --log)
      LOG_FILE="${2:-}"
      shift 2
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    --once)
      RUN_ONCE=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if ! [[ "${INTERVAL_SECONDS}" =~ ^[0-9]+$ ]] || (( INTERVAL_SECONDS <= 0 )); then
  echo "--interval must be a positive integer (seconds), got: ${INTERVAL_SECONDS}" >&2
  exit 2
fi

if [[ -n "${LOG_FILE}" ]]; then
  mkdir -p "$(dirname "${LOG_FILE}")" 2>/dev/null || true
fi

trap 'log "Exiting."; exit 0' INT TERM

if (( RUN_ONCE )); then
  dedupe_once
  exit 0
fi

while true; do
  dedupe_once
  sleep "${INTERVAL_SECONDS}"
done
