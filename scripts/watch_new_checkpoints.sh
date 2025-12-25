#!/usr/bin/env bash

set -euo pipefail

usage() {
  cat <<'EOF'
Usage: scripts/watch_new_checkpoints.sh [options]

Options:
  --watch-dir PATH        Directory that receives new checkpoint-* folders.
  --eval-script PATH      Path to eval.sh (local mode only; default: repo_root/eval.sh).
  --log-dir PATH          Where auto-eval logs & state are stored (default: repo_root/logs/auto_eval).
  --poll-interval SECS    Interval for polling fallback when inotifywait is missing (default: 300).
  --process-existing      Also enqueue existing checkpoint-* folders on startup.
  --run-mode MODE         "local" (default) or "det" to submit Determined jobs.
  --det-config PATH       Determined YAML used in det mode (default: det_yaml/det-eval_4_24g.yaml).
  --det-context PATH      Context directory for det uploads (default: repo root).
  --det-cli PATH          det CLI executable name/path (default: det).
  --det-extra-arg ARG     Extra flag passed to det experiment create (repeatable).
  --results-root PATH     Base directory where eval outputs are written (default: repo_root/data/evals).
  --skip-existing-results Skip checkpoints that already have results (default: enabled; use --no-skip-existing-results to disable).
  -h, --help              Show this help and exit.

Environment variables (override the same options): WATCH_DIR, EVAL_SCRIPT, AUTO_EVAL_LOG_DIR,
AUTO_EVAL_POLL_INTERVAL, AUTO_EVAL_PROCESS_EXISTING, AUTO_EVAL_RUN_MODE,
AUTO_EVAL_DET_CONFIG, AUTO_EVAL_DET_CONTEXT, AUTO_EVAL_DET_CLI, AUTO_EVAL_DET_EXTRA_ARGS
(comma separated), AUTO_EVAL_RESULTS_ROOT, AUTO_EVAL_SKIP_EXISTING_RESULTS (0/1).
EOF
}

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

DEFAULT_WATCH_DIR="/run/determined/NAS1/public/xuexiang/Intuitor_ckpt/Qwen2.5-Intuitor-3B_20251104_084223"

WATCH_DIR="${WATCH_DIR:-$DEFAULT_WATCH_DIR}"
EVAL_SCRIPT="${EVAL_SCRIPT:-$PROJECT_ROOT/eval.sh}"
LOG_DIR="${AUTO_EVAL_LOG_DIR:-$PROJECT_ROOT/logs/auto_eval}"
POLL_INTERVAL="${AUTO_EVAL_POLL_INTERVAL:-300}"
PROCESS_EXISTING="${AUTO_EVAL_PROCESS_EXISTING:-0}"
RUN_MODE="${AUTO_EVAL_RUN_MODE:-local}"
DET_CONFIG="${AUTO_EVAL_DET_CONFIG:-$PROJECT_ROOT/det_yaml/det-eval_4_24g.yaml}"
DET_CONTEXT="${AUTO_EVAL_DET_CONTEXT:-$PROJECT_ROOT}"
DET_CLI="${AUTO_EVAL_DET_CLI:-det}"
declare -a DET_EXTRA_ARGS=()
if [[ -n "${AUTO_EVAL_DET_EXTRA_ARGS:-}" ]]; then
  IFS=',' read -r -a DET_EXTRA_ARGS <<< "${AUTO_EVAL_DET_EXTRA_ARGS}"
fi
RESULTS_ROOT="${AUTO_EVAL_RESULTS_ROOT:-$PROJECT_ROOT/data/evals}"
SKIP_EXISTING_RESULTS="${AUTO_EVAL_SKIP_EXISTING_RESULTS:-1}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --watch-dir)
      [[ $# -ge 2 ]] || usage
      WATCH_DIR="$2"
      shift 2
      ;;
    --eval-script)
      [[ $# -ge 2 ]] || usage
      EVAL_SCRIPT="$2"
      shift 2
      ;;
    --log-dir)
      [[ $# -ge 2 ]] || usage
      LOG_DIR="$2"
      shift 2
      ;;
    --poll-interval)
      [[ $# -ge 2 ]] || usage
      POLL_INTERVAL="$2"
      shift 2
      ;;
    --process-existing)
      PROCESS_EXISTING=1
      shift
      ;;
    --run-mode)
      [[ $# -ge 2 ]] || usage
      RUN_MODE="$2"
      shift 2
      ;;
    --det-config)
      [[ $# -ge 2 ]] || usage
      DET_CONFIG="$2"
      shift 2
      ;;
    --det-context)
      [[ $# -ge 2 ]] || usage
      DET_CONTEXT="$2"
      shift 2
      ;;
    --det-cli)
      [[ $# -ge 2 ]] || usage
      DET_CLI="$2"
      shift 2
      ;;
    --det-extra-arg)
      [[ $# -ge 2 ]] || usage
      DET_EXTRA_ARGS+=("$2")
      shift 2
      ;;
    --results-root)
      [[ $# -ge 2 ]] || usage
      RESULTS_ROOT="$2"
      shift 2
      ;;
    --skip-existing-results)
      SKIP_EXISTING_RESULTS=1
      shift
      ;;
    --no-skip-existing-results)
      SKIP_EXISTING_RESULTS=0
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage
      exit 1
      ;;
  esac
done

RUN_MODE=$(echo "$RUN_MODE" | tr 'A-Z' 'a-z')
if [[ "$RUN_MODE" != "local" && "$RUN_MODE" != "det" ]]; then
  echo "Unsupported run mode: $RUN_MODE" >&2
  exit 1
fi

if [[ ! -e "$WATCH_DIR" ]]; then
  echo "Watch directory does not exist; creating: $WATCH_DIR" >&2
  if ! mkdir -p "$WATCH_DIR"; then
    echo "Failed to create watch directory: $WATCH_DIR" >&2
    exit 1
  fi
fi

if [[ ! -d "$WATCH_DIR" ]]; then
  echo "Watch path is not a directory: $WATCH_DIR" >&2
  exit 1
fi

if [[ "$RUN_MODE" == "local" && ! -x "$EVAL_SCRIPT" ]]; then
  echo "Eval script is not executable: $EVAL_SCRIPT" >&2
  echo "Run 'chmod +x $EVAL_SCRIPT' if needed." >&2
  exit 1
fi

if [[ "$RUN_MODE" == "det" ]]; then
  if [[ ! -f "$DET_CONFIG" ]]; then
    echo "det config not found: $DET_CONFIG" >&2
    exit 1
  fi
  if [[ ! -d "$DET_CONTEXT" ]]; then
    echo "det context directory not found: $DET_CONTEXT" >&2
    exit 1
  fi
  if ! command -v "$DET_CLI" >/dev/null 2>&1; then
    echo "det CLI not found: $DET_CLI" >&2
    exit 1
  fi
fi

mkdir -p "$LOG_DIR"
STATE_FILE="$LOG_DIR/processed_checkpoints.txt"
touch "$STATE_FILE"

declare -A PROCESSED=()
while IFS= read -r line || [[ -n "$line" ]]; do
  [[ -z "$line" ]] && continue
  PROCESSED["$line"]=1
done < "$STATE_FILE"

checkpoint_eval_dir() {
  local path="${1%/}"
  local parent child
  parent="$(basename "$(dirname "$path")")"
  child="$(basename "$path")"
  echo "$RESULTS_ROOT/$parent/$child"
}

checkpoint_has_results() {
  local eval_dir
  eval_dir="$(checkpoint_eval_dir "$1")"
  [[ -d "$eval_dir/results" ]]
}

declare -a DET_TMP_CONFIGS=()

cleanup_and_exit() {
  for tmp in "${DET_TMP_CONFIGS[@]}"; do
    [[ -n "$tmp" && -f "$tmp" ]] && rm -f "$tmp"
  done
  log "Stopping watcher"
  exit 0
}

log() {
  printf '[%(%Y-%m-%dT%H:%M:%S%z)T] %s\n' -1 "$*"
}

note_processed() {
  local path="$1"
  if [[ -z "${PROCESSED["$path"]:-}" ]]; then
    PROCESSED["$path"]=1
    echo "$path" >> "$STATE_FILE"
  fi
}

create_det_config() {
  local checkpoint="$1"
  local tmpfile
  tmpfile="$(mktemp "/tmp/det_eval_${2}.XXXX.yaml")"
  if ! python - "$DET_CONFIG" "$tmpfile" "$checkpoint" <<'PY'; then
import sys
try:
    import yaml
except ImportError:
    sys.stderr.write("PyYAML is required for det mode. Install via 'pip install pyyaml'.\n")
    sys.exit(1)

src, dst, model_path = sys.argv[1:4]
with open(src, 'r', encoding='utf-8') as f:
    data = yaml.safe_load(f)

env = data.setdefault('environment', {})
env_vars = env.setdefault('environment_variables', [])
if isinstance(env_vars, dict):
    env_vars = [f"{k}={v}" for k, v in env_vars.items()]

filtered = []
for item in env_vars:
    if isinstance(item, str) and item.startswith('MODEL_PATH='):
        continue
    filtered.append(item)
filtered.append(f"MODEL_PATH={model_path}")
env['environment_variables'] = filtered

with open(dst, 'w', encoding='utf-8') as f:
    yaml.safe_dump(data, f, sort_keys=False)
PY
    rm -f "$tmpfile"
    return 1
  fi
  DET_TMP_CONFIGS+=("$tmpfile")
  echo "$tmpfile"
}

submit_det_job() {
  local checkpoint="$1"
  local label="$2"
  local timestamp="$(date +%Y%m%d_%H%M%S)"
  local logfile="$LOG_DIR/${label}_${timestamp}_det.log"
  local tmp_config
  tmp_config="$(create_det_config "$checkpoint" "$label")" || return

  log "Submitting Determined eval for $label (config: $tmp_config)"
  local cmd=("$DET_CLI" experiment create "$tmp_config" "$DET_CONTEXT")
  if [[ ${#DET_EXTRA_ARGS[@]} -gt 0 ]]; then
    cmd+=("${DET_EXTRA_ARGS[@]}")
  fi
  (
    set +e
    "${cmd[@]}" >"$logfile" 2>&1
    status=$?
    if [[ $status -ne 0 ]]; then
      log "det submission failed for $label (exit $status). See $logfile"
    else
      log "det submission succeeded for $label. See $logfile for experiment ID."
    fi
    exit $status
  ) &
}

launch_eval() {
  local checkpoint="$1"
  local label="$2"
  if [[ "$RUN_MODE" == "det" ]]; then
    submit_det_job "$checkpoint" "$label"
    return
  fi

  local timestamp
  timestamp="$(date +%Y%m%d_%H%M%S)"
  local logfile="$LOG_DIR/${label}_${timestamp}.log"

  log "Running eval for $label (log: $logfile)"
  (
    set +e
    bash "$EVAL_SCRIPT" "$checkpoint" >"$logfile" 2>&1
    status=$?
    printf '\nExit code: %s\n' "$status" >>"$logfile"
    if [[ $status -ne 0 ]]; then
      log "Eval failed for $label (exit $status). See $logfile"
    else
      log "Eval finished for $label"
    fi
    exit $status
  ) &
}

process_checkpoint() {
  local path="${1%/}"
  [[ -d "$path" ]] || return

  local base
  base="$(basename "$path")"
  [[ "$base" == checkpoint-* ]] || return

  if [[ -n "${PROCESSED["$path"]:-}" ]]; then
    return
  fi

  if [[ "$SKIP_EXISTING_RESULTS" == "1" ]] && checkpoint_has_results "$path"; then
    local eval_dir
    eval_dir="$(checkpoint_eval_dir "$path")/results"
    log "Skip $base because results already exist at $eval_dir"
    note_processed "$path"
    return
  fi

  note_processed "$path"
  launch_eval "$path" "$base"
}

process_existing_once() {
  while IFS= read -r dir; do
    process_checkpoint "$dir"
  done < <(find "$WATCH_DIR" -maxdepth 1 -mindepth 1 -type d -name 'checkpoint-*' | sort)
}

trap cleanup_and_exit INT TERM

if [[ "$PROCESS_EXISTING" == "1" ]]; then
  log "Processing existing checkpoint-* folders before watching"
  process_existing_once
fi

watch_with_inotify() {
  log "Watching $WATCH_DIR via inotifywait"
  while IFS= read -r new_path; do
    process_checkpoint "$new_path"
  done < <(inotifywait -m -e create -e moved_to --format '%w%f' "$WATCH_DIR")
}

watch_with_polling() {
  log "inotifywait not found; falling back to polling every ${POLL_INTERVAL}s"
  while true; do
    process_existing_once
    sleep "$POLL_INTERVAL"
  done
}

if command -v inotifywait >/dev/null 2>&1; then
  watch_with_inotify
else
  watch_with_polling
fi
