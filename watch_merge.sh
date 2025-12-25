#!/usr/bin/env bash
# Watch one or more RUN_ROOT/ckpt folders for new checkpoint-* dirs and trigger merge.sh.
# 轮询一个或多个 RUN_ROOT/ckpt 目录，一旦发现新的 checkpoint-* 目录，就调用 merge.sh 合并。
#
# Usage:
#   bash watch_merge.sh RUN_ROOT [RUN_ROOT ...] [poll_interval_seconds]
#   bash watch_merge.sh -i 60 RUN_ROOT [RUN_ROOT ...]
#
# nohup example:
#   nohup bash watch_merge.sh RUN_ROOT1 RUN_ROOT2 60 > watch_merge.log 2>&1 &
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash watch_merge.sh RUN_ROOT [RUN_ROOT ...] [poll_interval_seconds]
  bash watch_merge.sh -i 60 RUN_ROOT [RUN_ROOT ...]

Notes:
  - Each RUN_ROOT is a directory that contains a 'ckpt/' folder.
  - You can also pass a '.../ckpt' directory directly.
EOF
}

INTERVAL=60
interval_set=false
inputs=()

while [[ $# -gt 0 ]]; do
  case "${1}" in
    -h|--help)
      usage
      exit 0
      ;;
    -i|--interval)
      if [[ $# -lt 2 ]]; then
        echo "Missing value for ${1}" >&2
        exit 1
      fi
      INTERVAL="${2}"
      interval_set=true
      shift 2
      ;;
    *)
      inputs+=("${1}")
      shift
      ;;
  esac
done

if [[ ${#inputs[@]} -eq 0 ]]; then
  usage
  exit 1
fi

# Backward-compatible: last positional numeric arg is treated as interval.
if [[ "${interval_set}" == false && ${#inputs[@]} -ge 2 && "${inputs[-1]}" =~ ^[0-9]+$ ]]; then
  INTERVAL="${inputs[-1]}"
  unset 'inputs[-1]'
fi

if ! [[ "${INTERVAL}" =~ ^[0-9]+$ && "${INTERVAL}" -gt 0 ]]; then
  echo "Invalid interval: ${INTERVAL} (must be a positive integer seconds)" >&2
  exit 1
fi

SCRIPT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

run_roots=()
ckpt_roots=()
for input in "${inputs[@]}"; do
  if [[ -d "${input}" && "$(basename "${input}")" == "ckpt" ]]; then
    ckpt_root="${input}"
    run_root="$(dirname "${ckpt_root}")"
  else
    run_root="${input}"
    ckpt_root="${run_root}/ckpt"
  fi
  run_roots+=("${run_root}")
  ckpt_roots+=("${ckpt_root}")
done

echo "Watching the following run roots (interval=${INTERVAL}s):"
for run_root in "${run_roots[@]}"; do
  echo "  ${run_root}"
done

declare -A last_status=()

while true; do
  for i in "${!run_roots[@]}"; do
    RUN_ROOT="${run_roots[$i]}"
    CKPT_ROOT="${ckpt_roots[$i]}"

    if [[ ! -d "${RUN_ROOT}" ]]; then
      if [[ "${last_status[${RUN_ROOT}]-}" != "missing_run_root" ]]; then
        echo "$(date '+%F %T') run root not found yet: ${RUN_ROOT} (waiting...)"
        last_status["${RUN_ROOT}"]="missing_run_root"
      fi
      continue
    fi

    if [[ ! -d "${CKPT_ROOT}" ]]; then
      if [[ "${last_status[${RUN_ROOT}]-}" != "missing_ckpt_root" ]]; then
        echo "$(date '+%F %T') ckpt dir not found yet: ${CKPT_ROOT} (waiting...)"
        last_status["${RUN_ROOT}"]="missing_ckpt_root"
      fi
      continue
    fi
    if [[ "${last_status[${RUN_ROOT}]-}" != "ok" ]]; then
      echo "$(date '+%F %T') found ckpt dir: ${CKPT_ROOT}"
      last_status["${RUN_ROOT}"]="ok"
    fi

    needs_merge=false  # 标记是否检测到尚未合并的新 checkpoint
    shopt -s nullglob
    for ckpt in "${CKPT_ROOT}"/checkpoint*; do
      [[ -d "${ckpt}" ]] || continue
      name="$(basename "${ckpt}")"
      out="${RUN_ROOT}/merged/${name}"
      if [[ -d "${out}" ]]; then
        continue  # already merged
      fi
      needs_merge=true
      break
    done
    shopt -u nullglob

    if [[ "${needs_merge}" == true ]]; then
      echo "$(date '+%F %T') Detected new checkpoint under ${CKPT_ROOT}, running merge.sh (RUN_ROOT=${RUN_ROOT})"
      if ! RUN_ROOT="${RUN_ROOT}" bash "${SCRIPT_DIR}/merge.sh"; then
        echo "$(date '+%F %T') merge.sh failed for RUN_ROOT=${RUN_ROOT} (will retry in next scan)" >&2
      fi
    fi
  done

  sleep "${INTERVAL}"
done
