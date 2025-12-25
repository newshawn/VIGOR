#!/usr/bin/env bash
set -euo pipefail

log() {
  echo "[bootstrap_h800_venv] $*"
}

WUSER="${WUSER:-wenxuexiang}"
WORK_ROOT="${WORK_ROOT:-/run/determined/localcq1/${WUSER}}"
TMP_ROOT="${TMP_ROOT:-${WORK_ROOT}/tmp}"

VENV_DST="${VENV_DST:-${WORK_ROOT}/Intuitor/open-r1-intuitor/openr1_intuitor}"

# Where you might have copied an existing venv from node1 (read-only /home or root-owned path)
VENV_SRC_1="${VENV_SRC_1:-/run/determined/localcq1/Intuitor/open-r1-intuitor/openr1_intuitor}"
VENV_SRC_2="${VENV_SRC_2:-/home/${WUSER}/projects/Intuitor/open-r1-intuitor/openr1_intuitor}"

# uv-managed Python (copied from /home to localcq1 for H800)
UV_PY_TAG="${UV_PY_TAG:-cpython-3.11.13-linux-x86_64-gnu}"
UV_SRC="${UV_SRC:-/home/${WUSER}/.local/share/uv/python/${UV_PY_TAG}}"
UV_DST="${UV_DST:-${WORK_ROOT}/uv/python/${UV_PY_TAG}}"

mkdir -p "${TMP_ROOT}"
export TMPDIR="${TMP_ROOT}"
export TMP="${TMP_ROOT}"
export TEMP="${TMP_ROOT}"

log "TMPDIR=${TMPDIR}"
log "WORK_ROOT=${WORK_ROOT}"
log "VENV_DST=${VENV_DST}"

copy_dir() {
  local src="$1"
  local dst="$2"
  if command -v rsync >/dev/null 2>&1; then
    rsync -aH "$src/" "$dst/"
  else
    mkdir -p "$(dirname "$dst")"
    cp -a "$src" "$(dirname "$dst")/"
  fi
}

if [[ ! -x "${UV_DST}/bin/python3.11" ]]; then
  if [[ -x "${UV_SRC}/bin/python3.11" ]]; then
    log "Copying uv python from ${UV_SRC} -> ${UV_DST} (one-time)"
    mkdir -p "$(dirname "${UV_DST}")"
    copy_dir "${UV_SRC}" "${UV_DST}"
  else
    log "UV python not found at ${UV_SRC}; will fall back to system python3.11 if available"
  fi
fi

BASE_PY=""
if [[ -x "${UV_DST}/bin/python3.11" ]]; then
  BASE_PY="${UV_DST}/bin/python3.11"
elif command -v python3.11 >/dev/null 2>&1; then
  BASE_PY="$(command -v python3.11)"
elif command -v python3 >/dev/null 2>&1; then
  BASE_PY="$(command -v python3)"
else
  log "ERROR: no usable base python found (need python3.11 or copied uv python)."
  exit 2
fi

log "BASE_PY=${BASE_PY}"
log "BASE_PY version: $(${BASE_PY} -V 2>&1 || true)"

# If destination venv is missing, try copying a prebuilt venv first (keeps site-packages).
if [[ ! -d "${VENV_DST}" ]]; then
  for candidate in "${VENV_SRC_1}" "${VENV_SRC_2}"; do
    if [[ -d "${candidate}" ]]; then
      log "VENV_DST missing; copying venv from ${candidate} -> ${VENV_DST}"
      mkdir -p "$(dirname "${VENV_DST}")"
      if command -v rsync >/dev/null 2>&1; then
        rsync -aH "${candidate}/" "${VENV_DST}/"
      else
        cp -a "${candidate}" "$(dirname "${VENV_DST}")/"
      fi
      break
    fi
  done
fi

# Upgrade (or create) the venv to point at BASE_PY, so it no longer depends on /home uv python.
if [[ -d "${VENV_DST}" ]]; then
  log "Upgrading existing venv in-place: ${VENV_DST}"
  "${BASE_PY}" -m venv --upgrade "${VENV_DST}"
else
  log "Creating new venv: ${VENV_DST}"
  mkdir -p "$(dirname "${VENV_DST}")"
  "${BASE_PY}" -m venv "${VENV_DST}"
fi

if [[ -f "${VENV_DST}/pyvenv.cfg" ]]; then
  log "pyvenv.cfg home: $(grep -E '^home\\s*=' -m1 "${VENV_DST}/pyvenv.cfg" || true)"
fi

log "VENV python: $("${VENV_DST}/bin/python" -c 'import sys; print(sys.executable)' 2>/dev/null || true)"
log "Done"

