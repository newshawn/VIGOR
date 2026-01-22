#!/usr/bin/env bash
set -e

PYTHON_VERSION="${1:-3.11.13}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${VENV_DIR:-$REPO_ROOT/.venv}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$REPO_ROOT/.uv-cache}"
cd "$REPO_ROOT"

# Some git-based deps (e.g. lighteval) may include Git LFS assets that can fail to download
# when network is restricted or the upstream LFS object is missing.
SKIP_GIT_LFS_SMUDGE="${SKIP_GIT_LFS_SMUDGE:-1}"
if [ "$SKIP_GIT_LFS_SMUDGE" = "1" ]; then
  export GIT_LFS_SKIP_SMUDGE=1
fi

# Install profiles:
# - cpu (default): skips GPU-only packages that often fail to build without CUDA
# - gpu: installs the full requirements.txt set (still requires a working CUDA toolchain)
INSTALL_PROFILE="${INSTALL_PROFILE:-cpu}"
DEFER_GPU_PACKAGES="${DEFER_GPU_PACKAGES:-1}"
VLLM_VERSION="${VLLM_VERSION:-0.8.4}"
FLASH_ATTN_VERSION="${FLASH_ATTN_VERSION:-2.7.4.post1}"

if ! command -v uv >/dev/null 2>&1; then
  echo "[ERROR] uv not found in PATH. Install uv first: https://docs.astral.sh/uv/" >&2
  exit 1
fi

if [ -f "$VENV_DIR/bin/activate" ]; then
  echo "[INFO] Reusing existing virtualenv at: $VENV_DIR"
else
  uv venv "$VENV_DIR" --python "$PYTHON_VERSION"
fi

# shellcheck disable=SC1090
source "$VENV_DIR/bin/activate"
export UV_PROJECT_ENVIRONMENT="$VENV_DIR"

uv pip install --upgrade pip
# Some VCS sdists (e.g. TRL) assume setuptools is present at build/metadata time.
uv pip install --upgrade setuptools wheel

REQ_FILE="$REPO_ROOT/requirements.txt"
TMP_REQ_FILE=""
cleanup() {
  if [ -n "$TMP_REQ_FILE" ] && [ -f "$TMP_REQ_FILE" ]; then
    rm -f "$TMP_REQ_FILE"
  fi
}
trap cleanup EXIT

if [ "$INSTALL_PROFILE" = "cpu" ]; then
  TMP_REQ_FILE="$(mktemp)"
  # Keep this filter conservative: only remove packages that are typically CUDA/GPU-specific builds.
  grep -Ev '^(flash[_-]attn|xformers|vllm|triton|deepspeed|cupy-cuda|nvidia-|liger_kernel)\\b' "$REQ_FILE" >"$TMP_REQ_FILE"
  REQ_FILE="$TMP_REQ_FILE"
  echo "[INFO] INSTALL_PROFILE=cpu: skipping GPU-only deps (flash-attn, xformers, vllm, triton, deepspeed, cupy-cuda*, nvidia-*, liger_kernel)"
elif [ "$INSTALL_PROFILE" = "gpu" ]; then
  echo "[INFO] INSTALL_PROFILE=gpu: installing full requirements.txt"
else
  echo "[ERROR] Unknown INSTALL_PROFILE='$INSTALL_PROFILE' (expected 'cpu' or 'gpu')" >&2
  exit 1
fi

if [ "$INSTALL_PROFILE" = "gpu" ]; then
  # Some packages (flash-attn, xformers, etc.) need torch at build/import time but may not declare it.
  # Install torch first and avoid build isolation so builds can import torch from the environment.
  uv pip install "torch==2.6.0" "torchvision==0.21.0" "torchaudio==2.6.0"

  if [ "$DEFER_GPU_PACKAGES" = "1" ]; then
    TMP_REQ_FILE="$(mktemp)"
    # Install these last to reduce build-order issues (mirrors common OpenR1 setups).
    grep -Ev '^(flash[_-]attn|vllm|torch|torchvision|torchaudio)\\b' "$REQ_FILE" >"$TMP_REQ_FILE"
    REQ_FILE="$TMP_REQ_FILE"
    echo "[INFO] DEFER_GPU_PACKAGES=1: will install vLLM/flash-attn after base deps"
  fi

  uv pip install -r "$REQ_FILE" --no-build-isolation

  if [ "$DEFER_GPU_PACKAGES" = "1" ]; then
    uv pip install "vllm==${VLLM_VERSION}"
    uv pip install "flash-attn==${FLASH_ATTN_VERSION}" --no-build-isolation
  fi
else
  uv pip install -r "$REQ_FILE"
fi

uv pip install -e "$REPO_ROOT" --no-deps --no-build-isolation

echo "[OK] Virtualenv ready at: $VENV_DIR"
