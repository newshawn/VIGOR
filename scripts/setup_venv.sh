#!/usr/bin/env bash
set -e

PYTHON_VERSION="${1:-3.11.13}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${VENV_DIR:-$REPO_ROOT/.venv}"

if ! command -v uv >/dev/null 2>&1; then
  echo "[ERROR] uv not found in PATH. Install uv first: https://docs.astral.sh/uv/" >&2
  exit 1
fi

uv venv "$VENV_DIR" --python "$PYTHON_VERSION"

# shellcheck disable=SC1090
source "$VENV_DIR/bin/activate"
export UV_PROJECT_ENVIRONMENT="$VENV_DIR"

uv pip install --upgrade pip
uv pip install -r "$REPO_ROOT/requirements.txt"
uv pip install -e . --no-deps

echo "[OK] Virtualenv ready at: $VENV_DIR"
