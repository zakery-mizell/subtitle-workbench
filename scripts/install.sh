#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT/scripts/resolve-venv.sh"
VENV="$(resolve_workbench_venv "$ROOT")"

find_python312() {
  for candidate in python3.12 /opt/homebrew/opt/python@3.12/bin/python3.12 /usr/local/opt/python@3.12/bin/python3.12; do
    if command -v "$candidate" >/dev/null 2>&1 || [[ -x "$candidate" ]]; then
      echo "$candidate"
      return 0
    fi
  done
  return 1
}

if ! PYTHON="$(find_python312)"; then
  echo "Python 3.12 was not found. Install it first, e.g.: brew install python@3.12" >&2
  exit 1
fi

if ! command -v ffmpeg >/dev/null 2>&1; then
  echo "ffmpeg was not found on PATH. Install it first, e.g.: brew install ffmpeg" >&2
  exit 1
fi

if [[ ! -x "$VENV/bin/python" ]]; then
  mkdir -p "$(dirname "$VENV")"
  "$PYTHON" -m venv "$VENV"
fi

"$VENV/bin/python" -m pip install --upgrade pip
# Default PyPI torch wheels on macOS include MPS (Apple Silicon GPU) support.
"$VENV/bin/pip" install torch==2.8.0 torchaudio==2.8.0
"$VENV/bin/pip" install -r "$ROOT/backend/requirements.txt"
# AI denoiser (MossFormer2-SE-48K). Optional: mastering degrades gracefully without it.
# clearvoice pins numpy<2, so install it without deps and provide the runtime deps ourselves.
if "$VENV/bin/pip" install -r "$ROOT/backend/requirements-denoise.txt" && "$VENV/bin/pip" install --no-deps clearvoice; then
  echo "AI denoiser installed."
else
  echo "WARNING: clearvoice install failed; AI denoising will be skipped." >&2
fi

(cd "$ROOT/frontend" && npm install)

echo "Install complete."
