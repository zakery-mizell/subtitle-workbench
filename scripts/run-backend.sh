#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT/scripts/resolve-venv.sh"
VENV="$(resolve_workbench_venv "$ROOT")"

if [[ ! -x "$VENV/bin/python" ]]; then
  echo "Virtual environment missing. Run scripts/install.sh first." >&2
  exit 1
fi

# Fall back to CPU for any torch op MPS does not implement yet.
export PYTORCH_ENABLE_MPS_FALLBACK=1

cd "$ROOT"
# --reload-dir backend: only watch source code. Model downloads write .py-named
# files into the models/ HF caches, which would restart the server mid-job and
# lose the in-process job registry.
exec "$VENV/bin/python" -m uvicorn backend.app.main:app --reload --reload-dir backend --host 127.0.0.1 --port 8000
