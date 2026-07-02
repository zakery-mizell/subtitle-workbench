#!/usr/bin/env bash
# Double-click launcher for macOS: starts the backend and frontend,
# then opens http://localhost:5173 in the default browser.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ ! -x "$ROOT/scripts/run-app.sh" ]]; then
  echo "scripts/run-app.sh is missing. Re-clone the repo or run scripts/install.sh first."
  read -r -p "Press Enter to close..."
  exit 1
fi

source "$ROOT/scripts/resolve-venv.sh"
VENV="$(resolve_workbench_venv "$ROOT")"
if [[ ! -x "$VENV/bin/python" ]]; then
  echo "No Python environment found. Running the installer first (this can take a while)..."
  "$ROOT/scripts/install.sh"
fi

echo "Starting Subtitle Workbench... close this window (or press Ctrl+C) to stop it."
exec "$ROOT/scripts/run-app.sh"
