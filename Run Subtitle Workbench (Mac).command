#!/usr/bin/env bash
# Double-click launcher for macOS: starts the backend and frontend,
# then opens http://localhost:5173 in the default browser.
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Keep the Terminal window open so any error stays readable on a failure.
fail() {
  echo ""
  echo "Something went wrong: $1"
  read -r -p "Press Enter to close this window..."
  exit 1
}

if [[ ! -x "$ROOT/scripts/run-app.sh" ]]; then
  fail "scripts/run-app.sh is missing. Re-clone the repo or run scripts/install.sh first."
fi

source "$ROOT/scripts/resolve-venv.sh"
VENV="$(resolve_workbench_venv "$ROOT")"
if [[ ! -x "$VENV/bin/python" ]]; then
  echo "No Python environment found. Running the installer first (this can take a while)..."
  "$ROOT/scripts/install.sh" || fail "Install failed. If it says Python 3.12 was not found, run: brew install python@3.12"
fi

echo "Starting Subtitle Workbench... close this window (or press Ctrl+C) to stop it."
exec "$ROOT/scripts/run-app.sh"
