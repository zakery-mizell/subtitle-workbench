#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

"$ROOT/scripts/run-backend.sh" &
BACKEND_PID=$!
"$ROOT/scripts/run-frontend.sh" &
FRONTEND_PID=$!

trap 'kill "$BACKEND_PID" "$FRONTEND_PID" 2>/dev/null || true' EXIT INT TERM

sleep 2
open "http://localhost:5173" 2>/dev/null || true

wait
