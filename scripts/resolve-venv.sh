#!/usr/bin/env bash
# Resolves the workbench venv path, honoring SUBTITLE_WORKBENCH_VENV or .venv-path.
resolve_workbench_venv() {
  local root="$1"
  local override="${SUBTITLE_WORKBENCH_VENV:-}"
  if [[ -z "$override" && -f "$root/.venv-path" ]]; then
    override="$(tr -d '[:space:]' < "$root/.venv-path")"
  fi
  if [[ -n "$override" ]]; then
    echo "$override"
  else
    echo "$root/.venv"
  fi
}
