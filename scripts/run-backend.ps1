$root = Split-Path -Parent $PSScriptRoot
. (Join-Path $PSScriptRoot "resolve-venv.ps1")

$venv = Resolve-WorkbenchVenv -Root $root
$venvPython = Join-Path $venv "Scripts\python.exe"

if (-not (Test-Path $venvPython)) {
  throw "Virtual environment missing. Run scripts/install.ps1 first."
}

Push-Location $root
try {
  # --reload-dir backend: only watch source code. Model downloads write
  # .py-named files into the models/ HF caches, which would restart the
  # server mid-job and lose the in-process job registry.
  & $venvPython -m uvicorn backend.app.main:app --reload --reload-dir backend --host 127.0.0.1 --port 8000
} finally {
  Pop-Location
}
