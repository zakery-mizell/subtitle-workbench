$root = Split-Path -Parent $PSScriptRoot
. (Join-Path $PSScriptRoot "resolve-venv.ps1")

$venv = Resolve-WorkbenchVenv -Root $root
$venvPython = Join-Path $venv "Scripts\python.exe"

if (-not (Test-Path $venvPython)) {
  throw "Virtual environment missing. Run scripts/install.ps1 first."
}

Push-Location $root
try {
  & $venvPython -m uvicorn backend.app.main:app --reload --host 127.0.0.1 --port 8000
} finally {
  Pop-Location
}
