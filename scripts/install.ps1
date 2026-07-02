$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $PSScriptRoot
. (Join-Path $PSScriptRoot "resolve-venv.ps1")

$venv = Resolve-WorkbenchVenv -Root $root
$venvPython = Join-Path $venv "Scripts\python.exe"
$venvPip = Join-Path $venv "Scripts\pip.exe"

function Resolve-Python312 {
  $pyLauncher = Get-Command py -ErrorAction SilentlyContinue
  if ($pyLauncher) {
    try {
      $null = & $pyLauncher.Source -3.12 -c "import sys; print(sys.executable)"
      if ($LASTEXITCODE -eq 0) {
        return $pyLauncher.Source
      }
    } catch {
    }
  }

  $pythonCommand = Get-Command python -ErrorAction SilentlyContinue
  if ($pythonCommand) {
    try {
      $version = & $pythonCommand.Source -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"
      if ($LASTEXITCODE -eq 0 -and $version -eq "3.12") {
        return $pythonCommand.Source
      }
    } catch {
    }
  }

  throw "Python 3.12 was not found. Install it and make sure either 'py -3.12' or 'python' resolves to Python 3.12."
}

$python = Resolve-Python312

if (-not (Test-Path $venvPython)) {
  New-Item -ItemType Directory -Force -Path (Split-Path -Parent $venv) | Out-Null
  if ($python -like "*py.exe") {
    & $python -3.12 -m venv $venv
  } else {
    & $python -m venv $venv
  }
}

& $venvPython -m pip install --upgrade pip
& $venvPip install --index-url https://download.pytorch.org/whl/cu128 torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0
& $venvPip install -r (Join-Path $root "backend\requirements.txt")
# AI denoiser (MossFormer2-SE-48K). Optional: mastering degrades gracefully without it.
# clearvoice pins numpy<2, so install it without deps and provide the runtime deps ourselves.
try {
  & $venvPip install -r (Join-Path $root "backend\requirements-denoise.txt")
  & $venvPip install --no-deps clearvoice
} catch {
  Write-Warning "clearvoice install failed; AI denoising will be skipped."
}

Push-Location (Join-Path $root "frontend")
try {
  npm install
} finally {
  Pop-Location
}

Write-Host "Install complete."
