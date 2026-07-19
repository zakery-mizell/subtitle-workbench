# Install the UniSE overlap-separation engine: vendored model code, python
# dependencies, and model checkpoints (~2.8 GB into checkpoints/unise).
# Keep in sync with install-unise.sh.
$ErrorActionPreference = "Stop"

$RootDir = Split-Path -Parent $PSScriptRoot
$VendorDir = Join-Path $RootDir "vendor/unified-audio"
. (Join-Path $PSScriptRoot "resolve-venv.ps1")
$VenvDir = Resolve-WorkbenchVenv -Root $RootDir
$PythonBin = Join-Path $VenvDir "Scripts\python.exe"
if (-not (Test-Path $PythonBin)) {
    $PythonBin = Join-Path $VenvDir "python.exe"
}

Write-Host "==> Vendoring QuarkAudio-UniSE model code"
if (-not (Test-Path (Join-Path $VendorDir "QuarkAudio-UniSE/model/model.py"))) {
    if (Test-Path $VendorDir) { Remove-Item -Recurse -Force $VendorDir }
    git clone --depth 1 https://github.com/alibaba/unified-audio $VendorDir
} else {
    Write-Host "    already present, skipping clone"
}

Write-Host "==> Installing python dependencies"
& $PythonBin -m pip install -r (Join-Path $RootDir "backend/requirements-separation.txt")

Write-Host "==> Downloading checkpoints (UniSE LM + BiCodec + wav2vec2, ~2.8 GB)"
$DownloadScript = @"
from pathlib import Path

from huggingface_hub import hf_hub_download, snapshot_download

checkpoints = Path(r'$RootDir') / 'checkpoints' / 'unise'
checkpoints.mkdir(parents=True, exist_ok=True)
snapshot_download(
    'SparkAudio/Spark-TTS-0.5B',
    local_dir=str(checkpoints),
    allow_patterns=['BiCodec/*', 'wav2vec2-large-xlsr-53/*', 'config.yaml'],
)
hf_hub_download(
    'QuarkAudio/QuarkAudio-UniSE',
    'epoch=20-step=109367.ckpt',
    local_dir=str(checkpoints),
)
print(f'Checkpoints ready in {checkpoints}')
"@
& $PythonBin -c $DownloadScript

Write-Host "==> UniSE install complete"
