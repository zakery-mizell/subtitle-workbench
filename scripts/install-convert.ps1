# Install the Seed-VC (V2) voice-conversion engine: vendored model code, the two
# missing python deps, and every model checkpoint (into the models/ HF cache).
# Keep in sync with install-convert.sh.
$ErrorActionPreference = "Stop"

$RootDir = Split-Path -Parent $PSScriptRoot
$VendorDir = Join-Path $RootDir "vendor/seed-vc"
. (Join-Path $PSScriptRoot "resolve-venv.ps1")
$VenvDir = Resolve-WorkbenchVenv -Root $RootDir
$PythonBin = Join-Path $VenvDir "Scripts\python.exe"
if (-not (Test-Path $PythonBin)) {
    $PythonBin = Join-Path $VenvDir "python.exe"
}

Write-Host "==> Vendoring Seed-VC model code"
if (-not (Test-Path (Join-Path $VendorDir "inference_v2.py"))) {
    if (Test-Path $VendorDir) { Remove-Item -Recurse -Force $VendorDir }
    git clone --depth 1 https://github.com/Plachtaa/seed-vc $VendorDir
} else {
    Write-Host "    already present, skipping clone"
}

Write-Host "==> Installing python dependencies"
# Only hydra-core and munch are missing from this venv. Do NOT install Seed-VC's
# upstream requirements.txt: it pins torch 2.4 / transformers 4.46 / numpy 1.26
# and would break the rest of the app.
& $PythonBin -m pip install hydra-core munch

Write-Host "==> Downloading Seed-VC checkpoints (~1.6 GB into models/)"
$DownloadScript = @"
import os
os.environ.setdefault('HF_HOME', os.path.join(r'$RootDir', 'models'))

from huggingface_hub import hf_hub_download, snapshot_download

# Seed-VC V2 weights + the quantizer/style checkpoints the wrapper loads directly.
hf_hub_download('Plachta/Seed-VC', 'v2/cfm_small.pth')
hf_hub_download('Plachta/Seed-VC', 'v2/ar_base.pth')
hf_hub_download('Plachta/ASTRAL-quantization', 'bsq32/bsq32_light.pth')
hf_hub_download('Plachta/ASTRAL-quantization', 'bsq2048/bsq2048_light.pth')
hf_hub_download('funasr/campplus', 'campplus_cn_common.bin')

# Content encoder, vocoder, and the whisper-small tokenizer the wrapper instantiates.
snapshot_download('facebook/hubert-large-ll60k')
snapshot_download('nvidia/bigvgan_v2_22khz_80band_256x')
snapshot_download('openai/whisper-small', allow_patterns=['*.json', '*.txt', '*.model'])
print('Seed-VC checkpoints ready in', os.environ['HF_HOME'])
"@
& $PythonBin -c $DownloadScript

Write-Host "==> Seed-VC install complete"
