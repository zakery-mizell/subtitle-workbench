# Install the Diamond speech-restoration engine (nineninesix/diamond-1.0):
# python dependencies and the model checkpoint (diamond.safetensors +
# diamond.json into the models/ HF cache). Keep in sync with install-restore.sh.
$ErrorActionPreference = "Stop"

$RootDir = Split-Path -Parent $PSScriptRoot
. (Join-Path $PSScriptRoot "resolve-venv.ps1")
$VenvDir = Resolve-WorkbenchVenv -Root $RootDir
$PythonBin = Join-Path $VenvDir "Scripts\python.exe"
if (-not (Test-Path $PythonBin)) {
    $PythonBin = Join-Path $VenvDir "python.exe"
}

Write-Host "==> Installing Diamond python dependencies"
# descript-audiotools pins protobuf<3.20, but this venv needs protobuf 7
# (onnxruntime). Install the descript packages WITHOUT their deps so pip does
# not downgrade protobuf, then add the runtime deps they actually need, then
# the Diamond inference package itself. pip prints a protobuf "dependency
# resolver" conflict warning here -- it is expected and harmless.
& $PythonBin -m pip install --no-deps descript-audio-codec descript-audiotools
& $PythonBin -m pip install argbind ffmpy flatten-dict markdown2 pystoi randomname fire torch-stoi tensorboard importlib_resources
& $PythonBin -m pip install --no-deps "git+https://github.com/nineninesix-ai/diamond-inference"

Write-Host "==> Downloading Diamond checkpoint (nineninesix/diamond-1.0)"
$DownloadScript = @"
import os
os.environ.setdefault('HF_HOME', os.path.join(r'$RootDir', 'models'))

from huggingface_hub import hf_hub_download

# Both files land in the same snapshot dir; the .json config must sit next to
# the .safetensors weights.
ckpt = hf_hub_download('nineninesix/diamond-1.0', 'diamond.safetensors')
hf_hub_download('nineninesix/diamond-1.0', 'diamond.json')
print(f'Checkpoint ready at {ckpt}')
"@
& $PythonBin -c $DownloadScript

Write-Host "==> Diamond restore install complete"
