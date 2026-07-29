# Install the Sidon speech-restoration engine (sarulab-speech/sidon-v0.1):
# the TorchScript checkpoints plus the log-mel front-end config, into the
# models/ HF cache. Keep in sync with install-sidon.sh.
#
# Sidon needs no packages beyond the base install (torch, torchaudio,
# transformers, huggingface_hub are all already there), so this script only
# verifies them and fetches weights.
$ErrorActionPreference = "Stop"

$RootDir = Split-Path -Parent $PSScriptRoot
. (Join-Path $PSScriptRoot "resolve-venv.ps1")
$VenvDir = Resolve-WorkbenchVenv -Root $RootDir
$PythonBin = Join-Path $VenvDir "Scripts\python.exe"
if (-not (Test-Path $PythonBin)) {
    $PythonBin = Join-Path $VenvDir "python.exe"
}

Write-Host "==> Checking Sidon python dependencies"
$CheckScript = @"
import sys

missing = []
for package in ('torch', 'torchaudio', 'transformers', 'huggingface_hub'):
    try:
        __import__(package)
    except ImportError:
        missing.append(package)
if missing:
    sys.exit(
        'Missing: ' + ', '.join(missing) + '. These ship with the base install; '
        'run scripts/install.ps1 first.'
    )
print('All Sidon dependencies are already installed.')
"@
& $PythonBin -c $CheckScript

Write-Host "==> Downloading Sidon checkpoint (sarulab-speech/sidon-v0.1)"
$DownloadScript = @"
import os
os.environ.setdefault('HF_HOME', os.path.join(r'$RootDir', 'models'))

import torch
import transformers
from huggingface_hub import hf_hub_download

# The archives are torch.jit.trace exports with device-pinned constants, so the
# cpu and cuda pairs are NOT interchangeable -- fetch only the one this machine
# will actually load (about 1 GB), matching restore/sidon_engine.py's policy.
variant = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Fetching the {variant} variant (~1 GB)')
for name in (f'feature_extractor_{variant}.pt', f'decoder_{variant}.pt'):
    print(f'  {name} -> ' + hf_hub_download('sarulab-speech/sidon-v0.1', name))

# The predictor was exported from w2v-BERT 2.0's input_features, so the log-mel
# front-end comes from that repo (a small config, not the model weights).
transformers.SeamlessM4TFeatureExtractor.from_pretrained('facebook/w2v-bert-2.0')
print('Log-mel front-end ready (facebook/w2v-bert-2.0)')
"@
& $PythonBin -c $DownloadScript

Write-Host "==> Sidon restore install complete"
