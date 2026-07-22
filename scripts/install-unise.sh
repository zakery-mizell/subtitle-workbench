#!/usr/bin/env bash
# Install the UniSE overlap-separation engine: vendored model code, python
# dependencies, and model checkpoints (~2.8 GB into checkpoints/unise).
# Keep in sync with install-unise.ps1.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENDOR_DIR="$ROOT_DIR/vendor/unified-audio"
source "$ROOT_DIR/scripts/resolve-venv.sh"
PYTHON_BIN="$(resolve_workbench_venv "$ROOT_DIR")/bin/python"

echo "==> Vendoring QuarkAudio-UniSE model code"
if [ ! -f "$VENDOR_DIR/QuarkAudio-UniSE/model/model.py" ]; then
  rm -rf "$VENDOR_DIR"
  git clone --depth 1 https://github.com/alibaba/unified-audio "$VENDOR_DIR"
else
  echo "    already present, skipping clone"
fi

echo "==> Installing python dependencies"
"$PYTHON_BIN" -m pip install -r "$ROOT_DIR/backend/requirements-separation.txt"

echo "==> Downloading checkpoints (UniSE LM + BiCodec + wav2vec2, ~2.8 GB)"
SUBTITLE_WORKBENCH_ROOT="$ROOT_DIR" "$PYTHON_BIN" - <<'EOF'
import os
from pathlib import Path

from huggingface_hub import hf_hub_download, snapshot_download

checkpoints = Path(os.environ["SUBTITLE_WORKBENCH_ROOT"]) / "checkpoints" / "unise"
checkpoints.mkdir(parents=True, exist_ok=True)
snapshot_download(
    "SparkAudio/Spark-TTS-0.5B",
    local_dir=str(checkpoints),
    allow_patterns=["BiCodec/*", "wav2vec2-large-xlsr-53/*", "config.yaml"],
)
hf_hub_download(
    "QuarkAudio/QuarkAudio-UniSE",
    "epoch=20-step=109367.ckpt",
    local_dir=str(checkpoints),
)
print(f"Checkpoints ready in {checkpoints}")
EOF

echo "==> UniSE install complete"
