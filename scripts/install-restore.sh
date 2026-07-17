#!/usr/bin/env bash
# Install the Diamond speech-restoration engine (nineninesix/diamond-1.0):
# python dependencies and the model checkpoint (diamond.safetensors +
# diamond.json into the models/ HF cache). Keep in sync with install-restore.ps1.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="$("$ROOT_DIR/scripts/resolve-venv.sh")/python"

echo "==> Installing Diamond python dependencies"
# descript-audiotools pins protobuf<3.20, but this venv needs protobuf 7
# (onnxruntime). Install the descript packages WITHOUT their deps so pip does
# not downgrade protobuf, then add the runtime deps they actually need, then
# the Diamond inference package itself. pip prints a protobuf "dependency
# resolver" conflict warning here -- it is expected and harmless.
"$PYTHON_BIN" -m pip install --no-deps descript-audio-codec descript-audiotools
"$PYTHON_BIN" -m pip install argbind ffmpy flatten-dict markdown2 pystoi randomname fire torch-stoi tensorboard importlib_resources
"$PYTHON_BIN" -m pip install --no-deps "git+https://github.com/nineninesix-ai/diamond-inference"

echo "==> Downloading Diamond checkpoint (nineninesix/diamond-1.0)"
SUBTITLE_WORKBENCH_ROOT="$ROOT_DIR" "$PYTHON_BIN" - <<'EOF'
import os
os.environ.setdefault("HF_HOME", os.path.join(os.environ["SUBTITLE_WORKBENCH_ROOT"], "models"))

from huggingface_hub import hf_hub_download

# Both files land in the same snapshot dir; the .json config must sit next to
# the .safetensors weights.
ckpt = hf_hub_download("nineninesix/diamond-1.0", "diamond.safetensors")
hf_hub_download("nineninesix/diamond-1.0", "diamond.json")
print(f"Checkpoint ready at {ckpt}")
EOF

echo "==> Diamond restore install complete"
