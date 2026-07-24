#!/usr/bin/env bash
# Install the Seed-VC (V2) voice-conversion engine: vendored model code, the two
# missing python deps, and every model checkpoint (into the models/ HF cache).
# Keep in sync with install-convert.ps1.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENDOR_DIR="$ROOT_DIR/vendor/seed-vc"
source "$ROOT_DIR/scripts/resolve-venv.sh"
PYTHON_BIN="$(resolve_workbench_venv "$ROOT_DIR")/bin/python"

echo "==> Vendoring Seed-VC model code"
if [ ! -f "$VENDOR_DIR/inference_v2.py" ]; then
  rm -rf "$VENDOR_DIR"
  git clone --depth 1 https://github.com/Plachtaa/seed-vc "$VENDOR_DIR"
else
  echo "    already present, skipping clone"
fi

echo "==> Installing python dependencies"
# Only hydra-core and munch are missing from this venv. Do NOT install Seed-VC's
# upstream requirements.txt: it pins torch 2.4 / transformers 4.46 / numpy 1.26
# and would break the rest of the app.
"$PYTHON_BIN" -m pip install hydra-core munch

echo "==> Downloading Seed-VC checkpoints (~1.6 GB into models/)"
SUBTITLE_WORKBENCH_ROOT="$ROOT_DIR" "$PYTHON_BIN" - <<'EOF'
import os
os.environ.setdefault("HF_HOME", os.path.join(os.environ["SUBTITLE_WORKBENCH_ROOT"], "models"))

from huggingface_hub import hf_hub_download, snapshot_download

# Seed-VC V2 weights + the quantizer/style checkpoints the wrapper loads directly.
hf_hub_download("Plachta/Seed-VC", "v2/cfm_small.pth")
hf_hub_download("Plachta/Seed-VC", "v2/ar_base.pth")
hf_hub_download("Plachta/ASTRAL-quantization", "bsq32/bsq32_light.pth")
hf_hub_download("Plachta/ASTRAL-quantization", "bsq2048/bsq2048_light.pth")
hf_hub_download("funasr/campplus", "campplus_cn_common.bin")

# Content encoder, vocoder, and the whisper-small tokenizer the wrapper instantiates.
snapshot_download("facebook/hubert-large-ll60k")
snapshot_download("nvidia/bigvgan_v2_22khz_80band_256x")
snapshot_download("openai/whisper-small", allow_patterns=["*.json", "*.txt", "*.model"])
print("Seed-VC checkpoints ready in", os.environ["HF_HOME"])
EOF

echo "==> Seed-VC install complete"
