#!/usr/bin/env bash
# Install the F5-TTS speech-edit engine (SWivid/F5-TTS, F5TTS_v1_Base): python
# dependencies and the checkpoint + vocoder (into the models/ HF cache). Keep in
# sync with install-speechedit.ps1.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT_DIR/scripts/resolve-venv.sh"
PYTHON_BIN="$(resolve_workbench_venv "$ROOT_DIR")/bin/python"

echo "==> Installing F5-TTS python dependencies"
# f5-tts's pyproject drags in gradio/bitsandbytes/etc. that inference never
# imports, and gradio's fastapi/pydantic pins collide with this app -- so install
# f5-tts WITHOUT its deps. The wheel still ships configs/*.yaml and the bundled
# vocab. Then add the runtime deps traced through real imports. The last three
# (ema_pytorch, wandb, datasets) are IMPORT-TIME-ONLY landmines: f5_tts/model/
# __init__.py unconditionally imports the Trainer (-> ema_pytorch, wandb) and
# dataset.py (-> datasets); inference uses none of them but the imports must
# resolve. hydra-core is shared with the Convert engine; listed here so this
# script stands alone. Everything else (torch, torchaudio, transformers, numpy,
# librosa, x-transformers, accelerate, soundfile, pydub, click, omegaconf,
# matplotlib, safetensors, tqdm, torchcodec) already lives in the venv.
"$PYTHON_BIN" -m pip install --no-deps f5-tts
"$PYTHON_BIN" -m pip install cached_path hydra-core vocos torchdiffeq pypinyin rjieba tomli ema_pytorch wandb datasets

echo "==> Downloading F5-TTS checkpoint + vocos vocoder (into models/)"
SUBTITLE_WORKBENCH_ROOT="$ROOT_DIR" "$PYTHON_BIN" - <<'EOF'
import os
os.environ["HF_HOME"] = os.path.join(os.environ["SUBTITLE_WORKBENCH_ROOT"], "models")

from huggingface_hub import hf_hub_download

cache_dir = os.path.join(os.environ["HF_HOME"], "hub")

# F5TTS_v1_Base flow-matching checkpoint (~1.35 GB).
ckpt = hf_hub_download("SWivid/F5-TTS", "F5TTS_v1_Base/model_1250000.safetensors", cache_dir=cache_dir)
print(f"Checkpoint ready at {ckpt}")

# vocos mel-24khz vocoder (~54 MB): config + weights.
hf_hub_download("charactr/vocos-mel-24khz", "config.yaml", cache_dir=cache_dir)
voc = hf_hub_download("charactr/vocos-mel-24khz", "pytorch_model.bin", cache_dir=cache_dir)
print(f"Vocoder ready at {voc}")
EOF

echo "==> Verifying dependency graph (expected f5-tts noise below)"
# --no-deps means pip check WILL flag f5-tts's own unmet gradio/bitsandbytes pins.
# That noise is expected and harmless (we skipped those pins on purpose); only
# non-f5-tts complaints matter. Never fail the script on it.
if ! "$PYTHON_BIN" -m pip check; then
  echo "    WARNING: pip check reported conflicts. The f5-tts gradio/bitsandbytes"
  echo "    complaints are EXPECTED (installed with --no-deps); investigate only"
  echo "    lines that mention other packages."
fi

echo "==> F5-TTS speech-edit install complete"
