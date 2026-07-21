#!/usr/bin/env bash
# Install the Qwen3-TTS voice-cloning engine (Qwen/Qwen3-TTS-12Hz-*-Base):
# python dependencies and the default model checkpoint (into the models/ HF
# cache). Keep in sync with install-tts.ps1.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT_DIR/scripts/resolve-venv.sh"
PYTHON_BIN="$(resolve_workbench_venv "$ROOT_DIR")/bin/python"

echo "==> Installing Qwen3-TTS python dependencies"
# qwen-tts's normal install pins transformers==4.57.3 / accelerate==1.12.0 and
# drags in gradio, which would fight this venv (transformers 4.57.6 works). So
# install qwen-tts WITHOUT its deps, then add accelerate + sox unpinned. pysox
# is an import-time dep (the 25 Hz tokenizer imports it at module level) but
# the sox BINARY is never invoked by the 12 Hz models, so its "SoX could not
# be found" notice on import is harmless. gradio (demo UI only) is NOT needed;
# librosa/onnxruntime/einops/soundfile/torchaudio already present.
"$PYTHON_BIN" -m pip install --no-deps qwen-tts==0.1.1
"$PYTHON_BIN" -m pip install accelerate sox

echo "==> Downloading Qwen3-TTS checkpoint (Qwen/Qwen3-TTS-12Hz-1.7B-Base)"
SUBTITLE_WORKBENCH_ROOT="$ROOT_DIR" "$PYTHON_BIN" - <<'EOF'
import os
os.environ.setdefault("HF_HOME", os.path.join(os.environ["SUBTITLE_WORKBENCH_ROOT"], "models"))

from huggingface_hub import snapshot_download

path = snapshot_download("Qwen/Qwen3-TTS-12Hz-1.7B-Base")
print(f"Checkpoint ready at {path}")
EOF

echo "==> The 0.6b variant (Qwen/Qwen3-TTS-12Hz-0.6B-Base) downloads lazily on first use if selected."
echo "==> Qwen3-TTS install complete"
