from __future__ import annotations

"""Inference wrapper around QuarkAudio-UniSE (vendored in vendor/unified-audio).

UniSE is an autoregressive LM over BiCodec speech tokens. It exposes three
task heads we use:
  - "se":   restore/denoise the mixture (no enrollment)
  - "tse":  reconstruct only the enrolled speaker from the mixture
  - "rtse": reconstruct everything except the enrolled speaker

All audio in this module is mono float32 at 16 kHz. The model generates in
fixed 5-second windows; output is re-synthesized speech (generative), so it
must be blended back into the original recording rather than used as-is.
"""

import math
import os
import sys
import threading
from pathlib import Path

import numpy as np

from ..config import settings
from ..device import select_device

UNISE_SAMPLE_RATE = 16000
WINDOW_SECONDS = 5.0
WINDOW_SAMPLES = int(WINDOW_SECONDS * UNISE_SAMPLE_RATE)
# Consecutive windows are generated independently, so their seams need a
# crossfade; without the overlap a hard concatenation clicks every 5 s.
WINDOW_OVERLAP_SECONDS = 0.5
WINDOW_OVERLAP_SAMPLES = int(WINDOW_OVERLAP_SECONDS * UNISE_SAMPLE_RATE)
WINDOW_HOP_SAMPLES = WINDOW_SAMPLES - WINDOW_OVERLAP_SAMPLES
ENROLL_SECONDS = 5.0
ENROLL_SAMPLES = int(ENROLL_SECONDS * UNISE_SAMPLE_RATE)

ROOT_DIR = Path(__file__).resolve().parents[3]
VENDOR_DIR = ROOT_DIR / "vendor" / "unified-audio" / "QuarkAudio-UniSE"
CHECKPOINT_DIR = ROOT_DIR / "checkpoints" / "unise"
LM_CHECKPOINT = CHECKPOINT_DIR / "epoch=20-step=109367.ckpt"

# Mirrors vendor/unified-audio/QuarkAudio-UniSE/conf/config.yaml (inference-relevant keys).
MODEL_CONFIG = {
    "codec_ckpt_dir": str(CHECKPOINT_DIR),
    "stft_config": {"hop_length": 320, "win_length": 640, "n_fft": 640, "n_mels": 80},
    "llm_config": {
        "num_tasks": 3,
        "task_map": {"se": 0, "tse": 1, "rtse": 2},
        "feats_dim": 768,
        "llm_base_config": {
            "cond_dim": 80,
            "global_size": 4096,
            "semantic_size": 8192,
            "hidden_size": 512,
            "num_layers": 12,
            "num_attention_heads": 8,
            "dropout_p": 0.1,
            "max_position_embeddings": 4096,
            "label_smoothing": 0.1,
            "conformer_params": {
                "num_layers": 6,
                "dim": 512,
                "heads": 8,
                "dim_head": 64,
                "depthwise_conv_kernel_size": 31,
                "ff_mult": 4,
                "dropout": 0.1,
                "qk_norm": None,
                "pe_attn_head": 1,
            },
        },
    },
}

_lock = threading.Lock()
_engine_cache: dict[str, "UniSEEngine"] = {}


class SeparationUnavailable(RuntimeError):
    """Raised when UniSE cannot run at all (missing vendor code/checkpoints/deps)."""


def _missing_asset_message() -> str | None:
    if not (VENDOR_DIR / "model" / "model.py").is_file():
        return (
            "The UniSE model code is missing. Run scripts/install-unise.sh "
            "(or .ps1) to clone it into vendor/unified-audio."
        )
    if not LM_CHECKPOINT.is_file():
        return (
            "The UniSE checkpoint is missing. Run scripts/install-unise.sh "
            "(or .ps1) to download it into checkpoints/unise."
        )
    for required in ("BiCodec/model.safetensors", "wav2vec2-large-xlsr-53/pytorch_model.bin", "config.yaml"):
        if not (CHECKPOINT_DIR / required).is_file():
            return (
                f"The BiCodec codec files are missing ({required}). Run "
                "scripts/install-unise.sh (or .ps1) to download them."
            )
    return None


class UniSEEngine:
    """Loaded UniSE model bound to one torch device."""

    def __init__(self, device: str) -> None:
        message = _missing_asset_message()
        if message:
            raise SeparationUnavailable(message)

        # Hard-set, not setdefault: transcribe_with_whisperx points HF_HOME at
        # the whisper cache process-wide, so after any transcription in this
        # process a setdefault is a no-op and the vendored model's ambient-cache
        # downloads (e.g. microsoft/wavlm-base-plus from model/model.py) silently
        # re-download into models/whisper.
        os.environ["HF_HOME"] = str(settings.model_cache_dir)
        import torch

        vendor_path = str(VENDOR_DIR)
        if vendor_path not in sys.path:
            sys.path.insert(0, vendor_path)
        try:
            from model import Model  # vendored QuarkAudio-UniSE package

            from .transformers_compat import patch_vendored_unise

            patch_vendored_unise()
        except ImportError as exc:
            raise SeparationUnavailable(
                "UniSE python dependencies are missing. Install them with: "
                "pip install -r backend/requirements-separation.txt"
            ) from exc

        self._torch = torch
        model = Model(config=MODEL_CONFIG)
        checkpoint = torch.load(str(LM_CHECKPOINT), map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()
        self.device = device
        self.model = model.to(torch.device(device))

    # ---- feature helpers -------------------------------------------------

    def _to_tensor(self, wav: np.ndarray):
        tensor = self._torch.from_numpy(np.ascontiguousarray(wav, dtype=np.float32))
        if tensor.ndim == 1:
            tensor = tensor[None, :]
        return tensor.to(self._torch.device(self.device))

    @staticmethod
    def _window_starts(length: int) -> list[int]:
        """Offsets of the overlapping 5 s windows that cover `length` samples."""
        starts = [0]
        while starts[-1] + WINDOW_SAMPLES < length:
            starts.append(starts[-1] + WINDOW_HOP_SAMPLES)
        return starts

    def _windows(self, mix: np.ndarray) -> np.ndarray:
        """Overlapping 5 s windows, shaped (n_windows, WINDOW_SAMPLES).

        The last window is wrap-padded from the start of the region; only its
        non-padded part is kept when the outputs are reassembled. Kept as a
        numpy array on the host: a full-length recording has hundreds of
        windows, and they are uploaded to the device one at a time.
        """
        n = mix.shape[-1]
        starts = self._window_starts(n)
        padded = np.pad(mix, (0, starts[-1] + WINDOW_SAMPLES - n), mode="wrap")
        return np.stack([padded[start : start + WINDOW_SAMPLES] for start in starts])

    @staticmethod
    def _reassemble(pieces: list[np.ndarray], starts: list[int], length: int) -> np.ndarray:
        """Lay the window outputs back on the timeline, crossfading their seams."""
        out = np.zeros(length, dtype=np.float32)
        for index, (start, piece) in enumerate(zip(starts, pieces)):
            valid = min(WINDOW_SAMPLES, length - start)
            if valid <= 0:
                break
            # The generator's output length can drift from the window it was given.
            window = np.zeros(WINDOW_SAMPLES, dtype=np.float32)
            usable = min(WINDOW_SAMPLES, piece.shape[-1])
            window[:usable] = piece[:usable]
            fade = min(WINDOW_OVERLAP_SAMPLES, valid) if index else 0
            if fade:
                theta = np.linspace(0.0, np.pi / 2.0, fade, dtype=np.float32)
                out[start : start + fade] *= np.cos(theta)
                out[start : start + fade] += window[:fade] * np.sin(theta)
            out[start + fade : start + valid] = window[fade:valid]
        return out

    @staticmethod
    def _peak_normalize(wav: np.ndarray, peak: float = 0.95) -> np.ndarray:
        top = float(np.max(np.abs(wav))) if wav.size else 0.0
        if top < 1e-6:
            return wav.astype(np.float32)
        return (wav * (peak / top)).astype(np.float32)

    def prepare_enrollment(self, voice: np.ndarray) -> np.ndarray:
        """Trim/tile a solo-speech sample to exactly 5 s of enrollment audio."""
        voice = np.asarray(voice, dtype=np.float32).reshape(-1)
        if voice.size == 0:
            raise ValueError("Enrollment audio is empty")
        if voice.size < ENROLL_SAMPLES:
            voice = np.tile(voice, math.ceil(ENROLL_SAMPLES / voice.size))
        return self._peak_normalize(voice[:ENROLL_SAMPLES], peak=0.9)

    # ---- inference -------------------------------------------------------

    @staticmethod
    def _report(progress, fraction: float) -> None:
        if progress:
            progress(min(1.0, max(0.0, fraction)))

    def run_task(
        self,
        task: str,
        mix: np.ndarray,
        enrollment: np.ndarray | None = None,
        progress=None,
    ) -> np.ndarray:
        """Run one UniSE task over the full mixture; returns 16 kHz mono audio.

        Windows are generated one at a time (not batched) so progress can be
        reported and peak memory stays flat for long regions.
        """
        if task not in ("se", "tse", "rtse"):
            raise ValueError(f"Unknown UniSE task: {task}")
        if task in ("tse", "rtse") and enrollment is None:
            raise ValueError(f"Task '{task}' requires enrollment audio")

        torch = self._torch
        mix = np.asarray(mix, dtype=np.float32).reshape(-1)
        length = mix.shape[-1]
        normalized = self._peak_normalize(mix)
        starts = self._window_starts(length)
        windows = self._windows(normalized)

        enroll_mel = enroll_feats = None
        if enrollment is not None:
            enroll = self._to_tensor(self.prepare_enrollment(enrollment))
            with torch.no_grad():
                enroll_mel = self.model.stft_logmel(enroll)
                enroll_feats = self.model.extract_semantic_features(enroll)

        outputs: list[np.ndarray] = []
        with torch.no_grad():
            for index in range(windows.shape[0]):
                window = self._to_tensor(windows[index])
                mix_mel = self.model.stft_logmel(window)
                mix_feats = self.model.extract_semantic_features(window)
                global_ids, semantic_ids = self.model.dnn.generate(
                    task_name=task,
                    enroll_mel=enroll_mel,
                    enroll_feats=enroll_feats,
                    mix_mel=mix_mel,
                    mix_feats=mix_feats,
                    do_sample=False,
                )
                estimate = self.model.tokenizer.detokenize(global_ids.unsqueeze(1), semantic_ids)
                outputs.append(estimate.reshape(-1).cpu().numpy())
                self._report(progress, (index + 1) / windows.shape[0])

        return self._reassemble(outputs, starts, length)


def load_engine(device_preference: str | None = None) -> UniSEEngine:
    """Load (or reuse) the UniSE engine, falling back to CPU if the GPU fails."""
    device = select_device(device_preference or settings.mastering_device)
    with _lock:
        cached = _engine_cache.get(device) or _engine_cache.get("cpu")
        if cached is not None:
            return cached
        try:
            engine = UniSEEngine(device)
        except SeparationUnavailable:
            raise
        except Exception:
            if device == "cpu":
                raise
            engine = UniSEEngine("cpu")
        _engine_cache[engine.device] = engine
        return engine
