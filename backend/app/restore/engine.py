"""Diamond speech-restoration engine (nineninesix/diamond-1.0).

Diamond is a sequence-to-sequence RQ-Transformer that resynthesizes clean
speech from degraded input. Output is generative and always 44.1 kHz, so it is
meant to replace the input rather than blend into it. All audio in this module
is mono float32; the model resamples to 24 kHz internally.

Device policy: Diamond is a small autoregressive model whose per-step kernel
launch overhead dominates runtime. Measured on this Mac (Apple Silicon): CPU
RTF ~11, MPS RTF ~20 -- MPS is SLOWER than CPU because the tiny autoregressive
kernels are launch-bound. So the auto policy is: cuda if available, else cpu,
and NEVER auto-pick mps. mps is only used when RESTORE_DEVICE explicitly names
it.
"""
from __future__ import annotations

import os
import threading
from typing import Callable

import numpy as np

from ..config import settings

RESTORE_INPUT_RATE = 24000  # model resamples to 24 kHz internally; feed it 24 kHz
RESTORE_OUTPUT_RATE = 44100  # Diamond always returns 44.1 kHz

# Segment long audio for bounded memory + progress reporting; crossfade the
# per-segment outputs at the OUTPUT rate (mirror mastering/denoise.py's 30s/1s).
SEGMENT_SECONDS = 30.0
SEGMENT_OVERLAP_SECONDS = 1.0

CHECKPOINT_REPO = "nineninesix/diamond-1.0"

_lock = threading.Lock()
_engine_cache: dict[str, "RestoreEngine"] = {}


class RestoreUnavailable(RuntimeError):
    """Raised when Diamond cannot run at all (missing package/checkpoint)."""


def _resolve_device() -> str:
    """cuda if available, else cpu; never auto-mps (see module docstring)."""
    configured = settings.restore_device
    if configured:
        return configured
    try:
        import torch
    except ImportError:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


class RestoreEngine:
    """Loaded Diamond model bound to one torch device."""

    def __init__(self, device: str) -> None:
        os.environ.setdefault("HF_HOME", str(settings.model_cache_dir))
        try:
            from huggingface_hub import hf_hub_download
        except ImportError as exc:
            raise RestoreUnavailable(
                "The 'huggingface_hub' package is missing, so speech restoration "
                "was skipped. Install the restore dependencies with "
                "scripts/install-restore.sh (or .ps1)."
            ) from exc
        try:
            from diamond import Diamond
            from diamond.infer import SampleConfig
        except ImportError as exc:
            raise RestoreUnavailable(
                "The 'diamond' speech-restoration package is not installed, so "
                "restoration was skipped. Install it with "
                "scripts/install-restore.sh (or .ps1)."
            ) from exc

        try:
            # Both files must land in the same snapshot dir; the .json config
            # sits next to the .safetensors weights (hf_hub_download does this).
            checkpoint = hf_hub_download(CHECKPOINT_REPO, "diamond.safetensors")
            hf_hub_download(CHECKPOINT_REPO, "diamond.json")
        except Exception as exc:  # download/network failures
            raise RestoreUnavailable(
                f"The Diamond checkpoint ({CHECKPOINT_REPO}) could not be "
                f"downloaded. Details: {exc}"
            ) from exc

        self._SampleConfig = SampleConfig
        self.device = device
        self.model = Diamond.from_pretrained(checkpoint, device=device)

    def restore_segment(
        self,
        segment: np.ndarray,
        sr: int,
        *,
        chunk_sec: float,
        overlap_sec: float,
        rep_penalty: float,
    ) -> np.ndarray:
        """Restore one mono segment; returns 44.1 kHz mono float32.

        trim_leadin/normalize MUST stay False here: trim_leadin would shift the
        segment alignment and normalize would pump gain per segment.
        """
        restored, out_sr = self.model.restore(
            segment,
            sr,
            chunk_sec=chunk_sec,
            overlap_sec=overlap_sec,
            warmup_sec=1.0,
            tail_pad_sec=1.0,
            normalize=False,
            trim_leadin=False,
            recover_collapse=True,
            sample=self._SampleConfig(rep_penalty=rep_penalty),
        )
        if out_sr != RESTORE_OUTPUT_RATE:  # documented invariant; guard anyway
            raise RestoreUnavailable(
                f"Diamond returned {out_sr} Hz, expected {RESTORE_OUTPUT_RATE} Hz."
            )
        return np.asarray(restored, dtype=np.float32).reshape(-1)


def load_engine(device_preference: str | None = None) -> RestoreEngine:
    """Load (or reuse) the Diamond engine, falling back to CPU if the GPU fails."""
    device = device_preference or _resolve_device()
    with _lock:
        cached = _engine_cache.get(device) or _engine_cache.get("cpu")
        if cached is not None:
            return cached
        try:
            engine = RestoreEngine(device)
        except RestoreUnavailable:
            raise
        except Exception:
            if device == "cpu":
                raise
            engine = RestoreEngine("cpu")
        _engine_cache[engine.device] = engine
        return engine


def _format_clock(seconds: float) -> str:
    total = max(0, int(round(seconds)))
    return f"{total // 60}:{total % 60:02d}"


def restore_waveform(
    wav: np.ndarray,
    sr: int,
    *,
    chunk_sec: float = 2.5,
    overlap_sec: float = 0.4,
    rep_penalty: float = 1.3,
    progress: Callable[[float, str], None] | None = None,
) -> tuple[np.ndarray, int]:
    """Restore a full mono waveform in overlapping segments; returns (audio, 44100).

    The HF demo Space seeds both RNGs for reproducibility, so we do the same.
    Segments are restored independently and crossfaded at the OUTPUT rate; each
    input segment of `seg_len` samples yields a deterministic output length of
    round(seg_len / sr * 44100).
    """
    import torch

    torch.manual_seed(42)
    np.random.seed(42)

    engine = load_engine()

    wav = np.asarray(wav, dtype=np.float32).reshape(-1)
    n = wav.size
    if n == 0:
        raise ValueError("Restore input is empty.")

    seg_len = int(SEGMENT_SECONDS * sr)
    overlap_in = int(SEGMENT_OVERLAP_SECONDS * sr)
    step = seg_len - overlap_in
    starts = list(range(0, max(n - overlap_in, 1), step)) if n > seg_len else [0]
    total_segments = len(starts)

    out_rate = RESTORE_OUTPUT_RATE
    total_out = round(n / sr * out_rate)
    out = np.zeros(total_out, dtype=np.float32)
    weight = np.zeros(total_out, dtype=np.float32)
    overlap_out = int(SEGMENT_OVERLAP_SECONDS * out_rate)

    total_label = _format_clock(n / sr)

    for index, start in enumerate(starts):
        end = min(n, start + seg_len)
        segment = wav[start:end]
        try:
            restored = engine.restore_segment(
                segment, sr, chunk_sec=chunk_sec, overlap_sec=overlap_sec, rep_penalty=rep_penalty
            )
        except RestoreUnavailable:
            raise
        except Exception:
            if engine.device == "cpu":
                raise
            engine = load_engine("cpu")
            restored = engine.restore_segment(
                segment, sr, chunk_sec=chunk_sec, overlap_sec=overlap_sec, rep_penalty=rep_penalty
            )

        out_start = round(start / sr * out_rate)
        length = min(restored.size, total_out - out_start)
        if length <= 0:
            continue
        restored = restored[:length]
        ramp = np.ones(length, dtype=np.float32)
        fade = min(overlap_out, length)
        if start > 0 and fade > 0:
            ramp[:fade] = np.linspace(0.0, 1.0, fade, dtype=np.float32)
        if end < n and fade > 0:
            ramp[-fade:] = np.minimum(ramp[-fade:], np.linspace(1.0, 0.0, fade, dtype=np.float32))
        out[out_start : out_start + length] += restored * ramp
        weight[out_start : out_start + length] += ramp

        if progress:
            progress((index + 1) / total_segments, f"Restoring {_format_clock(end / sr)} / {total_label}")

    restored_full = out / np.maximum(weight, 1e-6)
    peak = float(np.max(np.abs(restored_full))) if restored_full.size else 0.0
    if peak > 1.0:  # single global peak normalize; never boost quiet audio
        restored_full = restored_full * (0.99 / peak)
    return restored_full.astype(np.float32), out_rate
