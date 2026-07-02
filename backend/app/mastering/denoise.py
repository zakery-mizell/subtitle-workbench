from __future__ import annotations

import os
from typing import Callable

import numpy as np

from ..config import settings
from ..device import select_device
from .audio_io import MasterAudio

CHUNK_SECONDS = 30.0
OVERLAP_SECONDS = 1.0
MODEL_NAME = "MossFormer2_SE_48K"

_model_cache: dict[str, object] = {}


class DenoiseUnavailable(RuntimeError):
    """Raised when the denoiser cannot run at all (missing package/model)."""


def _load_model() -> object:
    if MODEL_NAME in _model_cache:
        return _model_cache[MODEL_NAME]

    os.environ.setdefault("HF_HOME", settings.model_cache_dir)
    try:
        from clearvoice import ClearVoice
    except ImportError as exc:
        raise DenoiseUnavailable(
            "The 'clearvoice' package is not installed, so AI denoising was skipped. "
            "Install it with: pip install clearvoice"
        ) from exc

    try:
        model = ClearVoice(task="speech_enhancement", model_names=[MODEL_NAME])
    except Exception as exc:  # model download/load failures
        raise DenoiseUnavailable(f"The {MODEL_NAME} denoiser model could not be loaded. Details: {exc}") from exc

    _model_cache[MODEL_NAME] = model
    return model


def _enhance_chunk(model: object, chunk: np.ndarray, sample_rate: int, temp_dir: str) -> np.ndarray:
    """Run one mono chunk through ClearVoice via a temp wav file."""
    import soundfile as sf
    from tempfile import NamedTemporaryFile

    os.makedirs(temp_dir, exist_ok=True)
    with NamedTemporaryFile(delete=False, suffix=".wav", dir=temp_dir) as handle:
        chunk_path = handle.name
    try:
        sf.write(chunk_path, chunk.astype(np.float32), sample_rate)
        enhanced = model(input_path=chunk_path, online_write=False)
        enhanced = np.asarray(enhanced, dtype=np.float32).reshape(-1)
        if enhanced.size < chunk.size:
            enhanced = np.pad(enhanced, (0, chunk.size - enhanced.size))
        return enhanced[: chunk.size]
    finally:
        try:
            os.unlink(chunk_path)
        except FileNotFoundError:
            pass


def denoise(
    audio: MasterAudio,
    amount: float,
    progress: Callable[[float], None] | None = None,
) -> tuple[MasterAudio, str]:
    """Denoise per channel in overlapping chunks; returns (audio, device_used).

    Raises DenoiseUnavailable when the model cannot run; the pipeline turns
    that into a warning and keeps the unprocessed audio.
    """
    model = _load_model()
    device = select_device(settings.mastering_device)

    sr = audio.sample_rate
    chunk = int(CHUNK_SECONDS * sr)
    overlap = int(OVERLAP_SECONDS * sr)
    step = chunk - overlap
    n = audio.samples.shape[1]
    wet = np.empty_like(audio.samples)

    starts = list(range(0, max(n - overlap, 1), step)) if n > chunk else [0]
    total_chunks = len(starts) * audio.channels
    done = 0

    for channel_index in range(audio.channels):
        channel = audio.samples[channel_index]
        out = np.zeros(n, dtype=np.float32)
        weight = np.zeros(n, dtype=np.float32)
        for start in starts:
            end = min(n, start + chunk)
            enhanced = _enhance_chunk(model, channel[start:end], sr, settings.temp_upload_dir)
            ramp = np.ones(end - start, dtype=np.float32)
            fade = min(overlap, end - start)
            if start > 0 and fade > 0:
                ramp[:fade] = np.linspace(0.0, 1.0, fade, dtype=np.float32)
            if end < n and fade > 0:
                ramp[-fade:] = np.minimum(ramp[-fade:], np.linspace(1.0, 0.0, fade, dtype=np.float32))
            out[start:end] += enhanced * ramp
            weight[start:end] += ramp
            done += 1
            if progress:
                progress(done / total_chunks)
        wet[channel_index] = out / np.maximum(weight, 1e-6)

    mix = float(np.clip(amount / 100.0, 0.0, 1.0))
    audio.samples = (wet * mix + audio.samples * (1.0 - mix)).astype(np.float32)
    return audio, device
