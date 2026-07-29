"""Sidon speech-restoration engine (sarulab-speech/sidon-v0.1).

Sidon restores degraded speech in two stages: a LoRA-adapted w2v-BERT 2.0
feature predictor cleanses the SSL features of the noisy input, and a decoder
vocoder resynthesises a waveform from those cleansed features. Output is always
48 kHz. All audio in this module is mono float32.

Unlike Diamond (see engine.py) it is multilingual and far cheaper: measured on
this Mac (M1 Max, CPU) at RTF ~0.2 -- about 5x FASTER than real time, against
Diamond's ~11x slower. It is also the more conservative repair: it cleanses
features rather than regenerating a token stream, so it does not invent words
the way Diamond can on badly clipped input.

Device policy: the published checkpoints are torch.jit.trace exports whose
constants are pinned to the tracing device, and they ship as separate
_cpu.pt / _cuda.pt pairs. Load the artifact that MATCHES the device and never
move a loaded module across devices. mps is unsupported in both directions:
map_location='mps' dies on a float64 constant in the archive, and loading on
cpu then calling .to('mps') fails at runtime with "Passed CPU tensor to MPS
op". So the auto policy is cuda if available, else cpu -- and an explicit
SIDON_DEVICE=mps is rejected rather than silently producing garbage.

Chunking: the feature predictor runs full self-attention over a chunk, so peak
memory scales with chunk length. Measured over a 150 s clip on CPU, with the
log-mel cosine taken against the 96 s render (the demo Space's setting):

    chunk    peak RSS    log-mel cosine vs 96 s
     96 s     23.9 GB     1.000 (reference)
     30 s     11.7 GB     0.993
     15 s      6.0 GB     0.989
     10 s      4.7 GB     0.982

Shorter chunks cost a little quality because the predictor sees less context.
The default (15 s) is chosen so a CUDA run fits in a mid-size card's VRAM; on a
64 GB Mac 30 s is comfortable. Waveform correlation between two renders is
meaningless here -- the vocoder regenerates phase, so two perceptually
identical renders can have a waveform cosine near zero. Compare log-mels.

Chunk seams do NOT accumulate timing drift: each chunk after the first carries
one feature frame of context forward and trims one frame of output, so its
contribution is exactly chunk_sec * 48000 samples. Measured envelope lag
against the input was 0 to -1 frames (<= 10 ms) at both 2 and 6 chunks, so
restored audio stays aligned to the workspace timeline.
"""
from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import Callable

import numpy as np

from ..config import settings
from .engine import RestoreUnavailable

SIDON_INPUT_RATE = 16000  # the feature predictor's rate; input is resampled here
SIDON_OUTPUT_RATE = 48000  # Sidon always returns 48 kHz

CHECKPOINT_REPO = "sarulab-speech/sidon-v0.1"
PREPROCESSOR_REPO = "facebook/w2v-bert-2.0"  # ships the SeamlessM4T log-mel front-end

DEFAULT_CHUNK_SECONDS = 15.0
# One stacked feature frame is 20 ms, which the decoder expands by x960.
FRAME_SAMPLES_OUT = 960
# The reference implementation pads the tail so the last words are not cut off
# by the frame boundary; the surplus is trimmed off the final output.
TAIL_PAD_SAMPLES = 24000  # 1.5 s at 16 kHz
# The log-mel front-end needs a few frames to produce anything; fold a shorter
# trailing chunk back into its predecessor rather than feeding it a stub.
MIN_CHUNK_SAMPLES = 4800  # 0.3 s at 16 kHz
# Front-end padding applied per chunk by the reference implementation.
EDGE_PAD_SAMPLES = 160

_lock = threading.Lock()
_engine_cache: dict[str, "SidonEngine"] = {}


def _resolve_device() -> str:
    """cuda if available, else cpu; mps is unsupported (see module docstring)."""
    configured = settings.sidon_device
    if configured:
        if configured.startswith("mps"):
            raise RestoreUnavailable(
                "SIDON_DEVICE=mps is not supported: the published Sidon checkpoints "
                "are traced with device-pinned constants and fail on MPS. Use cpu "
                "(about 5x faster than real time) or cuda."
            )
        return configured
    try:
        import torch
    except ImportError:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


class SidonEngine:
    """The two loaded Sidon TorchScript modules bound to one torch device."""

    def __init__(self, device: str) -> None:
        # Hard-set, not setdefault, for the same reason as the Diamond engine:
        # transcribe_with_whisperx force-points HF_HOME at the whisper cache
        # process-wide, and a job transcribes BEFORE loading this engine.
        os.environ["HF_HOME"] = str(settings.model_cache_dir)
        try:
            import torch
            import transformers
            from huggingface_hub import hf_hub_download
        except ImportError as exc:
            raise RestoreUnavailable(
                "Sidon needs torch, torchaudio and transformers, which are missing. "
                "Install them with scripts/install-sidon.sh (or .ps1)."
            ) from exc

        # The archives are traced per device and are not interchangeable.
        variant = "cuda" if device.startswith("cuda") else "cpu"
        cache_dir = str(Path(settings.model_cache_dir) / "hub")
        try:
            fe_path = hf_hub_download(
                CHECKPOINT_REPO, f"feature_extractor_{variant}.pt", cache_dir=cache_dir
            )
            decoder_path = hf_hub_download(
                CHECKPOINT_REPO, f"decoder_{variant}.pt", cache_dir=cache_dir
            )
        except Exception as exc:  # download/network failures
            raise RestoreUnavailable(
                f"The Sidon checkpoint ({CHECKPOINT_REPO}) could not be downloaded. "
                f"Details: {exc}"
            ) from exc

        try:
            self.preprocessor = transformers.SeamlessM4TFeatureExtractor.from_pretrained(
                PREPROCESSOR_REPO
            )
        except Exception as exc:
            raise RestoreUnavailable(
                f"Sidon's log-mel front-end ({PREPROCESSOR_REPO}) could not be loaded. "
                f"Details: {exc}"
            ) from exc

        self.device = device
        self.feature_extractor = torch.jit.load(fe_path, map_location=device).eval()
        self.decoder = torch.jit.load(decoder_path, map_location=device).eval()

    def restore_chunk(self, chunk, feature_cache):
        """Restore one 16 kHz chunk; returns (output samples, next feature cache).

        The cache is the chunk's last feature frame, prepended to the next
        chunk so the vocoder has continuity across the seam; the matching
        trailing frame is dropped from this chunk's output so the seam neither
        duplicates nor drops audio.
        """
        import torch

        inputs = self.preprocessor(
            torch.nn.functional.pad(chunk, (EDGE_PAD_SAMPLES, EDGE_PAD_SAMPLES)),
            sampling_rate=SIDON_INPUT_RATE,
            return_tensors="pt",
        )
        features = self.feature_extractor(inputs["input_features"].to(self.device))[
            "last_hidden_state"
        ]
        if feature_cache is not None:
            features = torch.cat([feature_cache, features], dim=1)
        audio = self.decoder(features.transpose(1, 2)).view(-1)[:-FRAME_SAMPLES_OUT]
        return audio, features[:, -1:]


def load_engine(device_preference: str | None = None) -> SidonEngine:
    """Load (or reuse) the Sidon engine, falling back to CPU if the GPU fails."""
    device = device_preference or _resolve_device()
    with _lock:
        cached = _engine_cache.get(device) or _engine_cache.get("cpu")
        if cached is not None:
            return cached
        try:
            engine = SidonEngine(device)
        except RestoreUnavailable:
            raise
        except Exception:
            if device == "cpu":
                raise
            engine = SidonEngine("cpu")
        _engine_cache[engine.device] = engine
        return engine


def _format_clock(seconds: float) -> str:
    total = max(0, int(round(seconds)))
    return f"{total // 60}:{total % 60:02d}"


def _chunk_bounds(total: int, chunk_samples: int) -> list[tuple[int, int]]:
    starts = list(range(0, total, chunk_samples))
    if len(starts) > 1 and total - starts[-1] < MIN_CHUNK_SAMPLES:
        starts.pop()  # fold a stub tail into its predecessor
    return [(start, min(total, start + chunk_samples) if index + 1 < len(starts) else total)
            for index, start in enumerate(starts)]


def restore_waveform(
    wav: np.ndarray,
    sr: int,
    *,
    chunk_sec: float = DEFAULT_CHUNK_SECONDS,
    progress: Callable[[float, str], None] | None = None,
) -> tuple[np.ndarray, int]:
    """Restore a full mono waveform with Sidon; returns (audio, 48000).

    Input may be at any sample rate; it is high-passed and resampled to 16 kHz
    here. The output length is pinned to round(n / sr * 48000) so the restored
    audio lines up with the source timeline sample for sample.

    Level is preserved rather than normalised: the model wants a 0.9-peak input,
    so the waveform is scaled into that range and the output is scaled back to
    the input's original peak. A quiet recording stays quiet.
    """
    import torch
    import torchaudio

    engine = load_engine()

    wav = np.asarray(wav, dtype=np.float32).reshape(-1)
    n = wav.size
    if n == 0:
        raise ValueError("Restore input is empty.")

    target_out = round(n / sr * SIDON_OUTPUT_RATE)
    total_label = _format_clock(n / sr)

    signal = torch.from_numpy(wav).view(1, -1)
    peak = float(signal.abs().max())
    if peak > 0:
        signal = 0.9 * (signal / peak)
    signal = torchaudio.functional.highpass_biquad(signal, sr, 50)
    if sr != SIDON_INPUT_RATE:
        signal = torchaudio.functional.resample(signal, sr, SIDON_INPUT_RATE)
    signal = torch.nn.functional.pad(signal, (0, TAIL_PAD_SAMPLES)).view(-1)

    chunk_samples = max(MIN_CHUNK_SAMPLES, int(chunk_sec * SIDON_INPUT_RATE))
    bounds = _chunk_bounds(signal.numel(), chunk_samples)

    pieces: list[np.ndarray] = []
    feature_cache = None
    with torch.inference_mode():
        for index, (start, end) in enumerate(bounds):
            try:
                audio, feature_cache = engine.restore_chunk(signal[start:end], feature_cache)
            except RestoreUnavailable:
                raise
            except Exception:
                if engine.device == "cpu":
                    raise
                engine = load_engine("cpu")
                feature_cache = None  # the cached frame lives on the dead device
                audio, feature_cache = engine.restore_chunk(signal[start:end], feature_cache)
            pieces.append(audio.float().cpu().numpy())
            if progress:
                done = min(n / sr, end / SIDON_INPUT_RATE)
                progress(
                    (index + 1) / len(bounds),
                    f"Restoring {_format_clock(done)} / {total_label}",
                )

    restored = np.concatenate(pieces) if len(pieces) > 1 else pieces[0]
    if restored.size < target_out:  # short only if the model returned less than promised
        restored = np.pad(restored, (0, target_out - restored.size))
    restored = restored[:target_out]

    out_peak = float(np.max(np.abs(restored))) if restored.size else 0.0
    if out_peak > 0 and peak > 0:
        restored = restored * (min(peak, 0.99) / out_peak)
    return restored.astype(np.float32), SIDON_OUTPUT_RATE
