"""Seed-VC (V2) zero-shot voice-conversion engine (vendored in vendor/seed-vc).

Seed-VC re-renders a source performance in the timbre of a reference clip: the
words, timing, and prosody come from the source, the voice from the reference.
This uses the V2 hubert-bsqvae-small model ("best in suppressing source speaker
traits"), which is speech-only -- there is no singing/f0 mode. All audio is mono
float32; output is 22.05 kHz. The wrapper loads the source and reference from
file PATHS and resamples internally; the reference is silently clipped to 25 s
and long sources are chunked internally (30 s windows, 5 s overlap, cosine
crossfade), so no external chunking is needed.

Device policy: like the compute-bound TTS transformer (and unlike the tiny
Diamond restore model), the diffusion + AR stack pays off on a GPU, so the auto
policy DOES pick MPS on Apple Silicon: CONVERSION_DEVICE override, else cuda,
else mps, else cpu. dtype is float16 on cuda but float32 on mps/cpu (this repo's
mps-fp32 rule; the CFM diffusion is forced fp32 internally regardless).

Vendored, not pip-installable: the tree lives in vendor/seed-vc and must be on
sys.path before its modules import. Two upstream quirks are handled here:
`hf_utils.load_custom_model_from_hf` hard-codes a CWD-relative cache_dir that
ignores HF_HOME, so it is monkeypatched to a plain hf_hub_download; and the V1
`inference.py` module is NEVER imported (it pins HF_HUB_CACHE to ./checkpoints at
import time). Streaming mp3-encodes each chunk via pydub, so ffmpeg must be on
PATH (already an app-wide requirement).
"""
from __future__ import annotations

import math
import os
import sys
import threading
import time
from pathlib import Path
from typing import Callable

import numpy as np

from ..config import settings

SEEDVC_SR = 22050
MAX_REFERENCE_SECONDS = 25.0

ROOT_DIR = Path(__file__).resolve().parents[3]
VENDOR_DIR = ROOT_DIR / "vendor" / "seed-vc"

_lock = threading.Lock()
_engine_cache: dict[str, "ConversionEngine"] = {}


class ConversionUnavailable(RuntimeError):
    """Raised when Seed-VC cannot run at all (missing vendor code/deps/checkpoints)."""


def _resolve_device() -> str:
    """cuda, else mps, else cpu (mps IS auto-picked here; see module docstring)."""
    configured = settings.conversion_device
    if configured:
        return configured
    try:
        import torch
    except ImportError:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _dtype_for(device: str):
    import torch

    # fp16 speeds up cuda; mps/cpu stay fp32 (this repo's mps-fp32 rule).
    return torch.float16 if device == "cuda" else torch.float32


class ConversionEngine:
    """Loaded Seed-VC V2 wrapper bound to one torch device."""

    def __init__(self, device: str) -> None:
        # Hard-set, not setdefault: transcribe_with_whisperx points HF_HOME at
        # the whisper cache process-wide, so after any transcription a setdefault
        # is a no-op and the wrapper's ambient-cache downloads (hubert, bigvgan,
        # the whisper-small tokenizer) would silently re-download into
        # models/whisper.
        os.environ["HF_HOME"] = str(settings.model_cache_dir)
        if not VENDOR_DIR.is_dir():
            raise ConversionUnavailable(
                "The Seed-VC model code is missing. Run scripts/install-convert.sh "
                "(or .ps1) to clone it into vendor/seed-vc."
            )

        import torch
        import yaml

        vendor_path = str(VENDOR_DIR)
        if vendor_path not in sys.path:
            sys.path.insert(0, vendor_path)
        try:
            import hydra
            from omegaconf import DictConfig
        except ImportError as exc:
            raise ConversionUnavailable(
                "Seed-VC python dependencies are missing (hydra-core/omegaconf). "
                "Install them with scripts/install-convert.sh (or .ps1)."
            ) from exc

        # hf_utils.load_custom_model_from_hf hard-codes cache_dir="./checkpoints"
        # (CWD-relative, ignores HF_HOME). Replace it with a plain download so the
        # Plachta/ASTRAL/campplus checkpoints land in HF_HOME like everything else.
        import hf_utils
        from huggingface_hub import hf_hub_download

        def _load_from_hf(repo_id, model_filename="pytorch_model.bin", config_filename=None):
            path = hf_hub_download(repo_id=repo_id, filename=model_filename)
            if config_filename is None:
                return path
            return path, hf_hub_download(repo_id=repo_id, filename=config_filename)

        hf_utils.load_custom_model_from_hf = _load_from_hf

        dtype = _dtype_for(device)
        # instantiate downloads hubert/bigvgan/whisper-tokenizer and load_checkpoints
        # fetches the Seed-VC checkpoints; wrap those so a missing/failed download
        # surfaces as ConversionUnavailable.
        try:
            config_path = VENDOR_DIR / "configs" / "v2" / "vc_wrapper.yaml"
            cfg = DictConfig(yaml.safe_load(open(str(config_path), "r")))
            wrapper = hydra.utils.instantiate(cfg)
            wrapper.load_checkpoints(ar_checkpoint_path=None, cfm_checkpoint_path=None)
        except Exception as exc:  # download/instantiate failures
            raise ConversionUnavailable(
                f"The Seed-VC checkpoints could not be downloaded. Details: {exc}"
            ) from exc

        # Left unwrapped so a device-specific move/cache-setup failure surfaces as
        # a plain exception, which load_engine catches to fall back to CPU.
        torch_device = torch.device(device)
        wrapper.to(torch_device)
        wrapper.eval()
        wrapper.setup_ar_caches(max_batch_size=1, max_seq_len=4096, dtype=dtype, device=torch_device)

        self._torch = torch
        self.device = device
        self.torch_device = torch_device
        self.dtype = dtype
        self.wrapper = wrapper


def load_engine(device_preference: str | None = None) -> ConversionEngine:
    """Load (or reuse) the Seed-VC engine, falling back to CPU if the GPU fails."""
    device = device_preference or _resolve_device()
    with _lock:
        cached = _engine_cache.get(device) or _engine_cache.get("cpu")
        if cached is not None:
            return cached
        try:
            engine = ConversionEngine(device)
        except ConversionUnavailable:
            raise
        except Exception:
            if device == "cpu":
                raise
            engine = ConversionEngine("cpu")
        _engine_cache[engine.device] = engine
        return engine


def expected_chunks(source_duration_sec: float) -> int:
    """Estimate the number of streamed chunks for progress reporting.

    The wrapper chunks the source into 30 s windows with 5 s overlap (25 s of new
    audio per window), so this bounds the yield count closely enough to drive a
    monotonic progress bar without knowing it ahead of time.
    """
    return max(1, math.ceil(source_duration_sec / 25.0))


# A chunk takes on the order of a minute on Apple Silicon (several with
# convert_style), so ticking only at chunk boundaries reads as a hang. Until the
# first chunk lands there is no measurement, so start from this guess and refine
# with the running mean of measured chunk times.
INITIAL_CHUNK_SECONDS = 90.0
_TICK_INTERVAL_SECONDS = 2.0


def format_eta(seconds: float) -> str:
    """~ETA as a compact human string: "40 s", "12 min", "1 h 05 min"."""
    seconds = max(seconds, 0.0)
    if seconds < 60:
        return f"{int(round(seconds))} s"
    minutes = seconds / 60
    if minutes < 60:
        return f"{int(round(minutes))} min"
    hours = int(minutes // 60)
    return f"{hours} h {int(round(minutes - hours * 60)):02d} min"


class _ChunkTicker:
    """Publishes smooth intra-chunk progress from wall-clock time.

    The streaming generator only reports at chunk boundaries; between them a
    daemon thread interpolates elapsed time against the per-chunk estimate (capped
    below the boundary so real completions stay ahead) and stamps every message
    with the chunk counter and a remaining-time estimate. Emitted fractions are
    clamped monotonic because a growing estimate could otherwise step backwards.
    """

    def __init__(self, expected: int, progress: Callable[[float, str], None]) -> None:
        self._expected = expected
        self._progress = progress
        self._durations: list[float] = []
        self._chunk_started = time.monotonic()
        self._done = 0
        self._last_fraction = 0.0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def __enter__(self) -> "_ChunkTicker":
        self._emit(0.0)
        self._thread.start()
        return self

    def __exit__(self, *_exc) -> None:
        self._stop.set()
        self._thread.join()

    def chunk_done(self) -> None:
        now = time.monotonic()
        self._durations.append(now - self._chunk_started)
        self._chunk_started = now
        self._done += 1
        self._emit(0.0)

    def _estimate(self) -> float:
        if not self._durations:
            return INITIAL_CHUNK_SECONDS
        return sum(self._durations) / len(self._durations)

    def _emit(self, within_chunk: float) -> None:
        done = min(self._done + within_chunk, float(self._expected))
        fraction = max(min(done / self._expected, 0.97), self._last_fraction)
        self._last_fraction = fraction
        remaining = max(self._expected - done, 0.0) * self._estimate()
        current = min(self._done + 1, self._expected)
        self._progress(
            fraction,
            f"Converting chunk {current}/{self._expected} (~{format_eta(remaining)} left)",
        )

    def _run(self) -> None:
        while not self._stop.wait(_TICK_INTERVAL_SECONDS):
            elapsed = time.monotonic() - self._chunk_started
            self._emit(min(elapsed / self._estimate(), 0.95))


def convert_voice(
    source_path: str,
    ref_path: str,
    *,
    diffusion_steps: int,
    length_adjust: float,
    intelligibility_cfg: float,
    similarity_cfg: float,
    convert_style: bool,
    source_duration_sec: float,
    top_p: float = 0.9,
    temperature: float = 1.0,
    repetition_penalty: float = 1.0,
    progress: Callable[[float, str], None] | None = None,
) -> tuple[np.ndarray, int]:
    """Convert the source performance into the reference timbre; returns (audio, sr).

    Seeds both RNGs for reproducibility, then drives the streaming generator to
    completion. stream_output=True is MANDATORY -- with False the generator
    yields nothing. Each yield is (mp3_bytes, full_audio); full_audio is None
    until the final chunk, then (sr, np.ndarray) with sr == 22050.
    """
    import torch

    torch.manual_seed(42)
    np.random.seed(42)

    engine = load_engine()

    generator = engine.wrapper.convert_voice_with_streaming(
        source_audio_path=source_path,
        target_audio_path=ref_path,
        diffusion_steps=diffusion_steps,
        length_adjust=length_adjust,
        intelligebility_cfg_rate=intelligibility_cfg,  # upstream misspelling; keep here only
        similarity_cfg_rate=similarity_cfg,
        top_p=top_p,
        temperature=temperature,
        repetition_penalty=repetition_penalty,
        convert_style=convert_style,
        anonymization_only=False,
        device=engine.torch_device,
        dtype=engine.dtype,
        stream_output=True,
    )

    expected = expected_chunks(source_duration_sec)
    out_sr = SEEDVC_SR
    final_audio: np.ndarray | None = None
    with _ChunkTicker(expected, progress or (lambda _fraction, _message: None)) as ticker:
        for _mp3_bytes, full_audio in generator:
            ticker.chunk_done()
            if full_audio is not None:
                out_sr, final_audio = full_audio

    if final_audio is None:
        raise ConversionUnavailable("Seed-VC produced no audio for this pair of clips.")
    return np.asarray(final_audio, dtype=np.float32).reshape(-1), int(out_sr)
