"""Qwen3-TTS voice-cloning engine (Qwen/Qwen3-TTS-12Hz-*-Base).

Qwen3-TTS Base is an open-weights speech LM that clones a voice from a short
reference clip and reads arbitrary text in it. Two clone modes exist: ICL mode
conditions on the reference speech AND its transcript (higher fidelity, needs a
transcript), while x-vector-only mode clones from the speaker embedding alone
(no transcript, lower fidelity). All inference here is local; there is no cloud
API. Reference and output audio are mono float32; the model resamples to its own
rates internally.

Device policy: unlike the tiny autoregressive Diamond restore model, this
0.6B/1.7B transformer is compute-bound, so GPU acceleration pays off and the
auto policy DOES pick MPS on Apple Silicon: TTS_DEVICE override, else cuda if
available, else mps if available, else cpu. dtype is bfloat16 on cuda but
float32 on mps and cpu -- fp16/bf16 voice cloning is broken on MPS. Attention is
"flash_attention_2" only when the device is cuda AND flash_attn imports;
otherwise "sdpa". Importing qwen_tts prints a harmless flash-attn warning.
"""
from __future__ import annotations

import os
import re
import threading
from typing import Callable

import numpy as np

from ..config import settings

MODEL_REPOS = {
    "1.7b": "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
    "0.6b": "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
}

# Silence stitched between chunks of a multi-chunk synthesis.
CHUNK_GAP_SECONDS = 0.12

# Runaway-generation guard. A reference transcript that mismatches the
# reference audio can send ICL generation off the rails: instead of stopping
# after the sentence, the model rambles until its (huge) default token budget
# runs out -- measured >24 minutes for one sentence on MPS. The 12 Hz models
# emit 12 codec frames per second of audio, so cap new tokens per chunk at a
# generous multiple of the text length: ~3 frames/char covers even slow CJK
# speech, plus a flat allowance for leading/trailing silence.
MAX_NEW_TOKENS_PER_CHAR = 3
MAX_NEW_TOKENS_FLOOR = 160

# Sentence-boundary splitter that keeps the delimiter with its sentence; the
# trailing alternative captures a final run with no terminal punctuation.
_SENTENCE_RE = re.compile(r"[^.!?…。！？]*[.!?…。！？]+|[^.!?…。！？]+$")

_lock = threading.Lock()
_engine_cache: dict[tuple[str, str], "TtsEngine"] = {}


class TtsUnavailable(RuntimeError):
    """Raised when Qwen3-TTS cannot run at all (missing package/checkpoint)."""


def _resolve_device() -> str:
    """cuda, else mps, else cpu (mps IS auto-picked here; see module docstring)."""
    configured = settings.tts_device
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

    # bf16 speeds up cuda; mps/cpu must stay fp32 (fp16/bf16 clone is broken on mps).
    return torch.bfloat16 if device == "cuda" else torch.float32


def _attn_impl_for(device: str) -> str:
    if device == "cuda":
        try:
            import flash_attn  # noqa: F401
        except ImportError:
            return "sdpa"
        return "flash_attention_2"
    return "sdpa"


class TtsEngine:
    """Loaded Qwen3-TTS model bound to one torch device."""

    def __init__(self, model_size: str, device: str) -> None:
        os.environ.setdefault("HF_HOME", str(settings.model_cache_dir))
        if model_size not in MODEL_REPOS:
            raise TtsUnavailable(f"Unknown Qwen3-TTS model size: {model_size!r}.")
        try:
            from qwen_tts import Qwen3TTSModel
        except ImportError as exc:
            raise TtsUnavailable(
                "The 'qwen-tts' voice-cloning package is not installed, so voice "
                "cloning was skipped. Install it with scripts/install-tts.sh (or .ps1)."
            ) from exc

        repo_id = MODEL_REPOS[model_size]
        # Left unwrapped so a device-specific load/download failure surfaces as a
        # plain exception, which load_engine catches to fall back to CPU.
        self.model = Qwen3TTSModel.from_pretrained(
            repo_id,
            device_map=device,
            dtype=_dtype_for(device),
            attn_implementation=_attn_impl_for(device),
        )
        self.model_size = model_size
        self.device = device

    def clone_prompt(
        self,
        ref_wav: np.ndarray,
        ref_sr: int,
        ref_text: str | None,
        x_vector_only: bool,
    ):
        """Build the voice-clone prompt items once for a reference clip.

        ICL mode (x_vector_only=False) requires a non-empty ref_text; x-vector
        mode ignores it and clones from the speaker embedding alone.
        """
        wav = np.asarray(ref_wav, dtype=np.float32).reshape(-1)
        return self.model.create_voice_clone_prompt(
            ref_audio=(wav, ref_sr),
            ref_text=ref_text,
            x_vector_only_mode=x_vector_only,
        )

    def generate(self, text_chunk: str, language: str, prompt_items) -> tuple[np.ndarray, int]:
        """Synthesize one text chunk with a prepared clone prompt."""
        wavs, sr = self.model.generate_voice_clone(
            text=text_chunk,
            language=language,
            voice_clone_prompt=prompt_items,
            max_new_tokens=max_new_tokens_for(text_chunk),
        )
        return np.asarray(wavs[0], dtype=np.float32).reshape(-1), int(sr)


def load_engine(model_size: str, device_preference: str | None = None) -> TtsEngine:
    """Load (or reuse) a Qwen3-TTS engine, falling back to CPU if the GPU fails."""
    device = device_preference or _resolve_device()
    with _lock:
        cached = _engine_cache.get((model_size, device)) or _engine_cache.get((model_size, "cpu"))
        if cached is not None:
            return cached
        try:
            engine = TtsEngine(model_size, device)
        except TtsUnavailable:
            raise
        except Exception:
            if device == "cpu":
                raise
            engine = TtsEngine(model_size, "cpu")
        _engine_cache[(engine.model_size, engine.device)] = engine
        return engine


def max_new_tokens_for(text_chunk: str) -> int:
    """Codec-token budget for one chunk (see the runaway guard note above)."""
    return MAX_NEW_TOKENS_PER_CHAR * len(text_chunk) + MAX_NEW_TOKENS_FLOOR


def _hard_split(sentence: str, max_chars: int) -> list[str]:
    """Split a single over-long sentence greedily on whitespace."""
    chunks: list[str] = []
    current = ""
    for word in sentence.split():
        candidate = f"{current} {word}" if current else word
        if current and len(candidate) > max_chars:
            chunks.append(current)
            current = word
        else:
            current = candidate
    if current:
        chunks.append(current)
    return chunks


def split_text_into_chunks(text: str, max_chars: int = 300) -> list[str]:
    """Split text into <= max_chars chunks on sentence boundaries.

    Sentences are packed greedily up to max_chars; a single sentence longer than
    max_chars is hard-split on whitespace (an unbreakable word stays whole). Pure
    and unit-testable -- no model state is touched.
    """
    sentences = [s.strip() for s in _SENTENCE_RE.findall(text) if s.strip()]
    chunks: list[str] = []
    current = ""
    for sentence in sentences:
        if len(sentence) > max_chars:
            if current:
                chunks.append(current)
                current = ""
            chunks.extend(_hard_split(sentence, max_chars))
            continue
        candidate = f"{current} {sentence}" if current else sentence
        if current and len(candidate) > max_chars:
            chunks.append(current)
            current = sentence
        else:
            current = candidate
    if current:
        chunks.append(current)
    return chunks


def synthesize_speech(
    text: str,
    *,
    language: str,
    ref_wav: np.ndarray,
    ref_sr: int,
    ref_text: str | None,
    x_vector_only: bool,
    model_size: str,
    progress: Callable[[float, str], None] | None = None,
) -> tuple[np.ndarray, int]:
    """Clone the reference voice and synthesize `text`; returns (audio, sr).

    Mirrors restore_waveform: seed both RNGs for reproducibility, build the clone
    prompt once, render chunk by chunk, and normalize once at the end. A non-CPU
    generate failure reloads the engine on CPU once (rebuilding the prompt on the
    new device) and retries that chunk.
    """
    import torch

    torch.manual_seed(42)
    np.random.seed(42)

    engine = load_engine(model_size)

    chunks = split_text_into_chunks(text)
    if not chunks:
        raise ValueError("Synthesis text is empty.")

    prompt_items = engine.clone_prompt(ref_wav, ref_sr, ref_text, x_vector_only)

    total = len(chunks)
    rendered: list[np.ndarray] = []
    out_sr = 0
    for index, chunk in enumerate(chunks):
        try:
            wav, out_sr = engine.generate(chunk, language, prompt_items)
        except Exception:
            if engine.device == "cpu":
                raise
            engine = load_engine(model_size, "cpu")
            prompt_items = engine.clone_prompt(ref_wav, ref_sr, ref_text, x_vector_only)
            wav, out_sr = engine.generate(chunk, language, prompt_items)
        rendered.append(wav)
        if progress:
            progress((index + 1) / total, f"Synthesizing chunk {index + 1} / {total}")

    gap = np.zeros(int(CHUNK_GAP_SECONDS * out_sr), dtype=np.float32)
    pieces: list[np.ndarray] = []
    for index, wav in enumerate(rendered):
        if index:
            pieces.append(gap)
        pieces.append(wav)
    audio = np.concatenate(pieces).astype(np.float32)

    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if peak > 1.0:  # single global peak normalize; never boost quiet audio
        audio = audio * (0.99 / peak)
    return audio.astype(np.float32), out_sr
