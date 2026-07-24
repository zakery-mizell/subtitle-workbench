"""F5-TTS speech-edit engine (SWivid/F5-TTS, F5TTS_v1_Base).

F5-TTS is a flow-matching text-to-speech model that operates in the mel domain.
It powers two modes here. In GENERATE mode it does zero-shot voice cloning: give
it a short reference clip plus that clip's transcript and it reads arbitrary text
in the reference voice. In EDIT mode it does mel-domain infilling: the surrounding
audio's mel frames are kept fixed and only the frames under an edit mask are
regenerated, conditioned on the kept context and a corrected transcript, so a
patched span inherits the surrounding voice and prosody. Both modes emit 24 kHz
mono float32.

Device policy: this is a compute-bound transformer, so GPU acceleration pays off
and the auto policy DOES pick MPS on Apple Silicon: SPEECHEDIT_DEVICE override,
else cuda, else mps, else cpu. dtype is fp16 only on cuda cc>=7; everywhere else
F5-TTS enforces fp32 itself (MPS-safe by design), so we never set dtype.

HF cache: the checkpoint (F5TTS_v1_Base/model_1250000.safetensors) and the vocos
vocoder resolve through cached_path / hf_hub_download. F5TTS honors the explicit
hf_cache_dir we pass, but cached_path falls back to HF_HOME on some code paths, so
we ALSO hard-set HF_HOME (assignment, not setdefault) for the same reason the
restore and separation engines do: transcribe_with_whisperx force-points HF_HOME
at the whisper cache process-wide and a job transcribes BEFORE loading this engine,
so a setdefault would be a no-op and downloads would land in models/whisper.
"""
from __future__ import annotations

import os
import threading
from pathlib import Path

import numpy as np

from ..config import settings

# Upstream mel constants (f5_tts.infer.utils_infer.hop_length / target_sample_rate
# and the 100-channel mel). Duplicated here so the pure frame math is importable
# without pulling in torch or f5_tts; the model methods assert they still match.
HOP_LENGTH = 256
TARGET_SAMPLE_RATE = 24000
N_MEL_CHANNELS = 100

# speech_edit.py normalizes quiet audio up to this RMS before infilling and
# un-scales the output by the same factor.
TARGET_RMS = 0.1

_lock = threading.Lock()
_engine_cache: dict[str, "SpeechEditEngine"] = {}


class SpeechEditUnavailable(RuntimeError):
    """Raised when F5-TTS cannot run at all (missing package/checkpoint)."""


def _resolve_device() -> str:
    """cuda, else mps, else cpu (mps IS auto-picked here; see module docstring)."""
    configured = settings.speechedit_device
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


def seconds_to_frames(seconds: float) -> int:
    """Mel-frame count for a duration (round(sec * sr / hop)); matches speech_edit.py."""
    return int(round(max(0.0, seconds) * TARGET_SAMPLE_RATE / HOP_LENGTH))


def rms_normalize_factor(audio: np.ndarray, target_rms: float = TARGET_RMS) -> float:
    """Scale factor that lifts quiet audio up to target_rms (never attenuates).

    Mirrors speech_edit.py: measure RMS, and only if it is below target return the
    boost factor (the caller multiplies input by it and divides the output back).
    Silent audio (rms 0) and already-loud audio return 1.0.
    """
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    if audio.size == 0:
        return 1.0
    rms = float(np.sqrt(np.mean(np.square(audio))))
    if rms <= 0.0 or rms >= target_rms:
        return 1.0
    return target_rms / rms


def build_edit_arrays(
    n_frames: int,
    edits: list[tuple[int, int, int]],
) -> tuple[list[tuple], list[tuple[bool, int]]]:
    """Plan the infill mel and its edit mask from per-window edits (frame units).

    `edits` is sorted `(start_frame, orig_end_frame, dur_frames)` triples where
    orig_end_frame indexes the ORIGINAL mel (the kept audio resumes there) and
    dur_frames is how many ZERO frames to insert for the regenerated span (it may
    differ from the original span when a fixed duration was requested).

    Returns `(plan, mask_runs)`:
      - plan: ordered `("keep", a, b)` (copy original mel frames [a, b)) and
        `("zero", count)` (insert `count` zero frames) segments.
      - mask_runs: matching run-length `(is_keep, length)` pairs; True marks frames
        the model must preserve, False the frames it regenerates.
    Pure integer arithmetic so tests cover it without a model.
    """
    plan: list[tuple] = []
    mask_runs: list[tuple[bool, int]] = []
    offset = 0
    for start_f, orig_end_f, dur_f in edits:
        keep_len = start_f - offset
        if keep_len > 0:
            plan.append(("keep", offset, start_f))
            mask_runs.append((True, keep_len))
        if dur_f > 0:
            plan.append(("zero", dur_f))
            mask_runs.append((False, dur_f))
        offset = orig_end_f
    if offset < n_frames:
        plan.append(("keep", offset, n_frames))
        mask_runs.append((True, n_frames - offset))
    return plan, mask_runs


class SpeechEditEngine:
    """One loaded F5-TTS instance bound to a torch device (both modes share it)."""

    def __init__(self, device: str) -> None:
        # Hard-set, not setdefault -- see module docstring for the whisper HF_HOME
        # collision that makes setdefault a silent no-op.
        os.environ["HF_HOME"] = str(settings.model_cache_dir)
        try:
            from f5_tts.api import F5TTS
        except ImportError as exc:
            raise SpeechEditUnavailable(
                "The 'f5-tts' speech-edit package is not installed, so patching was "
                "skipped. Install it with scripts/install-speechedit.sh (or .ps1)."
            ) from exc

        # Left unwrapped so a device-specific load failure surfaces as a plain
        # exception, which load_engine catches to fall back to CPU. F5TTS picks
        # fp16 only on cuda cc>=7 and fp32 elsewhere on its own, so we pass no dtype.
        try:
            self.model = F5TTS(
                model="F5TTS_v1_Base",
                device=device,
                hf_cache_dir=str(Path(settings.model_cache_dir) / "hub"),
            )
        except SpeechEditUnavailable:
            raise
        except ImportError:
            raise
        self.device = device

    def generate(
        self,
        ref_path: str,
        ref_text: str,
        gen_text: str,
        *,
        nfe_step: int,
        speed: float,
        seed: int | None,
    ) -> tuple[np.ndarray, int]:
        """Zero-shot clone: read `gen_text` in the reference voice; returns (wav, sr).

        ref_text MUST be non-empty -- an empty one triggers F5TTS's built-in
        whisper-large-v3-turbo download; the job supplies WhisperX text instead.
        """
        wav, sr, _spec = self.model.infer(
            ref_file=ref_path,
            ref_text=ref_text,
            gen_text=gen_text,
            target_rms=TARGET_RMS,
            cross_fade_duration=0.15,
            sway_sampling_coef=-1,
            cfg_strength=2,
            nfe_step=nfe_step,
            speed=speed,
            seed=seed,
        )
        return np.asarray(wav, dtype=np.float32).reshape(-1), int(sr)

    def edit_window(
        self,
        window_wav: np.ndarray,
        edits_local: list[tuple[float, float, float | None]],
        window_text: str,
        *,
        nfe_step: int,
        seed: int | None,
    ) -> np.ndarray:
        """Mel-infill the edit spans of one window; returns 24 kHz float32 audio.

        `edits_local` are `(start_s, end_s, fix_duration_s|None)` in window-local
        seconds. Mirrors infer/speech_edit.py: RMS-normalize, build a kept/zero mel
        with a matching edit mask, sample under `torch.inference_mode()`, vocode,
        then un-normalize.
        """
        import torch
        from f5_tts.infer.utils_infer import hop_length, target_sample_rate
        from f5_tts.model.utils import convert_char_to_pinyin

        # The pure frame math above hard-codes these; fail loud if upstream drifts.
        assert hop_length == HOP_LENGTH and target_sample_rate == TARGET_SAMPLE_RATE

        audio = np.asarray(window_wav, dtype=np.float32).reshape(-1)
        factor = rms_normalize_factor(audio, TARGET_RMS)

        model = self.model.ema_model
        vocoder = self.model.vocoder

        with torch.inference_mode():
            audio_t = torch.from_numpy(audio * factor).to(self.device).unsqueeze(0)
            # mel_spec -> (1, n_mel, T); build the cond along the frame axis in (1, T, n_mel).
            mel = model.mel_spec(audio_t)
            mel_bt = mel.permute(0, 2, 1)
            n_frames = mel_bt.shape[1]

            edits_frames: list[tuple[int, int, int]] = []
            for start_s, end_s, fix_dur in edits_local:
                dur_s = fix_dur if fix_dur is not None else (end_s - start_s)
                edits_frames.append(
                    (seconds_to_frames(start_s), seconds_to_frames(end_s), seconds_to_frames(dur_s))
                )
            plan, mask_runs = build_edit_arrays(n_frames, edits_frames)

            parts = []
            for seg in plan:
                if seg[0] == "keep":
                    parts.append(mel_bt[:, seg[1]:seg[2], :])
                else:
                    parts.append(
                        torch.zeros(1, seg[1], mel_bt.shape[-1], device=self.device, dtype=mel_bt.dtype)
                    )
            cond = torch.cat(parts, dim=1)

            mask_parts = [torch.full((length,), keep, dtype=torch.bool) for keep, length in mask_runs]
            edit_mask = torch.cat(mask_parts).unsqueeze(0).to(self.device)

            text_list = convert_char_to_pinyin([window_text])
            generated, _trajectory = model.sample(
                cond=cond,
                text=text_list,
                duration=cond.shape[1],
                steps=nfe_step,
                cfg_strength=2,
                sway_sampling_coef=-1.0,
                seed=seed,
                edit_mask=edit_mask,
            )
            gen_mel = generated.float().permute(0, 2, 1)
            wav = vocoder.decode(gen_mel)
            out = wav.squeeze().detach().cpu().numpy().astype(np.float32).reshape(-1)

        if factor != 1.0:
            out = out / factor
        return out


def load_engine(device_preference: str | None = None) -> SpeechEditEngine:
    """Load (or reuse) the F5-TTS engine, falling back to CPU if the GPU fails."""
    device = device_preference or _resolve_device()
    with _lock:
        cached = _engine_cache.get(device) or _engine_cache.get("cpu")
        if cached is not None:
            return cached
        try:
            engine = SpeechEditEngine(device)
        except SpeechEditUnavailable:
            raise
        except Exception:
            if device == "cpu":
                raise
            engine = SpeechEditEngine("cpu")
        _engine_cache[engine.device] = engine
        return engine
