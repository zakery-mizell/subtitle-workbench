"""TTS job: clone a reference voice with Qwen3-TTS and synthesize text.

Mirrors the restore job shape (token, artifact dir, staged ProgressReporter) so
the API endpoints and frontend polling work the same way.
"""
from __future__ import annotations

import subprocess
import uuid
from pathlib import Path

import numpy as np

from ..config import settings
from ..jobs import ProgressReporter
from ..mastering.audio_io import MasterAudio, encode_master
from ..schemas import WarningItem
from .engine import TtsUnavailable, load_engine, synthesize_speech
from .schemas import TtsParams, TtsResult

TTS_REFERENCE_RATE = 24000  # Qwen resamples internally; feed it 24 kHz mono
MAX_REFERENCE_SECONDS = 30.0

# Our language names mapped to whisper ISO codes for reference transcription.
_WHISPER_LANG: dict[str, str | None] = {
    "Auto": None,
    "English": "en",
    "Chinese": "zh",
    "German": "de",
    "Italian": "it",
    "Portuguese": "pt",
    "Spanish": "es",
    "Japanese": "ja",
    "Korean": "ko",
    "French": "fr",
    "Russian": "ru",
}


def tts_output_dir() -> Path:
    return Path(settings.mastering_output_dir).parent / "tts"


def find_tts_artifact(token: str) -> Path | None:
    output_dir = tts_output_dir()
    if not output_dir.is_dir() or not token.replace("t_", "").isalnum():
        return None
    matches = sorted(output_dir.glob(f"{token}__*"))
    return matches[0] if matches else None


def _output_paths(source_filename: str, fmt: str) -> tuple[str, Path]:
    token = f"t_{uuid.uuid4().hex[:12]}"
    stem = Path(source_filename or "voice").stem or "voice"
    output_dir = tts_output_dir()
    output_dir.mkdir(parents=True, exist_ok=True)
    return token, output_dir / f"{token}__{stem}.qwen3tts.{fmt}"


def decode_reference(source_path: str) -> tuple[np.ndarray, bool]:
    """Decode the reference clip to mono float32 24 kHz, trimmed to 30 s.

    Returns (audio, truncated); the caller emits the truncation warning since
    trimming happens here but the warnings list lives in the job driver.
    """
    command = [
        "ffmpeg", "-v", "error", "-i", source_path, "-vn", "-ac", "1",
        "-ar", str(TTS_REFERENCE_RATE), "-f", "f32le", "-acodec", "pcm_f32le", "pipe:1",
    ]
    try:
        completed = subprocess.run(command, check=True, capture_output=True)
    except FileNotFoundError as exc:
        raise RuntimeError("FFmpeg is required for voice cloning.") from exc
    except subprocess.CalledProcessError as exc:
        detail = exc.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"Could not decode the reference audio. Details: {detail}") from exc
    audio = np.frombuffer(completed.stdout, dtype=np.float32)
    if audio.size == 0:
        raise RuntimeError("No decodable audio samples were found in the reference clip.")
    max_samples = int(MAX_REFERENCE_SECONDS * TTS_REFERENCE_RATE)
    truncated = audio.size > max_samples
    if truncated:
        audio = audio[:max_samples]
    return audio.copy(), truncated


def _resolve_ref_text(
    source_path: str,
    params: TtsParams,
    warnings: list[WarningItem],
) -> tuple[str | None, str]:
    """Return (ref_text, clone_mode): ICL "transcript" when a transcript exists.

    A manual transcript wins; otherwise auto-transcribe the clip when enabled.
    Any failure (exception or empty text) falls back to x-vector-only mode.
    """
    manual = (params.ref_text or "").strip()
    if manual:
        return manual, "transcript"
    if not params.auto_ref_text:
        return None, "voice-signature"

    try:
        from ..whisperx_transcription import transcribe_with_whisperx

        result, _, _ = transcribe_with_whisperx(
            source_path,
            model_name=params.whisper_model,
            requested_language=_WHISPER_LANG.get(params.language),
            hotwords=None,
        )
        text = " ".join(
            str(segment.get("text") or "").strip()
            for segment in (result.get("segments") or [])
        ).strip()
        if not text:
            raise ValueError("The reference transcript came back empty.")
        return text, "transcript"
    except Exception:
        warnings.append(
            WarningItem(
                code="tts_ref_transcript_failed",
                message=(
                    "The reference clip could not be transcribed automatically, so the "
                    "voice was cloned from its speaker signature alone, which reduces "
                    "fidelity. Type the reference transcript manually for the best result."
                ),
            )
        )
        return None, "voice-signature"


def run_tts(
    source_path: str,
    source_filename: str,
    params: TtsParams,
    reporter: ProgressReporter,
) -> TtsResult:
    """Clone the reference voice and synthesize params.text end to end."""
    warnings: list[WarningItem] = []

    reporter.stage("decode", 0.0, 0.05, "Decoding reference audio")
    ref_wav, truncated = decode_reference(source_path)
    if truncated:
        warnings.append(
            WarningItem(
                code="tts_ref_truncated",
                message=f"The reference clip was trimmed to the first {int(MAX_REFERENCE_SECONDS)} seconds.",
            )
        )

    reporter.stage("ref_transcript", 0.05, 0.18, "Preparing the reference transcript")
    ref_text, clone_mode = _resolve_ref_text(source_path, params, warnings)

    reporter.stage("load_model", 0.18, 0.30, f"Loading Qwen3-TTS {params.model_size} model")
    try:
        engine = load_engine(params.model_size)
    except TtsUnavailable as exc:
        raise RuntimeError(str(exc)) from exc

    reporter.stage("synthesize", 0.30, 0.94, "Synthesizing speech")
    audio, out_sr = synthesize_speech(
        params.text,
        language=params.language,
        ref_wav=ref_wav,
        ref_sr=TTS_REFERENCE_RATE,
        ref_text=ref_text,
        x_vector_only=clone_mode == "voice-signature",
        model_size=params.model_size,
        progress=reporter.tick,
    )

    reporter.stage("encode", 0.94, 1.0, f"Encoding {params.output_format}")
    token, output_path = _output_paths(source_filename, params.output_format)
    encode_master(
        MasterAudio(samples=audio[None, :], sample_rate=out_sr),
        str(output_path),
        params.output_format,
    )

    return TtsResult(
        token=token,
        filename=output_path.name.split("__", 1)[1],
        output_format=params.output_format,
        sample_rate=out_sr,
        duration_sec=round(audio.size / out_sr, 3),
        device_used=engine.device,
        model_size=params.model_size,
        clone_mode=clone_mode,
        ref_text_used=ref_text,
        warnings=warnings,
    )
