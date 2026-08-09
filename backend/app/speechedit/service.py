"""Speech-edit job: patch garbled spans in place, or clone-and-speak with F5-TTS.

Mirrors the conversion/tts job shape (token, artifact dir, staged ProgressReporter)
so the API endpoints and frontend polling work the same way.

The EDIT-mode windowing layer is the novel part and lives here as PURE functions
(validation, window computation, word-replacement text building, splice math) so
tests exercise it without loading a model. F5-TTS handles ~<=30 s of context well,
so each edit span is patched inside a padded window cut from the full recording and
spliced back with short crossfades; everything outside a window stays bit-identical.
"""
from __future__ import annotations

import subprocess
import uuid
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

import numpy as np

from ..config import settings
from ..jobs import ProgressReporter
from ..mastering.audio_io import MasterAudio, encode_master
from ..schemas import WarningItem
from .engine import (
    TARGET_SAMPLE_RATE,
    SpeechEditUnavailable,
    load_engine,
)
from .schemas import PatchEdit, PatchRegion, SpeechEditParams, SpeechEditResult

SPEECHEDIT_RATE = TARGET_SAMPLE_RATE  # 24 kHz mono, both modes

# Windowing constants (see module docstring / the edit-mode design in the spec).
WINDOW_PAD_SECONDS = 4.0
MAX_WINDOW_SECONDS = 25.0
MAX_SPAN_SECONDS = 20.0
CROSSFADE_SECONDS = 0.15

# Reference clips are auto-clipped to <=12 s upstream, so pre-trim ourselves and
# transcribe the trimmed audio -- otherwise the WhisperX transcript would describe
# more audio than the model sees and derail cloning.
MAX_REFERENCE_SECONDS = 12.0

# --------------------------------------------------------------------------- #
# Artifact helpers                                                            #
# --------------------------------------------------------------------------- #
def speechedit_output_dir() -> Path:
    return Path(settings.mastering_output_dir).parent / "speechedit"


def find_speechedit_artifact(token: str) -> Path | None:
    output_dir = speechedit_output_dir()
    if not output_dir.is_dir() or not token.replace("se_", "").isalnum():
        return None
    matches = sorted(output_dir.glob(f"{token}__*"))
    return matches[0] if matches else None


def _output_paths(source_filename: str, fmt: str) -> tuple[str, Path]:
    token = f"se_{uuid.uuid4().hex[:12]}"
    stem = Path(source_filename or "audio").stem or "audio"
    output_dir = speechedit_output_dir()
    output_dir.mkdir(parents=True, exist_ok=True)
    return token, output_dir / f"{token}__{stem}.f5tts.{fmt}"


# --------------------------------------------------------------------------- #
# Pure windowing / mask / splice math (unit-tested without a model)           #
# --------------------------------------------------------------------------- #
def validate_edits(edits: list[PatchEdit], audio_duration: float) -> None:
    """Reject malformed edit lists with 422-style messages (raises ValueError).

    Requires: at least one edit; each end_s > start_s and new_text non-empty
    (already enforced by the schema, re-checked defensively); spans within the
    audio; spans <= MAX_SPAN_SECONDS; sorted by start and non-overlapping.
    """
    if not edits:
        raise ValueError("Add at least one span to patch.")
    ordered = sorted(edits, key=lambda e: e.start_s)
    prev_end = 0.0
    for edit in ordered:
        if edit.end_s <= edit.start_s:
            raise ValueError(f"Edit end ({edit.end_s}s) must be after its start ({edit.start_s}s).")
        if not (edit.new_text or "").strip() and not (edit.window_text or "").strip():
            raise ValueError("Every edit needs replacement text.")
        if edit.start_s < 0 or edit.end_s > audio_duration + 1e-3:
            raise ValueError(
                f"Edit {edit.start_s}-{edit.end_s}s falls outside the recording "
                f"(0-{round(audio_duration, 2)}s)."
            )
        if edit.end_s - edit.start_s > MAX_SPAN_SECONDS:
            raise ValueError(
                f"Edit {edit.start_s}-{edit.end_s}s is longer than the {int(MAX_SPAN_SECONDS)}s "
                "per-span limit; split it into smaller edits."
            )
        if edit.start_s < prev_end:
            raise ValueError("Edits overlap; give each span a distinct, non-overlapping range.")
        prev_end = edit.end_s


def compute_window(start_s: float, end_s: float, audio_duration: float) -> tuple[float, float]:
    """Padded, clamped window `[w0, w1]` around an edit span.

    Pads WINDOW_PAD_SECONDS each side, clamps to the audio bounds, and caps the
    total at MAX_WINDOW_SECONDS by shrinking the pad symmetrically. The span must
    already be <= MAX_SPAN_SECONDS (validate_edits enforces this).
    """
    w0 = max(0.0, start_s - WINDOW_PAD_SECONDS)
    w1 = min(audio_duration, end_s + WINDOW_PAD_SECONDS)
    if w1 - w0 > MAX_WINDOW_SECONDS:
        span_len = end_s - start_s
        pad = max(0.0, (MAX_WINDOW_SECONDS - span_len) / 2.0)
        w0 = max(0.0, start_s - pad)
        w1 = min(audio_duration, end_s + pad)
    return w0, w1


def extract_words(whisper_result: dict[str, Any]) -> list[dict[str, Any]]:
    """Flatten a WhisperX result to `{text, start, end}` words, in order.

    Uses aligned word timings when present, else falls back to segment granularity
    (one entry per segment spanning its bounds). Pure and unit-testable.
    """
    words: list[dict[str, Any]] = []
    for segment in whisper_result.get("segments") or []:
        seg_start = float(segment.get("start") or 0.0)
        seg_end = float(segment.get("end") or seg_start)
        seg_words = segment.get("words") or []
        if seg_words:
            for word in seg_words:
                text = str(word.get("word") or "").strip()
                if not text:
                    continue
                raw_start = word.get("start")
                raw_end = word.get("end")
                start = float(raw_start) if raw_start is not None else seg_start
                end = float(raw_end) if raw_end is not None else start
                words.append({"text": text, "start": start, "end": max(start, end)})
        else:
            text = str(segment.get("text") or "").strip()
            if text:
                words.append({"text": text, "start": seg_start, "end": max(seg_start, seg_end)})
    return words


def words_in_window(words: list[dict[str, Any]], w0: float, w1: float) -> list[dict[str, Any]]:
    """Words overlapping `[w0, w1]` (word.end > w0 and word.start < w1)."""
    return [w for w in words if w["end"] > w0 and w["start"] < w1]


def build_window_target_text(
    window_words: list[dict[str, Any]],
    span_start: float,
    span_end: float,
    new_text: str,
) -> str:
    """Window transcript with the edited span's words swapped for `new_text`.

    Words entirely before the span (end <= span_start) and entirely after it
    (start >= span_end) are kept verbatim; every word that overlaps the span --
    including ones straddling either boundary -- is dropped and replaced by the
    single `new_text` run. Joined with spaces. Pure and unit-testable.
    """
    before = [w["text"] for w in window_words if w["end"] <= span_start]
    after = [w["text"] for w in window_words if w["start"] >= span_end]
    parts = before + [new_text.strip()] + after
    return " ".join(part for part in parts if part).strip()


def _linear_crossfade(fade_out: np.ndarray, fade_in: np.ndarray) -> np.ndarray:
    """Equal-length linear crossfade: fade_out ramps down, fade_in ramps up."""
    n = len(fade_out)
    ramp = np.linspace(0.0, 1.0, n, endpoint=False, dtype=np.float32)
    return (fade_out * (1.0 - ramp) + fade_in * ramp).astype(np.float32)


def crossfade_splice(
    audio: np.ndarray,
    start_sample: int,
    end_sample: int,
    edited: np.ndarray,
    fade_samples: int,
) -> np.ndarray:
    """Replace `audio[start_sample:end_sample]` with `edited`, crossfading both seams.

    The crossfades sit INSIDE the window so `audio[:start_sample]` and
    `audio[end_sample:]` stay bit-identical (the headline guarantee). The window
    length becomes len(edited), so the output length shifts by
    len(edited) - (end_sample - start_sample). Pure numpy; unit-testable.
    """
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    edited = np.asarray(edited, dtype=np.float32).reshape(-1)
    fade = int(fade_samples)
    window_len = end_sample - start_sample
    # Degenerate guard: too little room to crossfade -> hard replace.
    if fade < 1 or window_len < 2 * fade or len(edited) < 2 * fade:
        return np.concatenate([audio[:start_sample], edited, audio[end_sample:]]).astype(np.float32)

    head = audio[:start_sample]
    tail = audio[end_sample:]
    left_cf = _linear_crossfade(audio[start_sample:start_sample + fade], edited[:fade])
    middle = edited[fade:len(edited) - fade]
    right_cf = _linear_crossfade(edited[len(edited) - fade:], audio[end_sample - fade:end_sample])
    return np.concatenate([head, left_cf, middle, right_cf, tail]).astype(np.float32)


def seconds_to_samples(seconds: float, sample_rate: int = SPEECHEDIT_RATE) -> int:
    return int(round(max(0.0, seconds) * sample_rate))


# --------------------------------------------------------------------------- #
# Decode / transcription helpers                                             #
# --------------------------------------------------------------------------- #
def decode_audio(source_path: str) -> np.ndarray:
    """Decode any audio/video upload to mono float32 24 kHz (no trim)."""
    command = [
        "ffmpeg", "-v", "error", "-i", source_path, "-vn", "-ac", "1",
        "-ar", str(SPEECHEDIT_RATE), "-f", "f32le", "-acodec", "pcm_f32le", "pipe:1",
    ]
    try:
        completed = subprocess.run(command, check=True, capture_output=True)
    except FileNotFoundError as exc:
        raise RuntimeError("FFmpeg is required for speech editing.") from exc
    except subprocess.CalledProcessError as exc:
        detail = exc.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"Could not decode the audio. Details: {detail}") from exc
    audio = np.frombuffer(completed.stdout, dtype=np.float32)
    if audio.size == 0:
        raise RuntimeError("No decodable audio samples were found in this file.")
    return audio.copy()


def _make_temp_wav() -> str:
    Path(settings.temp_upload_dir).mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(delete=False, suffix=".wav", dir=settings.temp_upload_dir) as handle:
        return handle.name


def _transcribe_wav(audio: np.ndarray, params: SpeechEditParams) -> dict[str, Any]:
    """Write `audio` to a temp wav and run the app's WhisperX with word timings."""
    import soundfile as sf

    from ..whisperx_transcription import transcribe_with_whisperx

    ref_path = _make_temp_wav()
    try:
        sf.write(ref_path, audio, SPEECHEDIT_RATE)
        result, _, _ = transcribe_with_whisperx(
            ref_path,
            model_name=params.whisper_model,
            requested_language="en",
            hotwords=None,
        )
        return result
    finally:
        try:
            Path(ref_path).unlink()
        except FileNotFoundError:
            pass


# --------------------------------------------------------------------------- #
# Job driver                                                                  #
# --------------------------------------------------------------------------- #
def run_speechedit(
    source_path: str,
    source_filename: str,
    params: SpeechEditParams,
    reporter: ProgressReporter,
) -> SpeechEditResult:
    if params.mode == "generate":
        return _run_generate(source_path, source_filename, params, reporter)
    return _run_edit(source_path, source_filename, params, reporter)


def _run_edit(
    source_path: str,
    source_filename: str,
    params: SpeechEditParams,
    reporter: ProgressReporter,
) -> SpeechEditResult:
    """Patch each edit span in place; audio outside the windows stays bit-identical."""
    warnings: list[WarningItem] = []

    reporter.stage("decode", 0.0, 0.05, "Decoding recording")
    audio = decode_audio(source_path)
    duration = audio.size / SPEECHEDIT_RATE
    validate_edits(params.edits, duration)

    reporter.stage("transcribe", 0.05, 0.25, "Transcribing recording")
    words: list[dict[str, Any]] | None
    try:
        words = extract_words(_transcribe_wav(audio, params))
    except Exception:
        words = None  # only fatal for edits lacking a manual window_text override

    reporter.stage("load_model", 0.25, 0.40, "Loading F5-TTS model")
    try:
        engine = load_engine()
    except SpeechEditUnavailable as exc:
        raise RuntimeError(str(exc)) from exc

    fade_samples = seconds_to_samples(CROSSFADE_SECONDS)
    ordered = sorted(params.edits, key=lambda e: e.start_s)

    # Precompute each edit's window + target text (forward order, for the result
    # metadata), then splice back-to-front so earlier splices do not shift the
    # sample bounds of later (higher-timestamp) edits.
    regions: list[PatchRegion] = []
    plans: list[tuple[PatchEdit, float, float, str]] = []
    for edit in ordered:
        w0, w1 = compute_window(edit.start_s, edit.end_s, duration)
        override = (edit.window_text or "").strip()
        if override:
            window_text = override
        elif words is not None:
            window_text = build_window_target_text(
                words_in_window(words, w0, w1), edit.start_s, edit.end_s, edit.new_text
            )
        else:
            raise RuntimeError(
                f"The recording could not be transcribed, so the {round(edit.start_s, 2)}-"
                f"{round(edit.end_s, 2)}s edit has no window transcript. Supply a window_text "
                "override for it and try again."
            )
        plans.append((edit, w0, w1, window_text))
        regions.append(
            PatchRegion(
                start_s=round(edit.start_s, 3),
                end_s=round(edit.end_s, 3),
                window_start_s=round(w0, 3),
                window_end_s=round(w1, 3),
                text_used=window_text,
            )
        )

    reporter.stage("patch", 0.40, 0.94, "Patching audio")
    total = len(plans)
    for done, (edit, w0, w1, window_text) in enumerate(reversed(plans)):
        s0 = seconds_to_samples(w0)
        s1 = seconds_to_samples(w1)
        window_wav = audio[s0:s1]
        local_start = edit.start_s - w0
        local_end = edit.end_s - w0
        edited = engine.edit_window(
            window_wav,
            [(local_start, local_end, edit.fix_duration_s)],
            window_text,
            nfe_step=params.nfe_step,
            seed=params.seed,
        )
        audio = crossfade_splice(audio, s0, s1, edited, fade_samples)
        reporter.tick((done + 1) / total, f"Patched span {done + 1} / {total}")

    reporter.stage("encode", 0.94, 1.0, f"Encoding {params.output_format}")
    token, output_path = _output_paths(source_filename, params.output_format)
    encode_master(
        MasterAudio(samples=audio[None, :], sample_rate=SPEECHEDIT_RATE),
        str(output_path),
        params.output_format,
    )

    return SpeechEditResult(
        token=token,
        filename=output_path.name.split("__", 1)[1],
        output_format=params.output_format,
        sample_rate=SPEECHEDIT_RATE,
        duration_sec=round(audio.size / SPEECHEDIT_RATE, 3),
        device_used=engine.device,
        mode="edit",
        regions=regions,
        warnings=warnings,
    )


def _run_generate(
    source_path: str,
    source_filename: str,
    params: SpeechEditParams,
    reporter: ProgressReporter,
) -> SpeechEditResult:
    """Zero-shot voice clone: read gen_text in the uploaded reference voice."""
    warnings: list[WarningItem] = []
    if not (params.gen_text or "").strip():
        raise RuntimeError("Enter the text you want the cloned voice to speak.")

    reporter.stage("decode", 0.0, 0.05, "Decoding reference audio")
    ref_wav = decode_audio(source_path)
    max_samples = int(MAX_REFERENCE_SECONDS * SPEECHEDIT_RATE)
    if ref_wav.size > max_samples:
        ref_wav = ref_wav[:max_samples]
        warnings.append(
            WarningItem(
                code="patch_ref_truncated",
                message=f"The reference clip was trimmed to the first {int(MAX_REFERENCE_SECONDS)} seconds.",
            )
        )

    reporter.stage("ref_transcript", 0.05, 0.18, "Preparing the reference transcript")
    # F5-TTS REQUIRES a reference transcript -- there is no x-vector fallback like
    # Qwen had -- so a manual transcript wins and, when absent, a failed or empty
    # auto-transcription fails the job with a clear message.
    ref_text = (params.ref_text or "").strip()
    if not ref_text:
        if not params.auto_ref_text:
            raise RuntimeError(
                "F5-TTS needs the reference clip's transcript. Type what the clip says, "
                "or enable auto-transcribe."
            )
        try:
            ref_text = " ".join(
                w["text"] for w in extract_words(_transcribe_wav(ref_wav, params))
            ).strip()
        except Exception as exc:
            raise RuntimeError(
                "The reference clip could not be transcribed automatically (patch_ref_transcript_failed). "
                f"Type the reference transcript manually and try again. Details: {exc}"
            ) from exc
        if not ref_text:
            raise RuntimeError(
                "The reference transcript came back empty (patch_ref_transcript_failed). "
                "Type the reference transcript manually and try again."
            )

    reporter.stage("load_model", 0.18, 0.30, "Loading F5-TTS model")
    try:
        engine = load_engine()
    except SpeechEditUnavailable as exc:
        raise RuntimeError(str(exc)) from exc

    reporter.stage("synthesize", 0.30, 0.94, "Synthesizing speech")
    ref_path = _make_temp_wav()
    try:
        import soundfile as sf

        sf.write(ref_path, ref_wav, SPEECHEDIT_RATE)
        audio, out_sr = engine.generate(
            ref_path,
            ref_text,
            params.gen_text,
            nfe_step=params.nfe_step,
            speed=params.speed,
            seed=params.seed,
        )
    finally:
        try:
            Path(ref_path).unlink()
        except FileNotFoundError:
            pass

    reporter.stage("encode", 0.94, 1.0, f"Encoding {params.output_format}")
    token, output_path = _output_paths(source_filename, params.output_format)
    encode_master(
        MasterAudio(samples=audio[None, :], sample_rate=out_sr),
        str(output_path),
        params.output_format,
    )

    return SpeechEditResult(
        token=token,
        filename=output_path.name.split("__", 1)[1],
        output_format=params.output_format,
        sample_rate=out_sr,
        duration_sec=round(audio.size / out_sr, 3),
        device_used=engine.device,
        mode="generate",
        regions=[],
        ref_text_used=ref_text,
        warnings=warnings,
    )
