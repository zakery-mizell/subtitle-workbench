from __future__ import annotations

"""Separation job: run UniSE on overlap regions and render the chosen blends.

Mirrors the mastering pipeline's shape (token, artifact dir, ProgressReporter)
so the API endpoints and frontend polling can work the same way.
"""

import shutil
import subprocess
import uuid
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np

from ..config import settings
from ..diarization import SpeakerTurn
from ..jobs import ProgressReporter
from ..mastering.audio_io import MasterAudio, decode_master, encode_master
from ..restore import sidon_engine
from ..restore.engine import RestoreUnavailable, restore_waveform
from ..schemas import WarningItem
from . import blend
from .overlap import pick_enrollment_span
from .schemas import (
    RegionReport,
    SeparationParams,
    SeparationResult,
    SoloRegionReport,
    SoloTrackOut,
    SoloTracksParams,
    SoloTracksResult,
    StemWord,
)
from .unise_engine import UNISE_SAMPLE_RATE, SeparationUnavailable, load_engine

REGION_PAD_SECONDS = 0.35
MIN_REGION_SECONDS = 0.25


def separation_output_dir() -> Path:
    return Path(settings.mastering_output_dir).parent / "separation"


def find_separation_artifact(token: str) -> Path | None:
    output_dir = separation_output_dir()
    if not output_dir.is_dir() or not token.replace("s_", "").isalnum():
        return None
    matches = sorted(output_dir.glob(f"{token}__*"))
    return matches[0] if matches else None


def _output_paths(source_filename: str, fmt: str, suffix: str = "separated") -> tuple[str, Path]:
    token = f"s_{uuid.uuid4().hex[:12]}"
    stem = Path(source_filename or "audio").stem or "audio"
    output_dir = separation_output_dir()
    output_dir.mkdir(parents=True, exist_ok=True)
    return token, output_dir / f"{token}__{stem}.{suffix}.{fmt}"


def decode_mono_16k(source_path: str) -> np.ndarray:
    command = [
        "ffmpeg", "-v", "error", "-i", source_path, "-vn", "-ac", "1",
        "-ar", str(UNISE_SAMPLE_RATE), "-f", "f32le", "-acodec", "pcm_f32le", "pipe:1",
    ]
    try:
        completed = subprocess.run(command, check=True, capture_output=True)
    except FileNotFoundError as exc:
        raise RuntimeError("FFmpeg is required for speaker separation.") from exc
    except subprocess.CalledProcessError as exc:
        detail = exc.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"Could not decode audio for separation. Details: {detail}") from exc
    audio = np.frombuffer(completed.stdout, dtype=np.float32)
    if audio.size == 0:
        raise RuntimeError("No decodable audio samples were found in this file.")
    return audio.copy()


def _seconds_slice(audio: np.ndarray, start: float, end: float) -> np.ndarray:
    lo = max(0, int(start * UNISE_SAMPLE_RATE))
    hi = min(audio.size, int(end * UNISE_SAMPLE_RATE))
    return audio[lo:hi]


def _transcribe_stem(
    stem: np.ndarray,
    window_start: float,
    region_start: float,
    region_end: float,
    model_name: str,
    language: str | None,
) -> list[StemWord]:
    import soundfile as sf

    from ..whisperx_transcription import transcribe_with_whisperx

    Path(settings.temp_upload_dir).mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(delete=False, suffix=".wav", dir=settings.temp_upload_dir) as handle:
        stem_path = handle.name
    try:
        sf.write(stem_path, stem, UNISE_SAMPLE_RATE)
        result, _, _ = transcribe_with_whisperx(
            stem_path,
            model_name=model_name,
            requested_language=language,
            hotwords=None,
        )
    finally:
        try:
            Path(stem_path).unlink()
        except FileNotFoundError:
            pass

    words: list[StemWord] = []
    for segment in result.get("segments") or []:
        for word in segment.get("words") or []:
            text = str(word.get("word") or "").strip()
            if not text:
                continue
            start = float(word.get("start", segment.get("start", 0.0))) + window_start
            end = float(word.get("end", start)) + window_start
            # Keep only words that genuinely lie in the overlap region; the
            # padded window edges duplicate words the base transcript already has.
            if end <= region_start + 0.05 or start >= region_end - 0.05:
                continue
            words.append(StemWord(text=text, start=round(start, 3), end=round(end, 3)))
    return words


def run_solo_tracks(
    source_path: str,
    source_filename: str,
    params: SoloTracksParams,
    reporter: ProgressReporter,
) -> SoloTracksResult:
    """Render one playback track per speaker involved in any overlap.

    Each track is the untouched original except inside overlap regions, where
    the mixture is replaced by that speaker's isolated voice (UniSE tse). When
    `params.speaker_regions` are supplied the track is additionally gated to that
    speaker's silence-snapped regions, so it stands alone on the original
    timeline: same length and timings, everyone else silent.
    """
    warnings: list[WarningItem] = []

    turns = [
        SpeakerTurn(start=turn.start, end=turn.end, label=str(turn.speaker_index))
        for turn in params.turns
    ]
    speaker_indices = sorted(
        {index for region in params.regions for index in region.speaker_indices}
        | {region.speaker_index for region in params.speaker_regions}
    )

    work_items = [
        (speaker, region)
        for speaker in speaker_indices
        for region in params.regions
        if speaker in region.speaker_indices
    ]

    reporter.stage("decode", 0.0, 0.06, "Decoding audio")
    original = decode_master(source_path)

    # A gating-only job (clean recording, no overlaps) never touches UniSE, so it
    # must not require the model to be installed or a second decode pass.
    engine = None
    mix16 = None
    duration16 = 0.0
    if work_items:
        mix16 = decode_mono_16k(source_path)
        duration16 = mix16.size / UNISE_SAMPLE_RATE
        reporter.stage("load_model", 0.06, 0.16, "Loading UniSE separation model")
        try:
            engine = load_engine()
        except SeparationUnavailable as exc:
            raise RuntimeError(str(exc)) from exc

    tracks: list[SoloTrackOut] = []
    reports: list[SoloRegionReport] = []
    span_start, span_end = 0.16, 0.90
    done = 0
    if not work_items:
        reporter.stage("gate", 0.06, 0.90, "Preparing per-speaker tracks")

    for speaker in speaker_indices:
        samples = original.samples.copy()
        applied_any = False

        regions = [region for region in params.regions if speaker in region.speaker_indices]
        enrollment_span = (
            pick_enrollment_span(turns, speaker, near=regions[0].start) if regions else None
        )
        enroll_audio = _seconds_slice(mix16, *enrollment_span) if enrollment_span and mix16 is not None else None

        for region in regions:
            stage_lo = span_start + (span_end - span_start) * done / len(work_items)
            stage_hi = span_start + (span_end - span_start) * (done + 1) / len(work_items)
            done += 1
            reporter.stage(
                "separate",
                stage_lo,
                stage_hi,
                f"Isolating speaker {speaker + 1} in overlap at {region.start:.1f}s",
            )
            report = SoloRegionReport(
                start=region.start, end=region.end, speaker_index=speaker, applied=False
            )
            reports.append(report)

            if enroll_audio is None:
                report.detail = "No clean solo speech was found for this speaker."
                continue
            if region.end - region.start < MIN_REGION_SECONDS or region.start >= duration16:
                report.detail = "Region is too short to process."
                continue

            window_start = max(0.0, region.start - REGION_PAD_SECONDS)
            window_end = min(duration16, region.end + REGION_PAD_SECONDS)
            window_audio = _seconds_slice(mix16, window_start, window_end)

            try:
                stem = engine.run_task("tse", window_audio, enroll_audio, progress=reporter.tick)
            except Exception:
                if engine.device == "cpu":
                    raise
                warnings.append(
                    WarningItem(
                        code="separation_cpu_fallback",
                        message="Separation failed on the GPU and was retried on the CPU, which is slower.",
                    )
                )
                engine = load_engine("cpu")
                stem = engine.run_task("tse", window_audio, enroll_audio, progress=reporter.tick)

            blend.apply_replace(
                samples,
                original.sample_rate,
                blend.RegionRender(
                    start=window_start, stem=stem, region_start=region.start, region_end=region.end
                ),
                UNISE_SAMPLE_RATE,
            )
            report.applied = True
            applied_any = True

        if regions and enrollment_span is None:
            warnings.append(
                WarningItem(
                    code="separation_no_enrollment",
                    message=(
                        f"Speaker {speaker + 1} never speaks alone, so their solo track "
                        "could not isolate the overlaps."
                    ),
                )
            )

        # Gate to this speaker's regions. Their edges already sit in silence, so
        # the equal-power crossfades only guard against clicks at the few splits
        # that a genuine no-pause handoff forces mid-speech.
        gated = blend.apply_region_gate(
            samples,
            original.sample_rate,
            [(region.start, region.end) for region in params.speaker_regions if region.speaker_index == speaker],
        )
        # A gated track is worth emitting on its own; without gating an untouched
        # copy of the original would be pointless.
        if not applied_any and not gated:
            continue

        track_samples = samples
        track_sr = original.sample_rate
        if params.restore:
            position = f"{speaker_indices.index(speaker) + 1}/{len(speaker_indices)}"
            reporter.stage("restore", 0.90, 0.98, f"Restoring speaker {position}")

            def restore_progress(fraction: float, message: str | None = None) -> None:
                label = None
                if message:
                    label = f"Restoring speaker {position} — {message.removeprefix('Restoring ').strip()}"
                reporter.tick(fraction, label)

            # The assembled track can be stereo (channels, n); both engines are mono.
            mono = samples.mean(axis=0).astype(np.float32) if samples.shape[0] > 1 else samples[0]
            engine_restore = (
                sidon_engine.restore_waveform
                if params.restore_engine == "sidon"
                else restore_waveform
            )
            try:
                restored, out_sr = engine_restore(mono, original.sample_rate, progress=restore_progress)
                track_samples = restored[None, :]
                track_sr = out_sr
            except RestoreUnavailable as exc:
                warnings.append(WarningItem(code="restore_unavailable", message=str(exc)))
            except Exception as exc:  # graceful degradation: keep the unrestored track
                warnings.append(
                    WarningItem(
                        code="restore_failed",
                        message=f"Speaker {speaker + 1}'s track could not be restored. Details: {exc}",
                    )
                )

        # A gated track is mostly digital silence, so its bitrate is wildly
        # uneven and a FLAC without a seektable seeks tens of seconds off (the
        # browser interpolates time from byte offsets). Fall back to a container
        # that seeks exactly on its own when the flac tools are missing.
        track_format = params.output.format
        if gated and track_format == "flac" and shutil.which("metaflac") is None:
            track_format = "wav"
            if not any(warning.code == "separation_seektable_missing" for warning in warnings):
                warnings.append(
                    WarningItem(
                        code="separation_seektable_missing",
                        message=(
                            "Speaker tracks were written as WAV because exact seeking in a gated "
                            "FLAC needs the `flac` command-line tools. Install flac for much "
                            "smaller files."
                        ),
                    )
                )

        token, output_path = _output_paths(source_filename, track_format, suffix=f"solo{speaker}")
        encode_master(
            MasterAudio(samples=track_samples, sample_rate=track_sr),
            str(output_path),
            track_format,
            params.output.bitrate_kbps,
        )
        tracks.append(
            SoloTrackOut(
                speaker_index=speaker,
                token=token,
                output_filename=output_path.name.split("__", 1)[1],
            )
        )

    reporter.stage("encode", 0.90, 1.0, "Finishing solo tracks")
    if not tracks:
        warnings.append(
            WarningItem(
                code="separation_nothing_applied",
                message="No solo tracks could be prepared for this recording.",
            )
        )

    return SoloTracksResult(
        tracks=tracks,
        regions=reports,
        output_format=params.output.format,
        device_used=engine.device if engine is not None else "none",
        warnings=warnings,
    )


def run_separation(
    source_path: str,
    source_filename: str,
    params: SeparationParams,
    reporter: ProgressReporter,
) -> SeparationResult:
    warnings: list[WarningItem] = []

    reporter.stage("decode", 0.0, 0.06, "Decoding audio")
    original = decode_master(source_path)
    mix16 = decode_mono_16k(source_path)
    duration16 = mix16.size / UNISE_SAMPLE_RATE

    reporter.stage("load_model", 0.06, 0.16, "Loading UniSE separation model")
    try:
        engine = load_engine()
    except SeparationUnavailable as exc:
        raise RuntimeError(str(exc)) from exc

    turns = [
        SpeakerTurn(start=turn.start, end=turn.end, label=str(turn.speaker_index))
        for turn in params.turns
    ]

    samples = original.samples  # blended in place per region
    reports: list[RegionReport] = []
    total_regions = len(params.regions)
    span_start, span_end = 0.16, 0.90

    for index, region in enumerate(params.regions):
        stage_lo = span_start + (span_end - span_start) * index / total_regions
        stage_hi = span_start + (span_end - span_start) * (index + 1) / total_regions
        reporter.stage(
            "separate",
            stage_lo,
            stage_hi,
            f"Separating overlap {index + 1} of {total_regions} ({region.start:.1f}s)",
        )

        report = RegionReport(
            start=region.start,
            end=region.end,
            mode=region.mode,
            target_speaker_index=region.target_speaker_index,
            applied=False,
        )
        reports.append(report)

        if region.end - region.start < MIN_REGION_SECONDS or region.start >= duration16:
            report.detail = "Region is too short to process."
            continue

        enrollment_span = pick_enrollment_span(turns, region.target_speaker_index, near=region.start)
        if enrollment_span is None:
            report.detail = (
                "No clean solo speech was found for this speaker, so there was no "
                "voice sample to separate with."
            )
            warnings.append(
                WarningItem(
                    code="separation_no_enrollment",
                    message=f"Skipped the overlap at {region.start:.1f}s: {report.detail}",
                )
            )
            continue

        window_start = max(0.0, region.start - REGION_PAD_SECONDS)
        window_end = min(duration16, region.end + REGION_PAD_SECONDS)
        window_audio = _seconds_slice(mix16, window_start, window_end)
        enroll_audio = _seconds_slice(mix16, *enrollment_span)
        task = "tse" if region.mode == "spotlight" else "rtse"

        try:
            stem = engine.run_task(
                task,
                window_audio,
                enroll_audio,
                progress=lambda fraction: reporter.tick(fraction * 0.8),
            )
        except Exception:
            if engine.device == "cpu":
                raise
            warnings.append(
                WarningItem(
                    code="separation_cpu_fallback",
                    message="Separation failed on the GPU and was retried on the CPU, which is slower.",
                )
            )
            engine = load_engine("cpu")
            stem = engine.run_task(
                task,
                window_audio,
                enroll_audio,
                progress=lambda fraction: reporter.tick(fraction * 0.8),
            )

        render = blend.RegionRender(
            start=window_start,
            stem=stem,
            region_start=region.start,
            region_end=region.end,
        )
        if region.mode == "spotlight":
            blend.apply_spotlight(
                samples, original.sample_rate, render, UNISE_SAMPLE_RATE, duck_db=params.duck_db
            )
        else:
            blend.apply_replace(samples, original.sample_rate, render, UNISE_SAMPLE_RATE)

        report.applied = True
        report.enrollment_start = round(enrollment_span[0], 3)
        report.enrollment_end = round(enrollment_span[1], 3)

        if params.transcribe_stems and region.mode == "spotlight":
            reporter.tick(0.85, "Transcribing the separated voice")
            try:
                report.words = _transcribe_stem(
                    stem,
                    window_start,
                    region.start,
                    region.end,
                    params.transcribe_model,
                    params.language or settings.default_language,
                )
            except Exception as exc:
                report.detail = f"The separated voice could not be transcribed. Details: {exc}"
                warnings.append(
                    WarningItem(code="separation_transcribe_failed", message=report.detail)
                )

    reporter.stage("encode", 0.90, 1.0, f"Encoding {params.output.format}")
    token, output_path = _output_paths(source_filename, params.output.format)
    encode_master(original, str(output_path), params.output.format, params.output.bitrate_kbps)

    if not any(report.applied for report in reports):
        warnings.append(
            WarningItem(
                code="separation_nothing_applied",
                message="No overlap regions could be processed; the output matches the original audio.",
            )
        )

    return SeparationResult(
        token=token,
        output_filename=output_path.name.split("__", 1)[1],
        output_format=params.output.format,
        device_used=engine.device,
        regions=reports,
        warnings=warnings,
    )
