from __future__ import annotations

"""Separation job: run UniSE on overlap/handoff windows and render blends.

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
from .handoff import (
    TARGET_WINDOW_MERGE_GAP_SECONDS,
    HandoffWindow,
    audit_handoff_assignments,
    handoff_windows,
)
from .overlap import pick_enrollment_span, speaker_turn_spans
from .schemas import (
    HandoffCorrection,
    RegionReport,
    SeparationParams,
    SeparationResult,
    SoloRegionReport,
    SoloTrackOut,
    SoloTracksParams,
    SoloTracksResult,
    StemWord,
)
from .speaker_verify import SpeakerVerificationUnavailable, verify_isolated_utterance_speakers
from .unise_engine import UNISE_SAMPLE_RATE, WINDOW_SECONDS, SeparationUnavailable, load_engine

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


def _processing_window(start: float, end: float, duration: float) -> tuple[float, float]:
    """Add context without ever handing UniSE a wrap-padded short window."""
    desired = max(WINDOW_SECONDS, end - start + 2 * REGION_PAD_SECONDS)
    desired = min(duration, desired)
    center = (start + end) / 2.0
    window_start = max(0.0, center - desired / 2.0)
    window_end = min(duration, window_start + desired)
    window_start = max(0.0, window_end - desired)
    return window_start, window_end


def _requested_language(language: str | None) -> str | None:
    """All mixture, stem, and handoff transcription is English-only."""
    return "en"


def _whisperx_result(
    stem: np.ndarray,
    model_name: str,
    language: str | None,
    sample_rate: int = UNISE_SAMPLE_RATE,
) -> dict:
    """Run WhisperX over a mono array via a temp WAV."""
    import soundfile as sf

    from ..whisperx_transcription import transcribe_with_whisperx

    Path(settings.temp_upload_dir).mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(delete=False, suffix=".wav", dir=settings.temp_upload_dir) as handle:
        stem_path = handle.name
    try:
        sf.write(stem_path, stem, sample_rate)
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
    return result


def _result_words(result: dict, offset: float = 0.0) -> list[StemWord]:
    words: list[StemWord] = []
    for segment in result.get("segments") or []:
        for word in segment.get("words") or []:
            text = str(word.get("word") or "").strip()
            if not text:
                continue
            start = float(word.get("start", segment.get("start", 0.0))) + offset
            end = float(word.get("end", start)) + offset
            raw_confidence = word.get("probability", word.get("score"))
            confidence = None if raw_confidence is None else max(0.0, min(1.0, float(raw_confidence)))
            words.append(
                StemWord(
                    text=text,
                    start=round(start, 3),
                    end=round(end, 3),
                    confidence=confidence,
                )
            )
    return words


def _words_in_spans(words: list[StemWord], spans: list[tuple[float, float]]) -> list[StemWord]:
    """Keep words whose midpoint survives the playback track's safety gate."""
    return [
        word
        for word in words
        if any(start <= (word.start + word.end) / 2.0 <= end for start, end in spans)
    ]


def _transcribe_stem(
    stem: np.ndarray,
    window_start: float,
    region_start: float,
    region_end: float,
    model_name: str,
    language: str | None,
) -> list[StemWord]:
    result = _whisperx_result(stem, model_name, language)
    # Keep only words that genuinely lie in the overlap region; the padded
    # window edges duplicate words the base transcript already has.
    return [
        word
        for word in _result_words(result, offset=window_start)
        if word.end > region_start + 0.05 and word.start < region_end - 0.05
    ]


def _finalize_track(
    samples: np.ndarray,
    sample_rate: int,
    speaker: int,
    position: str,
    params: SoloTracksParams,
    reporter: ProgressReporter,
    warnings: list[WarningItem],
    gated: bool,
    source_filename: str,
    stage_lo: float = 0.90,
    stage_hi: float = 0.98,
) -> SoloTrackOut:
    """Optionally restore, then encode one per-speaker track to an artifact."""
    track_samples = samples
    track_sr = sample_rate
    if params.restore:
        reporter.stage("restore", stage_lo, stage_hi, f"Restoring speaker {position}")

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
            restored, out_sr = engine_restore(mono, sample_rate, progress=restore_progress)
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
    return SoloTrackOut(
        speaker_index=speaker,
        token=token,
        output_filename=output_path.name.split("__", 1)[1],
    )


def _target_windows(params: SoloTracksParams, duration: float) -> list[HandoffWindow]:
    """Merge overlaps and audit handoffs into the only spans UniSE will inspect."""
    windows = handoff_windows(params.audit_words if params.audit_handoffs else [])
    windows.extend(
        HandoffWindow(
            start=max(0.0, region.start),
            end=min(duration, region.end),
            speaker_indices=tuple(sorted(set(region.speaker_indices))),
        )
        for region in params.regions
        if region.start < duration and region.end > region.start
    )

    merged: list[HandoffWindow] = []
    for candidate in sorted(windows, key=lambda item: (item.start, item.end)):
        window = HandoffWindow(
            start=max(0.0, min(duration, candidate.start)),
            end=max(0.0, min(duration, candidate.end)),
            speaker_indices=candidate.speaker_indices,
        )
        if window.end - window.start < MIN_REGION_SECONDS:
            continue
        if merged and window.start <= merged[-1].end + TARGET_WINDOW_MERGE_GAP_SECONDS:
            previous = merged[-1]
            merged[-1] = HandoffWindow(
                start=previous.start,
                end=max(previous.end, window.end),
                speaker_indices=tuple(sorted(set(previous.speaker_indices) | set(window.speaker_indices))),
            )
        else:
            merged.append(window)
    return merged


def run_targeted_stems(
    source_path: str,
    source_filename: str,
    params: SoloTracksParams,
    reporter: ProgressReporter,
) -> SoloTracksResult:
    """Audit handoffs with UniSE, but replace audio only at true overlaps."""
    warnings: list[WarningItem] = []

    turns = [
        SpeakerTurn(start=turn.start, end=turn.end, label=str(turn.speaker_index))
        for turn in params.turns
    ]
    speakers = sorted({turn.speaker_index for turn in params.turns})

    reporter.stage("decode", 0.0, 0.05, "Decoding audio")
    original = decode_master(source_path)
    mix16 = decode_mono_16k(source_path)
    duration16 = mix16.size / UNISE_SAMPLE_RATE

    windows = _target_windows(params, duration16)
    work_items = [
        (speaker, window)
        for window in windows
        for speaker in window.speaker_indices
        if speaker in speakers
    ]

    engine = None
    if work_items:
        reporter.stage("load_model", 0.05, 0.12, "Loading UniSE separation model")
        try:
            engine = load_engine()
        except SeparationUnavailable as exc:
            raise RuntimeError(str(exc)) from exc

    tracks: list[SoloTrackOut] = []
    reports: list[SoloRegionReport] = []
    audit_words_by_speaker: dict[int, list[StemWord]] = {}
    span_start, span_end = 0.12, 0.98
    for order, speaker in enumerate(speakers):
        lo = span_start + (span_end - span_start) * order / len(speakers)
        hi = span_start + (span_end - span_start) * (order + 1) / len(speakers)
        position = f"{order + 1}/{len(speakers)}"
        gate_spans = [
            (region.start, region.end)
            for region in params.speaker_regions
            if region.speaker_index == speaker
        ] or speaker_turn_spans(turns, speaker)
        if not gate_spans:
            continue

        samples = original.samples.copy()
        speaker_windows = [window for candidate, window in work_items if candidate == speaker]
        separated = False
        detail: str | None = None
        enrollment_missing = False
        for window_index, window in enumerate(speaker_windows):
            window_position = work_items.index((speaker, window))
            work_lo = span_start + (span_end - span_start) * window_position / max(1, len(work_items))
            work_hi = span_start + (span_end - span_start) * (window_position + 1) / max(1, len(work_items))
            report = SoloRegionReport(
                start=round(window.start, 3),
                end=round(window.end, 3),
                speaker_index=speaker,
                applied=False,
            )
            reports.append(report)

            enrollment = pick_enrollment_span(
                turns,
                speaker,
                near=(window.start + window.end) / 2.0,
            )
            if enrollment is None:
                enrollment_missing = True
                report.detail = "No clean solo speech was found for this speaker."
                continue

            window_start, window_end = _processing_window(window.start, window.end, duration16)
            window_audio = _seconds_slice(mix16, window_start, window_end)
            enroll_audio = _seconds_slice(mix16, *enrollment)
            reporter.stage(
                "separate",
                work_lo,
                work_hi,
                (
                    f"Checking speaker {speaker + 1} near overlap or handoff "
                    f"{window_index + 1}/{len(speaker_windows)}"
                ),
            )
            try:
                assert engine is not None
                stem = engine.run_task("tse", window_audio, enroll_audio, progress=reporter.tick)
            except Exception:
                assert engine is not None
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

            # The separated window is transcribed before it is blended or gated,
            # so a bad diarization edge cannot erase the evidence that corrects it.
            if params.audit_handoffs and params.audit_words:
                try:
                    raw_words = _result_words(
                        _whisperx_result(
                            stem,
                            params.transcribe_model,
                            _requested_language(params.language),
                        ),
                        offset=window_start,
                    )
                    audit_words_by_speaker.setdefault(speaker, []).extend(
                        _words_in_spans(raw_words, [(window.start, window.end)])
                    )
                except Exception as exc:
                    warnings.append(
                        WarningItem(
                            code="separation_transcribe_failed",
                            message=(
                                f"Speaker {speaker + 1}'s handoff window could not be "
                                f"transcribed. Details: {exc}"
                            ),
                        )
                    )

            # Handoff windows are useful as *evidence*, but ordinary turn-taking
            # already contains a clean original voice. Re-synthesizing those
            # stretches added warble to otherwise good audio. Confine audible
            # replacement to genuine simultaneous-speech regions.
            replacement_regions = [
                region
                for region in params.regions
                if speaker in region.speaker_indices
                and region.start < window.end
                and region.end > window.start
            ]
            for region in replacement_regions:
                region_start = max(window.start, region.start)
                region_end = min(window.end, region.end)
                if region_end <= region_start:
                    continue
                blend.apply_replace(
                    samples,
                    original.sample_rate,
                    blend.RegionRender(
                        start=window_start,
                        stem=stem,
                        region_start=region_start,
                        region_end=region_end,
                    ),
                    UNISE_SAMPLE_RATE,
                )
                report.applied = True
                separated = True
            if not report.applied:
                report.detail = "Voice checked; original audio preserved because the speakers did not overlap."

        if enrollment_missing:
            detail = (
                "At least one overlap or handoff could not be isolated because this "
                "speaker never had a clean solo voice sample nearby."
            )
            warnings.append(
                WarningItem(code="separation_no_enrollment", message=f"Speaker {speaker + 1}: {detail}")
            )

        # Outside the targeted windows the track is still the untouched original.
        # Gate the assembled track only after the raw UniSE evidence is captured.
        blend.apply_region_gate(samples, original.sample_rate, gate_spans)

        words: list[StemWord] | None = None
        if params.transcribe:
            reporter.stage(
                "transcribe",
                lo + (hi - lo) * 0.72,
                lo + (hi - lo) * 0.86,
                f"Transcribing speaker {speaker + 1}'s assembled track",
            )
            mono = samples.mean(axis=0).astype(np.float32) if samples.shape[0] > 1 else samples[0]
            try:
                words = _words_in_spans(
                    _result_words(
                        _whisperx_result(
                            mono,
                            params.transcribe_model,
                            _requested_language(params.language),
                            original.sample_rate,
                        )
                    ),
                    gate_spans,
                )
            except Exception as exc:
                warnings.append(
                    WarningItem(
                        code="separation_transcribe_failed",
                        message=f"Speaker {speaker + 1}'s stem could not be transcribed. Details: {exc}",
                    )
                )

        track = _finalize_track(
            samples,
            original.sample_rate,
            speaker,
            position,
            params,
            reporter,
            warnings,
            gated=True,
            source_filename=source_filename,
            stage_lo=lo + (hi - lo) * 0.86,
            stage_hi=hi,
        )
        track.separated = separated
        track.words = words
        track.detail = detail
        tracks.append(track)

    reporter.stage("encode", 0.98, 1.0, "Finishing targeted speaker tracks")
    if not tracks:
        warnings.append(
            WarningItem(
                code="separation_nothing_applied",
                message="No targeted speaker tracks could be prepared for this recording.",
            )
        )

    stem_corrections, audited = audit_handoff_assignments(
        params.audit_words if params.audit_handoffs else [],
        audit_words_by_speaker,
    )
    verified_corrections: list[HandoffCorrection] = []
    if params.audit_handoffs and params.audit_words:
        try:
            verified_corrections = verify_isolated_utterance_speakers(
                params.audit_words,
                mix16,
                turns,
            )
        except SpeakerVerificationUnavailable as exc:
            warnings.append(
                WarningItem(
                    code="speaker_verification_unavailable",
                    message=f"Short isolated replies could not be voice-verified. Details: {exc}",
                )
            )
        except Exception as exc:
            warnings.append(
                WarningItem(
                    code="speaker_verification_failed",
                    message=f"Short isolated reply verification failed. Details: {exc}",
                )
            )

    corrections_by_word = {item.word_id: item for item in stem_corrections}
    # Original-audio speaker embeddings are more trustworthy than generated
    # stem leakage for pause-bounded short replies.
    corrections_by_word.update({item.word_id: item for item in verified_corrections})
    corrections = sorted(corrections_by_word.values(), key=lambda item: (item.boundary_time, item.word_id))
    if corrections:
        warnings.append(
            WarningItem(
                code="handoff_speakers_corrected",
                message=(
                    f"The voice audit confirmed and corrected {len(corrections)} word-level speaker "
                    f"assignment{'s' if len(corrections) != 1 else ''} near handoffs or isolated replies."
                ),
            )
        )

    return SoloTracksResult(
        tracks=tracks,
        regions=reports,
        output_format=params.output.format,
        device_used=engine.device if engine is not None else "none",
        warnings=warnings,
        handoff_corrections=corrections,
        handoffs_audited=audited,
    )


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

    mode="targeted" dispatches to run_targeted_stems instead: UniSE only sees
    short merged windows around overlaps and rapid handoffs.
    """
    warnings: list[WarningItem] = []

    if params.mode == "targeted":
        if params.turns:
            return run_targeted_stems(source_path, source_filename, params, reporter)
        # Targeted extraction needs diarization turns for enrollment and gating;
        # without them the legacy gated behaviour is everything still possible.
        warnings.append(
            WarningItem(
                code="separation_targeted_needs_turns",
                message=(
                    "Targeted per-speaker extraction needs diarization turns; the tracks were "
                    "prepared with region gating instead."
                ),
            )
        )

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
        enroll_cache: dict[tuple[float, float], np.ndarray] = {}
        enrollment_missing = False

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

            # Per region, not once per speaker: the closest solo speech has the
            # mic/room conditions of *this* overlap.
            enrollment_span = pick_enrollment_span(turns, speaker, near=region.start)
            enroll_audio = None
            if enrollment_span is not None and mix16 is not None:
                enroll_audio = enroll_cache.get(enrollment_span)
                if enroll_audio is None:
                    enroll_audio = _seconds_slice(mix16, *enrollment_span)
                    enroll_cache[enrollment_span] = enroll_audio
            if enroll_audio is None:
                enrollment_missing = True
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

        if regions and enrollment_missing:
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

        # The speaker-based transcript: even a gated track (original audio,
        # everyone else silenced) transcribes into a per-speaker word list.
        words: list[StemWord] | None = None
        if params.transcribe:
            reporter.stage("transcribe", 0.88, 0.90, f"Transcribing speaker {speaker + 1}'s track")
            mono = samples.mean(axis=0).astype(np.float32) if samples.shape[0] > 1 else samples[0]
            try:
                words = _result_words(
                    _whisperx_result(
                        mono,
                        params.transcribe_model,
                        _requested_language(params.language),
                        original.sample_rate,
                    )
                )
            except Exception as exc:
                warnings.append(
                    WarningItem(
                        code="separation_transcribe_failed",
                        message=f"Speaker {speaker + 1}'s track could not be transcribed. Details: {exc}",
                    )
                )

        position = f"{speaker_indices.index(speaker) + 1}/{len(speaker_indices)}"
        track = _finalize_track(
            samples,
            original.sample_rate,
            speaker,
            position,
            params,
            reporter,
            warnings,
            gated=gated,
            source_filename=source_filename,
        )
        track.words = words
        tracks.append(track)

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
                    _requested_language(params.language),
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
