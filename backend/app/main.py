from __future__ import annotations

import json
import math
import os
import subprocess
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

from fastapi import FastAPI, File, Form, UploadFile
from fastapi import HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import ValidationError

from fastapi.responses import FileResponse, PlainTextResponse

from .config import settings
from .diarization import assign_speaker_id, run_diarization
from .jobs import registry as job_registry
from .mastering.cutting import CutRegion, export_audacity_labels
from .mastering.pipeline import find_master_artifact, run_mastering
from .mastering.schemas import JobStatusResponse, MasterJobResponse, MasteringParams
from .schemas import CapabilitiesResponse, RetranscribeRangeResponse, SpeakerAssignmentMode, SpeakerInput, TranscriptResponse, WarningItem, WaveformAnalysisResponse, WordToken
from .text_processing import build_captions, build_guide_blocks, build_paragraphs, build_words, remove_disfluencies
from .waveform_analysis import analyze_waveform
from .whisperx_transcription import transcribe_with_whisperx

if os.name != "nt":
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

app = FastAPI(title=settings.app_name)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[settings.cors_origin, "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/capabilities", response_model=CapabilitiesResponse)
def capabilities() -> CapabilitiesResponse:
    return CapabilitiesResponse(diarization_configured=bool(settings.diarization_auth_token))


@app.post("/api/analyze-waveform", response_model=WaveformAnalysisResponse)
async def analyze_waveform_endpoint(audio: UploadFile = File(...)) -> WaveformAnalysisResponse:
    temp_path = await save_upload_to_temp(audio)
    try:
        return analyze_waveform(temp_path)
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        delete_file_quietly(temp_path)


def parse_speakers_json(speakers_json: str) -> list[SpeakerInput]:
    try:
        payload = json.loads(speakers_json)
        if not isinstance(payload, list):
            raise TypeError("speakers_json must decode to a list")
        return [SpeakerInput.model_validate(item) for item in payload]
    except (json.JSONDecodeError, ValidationError, TypeError) as exc:
        raise HTTPException(status_code=422, detail="speakers_json must be a JSON array of speaker objects with id and name fields.") from exc


def validate_speaker_request(speakers: list[SpeakerInput], speaker_count: int) -> list[SpeakerInput]:
    if speaker_count < 1:
        raise HTTPException(status_code=422, detail="speaker_count must be at least 1.")
    if not speakers:
        raise HTTPException(status_code=422, detail="At least one speaker must be provided.")
    if len(speakers) != speaker_count:
        raise HTTPException(status_code=422, detail="speaker_count must match the number of speakers in speakers_json.")

    normalized_speakers: list[SpeakerInput] = []
    seen_ids: set[int] = set()
    for speaker in speakers:
        normalized_name = speaker.name.strip()
        if not normalized_name:
            raise HTTPException(status_code=422, detail="Speaker names must not be blank.")
        if speaker.id in seen_ids:
            raise HTTPException(status_code=422, detail="Speaker ids in speakers_json must be unique.")
        seen_ids.add(speaker.id)
        normalized_speakers.append(speaker.model_copy(update={"name": normalized_name}))

    return normalized_speakers


def resolve_requested_language(language: str | None, default_language: str | None) -> str | None:
    normalized = (language or default_language or "").strip()
    return normalized or None


def coerce_timestamp(value: object, fallback: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return fallback
    if not math.isfinite(parsed):
        return fallback
    return parsed


def prepare_diarization_audio(source_path: str) -> str:
    source = Path(source_path)
    if source.suffix.lower() == ".wav":
        return source_path

    with named_temp_upload_file(".wav") as handle:
        normalized_path = handle.name

    command = [
        "ffmpeg",
        "-y",
        "-i",
        source_path,
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-c:a",
        "pcm_s16le",
        normalized_path,
    ]

    try:
        subprocess.run(command, check=True, capture_output=True, text=True)
        return normalized_path
    except Exception:
        try:
            os.unlink(normalized_path)
        except FileNotFoundError:
            pass
        return source_path


async def save_upload_to_temp(audio: UploadFile) -> str:
    with named_temp_upload_file(Path(audio.filename or "audio").suffix or ".wav") as handle:
        temp_path = handle.name
        while chunk := await audio.read(1024 * 1024):
            handle.write(chunk)
    return temp_path


def clip_audio_range(source_path: str, start_seconds: float, end_seconds: float) -> str:
    duration = max(0.1, end_seconds - start_seconds)
    with named_temp_upload_file(".wav") as handle:
        clip_path = handle.name

    command = [
        "ffmpeg",
        "-y",
        "-ss",
        f"{start_seconds:.3f}",
        "-t",
        f"{duration:.3f}",
        "-i",
        source_path,
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-c:a",
        "pcm_s16le",
        clip_path,
    ]

    try:
        subprocess.run(command, check=True, capture_output=True, text=True)
        return clip_path
    except Exception as exc:
        try:
            os.unlink(clip_path)
        except FileNotFoundError:
            pass
        raise HTTPException(status_code=500, detail=f"Could not extract the requested audio range. Details: {exc}") from exc


def delete_file_quietly(path: str | None) -> None:
    if not path:
        return
    try:
        os.unlink(path)
    except FileNotFoundError:
        pass


def named_temp_upload_file(suffix: str):
    Path(settings.temp_upload_dir).mkdir(parents=True, exist_ok=True)
    return NamedTemporaryFile(delete=False, suffix=suffix, dir=settings.temp_upload_dir)


def shift_segment_times(segments: list[dict[str, Any]], offset_seconds: float) -> list[dict[str, Any]]:
    shifted: list[dict[str, Any]] = []
    for segment in segments:
        next_segment = dict(segment)
        segment_start = coerce_timestamp(segment.get("start"), 0.0)
        next_segment["start"] = segment_start + offset_seconds
        next_segment["end"] = coerce_timestamp(segment.get("end"), segment_start) + offset_seconds
        segment_words = []
        for word in segment.get("words") or []:
            next_word = dict(word)
            word_start = coerce_timestamp(word.get("start"), segment_start)
            if "start" in word:
                next_word["start"] = word_start + offset_seconds
            if "end" in word:
                next_word["end"] = coerce_timestamp(word["end"], word_start) + offset_seconds
            segment_words.append(next_word)
        if segment_words:
            next_segment["words"] = segment_words
        shifted.append(next_segment)
    return shifted


def trim_words_to_range(words: list[WordToken], start_seconds: float, end_seconds: float) -> list[WordToken]:
    trimmed: list[WordToken] = []
    for word in words:
        if word.end <= start_seconds or word.start >= end_seconds:
            continue
        trimmed.append(
            word.model_copy(
                update={
                    "start": max(word.start, start_seconds),
                    "end": min(word.end, end_seconds),
                }
            )
        )
    return trimmed


def default_speaker_lookup(_: float, __: float) -> tuple[int | None, str | None]:
    return None, None


@app.post("/api/transcribe", response_model=TranscriptResponse)
async def transcribe(
    audio: UploadFile = File(...),
    model: str = Form("large-v3"),
    speaker_count: int = Form(1, ge=1, le=12),
    speakers_json: str = Form("[{\"id\":0,\"name\":\"Speaker 1\"}]"),
    speaker_assignment_mode: SpeakerAssignmentMode = Form("segment"),
    language: str | None = Form(None),
    hotwords: str | None = Form(None),
    remove_disfluencies_enabled: bool = Form(False, alias="remove_disfluencies"),
):
    speakers = validate_speaker_request(parse_speakers_json(speakers_json), speaker_count)
    warnings: list[WarningItem] = []
    requested_language = resolve_requested_language(language, settings.default_language)
    diarization_audio_path = None

    temp_path = await save_upload_to_temp(audio)

    try:
        result, transcription_warnings, gpu_enabled = transcribe_with_whisperx(
            temp_path,
            model_name=model,
            requested_language=requested_language,
            hotwords=hotwords,
        )
        warnings.extend(transcription_warnings)

        segments = result.get("segments") or []
        duration = coerce_timestamp(segments[-1].get("end"), 0.0) if segments else None

        diarization_turns = []
        if speaker_count > 1:
            diarization_cutoff = settings.diarization_max_duration_seconds
            if duration and diarization_cutoff > 0 and duration > diarization_cutoff:
                if diarization_cutoff < 60:
                    cutoff_value = int(diarization_cutoff)
                    cutoff_unit = "second" if cutoff_value == 1 else "seconds"
                else:
                    cutoff_value = int(diarization_cutoff // 60)
                    cutoff_unit = "minute" if cutoff_value == 1 else "minutes"
                cutoff_label = f"{cutoff_value} {cutoff_unit}"
                warnings.append(
                    WarningItem(
                        code="speaker_fallback",
                        message=(
                            "Speaker diarization was skipped because this file is longer than "
                            f"{cutoff_label}. The transcript used the first speaker name "
                            "so the subtitles could finish rendering. Raise DIARIZATION_MAX_DURATION_SECONDS in .env "
                            "or set it to 0 to force diarization on longer files."
                        ),
                    )
                )
            else:
                try:
                    diarization_audio_path = prepare_diarization_audio(temp_path)
                    diarization_turns = run_diarization(
                        diarization_audio_path,
                        num_speakers=speaker_count,
                        auth_token=settings.diarization_auth_token,
                        cache_dir=settings.whisper_cache_dir,
                    )
                    if not diarization_turns:
                        warnings.append(
                            WarningItem(
                                code="speaker_fallback",
                                message=(
                                    "Speaker diarization was not available. The transcript uses the first speaker "
                                    "name as a fallback. Set DIARIZATION_AUTH_TOKEN in .env to enable pyannote diarization."
                                ),
                            )
                        )
                except Exception as exc:
                    warnings.append(
                        WarningItem(
                            code="speaker_fallback",
                            message=(
                                "Speaker diarization failed and the transcript fell back to a single speaker. "
                                f"Details: {exc}"
                            ),
                        )
                    )

        def speaker_lookup(start: float, end: float) -> tuple[int | None, str | None]:
            return assign_speaker_id(
                start=start,
                end=end,
                turns=diarization_turns,
                requested_speakers=[speaker.model_dump() for speaker in speakers],
            )

        words = build_words(
            segments,
            speaker_lookup,
            speaker_assignment_mode=speaker_assignment_mode,
        )
        if remove_disfluencies_enabled:
            words = remove_disfluencies(words)
        paragraphs = build_paragraphs(words)
        captions = build_captions(words)
        guide_blocks = build_guide_blocks(words, captions)

        return TranscriptResponse(
            audio_filename=audio.filename or "audio",
            duration=duration,
            speakers=speakers,
            words=words,
            paragraphs=paragraphs,
            captions=captions,
            guide_blocks=guide_blocks,
            warnings=warnings,
            model=model,
            speaker_assignment_mode=speaker_assignment_mode,
            language=result.get("language"),
            gpu_enabled=gpu_enabled,
        )
    finally:
        if diarization_audio_path and diarization_audio_path != temp_path:
            delete_file_quietly(diarization_audio_path)
        delete_file_quietly(temp_path)


@app.post("/api/retranscribe-range", response_model=RetranscribeRangeResponse)
async def retranscribe_range(
    audio: UploadFile = File(...),
    model: str = Form("large-v3"),
    start_seconds: float = Form(..., ge=0),
    end_seconds: float = Form(..., gt=0),
    hotwords: str | None = Form(None),
    language: str | None = Form(None),
    remove_disfluencies_enabled: bool = Form(False, alias="remove_disfluencies"),
):
    if not math.isfinite(start_seconds) or not math.isfinite(end_seconds):
        raise HTTPException(status_code=400, detail="The selected range must use finite timestamps.")
    if end_seconds <= start_seconds:
        raise HTTPException(status_code=400, detail="The selected range must have a positive duration.")

    requested_language = resolve_requested_language(language, settings.default_language)
    clip_padding_seconds = 0.75
    clip_start = max(0.0, start_seconds - clip_padding_seconds)
    clip_end = end_seconds + clip_padding_seconds

    temp_path = await save_upload_to_temp(audio)
    clip_path = None

    try:
        clip_path = clip_audio_range(temp_path, clip_start, clip_end)
        result, warnings, gpu_enabled = transcribe_with_whisperx(
            clip_path,
            model_name=model,
            requested_language=requested_language,
            hotwords=hotwords,
        )
        shifted_segments = shift_segment_times(result.get("segments") or [], clip_start)
        words = trim_words_to_range(
            build_words(
                shifted_segments,
                default_speaker_lookup,
                speaker_assignment_mode="segment",
            ),
            start_seconds,
            end_seconds,
        )
        if remove_disfluencies_enabled:
            words = remove_disfluencies(words)
        paragraphs = build_paragraphs(words)
        captions = build_captions(words)

        if not words:
            warnings.append(
                WarningItem(
                    code="retranscribe_empty",
                    message="WhisperX did not return any words for the selected range.",
                )
            )

        return RetranscribeRangeResponse(
            start=start_seconds,
            end=end_seconds,
            words=words,
            paragraphs=paragraphs,
            captions=captions,
            warnings=warnings,
            model=model,
            language=result.get("language"),
            gpu_enabled=gpu_enabled,
        )
    finally:
        delete_file_quietly(clip_path)
        delete_file_quietly(temp_path)


def parse_mastering_params(params_json: str) -> MasteringParams:
    try:
        return MasteringParams.model_validate_json(params_json)
    except ValidationError as exc:
        raise HTTPException(status_code=422, detail=f"params_json is not a valid mastering configuration. Details: {exc.error_count()} invalid field(s).") from exc


def parse_words_json(words_json: str | None) -> list[dict[str, Any]] | None:
    if not words_json:
        return None
    try:
        payload = json.loads(words_json)
        if not isinstance(payload, list):
            raise TypeError("words_json must decode to a list")
        return [dict(item) for item in payload if isinstance(item, dict)]
    except (json.JSONDecodeError, TypeError) as exc:
        raise HTTPException(status_code=422, detail="words_json must be a JSON array of word objects.") from exc


@app.post("/api/master", response_model=MasterJobResponse)
async def master_audio(
    audio: UploadFile = File(...),
    params_json: str = Form("{}"),
    words_json: str | None = Form(None),
) -> MasterJobResponse:
    params = parse_mastering_params(params_json)
    words = parse_words_json(words_json)
    source_filename = audio.filename or "audio"
    temp_path = await save_upload_to_temp(audio)

    job_registry.purge_expired(settings.mastering_job_ttl_seconds, delete_artifact=delete_file_quietly)

    def run(job, reporter):
        try:
            result = run_mastering(temp_path, source_filename, params, words, reporter)
        finally:
            delete_file_quietly(temp_path)
        artifact = find_master_artifact(result.token)
        if artifact:
            job.artifacts.append(str(artifact))
        return result

    return MasterJobResponse(job_id=job_registry.submit(run))


@app.get("/api/jobs/{job_id}", response_model=JobStatusResponse)
def job_status(job_id: str) -> JobStatusResponse:
    job = job_registry.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Unknown job id.")
    return JobStatusResponse(
        id=job.id,
        status=job.status,
        stage=job.stage,
        progress=round(job.progress, 4),
        message=job.message,
        error=job.error,
        result=job.result,
    )


def resolve_master_artifact(token: str) -> Path:
    artifact = find_master_artifact(token)
    if artifact is None or not artifact.is_file():
        raise HTTPException(status_code=404, detail="No processed audio was found for this token. It may have expired.")
    return artifact


MASTER_MEDIA_TYPES = {
    ".wav": "audio/wav",
    ".flac": "audio/flac",
    ".mp3": "audio/mpeg",
    ".aac": "audio/aac",
    ".opus": "audio/ogg",
}


@app.get("/api/master/{token}/audio")
def master_audio_file(token: str) -> FileResponse:
    artifact = resolve_master_artifact(token)
    media_type = MASTER_MEDIA_TYPES.get(artifact.suffix.lower(), "application/octet-stream")
    download_name = artifact.name.split("__", 1)[-1]
    return FileResponse(str(artifact), media_type=media_type, filename=download_name)


@app.get("/api/master/{token}/waveform", response_model=WaveformAnalysisResponse)
def master_waveform(token: str) -> WaveformAnalysisResponse:
    artifact = resolve_master_artifact(token)
    try:
        return analyze_waveform(str(artifact))
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/api/master/{token}/cut-list")
def master_cut_list(token: str, format: str = "json"):
    job = next(
        (candidate for candidate in job_registry.all_jobs() if getattr(candidate.result, "token", None) == token),
        None,
    )
    if job is None or job.result is None:
        raise HTTPException(status_code=404, detail="No cut list was found for this token. It may have expired.")
    if format == "audacity":
        cuts = [CutRegion(start=cut.start, end=cut.end, reason=cut.reason, label=cut.label) for cut in job.result.cut_list]
        return PlainTextResponse(export_audacity_labels(cuts), media_type="text/plain")
    return {"cut_list": [cut.model_dump() for cut in job.result.cut_list]}


@app.delete("/api/master/{token}")
def delete_master(token: str) -> dict[str, str]:
    artifact = find_master_artifact(token)
    if artifact:
        delete_file_quietly(str(artifact))
    return {"status": "deleted"}
