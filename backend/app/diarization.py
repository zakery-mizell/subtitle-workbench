from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Mapping

from .runtime_warnings import suppress_known_audio_stack_warnings

# pyannote community-1 (pyannote.audio 4.x). Its "exclusive" output assigns a
# single most-likely-transcribed speaker at every moment, which keeps word- and
# caption-level speaker assignment stable through overlaps and backchannels.
DIARIZATION_MODEL = "pyannote/speaker-diarization-community-1"


@dataclass(slots=True)
class SpeakerTurn:
    start: float
    end: float
    label: str


def _emit_progress(message: str) -> None:
    print(f"[WhisperX] {message}", flush=True)


def _prepare_waveform(audio_source: str | Mapping[str, Any]) -> dict[str, Any]:
    """Build pyannote's in-memory audio dict.

    Decoding happens here (via WhisperX's ffmpeg loader) instead of inside
    pyannote because pyannote 4.x decodes files with torchcodec, which is not
    reliably importable on every platform we run on.
    """
    import torch

    if isinstance(audio_source, Mapping):
        waveform = audio_source.get("waveform")
        if waveform is None:
            raise RuntimeError("Diarization waveform was missing")
        sample_rate = int(audio_source.get("sample_rate") or 16000)
    else:
        from whisperx.audio import SAMPLE_RATE, load_audio

        waveform = load_audio(audio_source)
        sample_rate = SAMPLE_RATE

    if not torch.is_tensor(waveform):
        waveform = torch.from_numpy(waveform)
    waveform = waveform.detach().cpu().float()
    if waveform.ndim == 1:
        waveform = waveform[None, :]
    elif waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    return {"waveform": waveform, "sample_rate": sample_rate}


def _resolve_device(torch: Any) -> Any:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _annotation_turns(annotation: Any) -> list[SpeakerTurn]:
    turns: list[SpeakerTurn] = []
    for segment, _, label in annotation.itertracks(yield_label=True):
        turns.append(
            SpeakerTurn(
                start=float(segment.start),
                end=float(segment.end),
                label=str(label),
            )
        )
    return turns


def _extract_turns(output: Any) -> tuple[list[SpeakerTurn], list[SpeakerTurn]]:
    """Return (exclusive_turns, raw_turns), both relabeled by appearance order.

    The exclusive annotation assigns one speaker per moment (best for caption
    attribution); the raw annotation keeps overlapping turns (needed to find
    overlap regions and per-speaker solo spans). Labels are relabeled with one
    shared mapping so indices agree between the two lists.
    """
    exclusive = getattr(output, "exclusive_speaker_diarization", None)
    raw = getattr(output, "speaker_diarization", None)
    if exclusive is None and raw is None:
        exclusive = raw = output
    elif exclusive is None:
        exclusive = raw
    elif raw is None:
        raw = exclusive

    exclusive_turns = _annotation_turns(exclusive)
    raw_turns = _annotation_turns(raw)
    rank = _appearance_rank(exclusive_turns + raw_turns)
    return _relabel(exclusive_turns, rank), _relabel(raw_turns, rank)


def _appearance_rank(turns: list[SpeakerTurn]) -> dict[str, int]:
    """Map cluster labels to speaking order, so speaker 0 is the first voice heard.

    pyannote's cluster labels (SPEAKER_00, SPEAKER_01, ...) are arbitrary and
    do not follow speaking order, which made "Speaker 1" land on the wrong
    person. Requested speaker names are matched by index, so index = order of
    first appearance is the intuitive mapping.
    """
    ordered = sorted(turns, key=lambda turn: (turn.start, turn.end))
    appearance_rank: dict[str, int] = {}
    for turn in ordered:
        if turn.label not in appearance_rank:
            appearance_rank[turn.label] = len(appearance_rank)
    return appearance_rank


def _relabel(turns: list[SpeakerTurn], rank: dict[str, int]) -> list[SpeakerTurn]:
    ordered = sorted(turns, key=lambda turn: (turn.start, turn.end))
    return [
        SpeakerTurn(start=turn.start, end=turn.end, label=str(rank[turn.label]))
        for turn in ordered
    ]


def run_diarization(
    audio_source: str | Mapping[str, Any],
    num_speakers: int,
    auth_token: str | None,
    cache_dir: str | None = None,
) -> tuple[list[SpeakerTurn], list[SpeakerTurn]]:
    """Diarize and return (exclusive_turns, raw_turns).

    Exclusive turns never overlap and drive caption speaker attribution; raw
    turns keep genuine overlaps and drive overlap detection/separation.
    """
    if num_speakers <= 1 or not auth_token:
        return [], []

    try:
        with suppress_known_audio_stack_warnings():
            import torch
            from pyannote.audio import Pipeline
    except Exception as exc:  # pragma: no cover - optional dependency path
        raise RuntimeError("pyannote.audio diarization is not installed") from exc

    with suppress_known_audio_stack_warnings():
        audio = _prepare_waveform(audio_source)
        device = _resolve_device(torch)

        _emit_progress(f"Loading diarization pipeline '{DIARIZATION_MODEL}'...")
        pipeline = Pipeline.from_pretrained(
            DIARIZATION_MODEL,
            token=auth_token,
            cache_dir=cache_dir,
        )
        if pipeline is None:
            raise RuntimeError(
                f"Could not load '{DIARIZATION_MODEL}'. Accept its gated access on Hugging Face "
                "and make sure DIARIZATION_AUTH_TOKEN belongs to that account."
            )

        try:
            _emit_progress(
                f"Running speaker diarization for {num_speakers} speakers on {device.type.upper()}..."
            )
            try:
                pipeline.to(device)
                # pyannote mutates the audio dict, so hand it a copy per attempt.
                output = pipeline(dict(audio), num_speakers=num_speakers)
            except Exception:
                if device.type == "cpu":
                    raise
                _emit_progress(f"Diarization failed on {device.type.upper()}; retrying on CPU...")
                pipeline.to(torch.device("cpu"))
                output = pipeline(dict(audio), num_speakers=num_speakers)
            _emit_progress("Speaker diarization complete.")
        finally:
            del pipeline
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    return _extract_turns(output)


def assign_speaker_id(
    start: float,
    end: float,
    turns: list[SpeakerTurn],
    requested_speakers: list[dict[str, Any]],
) -> tuple[int | None, str | None]:
    if not turns:
        if requested_speakers:
            return requested_speakers[0]["id"], requested_speakers[0]["name"]
        return None, None

    overlaps: dict[str, float] = {}
    for turn in turns:
        overlap = max(0.0, min(end, turn.end) - max(start, turn.start))
        if overlap > 0:
            overlaps[turn.label] = overlaps.get(turn.label, 0.0) + overlap

    if not overlaps:
        return None, None

    ranked_labels = sorted(overlaps.items(), key=lambda item: item[1], reverse=True)
    label = ranked_labels[0][0]
    raw_index = resolve_speaker_index(label, turns)
    if requested_speakers:
        speaker = requested_speakers[min(raw_index, len(requested_speakers) - 1)]
        return speaker["id"], speaker["name"]
    return raw_index, f"Speaker {raw_index + 1}"


def resolve_speaker_index(label: str, turns: list[SpeakerTurn]) -> int:
    match = re.search(r"(\d+)$", label)
    if match:
        return int(match.group(1))

    ordered_labels = list(dict.fromkeys(turn.label for turn in turns))
    try:
        return ordered_labels.index(label)
    except ValueError:
        return 0
