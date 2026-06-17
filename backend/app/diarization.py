from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Mapping

from .runtime_warnings import suppress_known_audio_stack_warnings


@dataclass(slots=True)
class SpeakerTurn:
    start: float
    end: float
    label: str


def _emit_progress(message: str) -> None:
    print(f"[WhisperX] {message}", flush=True)


def run_diarization(
    audio_source: str | Mapping[str, Any],
    num_speakers: int,
    auth_token: str | None,
    cache_dir: str | None = None,
) -> list[SpeakerTurn]:
    if num_speakers <= 1 or not auth_token:
        return []

    try:
        with suppress_known_audio_stack_warnings():
            import torch
            from whisperx.diarize import DiarizationPipeline
    except Exception as exc:  # pragma: no cover - optional dependency path
        raise RuntimeError("WhisperX diarization is not installed") from exc

    with suppress_known_audio_stack_warnings():
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _emit_progress(f"Running speaker diarization for {num_speakers} speakers on {device.upper()}...")
        diarizer = DiarizationPipeline(
            token=auth_token,
            device=device,
            cache_dir=cache_dir,
        )

        if isinstance(audio_source, Mapping):
            waveform = audio_source.get("waveform")
            if waveform is None:
                raise RuntimeError("Diarization waveform was missing")
            if hasattr(waveform, "detach"):
                waveform = waveform.detach()
            if hasattr(waveform, "cpu"):
                waveform = waveform.cpu()
            if hasattr(waveform, "numpy"):
                waveform = waveform.numpy()
            if getattr(waveform, "ndim", 1) > 1:
                waveform = waveform.mean(axis=0)
            diarization_df = diarizer(waveform, num_speakers=num_speakers)
        else:
            diarization_df = diarizer(audio_source, num_speakers=num_speakers)
        _emit_progress("Speaker diarization complete.")

    turns: list[SpeakerTurn] = []
    for row in diarization_df.itertuples(index=False):
        turns.append(
            SpeakerTurn(
                start=float(row.start),
                end=float(row.end),
                label=str(row.speaker),
            )
        )
    return turns


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
