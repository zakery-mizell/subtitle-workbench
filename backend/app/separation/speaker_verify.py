from __future__ import annotations

"""Original-audio speaker verification for short, easily missed replies."""

import threading
from pathlib import Path

import numpy as np

from ..config import settings
from ..diarization import SpeakerTurn
from ..runtime_warnings import suppress_known_audio_stack_warnings
from .handoff import isolated_utterances
from .overlap import pick_enrollment_span
from .schemas import AuditWordInput, HandoffCorrection

MIN_WINNER_SIMILARITY = 0.20
MIN_RUNNER_UP_MARGIN = 0.10
MIN_CURRENT_SPEAKER_MARGIN = 0.15
VERIFY_CONTEXT_SECONDS = 0.10

_lock = threading.Lock()
_run_lock = threading.Lock()
_inference = None


class SpeakerVerificationUnavailable(RuntimeError):
    pass


def _load_inference():
    global _inference
    with _lock:
        if _inference is not None:
            return _inference
        snapshots = (
            Path(settings.whisper_cache_dir)
            / "models--pyannote--speaker-diarization-community-1"
            / "snapshots"
        )
        checkpoints = sorted(snapshots.glob("*/embedding/pytorch_model.bin"))
        if not checkpoints:
            raise SpeakerVerificationUnavailable(
                "The cached pyannote speaker-embedding model was not found."
            )
        try:
            with suppress_known_audio_stack_warnings():
                from pyannote.audio import Inference, Model

                model = Model.from_pretrained(checkpoints[-1])
            if model is None:
                raise RuntimeError("the embedding checkpoint could not be loaded")
            _inference = Inference(model, window="whole")
        except Exception as exc:
            raise SpeakerVerificationUnavailable(str(exc)) from exc
        return _inference


def _embedding(audio: np.ndarray) -> np.ndarray:
    if audio.size < 320:
        raise ValueError("speaker-verification audio is too short")
    import torch

    inference = _load_inference()
    waveform = torch.from_numpy(np.ascontiguousarray(audio, dtype=np.float32))[None, :]
    with _run_lock, suppress_known_audio_stack_warnings():
        vector = np.asarray(inference({"waveform": waveform, "sample_rate": 16000}), dtype=np.float32)
    norm = float(np.linalg.norm(vector))
    if not np.isfinite(norm) or norm <= 1e-8:
        raise ValueError("speaker-verification embedding was empty")
    return vector / norm


def choose_verified_speaker(
    scores: dict[int, float],
    current_speaker: int,
) -> tuple[int, float] | None:
    """Return one clear alternative winner and a normalized confidence."""
    if current_speaker not in scores or len(scores) < 2:
        return None
    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    winner, winner_score = ranked[0]
    runner_up_score = ranked[1][1]
    current_score = scores[current_speaker]
    if (
        winner == current_speaker
        or winner_score < MIN_WINNER_SIMILARITY
        or winner_score - runner_up_score < MIN_RUNNER_UP_MARGIN
        or winner_score - current_score < MIN_CURRENT_SPEAKER_MARGIN
    ):
        return None
    confidence = min(
        1.0,
        max(
            0.0,
            0.55
            + 0.55 * (winner_score - current_score)
            + 0.35 * (winner_score - runner_up_score),
        ),
    )
    return winner, round(confidence, 3)


def verify_isolated_utterance_speakers(
    words: list[AuditWordInput],
    mixture_16k: np.ndarray,
    turns: list[SpeakerTurn],
) -> list[HandoffCorrection]:
    """Verify pause-bounded short utterances against clean original enrollments."""
    candidates = isolated_utterances(words)
    speakers = sorted({turn.label for turn in turns}, key=int)
    if not candidates or len(speakers) < 2:
        return []

    duration = mixture_16k.size / 16000.0
    corrections: list[HandoffCorrection] = []
    enrollment_cache: dict[tuple[float, float], np.ndarray] = {}
    for candidate in candidates:
        enrollment_embeddings: dict[int, np.ndarray] = {}
        near = (candidate.start + candidate.end) / 2.0
        for raw_speaker in speakers:
            speaker = int(raw_speaker)
            enrollment = pick_enrollment_span(turns, speaker, near=near)
            if enrollment is None:
                continue
            if enrollment not in enrollment_cache:
                enroll_start, enroll_end = enrollment
                enrollment_cache[enrollment] = _embedding(
                    mixture_16k[int(enroll_start * 16000) : int(enroll_end * 16000)]
                )
            enrollment_embeddings[speaker] = enrollment_cache[enrollment]

        start = max(0.0, candidate.start - VERIFY_CONTEXT_SECONDS)
        end = min(duration, candidate.end + VERIFY_CONTEXT_SECONDS)
        probe = _embedding(mixture_16k[int(start * 16000) : int(end * 16000)])
        scores = {
            speaker: float(np.dot(probe, enrollment))
            for speaker, enrollment in enrollment_embeddings.items()
        }
        chosen = choose_verified_speaker(scores, candidate.speaker_index)
        if chosen is None:
            continue
        speaker, confidence = chosen
        for word_id in candidate.word_ids:
            corrections.append(
                HandoffCorrection(
                    word_id=word_id,
                    from_speaker_index=candidate.speaker_index,
                    to_speaker_index=speaker,
                    confidence=confidence,
                    boundary_time=round((candidate.start + candidate.end) / 2.0, 3),
                )
            )
    return corrections
