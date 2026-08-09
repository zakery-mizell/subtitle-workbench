from __future__ import annotations

"""Persistent voice profiles used to name anonymous diarization clusters.

Speaker diarization can tell voices apart, but its cluster labels carry no
identity.  A saved profile supplies that missing identity so a future recording
can be processed from the original mix alone.
"""

import hashlib
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np
from scipy.optimize import linear_sum_assignment

from .config import settings
from .diarization import SpeakerTurn
from .separation.speaker_verify import _embedding

SAMPLE_RATE = 16_000
REFERENCE_SECONDS = 30
MIN_REFERENCE_SECONDS = 3.0
MIN_MATCH_SIMILARITY = 0.45
MIN_MATCH_MARGIN = 0.10


@dataclass(frozen=True, slots=True)
class SpeakerProfileMatch:
    name: str
    cluster_label: str
    speaker_index: int
    similarity: float


def _profile_key(name: str) -> str:
    normalized = name.strip().casefold()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:20]


def _profile_path(name: str) -> Path:
    return Path(settings.speaker_profile_dir) / f"{_profile_key(name)}.npz"


def _decode_mono_16k(source_path: str) -> np.ndarray:
    command = [
        "ffmpeg",
        "-v",
        "error",
        "-i",
        source_path,
        "-vn",
        "-ac",
        "1",
        "-ar",
        str(SAMPLE_RATE),
        "-f",
        "f32le",
        "-acodec",
        "pcm_f32le",
        "pipe:1",
    ]
    try:
        completed = subprocess.run(command, check=True, capture_output=True)
    except FileNotFoundError as exc:
        raise RuntimeError("FFmpeg is required to learn a speaker voice.") from exc
    except subprocess.CalledProcessError as exc:
        detail = exc.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"Could not decode the voice reference. Details: {detail}") from exc
    audio = np.frombuffer(completed.stdout, dtype=np.float32)
    if audio.size == 0:
        raise RuntimeError("The voice reference contained no decodable audio.")
    return audio.copy()


def _strongest_reference_audio(audio: np.ndarray) -> np.ndarray:
    """Keep the strongest one-second frames, discarding gated digital silence."""
    frame = SAMPLE_RATE
    frames = [audio[start : start + frame] for start in range(0, audio.size - frame + 1, frame)]
    ranked = sorted(
        frames,
        key=lambda samples: float(np.sqrt(np.mean(np.square(samples), dtype=np.float64))),
        reverse=True,
    )
    voiced = [
        samples
        for samples in ranked[:REFERENCE_SECONDS]
        if float(np.sqrt(np.mean(np.square(samples), dtype=np.float64))) > 1e-4
    ]
    if sum(samples.size for samples in voiced) < int(MIN_REFERENCE_SECONDS * SAMPLE_RATE):
        raise RuntimeError("The voice reference needs at least three seconds of clear speech.")
    return np.concatenate(voiced)


def enroll_speaker_profile(name: str, source_path: str) -> str:
    clean_name = name.strip()
    if not clean_name:
        raise ValueError("Speaker name must not be blank.")
    embedding = _embedding(_strongest_reference_audio(_decode_mono_16k(source_path)))
    output_dir = Path(settings.speaker_profile_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    destination = _profile_path(clean_name)
    with NamedTemporaryFile(dir=output_dir, suffix=".npz", delete=False) as handle:
        temp_path = Path(handle.name)
    try:
        np.savez(temp_path, name=np.asarray(clean_name), embedding=embedding.astype(np.float32))
        os.replace(temp_path, destination)
    finally:
        temp_path.unlink(missing_ok=True)
    return clean_name


def list_speaker_profiles() -> list[str]:
    directory = Path(settings.speaker_profile_dir)
    if not directory.is_dir():
        return []
    names: list[str] = []
    for path in directory.glob("*.npz"):
        try:
            with np.load(path, allow_pickle=False) as payload:
                name = str(payload["name"].item()).strip()
            if name:
                names.append(name)
        except Exception:
            continue
    return sorted(set(names), key=str.casefold)


def _load_requested_profiles(names: list[str]) -> dict[int, tuple[str, np.ndarray]]:
    profiles: dict[int, tuple[str, np.ndarray]] = {}
    for index, name in enumerate(names):
        path = _profile_path(name)
        if not path.is_file():
            continue
        try:
            with np.load(path, allow_pickle=False) as payload:
                stored_name = str(payload["name"].item()).strip()
                vector = np.asarray(payload["embedding"], dtype=np.float32)
            norm = float(np.linalg.norm(vector))
            if stored_name.casefold() == name.strip().casefold() and norm > 1e-8:
                profiles[index] = (stored_name, vector / norm)
        except Exception:
            continue
    return profiles


def _cluster_reference_audio(audio: np.ndarray, turns: list[SpeakerTurn], label: str) -> np.ndarray | None:
    pieces: list[np.ndarray] = []
    total = 0
    for turn in sorted((turn for turn in turns if turn.label == label), key=lambda turn: turn.end - turn.start, reverse=True):
        start = max(0, int(turn.start * SAMPLE_RATE))
        end = min(audio.size, int(turn.end * SAMPLE_RATE))
        if end - start < SAMPLE_RATE // 2:
            continue
        pieces.append(audio[start:end])
        total += end - start
        if total >= REFERENCE_SECONDS * SAMPLE_RATE:
            break
    if total < int(MIN_REFERENCE_SECONDS * SAMPLE_RATE):
        return None
    return np.concatenate(pieces)[: REFERENCE_SECONDS * SAMPLE_RATE]


def choose_profile_mapping(
    similarities: np.ndarray,
    profile_indices: list[int],
    cluster_labels: list[str],
    names: list[str],
) -> list[SpeakerProfileMatch]:
    """Choose a one-to-one profile/cluster mapping, rejecting weak matches."""
    if similarities.size == 0:
        return []
    rows, columns = linear_sum_assignment(-similarities)
    matches: list[SpeakerProfileMatch] = []
    for row, column in zip(rows.tolist(), columns.tolist()):
        score = float(similarities[row, column])
        alternatives = np.delete(similarities[row], column)
        runner_up = float(np.max(alternatives)) if alternatives.size else -1.0
        if score < MIN_MATCH_SIMILARITY or score - runner_up < MIN_MATCH_MARGIN:
            continue
        speaker_index = profile_indices[row]
        matches.append(
            SpeakerProfileMatch(
                name=names[speaker_index],
                cluster_label=cluster_labels[column],
                speaker_index=speaker_index,
                similarity=round(score, 3),
            )
        )
    return matches


def apply_speaker_profiles(
    source_path: str,
    exclusive_turns: list[SpeakerTurn],
    raw_turns: list[SpeakerTurn],
    requested_names: list[str],
) -> tuple[list[SpeakerTurn], list[SpeakerTurn], list[SpeakerProfileMatch]]:
    """Relabel diarization clusters to requested speaker order using saved voices."""
    profiles = _load_requested_profiles(requested_names)
    cluster_labels = sorted({turn.label for turn in raw_turns}, key=lambda label: int(label))
    if not profiles or not cluster_labels:
        return exclusive_turns, raw_turns, []

    audio = _decode_mono_16k(source_path)
    cluster_embeddings: dict[str, np.ndarray] = {}
    for label in cluster_labels:
        reference = _cluster_reference_audio(audio, raw_turns, label)
        if reference is not None:
            cluster_embeddings[label] = _embedding(reference)
    usable_labels = [label for label in cluster_labels if label in cluster_embeddings]
    if not usable_labels:
        return exclusive_turns, raw_turns, []

    profile_indices = sorted(profiles)
    similarities = np.asarray(
        [
            [float(np.dot(profiles[index][1], cluster_embeddings[label])) for label in usable_labels]
            for index in profile_indices
        ],
        dtype=np.float32,
    )
    matches = choose_profile_mapping(similarities, profile_indices, usable_labels, requested_names)
    if not matches:
        return exclusive_turns, raw_turns, []

    label_map = {match.cluster_label: str(match.speaker_index) for match in matches}
    used_targets = {match.speaker_index for match in matches}
    remaining_targets = [index for index in range(len(requested_names)) if index not in used_targets]
    remaining_labels = [label for label in cluster_labels if label not in label_map]
    for label, target in zip(remaining_labels, remaining_targets):
        label_map[label] = str(target)

    def relabel(turns: list[SpeakerTurn]) -> list[SpeakerTurn]:
        return [
            SpeakerTurn(start=turn.start, end=turn.end, label=label_map.get(turn.label, turn.label))
            for turn in turns
        ]

    return relabel(exclusive_turns), relabel(raw_turns), matches
