from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from .audio_io import MasterAudio

SegmentKind = Literal["speech", "music", "background"]

HOP_SECONDS = 0.1
MUSIC_MIN_SECONDS = 8.0
MUSIC_ACTIVE_FRACTION = 0.95
MUSIC_MAX_MODULATION = 0.35


@dataclass
class Segment:
    start: float
    end: float
    kind: SegmentKind


def frame_rms(audio: MasterAudio, hop_s: float = HOP_SECONDS) -> np.ndarray:
    hop = max(1, int(round(hop_s * audio.sample_rate)))
    mono = np.mean(audio.samples, axis=0)
    usable = mono[: (mono.size // hop) * hop]
    if usable.size == 0:
        return np.array([])
    frames = usable.reshape(-1, hop)
    return np.sqrt(np.mean(frames * frames, axis=1))


def _smooth(values: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0 or values.size == 0:
        return values
    kernel = np.ones(radius * 2 + 1) / (radius * 2 + 1)
    return np.convolve(values, kernel, mode="same")


def _adaptive_threshold(values: np.ndarray) -> float:
    """Noise-floor-anchored activity threshold.

    Deliberately more permissive than the waveform-display threshold: for
    mastering decisions, treating quiet speech as background (and cutting or
    refusing to level it) is far worse than letting some noise count as
    speech, especially since classification runs after denoising.
    """
    if values.size == 0:
        return 1.0
    # p5, not p20: the noise floor must come from the very quietest frames,
    # or files with little silence get their quietest speaker misread as noise.
    noise_floor = float(np.percentile(values, 5.0))
    speech_level = float(np.percentile(values, 90.0))
    if speech_level <= 0:
        return 1.0
    threshold = max(
        noise_floor * 2.0,
        noise_floor + (speech_level - noise_floor) * 0.05,
        speech_level * 0.03,
        0.0005,
    )
    return min(threshold, max(speech_level * 0.82, 0.0005))


def _modulation_index(rms: np.ndarray) -> float:
    """Envelope fluctuation of an active region; speech is highly modulated."""
    if rms.size < 4:
        return 1.0
    mean = float(np.mean(rms))
    if mean <= 0:
        return 1.0
    return float(np.std(rms) / mean)


def classify_segments(audio: MasterAudio, hop_s: float = HOP_SECONDS) -> list[Segment]:
    rms = frame_rms(audio, hop_s)
    if rms.size == 0:
        return []

    smoothed = _smooth(rms, radius=2)
    threshold = _adaptive_threshold(smoothed)
    off_threshold = threshold * 0.7
    active = smoothed >= off_threshold

    segments: list[Segment] = []
    index = 0
    total = active.size
    while index < total:
        state = bool(active[index])
        start = index
        while index < total and bool(active[index]) == state:
            index += 1
        start_s = start * hop_s
        end_s = index * hop_s
        if not state:
            segments.append(Segment(start=start_s, end=end_s, kind="background"))
            continue

        region = smoothed[start:index]
        duration = end_s - start_s
        strongly_active = float(np.mean(region >= threshold)) if region.size else 0.0
        # Modulation must come from the raw envelope: smoothing erases the
        # syllabic fluctuation that distinguishes speech from music.
        if (
            duration >= MUSIC_MIN_SECONDS
            and strongly_active >= MUSIC_ACTIVE_FRACTION
            and _modulation_index(rms[start:index]) <= MUSIC_MAX_MODULATION
        ):
            kind: SegmentKind = "music"
        else:
            kind = "speech"
        segments.append(Segment(start=start_s, end=end_s, kind=kind))

    return segments
