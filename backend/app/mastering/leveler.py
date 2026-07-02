from __future__ import annotations

import numpy as np

from .audio_io import MasterAudio
from .classify import Segment
from .loudness import SILENCE_LUFS, short_term_loudness
from .schemas import LevelerStrength

HOP_SECONDS = 0.1
MOMENTARY_WINDOW_SECONDS = 0.4
LOCAL_SMOOTH_SECONDS = 3.0

STRENGTH_CLAMPS_DB: dict[str, float] = {"tight": 12.0, "moderate": 9.0, "soft": 6.0}
MUSIC_CLAMP_DB = 3.0
GAIN_DOWN_SECONDS = 0.4
GAIN_UP_SECONDS = 2.0


def _frame_kinds(segments: list[Segment], frame_times: np.ndarray) -> np.ndarray:
    """Kind per loudness frame: 0 background, 1 speech, 2 music."""
    kinds = np.zeros(frame_times.size, dtype=np.int8)
    codes = {"background": 0, "speech": 1, "music": 2}
    for segment in segments:
        mask = (frame_times >= segment.start) & (frame_times < segment.end)
        kinds[mask] = codes[segment.kind]
    return kinds


def _segment_local_smooth(values: np.ndarray, mask: np.ndarray, radius: int) -> np.ndarray:
    """Edge-truncated moving average computed only over `mask` frames.

    Frames outside the mask contribute nothing, so silence around a segment
    never drags the measured speech loudness down.
    """
    weights = mask.astype(np.float64)
    padded = np.where(mask, values, 0.0)
    kernel = np.ones(radius * 2 + 1)
    sums = np.convolve(padded, kernel, mode="same")
    counts = np.convolve(weights, kernel, mode="same")
    out = np.full(values.size, np.nan)
    valid = counts > 0
    out[valid] = sums[valid] / counts[valid]
    return out


def _asymmetric_smooth(gains_db: np.ndarray, hop_s: float) -> np.ndarray:
    """Fast gain decrease, slow increase, so boosts never pump noise onsets."""
    down_alpha = float(np.exp(-hop_s / GAIN_DOWN_SECONDS))
    up_alpha = float(np.exp(-hop_s / GAIN_UP_SECONDS))
    smoothed = np.empty_like(gains_db)
    current = 0.0
    for index in range(gains_db.size):
        target = gains_db[index]
        alpha = down_alpha if target < current else up_alpha
        current = target + (current - target) * alpha
        smoothed[index] = current
    return smoothed


def compute_leveler_gain(
    audio: MasterAudio,
    segments: list[Segment],
    strength: LevelerStrength,
    hop_s: float = HOP_SECONDS,
) -> np.ndarray:
    """Per-frame gain (dB) that pulls speech toward uniform loudness.

    Uses momentary loudness smoothed only across nearby speech frames, so a
    quiet speaker inside a long segment is boosted while silence around
    segment edges cannot fake a low reading. Background frames are never
    boosted; music segments get one conservative flat gain.
    """
    momentary = short_term_loudness(audio, window_s=MOMENTARY_WINDOW_SECONDS, hop_s=hop_s)
    if momentary.size == 0:
        return np.array([])

    frame_times = np.arange(momentary.size) * hop_s + MOMENTARY_WINDOW_SECONDS / 2.0
    kinds = _frame_kinds(segments, frame_times)
    clamp = STRENGTH_CLAMPS_DB[strength]

    speech_mask = (kinds == 1) & (momentary > SILENCE_LUFS)
    if not np.any(speech_mask):
        return np.zeros(momentary.size)

    radius = max(1, int(round(LOCAL_SMOOTH_SECONDS / hop_s / 2.0)))
    speech_loudness = _segment_local_smooth(momentary, speech_mask, radius)
    target = float(np.median(speech_loudness[speech_mask]))

    gains = np.zeros(momentary.size)
    gains[speech_mask] = np.clip(target - speech_loudness[speech_mask], -clamp, clamp)

    # One flat, conservative gain per music segment.
    for segment in segments:
        if segment.kind != "music":
            continue
        mask = (frame_times >= segment.start) & (frame_times < segment.end)
        region = momentary[mask]
        valid = region[region > SILENCE_LUFS]
        if valid.size == 0:
            continue
        gains[mask] = float(np.clip(target - np.median(valid), -MUSIC_CLAMP_DB, MUSIC_CLAMP_DB))

    # Background frames stay at unity or follow downward gains only.
    gains[kinds == 0] = np.minimum(0.0, gains[kinds == 0])

    return _asymmetric_smooth(gains, hop_s)


def apply_gain_curve(
    audio: MasterAudio,
    gains_db: np.ndarray,
    hop_s: float = HOP_SECONDS,
    frame_offset_s: float = MOMENTARY_WINDOW_SECONDS / 2.0,
) -> MasterAudio:
    """Apply a per-frame dB gain curve; `frame_offset_s` is the window-center
    offset used when the curve was computed."""
    if gains_db.size == 0:
        return audio
    n = audio.samples.shape[1]
    frame_positions = (np.arange(gains_db.size) * hop_s + frame_offset_s) * audio.sample_rate
    gain_curve = np.power(10.0, np.interp(np.arange(n), frame_positions, gains_db) / 20.0).astype(np.float32)
    audio.samples *= gain_curve
    return audio
