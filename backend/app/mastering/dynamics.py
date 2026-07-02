from __future__ import annotations

import numpy as np
from scipy.ndimage import minimum_filter1d
from scipy.signal import resample_poly

from .audio_io import MasterAudio

LIMITER_BLOCK_SECONDS = 0.005
LIMITER_SAFETY_DB = 0.1
TRUE_PEAK_OVERSAMPLE = 4


def _block_peaks(audio: MasterAudio, block: int) -> np.ndarray:
    """Per-block true-peak estimate (4x oversampled), max across channels."""
    n = audio.samples.shape[1]
    block_count = (n + block - 1) // block
    peaks = np.zeros(block_count, dtype=np.float64)
    for channel in audio.samples:
        oversampled = np.abs(resample_poly(channel.astype(np.float64), TRUE_PEAK_OVERSAMPLE, 1))
        os_block = block * TRUE_PEAK_OVERSAMPLE
        padded_len = block_count * os_block
        padded = np.zeros(padded_len, dtype=np.float64)
        padded[: min(oversampled.size, padded_len)] = oversampled[:padded_len]
        peaks = np.maximum(peaks, padded.reshape(block_count, os_block).max(axis=1))
    return peaks


def _release_smooth(required_gain: np.ndarray, alpha: float) -> np.ndarray:
    """Instant attack (values already lookahead-minimized), exponential release."""
    smoothed = np.empty_like(required_gain)
    gain = 1.0
    for index in range(required_gain.size):
        gain = min(required_gain[index], 1.0 - (1.0 - gain) * alpha)
        smoothed[index] = gain
    return smoothed


def true_peak_limiter(
    audio: MasterAudio,
    ceiling_dbtp: float = -1.0,
    lookahead_ms: float = 5.0,
    release_ms: float = 60.0,
) -> MasterAudio:
    n = audio.samples.shape[1]
    if n == 0:
        return audio

    ceiling = float(np.power(10.0, (ceiling_dbtp - LIMITER_SAFETY_DB) / 20.0))
    block = max(1, int(round(LIMITER_BLOCK_SECONDS * audio.sample_rate)))
    peaks = _block_peaks(audio, block)
    if float(peaks.max(initial=0.0)) <= ceiling:
        return audio

    required = np.minimum(1.0, ceiling / np.maximum(peaks, 1e-12))

    lookahead_blocks = max(1, int(round((lookahead_ms / 1000.0) / LIMITER_BLOCK_SECONDS)))
    # Pull each reduction earlier so the gain is already down when the peak hits.
    required = minimum_filter1d(required, size=lookahead_blocks + 1, origin=-((lookahead_blocks + 1) // 2))

    alpha = float(np.exp(-LIMITER_BLOCK_SECONDS / max(release_ms / 1000.0, 1e-3)))
    gains = _release_smooth(required, alpha)

    block_times = (np.arange(gains.size) + 0.5) * block
    sample_positions = np.arange(n)
    gain_curve = np.interp(sample_positions, block_times, gains).astype(np.float32)
    audio.samples *= gain_curve
    return audio


def soft_knee_compressor(
    audio: MasterAudio,
    threshold_db: float = -20.0,
    ratio: float = 1.8,
    knee_db: float = 6.0,
    attack_ms: float = 15.0,
    release_ms: float = 150.0,
) -> MasterAudio:
    """Channel-linked soft-knee downward compressor on a 5 ms detector grid."""
    n = audio.samples.shape[1]
    if n == 0 or ratio <= 1.0:
        return audio

    block = max(1, int(round(LIMITER_BLOCK_SECONDS * audio.sample_rate)))
    block_count = (n + block - 1) // block
    mono = np.max(np.abs(audio.samples), axis=0)
    padded = np.zeros(block_count * block, dtype=np.float64)
    padded[:n] = mono
    detector = np.sqrt(np.mean(padded.reshape(block_count, block) ** 2, axis=1))
    level_db = 20.0 * np.log10(np.maximum(detector, 1e-9))

    # Static soft-knee gain curve.
    overshoot = level_db - threshold_db
    half_knee = knee_db / 2.0
    reduction_db = np.where(
        overshoot <= -half_knee,
        0.0,
        np.where(
            overshoot >= half_knee,
            overshoot * (1.0 / ratio - 1.0),
            (1.0 / ratio - 1.0) * (overshoot + half_knee) ** 2 / (2.0 * knee_db),
        ),
    )

    attack_alpha = float(np.exp(-LIMITER_BLOCK_SECONDS / max(attack_ms / 1000.0, 1e-3)))
    release_alpha = float(np.exp(-LIMITER_BLOCK_SECONDS / max(release_ms / 1000.0, 1e-3)))
    smoothed = np.empty_like(reduction_db)
    current = 0.0
    for index in range(reduction_db.size):
        target = reduction_db[index]
        alpha = attack_alpha if target < current else release_alpha
        current = target + (current - target) * alpha
        smoothed[index] = current

    block_times = (np.arange(smoothed.size) + 0.5) * block
    gain_curve = np.power(10.0, np.interp(np.arange(n), block_times, smoothed) / 20.0).astype(np.float32)
    audio.samples *= gain_curve
    return audio
