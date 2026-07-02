from __future__ import annotations

import numpy as np
import pyloudnorm
from scipy.signal import lfilter, resample_poly

from .audio_io import MasterAudio
from .schemas import LoudnessStats

SILENCE_LUFS = -70.0
TRUE_PEAK_OVERSAMPLE = 4


def _k_weight(audio: MasterAudio) -> np.ndarray:
    """Apply the ITU-R BS.1770 K-weighting pre-filter chain per channel."""
    meter = pyloudnorm.Meter(audio.sample_rate)
    weighted = audio.samples.astype(np.float64)
    for stage in meter._filters.values():  # noqa: SLF001 - pyloudnorm keeps coefficients here
        weighted = lfilter(stage.b, stage.a, weighted, axis=1)
    return weighted


def _windowed_loudness(weighted: np.ndarray, sample_rate: int, window_s: float, hop_s: float) -> np.ndarray:
    """Loudness (LUFS) of overlapping windows, vectorized via cumulative sums."""
    window = int(round(window_s * sample_rate))
    hop = int(round(hop_s * sample_rate))
    n = weighted.shape[1]
    if n < window:
        window = n
    if window <= 0:
        return np.array([])

    # Channel-summed squared signal (unity channel weights for mono/stereo).
    squared = np.sum(weighted * weighted, axis=0)
    cumulative = np.concatenate(([0.0], np.cumsum(squared)))
    starts = np.arange(0, n - window + 1, hop)
    mean_square = (cumulative[starts + window] - cumulative[starts]) / window
    return -0.691 + 10.0 * np.log10(np.maximum(mean_square, 1e-12))


def short_term_loudness(audio: MasterAudio, window_s: float = 3.0, hop_s: float = 0.1) -> np.ndarray:
    return _windowed_loudness(_k_weight(audio), audio.sample_rate, window_s, hop_s)


def integrated_loudness(audio: MasterAudio) -> float:
    meter = pyloudnorm.Meter(audio.sample_rate)
    value = meter.integrated_loudness(np.ascontiguousarray(audio.samples.T, dtype=np.float64))
    if not np.isfinite(value):
        return SILENCE_LUFS
    return float(max(value, SILENCE_LUFS))


def loudness_range(short_term: np.ndarray) -> float:
    """EBU R128 / Tech 3342 loudness range from short-term loudness values."""
    if short_term.size == 0:
        return 0.0
    above_absolute = short_term[short_term > SILENCE_LUFS]
    if above_absolute.size == 0:
        return 0.0
    # Relative gate: -20 LU below the power-mean of absolutely gated values.
    power_mean = 10.0 * np.log10(np.mean(np.power(10.0, above_absolute / 10.0)))
    gated = above_absolute[above_absolute > power_mean - 20.0]
    if gated.size < 2:
        return 0.0
    low, high = np.percentile(gated, [10.0, 95.0])
    return float(max(0.0, high - low))


def true_peak_dbtp(audio: MasterAudio) -> float:
    peak = 0.0
    for channel in audio.samples:
        oversampled = resample_poly(channel.astype(np.float64), TRUE_PEAK_OVERSAMPLE, 1)
        peak = max(peak, float(np.max(np.abs(oversampled)))) if oversampled.size else peak
    if peak <= 0.0:
        return -120.0
    return float(20.0 * np.log10(peak))


def noise_floor_dbfs(audio: MasterAudio, frame_s: float = 0.1) -> float:
    frame = max(1, int(round(frame_s * audio.sample_rate)))
    mono = np.mean(audio.samples, axis=0)
    usable = mono[: (mono.size // frame) * frame]
    if usable.size == 0:
        return -120.0
    frames = usable.reshape(-1, frame)
    rms = np.sqrt(np.mean(frames * frames, axis=1))
    floor = float(np.percentile(rms[rms > 0], 10.0)) if np.any(rms > 0) else 0.0
    if floor <= 0.0:
        return -120.0
    return float(max(-120.0, 20.0 * np.log10(floor)))


def measure_loudness(audio: MasterAudio) -> LoudnessStats:
    return LoudnessStats(
        integrated_lufs=round(integrated_loudness(audio), 2),
        lra=round(loudness_range(short_term_loudness(audio)), 2),
        true_peak_dbtp=round(true_peak_dbtp(audio), 2),
        noise_floor_dbfs=round(noise_floor_dbfs(audio), 2),
    )


def apply_gain_db(audio: MasterAudio, gain_db: float) -> MasterAudio:
    audio.samples *= np.float32(np.power(10.0, gain_db / 20.0))
    return audio


def normalize_loudness(
    audio: MasterAudio,
    target_lufs: float,
    true_peak_ceiling_dbtp: float,
    max_iterations: int = 3,
    tolerance_lu: float = 0.5,
) -> MasterAudio:
    """Measure → gain → limit → re-measure until within tolerance of target."""
    from .dynamics import true_peak_limiter

    for _ in range(max_iterations):
        measured = integrated_loudness(audio)
        if measured <= SILENCE_LUFS:
            break
        offset = target_lufs - measured
        if abs(offset) <= tolerance_lu:
            break
        apply_gain_db(audio, offset)
        audio = true_peak_limiter(audio, ceiling_dbtp=true_peak_ceiling_dbtp)
    return audio
