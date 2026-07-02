from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import iirnotch, sosfiltfilt, tf2sos, welch

from .audio_io import MasterAudio

HUM_CANDIDATES = (50.0, 60.0)
HUM_PROMINENCE_DB = 10.0
HUM_SEARCH_TOLERANCE_HZ = 2.0


@dataclass
class HumProfile:
    base_frequency: float
    harmonics: list[float]


def _band_level_db(frequencies: np.ndarray, psd: np.ndarray, center: float, width: float) -> float:
    mask = (frequencies >= center - width) & (frequencies <= center + width)
    if not np.any(mask):
        return -200.0
    return float(10.0 * np.log10(np.maximum(np.max(psd[mask]), 1e-20)))


def _peak_prominence_db(frequencies: np.ndarray, psd: np.ndarray, center: float) -> float:
    """Peak level near `center` relative to the surrounding spectrum."""
    peak = _band_level_db(frequencies, psd, center, HUM_SEARCH_TOLERANCE_HZ)
    surround_mask = (
        (frequencies >= center - 12.0)
        & (frequencies <= center + 12.0)
        & ((frequencies < center - 4.0) | (frequencies > center + 4.0))
    )
    if not np.any(surround_mask):
        return 0.0
    surround = float(10.0 * np.log10(np.maximum(np.median(psd[surround_mask]), 1e-20)))
    return peak - surround


def detect_hum(
    audio: MasterAudio,
    candidates: tuple[float, ...] = HUM_CANDIDATES,
    max_harmonics: int = 8,
) -> HumProfile | None:
    mono = np.mean(audio.samples, axis=0)
    if mono.size < audio.sample_rate:
        return None

    # Welch PSD with ~1 Hz resolution to resolve the 50/60 Hz split.
    nperseg = min(mono.size, audio.sample_rate)
    frequencies, psd = welch(mono, fs=audio.sample_rate, nperseg=nperseg)

    best: tuple[float, float] | None = None
    for candidate in candidates:
        prominence = _peak_prominence_db(frequencies, psd, candidate)
        if prominence >= HUM_PROMINENCE_DB and (best is None or prominence > best[1]):
            best = (candidate, prominence)
    if best is None:
        return None

    base = best[0]
    harmonics = [base]
    for order in range(2, max_harmonics + 1):
        frequency = base * order
        if frequency >= audio.sample_rate / 2.0 - 100.0:
            break
        if _peak_prominence_db(frequencies, psd, frequency) >= HUM_PROMINENCE_DB * 0.6:
            harmonics.append(frequency)
    return HumProfile(base_frequency=base, harmonics=harmonics)


def remove_hum(audio: MasterAudio, profile: HumProfile, q: float = 35.0) -> MasterAudio:
    for frequency in profile.harmonics:
        b, a = iirnotch(frequency, q, fs=audio.sample_rate)
        sos = tf2sos(b, a)
        audio.samples = sosfiltfilt(sos, audio.samples, axis=1).astype(np.float32)
    return audio
