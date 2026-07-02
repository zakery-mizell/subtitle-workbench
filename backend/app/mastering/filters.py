from __future__ import annotations

import numpy as np
from scipy.signal import butter, sosfiltfilt, welch

from .audio_io import MasterAudio

MIN_CUTOFF_HZ = 40.0
MAX_CUTOFF_HZ = 120.0
DEFAULT_CUTOFF_HZ = 80.0


def estimate_voice_fundamental(audio: MasterAudio) -> float:
    """Estimate the dominant voice fundamental from the long-term spectrum.

    Looks for the strongest peak in the 60-320 Hz range; used to keep the
    adaptive high-pass below the speaker's fundamental.
    """
    mono = np.mean(audio.samples, axis=0)
    if mono.size < audio.sample_rate:
        return 120.0

    nperseg = min(mono.size, audio.sample_rate)
    frequencies, psd = welch(mono, fs=audio.sample_rate, nperseg=nperseg)
    mask = (frequencies >= 60.0) & (frequencies <= 320.0)
    if not np.any(mask):
        return 120.0
    return float(frequencies[mask][np.argmax(psd[mask])])


def adaptive_cutoff(audio: MasterAudio) -> float:
    fundamental = estimate_voice_fundamental(audio)
    return float(np.clip(fundamental * 0.7, MIN_CUTOFF_HZ, MAX_CUTOFF_HZ))


def high_pass(audio: MasterAudio, cutoff_hz: float, order: int = 4) -> MasterAudio:
    sos = butter(order, cutoff_hz, btype="highpass", fs=audio.sample_rate, output="sos")
    audio.samples = sosfiltfilt(sos, audio.samples, axis=1).astype(np.float32)
    return audio
