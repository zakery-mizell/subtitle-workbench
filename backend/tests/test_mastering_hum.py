import unittest

import numpy as np

from backend.app.mastering.audio_io import MasterAudio
from backend.app.mastering.hum import detect_hum, remove_hum

SAMPLE_RATE = 48000


def band_rms(samples: np.ndarray, frequency: float, sample_rate: int) -> float:
    """RMS of a narrow band around `frequency` via correlation with a sine."""
    t = np.arange(samples.size) / sample_rate
    probe_sin = np.sin(2.0 * np.pi * frequency * t)
    probe_cos = np.cos(2.0 * np.pi * frequency * t)
    amplitude = 2.0 * np.hypot(np.mean(samples * probe_sin), np.mean(samples * probe_cos))
    return float(amplitude / np.sqrt(2.0))


def hum_signal(base: float, seconds: float = 6.0) -> MasterAudio:
    rng = np.random.default_rng(11)
    t = np.arange(int(seconds * SAMPLE_RATE)) / SAMPLE_RATE
    speech_like = 0.1 * rng.standard_normal(t.size)
    hum = 0.05 * np.sin(2.0 * np.pi * base * t)
    hum += 0.02 * np.sin(2.0 * np.pi * base * 2 * t)
    hum += 0.01 * np.sin(2.0 * np.pi * base * 3 * t)
    tone = 0.05 * np.sin(2.0 * np.pi * 1000.0 * t)
    samples = (speech_like + hum + tone).astype(np.float32)
    return MasterAudio(samples=samples[np.newaxis, :], sample_rate=SAMPLE_RATE)


class HumDetectionTests(unittest.TestCase):
    def test_detects_sixty_hz_hum(self) -> None:
        profile = detect_hum(hum_signal(60.0))
        self.assertIsNotNone(profile)
        self.assertEqual(profile.base_frequency, 60.0)
        self.assertGreaterEqual(len(profile.harmonics), 2)

    def test_detects_fifty_hz_hum(self) -> None:
        profile = detect_hum(hum_signal(50.0))
        self.assertIsNotNone(profile)
        self.assertEqual(profile.base_frequency, 50.0)

    def test_no_detection_without_hum(self) -> None:
        rng = np.random.default_rng(3)
        samples = (0.1 * rng.standard_normal(SAMPLE_RATE * 4)).astype(np.float32)
        audio = MasterAudio(samples=samples[np.newaxis, :], sample_rate=SAMPLE_RATE)
        self.assertIsNone(detect_hum(audio))


class HumRemovalTests(unittest.TestCase):
    def test_notches_hum_but_preserves_speech_band(self) -> None:
        audio = hum_signal(60.0)
        hum_before = band_rms(audio.samples[0], 60.0, SAMPLE_RATE)
        tone_before = band_rms(audio.samples[0], 1000.0, SAMPLE_RATE)

        profile = detect_hum(audio)
        self.assertIsNotNone(profile)
        cleaned = remove_hum(audio, profile)

        hum_after = band_rms(cleaned.samples[0], 60.0, SAMPLE_RATE)
        tone_after = band_rms(cleaned.samples[0], 1000.0, SAMPLE_RATE)

        hum_reduction_db = 20.0 * np.log10(hum_before / max(hum_after, 1e-9))
        tone_change_db = abs(20.0 * np.log10(tone_after / tone_before))
        self.assertGreater(hum_reduction_db, 20.0)
        self.assertLess(tone_change_db, 1.0)


if __name__ == "__main__":
    unittest.main()
