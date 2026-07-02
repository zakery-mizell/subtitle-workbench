import unittest

import numpy as np

from backend.app.mastering.audio_io import MasterAudio
from backend.app.mastering.loudness import (
    integrated_loudness,
    loudness_range,
    measure_loudness,
    noise_floor_dbfs,
    normalize_loudness,
    short_term_loudness,
    true_peak_dbtp,
)

SAMPLE_RATE = 48000


def sine(frequency: float, seconds: float, amplitude: float, channels: int = 1) -> MasterAudio:
    t = np.arange(int(seconds * SAMPLE_RATE)) / SAMPLE_RATE
    wave = (amplitude * np.sin(2.0 * np.pi * frequency * t)).astype(np.float32)
    return MasterAudio(samples=np.tile(wave, (channels, 1)), sample_rate=SAMPLE_RATE)


class LoudnessMeasurementTests(unittest.TestCase):
    def test_halving_amplitude_drops_loudness_six_db(self) -> None:
        loud = integrated_loudness(sine(997.0, 10.0, 0.5))
        quiet = integrated_loudness(sine(997.0, 10.0, 0.25))
        self.assertAlmostEqual(loud - quiet, 6.02, delta=0.3)

    def test_short_term_matches_integrated_for_steady_signal(self) -> None:
        audio = sine(997.0, 10.0, 0.3)
        integrated = integrated_loudness(audio)
        short_term = short_term_loudness(audio)
        self.assertGreater(short_term.size, 0)
        self.assertAlmostEqual(float(np.median(short_term)), integrated, delta=0.5)

    def test_loudness_range_of_two_level_signal(self) -> None:
        quiet = sine(997.0, 10.0, 0.05).samples
        loud = sine(997.0, 10.0, 0.5).samples
        audio = MasterAudio(samples=np.concatenate([quiet, loud], axis=1), sample_rate=SAMPLE_RATE)
        lra = loudness_range(short_term_loudness(audio))
        # The levels differ by 20 dB; gating and percentiles compress that some.
        self.assertGreater(lra, 8.0)
        self.assertLess(lra, 22.0)

    def test_loudness_range_of_steady_signal_is_near_zero(self) -> None:
        lra = loudness_range(short_term_loudness(sine(997.0, 10.0, 0.3)))
        self.assertLess(lra, 1.0)

    def test_true_peak_catches_inter_sample_peaks(self) -> None:
        # A sine near Nyquist/4 with phase offset puts true peaks between samples.
        t = np.arange(SAMPLE_RATE) / SAMPLE_RATE
        wave = (0.99 * np.sin(2.0 * np.pi * 11997.0 * t + 0.3)).astype(np.float32)
        audio = MasterAudio(samples=wave[np.newaxis, :], sample_rate=SAMPLE_RATE)
        sample_peak_db = 20.0 * np.log10(float(np.max(np.abs(wave))))
        true_peak = true_peak_dbtp(audio)
        self.assertGreaterEqual(true_peak, sample_peak_db - 0.05)

    def test_noise_floor_reflects_quiet_portions(self) -> None:
        rng = np.random.default_rng(7)
        noise = (0.001 * rng.standard_normal(SAMPLE_RATE * 2)).astype(np.float32)
        speech = (0.3 * rng.standard_normal(SAMPLE_RATE * 2)).astype(np.float32)
        audio = MasterAudio(
            samples=np.concatenate([noise, speech])[np.newaxis, :], sample_rate=SAMPLE_RATE
        )
        floor = noise_floor_dbfs(audio)
        self.assertLess(floor, -50.0)
        self.assertGreater(floor, -80.0)


class LoudnessNormalizationTests(unittest.TestCase):
    def test_normalization_hits_target_within_tolerance(self) -> None:
        audio = sine(997.0, 10.0, 0.05)
        normalized = normalize_loudness(audio, target_lufs=-16.0, true_peak_ceiling_dbtp=-1.0)
        self.assertAlmostEqual(integrated_loudness(normalized), -16.0, delta=0.5)

    def test_normalization_respects_true_peak_ceiling(self) -> None:
        # A loud signal pushed toward a hot target must still respect the ceiling.
        audio = sine(997.0, 10.0, 0.4)
        normalized = normalize_loudness(audio, target_lufs=-6.0, true_peak_ceiling_dbtp=-1.0)
        self.assertLessEqual(true_peak_dbtp(normalized), -0.9)

    def test_measure_loudness_returns_all_stats(self) -> None:
        stats = measure_loudness(sine(997.0, 5.0, 0.2))
        self.assertLess(stats.integrated_lufs, 0.0)
        self.assertGreaterEqual(stats.lra, 0.0)
        self.assertLess(stats.true_peak_dbtp, 0.0)


if __name__ == "__main__":
    unittest.main()
