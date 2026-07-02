import unittest

import numpy as np

from backend.app.mastering.audio_io import MasterAudio
from backend.app.mastering.dynamics import soft_knee_compressor, true_peak_limiter
from backend.app.mastering.loudness import true_peak_dbtp

SAMPLE_RATE = 48000


def sine(frequency: float, seconds: float, amplitude: float) -> MasterAudio:
    t = np.arange(int(seconds * SAMPLE_RATE)) / SAMPLE_RATE
    wave = (amplitude * np.sin(2.0 * np.pi * frequency * t)).astype(np.float32)
    return MasterAudio(samples=wave[np.newaxis, :], sample_rate=SAMPLE_RATE)


class TruePeakLimiterTests(unittest.TestCase):
    def test_limiter_holds_ceiling(self) -> None:
        audio = sine(997.0, 3.0, 0.99)
        limited = true_peak_limiter(audio, ceiling_dbtp=-3.0)
        self.assertLessEqual(true_peak_dbtp(limited), -2.8)

    def test_limiter_leaves_quiet_audio_untouched(self) -> None:
        audio = sine(997.0, 2.0, 0.05)
        original = audio.samples.copy()
        limited = true_peak_limiter(audio, ceiling_dbtp=-1.0)
        np.testing.assert_array_equal(limited.samples, original)

    def test_limiter_handles_inter_sample_peaks(self) -> None:
        t = np.arange(SAMPLE_RATE * 2) / SAMPLE_RATE
        wave = (0.99 * np.sin(2.0 * np.pi * 11997.0 * t + 0.3)).astype(np.float32)
        audio = MasterAudio(samples=wave[np.newaxis, :], sample_rate=SAMPLE_RATE)
        limited = true_peak_limiter(audio, ceiling_dbtp=-1.0)
        self.assertLessEqual(true_peak_dbtp(limited), -0.7)

    def test_limiter_only_ducks_the_loud_section(self) -> None:
        quiet = sine(500.0, 1.0, 0.1).samples
        loud = sine(500.0, 1.0, 0.98).samples
        audio = MasterAudio(samples=np.concatenate([quiet, loud], axis=1), sample_rate=SAMPLE_RATE)
        limited = true_peak_limiter(audio, ceiling_dbtp=-6.0)
        # The first half (away from the lookahead boundary) keeps its level.
        first_half = limited.samples[0, : SAMPLE_RATE // 2]
        self.assertAlmostEqual(float(np.max(np.abs(first_half))), 0.1, delta=0.005)


class CompressorTests(unittest.TestCase):
    def test_compressor_reduces_level_difference(self) -> None:
        quiet = sine(500.0, 1.0, 0.05).samples
        loud = sine(500.0, 1.0, 0.8).samples
        audio = MasterAudio(samples=np.concatenate([quiet, loud], axis=1), sample_rate=SAMPLE_RATE)

        def level_diff_db(samples: np.ndarray) -> float:
            first = np.sqrt(np.mean(samples[0, : SAMPLE_RATE // 2] ** 2))
            second = np.sqrt(np.mean(samples[0, -SAMPLE_RATE // 2 :] ** 2))
            return 20.0 * float(np.log10(second / first))

        before = level_diff_db(audio.samples)
        compressed = soft_knee_compressor(audio, threshold_db=-20.0, ratio=2.0)
        after = level_diff_db(compressed.samples)
        self.assertLess(after, before - 1.0)

    def test_unity_ratio_is_a_no_op(self) -> None:
        audio = sine(500.0, 1.0, 0.5)
        original = audio.samples.copy()
        result = soft_knee_compressor(audio, ratio=1.0)
        np.testing.assert_array_equal(result.samples, original)


if __name__ == "__main__":
    unittest.main()
