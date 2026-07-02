import unittest

import numpy as np

from backend.app.mastering.audio_io import MasterAudio
from backend.app.mastering.classify import Segment, classify_segments
from backend.app.mastering.leveler import apply_gain_curve, compute_leveler_gain
from backend.app.mastering.loudness import SILENCE_LUFS, short_term_loudness

SAMPLE_RATE = 48000


def speech_like(seconds: float, amplitude: float, seed: int) -> np.ndarray:
    """Amplitude-modulated noise, roughly syllabic at 4 Hz."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(seconds * SAMPLE_RATE)) / SAMPLE_RATE
    modulation = 0.55 + 0.45 * np.sin(2.0 * np.pi * 4.0 * t)
    return (amplitude * modulation * rng.standard_normal(t.size)).astype(np.float32)


def two_speaker_audio() -> tuple[MasterAudio, list[Segment]]:
    loud = speech_like(8.0, 0.4, seed=1)
    gap = np.zeros(SAMPLE_RATE * 2, dtype=np.float32)
    quiet = speech_like(8.0, 0.05, seed=2)
    samples = np.concatenate([loud, gap, quiet])
    audio = MasterAudio(samples=samples[np.newaxis, :], sample_rate=SAMPLE_RATE)
    segments = [
        Segment(start=0.0, end=8.0, kind="speech"),
        Segment(start=8.0, end=10.0, kind="background"),
        Segment(start=10.0, end=18.0, kind="speech"),
    ]
    return audio, segments


class ClassifyTests(unittest.TestCase):
    def test_speech_and_background_are_separated(self) -> None:
        audio, _ = two_speaker_audio()
        segments = classify_segments(audio)
        kinds = {segment.kind for segment in segments}
        self.assertIn("speech", kinds)
        self.assertIn("background", kinds)

    def test_quiet_speaker_is_not_background(self) -> None:
        # A speaker ~8x quieter than the other must still register as active,
        # otherwise silence cutting would delete them from the episode.
        loud = speech_like(6.0, 0.30, seed=4)
        gap = np.zeros(SAMPLE_RATE * 2, dtype=np.float32)
        rng = np.random.default_rng(9)
        noise_floor = (0.005 * rng.standard_normal(SAMPLE_RATE * 2)).astype(np.float32)
        quiet = speech_like(6.0, 0.04, seed=5)
        samples = np.concatenate([loud, gap + noise_floor, quiet])
        audio = MasterAudio(samples=samples[np.newaxis, :], sample_rate=SAMPLE_RATE)
        segments = classify_segments(audio)
        active_spans = [(s.start, s.end) for s in segments if s.kind != "background"]
        # The quiet speaker's region (8-14 s) must be covered by active spans.
        covered = sum(max(0.0, min(end, 14.0) - max(start, 8.5)) for start, end in active_spans)
        self.assertGreater(covered, 4.5)

    def test_sustained_flat_tone_is_music(self) -> None:
        t = np.arange(SAMPLE_RATE * 12) / SAMPLE_RATE
        chord = sum(np.sin(2.0 * np.pi * f * t) for f in (220.0, 277.2, 329.6))
        samples = (0.2 * chord / 3.0).astype(np.float32)
        audio = MasterAudio(samples=samples[np.newaxis, :], sample_rate=SAMPLE_RATE)
        segments = classify_segments(audio)
        self.assertTrue(any(segment.kind == "music" for segment in segments))


class LevelerTests(unittest.TestCase):
    def test_leveler_reduces_spread_between_speakers(self) -> None:
        audio, segments = two_speaker_audio()

        def speech_spread(current: MasterAudio) -> float:
            stl = short_term_loudness(current)
            first = np.median(stl[10:60])
            second = np.median(stl[110:160])
            return float(abs(first - second))

        spread_before = speech_spread(audio)
        gains = compute_leveler_gain(audio, segments, strength="moderate")
        leveled = apply_gain_curve(audio, gains)
        spread_after = speech_spread(leveled)
        self.assertLess(spread_after, spread_before * 0.6)

    def test_background_is_never_boosted(self) -> None:
        audio, segments = two_speaker_audio()
        gains = compute_leveler_gain(audio, segments, strength="tight")
        background_frames = gains[82:98]  # inside the 8-10 s gap
        self.assertTrue(np.all(background_frames <= 0.5))

    def test_strength_clamps_boost(self) -> None:
        audio, segments = two_speaker_audio()
        gains = compute_leveler_gain(audio, segments, strength="soft")
        self.assertLessEqual(float(np.max(gains)), 6.0 + 1e-6)

    def test_silent_input_produces_no_gain(self) -> None:
        samples = np.zeros((1, SAMPLE_RATE * 4), dtype=np.float32)
        audio = MasterAudio(samples=samples, sample_rate=SAMPLE_RATE)
        gains = compute_leveler_gain(audio, [Segment(start=0.0, end=4.0, kind="background")], strength="moderate")
        self.assertTrue(np.all(gains <= 0.0) if gains.size else True)


if __name__ == "__main__":
    unittest.main()
