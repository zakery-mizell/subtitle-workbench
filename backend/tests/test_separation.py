from __future__ import annotations

import unittest

import numpy as np

from backend.app.diarization import SpeakerTurn
from backend.app.separation import blend
from backend.app.separation.overlap import (
    find_overlap_regions,
    find_solo_spans,
    pick_enrollment_span,
)


def turns_two_speaker_overlap() -> list[SpeakerTurn]:
    # A: 0-8 solo, B: 8.4-16 solo, then both speak 16.5-22, then A alone to 24.
    return [
        SpeakerTurn(start=0.0, end=8.0, label="0"),
        SpeakerTurn(start=8.4, end=16.0, label="1"),
        SpeakerTurn(start=16.5, end=22.0, label="0"),
        SpeakerTurn(start=16.5, end=22.0, label="1"),
        SpeakerTurn(start=22.0, end=24.0, label="0"),
    ]


class OverlapDetectionTests(unittest.TestCase):
    def test_finds_two_speaker_overlap(self) -> None:
        regions = find_overlap_regions(turns_two_speaker_overlap())
        self.assertEqual(len(regions), 1)
        region = regions[0]
        self.assertAlmostEqual(region.start, 16.5, places=2)
        self.assertAlmostEqual(region.end, 22.0, places=2)
        self.assertEqual(region.speaker_indices, [0, 1])

    def test_ignores_sub_threshold_blips(self) -> None:
        turns = [
            SpeakerTurn(start=0.0, end=10.0, label="0"),
            SpeakerTurn(start=5.0, end=5.2, label="1"),  # 200 ms backchannel
        ]
        self.assertEqual(find_overlap_regions(turns), [])

    def test_merges_nearby_overlaps(self) -> None:
        turns = [
            SpeakerTurn(start=0.0, end=10.0, label="0"),
            SpeakerTurn(start=2.0, end=3.0, label="1"),
            SpeakerTurn(start=3.2, end=4.2, label="1"),  # gap 0.2 < merge_gap
        ]
        regions = find_overlap_regions(turns)
        self.assertEqual(len(regions), 1)
        self.assertAlmostEqual(regions[0].start, 2.0, places=2)
        self.assertAlmostEqual(regions[0].end, 4.2, places=2)

    def test_no_overlap_for_single_speaker(self) -> None:
        self.assertEqual(find_overlap_regions([SpeakerTurn(start=0.0, end=5.0, label="0")]), [])
        self.assertEqual(find_overlap_regions([]), [])

    def test_three_speaker_overlap_lists_all(self) -> None:
        turns = [
            SpeakerTurn(start=0.0, end=6.0, label="0"),
            SpeakerTurn(start=1.0, end=6.0, label="1"),
            SpeakerTurn(start=2.0, end=6.0, label="2"),
        ]
        regions = find_overlap_regions(turns)
        self.assertEqual(len(regions), 1)
        self.assertEqual(regions[0].speaker_indices, [0, 1, 2])


class SoloSpanTests(unittest.TestCase):
    def test_solo_spans_exclude_overlap(self) -> None:
        spans = find_solo_spans(turns_two_speaker_overlap(), 0)
        self.assertAlmostEqual(spans[0][0], 0.0, places=2)
        self.assertAlmostEqual(spans[0][1], 8.0, places=2)
        # The 16.5-22 stretch is shared, so it is not solo for speaker 0.
        self.assertTrue(all(not (start < 17.0 < end) for start, end in spans))

    def test_enrollment_prefers_nearby_long_span(self) -> None:
        span = pick_enrollment_span(turns_two_speaker_overlap(), 1, near=16.5)
        self.assertIsNotNone(span)
        start, end = span
        self.assertGreaterEqual(start, 8.4)
        self.assertLessEqual(end, 16.0)
        self.assertAlmostEqual(end - start, 5.0, places=2)
        # Nearest edge of the solo span to the overlap should be kept.
        self.assertAlmostEqual(end, 16.0, places=2)

    def test_enrollment_missing_when_speaker_never_solo(self) -> None:
        turns = [
            SpeakerTurn(start=0.0, end=5.0, label="0"),
            SpeakerTurn(start=0.0, end=5.0, label="1"),
        ]
        self.assertIsNone(pick_enrollment_span(turns, 0, near=2.0))


class BlendTests(unittest.TestCase):
    def _tone(self, seconds: float, sr: int, freq: float = 220.0) -> np.ndarray:
        t = np.arange(int(seconds * sr)) / sr
        return (0.5 * np.sin(2 * np.pi * freq * t)).astype(np.float32)

    def test_spotlight_ducks_original_and_adds_stem(self) -> None:
        sr = 48000
        samples = self._tone(10.0, sr)[None, :].copy()
        stem = self._tone(3.0, 16000, freq=440.0)
        render = blend.RegionRender(start=3.0, stem=stem, region_start=3.5, region_end=5.5)
        before_outside = samples[0, : int(3.4 * sr)].copy()
        blend.apply_spotlight(samples, sr, render, 16000, duck_db=-60.0)

        # Untouched outside the region.
        np.testing.assert_array_equal(samples[0, : int(3.4 * sr)], before_outside)
        # Inside the region the original is ducked to near silence, so what
        # remains must be the (RMS-matched) 440 Hz stem, not the 220 Hz bed.
        middle = samples[0, int(4.4 * sr) : int(4.6 * sr)]
        spectrum = np.abs(np.fft.rfft(middle))
        peak_hz = np.fft.rfftfreq(middle.size, 1 / sr)[int(np.argmax(spectrum))]
        self.assertAlmostEqual(peak_hz, 440.0, delta=15.0)

    def test_replace_swaps_region_content(self) -> None:
        sr = 48000
        samples = self._tone(10.0, sr)[None, :].copy()
        stem = self._tone(3.0, 16000, freq=880.0)
        render = blend.RegionRender(start=3.0, stem=stem, region_start=3.5, region_end=5.5)
        blend.apply_replace(samples, sr, render, 16000)

        middle = samples[0, int(4.4 * sr) : int(4.6 * sr)]
        spectrum = np.abs(np.fft.rfft(middle))
        peak_hz = np.fft.rfftfreq(middle.size, 1 / sr)[int(np.argmax(spectrum))]
        self.assertAlmostEqual(peak_hz, 880.0, delta=15.0)

    def test_region_beyond_audio_is_clamped(self) -> None:
        sr = 48000
        samples = self._tone(2.0, sr)[None, :].copy()
        stem = self._tone(3.0, 16000)
        render = blend.RegionRender(start=1.0, stem=stem, region_start=1.5, region_end=9.0)
        blend.apply_replace(samples, sr, render, 16000)  # must not raise
        self.assertEqual(samples.shape[1], int(2.0 * sr))

    def test_stereo_gets_stem_on_both_channels(self) -> None:
        sr = 48000
        samples = np.stack([self._tone(6.0, sr), self._tone(6.0, sr)]).copy()
        stem = self._tone(2.0, 16000, freq=660.0)
        render = blend.RegionRender(start=2.0, stem=stem, region_start=2.2, region_end=3.8)
        blend.apply_spotlight(samples, sr, render, 16000)
        mid = slice(int(2.8 * sr), int(3.2 * sr))
        self.assertGreater(float(np.abs(samples[0, mid]).max()), 0.0)
        np.testing.assert_array_equal(samples[0, mid], samples[1, mid])


class RegionGateTests(unittest.TestCase):
    """Gating an exported solo track to one speaker's silence-snapped regions."""

    def _tone(self, seconds: float, sr: int, freq: float = 220.0) -> np.ndarray:
        t = np.arange(int(seconds * sr)) / sr
        return (0.5 * np.sin(2 * np.pi * freq * t)).astype(np.float32)

    def test_envelope_is_one_inside_and_zero_outside(self) -> None:
        sr = 48000
        envelope = blend.region_envelope(int(10.0 * sr), sr, [(2.0, 4.0)])
        self.assertEqual(envelope.size, int(10.0 * sr))
        self.assertAlmostEqual(float(envelope[int(3.0 * sr)]), 1.0, places=6)
        self.assertEqual(float(envelope[int(1.0 * sr)]), 0.0)
        self.assertEqual(float(envelope[int(6.0 * sr)]), 0.0)
        # Edges themselves are the start of the fade, not full gain.
        self.assertLess(float(envelope[int(2.0 * sr)]), 0.1)
        self.assertLess(float(envelope[int(4.0 * sr) - 1]), 0.1)

    def test_envelope_fades_are_monotonic(self) -> None:
        sr = 48000
        fade = int(blend.DEFAULT_CROSSFADE_SECONDS * sr)
        envelope = blend.region_envelope(int(10.0 * sr), sr, [(2.0, 4.0)])
        rise = envelope[int(2.0 * sr) : int(2.0 * sr) + fade]
        fall = envelope[int(4.0 * sr) - fade : int(4.0 * sr)]
        self.assertTrue(np.all(np.diff(rise) >= 0.0))
        self.assertTrue(np.all(np.diff(fall) <= 0.0))
        self.assertAlmostEqual(float(rise[0]), 0.0, places=6)
        self.assertAlmostEqual(float(fall[-1]), 0.0, places=6)

    def test_envelope_honours_every_region_and_merges_touching_ones(self) -> None:
        sr = 48000
        envelope = blend.region_envelope(int(12.0 * sr), sr, [(1.0, 2.0), (5.0, 6.0)])
        self.assertAlmostEqual(float(envelope[int(1.5 * sr)]), 1.0, places=6)
        self.assertEqual(float(envelope[int(3.5 * sr)]), 0.0)
        self.assertAlmostEqual(float(envelope[int(5.5 * sr)]), 1.0, places=6)
        # Touching regions must not dip at the joint: one fade, not two.
        joined = blend.region_envelope(int(12.0 * sr), sr, [(1.0, 3.0), (3.0, 5.0)])
        self.assertAlmostEqual(float(joined[int(3.0 * sr)]), 1.0, places=6)

    def test_gate_silences_outside_and_keeps_length(self) -> None:
        sr = 48000
        samples = self._tone(8.0, sr)[None, :].copy()
        applied = blend.apply_region_gate(samples, sr, [(2.0, 5.0)])
        self.assertTrue(applied)
        self.assertEqual(samples.shape[1], int(8.0 * sr))
        self.assertGreater(float(np.abs(samples[0, int(3.0 * sr) : int(4.0 * sr)]).max()), 0.4)
        self.assertEqual(float(np.abs(samples[0, : int(1.9 * sr)]).max()), 0.0)
        self.assertEqual(float(np.abs(samples[0, int(5.1 * sr) :]).max()), 0.0)

    def test_gate_applies_to_every_channel(self) -> None:
        sr = 48000
        samples = np.stack([self._tone(6.0, sr), self._tone(6.0, sr, freq=330.0)]).copy()
        blend.apply_region_gate(samples, sr, [(1.0, 2.0)])
        self.assertEqual(float(np.abs(samples[0, int(3.0 * sr) :]).max()), 0.0)
        self.assertEqual(float(np.abs(samples[1, int(3.0 * sr) :]).max()), 0.0)

    def test_no_regions_leaves_samples_byte_identical(self) -> None:
        sr = 48000
        samples = self._tone(4.0, sr)[None, :].copy()
        before = samples.tobytes()
        self.assertFalse(blend.apply_region_gate(samples, sr, []))
        self.assertEqual(samples.tobytes(), before)

    def test_regions_beyond_the_audio_are_clamped(self) -> None:
        sr = 48000
        samples = self._tone(2.0, sr)[None, :].copy()
        self.assertTrue(blend.apply_region_gate(samples, sr, [(1.0, 9.0)]))  # must not raise
        self.assertEqual(samples.shape[1], int(2.0 * sr))
        self.assertGreater(float(np.abs(samples[0, int(1.5 * sr) :]).max()), 0.0)


if __name__ == "__main__":
    unittest.main()
