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


if __name__ == "__main__":
    unittest.main()
