import unittest

import numpy as np

from backend.app.mastering.audio_io import MasterAudio
from backend.app.mastering.classify import Segment
from backend.app.mastering.cutting import (
    CutRegion,
    apply_cuts,
    build_silence_cuts,
    clamp_cuts_to_duration,
    detect_filler_regions,
    export_audacity_labels,
    merge_cut_regions,
    remap_timestamp,
    total_cut_duration,
)

SAMPLE_RATE = 48000


class SilenceCutTests(unittest.TestCase):
    def test_long_pause_is_trimmed_and_short_pause_kept(self) -> None:
        segments = [
            Segment(start=0.0, end=5.0, kind="speech"),
            Segment(start=5.0, end=5.8, kind="background"),  # short: keep
            Segment(start=5.8, end=10.0, kind="speech"),
            Segment(start=10.0, end=14.0, kind="background"),  # long: trim
            Segment(start=14.0, end=20.0, kind="speech"),
        ]
        cuts = build_silence_cuts(segments, duration=20.0, keep_pause_seconds=0.6, max_pause_seconds=1.5)
        self.assertEqual(len(cuts), 1)
        self.assertAlmostEqual(cuts[0].start, 10.3, places=3)
        self.assertAlmostEqual(cuts[0].end, 13.7, places=3)


class FillerCutTests(unittest.TestCase):
    def test_fillers_are_detected_from_words(self) -> None:
        words = [
            {"text": "So", "start": 0.0, "end": 0.3},
            {"text": "um,", "start": 0.4, "end": 0.7},
            {"text": "hello", "start": 0.9, "end": 1.3},
            {"text": "Uh", "start": 2.0, "end": 2.2},
        ]
        cuts = detect_filler_regions(words)
        self.assertEqual(len(cuts), 2)
        self.assertEqual(cuts[0].label, "um")
        self.assertAlmostEqual(cuts[0].start, 0.36, places=3)
        self.assertAlmostEqual(cuts[0].end, 0.74, places=3)
        self.assertEqual(cuts[1].label, "uh")

    def test_words_without_timestamps_are_skipped(self) -> None:
        cuts = detect_filler_regions([{"text": "um", "start": None, "end": None}])
        self.assertEqual(cuts, [])


class MergeAndRemapTests(unittest.TestCase):
    def test_overlapping_cuts_merge(self) -> None:
        cuts = [
            CutRegion(start=1.0, end=2.0, reason="silence", label="silence"),
            CutRegion(start=1.8, end=2.5, reason="filler", label="um"),
            CutRegion(start=5.0, end=6.0, reason="silence", label="silence"),
        ]
        merged = merge_cut_regions(cuts)
        self.assertEqual(len(merged), 2)
        self.assertAlmostEqual(merged[0].end, 2.5)

    def test_remap_timestamp_is_piecewise(self) -> None:
        cuts = [
            CutRegion(start=1.0, end=2.0, reason="silence", label="silence"),
            CutRegion(start=5.0, end=6.5, reason="silence", label="silence"),
        ]
        self.assertAlmostEqual(remap_timestamp(0.5, cuts), 0.5)
        self.assertAlmostEqual(remap_timestamp(1.5, cuts), 1.0)  # inside first cut
        self.assertAlmostEqual(remap_timestamp(3.0, cuts), 2.0)
        self.assertAlmostEqual(remap_timestamp(7.0, cuts), 4.5)

    def test_clamp_drops_tiny_and_out_of_range_cuts(self) -> None:
        cuts = [
            CutRegion(start=1.0, end=1.01, reason="silence", label="tiny"),
            CutRegion(start=9.5, end=12.0, reason="silence", label="tail"),
        ]
        clamped = clamp_cuts_to_duration(cuts, duration=10.0)
        self.assertEqual(len(clamped), 1)
        self.assertAlmostEqual(clamped[0].end, 10.0)


class ApplyCutsTests(unittest.TestCase):
    def test_apply_mode_shortens_audio_by_cut_duration(self) -> None:
        samples = np.ones((1, SAMPLE_RATE * 10), dtype=np.float32)
        audio = MasterAudio(samples=samples, sample_rate=SAMPLE_RATE)
        cuts = [CutRegion(start=2.0, end=4.0, reason="silence", label="silence")]
        result = apply_cuts(audio, cuts, mode="apply")
        expected = 10.0 - total_cut_duration(cuts)
        self.assertAlmostEqual(result.duration, expected, delta=0.02)

    def test_silence_mode_keeps_duration_and_zeroes_region(self) -> None:
        rng = np.random.default_rng(5)
        samples = (0.5 * rng.standard_normal(SAMPLE_RATE * 6)).astype(np.float32)[np.newaxis, :]
        audio = MasterAudio(samples=samples, sample_rate=SAMPLE_RATE)
        cuts = [CutRegion(start=2.0, end=3.0, reason="filler", label="um")]
        result = apply_cuts(audio, cuts, mode="silence")
        self.assertAlmostEqual(result.duration, 6.0, places=3)
        middle = result.samples[0, int(2.2 * SAMPLE_RATE) : int(2.8 * SAMPLE_RATE)]
        self.assertEqual(float(np.max(np.abs(middle))), 0.0)


class ExportTests(unittest.TestCase):
    def test_audacity_label_format(self) -> None:
        cuts = [CutRegion(start=1.5, end=2.25, reason="filler", label="um")]
        text = export_audacity_labels(cuts)
        self.assertEqual(text, "1.500000\t2.250000\tum\n")

    def test_empty_cut_list_exports_empty_string(self) -> None:
        self.assertEqual(export_audacity_labels([]), "")


if __name__ == "__main__":
    unittest.main()
