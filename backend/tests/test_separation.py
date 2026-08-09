from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from backend.app.diarization import SpeakerTurn
from backend.app.mastering.audio_io import MasterAudio, add_flac_seekpoints, encode_master

SEEKTABLE_BLOCK_TYPE = 3
from backend.app.separation import blend
from backend.app.separation.overlap import (
    find_overlap_regions,
    find_solo_spans,
    pick_enrollment_span,
    speaker_turn_spans,
)
from backend.app.separation.unise_engine import WINDOW_SAMPLES, UniSEEngine


def simultaneous_seconds(turns: list[SpeakerTurn], start: float = 0.0, end: float = 1e9) -> float:
    """Ground truth: seconds inside [start, end] where two or more speakers talk."""
    marks = sorted({turn.start for turn in turns} | {turn.end for turn in turns})
    total = 0.0
    for mark, next_mark in zip(marks, marks[1:]):
        lo, hi = max(mark, start), min(next_mark, end)
        if hi - lo <= 1e-6:
            continue
        active = {turn.label for turn in turns if turn.start <= lo + 1e-6 and turn.end >= hi - 1e-6}
        if len(active) >= 2:
            total += hi - lo
    return total


def turns_backchannels(count: int = 20) -> list[SpeakerTurn]:
    """One speaker talking straight through periodic 0.25 s "mm-hm"s from another."""
    turns = [SpeakerTurn(start=0.0, end=0.6 * count + 1.0, label="0")]
    for index in range(count):
        start = 0.5 + 0.6 * index
        turns.append(SpeakerTurn(start=round(start, 3), end=round(start + 0.25, 3), label="1"))
    return turns


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


class MergeChainTests(unittest.TestCase):
    """Merging must bridge diarization jitter without welding a whole conversation."""

    def test_backchannels_do_not_chain_into_one_region(self) -> None:
        turns = turns_backchannels()
        regions = find_overlap_regions(turns)

        self.assertGreater(len(regions), 1)
        for region in regions:
            self.assertLess(region.duration, 2.0)
            # No region may be mostly solo audio.
            inside = simultaneous_seconds(turns, region.start, region.end)
            self.assertGreaterEqual(inside / region.duration, 0.5)

        # Every genuine "mm-hm" is still accounted for, none of the solo speech is.
        found = sum(simultaneous_seconds(turns, region.start, region.end) for region in regions)
        self.assertAlmostEqual(found, simultaneous_seconds(turns), places=2)

    def test_backchannel_regions_do_not_swallow_solo_speech(self) -> None:
        turns = turns_backchannels()
        regions = find_overlap_regions(turns)
        span = sum(region.duration for region in regions)
        # The old chain reported the whole 11.65 s stretch as one overlap.
        self.assertLess(span, 10.0)

    def test_handoff_with_a_few_ms_gap_is_not_an_overlap(self) -> None:
        # Diarizer rounding, not two voices: 4 ms on either side of the handoff.
        gap = [SpeakerTurn(start=0.0, end=5.0, label="0"), SpeakerTurn(start=5.004, end=12.0, label="1")]
        self.assertEqual(find_overlap_regions(gap), [])
        touch = [SpeakerTurn(start=0.0, end=5.0, label="0"), SpeakerTurn(start=4.996, end=12.0, label="1")]
        self.assertEqual(find_overlap_regions(touch), [])

    def test_brisk_handoffs_never_chain_into_a_phantom_region(self) -> None:
        # Strict turn-taking, 4 ms of diarizer slop at each handoff. Nothing here
        # is simultaneous, but a phantom per handoff used to weld into one region.
        turns: list[SpeakerTurn] = []
        cursor = 0.0
        for index in range(10):
            turns.append(SpeakerTurn(start=round(cursor, 3), end=round(cursor + 0.3, 3), label=str(index % 2)))
            cursor += 0.304
        self.assertEqual(simultaneous_seconds(turns), 0.0)
        self.assertEqual(find_overlap_regions(turns), [])

    def test_isolated_blips_do_not_resurrect_by_chaining(self) -> None:
        turns = [SpeakerTurn(start=0.0, end=10.0, label="0")]
        for index in range(5):
            start = 2.0 + 0.3 * index
            turns.append(SpeakerTurn(start=round(start, 3), end=round(start + 0.1, 3), label="1"))
        self.assertEqual(find_overlap_regions(turns), [])

    def test_jittered_overlap_fragments_still_merge(self) -> None:
        turns = [
            SpeakerTurn(start=0.0, end=10.0, label="0"),
            SpeakerTurn(start=2.0, end=2.5, label="1"),
            SpeakerTurn(start=2.55, end=3.1, label="1"),
            SpeakerTurn(start=3.15, end=3.6, label="1"),
        ]
        regions = find_overlap_regions(turns)
        self.assertEqual(len(regions), 1)
        self.assertAlmostEqual(regions[0].start, 2.0, places=2)
        self.assertAlmostEqual(regions[0].end, 3.6, places=2)
        self.assertEqual(regions[0].speaker_indices, [0, 1])

    def test_artifact_speaker_is_left_out_of_speaker_indices(self) -> None:
        turns = [
            SpeakerTurn(start=5.0, end=6.0, label="0"),
            SpeakerTurn(start=5.0, end=6.0, label="1"),
            SpeakerTurn(start=5.4, end=5.434, label="2"),  # 34 ms diarizer artifact
        ]
        regions = find_overlap_regions(turns)
        self.assertEqual(len(regions), 1)
        self.assertEqual(regions[0].speaker_indices, [0, 1])

    def test_region_duration_uses_simultaneous_time_not_span(self) -> None:
        # Two 0.2 s overlaps 0.3 s apart: 0.4 s of speech spread over a 0.7 s span.
        turns = [
            SpeakerTurn(start=0.0, end=10.0, label="0"),
            SpeakerTurn(start=2.0, end=2.2, label="1"),
            SpeakerTurn(start=2.5, end=2.7, label="1"),
        ]
        self.assertEqual(find_overlap_regions(turns, min_duration=0.5), [])
        self.assertEqual(len(find_overlap_regions(turns, min_duration=0.4)), 1)


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

    def test_solo_spans_bridge_short_silence(self) -> None:
        turns = [
            SpeakerTurn(start=0.0, end=3.0, label="0"),
            SpeakerTurn(start=3.6, end=6.0, label="0"),  # 0.6 s breath
            SpeakerTurn(start=8.0, end=9.0, label="1"),
        ]
        self.assertEqual(find_solo_spans(turns, 0), [(0.0, 6.0)])

    def test_solo_spans_do_not_bridge_long_silence(self) -> None:
        turns = [
            SpeakerTurn(start=0.0, end=3.0, label="0"),
            SpeakerTurn(start=5.0, end=8.0, label="0"),  # 2 s: a different take on the room
            SpeakerTurn(start=9.0, end=10.0, label="1"),
        ]
        self.assertEqual(find_solo_spans(turns, 0), [(0.0, 3.0), (5.0, 8.0)])

    def test_solo_spans_do_not_bridge_another_voice(self) -> None:
        turns = [
            SpeakerTurn(start=0.0, end=3.0, label="0"),
            SpeakerTurn(start=3.1, end=3.4, label="1"),
            SpeakerTurn(start=3.5, end=6.0, label="0"),
        ]
        self.assertEqual(find_solo_spans(turns, 0), [(0.0, 3.0), (3.5, 6.0)])

    def test_enrollment_reaches_five_seconds_on_fragmented_speech(self) -> None:
        # Conversational pacing: 1 s of speech, 0.4 s pause. Nothing is ever 5 s
        # contiguous, but the bridged span is, so the "best" branch can run.
        turns: list[SpeakerTurn] = []
        cursor = 0.0
        for _ in range(8):
            turns.append(SpeakerTurn(start=round(cursor, 3), end=round(cursor + 1.0, 3), label="0"))
            cursor += 1.4
        turns.append(SpeakerTurn(start=round(cursor, 3), end=round(cursor + 2.0, 3), label="1"))

        span = pick_enrollment_span(turns, 0, near=cursor)
        self.assertIsNotNone(span)
        self.assertAlmostEqual(span[1] - span[0], 5.0, places=2)
        # Closest to the overlap, so it ends at the last solo speech.
        self.assertAlmostEqual(span[1], 10.8, places=2)

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


    def test_replace_crossfade_holds_power_at_the_fade_midpoint(self) -> None:
        sr = 48000
        rng = np.random.default_rng(7)
        samples = (rng.standard_normal(int(10.0 * sr)) * 0.1).astype(np.float32)[None, :].copy()
        stem = (rng.standard_normal(int(3.0 * 16000)) * 0.1).astype(np.float32)
        render = blend.RegionRender(start=3.0, stem=stem, region_start=3.5, region_end=5.5)
        blend.apply_replace(samples, sr, render, 16000)

        fade = int(blend.DEFAULT_CROSSFADE_SECONDS * sr)
        start = int(3.5 * sr)
        # The two signals are uncorrelated, so an equal-power pair keeps the sum
        # of their powers flat across the fade; 1 - blend dips 2.3 dB.
        midpoint = samples[0, start + fade // 4 : start + (3 * fade) // 4]
        interior = samples[0, int(4.4 * sr) : int(4.6 * sr)]
        dip_db = 20.0 * np.log10(float(np.sqrt(np.mean(midpoint**2)) / np.sqrt(np.mean(interior**2))))
        self.assertLess(abs(dip_db), 0.5)

    def test_stem_gain_is_capped_for_a_near_silent_stem(self) -> None:
        rng = np.random.default_rng(11)
        stem = (rng.standard_normal(16000) * 1e-5).astype(np.float32)
        reference = (rng.standard_normal(16000) * 0.5).astype(np.float32)
        matched = blend.match_stem_gain(stem, reference)
        ratio = float(np.sqrt(np.mean(matched.astype(np.float64) ** 2)) / np.sqrt(np.mean(stem.astype(np.float64) ** 2)))
        self.assertAlmostEqual(ratio, blend.MAX_STEM_GAIN, places=3)

    def test_stem_gain_still_matches_a_healthy_stem(self) -> None:
        rng = np.random.default_rng(13)
        stem = (rng.standard_normal(16000) * 0.3).astype(np.float32)
        reference = (rng.standard_normal(16000) * 0.2).astype(np.float32)
        matched = blend.match_stem_gain(stem, reference, headroom_db=0.0)
        self.assertAlmostEqual(
            float(np.sqrt(np.mean(matched.astype(np.float64) ** 2))),
            float(np.sqrt(np.mean(reference.astype(np.float64) ** 2))),
            places=4,
        )


class WindowReassemblyTests(unittest.TestCase):
    """UniSE generates independent 5 s windows; the seams must not click."""

    def test_window_starts_overlap_and_cover_the_input(self) -> None:
        self.assertEqual(UniSEEngine._window_starts(1000), [0])
        self.assertEqual(UniSEEngine._window_starts(WINDOW_SAMPLES), [0])
        starts = UniSEEngine._window_starts(int(12.5 * 16000))
        self.assertEqual(starts, [0, 72000, 144000])
        self.assertGreaterEqual(starts[-1] + WINDOW_SAMPLES, int(12.5 * 16000))
        for previous, current in zip(starts, starts[1:]):
            self.assertLess(current, previous + WINDOW_SAMPLES)

    def test_reassembly_keeps_length_and_crossfades_the_seams(self) -> None:
        length = int(12.5 * 16000)
        starts = UniSEEngine._window_starts(length)
        pieces = [np.full(WINDOW_SAMPLES, value, dtype=np.float32) for value in (1.0, 2.0, 3.0)]
        out = UniSEEngine._reassemble(pieces, starts, length)

        self.assertEqual(out.shape, (length,))
        self.assertAlmostEqual(float(out[0]), 1.0, places=5)
        self.assertAlmostEqual(float(out[-1]), 3.0, places=5)
        self.assertGreater(float(np.abs(out).min()), 0.0)  # no gaps
        # A hard concatenation would step by 1.0 at each seam.
        self.assertLess(float(np.abs(np.diff(out)).max()), 0.01)

    def test_reassembly_tolerates_short_generated_windows(self) -> None:
        length = int(12.5 * 16000)
        starts = UniSEEngine._window_starts(length)
        pieces = [np.ones(WINDOW_SAMPLES - 5, dtype=np.float32) for _ in starts]
        out = UniSEEngine._reassemble(pieces, starts, length)
        self.assertEqual(out.shape, (length,))


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


class SoloTrackEnrollmentTests(unittest.TestCase):
    """Enrollment is picked per region, so it tracks the mic/room of that overlap."""

    def test_enrollment_is_picked_for_every_region(self) -> None:
        from backend.app.separation import service
        from backend.app.separation.schemas import (
            OverlapRegionIn,
            SoloTracksParams,
            SpeakerRegionIn,
            TurnInput,
        )

        params = SoloTracksParams(
            regions=[
                OverlapRegionIn(start=5.0, end=6.0, speaker_indices=[0, 1]),
                OverlapRegionIn(start=15.0, end=16.0, speaker_indices=[0, 1]),
            ],
            turns=[
                TurnInput(start=0.0, end=5.5, speaker_index=0),
                TurnInput(start=5.0, end=20.0, speaker_index=1),
            ],
            speaker_regions=[SpeakerRegionIn(start=0.0, end=20.0, speaker_index=0)],
        )
        original = MasterAudio(samples=np.zeros((1, 48000 * 20), dtype=np.float32), sample_rate=48000)
        engine = mock.Mock(device="cpu")
        engine.run_task.side_effect = lambda *args, **kwargs: np.zeros(16000, dtype=np.float32)

        with tempfile.TemporaryDirectory() as directory:
            with (
                mock.patch.object(service, "decode_master", return_value=original),
                mock.patch.object(service, "decode_mono_16k", return_value=np.zeros(16000 * 20, dtype=np.float32)),
                mock.patch.object(service, "load_engine", return_value=engine),
                mock.patch.object(service, "encode_master"),
                mock.patch.object(service, "separation_output_dir", return_value=Path(directory)),
                mock.patch.object(
                    service, "pick_enrollment_span", return_value=(0.0, 5.0)
                ) as pick,
            ):
                result = service.run_solo_tracks("in.wav", "in.wav", params, mock.MagicMock())

        self.assertTrue(all(report.applied for report in result.regions))
        near_values = [call.kwargs["near"] for call in pick.call_args_list]
        self.assertEqual(near_values, [5.0, 15.0, 5.0, 15.0])


class SpeakerTurnSpanTests(unittest.TestCase):
    """The keep-envelope for a full-length stem: padded, merged, one speaker."""

    def test_spans_are_padded_and_clamped_at_zero(self) -> None:
        turns = [SpeakerTurn(start=0.1, end=2.0, label="0")]
        self.assertEqual(speaker_turn_spans(turns, 0, pad=0.35), [(0.0, 2.35)])

    def test_nearby_turns_merge_across_pad_and_gap(self) -> None:
        turns = [
            SpeakerTurn(start=1.0, end=2.0, label="0"),
            SpeakerTurn(start=3.0, end=4.0, label="0"),  # 1.0 s gap < 2*pad + merge
            SpeakerTurn(start=10.0, end=11.0, label="0"),
        ]
        spans = speaker_turn_spans(turns, 0, pad=0.35, merge_gap=0.6)
        self.assertEqual(len(spans), 2)
        self.assertAlmostEqual(spans[0][0], 0.65, places=3)
        self.assertAlmostEqual(spans[0][1], 4.35, places=3)

    def test_other_speakers_turns_are_ignored(self) -> None:
        turns = [
            SpeakerTurn(start=0.0, end=5.0, label="0"),
            SpeakerTurn(start=10.0, end=15.0, label="1"),
        ]
        spans = speaker_turn_spans(turns, 0)
        self.assertEqual(len(spans), 1)
        self.assertLess(spans[0][1], 6.0)

    def test_overlapped_speech_is_kept(self) -> None:
        # The speaker's turn during an overlap is still their speech.
        turns = [
            SpeakerTurn(start=0.0, end=10.0, label="0"),
            SpeakerTurn(start=4.0, end=6.0, label="1"),
        ]
        spans = speaker_turn_spans(turns, 1)
        self.assertEqual(len(spans), 1)
        self.assertLessEqual(spans[0][0], 4.0)
        self.assertGreaterEqual(spans[0][1], 6.0)


class TargetedStemTests(unittest.TestCase):
    """mode='targeted': UniSE sees only overlap and rapid-handoff windows."""

    DURATION = 24.0

    def _turns(self) -> list:
        from backend.app.separation.schemas import TurnInput

        # A solo, B solo, both 16.5-22, A solo to 24 (same shape as the
        # diarization fixtures above).
        return [
            TurnInput(start=0.0, end=8.0, speaker_index=0),
            TurnInput(start=8.4, end=16.0, speaker_index=1),
            TurnInput(start=16.5, end=22.0, speaker_index=0),
            TurnInput(start=16.5, end=22.0, speaker_index=1),
            TurnInput(start=22.0, end=24.0, speaker_index=0),
        ]

    def _run(self, params, engine=None, whisper_result=None, verified_corrections=None):
        from backend.app.separation import service

        n16 = int(self.DURATION * 16000)
        original = MasterAudio(
            samples=np.full((1, int(self.DURATION * 48000)), 0.25, dtype=np.float32),
            sample_rate=48000,
        )
        if engine is None:
            engine = mock.Mock(device="cpu")
            engine.run_task.side_effect = lambda task, mix, enroll, progress=None: np.full(
                mix.shape[-1], 0.5, dtype=np.float32
            )
        encoded: list[MasterAudio] = []

        with tempfile.TemporaryDirectory() as directory:
            with (
                mock.patch.object(service, "decode_master", return_value=original),
                mock.patch.object(
                    service, "decode_mono_16k", return_value=np.full(n16, 0.25, dtype=np.float32)
                ),
                mock.patch.object(service, "load_engine", return_value=engine),
                mock.patch.object(
                    service, "encode_master", side_effect=lambda audio, *a, **k: encoded.append(audio)
                ),
                mock.patch.object(service, "separation_output_dir", return_value=Path(directory)),
                mock.patch.object(
                    service, "_whisperx_result", return_value=whisper_result or {"segments": []}
                ) as whisper,
                mock.patch.object(
                    service,
                    "verify_isolated_utterance_speakers",
                    return_value=verified_corrections or [],
                ),
                mock.patch.object(service.shutil, "which", return_value="/usr/bin/metaflac"),
            ):
                result = service.run_solo_tracks("in.wav", "in.wav", params, mock.MagicMock())
        return result, engine, encoded, whisper

    def test_targeted_mode_extracts_only_the_overlap_window(self) -> None:
        from backend.app.separation.schemas import OverlapRegionIn, SoloTracksParams

        params = SoloTracksParams(
            mode="targeted",
            turns=self._turns(),
            regions=[OverlapRegionIn(start=16.5, end=22.0, speaker_indices=[0, 1])],
        )
        result, engine, encoded, _ = self._run(params)

        self.assertEqual([track.speaker_index for track in result.tracks], [0, 1])
        self.assertTrue(all(track.separated for track in result.tracks))
        self.assertEqual(len(result.regions), 2)
        # TSE ran once per relevant speaker, only on the padded overlap window.
        self.assertEqual(engine.run_task.call_count, 2)
        for call in engine.run_task.call_args_list:
            self.assertEqual(call.args[0], "tse")
            self.assertAlmostEqual(
                call.args[1].shape[-1],
                int((22.0 - 16.5 + 0.7) * 16000),
                delta=1,
            )
            self.assertLess(call.args[1].shape[-1], int(self.DURATION * 16000))

    def test_targeted_track_is_muted_where_the_speaker_never_talks(self) -> None:
        from backend.app.separation.schemas import OverlapRegionIn, SoloTracksParams

        params = SoloTracksParams(
            mode="targeted",
            turns=self._turns(),
            regions=[OverlapRegionIn(start=16.5, end=22.0, speaker_indices=[0, 1])],
        )
        _, _, encoded, _ = self._run(params)

        stem_b = encoded[1].samples[0]  # assembled track stays at the original 48 kHz
        sr = 48000
        # Speaker 1 does not talk during 0-8 s: their stem is silent there...
        self.assertEqual(float(np.abs(stem_b[: int(7.5 * sr)]).max()), 0.0)
        # ...but carries their voice mid-turn and inside the 16.5-22 s overlap.
        self.assertGreater(float(np.abs(stem_b[int(11.0 * sr) : int(12.0 * sr)]).max()), 0.1)
        self.assertGreater(float(np.abs(stem_b[int(18.0 * sr) : int(21.0 * sr)]).max()), 0.1)

    def test_targeted_mode_extracts_a_rapid_handoff_without_an_overlap(self) -> None:
        from backend.app.separation.schemas import AuditWordInput, SoloTracksParams

        params = SoloTracksParams(
            mode="targeted",
            turns=self._turns(),
            audit_words=[
                AuditWordInput(id="a", text="finish", start=7.7, end=8.0, speaker_index=0),
                AuditWordInput(id="b", text="right", start=8.2, end=8.45, speaker_index=1),
            ],
        )
        result, engine, _, whisper = self._run(params)

        self.assertEqual(result.handoffs_audited, 1)
        self.assertEqual(engine.run_task.call_count, 2)
        self.assertEqual(whisper.call_count, 2)  # one raw target window per relevant voice
        # The generated stems are evidence only. Clean turn-taking must keep the
        # original recording instead of replacing it with re-synthesized audio.
        self.assertTrue(all(not track.separated for track in result.tracks))
        self.assertTrue(all(not region.applied for region in result.regions))
        self.assertTrue(all("original audio preserved" in (region.detail or "") for region in result.regions))
        for call in engine.run_task.call_args_list:
            # UniSE natively consumes 5-second windows. Supplying less would
            # wrap the outgoing voice into the end and contaminate extraction.
            self.assertEqual(call.args[1].shape[-1], int(5.0 * 16000))
            self.assertLess(call.args[1].shape[-1], int(self.DURATION * 16000))

    def test_speaker_without_solo_speech_falls_back_to_gated_original(self) -> None:
        from backend.app.separation.schemas import OverlapRegionIn, SoloTracksParams, TurnInput

        turns = [
            TurnInput(start=0.0, end=20.0, speaker_index=0),
            TurnInput(start=4.0, end=6.0, speaker_index=1),  # only ever overlapped
        ]
        params = SoloTracksParams(
            mode="targeted",
            turns=turns,
            regions=[OverlapRegionIn(start=4.0, end=6.0, speaker_indices=[0, 1])],
        )
        result, engine, encoded, _ = self._run(params)

        self.assertEqual(engine.run_task.call_count, 1)  # speaker 0 only
        track_b = next(track for track in result.tracks if track.speaker_index == 1)
        self.assertFalse(track_b.separated)
        self.assertIsNotNone(track_b.detail)
        self.assertTrue(any(w.code == "separation_no_enrollment" for w in result.warnings))
        # The fallback is the original recording (48 kHz) gated to their turns.
        fallback = encoded[1].samples[0]
        sr = 48000
        self.assertGreater(float(np.abs(fallback[int(4.5 * sr) : int(5.5 * sr)]).max()), 0.1)
        self.assertEqual(float(np.abs(fallback[int(10.0 * sr) :]).max()), 0.0)

    def test_transcribe_attaches_words_to_each_track(self) -> None:
        from backend.app.separation.schemas import OverlapRegionIn, SoloTracksParams

        params = SoloTracksParams(
            mode="targeted",
            turns=self._turns(),
            regions=[OverlapRegionIn(start=16.5, end=22.0, speaker_indices=[0, 1])],
            transcribe=True,
        )
        whisper_result = {
            "segments": [
                {"start": 1.0, "words": [{"word": " hello ", "start": 1.0, "end": 1.4}]}
            ]
        }
        result, _, _, whisper = self._run(params, whisper_result=whisper_result)

        self.assertEqual(whisper.call_count, 2)
        track_a = next(track for track in result.tracks if track.speaker_index == 0)
        track_b = next(track for track in result.tracks if track.speaker_index == 1)
        self.assertEqual(len(track_a.words or []), 1)
        self.assertEqual(track_a.words[0].text, "hello")
        self.assertAlmostEqual(track_a.words[0].start, 1.0, places=3)
        # The assembled track transcript still follows the playback safety gate.
        self.assertEqual(track_b.words, [])

    def test_gated_mode_also_transcribes_when_asked(self) -> None:
        from backend.app.separation.schemas import SoloTracksParams, SpeakerRegionIn, TurnInput

        params = SoloTracksParams(
            turns=[TurnInput(start=0.0, end=20.0, speaker_index=0)],
            speaker_regions=[SpeakerRegionIn(start=0.0, end=20.0, speaker_index=0)],
            transcribe=True,
        )
        whisper_result = {
            "segments": [{"start": 2.0, "words": [{"word": "hi", "start": 2.0, "end": 2.2}]}]
        }
        result, _, _, whisper = self._run(params, whisper_result=whisper_result)

        self.assertEqual(whisper.call_count, 1)
        self.assertEqual(len(result.tracks), 1)
        self.assertEqual((result.tracks[0].words or [])[0].text, "hi")

    def test_original_audio_verification_correction_is_returned(self) -> None:
        from backend.app.separation.schemas import (
            AuditWordInput,
            HandoffCorrection,
            SoloTracksParams,
        )

        correction = HandoffCorrection(
            word_id="16-0",
            from_speaker_index=1,
            to_speaker_index=0,
            confidence=0.93,
            boundary_time=43.855,
        )
        params = SoloTracksParams(
            mode="targeted",
            turns=self._turns(),
            audit_words=[
                AuditWordInput(id="15-0", text="before", start=42.8, end=43.2, speaker_index=1),
                AuditWordInput(id="16-0", text="Never", start=43.7, end=43.95, speaker_index=1),
                AuditWordInput(id="17-0", text="after", start=44.8, end=45.1, speaker_index=0),
            ],
        )

        result, _, _, _ = self._run(params, verified_corrections=[correction])

        self.assertEqual(result.handoff_corrections, [correction])
        self.assertTrue(any(w.code == "handoff_speakers_corrected" for w in result.warnings))


class FlacSeektableTests(unittest.TestCase):
    """A gated FLAC without a seektable seeks tens of seconds off in a browser."""

    @staticmethod
    def _metadata_block_types(path: str) -> list[int]:
        types: list[int] = []
        with open(path, "rb") as handle:
            self_magic = handle.read(4)
            assert self_magic == b"fLaC"
            while True:
                header = handle.read(4)
                types.append(header[0] & 0x7F)
                size = int.from_bytes(b"\x00" + header[1:4], "big")
                handle.seek(size, 1)
                if header[0] & 0x80:
                    return types

    def test_encoded_flac_carries_a_seektable(self) -> None:
        if shutil.which("metaflac") is None:
            self.skipTest("metaflac is not installed")
        tone = (np.sin(np.linspace(0, 400 * 2 * np.pi, 48000 * 3, dtype=np.float32)) * 0.2)[None, :]
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / "tone.flac")
            encode_master(MasterAudio(samples=tone, sample_rate=48000), path, "flac")
            self.assertIn(SEEKTABLE_BLOCK_TYPE, self._metadata_block_types(path))

    def test_add_seekpoints_reports_false_without_metaflac(self) -> None:
        with mock.patch("backend.app.mastering.audio_io.shutil.which", return_value=None):
            self.assertFalse(add_flac_seekpoints("/nonexistent.flac"))

    def test_wav_output_is_untouched(self) -> None:
        tone = (np.sin(np.linspace(0, 400 * 2 * np.pi, 48000, dtype=np.float32)) * 0.2)[None, :]
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / "tone.wav")
            encode_master(MasterAudio(samples=tone, sample_rate=48000), path, "wav")
            with open(path, "rb") as handle:
                self.assertEqual(handle.read(4), b"RIFF")


if __name__ == "__main__":
    unittest.main()
