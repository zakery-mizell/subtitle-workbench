import unittest

from backend.app.waveform_analysis import FrameStats, _detect_speech_spans, _downsample_display_frames


class WaveformAnalysisTests(unittest.TestCase):
    def test_detect_speech_spans_merges_short_internal_gaps(self) -> None:
        frames = [
            FrameStats(time=index * 0.02, min=-rms, max=rms, rms=rms)
            for index, rms in enumerate(
                [0.001] * 8
                + [0.05] * 12
                + [0.006] * 3
                + [0.045] * 10
                + [0.001] * 8
            )
        ]
        smoothed = [frame.rms for frame in frames]

        spans = _detect_speech_spans(frames, smoothed, threshold=0.02)

        self.assertEqual(len(spans), 1)
        self.assertAlmostEqual(spans[0].start, 0.16, places=2)
        self.assertGreater(spans[0].end, 0.6)

    def test_downsample_display_frames_preserves_extremes(self) -> None:
        frames = [
            FrameStats(time=0.0, min=-0.1, max=0.2, rms=0.1),
            FrameStats(time=0.02, min=-0.6, max=0.3, rms=0.2),
            FrameStats(time=0.04, min=-0.2, max=0.9, rms=0.4),
            FrameStats(time=0.06, min=-0.4, max=0.1, rms=0.3),
        ]

        display = _downsample_display_frames(frames, limit=2)

        self.assertEqual(len(display), 2)
        self.assertEqual(display[0].min, -0.6)
        self.assertEqual(display[1].max, 0.9)
        self.assertAlmostEqual(display[0].rms, 0.15)


if __name__ == "__main__":
    unittest.main()
