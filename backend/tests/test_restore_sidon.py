from __future__ import annotations

import unittest

from backend.app.restore.schemas import RestoreParams
from backend.app.restore.sidon_engine import (
    MIN_CHUNK_SAMPLES,
    SIDON_INPUT_RATE,
    _chunk_bounds,
    _resolve_device,
)
from backend.app.separation.schemas import SoloTracksParams


class ChunkBoundsTests(unittest.TestCase):
    def test_chunks_cover_the_signal_exactly_once(self) -> None:
        total = 7 * SIDON_INPUT_RATE
        bounds = _chunk_bounds(total, 2 * SIDON_INPUT_RATE)
        self.assertEqual(bounds[0][0], 0)
        self.assertEqual(bounds[-1][1], total)
        for (_, end), (next_start, _) in zip(bounds, bounds[1:]):
            self.assertEqual(end, next_start)

    def test_short_tail_is_folded_into_its_predecessor(self) -> None:
        # A stub trailing chunk would produce too few mel frames to decode.
        total = 2 * SIDON_INPUT_RATE + (MIN_CHUNK_SAMPLES - 1)
        bounds = _chunk_bounds(total, SIDON_INPUT_RATE)
        self.assertEqual(len(bounds), 2)
        self.assertEqual(bounds[-1][1], total)
        self.assertGreaterEqual(bounds[-1][1] - bounds[-1][0], MIN_CHUNK_SAMPLES)

    def test_a_signal_shorter_than_one_chunk_is_a_single_chunk(self) -> None:
        self.assertEqual(_chunk_bounds(1000, SIDON_INPUT_RATE), [(0, 1000)])


class DevicePolicyTests(unittest.TestCase):
    def test_mps_is_rejected_rather_than_silently_producing_garbage(self) -> None:
        # The checkpoints are traced with device-pinned constants, so an MPS
        # load either raises deep in torch or feeds CPU tensors to MPS ops.
        from unittest import mock

        with mock.patch("backend.app.restore.sidon_engine.settings") as settings:
            settings.sidon_device = "mps"
            with self.assertRaises(RuntimeError) as caught:
                _resolve_device()
        self.assertIn("mps", str(caught.exception).lower())


class RestoreParamDefaultTests(unittest.TestCase):
    def test_sidon_is_the_default_engine_everywhere(self) -> None:
        self.assertEqual(RestoreParams().engine, "sidon")
        self.assertEqual(SoloTracksParams().restore_engine, "sidon")

    def test_diamond_stays_selectable_with_its_own_knobs(self) -> None:
        params = RestoreParams(engine="diamond", rep_penalty=1.5)
        self.assertEqual(params.engine, "diamond")
        self.assertAlmostEqual(params.rep_penalty, 1.5)

    def test_context_window_is_bounded(self) -> None:
        for bad in (1.0, 200.0):
            with self.assertRaises(ValueError):
                RestoreParams(sidon_chunk_sec=bad)


if __name__ == "__main__":
    unittest.main()
