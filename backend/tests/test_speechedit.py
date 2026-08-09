import time
import unittest
from unittest import mock

import numpy as np
from fastapi.testclient import TestClient
from pydantic import ValidationError

from backend.app import main
from backend.app.speechedit import service as se_service
from backend.app.speechedit.engine import (
    build_edit_arrays,
    rms_normalize_factor,
    seconds_to_frames,
)
from backend.app.speechedit.schemas import PatchEdit, SpeechEditParams
from backend.app.speechedit.service import (
    CROSSFADE_SECONDS,
    MAX_SPAN_SECONDS,
    MAX_WINDOW_SECONDS,
    build_window_target_text,
    compute_window,
    crossfade_splice,
    extract_words,
    seconds_to_samples,
    validate_edits,
    words_in_window,
)


# --------------------------------------------------------------------------- #
# Frame / mask math (engine)                                                  #
# --------------------------------------------------------------------------- #
class FrameMathTests(unittest.TestCase):
    def test_seconds_to_frames_rounds_to_hop(self) -> None:
        # 1 s at 24 kHz / hop 256 = 93.75 frames -> 94.
        self.assertEqual(seconds_to_frames(1.0), 94)
        self.assertEqual(seconds_to_frames(0.0), 0)
        self.assertEqual(seconds_to_frames(-5.0), 0)

    def test_build_edit_arrays_middle_span(self) -> None:
        plan, mask = build_edit_arrays(100, [(20, 30, 10)])
        self.assertEqual(plan, [("keep", 0, 20), ("zero", 10), ("keep", 30, 100)])
        self.assertEqual(mask, [(True, 20), (False, 10), (True, 70)])
        self.assertEqual(sum(length for _, length in mask), 100)

    def test_build_edit_arrays_fix_duration_changes_length(self) -> None:
        plan, mask = build_edit_arrays(100, [(20, 30, 5)])
        self.assertEqual(plan, [("keep", 0, 20), ("zero", 5), ("keep", 30, 100)])
        self.assertEqual(sum(length for _, length in mask), 95)

    def test_build_edit_arrays_span_at_start_drops_leading_keep(self) -> None:
        plan, mask = build_edit_arrays(100, [(0, 10, 10)])
        self.assertEqual(plan, [("zero", 10), ("keep", 10, 100)])
        self.assertEqual(mask, [(False, 10), (True, 90)])

    def test_build_edit_arrays_span_at_end_drops_trailing_keep(self) -> None:
        plan, mask = build_edit_arrays(100, [(90, 100, 10)])
        self.assertEqual(plan, [("keep", 0, 90), ("zero", 10)])
        self.assertEqual(mask, [(True, 90), (False, 10)])

    def test_build_edit_arrays_multiple_spans(self) -> None:
        plan, _ = build_edit_arrays(100, [(10, 20, 10), (40, 50, 10)])
        self.assertEqual(
            plan,
            [("keep", 0, 10), ("zero", 10), ("keep", 20, 40), ("zero", 10), ("keep", 50, 100)],
        )

    def test_rms_normalize_only_boosts_quiet_audio(self) -> None:
        loud = np.full(1000, 0.5, dtype=np.float32)  # rms 0.5 >= 0.1
        self.assertEqual(rms_normalize_factor(loud), 1.0)
        quiet = np.full(1000, 0.01, dtype=np.float32)  # rms 0.01 < 0.1
        self.assertAlmostEqual(rms_normalize_factor(quiet), 10.0, places=4)
        self.assertEqual(rms_normalize_factor(np.zeros(10, dtype=np.float32)), 1.0)
        self.assertEqual(rms_normalize_factor(np.array([], dtype=np.float32)), 1.0)


# --------------------------------------------------------------------------- #
# Edit validation                                                             #
# --------------------------------------------------------------------------- #
class ValidateEditsTests(unittest.TestCase):
    def _edit(self, start, end, **kw) -> PatchEdit:
        return PatchEdit(start_s=start, end_s=end, new_text=kw.pop("text", "hi"), **kw)

    def test_empty_list_rejected(self) -> None:
        with self.assertRaises(ValueError):
            validate_edits([], 60.0)

    def test_out_of_bounds_rejected(self) -> None:
        with self.assertRaises(ValueError):
            validate_edits([self._edit(50.0, 70.0)], 60.0)

    def test_overlap_rejected(self) -> None:
        with self.assertRaises(ValueError):
            validate_edits([self._edit(1.0, 5.0), self._edit(4.0, 8.0)], 60.0)

    def test_span_longer_than_limit_rejected(self) -> None:
        with self.assertRaises(ValueError):
            validate_edits([self._edit(0.0, MAX_SPAN_SECONDS + 1.0)], 60.0)

    def test_unsorted_but_non_overlapping_is_accepted(self) -> None:
        # validate_edits sorts internally, so caller order does not matter.
        validate_edits([self._edit(30.0, 32.0), self._edit(1.0, 2.0)], 60.0)

    def test_valid_edits_pass(self) -> None:
        validate_edits([self._edit(1.0, 2.0), self._edit(10.0, 12.0)], 60.0)


# --------------------------------------------------------------------------- #
# Window computation                                                          #
# --------------------------------------------------------------------------- #
class ComputeWindowTests(unittest.TestCase):
    def test_padding_in_the_middle(self) -> None:
        self.assertEqual(compute_window(10.0, 12.0, 100.0), (6.0, 16.0))

    def test_clamps_at_file_start(self) -> None:
        self.assertEqual(compute_window(1.0, 2.0, 100.0), (0.0, 6.0))

    def test_clamps_at_file_end(self) -> None:
        self.assertEqual(compute_window(96.0, 98.0, 100.0), (92.0, 100.0))

    def test_caps_total_at_max_window_shrinking_pad_symmetrically(self) -> None:
        # span 18 s + 4 s pad each side = 26 s > 25 s cap.
        w0, w1 = compute_window(5.0, 23.0, 100.0)
        self.assertAlmostEqual(w1 - w0, MAX_WINDOW_SECONDS, places=6)
        self.assertAlmostEqual(w0, 1.5, places=6)
        self.assertAlmostEqual(w1, 26.5, places=6)


# --------------------------------------------------------------------------- #
# Word flattening / window text building                                      #
# --------------------------------------------------------------------------- #
class WordTextTests(unittest.TestCase):
    def _words(self):
        return [
            {"text": "the", "start": 0.0, "end": 0.5},
            {"text": "quick", "start": 0.5, "end": 1.0},
            {"text": "brown", "start": 1.0, "end": 1.5},
            {"text": "fox", "start": 1.5, "end": 2.0},
            {"text": "jumps", "start": 2.0, "end": 2.5},
        ]

    def test_extract_words_uses_aligned_word_timings(self) -> None:
        result = {"segments": [{"start": 0.0, "end": 1.0, "words": [
            {"word": "hello", "start": 0.0, "end": 0.4},
            {"word": "there", "start": 0.5, "end": 1.0},
        ]}]}
        self.assertEqual(
            extract_words(result),
            [{"text": "hello", "start": 0.0, "end": 0.4}, {"text": "there", "start": 0.5, "end": 1.0}],
        )

    def test_extract_words_falls_back_to_segment_granularity(self) -> None:
        result = {"segments": [{"start": 2.0, "end": 4.0, "text": "no words here"}]}
        self.assertEqual(extract_words(result), [{"text": "no words here", "start": 2.0, "end": 4.0}])

    def test_words_in_window_overlap(self) -> None:
        got = [w["text"] for w in words_in_window(self._words(), 0.9, 1.6)]
        self.assertEqual(got, ["quick", "brown", "fox"])

    def test_replace_span_in_middle(self) -> None:
        # Span 1.0-1.5 covers "brown"; keep neighbours.
        text = build_window_target_text(self._words(), 1.0, 1.5, "grey")
        self.assertEqual(text, "the quick grey fox jumps")

    def test_straddling_words_count_as_inside(self) -> None:
        # Span 1.2-1.8 straddles "brown" (1.0-1.5) and "fox" (1.5-2.0); both replaced.
        text = build_window_target_text(self._words(), 1.2, 1.8, "grey wolf")
        self.assertEqual(text, "the quick grey wolf jumps")


# --------------------------------------------------------------------------- #
# Splice math                                                                 #
# --------------------------------------------------------------------------- #
class CrossfadeSpliceTests(unittest.TestCase):
    def test_outside_window_stays_bit_identical(self) -> None:
        audio = np.arange(1000, dtype=np.float32)
        edited = np.full(200, -1.0, dtype=np.float32)
        out = crossfade_splice(audio, 400, 600, edited, fade_samples=10)
        # Same-length edit -> same total length.
        self.assertEqual(len(out), 1000)
        np.testing.assert_array_equal(out[:400], audio[:400])
        np.testing.assert_array_equal(out[600:], audio[600:])

    def test_length_delta_when_edited_is_longer(self) -> None:
        audio = np.arange(1000, dtype=np.float32)
        edited = np.full(300, -1.0, dtype=np.float32)
        out = crossfade_splice(audio, 400, 600, edited, fade_samples=10)
        self.assertEqual(len(out), 1000 - 200 + 300)
        np.testing.assert_array_equal(out[:400], audio[:400])
        np.testing.assert_array_equal(out[-400:], audio[600:])

    def test_back_to_front_ordering_preserves_earlier_bounds(self) -> None:
        # Splicing the later window first must not shift the earlier window's
        # sample indices. Two windows: [100,200] and [600,700], edited longer.
        audio = np.arange(1000, dtype=np.float32)
        windows = [(100, 200), (600, 700)]
        edited = {w: np.full(150, float(-i - 1), dtype=np.float32) for i, w in enumerate(windows)}
        out = audio
        for s0, s1 in reversed(windows):
            out = crossfade_splice(out, s0, s1, edited[(s0, s1)], fade_samples=5)
        # Everything before the first window is untouched.
        np.testing.assert_array_equal(out[:100], audio[:100])
        # The first edit's fill is present near sample 100.
        self.assertEqual(out[125], -1.0)

    def test_degenerate_window_hard_replaces(self) -> None:
        audio = np.arange(100, dtype=np.float32)
        edited = np.full(5, 9.0, dtype=np.float32)
        out = crossfade_splice(audio, 40, 45, edited, fade_samples=10)  # window < 2*fade
        np.testing.assert_array_equal(out, np.concatenate([audio[:40], edited, audio[45:]]))

    def test_seconds_to_samples(self) -> None:
        self.assertEqual(seconds_to_samples(1.0), 24000)
        self.assertEqual(seconds_to_samples(CROSSFADE_SECONDS), 3600)


# --------------------------------------------------------------------------- #
# Params validation                                                           #
# --------------------------------------------------------------------------- #
class ParamsTests(unittest.TestCase):
    def test_defaults(self) -> None:
        params = SpeechEditParams()
        self.assertEqual(params.mode, "edit")
        self.assertEqual(params.nfe_step, 32)
        self.assertEqual(params.output_format, "flac")
        self.assertEqual(params.language, "English")

    def test_language_is_english_only(self) -> None:
        with self.assertRaises(ValidationError):
            SpeechEditParams(language="Spanish")

    def test_nfe_step_bounds(self) -> None:
        with self.assertRaises(ValidationError):
            SpeechEditParams(nfe_step=4)
        with self.assertRaises(ValidationError):
            SpeechEditParams(nfe_step=128)

    def test_patch_edit_requires_new_text(self) -> None:
        with self.assertRaises(ValidationError):
            PatchEdit(start_s=1.0, end_s=2.0, new_text="")

    def test_patch_edit_end_must_be_positive(self) -> None:
        with self.assertRaises(ValidationError):
            PatchEdit(start_s=0.0, end_s=0.0, new_text="hi")

    def test_fix_duration_upper_bound(self) -> None:
        with self.assertRaises(ValidationError):
            PatchEdit(start_s=0.0, end_s=1.0, new_text="hi", fix_duration_s=21.0)


# --------------------------------------------------------------------------- #
# Artifact helpers                                                            #
# --------------------------------------------------------------------------- #
class ArtifactHelperTests(unittest.TestCase):
    def test_find_rejects_malformed_token(self) -> None:
        self.assertIsNone(se_service.find_speechedit_artifact("../etc/passwd"))


# --------------------------------------------------------------------------- #
# Endpoint integration (mocked engine)                                        #
# --------------------------------------------------------------------------- #
class FakeEngine:
    device = "cpu"

    def edit_window(self, window_wav, edits_local, window_text, *, nfe_step, seed):
        # Return the window unchanged length so splices stay predictable.
        return np.asarray(window_wav, dtype=np.float32).reshape(-1)

    def generate(self, ref_path, ref_text, gen_text, *, nfe_step, speed, seed):
        sr = 24000
        t = np.arange(int(0.2 * sr)) / sr
        return (0.5 * np.sin(2 * np.pi * 220.0 * t)).astype(np.float32), sr


def fake_transcribe(*args, **kwargs):
    return (
        {"segments": [{"start": 0.0, "end": 3.0, "words": [
            {"word": "one", "start": 0.0, "end": 1.0},
            {"word": "two", "start": 1.0, "end": 2.0},
            {"word": "three", "start": 2.0, "end": 3.0},
        ]}]},
        [],
        False,
    )


def wait_for_done(client: TestClient, job_id: str, timeout_s: float = 5.0) -> dict:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        payload = client.get(f"/api/jobs/{job_id}").json()
        if payload["status"] in ("done", "error"):
            return payload
        time.sleep(0.02)
    raise AssertionError("job did not finish in time")


class SpeechEditEndpointTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = TestClient(main.app)
        self.engine = FakeEngine()
        patches = [
            mock.patch.object(se_service, "load_engine", return_value=self.engine),
            mock.patch.object(se_service, "decode_audio", return_value=np.zeros(24000 * 5, dtype=np.float32)),
        ]
        for patch in patches:
            patch.start()
            self.addCleanup(patch.stop)

    def _post(self, params_json: str) -> dict:
        response = self.client.post(
            "/api/speech-edit",
            files={"audio": ("rec.wav", b"RIFF0000WAVE", "audio/wav")},
            data={"params_json": params_json},
        )
        self.assertEqual(response.status_code, 200)
        return wait_for_done(self.client, response.json()["job_id"])

    def test_invalid_params_json_is_rejected(self) -> None:
        response = self.client.post(
            "/api/speech-edit",
            files={"audio": ("rec.wav", b"RIFF0000WAVE", "audio/wav")},
            data={"params_json": '{"nfe_step": 999}'},
        )
        self.assertEqual(response.status_code, 422)

    def test_edit_mode_runs_and_creates_artifact(self) -> None:
        with mock.patch("backend.app.whisperx_transcription.transcribe_with_whisperx", side_effect=fake_transcribe):
            payload = self._post(
                '{"mode": "edit", "output_format": "wav", '
                '"edits": [{"start_s": 1.0, "end_s": 2.0, "new_text": "TWO"}]}'
            )
        self.assertEqual(payload["status"], "done", payload.get("error"))
        result = payload["result"]
        self.assertTrue(result["token"].startswith("se_"))
        self.assertEqual(result["mode"], "edit")
        self.assertEqual(result["sample_rate"], 24000)
        self.assertEqual(len(result["regions"]), 1)
        self.assertIn("TWO", result["regions"][0]["text_used"])

        artifact = se_service.find_speechedit_artifact(result["token"])
        self.assertIsNotNone(artifact)
        self.assertTrue(artifact.name.endswith(".f5tts.wav"))
        self.assertEqual(self.client.get(f"/api/speech-edit/{result['token']}/audio").status_code, 200)
        self.assertEqual(self.client.head(f"/api/speech-edit/{result['token']}/audio").status_code, 200)
        self.assertEqual(self.client.delete(f"/api/speech-edit/{result['token']}").status_code, 200)
        self.assertEqual(self.client.get(f"/api/speech-edit/{result['token']}/audio").status_code, 404)

    def test_edit_mode_window_text_override_skips_transcription(self) -> None:
        with mock.patch(
            "backend.app.whisperx_transcription.transcribe_with_whisperx",
            side_effect=RuntimeError("asr down"),
        ):
            payload = self._post(
                '{"mode": "edit", "output_format": "wav", '
                '"edits": [{"start_s": 1.0, "end_s": 2.0, "new_text": "x", '
                '"window_text": "manual full window text"}]}'
            )
        self.assertEqual(payload["status"], "done", payload.get("error"))
        self.assertEqual(payload["result"]["regions"][0]["text_used"], "manual full window text")
        self.client.delete(f"/api/speech-edit/{payload['result']['token']}")

    def test_edit_mode_fails_without_transcript_or_override(self) -> None:
        with mock.patch(
            "backend.app.whisperx_transcription.transcribe_with_whisperx",
            side_effect=RuntimeError("asr down"),
        ):
            payload = self._post(
                '{"mode": "edit", "edits": [{"start_s": 1.0, "end_s": 2.0, "new_text": "x"}]}'
            )
        self.assertEqual(payload["status"], "error")
        self.assertIn("window_text", payload["error"])

    def test_edit_mode_rejects_empty_edits(self) -> None:
        payload = self._post('{"mode": "edit", "edits": []}')
        self.assertEqual(payload["status"], "error")

    def test_generate_mode_runs_and_reports_ref_text(self) -> None:
        with mock.patch("backend.app.whisperx_transcription.transcribe_with_whisperx", side_effect=fake_transcribe):
            payload = self._post('{"mode": "generate", "output_format": "wav", "gen_text": "Hello world."}')
        self.assertEqual(payload["status"], "done", payload.get("error"))
        result = payload["result"]
        self.assertEqual(result["mode"], "generate")
        self.assertEqual(result["ref_text_used"], "one two three")
        self.assertEqual(result["regions"], [])
        self.client.delete(f"/api/speech-edit/{result['token']}")

    def test_generate_mode_fails_when_ref_transcript_unavailable(self) -> None:
        with mock.patch(
            "backend.app.whisperx_transcription.transcribe_with_whisperx",
            side_effect=RuntimeError("asr down"),
        ):
            payload = self._post('{"mode": "generate", "output_format": "wav", "gen_text": "Hi."}')
        self.assertEqual(payload["status"], "error")
        self.assertIn("patch_ref_transcript_failed", payload["error"])

    def test_generate_mode_requires_gen_text(self) -> None:
        payload = self._post('{"mode": "generate", "gen_text": "   "}')
        self.assertEqual(payload["status"], "error")

    def test_unknown_token_returns_404(self) -> None:
        self.assertEqual(self.client.get("/api/speech-edit/se_zzz/audio").status_code, 404)
        self.assertEqual(self.client.get("/api/speech-edit/se_zzz/waveform").status_code, 404)


if __name__ == "__main__":
    unittest.main()
