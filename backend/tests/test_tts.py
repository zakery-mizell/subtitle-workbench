import time
import unittest
from unittest import mock

import numpy as np
from fastapi.testclient import TestClient
from pydantic import ValidationError

from backend.app import main
from backend.app.tts import engine as tts_engine
from backend.app.tts import service as tts_service
from backend.app.tts.engine import max_new_tokens_for, split_text_into_chunks
from backend.app.tts.schemas import TtsParams


class SplitTextTests(unittest.TestCase):
    def test_tts_language_is_english_only(self) -> None:
        self.assertEqual(TtsParams(text="Hello").language, "English")
        with self.assertRaises(ValidationError):
            TtsParams(text="Hola", language="Spanish")

    def test_single_chunk_when_under_limit(self) -> None:
        self.assertEqual(
            split_text_into_chunks("Hi there. How are you?", max_chars=300),
            ["Hi there. How are you?"],
        )

    def test_packs_sentences_greedily_up_to_limit(self) -> None:
        chunks = split_text_into_chunks("Alpha beta. Gamma delta. Epsilon zeta.", max_chars=15)
        self.assertEqual(chunks, ["Alpha beta.", "Gamma delta.", "Epsilon zeta."])
        self.assertTrue(all(len(c) <= 15 for c in chunks))

    def test_two_sentences_pack_into_one_chunk(self) -> None:
        # "One. Two." (8 chars) fits within 20 but the next sentence overflows.
        chunks = split_text_into_chunks("One. Two. Three four five six.", max_chars=20)
        self.assertEqual(chunks[0], "One. Two.")

    def test_long_sentence_hard_split_on_whitespace(self) -> None:
        sentence = " ".join(["word"] * 20)  # 99 chars, no terminal punctuation
        chunks = split_text_into_chunks(sentence, max_chars=24)
        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(len(c) <= 24 for c in chunks))
        # No word is dropped or reordered by the hard split.
        self.assertEqual(" ".join(chunks).split(), sentence.split())

    def test_unbreakable_long_word_stays_whole(self) -> None:
        word = "x" * 50
        self.assertEqual(split_text_into_chunks(word, max_chars=10), [word])

    def test_max_new_tokens_scales_with_chunk_length(self) -> None:
        # 12 codec frames/sec: the budget must comfortably cover normal speech
        # (a 120-char English sentence is ~8 s ≈ 96 frames) while bounding
        # runaway generation from a mismatched reference transcript.
        self.assertEqual(max_new_tokens_for(""), 160)
        self.assertEqual(max_new_tokens_for("a" * 120), 3 * 120 + 160)
        self.assertGreater(max_new_tokens_for("a" * 120), 96)


class FakeEngine:
    """Stand-in for TtsEngine that renders a short tone per chunk."""

    device = "cpu"

    def __init__(self) -> None:
        self.prompt_calls = 0
        self.generate_calls = 0

    def clone_prompt(self, ref_wav, ref_sr, ref_text, x_vector_only):
        self.prompt_calls += 1
        return [{"ref_text": ref_text, "x_vector_only": x_vector_only}]

    def generate(self, text_chunk, language, prompt_items):
        self.generate_calls += 1
        sr = 24000
        t = np.arange(int(0.2 * sr)) / sr
        return (0.5 * np.sin(2 * np.pi * 220.0 * t)).astype(np.float32), sr


def fake_transcribe(*args, **kwargs):
    return {"segments": [{"text": "hello there"}]}, [], False


def wait_for_done(client: TestClient, job_id: str, timeout_s: float = 5.0) -> dict:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        payload = client.get(f"/api/jobs/{job_id}").json()
        if payload["status"] in ("done", "error"):
            return payload
        time.sleep(0.02)
    raise AssertionError("job did not finish in time")


class TtsEndpointTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = TestClient(main.app)
        # Fake engine served to both the service (device report) and the engine
        # driver (synthesis); real decode/transcribe are patched away per test.
        self.engine = FakeEngine()
        patches = [
            mock.patch.object(tts_service, "load_engine", return_value=self.engine),
            mock.patch.object(tts_engine, "load_engine", return_value=self.engine),
            mock.patch.object(tts_service, "decode_reference", return_value=(np.zeros(2400, dtype=np.float32), False)),
        ]
        for patch in patches:
            patch.start()
            self.addCleanup(patch.stop)

    def _post(self, params_json: str) -> dict:
        response = self.client.post(
            "/api/tts",
            files={"audio": ("me.wav", b"RIFF0000WAVE", "audio/wav")},
            data={"params_json": params_json},
        )
        self.assertEqual(response.status_code, 200)
        return wait_for_done(self.client, response.json()["job_id"])

    def test_invalid_params_json_is_rejected(self) -> None:
        response = self.client.post(
            "/api/tts",
            files={"audio": ("me.wav", b"RIFF0000WAVE", "audio/wav")},
            data={"params_json": '{"text": "hi", "output_format": "banana"}'},
        )
        self.assertEqual(response.status_code, 422)

    def test_missing_text_is_rejected(self) -> None:
        response = self.client.post(
            "/api/tts",
            files={"audio": ("me.wav", b"RIFF0000WAVE", "audio/wav")},
            data={"params_json": "{}"},
        )
        self.assertEqual(response.status_code, 422)

    def test_transcript_clone_runs_to_completion_and_creates_artifact(self) -> None:
        with mock.patch("backend.app.whisperx_transcription.transcribe_with_whisperx", side_effect=fake_transcribe):
            payload = self._post('{"text": "Hello world. Goodbye now.", "output_format": "wav"}')

        self.assertEqual(payload["status"], "done")
        result = payload["result"]
        self.assertTrue(result["token"].startswith("t_"))
        self.assertEqual(result["clone_mode"], "transcript")
        self.assertEqual(result["ref_text_used"], "hello there")
        self.assertEqual(result["model_size"], "1.7b")
        self.assertEqual(result["device_used"], "cpu")
        self.assertEqual(result["sample_rate"], 24000)
        self.assertGreater(result["duration_sec"], 0.0)

        artifact = tts_service.find_tts_artifact(result["token"])
        self.assertIsNotNone(artifact)
        self.assertTrue(artifact.is_file())
        self.assertTrue(artifact.name.endswith(".qwen3tts.wav"))

        # Audio (GET + HEAD), then delete, then it is gone.
        self.assertEqual(self.client.get(f"/api/tts/{result['token']}/audio").status_code, 200)
        self.assertEqual(self.client.head(f"/api/tts/{result['token']}/audio").status_code, 200)
        self.assertEqual(self.client.delete(f"/api/tts/{result['token']}").status_code, 200)
        self.assertEqual(self.client.get(f"/api/tts/{result['token']}/audio").status_code, 404)

    def test_auto_transcribe_runs_on_the_trimmed_reference(self) -> None:
        # The transcript must describe the SAME audio the prompt encoder sees
        # (decode_reference's 30s-trimmed output, 2400 samples in this mock),
        # not the original upload -- a longer upload's transcript would derail
        # ICL generation into re-speaking the reference.
        import soundfile as sf

        seen_frames: list[int] = []

        def capturing_transcribe(audio_path, **kwargs):
            seen_frames.append(sf.info(audio_path).frames)
            return {"segments": [{"text": "hello there"}]}, [], False

        with mock.patch("backend.app.whisperx_transcription.transcribe_with_whisperx", side_effect=capturing_transcribe):
            payload = self._post('{"text": "Hi.", "output_format": "wav"}')

        self.assertEqual(payload["status"], "done")
        self.assertEqual(seen_frames, [2400])
        self.client.delete(f"/api/tts/{payload['result']['token']}")

    def test_auto_transcribe_failure_falls_back_to_voice_signature(self) -> None:
        with mock.patch("backend.app.whisperx_transcription.transcribe_with_whisperx", side_effect=RuntimeError("asr down")):
            payload = self._post('{"text": "Read this aloud.", "output_format": "wav"}')

        self.assertEqual(payload["status"], "done")
        result = payload["result"]
        self.assertEqual(result["clone_mode"], "voice-signature")
        self.assertIsNone(result["ref_text_used"])
        codes = [w["code"] for w in result["warnings"]]
        self.assertIn("tts_ref_transcript_failed", codes)

        tts_service.find_tts_artifact(result["token"]) and self.client.delete(f"/api/tts/{result['token']}")

    def test_unknown_tts_token_returns_404(self) -> None:
        self.assertEqual(self.client.get("/api/tts/t_zzz/audio").status_code, 404)
        self.assertEqual(self.client.get("/api/tts/t_zzz/waveform").status_code, 404)


if __name__ == "__main__":
    unittest.main()
