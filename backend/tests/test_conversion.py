import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

from fastapi.testclient import TestClient
from pydantic import ValidationError

from backend.app import main
from backend.app.conversion import service as conversion_service
from backend.app.conversion.engine import expected_chunks
from backend.app.conversion.schemas import ConversionParams


class ConversionParamsTests(unittest.TestCase):
    def test_defaults(self) -> None:
        params = ConversionParams()
        self.assertEqual(params.output_format, "flac")
        self.assertEqual(params.diffusion_steps, 50)
        self.assertEqual(params.length_adjust, 1.0)
        self.assertEqual(params.intelligibility_cfg, 0.7)
        self.assertEqual(params.similarity_cfg, 0.7)
        self.assertFalse(params.convert_style)

    def test_diffusion_steps_bounds(self) -> None:
        self.assertEqual(ConversionParams(diffusion_steps=10).diffusion_steps, 10)
        self.assertEqual(ConversionParams(diffusion_steps=100).diffusion_steps, 100)
        with self.assertRaises(ValidationError):
            ConversionParams(diffusion_steps=9)
        with self.assertRaises(ValidationError):
            ConversionParams(diffusion_steps=101)

    def test_length_adjust_bounds(self) -> None:
        with self.assertRaises(ValidationError):
            ConversionParams(length_adjust=0.4)
        with self.assertRaises(ValidationError):
            ConversionParams(length_adjust=2.1)

    def test_cfg_bounds(self) -> None:
        with self.assertRaises(ValidationError):
            ConversionParams(similarity_cfg=-0.1)
        with self.assertRaises(ValidationError):
            ConversionParams(intelligibility_cfg=1.1)

    def test_unknown_output_format_rejected(self) -> None:
        with self.assertRaises(ValidationError):
            ConversionParams.model_validate({"output_format": "banana"})


class ExpectedChunksTests(unittest.TestCase):
    def test_minimum_one_chunk(self) -> None:
        # 30 s windows minus 5 s overlap => 25 s of new audio per streamed chunk.
        self.assertEqual(expected_chunks(0.0), 1)
        self.assertEqual(expected_chunks(1.0), 1)
        self.assertEqual(expected_chunks(25.0), 1)

    def test_scales_with_25s_windows(self) -> None:
        self.assertEqual(expected_chunks(26.0), 2)
        self.assertEqual(expected_chunks(50.0), 2)
        self.assertEqual(expected_chunks(51.0), 3)


class FormatEtaTests(unittest.TestCase):
    def test_seconds_below_a_minute(self) -> None:
        from backend.app.conversion.engine import format_eta

        self.assertEqual(format_eta(0), "0 s")
        self.assertEqual(format_eta(41.4), "41 s")

    def test_minutes(self) -> None:
        from backend.app.conversion.engine import format_eta

        self.assertEqual(format_eta(90), "2 min")
        self.assertEqual(format_eta(59 * 60), "59 min")

    def test_hours_keep_two_digit_minutes(self) -> None:
        from backend.app.conversion.engine import format_eta

        self.assertEqual(format_eta(17 * 3600 + 5 * 60), "17 h 05 min")

    def test_negative_clamps_to_zero(self) -> None:
        from backend.app.conversion.engine import format_eta

        self.assertEqual(format_eta(-5), "0 s")


class ChunkTickerTests(unittest.TestCase):
    def test_progress_is_monotonic_and_messages_carry_chunk_counts(self) -> None:
        from backend.app.conversion.engine import _ChunkTicker

        emitted: list[tuple[float, str]] = []
        ticker = _ChunkTicker(4, lambda fraction, message: emitted.append((fraction, message)))
        ticker._emit(0.0)
        ticker._emit(0.5)  # mid-chunk tick
        ticker.chunk_done()
        ticker._emit(0.2)
        ticker.chunk_done()

        fractions = [fraction for fraction, _ in emitted]
        self.assertEqual(fractions, sorted(fractions))
        self.assertIn("chunk 1/4", emitted[0][1])
        self.assertIn("chunk 2/4", emitted[2][1])
        self.assertIn("left)", emitted[-1][1])

    def test_final_chunk_caps_at_ninety_seven_percent(self) -> None:
        from backend.app.conversion.engine import _ChunkTicker

        emitted: list[float] = []
        ticker = _ChunkTicker(2, lambda fraction, _message: emitted.append(fraction))
        ticker.chunk_done()
        ticker.chunk_done()
        self.assertAlmostEqual(emitted[-1], 0.97)


class ArtifactPathTests(unittest.TestCase):
    def test_output_paths_use_vc_token_and_seedvc_suffix(self) -> None:
        with TemporaryDirectory() as tmp:
            mastering = Path(tmp) / "mastering"
            with mock.patch.object(conversion_service.settings, "mastering_output_dir", str(mastering)):
                token, path = conversion_service._output_paths("phone call.m4a", "flac")
        self.assertTrue(token.startswith("vc_"))
        self.assertTrue(path.name.startswith(f"{token}__"))
        self.assertTrue(path.name.endswith(".seedvc.flac"))

    def test_find_artifact_rejects_bad_tokens_and_finds_created_file(self) -> None:
        with TemporaryDirectory() as tmp:
            mastering = Path(tmp) / "mastering"
            with mock.patch.object(conversion_service.settings, "mastering_output_dir", str(mastering)):
                # Missing directory -> None regardless of token.
                self.assertIsNone(conversion_service.find_conversion_artifact("vc_abc123"))

                output_dir = conversion_service.conversion_output_dir()
                output_dir.mkdir(parents=True, exist_ok=True)

                # Path-traversal / punctuation tokens are rejected before any glob.
                self.assertIsNone(conversion_service.find_conversion_artifact("vc_../etc"))
                self.assertIsNone(conversion_service.find_conversion_artifact("vc_bad token"))

                artifact = output_dir / "vc_abc123__clip.seedvc.flac"
                artifact.write_bytes(b"x")
                self.assertEqual(conversion_service.find_conversion_artifact("vc_abc123"), artifact)


class ConversionEndpointTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = TestClient(main.app)

    def test_invalid_params_json_is_rejected(self) -> None:
        response = self.client.post(
            "/api/convert",
            files={
                "audio": ("src.wav", b"RIFF0000WAVE", "audio/wav"),
                "reference": ("ref.wav", b"RIFF0000WAVE", "audio/wav"),
            },
            data={"params_json": '{"diffusion_steps": 500}'},
        )
        self.assertEqual(response.status_code, 422)

    def test_missing_reference_upload_is_rejected(self) -> None:
        response = self.client.post(
            "/api/convert",
            files={"audio": ("src.wav", b"RIFF0000WAVE", "audio/wav")},
            data={"params_json": "{}"},
        )
        self.assertEqual(response.status_code, 422)

    def test_unknown_conversion_token_returns_404(self) -> None:
        self.assertEqual(self.client.get("/api/convert/vc_missing/audio").status_code, 404)
        self.assertEqual(self.client.get("/api/convert/vc_missing/waveform").status_code, 404)


if __name__ == "__main__":
    unittest.main()
