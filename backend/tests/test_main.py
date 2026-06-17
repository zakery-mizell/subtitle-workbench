import unittest

from fastapi import HTTPException

from backend.app.diarization import SpeakerTurn, assign_speaker_id
from backend.app.main import parse_speakers_json, resolve_requested_language, shift_segment_times, validate_speaker_request


class MainApiTests(unittest.TestCase):
    def test_parse_speakers_json_rejects_invalid_payload(self) -> None:
        with self.assertRaises(HTTPException) as context:
            parse_speakers_json('{"bad": true}')

        self.assertEqual(context.exception.status_code, 422)
        self.assertIn("speakers_json", str(context.exception.detail))

    def test_parse_speakers_json_accepts_valid_payload(self) -> None:
        speakers = parse_speakers_json('[{"id": 0, "name": "Speaker 1"}]')

        self.assertEqual(len(speakers), 1)
        self.assertEqual(speakers[0].id, 0)
        self.assertEqual(speakers[0].name, "Speaker 1")

    def test_validate_speaker_request_rejects_count_mismatch(self) -> None:
        speakers = parse_speakers_json('[{"id": 0, "name": "Speaker 1"}]')

        with self.assertRaises(HTTPException) as context:
            validate_speaker_request(speakers, speaker_count=2)

        self.assertEqual(context.exception.status_code, 422)
        self.assertIn("speaker_count", str(context.exception.detail))

    def test_validate_speaker_request_rejects_blank_names(self) -> None:
        speakers = parse_speakers_json('[{"id": 0, "name": "   "}]')

        with self.assertRaises(HTTPException) as context:
            validate_speaker_request(speakers, speaker_count=1)

        self.assertEqual(context.exception.status_code, 422)
        self.assertIn("blank", str(context.exception.detail).lower())

    def test_assign_speaker_id_handles_non_numeric_labels(self) -> None:
        turns = [
            SpeakerTurn(start=0.0, end=1.0, label="SPEAKER_A"),
            SpeakerTurn(start=1.0, end=2.0, label="SPEAKER_B"),
        ]
        speaker_id, speaker_name = assign_speaker_id(
            start=1.1,
            end=1.9,
            turns=turns,
            requested_speakers=[
                {"id": 10, "name": "Alice"},
                {"id": 20, "name": "Bob"},
            ],
        )

        self.assertEqual(speaker_id, 20)
        self.assertEqual(speaker_name, "Bob")

    def test_resolve_requested_language_allows_auto_detection(self) -> None:
        self.assertIsNone(resolve_requested_language(None, None))
        self.assertEqual(resolve_requested_language("  es  ", None), "es")
        self.assertEqual(resolve_requested_language(None, "fr"), "fr")

    def test_shift_segment_times_tolerates_missing_word_timestamps(self) -> None:
        shifted = shift_segment_times(
            [
                {
                    "start": 1.5,
                    "end": 2.5,
                    "words": [
                        {"word": "hello", "start": None, "end": None},
                        {"word": "world", "start": 2.0, "end": None},
                    ],
                }
            ],
            3.0,
        )

        self.assertEqual(shifted[0]["start"], 4.5)
        self.assertEqual(shifted[0]["end"], 5.5)
        self.assertEqual(shifted[0]["words"][0]["start"], 4.5)
        self.assertEqual(shifted[0]["words"][0]["end"], 4.5)
        self.assertEqual(shifted[0]["words"][1]["start"], 5.0)
        self.assertEqual(shifted[0]["words"][1]["end"], 5.0)


if __name__ == "__main__":
    unittest.main()
