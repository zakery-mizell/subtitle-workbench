import unittest

from fastapi import HTTPException

from backend.app.diarization import SpeakerTurn, _extract_turns, assign_speaker_id
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

    def test_extract_turns_prefers_exclusive_diarization(self) -> None:
        class FakeSegment:
            def __init__(self, start: float, end: float) -> None:
                self.start = start
                self.end = end

        class FakeAnnotation:
            def __init__(self, tracks: list[tuple[float, float, str]]) -> None:
                self._tracks = tracks

            def itertracks(self, yield_label: bool = False):
                for start, end, label in self._tracks:
                    yield FakeSegment(start, end), None, label

        class FakeOutput:
            exclusive_speaker_diarization = FakeAnnotation([(0.0, 1.5, "SPEAKER_00")])
            speaker_diarization = FakeAnnotation([(0.0, 2.0, "SPEAKER_00"), (1.0, 2.0, "SPEAKER_01")])

        turns, raw_turns = _extract_turns(FakeOutput())
        self.assertEqual(turns, [SpeakerTurn(start=0.0, end=1.5, label="0")])
        # The raw annotation keeps the genuine overlap and shares labels.
        self.assertEqual(
            raw_turns,
            [
                SpeakerTurn(start=0.0, end=2.0, label="0"),
                SpeakerTurn(start=1.0, end=2.0, label="1"),
            ],
        )

        # Legacy pipelines return a bare annotation instead of a DiarizeOutput.
        legacy, legacy_raw = _extract_turns(FakeAnnotation([(0.5, 1.0, "SPEAKER_01")]))
        self.assertEqual(legacy, [SpeakerTurn(start=0.5, end=1.0, label="0")])
        self.assertEqual(legacy, legacy_raw)

        # Cluster labels are arbitrary: whoever speaks first becomes speaker 0.
        reordered, _ = _extract_turns(
            FakeAnnotation([(0.0, 1.0, "SPEAKER_01"), (1.0, 2.0, "SPEAKER_00"), (2.0, 3.0, "SPEAKER_01")])
        )
        self.assertEqual(
            reordered,
            [
                SpeakerTurn(start=0.0, end=1.0, label="0"),
                SpeakerTurn(start=1.0, end=2.0, label="1"),
                SpeakerTurn(start=2.0, end=3.0, label="0"),
            ],
        )

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
