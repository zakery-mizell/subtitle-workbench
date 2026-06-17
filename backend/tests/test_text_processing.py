import unittest

from backend.app.schemas import Caption, WordToken
from backend.app.text_processing import build_guide_blocks, build_words


class TextProcessingTests(unittest.TestCase):
    def test_build_words_tolerates_missing_alignment_values(self) -> None:
        words = build_words(
            [
                {
                    "start": 10.0,
                    "end": 12.0,
                    "avg_logprob": -0.5,
                    "words": [
                        {"word": "hello", "start": None, "end": None, "probability": None},
                        {"word": "world", "start": 11.2, "end": None, "score": None},
                    ],
                }
            ],
            lambda _start, _end: (0, "Speaker 1"),
        )

        self.assertEqual([word.text for word in words], ["hello", "world"])
        self.assertEqual(words[0].start, 10.0)
        self.assertGreater(words[0].end, words[0].start)
        self.assertAlmostEqual(words[0].confidence, 0.9)
        self.assertEqual(words[1].start, 11.2)
        self.assertGreater(words[1].end, words[1].start)
        self.assertAlmostEqual(words[1].confidence, 0.9)

    def test_build_words_estimates_word_timing_when_alignment_is_missing(self) -> None:
        words = build_words(
            [
                {
                    "start": 2.0,
                    "end": 5.0,
                    "text": "Hello world again.",
                    "avg_logprob": -0.25,
                }
            ],
            lambda _start, _end: (0, "Speaker 1"),
        )

        self.assertEqual([word.text for word in words], ["Hello", "world", "again."])
        self.assertEqual(words[0].start, 2.0)
        self.assertEqual(words[-1].end, 5.0)
        self.assertTrue(all(word.end > word.start for word in words))
        self.assertTrue(all(left.end <= right.start for left, right in zip(words, words[1:])))

    def test_build_guide_blocks_adds_silence_cut_and_repeat_blocks(self) -> None:
        words = [
            WordToken(id="w0", text="um", start=0.0, end=0.2, confidence=0.9, low_confidence=False),
            WordToken(id="w1", text="um", start=0.22, end=0.4, confidence=0.9, low_confidence=False),
            WordToken(id="w2", text="hello", start=12.0, end=12.3, confidence=0.9, low_confidence=False),
            WordToken(id="w3", text="hello", start=12.32, end=12.6, confidence=0.9, low_confidence=False),
        ]
        captions = [
            Caption(id="c0", start=0.0, end=0.4, lines=["um um"], word_ids=["w0", "w1"]),
            Caption(id="c1", start=12.0, end=12.5, lines=["hello hello"], word_ids=["w2", "w3"]),
        ]

        blocks = build_guide_blocks(words, captions)
        labels = [block.label for block in blocks]

        self.assertIn("CUT", labels)
        self.assertIn("REPEAT", labels)
        self.assertIn("SILENT", labels)


if __name__ == "__main__":
    unittest.main()
