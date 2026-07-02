import re
import unittest

from backend.app.schemas import WordToken
from backend.app.text_processing import (
    CAPTION_MAX_CHARS,
    CAPTION_MAX_LINE_CHARS,
    build_captions,
    split_caption_lines,
)

SENTENCE_END = re.compile(r"[.!?][\"')\]]?$")


def make_words(text: str, start: float = 0.0, gap: float = 0.05, word_duration: float = 0.25, speaker_id: int | None = 0) -> list[WordToken]:
    words: list[WordToken] = []
    cursor = start
    for index, token in enumerate(text.split()):
        words.append(
            WordToken(
                id=f"w-{index}-{start}",
                text=token,
                start=cursor,
                end=cursor + word_duration,
                confidence=0.9,
                low_confidence=False,
                speaker_id=speaker_id,
                speaker_name="Speaker 1" if speaker_id is not None else None,
            )
        )
        cursor += word_duration + gap
    return words


def caption_text(caption) -> str:
    return " ".join(caption.lines)


def internal_sentence_positions(caption) -> list[int]:
    """Token indexes (not last) that end a sentence inside the caption."""
    tokens = caption_text(caption).split()
    return [i for i in range(len(tokens) - 1) if SENTENCE_END.search(tokens[i])]


class SentenceBoundaryTests(unittest.TestCase):
    def test_each_sentence_gets_its_own_caption(self) -> None:
        words = make_words(
            "This is the first sentence of the recording. Here comes a second sentence right after it."
        )
        captions = build_captions(words)
        self.assertEqual(len(captions), 2)
        self.assertTrue(caption_text(captions[0]).endswith("recording."))
        self.assertTrue(caption_text(captions[1]).startswith("Here"))

    def test_sentence_tail_never_glues_to_next_sentence_head(self) -> None:
        words = make_words(
            "We talked about the roadmap for a while and eventually agreed on a plan. Then the meeting ended "
            "and everyone went home for the day. The next morning brought a new set of problems to solve."
        )
        for caption in build_captions(words):
            # A sentence end inside a caption would mean a next sentence started mid-caption.
            self.assertEqual(internal_sentence_positions(caption), [], caption_text(caption))

    def test_short_sentences_may_share_a_caption(self) -> None:
        words = make_words("Yeah. Exactly. That is what I meant.")
        captions = build_captions(words)
        self.assertEqual(caption_text(captions[0]), "Yeah. Exactly.")
        self.assertTrue(caption_text(captions[1]).startswith("That"))

    def test_short_sentence_merge_respects_pauses(self) -> None:
        first = make_words("Yeah.", start=0.0)
        second = make_words("Exactly.", start=first[-1].end + 1.0)
        captions = build_captions(first + second)
        self.assertEqual(len(captions), 2)

    def test_speaker_change_forces_a_boundary(self) -> None:
        first = make_words("I think we should ship it", speaker_id=0)
        second = make_words("no way that is ready", start=first[-1].end + 0.1, speaker_id=1)
        captions = build_captions(first + second)
        self.assertEqual(len(captions), 2)
        self.assertEqual(captions[0].speaker_id, 0)
        self.assertEqual(captions[1].speaker_id, 1)

    def test_abbreviations_do_not_end_sentences(self) -> None:
        words = make_words("Dr. Smith arrived early and greeted everyone warmly.")
        captions = build_captions(words)
        self.assertEqual(len(captions), 1)


class LongSentenceTests(unittest.TestCase):
    LONG_SENTENCE = (
        "The engineering team spent the entire afternoon debugging the deployment pipeline, "
        "reviewing every configuration file in the repository, and eventually discovered that "
        "a single missing environment variable had silently broken the staging environment for everyone."
    )

    def test_long_sentence_splits_into_capped_chunks(self) -> None:
        captions = build_captions(make_words(self.LONG_SENTENCE))
        self.assertGreater(len(captions), 1)
        for caption in captions:
            self.assertLessEqual(len(caption_text(caption)), CAPTION_MAX_CHARS, caption_text(caption))

    def test_long_sentence_chunks_never_merge_with_next_sentence(self) -> None:
        words = make_words(self.LONG_SENTENCE + " Nobody was surprised.")
        captions = build_captions(words)
        for caption in captions:
            self.assertEqual(internal_sentence_positions(caption), [], caption_text(caption))

    def test_split_prefers_clause_punctuation(self) -> None:
        captions = build_captions(make_words(self.LONG_SENTENCE))
        # At least one break should land right after a comma.
        boundary_tokens = [caption_text(c).split()[-1] for c in captions[:-1]]
        self.assertTrue(any(token.endswith(",") for token in boundary_tokens), boundary_tokens)


class LineRuleTests(unittest.TestCase):
    def test_lines_never_exceed_two(self) -> None:
        for text in (
            "Short.",
            "A medium length caption that needs two balanced lines to display nicely.",
            LongSentenceTests.LONG_SENTENCE,
        ):
            for caption in build_captions(make_words(text)):
                self.assertLessEqual(len(caption.lines), 2, caption.lines)

    def test_single_line_when_it_fits(self) -> None:
        lines = split_caption_lines("This fits on one line easily.")
        self.assertEqual(len(lines), 1)

    def test_two_lines_when_over_line_length(self) -> None:
        text = "This caption is clearly longer than the per line limit and must wrap."
        lines = split_caption_lines(text)
        self.assertEqual(len(lines), 2)
        for line in lines:
            self.assertLessEqual(len(line), CAPTION_MAX_LINE_CHARS + 14, line)

    def test_merged_sentences_break_lines_at_the_sentence_boundary(self) -> None:
        lines = split_caption_lines("Absolutely, I could not agree more. Let us do it.")
        self.assertEqual(lines[0], "Absolutely, I could not agree more.")
        self.assertEqual(lines[1], "Let us do it.")


class TimingTests(unittest.TestCase):
    def test_short_captions_extend_toward_min_duration(self) -> None:
        first = make_words("Yes.", start=0.0)
        second = make_words("The full explanation takes much longer than that.", start=5.0)
        captions = build_captions(first + second)
        self.assertGreaterEqual(captions[0].end - captions[0].start, 1.0)
        self.assertLessEqual(captions[0].end, 5.0)

    def test_blank_after_long_gaps(self) -> None:
        first = make_words("First thought ends here.", start=0.0)
        second = make_words("A new thought begins now.", start=first[-1].end + 4.0)
        captions = build_captions(first + second)
        self.assertTrue(captions[0].blank_after)
        self.assertFalse(captions[-1].blank_after)


class DeterminismTests(unittest.TestCase):
    def test_same_input_gives_same_output(self) -> None:
        words = make_words(LongSentenceTests.LONG_SENTENCE + " Nobody was surprised. Yeah. Exactly.")
        first = [(caption_text(c), c.start, c.end) for c in build_captions(words)]
        second = [(caption_text(c), c.start, c.end) for c in build_captions(words)]
        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
