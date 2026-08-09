from backend.app.separation.handoff import audit_handoff_assignments, handoff_windows
from backend.app.separation.schemas import AuditWordInput, StemWord


def base(word_id: str, text: str, start: float, end: float, speaker: int) -> AuditWordInput:
    return AuditWordInput(
        id=word_id,
        text=text,
        start=start,
        end=end,
        speaker_index=speaker,
    )


def stem(text: str, start: float, end: float, confidence: float = 0.95) -> StemWord:
    return StemWord(text=text, start=start, end=end, confidence=confidence)


def test_moves_word_that_only_appears_in_outgoing_speakers_stem() -> None:
    words = [
        base("w0", "We", 0.55, 0.72, 0),
        base("w1", "actually", 0.74, 1.05, 1),
        base("w2", "should", 1.08, 1.30, 1),
    ]
    corrections, audited = audit_handoff_assignments(
        words,
        {
            0: [stem("We", 0.55, 0.72), stem("actually,", 0.74, 1.05)],
            1: [stem("should", 1.08, 1.30)],
        },
    )

    assert audited == 1
    assert [(item.word_id, item.from_speaker_index, item.to_speaker_index) for item in corrections] == [
        ("w1", 1, 0)
    ]


def test_keeps_assignment_when_both_stems_contain_the_word() -> None:
    words = [
        base("w0", "No", 2.0, 2.18, 0),
        base("w1", "thanks", 2.20, 2.48, 1),
    ]
    corrections, audited = audit_handoff_assignments(
        words,
        {
            0: [stem("thanks", 2.20, 2.48, 0.82)],
            1: [stem("thanks", 2.20, 2.48, 0.96)],
        },
    )

    assert audited == 1
    assert corrections == []


def test_ignores_words_far_from_a_fast_handoff() -> None:
    words = [
        base("w0", "Earlier", 0.0, 0.3, 1),
        base("w1", "handoff", 5.0, 5.25, 0),
        base("w2", "now", 5.28, 5.45, 1),
    ]
    corrections, audited = audit_handoff_assignments(
        words,
        {
            0: [stem("Earlier", 0.0, 0.3)],
            1: [stem("handoff", 5.0, 5.25), stem("now", 5.28, 5.45)],
        },
    )

    assert audited == 1
    assert [item.word_id for item in corrections] == ["w1"]
    assert all(item.word_id != "w0" for item in corrections)


def test_short_acknowledgement_needs_a_corroborating_word() -> None:
    words = [
        base("w0", "finish", 7.0, 7.28, 0),
        base("w1", "I", 7.30, 7.36, 1),
        base("w2", "agree", 7.38, 7.62, 1),
    ]
    corrections, _ = audit_handoff_assignments(
        words,
        {
            0: [stem("finish", 7.0, 7.28), stem("I", 7.30, 7.36)],
            1: [stem("agree", 7.38, 7.62)],
        },
    )

    assert corrections == []


def test_short_word_moves_with_neighbouring_phrase() -> None:
    words = [
        base("w0", "finish", 8.0, 8.22, 0),
        base("w1", "I", 8.24, 8.30, 1),
        base("w2", "really", 8.31, 8.55, 1),
        base("w3", "agree", 8.58, 8.82, 1),
    ]
    corrections, _ = audit_handoff_assignments(
        words,
        {
            0: [
                stem("finish", 8.0, 8.22),
                stem("I", 8.24, 8.30),
                stem("really", 8.31, 8.55),
            ],
            1: [stem("agree", 8.58, 8.82)],
        },
    )

    assert [item.word_id for item in corrections] == ["w1", "w2"]


def test_handoff_windows_are_short_and_merge_nearby_boundaries() -> None:
    words = [
        base("w0", "one", 9.0, 9.2, 0),
        base("w1", "two", 9.22, 9.42, 1),
        base("w2", "three", 9.44, 9.64, 0),
        base("w3", "later", 20.0, 20.2, 1),
    ]

    windows = handoff_windows(words)

    assert len(windows) == 1
    assert windows[0].start > 7.0
    assert windows[0].end < 12.0
    assert windows[0].speaker_indices == (0, 1)


def test_leaky_stem_cannot_fragment_a_coherent_aligned_segment() -> None:
    words = [
        base("37-0", "So", 138.766, 139.006, 2),
        base("38-0", "That's", 139.338, 139.538, 1),
        base("38-1", "too", 139.578, 139.718, 1),
        base("38-2", "good", 139.738, 139.898, 1),
        base("38-3", "work", 139.979, 140.159, 1),
        base("38-4", "of", 140.199, 140.259, 1),
        base("38-5", "art", 140.339, 140.499, 1),
    ]
    leaking_previous_stem = [
        stem(word.text, word.start, word.end)
        for word in words[1:]
    ]

    corrections, audited = audit_handoff_assignments(
        words,
        {1: [], 2: leaking_previous_stem},
    )

    assert audited == 1
    assert corrections == []


def test_isolated_one_word_reply_is_windowed_and_compared_to_every_speaker() -> None:
    words = [
        base("15-8", "cry", 42.975, 43.255, 2),
        base("16-0", "Never", 43.735, 43.975, 2),  # entirely mislabeled; no boundary
        base("17-0", "No", 44.856, 45.016, 0),
        base("18-0", "Later", 50.0, 50.2, 1),
    ]

    windows = handoff_windows(words)
    assert any(window.start < 43.8 < window.end and window.speaker_indices == (0, 1, 2) for window in windows)

    corrections, audited = audit_handoff_assignments(
        words,
        {
            0: [],
            1: [stem("Never", 43.735, 43.975)],
            2: [],
        },
    )

    assert audited == 1
    assert [(item.word_id, item.from_speaker_index, item.to_speaker_index) for item in corrections] == [
        ("16-0", 2, 1)
    ]
