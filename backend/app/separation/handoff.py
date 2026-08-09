from __future__ import annotations

"""Conservative word-speaker reconciliation around rapid handoffs.

UniSE is an extractor, not a speaker classifier.  This module turns the
time-aligned transcripts of its *ungated* target-speaker stems into supporting
evidence for the mixture transcript.  It only proposes a change when the word
is close to an existing fast speaker boundary, is clearly present in the other
speaker's stem, and is clearly absent from the currently assigned stem.
"""

from dataclasses import dataclass
import math
import re
import unicodedata

from .schemas import AuditWordInput, HandoffCorrection, StemWord

MAX_HANDOFF_GAP_SECONDS = 0.45
MAX_HANDOFF_OVERLAP_SECONDS = 1.25
AUDIT_RADIUS_SECONDS = 1.0
TARGET_WINDOW_RADIUS_SECONDS = 1.25
TARGET_WINDOW_MERGE_GAP_SECONDS = 0.35
MAX_STEM_CENTER_DISTANCE_SECONDS = 0.5
MIN_ALTERNATIVE_EVIDENCE = 0.76
MAX_CURRENT_EVIDENCE = 0.44
MIN_EVIDENCE_MARGIN = 0.36
SEGMENT_DOMINANCE = 1.5
MIN_SEGMENT_GUARD_WORDS = 3
MAX_ISOLATED_UTTERANCE_WORDS = 3
MAX_ISOLATED_UTTERANCE_SECONDS = 1.25
MIN_ISOLATION_GAP_SECONDS = 0.30
MAX_ISOLATION_GAP_SECONDS = 2.0
MIN_BEST_ALTERNATIVE_MARGIN = 0.12
ALIGNED_WORD_ID = re.compile(r"^(\d+)-\d+$")


@dataclass(frozen=True, slots=True)
class _Boundary:
    time: float
    left_speaker: int
    right_speaker: int


@dataclass(frozen=True, slots=True)
class HandoffWindow:
    """A short UniSE work area around one or more rapid speaker changes."""

    start: float
    end: float
    speaker_indices: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class IsolatedUtterance:
    start: float
    end: float
    speaker_index: int
    word_ids: tuple[str, ...]


def _token(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text).casefold()
    return "".join(character for character in normalized if unicodedata.category(character)[0] in {"L", "N"})


def _boundaries(words: list[AuditWordInput]) -> list[_Boundary]:
    ordered = sorted(words, key=lambda word: (word.start, word.end, word.id))
    boundaries: list[_Boundary] = []
    for previous, current in zip(ordered, ordered[1:]):
        if previous.speaker_index == current.speaker_index:
            continue
        gap = current.start - previous.end
        if gap > MAX_HANDOFF_GAP_SECONDS or gap < -MAX_HANDOFF_OVERLAP_SECONDS:
            continue
        # In an overlap the midpoint of the two nearest edges remains inside the
        # contested area; in a handoff it lands in the gap between speakers.
        time = (previous.end + current.start) / 2.0
        candidate = _Boundary(time, previous.speaker_index, current.speaker_index)
        if boundaries and abs(boundaries[-1].time - time) < 0.08:
            continue
        boundaries.append(candidate)
    return boundaries


def isolated_utterances(words: list[AuditWordInput]) -> list[IsolatedUtterance]:
    """Find short pause-bounded ASR segments that can hide a missed handoff."""
    ordered = sorted(words, key=lambda word: (word.start, word.end, word.id))
    positions = {word.id: index for index, word in enumerate(ordered)}
    by_segment: dict[str, list[AuditWordInput]] = {}
    for word in ordered:
        match = ALIGNED_WORD_ID.fullmatch(word.id)
        if match is not None:
            by_segment.setdefault(match.group(1), []).append(word)

    candidates: list[IsolatedUtterance] = []
    for segment_words in by_segment.values():
        segment_words.sort(key=lambda word: (word.start, word.end, word.id))
        if not 1 <= len(segment_words) <= MAX_ISOLATED_UTTERANCE_WORDS:
            continue
        speakers = {word.speaker_index for word in segment_words}
        if len(speakers) != 1:
            continue
        start, end = segment_words[0].start, segment_words[-1].end
        if end - start > MAX_ISOLATED_UTTERANCE_SECONDS:
            continue
        first_position = positions[segment_words[0].id]
        last_position = positions[segment_words[-1].id]
        if first_position == 0 or last_position >= len(ordered) - 1:
            continue
        gap_before = start - ordered[first_position - 1].end
        gap_after = ordered[last_position + 1].start - end
        if not (
            MIN_ISOLATION_GAP_SECONDS <= gap_before <= MAX_ISOLATION_GAP_SECONDS
            and MIN_ISOLATION_GAP_SECONDS <= gap_after <= MAX_ISOLATION_GAP_SECONDS
        ):
            continue
        candidates.append(
            IsolatedUtterance(
                start=start,
                end=end,
                speaker_index=next(iter(speakers)),
                word_ids=tuple(word.id for word in segment_words),
            )
        )
    return candidates


def handoff_windows(words: list[AuditWordInput]) -> list[HandoffWindow]:
    """Return merged UniSE windows around handoffs and isolated utterances."""
    all_speakers = tuple(sorted({word.speaker_index for word in words}))
    raw = [
        HandoffWindow(
            start=max(0.0, boundary.time - TARGET_WINDOW_RADIUS_SECONDS),
            end=boundary.time + TARGET_WINDOW_RADIUS_SECONDS,
            speaker_indices=tuple(sorted({boundary.left_speaker, boundary.right_speaker})),
        )
        for boundary in _boundaries(words)
    ]
    if len(all_speakers) >= 2:
        raw.extend(
            HandoffWindow(
                start=max(0.0, (utterance.start + utterance.end) / 2.0 - TARGET_WINDOW_RADIUS_SECONDS),
                end=(utterance.start + utterance.end) / 2.0 + TARGET_WINDOW_RADIUS_SECONDS,
                speaker_indices=all_speakers,
            )
            for utterance in isolated_utterances(words)
        )
    raw.sort(key=lambda window: (window.start, window.end))
    merged: list[HandoffWindow] = []
    for window in raw:
        if merged and window.start <= merged[-1].end + TARGET_WINDOW_MERGE_GAP_SECONDS:
            previous = merged[-1]
            merged[-1] = HandoffWindow(
                start=previous.start,
                end=max(previous.end, window.end),
                speaker_indices=tuple(sorted(set(previous.speaker_indices) | set(window.speaker_indices))),
            )
        else:
            merged.append(window)
    return merged


def _time_evidence(word: AuditWordInput, stem_word: StemWord) -> float:
    overlap = max(0.0, min(word.end, stem_word.end) - max(word.start, stem_word.start))
    shorter = max(0.06, min(word.end - word.start, stem_word.end - stem_word.start))
    if overlap > 0:
        timing = 0.74 + 0.24 * min(1.0, overlap / shorter)
    else:
        center = (word.start + word.end) / 2.0
        stem_center = (stem_word.start + stem_word.end) / 2.0
        distance = abs(center - stem_center)
        if distance > MAX_STEM_CENTER_DISTANCE_SECONDS:
            return 0.0
        timing = 0.68 * math.exp(-distance / 0.24)

    confidence = stem_word.confidence
    if confidence is None:
        return timing
    return timing * (0.85 + 0.15 * max(0.0, min(1.0, confidence)))


def _evidence(word: AuditWordInput, stem_words: list[StemWord]) -> float:
    token = _token(word.text)
    if not token:
        return 0.0
    return max(
        (_time_evidence(word, candidate) for candidate in stem_words if _token(candidate.text) == token),
        default=0.0,
    )


def _segment_preferences(words: list[AuditWordInput]) -> dict[str, int]:
    """Return a clear within-ASR-segment speaker majority when one exists."""
    weights: dict[str, dict[int, float]] = {}
    counts: dict[str, int] = {}
    for word in words:
        match = ALIGNED_WORD_ID.fullmatch(word.id)
        if match is None:
            continue
        speaker_weights = weights.setdefault(match.group(1), {})
        counts[match.group(1)] = counts.get(match.group(1), 0) + 1
        speaker_weights[word.speaker_index] = speaker_weights.get(word.speaker_index, 0.0) + max(
            0.02, word.end - word.start
        )

    preferences: dict[str, int] = {}
    for segment, speaker_weights in weights.items():
        if counts.get(segment, 0) < MIN_SEGMENT_GUARD_WORDS:
            continue
        ranked = sorted(speaker_weights.items(), key=lambda item: item[1], reverse=True)
        if len(ranked) == 1 or ranked[0][1] >= ranked[1][1] * SEGMENT_DOMINANCE:
            preferences[segment] = ranked[0][0]
    return preferences


def audit_handoff_assignments(
    words: list[AuditWordInput],
    stem_words_by_speaker: dict[int, list[StemWord]],
) -> tuple[list[HandoffCorrection], int]:
    """Return high-confidence corrections and the number of boundaries audited."""
    boundaries = _boundaries(words)
    isolated = isolated_utterances(words)
    if (not boundaries and not isolated) or len(stem_words_by_speaker) < 2:
        return [], len(boundaries) + len(isolated)

    proposals: dict[str, HandoffCorrection] = {}
    evidence_by_word: dict[tuple[str, int], float] = {}
    segment_preferences = _segment_preferences(words)

    def evidence(word: AuditWordInput, speaker: int) -> float:
        key = (word.id, speaker)
        if key not in evidence_by_word:
            evidence_by_word[key] = _evidence(word, stem_words_by_speaker.get(speaker, []))
        return evidence_by_word[key]

    for boundary in boundaries:
        pair = {boundary.left_speaker, boundary.right_speaker}
        for word in words:
            if word.speaker_index not in pair:
                continue
            center = (word.start + word.end) / 2.0
            if abs(center - boundary.time) > AUDIT_RADIUS_SECONDS:
                continue
            alternative = (
                boundary.right_speaker
                if word.speaker_index == boundary.left_speaker
                else boundary.left_speaker
            )
            segment_match = ALIGNED_WORD_ID.fullmatch(word.id)
            preferred_speaker = (
                segment_preferences.get(segment_match.group(1)) if segment_match is not None else None
            )
            # UniSE extraction can leak a coherent utterance into the wrong
            # enrolled stem. It may repair an outlying segment-edge word, but it
            # may not fragment a segment whose current speaker already has a
            # clear aligned-word majority.
            if preferred_speaker == word.speaker_index and alternative != preferred_speaker:
                continue
            current_score = evidence(word, word.speaker_index)
            alternative_score = evidence(word, alternative)
            margin = alternative_score - current_score
            if (
                alternative_score < MIN_ALTERNATIVE_EVIDENCE
                or current_score > MAX_CURRENT_EVIDENCE
                or margin < MIN_EVIDENCE_MARGIN
            ):
                continue
            confidence = max(0.0, min(1.0, 0.55 * alternative_score + 0.45 * margin))
            correction = HandoffCorrection(
                word_id=word.id,
                from_speaker_index=word.speaker_index,
                to_speaker_index=alternative,
                confidence=round(confidence, 3),
                boundary_time=round(boundary.time, 3),
            )
            previous = proposals.get(word.id)
            if previous is None or correction.confidence > previous.confidence:
                proposals[word.id] = correction

    # A short pause-bounded utterance can be entirely assigned to the wrong
    # speaker, leaving no visible boundary to audit. Compare it against every
    # extracted speaker and require one unambiguous alternative winner.
    words_by_id = {word.id: word for word in words}
    for utterance in isolated:
        for word_id in utterance.word_ids:
            word = words_by_id[word_id]
            current_score = evidence(word, word.speaker_index)
            alternatives = sorted(
                (
                    (speaker, evidence(word, speaker))
                    for speaker in stem_words_by_speaker
                    if speaker != word.speaker_index
                ),
                key=lambda item: item[1],
                reverse=True,
            )
            if not alternatives:
                continue
            alternative, alternative_score = alternatives[0]
            runner_up = alternatives[1][1] if len(alternatives) > 1 else 0.0
            margin = alternative_score - current_score
            if (
                alternative_score < MIN_ALTERNATIVE_EVIDENCE
                or current_score > MAX_CURRENT_EVIDENCE
                or margin < MIN_EVIDENCE_MARGIN
                or alternative_score - runner_up < MIN_BEST_ALTERNATIVE_MARGIN
            ):
                continue
            confidence = max(0.0, min(1.0, 0.55 * alternative_score + 0.45 * margin))
            correction = HandoffCorrection(
                word_id=word.id,
                from_speaker_index=word.speaker_index,
                to_speaker_index=alternative,
                confidence=round(confidence, 3),
                boundary_time=round((utterance.start + utterance.end) / 2.0, 3),
            )
            previous = proposals.get(word.id)
            if previous is None or correction.confidence > previous.confidence:
                proposals[word.id] = correction

    # Tiny acknowledgement tokens are especially prone to appearing in both
    # generated stems.  Keep them only when a neighbouring word supports the
    # same direction of correction.
    ordered = sorted(words, key=lambda word: (word.start, word.end, word.id))
    position = {word.id: index for index, word in enumerate(ordered)}
    filtered: list[HandoffCorrection] = []
    for correction in proposals.values():
        source = next(word for word in ordered if word.id == correction.word_id)
        if len(_token(source.text)) > 2:
            filtered.append(correction)
            continue
        index = position[source.id]
        neighbours = ordered[max(0, index - 1) : index] + ordered[index + 1 : index + 2]
        corroborated = any(
            (proposal := proposals.get(neighbour.id)) is not None
            and proposal.from_speaker_index == correction.from_speaker_index
            and proposal.to_speaker_index == correction.to_speaker_index
            for neighbour in neighbours
        )
        if corroborated:
            filtered.append(correction)

    return (
        sorted(filtered, key=lambda item: (item.boundary_time, item.word_id)),
        len(boundaries) + len(isolated),
    )
