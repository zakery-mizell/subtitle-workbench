from __future__ import annotations

"""Overlap-region and solo-span math on diarization turns.

Works on the *raw* (non-exclusive) diarization annotation, where turns from
different speakers may overlap. Speaker labels are appearance-order indices
("0", "1", ...) as produced by app.diarization.
"""

from dataclasses import dataclass, field

from ..diarization import SpeakerTurn

MIN_OVERLAP_SECONDS = 0.4
MERGE_GAP_SECONDS = 0.4
MIN_ENROLL_SECONDS = 1.5
BEST_ENROLL_SECONDS = 5.0

# A stretch of simultaneous speech shorter than this is diarizer noise (a turn
# boundary that landed a few frames late), not two people actually talking.
MIN_SIMULTANEOUS_SECONDS = 0.12
# Merging across gaps must never turn a region into mostly solo audio.
MIN_REGION_DENSITY = 0.5
# A speaker only belongs to a region if they genuinely talk over someone in it.
MIN_SPEAKER_SHARE_SECONDS = 0.15
# Enrollment may span a pause; silence in the sample is harmless, other voices are not.
SOLO_BRIDGE_SECONDS = 1.0
# Padding around a speaker's turns when muting their full-length stem outside
# them: generous enough that breaths and clipped word tails survive diarizer
# jitter, small enough that hallucinated speech in long silences is cut.
GATE_PAD_SECONDS = 0.35
GATE_MERGE_SECONDS = 0.6


@dataclass(slots=True)
class OverlapRegion:
    start: float
    end: float
    speaker_indices: list[int] = field(default_factory=list)

    @property
    def duration(self) -> float:
        return self.end - self.start


@dataclass(slots=True)
class _Chunk:
    """One uninterrupted stretch of simultaneous speech (the active set may change)."""

    start: float
    end: float
    pieces: list[tuple[float, frozenset[int]]]  # (duration, active speakers)

    @property
    def duration(self) -> float:
        return self.end - self.start


def _boundaries(turns: list[SpeakerTurn]) -> list[float]:
    marks = {round(turn.start, 3) for turn in turns} | {round(turn.end, 3) for turn in turns}
    return sorted(marks)


def _active_at(turns: list[SpeakerTurn], start: float, end: float) -> set[int]:
    # Boundaries are rounded to 3 decimals, so allow a matching tolerance when
    # testing whether a turn covers the interval between two boundaries. Any
    # larger and a 1-4 ms handoff gap reads as both speakers being active.
    tolerance = 1e-3
    active: set[int] = set()
    for turn in turns:
        if turn.start <= start + tolerance and turn.end >= end - tolerance:
            active.add(int(turn.label))
    return active


def _simultaneous_chunks(turns: list[SpeakerTurn]) -> list[_Chunk]:
    """Contiguous stretches with two or more speakers active, in timeline order."""
    chunks: list[_Chunk] = []
    marks = _boundaries(turns)
    for start, end in zip(marks, marks[1:]):
        if end - start <= 1e-6:
            continue
        active = _active_at(turns, start, end)
        if len(active) < 2:
            continue
        previous = chunks[-1] if chunks else None
        if previous is not None and start - previous.end <= 1e-6:
            previous.end = end
            previous.pieces.append((end - start, frozenset(active)))
        else:
            chunks.append(_Chunk(start=start, end=end, pieces=[(end - start, frozenset(active))]))
    return chunks


def _region_speakers(group: list[_Chunk]) -> list[int]:
    """Speakers that actually talk over someone for a meaningful slice of the region."""
    totals: dict[int, float] = {}
    for chunk in group:
        for duration, active in chunk.pieces:
            for index in active:
                totals[index] = totals.get(index, 0.0) + duration
    ranked = sorted(totals, key=lambda index: (-totals[index], index))
    speakers = [index for index in ranked if totals[index] >= MIN_SPEAKER_SHARE_SECONDS]
    if len(speakers) < 2:
        # A region is simultaneous speech by definition, so never report fewer than two.
        speakers = ranked[:2]
    return sorted(speakers)


def find_overlap_regions(
    turns: list[SpeakerTurn],
    min_duration: float = MIN_OVERLAP_SECONDS,
    merge_gap: float = MERGE_GAP_SECONDS,
) -> list[OverlapRegion]:
    """Spans where two or more speakers talk at once, merged across tiny gaps.

    Merging bridges diarization jitter only: a region stops growing once the
    speech it actually contains drops below MIN_REGION_DENSITY of its span, so
    a speaker over periodic backchannels yields several tight regions instead of
    one that swallows minutes of solo audio.
    """
    if not turns:
        return []

    floor = min(MIN_SIMULTANEOUS_SECONDS, min_duration)
    chunks = [chunk for chunk in _simultaneous_chunks(turns) if chunk.duration >= floor]

    groups: list[list[_Chunk]] = []
    simultaneous: list[float] = []
    for chunk in chunks:
        if groups:
            current = groups[-1]
            gap = chunk.start - current[-1].end
            span = chunk.end - current[0].start
            if gap <= merge_gap and (simultaneous[-1] + chunk.duration) / span >= MIN_REGION_DENSITY:
                current.append(chunk)
                simultaneous[-1] += chunk.duration
                continue
        groups.append([chunk])
        simultaneous.append(chunk.duration)

    return [
        OverlapRegion(start=group[0].start, end=group[-1].end, speaker_indices=_region_speakers(group))
        for group, total in zip(groups, simultaneous)
        if total >= min_duration
    ]


def speaker_turn_spans(
    turns: list[SpeakerTurn],
    speaker_index: int,
    pad: float = GATE_PAD_SECONDS,
    merge_gap: float = GATE_MERGE_SECONDS,
) -> list[tuple[float, float]]:
    """Padded, merged spans where this speaker talks — alone or over someone.

    This is the keep-envelope for a full-length stem: everything outside it is
    silence as far as this speaker is concerned, so it can be muted without
    touching any of their actual speech.
    """
    spans = sorted(
        (max(0.0, turn.start - pad), turn.end + pad)
        for turn in turns
        if int(turn.label) == speaker_index
    )
    merged: list[list[float]] = []
    for start, end in spans:
        if merged and start <= merged[-1][1] + merge_gap:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return [(start, end) for start, end in merged]


def find_solo_spans(turns: list[SpeakerTurn], speaker_index: int) -> list[tuple[float, float]]:
    """Spans where exactly this speaker talks (no one else active).

    Short pauses are bridged, so a span may contain internal silence — that is
    fine for enrollment and keeps conversational speech from being shredded into
    unusably short fragments.
    """
    if not turns:
        return []

    spans: list[tuple[float, float]] = []
    blocked = True  # another voice (or nothing yet) sits between us and the last span
    marks = _boundaries(turns)
    for start, end in zip(marks, marks[1:]):
        if end - start <= 1e-6:
            continue
        active = _active_at(turns, start, end)
        if active == {speaker_index}:
            if spans and not blocked and start - spans[-1][1] <= SOLO_BRIDGE_SECONDS:
                spans[-1] = (spans[-1][0], end)
            else:
                spans.append((start, end))
            blocked = False
        elif active:
            blocked = True
    return spans


def pick_enrollment_span(
    turns: list[SpeakerTurn],
    speaker_index: int,
    near: float,
    min_duration: float = MIN_ENROLL_SECONDS,
    best_duration: float = BEST_ENROLL_SECONDS,
) -> tuple[float, float] | None:
    """Pick the solo span to use as the speaker's voice sample (enrollment).

    Prefers spans of at least `best_duration`; among those, the one closest to
    `near` (the overlap being processed) so the voice matches current mic/room
    conditions. Falls back to the longest span at or above `min_duration`.
    """
    spans = find_solo_spans(turns, speaker_index)
    usable = [span for span in spans if span[1] - span[0] >= min_duration]
    if not usable:
        return None

    def distance(span: tuple[float, float]) -> float:
        start, end = span
        if start <= near <= end:
            return 0.0
        return min(abs(near - end), abs(start - near))

    long_enough = [span for span in usable if span[1] - span[0] >= best_duration]
    if long_enough:
        best = min(long_enough, key=distance)
    else:
        best = max(usable, key=lambda span: span[1] - span[0])

    start, end = best
    if end - start > best_duration:
        # Trim to the best_duration window inside the span closest to `near`.
        if near <= start:
            end = start + best_duration
        elif near >= end:
            start = end - best_duration
        else:
            half = best_duration / 2.0
            center = min(max(near, start + half), end - half)
            start, end = center - half, center + half
    return (start, end)
