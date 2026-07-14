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


@dataclass(slots=True)
class OverlapRegion:
    start: float
    end: float
    speaker_indices: list[int] = field(default_factory=list)

    @property
    def duration(self) -> float:
        return self.end - self.start


def _boundaries(turns: list[SpeakerTurn]) -> list[float]:
    marks = {round(turn.start, 3) for turn in turns} | {round(turn.end, 3) for turn in turns}
    return sorted(marks)


def _active_at(turns: list[SpeakerTurn], start: float, end: float) -> set[int]:
    # Boundaries are rounded to 3 decimals, so allow a matching tolerance when
    # testing whether a turn covers the interval between two boundaries.
    tolerance = 5e-3
    active: set[int] = set()
    for turn in turns:
        if turn.start <= start + tolerance and turn.end >= end - tolerance:
            active.add(int(turn.label))
    return active


def find_overlap_regions(
    turns: list[SpeakerTurn],
    min_duration: float = MIN_OVERLAP_SECONDS,
    merge_gap: float = MERGE_GAP_SECONDS,
) -> list[OverlapRegion]:
    """Spans where two or more speakers talk at once, merged across tiny gaps."""
    if not turns:
        return []

    regions: list[OverlapRegion] = []
    marks = _boundaries(turns)
    for start, end in zip(marks, marks[1:]):
        if end - start <= 1e-6:
            continue
        active = _active_at(turns, start, end)
        if len(active) < 2:
            continue
        previous = regions[-1] if regions else None
        if previous is not None and start - previous.end <= merge_gap:
            previous.end = end
            previous.speaker_indices = sorted(set(previous.speaker_indices) | active)
        else:
            regions.append(OverlapRegion(start=start, end=end, speaker_indices=sorted(active)))

    return [region for region in regions if region.duration >= min_duration]


def find_solo_spans(turns: list[SpeakerTurn], speaker_index: int) -> list[tuple[float, float]]:
    """Spans where exactly this speaker talks (no one else active)."""
    if not turns:
        return []

    spans: list[tuple[float, float]] = []
    marks = _boundaries(turns)
    for start, end in zip(marks, marks[1:]):
        if end - start <= 1e-6:
            continue
        active = _active_at(turns, start, end)
        if active == {speaker_index}:
            if spans and abs(spans[-1][1] - start) <= 1e-6:
                spans[-1] = (spans[-1][0], end)
            else:
                spans.append((start, end))
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
