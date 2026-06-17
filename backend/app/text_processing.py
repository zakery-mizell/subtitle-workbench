from __future__ import annotations

import math
import re
from typing import Callable, Iterable

from .config import settings
from .schemas import Caption, GuideBlock, Paragraph, SpeakerAssignmentMode, WordToken


CLAUSE_BREAK = re.compile(r"(?<=[,;:!?])\s+")
WORD_RE = re.compile(r"[A-Za-z']+")
FILLERS = {"um", "uh", "erm", "hmm"}
DISFLUENCY_FILLERS = {"uh", "um", "ah", "er", "erm", "hmm", "mm", "mhm", "uhh", "umm"}
SHORT_STUTTER_WORDS = {"i", "a", "an", "the", "to", "we", "you", "he", "she", "it", "they"}
SENTENCE_END_RE = re.compile(r"[.!?][\"')\]]?$")
CLAUSE_END_RE = re.compile(r"[,;:][\"')\]]?$")
TITLECASE_TOKEN_RE = re.compile("^[A-Z][A-Za-z'\u2019-]+$")
HONORIFIC_RE = re.compile(r"^(mr|mrs|ms|dr|prof|sir|lady|lord|st)\.?$", re.IGNORECASE)
DOUBLE_QUOTE_RE = re.compile("[\"\u201C\u201D]")
TITLE_CONNECTORS = {
    "a",
    "an",
    "and",
    "for",
    "in",
    "of",
    "on",
    "the",
    "to",
}
ENTITY_INTRODUCERS = {"called", "named", "titled"}
WEAK_LINE_STARTS = {
    "and",
    "but",
    "or",
    "so",
    "because",
    "if",
    "then",
    "than",
    "that",
    "which",
    "who",
    "when",
    "where",
    "to",
    "of",
    "for",
    "with",
    "a",
    "an",
    "the",
}
WEAK_LINE_ENDS = {
    "a",
    "an",
    "the",
    "and",
    "but",
    "or",
    "so",
    "to",
    "of",
    "for",
    "with",
    "at",
    "by",
    "from",
    "in",
    "on",
    "if",
    "than",
    "that",
    "which",
    "who",
    "when",
    "where",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "being",
}


def build_words(
    segments: list[dict],
    speaker_lookup: Callable[[float, float], tuple[int | None, str | None]],
    speaker_assignment_mode: SpeakerAssignmentMode = "segment",
) -> list[WordToken]:
    words: list[WordToken] = []
    for segment_index, segment in enumerate(segments):
        segment_start = _coerce_float(segment.get("start"), 0.0)
        segment_end = _coerce_float(segment.get("end"), segment_start)
        segment_speaker_id, segment_speaker_name = speaker_lookup(segment_start, segment_end)
        segment_words = _segment_word_entries(segment, segment_start, segment_end)
        if not segment_words:
            continue

        for word_index, word in enumerate(segment_words):
            text = str(word.get("word", "")).strip()
            if not text:
                continue
            start = _coerce_float(word.get("start"), segment_start)
            end = max(start, _coerce_float(word.get("end"), start))
            confidence = _coerce_float(
                word.get("probability"),
                _coerce_float(word.get("score"), _segment_confidence(segment)),
            )
            if speaker_assignment_mode == "word":
                speaker_id, speaker_name = speaker_lookup(start, end)
                if speaker_id is None:
                    speaker_id, speaker_name = segment_speaker_id, segment_speaker_name
            else:
                speaker_id, speaker_name = segment_speaker_id, segment_speaker_name
            words.append(
                WordToken(
                    id=f"{segment_index}-{word_index}",
                    text=text,
                    start=start,
                    end=end,
                    confidence=confidence,
                    low_confidence=confidence < settings.low_confidence_threshold,
                    speaker_id=speaker_id,
                    speaker_name=speaker_name,
                )
            )
    return words


def _segment_word_entries(segment: dict, segment_start: float, segment_end: float) -> list[dict]:
    segment_words = segment.get("words") or []
    if segment_words:
        return _repair_word_timing_entries(segment_words, segment_start, segment_end)

    text = (segment.get("text") or "").strip()
    if not text:
        return []

    confidence = _segment_confidence(segment)
    return [
        {
            "word": token,
            "start": start,
            "end": end,
            "probability": confidence,
        }
        for token, start, end in _estimated_token_timings(_split_segment_text(text), segment_start, segment_end)
    ]


def _repair_word_timing_entries(segment_words: list[dict], segment_start: float, segment_end: float) -> list[dict]:
    raw_items = [(word, str(word.get("word", "")).strip()) for word in segment_words]
    raw_items = [(word, token) for word, token in raw_items if token]
    tokens = [token for _, token in raw_items]
    fallback_timings = _estimated_token_timings(tokens, segment_start, segment_end)
    repaired: list[dict] = []

    for (raw_word, _), fallback in zip(raw_items, fallback_timings):
        token, fallback_start, fallback_end = fallback
        if not token:
            continue

        raw_start = _coerce_optional_float(raw_word.get("start"))
        raw_end = _coerce_optional_float(raw_word.get("end"))
        start = raw_start if raw_start is not None else fallback_start
        end = raw_end if raw_end is not None else fallback_end

        if end < start:
            end = start

        if end == start and segment_end > segment_start:
            estimated_duration = max(0.08, min(0.35, (segment_end - segment_start) / max(len(fallback_timings), 1)))
            end = min(segment_end, start + estimated_duration) if start < segment_end else start + estimated_duration

        next_word = dict(raw_word)
        next_word["word"] = token
        next_word["start"] = start
        next_word["end"] = end
        repaired.append(next_word)

    return repaired


def _split_segment_text(text: str) -> list[str]:
    return re.findall(r"\S+", text)


def _estimated_token_timings(tokens: list[str], segment_start: float, segment_end: float) -> list[tuple[str, float, float]]:
    visible_tokens = [token for token in tokens if token.strip()]
    if not visible_tokens:
        return []

    if segment_end <= segment_start:
        return [(token, segment_start, segment_start) for token in visible_tokens]

    segment_duration = segment_end - segment_start
    weights = [max(1, len(normalize_word(token)) or len(token)) for token in visible_tokens]
    total_weight = sum(weights) or len(visible_tokens)
    timings: list[tuple[str, float, float]] = []
    elapsed_weight = 0

    for index, (token, weight) in enumerate(zip(visible_tokens, weights)):
        start = segment_start + (segment_duration * elapsed_weight / total_weight)
        elapsed_weight += weight
        end = segment_end if index == len(visible_tokens) - 1 else segment_start + (segment_duration * elapsed_weight / total_weight)
        timings.append((token, start, max(start, end)))

    return timings


def remove_disfluencies(words: list[WordToken]) -> list[WordToken]:
    if not words:
        return words

    cleaned: list[WordToken] = []
    for word in words:
        normalized = normalize_word(word.text)
        if normalized in DISFLUENCY_FILLERS:
            continue

        if cleaned:
            previous = cleaned[-1]
            previous_normalized = normalize_word(previous.text)
            if (
                normalized
                and normalized == previous_normalized
                and word.start - previous.end <= 0.35
                and (
                    len(normalized) <= 2
                    or normalized in SHORT_STUTTER_WORDS
                    or previous.low_confidence
                    or word.low_confidence
                )
            ):
                cleaned[-1] = word
                continue

        cleaned.append(word)

    return cleaned


def build_paragraphs(words: list[WordToken]) -> list[Paragraph]:
    if not words:
        return []

    paragraphs: list[Paragraph] = []
    current: list[WordToken] = []

    def flush():
        nonlocal current
        if not current:
            return
        paragraphs.append(
            Paragraph(
                id=f"p-{len(paragraphs)}",
                start=current[0].start,
                end=current[-1].end,
                speaker_id=current[0].speaker_id,
                speaker_name=current[0].speaker_name,
                text=_normalize_spacing(" ".join(word.text for word in current)),
                word_ids=[word.id for word in current],
            )
        )
        current = []

    for word in words:
        if not current:
            current.append(word)
            continue

        prev = current[-1]
        text_so_far = _normalize_spacing(" ".join(item.text for item in current))
        should_break = (
            word.speaker_id != prev.speaker_id
            or word.start - prev.end > 1.8
            or (len(text_so_far) > 380 and re.search(r"[.!?][\"')\]]?$", prev.text))
        )

        if should_break:
            flush()

        current.append(word)

    flush()
    return paragraphs


def build_captions(words: list[WordToken]) -> list[Caption]:
    captions: list[Caption] = []
    if not words:
        return captions

    bucket: list[WordToken] = []

    def flush(blank_after: bool = False):
        nonlocal bucket
        if not bucket:
            return
        text = _normalize_spacing(" ".join(word.text for word in bucket))
        lines = split_caption_lines(text)
        captions.append(
            Caption(
                id=f"c-{len(captions)}",
                start=bucket[0].start,
                end=bucket[-1].end,
                speaker_id=bucket[0].speaker_id,
                speaker_name=bucket[0].speaker_name,
                lines=lines,
                word_ids=[word.id for word in bucket],
                blank_after=blank_after,
            )
        )
        bucket = []

    for word in words:
        if not bucket:
            bucket.append(word)
            continue

        candidate = _normalize_spacing(" ".join(item.text for item in [*bucket, word]))
        prev = bucket[-1]
        sentence_break = re.search(r"[.!?][\"')\]]?$", prev.text) is not None
        long_pause = word.start - prev.end > 1.4
        speaker_change = word.speaker_id != prev.speaker_id
        sentence_overflow = sentence_break and len(candidate) > 64
        clause_overflow = len(candidate) > 104

        if speaker_change or long_pause or sentence_overflow:
            flush(blank_after=word.start - prev.end > 2.5)

        elif clause_overflow:
            flush()

        bucket.append(word)

    flush()
    return captions


def build_guide_blocks(words: list[WordToken], captions: list[Caption]) -> list[GuideBlock]:
    blocks: list[GuideBlock] = []

    for index in range(len(captions) - 1):
        current = captions[index]
        nxt = captions[index + 1]
        gap = nxt.start - current.end
        if gap >= settings.silence_seconds:
            blocks.append(
                GuideBlock(
                    id=f"g-{len(blocks)}",
                    start=current.end,
                    end=nxt.start,
                    label="SILENT",
                    reason=f"Silence longer than {settings.silence_seconds:.0f}s",
                    skip=True,
                )
            )

    for start, end, reason in detect_repetition_windows(words):
        label = "CUT" if any(normalize_word(word.text) in FILLERS for word in words if word.start < end and word.end > start) else "REPEAT"
        blocks.append(
            GuideBlock(
                id=f"g-{len(blocks)}",
                start=start,
                end=end,
                label=label,
                reason=reason,
                skip=True,
            )
        )

    return sorted(blocks, key=lambda item: (item.start, item.end))


def detect_repetition_windows(words: list[WordToken]) -> list[tuple[float, float, str]]:
    windows: list[tuple[float, float, str]] = []
    normalized = [normalize_word(word.text) for word in words]
    i = 1
    while i < len(words):
        if not normalized[i] or normalized[i] != normalized[i - 1]:
            i += 1
            continue

        start = i - 1
        end = i
        while end + 1 < len(words) and normalized[end + 1] == normalized[start]:
            end += 1

        repeated = normalized[start]
        if repeated and (repeated in FILLERS or end - start >= 1):
            windows.append(
                (
                    words[start].start,
                    words[end].end,
                    f"Repeated word sequence: {repeated}",
                )
            )
        i = end + 1

    for i in range(len(words) - 4):
        span = normalized[i : i + 5]
        if span[0] and span[:2] == span[2:4]:
            windows.append(
                (
                    words[i].start,
                    words[i + 3].end,
                    "Restarted phrase",
                )
            )
    return merge_windows(windows, max_gap=0.25)


def merge_windows(
    windows: Iterable[tuple[float, float, str]],
    max_gap: float,
) -> list[tuple[float, float, str]]:
    ordered = sorted(windows, key=lambda item: (item[0], item[1]))
    merged: list[tuple[float, float, str]] = []
    for start, end, reason in ordered:
        if not merged:
            merged.append((start, end, reason))
            continue
        prev_start, prev_end, prev_reason = merged[-1]
        if start - prev_end <= max_gap:
            merged[-1] = (prev_start, max(prev_end, end), prev_reason)
        else:
            merged.append((start, end, reason))
    return merged


def split_caption_lines(text: str, target_line_length: int = 64) -> list[str]:
    normalized = _normalize_spacing(text)
    if not normalized:
        return [""]

    if len(normalized) <= target_line_length:
        return [normalized]

    tokens = normalized.split()
    if len(tokens) < 2:
        return [normalized]

    best_lines = [normalized]
    best_score: float | None = None
    hard_cap = max(target_line_length + 28, 92)
    preferred_indexes = (
        [index for index in range(1, len(tokens)) if CLAUSE_END_RE.search(tokens[index - 1])]
        if len(normalized) <= hard_cap
        else []
    )
    candidate_indexes = preferred_indexes or (list(range(1, len(tokens))) if len(normalized) > hard_cap else [])

    if not candidate_indexes:
        return [normalized]

    for index in candidate_indexes:
        left = " ".join(tokens[:index]).strip()
        right = " ".join(tokens[index:]).strip()
        if not left or not right:
            continue

        score = _caption_split_score(left, right, target_line_length=target_line_length, hard_cap=hard_cap)
        if best_score is None or score < best_score:
            best_score = score
            best_lines = [left, right]

    return [_normalize_spacing(line) for line in best_lines]


def normalize_word(text: str) -> str:
    match = WORD_RE.search(text.lower())
    return match.group(0) if match else ""


def _segment_confidence(segment: dict) -> float:
    avg_logprob = float(segment.get("avg_logprob", -0.5))
    scaled = max(0.0, min(1.0, 1.0 + (avg_logprob / 5.0)))
    return scaled


def _coerce_float(value: object, fallback: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return fallback
    if not math.isfinite(parsed):
        return fallback
    return parsed


def _coerce_optional_float(value: object) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(parsed):
        return None
    return parsed


def _normalize_spacing(text: str) -> str:
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)
    text = re.sub(r"\(\s+", "(", text)
    text = re.sub(r"\s+\)", ")", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _strip_caption_token(text: str) -> str:
    text = re.sub("^[\"'(\\[\u2018\u201C]+", "", text)
    return re.sub("[\"')\\].,;:!?\u2019\u201D]+$", "", text)


def _looks_like_title_token(text: str) -> bool:
    return bool(TITLECASE_TOKEN_RE.match(_strip_caption_token(text)))


def _is_honorific(text: str) -> bool:
    return bool(HONORIFIC_RE.match(_strip_caption_token(text)))


def _split_crosses_quoted_span(left: str, right: str) -> bool:
    return len(DOUBLE_QUOTE_RE.findall(left)) % 2 == 1 and len(DOUBLE_QUOTE_RE.findall(right)) > 0


def _starts_sentence(text: str) -> bool:
    return bool(re.match("^[\"'(\\[\u2018\u201C]?[A-Z]", text))


def _is_name_or_title_boundary(last_word: str, first_word: str) -> bool:
    left_clean = _strip_caption_token(last_word)
    right_clean = _strip_caption_token(first_word)
    left_normalized = normalize_word(left_clean)
    right_normalized = normalize_word(right_clean)

    if not left_clean or not right_clean:
        return False

    if _is_honorific(left_clean) and _looks_like_title_token(right_clean):
        return True

    if _looks_like_title_token(left_clean) and _looks_like_title_token(right_clean):
        return True

    if _looks_like_title_token(left_clean) and right_normalized in TITLE_CONNECTORS:
        return True

    if left_normalized in TITLE_CONNECTORS and _looks_like_title_token(right_clean):
        return True

    if left_normalized in ENTITY_INTRODUCERS and _looks_like_title_token(right_clean):
        return True

    return False


def _caption_split_score(left: str, right: str, target_line_length: int, hard_cap: int) -> float:
    left_len = len(left)
    right_len = len(right)
    score = abs(left_len - right_len)
    total_len = left_len + right_len

    left_words = left.split()
    right_words = right.split()
    score += abs(len(left_words) - len(right_words)) * 0.8

    score += max(0, left_len - target_line_length) * 1.7
    score += max(0, right_len - target_line_length) * 1.7
    score += max(0, left_len - hard_cap) * 6
    score += max(0, right_len - hard_cap) * 6

    if left_len < 12:
        score += (12 - left_len) * 3
    if right_len < 12:
        score += (12 - right_len) * 3

    midpoint = total_len / 2
    score += abs(left_len - midpoint) * 0.35

    last_word = left_words[-1]
    first_word = normalize_word(right_words[0])
    last_normalized = normalize_word(last_word)

    if SENTENCE_END_RE.search(last_word) and _starts_sentence(right_words[0]):
        score += 52
    elif CLAUSE_END_RE.search(last_word):
        score -= 18

    if _split_crosses_quoted_span(left, right):
        score += 44

    if _is_name_or_title_boundary(last_word, right_words[0]):
        score += 52

    if last_normalized in WEAK_LINE_ENDS:
        score += 16
    if first_word in WEAK_LINE_STARTS:
        score += 14

    if right[0] in ",.;:!?)]}\"'":
        score += 40
    if left[-1] in "([{\"'":
        score += 28

    if 24 <= left_len <= target_line_length and 24 <= right_len <= target_line_length:
        score -= 6
    if len(left_words) < 2 or len(right_words) < 2:
        score += 24

    return score
