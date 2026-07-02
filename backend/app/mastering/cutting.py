from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from ..text_processing import DISFLUENCY_FILLERS
from .audio_io import MasterAudio
from .classify import Segment

CutReason = Literal["silence", "filler"]

CROSSFADE_SECONDS = 0.010
MIN_CUT_SECONDS = 0.08
FILLER_PAD_SECONDS = 0.04
WORD_CLEAN_RE = re.compile(r"[^a-z]+")


@dataclass
class CutRegion:
    start: float
    end: float
    reason: CutReason
    label: str


def build_silence_cuts(
    segments: list[Segment],
    duration: float,
    keep_pause_seconds: float = 0.6,
    max_pause_seconds: float = 1.5,
) -> list[CutRegion]:
    """Trim pauses longer than `max_pause_seconds` down to `keep_pause_seconds`.

    Intentional speech breaks up to the maximum are preserved untouched; longer
    gaps keep half the kept pause on each side and lose the middle.
    """
    keep = min(keep_pause_seconds, max_pause_seconds)
    cuts: list[CutRegion] = []
    for segment in segments:
        if segment.kind != "background":
            continue
        start = max(0.0, segment.start)
        end = min(duration, segment.end)
        gap = end - start
        if gap <= max_pause_seconds:
            continue
        cut_start = start + keep / 2.0
        cut_end = end - keep / 2.0
        if cut_end - cut_start >= MIN_CUT_SECONDS:
            cuts.append(CutRegion(start=cut_start, end=cut_end, reason="silence", label="silence"))
    return cuts


def detect_filler_regions(
    words: list[dict[str, Any]],
    lexicon: set[str] = DISFLUENCY_FILLERS,
    pad_seconds: float = FILLER_PAD_SECONDS,
) -> list[CutRegion]:
    cuts: list[CutRegion] = []
    for word in words:
        text = WORD_CLEAN_RE.sub("", str(word.get("text") or word.get("word") or "").lower())
        if text not in lexicon:
            continue
        try:
            start = float(word["start"])
            end = float(word["end"])
        except (KeyError, TypeError, ValueError):
            continue
        if end <= start:
            continue
        cuts.append(
            CutRegion(
                start=max(0.0, start - pad_seconds),
                end=end + pad_seconds,
                reason="filler",
                label=text,
            )
        )
    return cuts


def merge_cut_regions(cuts: list[CutRegion]) -> list[CutRegion]:
    if not cuts:
        return []
    ordered = sorted(cuts, key=lambda cut: cut.start)
    merged = [ordered[0]]
    for cut in ordered[1:]:
        previous = merged[-1]
        if cut.start <= previous.end:
            merged[-1] = CutRegion(
                start=previous.start,
                end=max(previous.end, cut.end),
                reason=previous.reason if previous.reason == cut.reason else "silence",
                label=previous.label if previous.reason == cut.reason else "silence+filler",
            )
        else:
            merged.append(cut)
    return merged


def clamp_cuts_to_duration(cuts: list[CutRegion], duration: float) -> list[CutRegion]:
    clamped: list[CutRegion] = []
    for cut in cuts:
        start = max(0.0, min(cut.start, duration))
        end = max(0.0, min(cut.end, duration))
        if end - start >= MIN_CUT_SECONDS:
            clamped.append(CutRegion(start=start, end=end, reason=cut.reason, label=cut.label))
    return clamped


def apply_cuts(
    audio: MasterAudio,
    cuts: list[CutRegion],
    mode: Literal["apply", "silence"],
    crossfade_seconds: float = CROSSFADE_SECONDS,
) -> MasterAudio:
    if not cuts:
        return audio

    sr = audio.sample_rate
    n = audio.samples.shape[1]
    fade = max(1, int(round(crossfade_seconds * sr)))

    if mode == "silence":
        for cut in cuts:
            start = int(cut.start * sr)
            end = min(n, int(cut.end * sr))
            if end <= start:
                continue
            audio.samples[:, start:end] = 0.0
            fade_in_end = min(n, end + fade)
            fade_out_start = max(0, start - fade)
            if start > fade_out_start:
                ramp = np.linspace(1.0, 0.0, start - fade_out_start, dtype=np.float32)
                audio.samples[:, fade_out_start:start] *= ramp
            if fade_in_end > end:
                ramp = np.linspace(0.0, 1.0, fade_in_end - end, dtype=np.float32)
                audio.samples[:, end:fade_in_end] *= ramp
        return audio

    keep_slices: list[np.ndarray] = []
    cursor = 0
    for cut in cuts:
        start = int(cut.start * sr)
        end = min(n, int(cut.end * sr))
        if start > cursor:
            keep_slices.append(audio.samples[:, cursor:start])
        cursor = max(cursor, end)
    if cursor < n:
        keep_slices.append(audio.samples[:, cursor:n])
    if not keep_slices:
        audio.samples = np.zeros((audio.channels, 0), dtype=np.float32)
        return audio

    # Short equal-power crossfades at each joint to avoid clicks.
    joined = keep_slices[0].copy()
    for piece in keep_slices[1:]:
        overlap = min(fade, joined.shape[1], piece.shape[1])
        if overlap > 0:
            fade_out = np.cos(np.linspace(0.0, np.pi / 2.0, overlap, dtype=np.float32)) ** 2
            fade_in = 1.0 - fade_out
            joined[:, -overlap:] = joined[:, -overlap:] * fade_out + piece[:, :overlap] * fade_in
            joined = np.concatenate([joined, piece[:, overlap:]], axis=1)
        else:
            joined = np.concatenate([joined, piece], axis=1)
    audio.samples = np.ascontiguousarray(joined, dtype=np.float32)
    return audio


def remap_timestamp(t: float, cuts: list[CutRegion]) -> float:
    """Map a timestamp on the original timeline onto the cut timeline."""
    removed = 0.0
    for cut in cuts:
        if t >= cut.end:
            removed += cut.end - cut.start
        elif t > cut.start:
            removed += t - cut.start
        else:
            break
    return max(0.0, t - removed)


def total_cut_duration(cuts: list[CutRegion]) -> float:
    return sum(cut.end - cut.start for cut in cuts)


def export_audacity_labels(cuts: list[CutRegion]) -> str:
    lines = [f"{cut.start:.6f}\t{cut.end:.6f}\t{cut.label}" for cut in cuts]
    return "\n".join(lines) + ("\n" if lines else "")
