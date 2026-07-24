from __future__ import annotations

"""Blend UniSE's 16 kHz re-synthesized stems back into the original recording.

The original keeps its native sample rate everywhere except inside overlap
regions, where we either duck-and-overlay the separated voice ("spotlight")
or replace the mixture with the everyone-but-X stem ("mute"). Equal-power
crossfades at region edges keep the transitions inaudible.
"""

from dataclasses import dataclass

import numpy as np
import soxr

DEFAULT_CROSSFADE_SECONDS = 0.06
DEFAULT_DUCK_DB = -11.0
STEM_HEADROOM_DB = -1.0


@dataclass(slots=True)
class RegionRender:
    """One processed overlap region, positioned on the original timeline."""

    start: float  # seconds, where `stem` begins on the original timeline
    stem: np.ndarray  # mono float32 at stem_rate
    region_start: float  # seconds, audible blend window
    region_end: float


def resample_to(stem: np.ndarray, stem_rate: int, target_rate: int) -> np.ndarray:
    if stem_rate == target_rate:
        return np.asarray(stem, dtype=np.float32)
    return soxr.resample(np.asarray(stem, dtype=np.float32), stem_rate, target_rate).astype(np.float32)


def _rms(x: np.ndarray) -> float:
    if x.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(x), dtype=np.float64)))


def match_stem_gain(stem: np.ndarray, reference: np.ndarray, headroom_db: float = STEM_HEADROOM_DB) -> np.ndarray:
    """Scale the stem so its RMS matches the reference region, with headroom."""
    stem_rms = _rms(stem)
    ref_rms = _rms(reference)
    if stem_rms < 1e-6 or ref_rms < 1e-6:
        return stem.astype(np.float32)
    gain = (ref_rms / stem_rms) * (10.0 ** (headroom_db / 20.0))
    return (stem * gain).astype(np.float32)


def _edge_ramps(length: int, fade: int) -> np.ndarray:
    """Envelope that rises 0→1 over `fade` samples and falls back at the end."""
    envelope = np.ones(length, dtype=np.float32)
    fade = min(fade, length // 2)
    if fade > 0:
        # Equal-power so the summed original+stem loudness stays steady.
        curve = np.sin(np.linspace(0.0, np.pi / 2.0, fade, dtype=np.float32))
        envelope[:fade] = curve
        envelope[length - fade :] = curve[::-1]
    return envelope


def region_envelope(
    total: int,
    sample_rate: int,
    regions: list[tuple[float, float]],
    crossfade_seconds: float = DEFAULT_CROSSFADE_SECONDS,
) -> np.ndarray:
    """Gain curve that is 1 inside `regions` and 0 outside, fading at the edges.

    Regions that touch or overlap are merged first so no fade dips appear at an
    internal joint.
    """
    envelope = np.zeros(total, dtype=np.float32)
    if total <= 0:
        return envelope

    merged: list[list[int]] = []
    for start, end in sorted(regions):
        lo = max(0, min(total, int(round(start * sample_rate))))
        hi = max(0, min(total, int(round(end * sample_rate))))
        if hi <= lo:
            continue
        if merged and lo <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], hi)
        else:
            merged.append([lo, hi])

    fade = int(crossfade_seconds * sample_rate)
    for lo, hi in merged:
        envelope[lo:hi] = _edge_ramps(hi - lo, fade)
    return envelope


def apply_region_gate(
    samples: np.ndarray,
    sample_rate: int,
    regions: list[tuple[float, float]],
    crossfade_seconds: float = DEFAULT_CROSSFADE_SECONDS,
) -> bool:
    """Silence everything outside `regions`, keeping the timeline intact.

    Mutates `samples` (channels, n) in place and reports whether it did anything;
    an empty region list leaves the audio untouched.
    """
    if not regions or samples.shape[1] <= 0:
        return False
    envelope = region_envelope(samples.shape[1], sample_rate, regions, crossfade_seconds)
    samples *= envelope[None, :]
    return True


def _region_bounds(render: RegionRender, sample_rate: int, total: int) -> tuple[int, int]:
    start = max(0, int(round(render.region_start * sample_rate)))
    end = min(total, int(round(render.region_end * sample_rate)))
    return start, end


def _stem_slice(render: RegionRender, stem_48k: np.ndarray, sample_rate: int, start: int, end: int) -> np.ndarray:
    """Cut the piece of the (already resampled) stem that covers [start, end)."""
    offset = start - int(round(render.start * sample_rate))
    piece = stem_48k[max(0, offset) : max(0, offset) + (end - start)]
    if piece.shape[0] < end - start:
        piece = np.pad(piece, (0, end - start - piece.shape[0]))
    return piece


def _limit_region(samples: np.ndarray, start: int, end: int, ceiling: float = 0.985) -> None:
    """Scale the blended region down if the overlay pushed it past the ceiling."""
    peak = float(np.max(np.abs(samples[:, start:end]))) if end > start else 0.0
    if peak > ceiling:
        samples[:, start:end] *= ceiling / peak


def apply_spotlight(
    samples: np.ndarray,
    sample_rate: int,
    render: RegionRender,
    stem_rate: int,
    duck_db: float = DEFAULT_DUCK_DB,
    crossfade_seconds: float = DEFAULT_CROSSFADE_SECONDS,
) -> None:
    """Duck the original inside the region and lay the separated voice on top.

    Mutates `samples` (channels, n) in place.
    """
    total = samples.shape[1]
    start, end = _region_bounds(render, sample_rate, total)
    if end - start <= 0:
        return

    stem_48k = resample_to(render.stem, stem_rate, sample_rate)
    piece = _stem_slice(render, stem_48k, sample_rate, start, end)

    mono_reference = samples[:, start:end].mean(axis=0)
    piece = match_stem_gain(piece, mono_reference)

    fade = int(crossfade_seconds * sample_rate)
    blend = _edge_ramps(end - start, fade)
    duck_gain = 10.0 ** (duck_db / 20.0)
    # Original: 1 outside, duck_gain inside, ramped at the edges.
    original_env = 1.0 - blend * (1.0 - duck_gain)

    samples[:, start:end] *= original_env
    samples[:, start:end] += piece * blend
    _limit_region(samples, start, end)


def apply_replace(
    samples: np.ndarray,
    sample_rate: int,
    render: RegionRender,
    stem_rate: int,
    crossfade_seconds: float = DEFAULT_CROSSFADE_SECONDS,
) -> None:
    """Replace the region with the stem (used for mute: stem = everyone but X).

    Mutates `samples` (channels, n) in place.
    """
    total = samples.shape[1]
    start, end = _region_bounds(render, sample_rate, total)
    if end - start <= 0:
        return

    stem_48k = resample_to(render.stem, stem_rate, sample_rate)
    piece = _stem_slice(render, stem_48k, sample_rate, start, end)

    mono_reference = samples[:, start:end].mean(axis=0)
    piece = match_stem_gain(piece, mono_reference, headroom_db=0.0)

    fade = int(crossfade_seconds * sample_rate)
    blend = _edge_ramps(end - start, fade)

    samples[:, start:end] *= 1.0 - blend
    samples[:, start:end] += piece * blend
    _limit_region(samples, start, end)
