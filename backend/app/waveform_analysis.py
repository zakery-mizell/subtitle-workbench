from __future__ import annotations

import array
import math
import subprocess
import sys
from dataclasses import dataclass

from .schemas import SpeechSpan, WaveformAnalysisResponse, WaveformFrame


SAMPLE_RATE = 16000
FRAME_DURATION_SECONDS = 0.02
DISPLAY_FRAME_LIMIT = 6000
PCM_MAX = 32768.0


@dataclass(slots=True)
class FrameStats:
    time: float
    min: float
    max: float
    rms: float


def analyze_waveform(source_path: str) -> WaveformAnalysisResponse:
    frames, sample_count = _extract_pcm_frames(source_path)
    if not frames or sample_count <= 0:
        raise RuntimeError("No decodable audio samples were found in this file.")

    smoothed_rms = _smooth([frame.rms for frame in frames], radius=2)
    threshold = _adaptive_threshold(smoothed_rms)
    speech_spans = _detect_speech_spans(frames, smoothed_rms, threshold)
    display_frames = _downsample_display_frames(frames, DISPLAY_FRAME_LIMIT)

    return WaveformAnalysisResponse(
        duration=sample_count / SAMPLE_RATE,
        sample_rate=SAMPLE_RATE,
        frame_duration=FRAME_DURATION_SECONDS,
        threshold=threshold,
        frames=[
            WaveformFrame(time=frame.time, min=frame.min, max=frame.max, rms=frame.rms)
            for frame in display_frames
        ],
        speech_spans=speech_spans,
        warnings=[],
    )


def _extract_pcm_frames(source_path: str) -> tuple[list[FrameStats], int]:
    command = [
        "ffmpeg",
        "-v",
        "error",
        "-i",
        source_path,
        "-vn",
        "-ac",
        "1",
        "-ar",
        str(SAMPLE_RATE),
        "-f",
        "s16le",
        "-acodec",
        "pcm_s16le",
        "pipe:1",
    ]

    try:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except FileNotFoundError as exc:
        raise RuntimeError("FFmpeg is required to analyze waveform timing.") from exc

    if process.stdout is None:
        raise RuntimeError("FFmpeg did not provide audio output.")

    frame_samples = max(1, round(SAMPLE_RATE * FRAME_DURATION_SECONDS))
    frame_bytes = frame_samples * 2
    pending = bytearray()
    frames: list[FrameStats] = []
    sample_count = 0

    while True:
        chunk = process.stdout.read(1024 * 128)
        if not chunk:
            break
        pending.extend(chunk)

        while len(pending) >= frame_bytes:
            frame_data = bytes(pending[:frame_bytes])
            del pending[:frame_bytes]
            frames.append(_stats_for_frame(frame_data, len(frames)))
            sample_count += frame_samples

    if len(pending) >= 2:
        usable_bytes = len(pending) - (len(pending) % 2)
        remainder = bytes(pending[:usable_bytes])
        frames.append(_stats_for_frame(remainder, len(frames)))
        sample_count += usable_bytes // 2

    stderr = process.stderr.read().decode("utf-8", errors="replace") if process.stderr else ""
    return_code = process.wait()
    if return_code != 0:
        detail = stderr.strip() or f"ffmpeg exited with status {return_code}"
        raise RuntimeError(f"Could not decode audio for waveform analysis. Details: {detail}")

    return frames, sample_count


def _stats_for_frame(frame_data: bytes, frame_index: int) -> FrameStats:
    samples = array.array("h")
    samples.frombytes(frame_data)
    if sys.byteorder != "little":
        samples.byteswap()

    if not samples:
        return FrameStats(time=frame_index * FRAME_DURATION_SECONDS, min=0.0, max=0.0, rms=0.0)

    min_sample = min(samples)
    max_sample = max(samples)
    sum_squares = sum(sample * sample for sample in samples)
    rms = math.sqrt(sum_squares / len(samples)) / PCM_MAX

    return FrameStats(
        time=frame_index * FRAME_DURATION_SECONDS,
        min=max(-1.0, min(1.0, min_sample / PCM_MAX)),
        max=max(-1.0, min(1.0, max_sample / PCM_MAX)),
        rms=max(0.0, min(1.0, rms)),
    )


def _smooth(values: list[float], radius: int) -> list[float]:
    if radius <= 0 or not values:
        return values[:]

    smoothed: list[float] = []
    for index in range(len(values)):
        start = max(0, index - radius)
        end = min(len(values), index + radius + 1)
        smoothed.append(sum(values[start:end]) / (end - start))
    return smoothed


def _adaptive_threshold(values: list[float]) -> float:
    if not values:
        return 1.0

    noise_floor = _percentile(values, 0.2)
    speech_level = _percentile(values, 0.9)
    if speech_level <= 0:
        return 1.0

    threshold = max(
        noise_floor * 2.4,
        noise_floor + (speech_level - noise_floor) * 0.28,
        speech_level * 0.12,
        0.003,
    )
    return min(threshold, max(speech_level * 0.82, 0.003))


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0

    ordered = sorted(values)
    raw_index = (len(ordered) - 1) * max(0.0, min(1.0, percentile))
    lower = math.floor(raw_index)
    upper = math.ceil(raw_index)
    if lower == upper:
        return ordered[lower]

    fraction = raw_index - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def _detect_speech_spans(
    frames: list[FrameStats],
    smoothed_rms: list[float],
    threshold: float,
) -> list[SpeechSpan]:
    if not frames or not smoothed_rms:
        return []

    on_threshold = threshold
    off_threshold = min(threshold * 0.85, max(threshold * 0.55, _percentile(smoothed_rms, 0.2) * 1.6))
    min_on_frames = max(1, round(0.06 / FRAME_DURATION_SECONDS))
    min_off_frames = max(1, round(0.1 / FRAME_DURATION_SECONDS))
    min_span_seconds = 0.08

    raw_spans: list[SpeechSpan] = []
    start_frame: int | None = None
    above_run = 0
    below_run = 0
    peak = 0.0

    for index, rms in enumerate(smoothed_rms):
        if start_frame is None:
            if rms >= on_threshold:
                above_run += 1
                if above_run >= min_on_frames:
                    start_frame = index - above_run + 1
                    peak = max(smoothed_rms[start_frame : index + 1])
                    below_run = 0
            else:
                above_run = 0
            continue

        peak = max(peak, rms, frames[index].rms)
        if rms < off_threshold:
            below_run += 1
            if below_run >= min_off_frames:
                end_frame = index - below_run + 1
                _append_span(raw_spans, frames, start_frame, end_frame, peak, min_span_seconds)
                start_frame = None
                above_run = 0
                below_run = 0
                peak = 0.0
        else:
            below_run = 0

    if start_frame is not None:
        _append_span(raw_spans, frames, start_frame, len(frames), peak, min_span_seconds)

    return _merge_close_spans(raw_spans, max_gap=0.14)


def _append_span(
    spans: list[SpeechSpan],
    frames: list[FrameStats],
    start_frame: int,
    end_frame: int,
    peak: float,
    min_span_seconds: float,
) -> None:
    start = max(0.0, frames[start_frame].time)
    end = min(frames[-1].time + FRAME_DURATION_SECONDS, max(start, end_frame * FRAME_DURATION_SECONDS))
    if end - start < min_span_seconds:
        return
    spans.append(SpeechSpan(start=start, end=end, peak=max(0.0, min(1.0, peak))))


def _merge_close_spans(spans: list[SpeechSpan], max_gap: float) -> list[SpeechSpan]:
    if not spans:
        return []

    merged: list[SpeechSpan] = []
    for span in spans:
        if not merged or span.start - merged[-1].end > max_gap:
            merged.append(span)
            continue

        previous = merged[-1]
        merged[-1] = SpeechSpan(
            start=previous.start,
            end=max(previous.end, span.end),
            peak=max(previous.peak, span.peak),
        )

    return merged


def _downsample_display_frames(frames: list[FrameStats], limit: int) -> list[FrameStats]:
    if len(frames) <= limit:
        return frames

    bucket_size = math.ceil(len(frames) / limit)
    display: list[FrameStats] = []
    for start in range(0, len(frames), bucket_size):
        bucket = frames[start : start + bucket_size]
        if not bucket:
            continue

        display.append(
            FrameStats(
                time=bucket[0].time,
                min=min(frame.min for frame in bucket),
                max=max(frame.max for frame in bucket),
                rms=sum(frame.rms for frame in bucket) / len(bucket),
            )
        )

    return display
