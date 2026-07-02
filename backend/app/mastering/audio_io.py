from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass

import numpy as np

MASTER_SAMPLE_RATE = 48000
MAX_MASTER_DURATION_SECONDS = 4.0 * 3600.0


@dataclass
class MasterAudio:
    samples: np.ndarray  # float32, shape (channels, n)
    sample_rate: int

    @property
    def channels(self) -> int:
        return int(self.samples.shape[0])

    @property
    def duration(self) -> float:
        return float(self.samples.shape[1]) / self.sample_rate


def probe_source(source_path: str) -> tuple[int, float]:
    """Return (channels, duration_seconds) for the first audio stream."""
    command = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "a:0",
        "-show_entries",
        "stream=channels:format=duration",
        "-of",
        "json",
        source_path,
    ]
    try:
        completed = subprocess.run(command, check=True, capture_output=True, text=True)
    except FileNotFoundError as exc:
        raise RuntimeError("FFmpeg (ffprobe) is required for audio mastering.") from exc
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"Could not probe the audio file. Details: {exc.stderr.strip()}") from exc

    payload = json.loads(completed.stdout or "{}")
    streams = payload.get("streams") or []
    if not streams:
        raise RuntimeError("No audio stream was found in this file.")
    channels = int(streams[0].get("channels") or 1)
    duration = float(payload.get("format", {}).get("duration") or 0.0)
    return channels, duration


def decode_master(
    source_path: str,
    target_sr: int = MASTER_SAMPLE_RATE,
    downmix_mono: bool = False,
) -> MasterAudio:
    """Decode any audio/video file to float32 PCM at the mastering rate."""
    source_channels, duration = probe_source(source_path)
    if duration > MAX_MASTER_DURATION_SECONDS:
        hours = MAX_MASTER_DURATION_SECONDS / 3600.0
        raise RuntimeError(
            f"This file is longer than {hours:.0f} hours, which exceeds the mastering limit. "
            "Split it into shorter parts first."
        )

    channels = 1 if downmix_mono else min(max(source_channels, 1), 2)
    command = [
        "ffmpeg",
        "-v",
        "error",
        "-i",
        source_path,
        "-vn",
        "-ac",
        str(channels),
        "-ar",
        str(target_sr),
        "-f",
        "f32le",
        "-acodec",
        "pcm_f32le",
        "pipe:1",
    ]
    try:
        completed = subprocess.run(command, check=True, capture_output=True)
    except FileNotFoundError as exc:
        raise RuntimeError("FFmpeg is required for audio mastering.") from exc
    except subprocess.CalledProcessError as exc:
        detail = exc.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"Could not decode audio for mastering. Details: {detail}") from exc

    raw = np.frombuffer(completed.stdout, dtype=np.float32)
    if raw.size == 0:
        raise RuntimeError("No decodable audio samples were found in this file.")
    usable = raw.size - (raw.size % channels)
    samples = raw[:usable].reshape(-1, channels).T.copy()
    return MasterAudio(samples=samples, sample_rate=target_sr)


ENCODER_ARGS: dict[str, list[str]] = {
    "mp3": ["-c:a", "libmp3lame"],
    "aac": ["-c:a", "aac"],
    "opus": ["-c:a", "libopus"],
    "flac": ["-c:a", "flac"],
    "wav": ["-c:a", "pcm_s16le"],
}

DEFAULT_BITRATES_KBPS: dict[str, int] = {"mp3": 192, "aac": 160, "opus": 128}


def encode_master(audio: MasterAudio, out_path: str, fmt: str, bitrate_kbps: int | None = None) -> None:
    """Encode float32 PCM to the requested format via ffmpeg stdin pipe."""
    if fmt not in ENCODER_ARGS:
        raise RuntimeError(f"Unsupported output format: {fmt}")

    command = [
        "ffmpeg",
        "-y",
        "-v",
        "error",
        "-f",
        "f32le",
        "-ar",
        str(audio.sample_rate),
        "-ac",
        str(audio.channels),
        "-i",
        "pipe:0",
        *ENCODER_ARGS[fmt],
    ]
    if fmt in DEFAULT_BITRATES_KBPS:
        bitrate = bitrate_kbps or DEFAULT_BITRATES_KBPS[fmt]
        command.extend(["-b:a", f"{bitrate}k"])
    command.append(out_path)

    interleaved = np.ascontiguousarray(audio.samples.T, dtype=np.float32)
    try:
        subprocess.run(command, check=True, input=interleaved.tobytes(), capture_output=True)
    except FileNotFoundError as exc:
        raise RuntimeError("FFmpeg is required for audio mastering.") from exc
    except subprocess.CalledProcessError as exc:
        detail = exc.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"Could not encode the processed audio. Details: {detail}") from exc


def remove_dc_offset(audio: MasterAudio) -> MasterAudio:
    audio.samples -= audio.samples.mean(axis=1, keepdims=True).astype(np.float32)
    return audio
