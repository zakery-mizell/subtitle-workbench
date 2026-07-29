"""Restore job: run Diamond over a whole file and encode the result.

Mirrors the mastering/separation job shape (token, artifact dir, staged
ProgressReporter) so the API endpoints and frontend polling work the same way.
"""
from __future__ import annotations

import subprocess
import uuid
from pathlib import Path

import numpy as np

from ..config import settings
from ..jobs import ProgressReporter
from ..mastering.audio_io import MasterAudio, encode_master
from ..schemas import WarningItem
from . import sidon_engine
from .engine import RESTORE_INPUT_RATE, RestoreUnavailable, load_engine, restore_waveform
from .schemas import RestoreParams, RestoreResult

ENGINE_LABELS = {"diamond": "Diamond", "sidon": "Sidon"}


def restore_output_dir() -> Path:
    return Path(settings.mastering_output_dir).parent / "restore"


def find_restore_artifact(token: str) -> Path | None:
    output_dir = restore_output_dir()
    if not output_dir.is_dir() or not token.replace("r_", "").isalnum():
        return None
    matches = sorted(output_dir.glob(f"{token}__*"))
    return matches[0] if matches else None


def _output_paths(source_filename: str, fmt: str) -> tuple[str, Path]:
    token = f"r_{uuid.uuid4().hex[:12]}"
    stem = Path(source_filename or "audio").stem or "audio"
    output_dir = restore_output_dir()
    output_dir.mkdir(parents=True, exist_ok=True)
    return token, output_dir / f"{token}__{stem}.restored.{fmt}"


def decode_mono(source_path: str, rate: int) -> np.ndarray:
    """Decode any audio/video file to mono float32 at the engine's input rate."""
    command = [
        "ffmpeg", "-v", "error", "-i", source_path, "-vn", "-ac", "1",
        "-ar", str(rate), "-f", "f32le", "-acodec", "pcm_f32le", "pipe:1",
    ]
    try:
        completed = subprocess.run(command, check=True, capture_output=True)
    except FileNotFoundError as exc:
        raise RuntimeError("FFmpeg is required for speech restoration.") from exc
    except subprocess.CalledProcessError as exc:
        detail = exc.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"Could not decode audio for restoration. Details: {detail}") from exc
    audio = np.frombuffer(completed.stdout, dtype=np.float32)
    if audio.size == 0:
        raise RuntimeError("No decodable audio samples were found in this file.")
    return audio.copy()


def run_restore(
    source_path: str,
    source_filename: str,
    params: RestoreParams,
    reporter: ProgressReporter,
) -> RestoreResult:
    """Restore a single audio file end to end with the selected model."""
    warnings: list[WarningItem] = []
    label = ENGINE_LABELS[params.engine]
    input_rate = (
        sidon_engine.SIDON_INPUT_RATE if params.engine == "sidon" else RESTORE_INPUT_RATE
    )

    reporter.stage("decode", 0.0, 0.06, "Decoding audio")
    wav = decode_mono(source_path, input_rate)

    reporter.stage("load_model", 0.06, 0.16, f"Loading {label} restoration model")
    try:
        engine = sidon_engine.load_engine() if params.engine == "sidon" else load_engine()
    except RestoreUnavailable as exc:
        raise RuntimeError(str(exc)) from exc

    reporter.stage("restore", 0.16, 0.92, "Restoring speech")
    if params.engine == "sidon":
        restored, out_sr = sidon_engine.restore_waveform(
            wav,
            input_rate,
            chunk_sec=params.sidon_chunk_sec,
            progress=reporter.tick,
        )
    else:
        restored, out_sr = restore_waveform(
            wav,
            input_rate,
            chunk_sec=params.chunk_sec,
            overlap_sec=params.overlap_sec,
            rep_penalty=params.rep_penalty,
            progress=reporter.tick,
        )

    reporter.stage("encode", 0.92, 1.0, f"Encoding {params.output_format}")
    token, output_path = _output_paths(source_filename, params.output_format)
    encode_master(
        MasterAudio(samples=restored[None, :], sample_rate=out_sr),
        str(output_path),
        params.output_format,
    )

    return RestoreResult(
        token=token,
        filename=output_path.name.split("__", 1)[1],
        engine=params.engine,
        output_format=params.output_format,
        sample_rate=out_sr,
        duration_sec=round(restored.size / out_sr, 3),
        device_used=engine.device,
        warnings=warnings,
    )
