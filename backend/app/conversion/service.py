"""Conversion job: re-voice a source performance with Seed-VC and encode it.

Mirrors the restore/tts job shape (token, artifact dir, staged ProgressReporter)
so the API endpoints and frontend polling work the same way. Both uploads are
pre-decoded with ffmpeg because they may be mp4/m4a containers librosa cannot
read reliably; the engine is handed the decoded wav paths.
"""
from __future__ import annotations

import subprocess
import uuid
import wave
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np

from ..config import settings
from ..jobs import ProgressReporter
from ..mastering.audio_io import MasterAudio, encode_master
from ..schemas import WarningItem
from .engine import SEEDVC_SR, MAX_REFERENCE_SECONDS, ConversionUnavailable, convert_voice, load_engine
from .schemas import ConversionParams, ConversionResult

MAX_REFERENCE_SAMPLES = int(MAX_REFERENCE_SECONDS * SEEDVC_SR)


def conversion_output_dir() -> Path:
    return Path(settings.mastering_output_dir).parent / "conversion"


def find_conversion_artifact(token: str) -> Path | None:
    output_dir = conversion_output_dir()
    if not output_dir.is_dir() or not token.replace("vc_", "").isalnum():
        return None
    matches = sorted(output_dir.glob(f"{token}__*"))
    return matches[0] if matches else None


def _output_paths(source_filename: str, fmt: str) -> tuple[str, Path]:
    token = f"vc_{uuid.uuid4().hex[:12]}"
    stem = Path(source_filename or "audio").stem or "audio"
    output_dir = conversion_output_dir()
    output_dir.mkdir(parents=True, exist_ok=True)
    return token, output_dir / f"{token}__{stem}.seedvc.{fmt}"


def _wav_frame_count(path: str) -> int:
    with wave.open(path, "rb") as handle:
        return handle.getnframes()


def decode_input(source_path: str, out_path: str) -> int:
    """Decode any audio/video upload to a mono 22.05 kHz 16-bit wav at out_path.

    Returns the decoded sample count so the caller can derive a duration (for the
    progress estimate and the reference-length check) without re-reading the file.
    """
    command = [
        "ffmpeg", "-v", "error", "-y", "-i", source_path, "-vn", "-ac", "1",
        "-ar", str(SEEDVC_SR), "-acodec", "pcm_s16le", out_path,
    ]
    try:
        subprocess.run(command, check=True, capture_output=True)
    except FileNotFoundError as exc:
        raise RuntimeError("FFmpeg is required for voice conversion.") from exc
    except subprocess.CalledProcessError as exc:
        detail = exc.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"Could not decode audio for conversion. Details: {detail}") from exc
    frames = _wav_frame_count(out_path)
    if frames == 0:
        raise RuntimeError("No decodable audio samples were found in this file.")
    return frames


def _make_temp_wav() -> str:
    Path(settings.temp_upload_dir).mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(delete=False, suffix=".wav", dir=settings.temp_upload_dir) as handle:
        return handle.name


def run_conversion(
    source_path: str,
    source_filename: str,
    ref_path: str,
    params: ConversionParams,
    reporter: ProgressReporter,
) -> ConversionResult:
    """Re-voice a source performance in the reference timbre end to end."""
    warnings: list[WarningItem] = []
    decoded_source = _make_temp_wav()
    decoded_ref = _make_temp_wav()
    try:
        reporter.stage("decode", 0.0, 0.05, "Decoding audio")
        source_frames = decode_input(source_path, decoded_source)
        ref_frames = decode_input(ref_path, decoded_ref)
        if ref_frames > MAX_REFERENCE_SAMPLES:
            warnings.append(
                WarningItem(
                    code="convert_ref_truncated",
                    message=(
                        f"The reference clip is longer than {int(MAX_REFERENCE_SECONDS)} seconds; "
                        f"the model uses only its first {int(MAX_REFERENCE_SECONDS)} seconds."
                    ),
                )
            )

        reporter.stage("load_model", 0.05, 0.30, "Loading Seed-VC model")
        try:
            engine = load_engine()
        except ConversionUnavailable as exc:
            raise RuntimeError(str(exc)) from exc

        reporter.stage("convert", 0.30, 0.94, "Converting voice")
        audio, out_sr = convert_voice(
            decoded_source,
            decoded_ref,
            diffusion_steps=params.diffusion_steps,
            length_adjust=params.length_adjust,
            intelligibility_cfg=params.intelligibility_cfg,
            similarity_cfg=params.similarity_cfg,
            convert_style=params.convert_style,
            source_duration_sec=source_frames / SEEDVC_SR,
            progress=reporter.tick,
        )

        reporter.stage("encode", 0.94, 1.0, f"Encoding {params.output_format}")
        token, output_path = _output_paths(source_filename, params.output_format)
        encode_master(
            MasterAudio(samples=audio[None, :], sample_rate=out_sr),
            str(output_path),
            params.output_format,
        )
    finally:
        for temp in (decoded_source, decoded_ref):
            try:
                Path(temp).unlink()
            except FileNotFoundError:
                pass

    return ConversionResult(
        token=token,
        filename=output_path.name.split("__", 1)[1],
        output_format=params.output_format,
        sample_rate=out_sr,
        duration_sec=round(audio.size / out_sr, 3),
        device_used=engine.device,
        diffusion_steps=params.diffusion_steps,
        warnings=warnings,
    )
