from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any

import numpy as np

from ..config import settings
from ..jobs import ProgressReporter
from ..schemas import WarningItem
from . import cutting as cutting_mod
from .audio_io import decode_master, encode_master, remove_dc_offset
from .classify import Segment, classify_segments
from .cutting import CutRegion
from .denoise import DenoiseUnavailable, denoise
from .dynamics import soft_knee_compressor, true_peak_limiter
from .filters import adaptive_cutoff, high_pass
from .hum import detect_hum, remove_hum
from .leveler import apply_gain_curve, compute_leveler_gain
from .loudness import measure_loudness, normalize_loudness
from .schemas import (
    CutRegionModel,
    DenoiseReport,
    HighPassReport,
    HumReport,
    LevelerReport,
    MasteringParams,
    MasteringReport,
    MasteringResult,
)


def _output_paths(source_filename: str, fmt: str) -> tuple[str, Path]:
    token = f"m_{uuid.uuid4().hex[:12]}"
    stem = Path(source_filename or "audio").stem or "audio"
    output_dir = Path(settings.mastering_output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    return token, output_dir / f"{token}__{stem}.master.{fmt}"


def find_master_artifact(token: str) -> Path | None:
    output_dir = Path(settings.mastering_output_dir)
    if not output_dir.is_dir() or not token.replace("m_", "").isalnum():
        return None
    matches = sorted(output_dir.glob(f"{token}__*"))
    return matches[0] if matches else None


def run_mastering(
    source_path: str,
    source_filename: str,
    params: MasteringParams,
    words: list[dict[str, Any]] | None,
    reporter: ProgressReporter,
) -> MasteringResult:
    warnings: list[WarningItem] = []

    reporter.stage("decode", 0.0, 0.05, "Decoding audio")
    audio = decode_master(source_path, downmix_mono=params.output.downmix_mono)
    audio = remove_dc_offset(audio)
    duration_before = audio.duration

    reporter.stage("measure", 0.05, 0.10, "Measuring input loudness")
    before_stats = measure_loudness(audio)

    hum_report = HumReport(detected=False)
    if params.hum_removal.enabled:
        reporter.stage("hum", 0.10, 0.15, "Detecting hum")
        candidates: tuple[float, ...]
        if params.hum_removal.base_frequency == "auto":
            candidates = (50.0, 60.0)
        else:
            candidates = (float(params.hum_removal.base_frequency),)
        profile = detect_hum(audio, candidates=candidates, max_harmonics=params.hum_removal.max_harmonics)
        if profile:
            audio = remove_hum(audio, profile)
            hum_report = HumReport(
                detected=True,
                base_frequency=profile.base_frequency,
                harmonics_notched=len(profile.harmonics),
            )

    denoise_report = DenoiseReport(applied=False)
    if params.denoise.enabled:
        reporter.stage("denoise", 0.15, 0.55, "Denoising speech (MossFormer2)")
        try:
            audio, device_used = denoise(audio, params.denoise.amount, progress=reporter.tick)
            denoise_report = DenoiseReport(applied=True, device_used=device_used)
            if device_used == "cpu":
                warnings.append(
                    WarningItem(
                        code="denoise_cpu_fallback",
                        message="AI denoising ran on the CPU, which is much slower than a GPU.",
                    )
                )
        except DenoiseUnavailable as exc:
            warnings.append(WarningItem(code="denoise_unavailable", message=str(exc)))

    high_pass_report = HighPassReport(applied=False)
    if params.high_pass.enabled:
        reporter.stage("high_pass", 0.55, 0.60, "Applying adaptive high-pass filter")
        cutoff = adaptive_cutoff(audio) if params.high_pass.cutoff == "auto" else float(params.high_pass.cutoff)
        audio = high_pass(audio, cutoff)
        high_pass_report = HighPassReport(applied=True, cutoff_hz=round(cutoff, 1))

    reporter.stage("classify", 0.60, 0.63, "Classifying speech and background")
    segments = classify_segments(audio)

    cut_list: list[CutRegion] = []
    cutting = params.cutting
    if cutting.silence.enabled or cutting.fillers.enabled:
        reporter.stage("cutting", 0.63, 0.70, "Finding cuts")
        if cutting.silence.enabled:
            cut_list.extend(
                cutting_mod.build_silence_cuts(
                    segments,
                    audio.duration,
                    keep_pause_seconds=cutting.silence.keep_pause_seconds,
                    max_pause_seconds=cutting.silence.max_pause_seconds,
                )
            )
        if cutting.fillers.enabled:
            if words:
                cut_list.extend(cutting_mod.detect_filler_regions(words))
            else:
                warnings.append(
                    WarningItem(
                        code="fillers_no_words",
                        message="Filler cutting needs a transcript. Run transcription first, then master again.",
                    )
                )
        cut_list = cutting_mod.clamp_cuts_to_duration(cutting_mod.merge_cut_regions(cut_list), audio.duration)
        if cut_list and cutting.apply_mode != "list_only":
            audio = cutting_mod.apply_cuts(audio, cut_list, mode=cutting.apply_mode)
            if cutting.apply_mode == "apply":
                # Remap the segments through the cut list rather than
                # re-classifying: with most silence gone, percentile-based
                # thresholds would mistake the quietest speaker for background.
                segments = [
                    Segment(
                        start=cutting_mod.remap_timestamp(segment.start, cut_list),
                        end=cutting_mod.remap_timestamp(segment.end, cut_list),
                        kind=segment.kind,
                    )
                    for segment in segments
                ]
                segments = [segment for segment in segments if segment.end - segment.start > 0.05]

    leveler_report = LevelerReport(applied=False)
    if params.leveler.enabled:
        reporter.stage("leveler", 0.70, 0.78, "Leveling speech")
        gains = compute_leveler_gain(audio, segments, params.leveler.strength)
        if gains.size:
            audio = apply_gain_curve(audio, gains)
            leveler_report = LevelerReport(
                applied=True,
                max_boost_db=round(float(np.max(gains)), 2),
                max_cut_db=round(float(-np.min(gains)), 2),
            )
        audio = soft_knee_compressor(audio)

    target_lufs, true_peak = params.loudness.resolved_targets()
    if params.loudness.enabled:
        reporter.stage("loudness", 0.78, 0.88, f"Normalizing to {target_lufs:g} LUFS")
        audio = normalize_loudness(audio, target_lufs, true_peak)
    audio = true_peak_limiter(audio, ceiling_dbtp=true_peak)

    reporter.stage("measure_after", 0.88, 0.93, "Measuring output loudness")
    after_stats = measure_loudness(audio)

    reporter.stage("encode", 0.93, 1.0, f"Encoding {params.output.format}")
    token, output_path = _output_paths(source_filename, params.output.format)
    encode_master(audio, str(output_path), params.output.format, params.output.bitrate_kbps)

    return MasteringResult(
        token=token,
        output_filename=output_path.name.split("__", 1)[1],
        output_format=params.output.format,
        duration_before=round(duration_before, 3),
        duration_after=round(audio.duration, 3),
        report=MasteringReport(
            before=before_stats,
            after=after_stats,
            hum=hum_report,
            denoise=denoise_report,
            high_pass=high_pass_report,
            leveler=leveler_report,
        ),
        cut_list=[
            CutRegionModel(start=round(cut.start, 3), end=round(cut.end, 3), reason=cut.reason, label=cut.label)
            for cut in cut_list
        ],
        warnings=warnings,
    )
