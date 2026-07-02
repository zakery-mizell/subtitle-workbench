from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from ..schemas import WarningItem

LevelerStrength = Literal["tight", "moderate", "soft"]
CutApplyMode = Literal["apply", "silence", "list_only"]
OutputFormat = Literal["wav", "flac", "mp3", "aac", "opus"]
LoudnessPreset = Literal["podcast", "streaming", "broadcast", "mono", "custom"]

LOUDNESS_PRESETS: dict[str, tuple[float, float]] = {
    # preset: (target_lufs, true_peak_dbtp)
    "podcast": (-16.0, -1.0),
    "streaming": (-14.0, -1.0),
    "broadcast": (-23.0, -1.0),
    "mono": (-19.0, -1.0),
}


class DenoiseParams(BaseModel):
    enabled: bool = False
    amount: float = Field(100.0, ge=0.0, le=100.0)


class HumRemovalParams(BaseModel):
    enabled: bool = False
    base_frequency: Literal["auto", "50", "60"] = "auto"
    max_harmonics: int = Field(8, ge=1, le=16)


class HighPassParams(BaseModel):
    enabled: bool = False
    cutoff: Literal["auto"] | float = "auto"


class LevelerParams(BaseModel):
    enabled: bool = False
    strength: LevelerStrength = "moderate"


class LoudnessParams(BaseModel):
    enabled: bool = True
    preset: LoudnessPreset = "podcast"
    target_lufs: float = Field(-16.0, ge=-40.0, le=-6.0)
    true_peak_dbtp: float = Field(-1.0, ge=-9.0, le=0.0)

    def resolved_targets(self) -> tuple[float, float]:
        if self.preset != "custom" and self.preset in LOUDNESS_PRESETS:
            return LOUDNESS_PRESETS[self.preset]
        return self.target_lufs, self.true_peak_dbtp


class SilenceCutParams(BaseModel):
    enabled: bool = False
    keep_pause_seconds: float = Field(0.6, ge=0.1, le=5.0)
    max_pause_seconds: float = Field(1.5, ge=0.2, le=10.0)


class FillerCutParams(BaseModel):
    enabled: bool = False
    engine: Literal["words"] = "words"


class CuttingParams(BaseModel):
    silence: SilenceCutParams = Field(default_factory=SilenceCutParams)
    fillers: FillerCutParams = Field(default_factory=FillerCutParams)
    apply_mode: CutApplyMode = "list_only"


class OutputParams(BaseModel):
    format: OutputFormat = "wav"
    bitrate_kbps: int | None = Field(None, ge=32, le=512)
    downmix_mono: bool = False


class MasteringParams(BaseModel):
    denoise: DenoiseParams = Field(default_factory=DenoiseParams)
    hum_removal: HumRemovalParams = Field(default_factory=HumRemovalParams)
    high_pass: HighPassParams = Field(default_factory=HighPassParams)
    leveler: LevelerParams = Field(default_factory=LevelerParams)
    loudness: LoudnessParams = Field(default_factory=LoudnessParams)
    cutting: CuttingParams = Field(default_factory=CuttingParams)
    output: OutputParams = Field(default_factory=OutputParams)


class LoudnessStats(BaseModel):
    integrated_lufs: float
    lra: float
    true_peak_dbtp: float
    noise_floor_dbfs: float


class HumReport(BaseModel):
    detected: bool
    base_frequency: float | None = None
    harmonics_notched: int = 0


class DenoiseReport(BaseModel):
    applied: bool
    device_used: str | None = None


class HighPassReport(BaseModel):
    applied: bool
    cutoff_hz: float | None = None


class LevelerReport(BaseModel):
    applied: bool
    max_boost_db: float | None = None
    max_cut_db: float | None = None


class MasteringReport(BaseModel):
    before: LoudnessStats
    after: LoudnessStats
    hum: HumReport = Field(default_factory=lambda: HumReport(detected=False))
    denoise: DenoiseReport = Field(default_factory=lambda: DenoiseReport(applied=False))
    high_pass: HighPassReport = Field(default_factory=lambda: HighPassReport(applied=False))
    leveler: LevelerReport = Field(default_factory=lambda: LevelerReport(applied=False))


class CutRegionModel(BaseModel):
    start: float
    end: float
    reason: Literal["silence", "filler"]
    label: str


class MasteringResult(BaseModel):
    token: str
    output_filename: str
    output_format: OutputFormat
    duration_before: float
    duration_after: float
    report: MasteringReport
    cut_list: list[CutRegionModel] = Field(default_factory=list)
    warnings: list[WarningItem] = Field(default_factory=list)


class MasterJobResponse(BaseModel):
    job_id: str


class JobStatusResponse(BaseModel):
    id: str
    status: Literal["queued", "running", "done", "error"]
    stage: str
    progress: float
    message: str | None = None
    error: str | None = None
    result: MasteringResult | None = None
