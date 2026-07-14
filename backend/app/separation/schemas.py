from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from ..schemas import WarningItem

SeparationMode = Literal["spotlight", "mute"]


class TurnInput(BaseModel):
    """One raw diarization turn, as previously returned by /api/transcribe."""

    start: float = Field(ge=0)
    end: float = Field(gt=0)
    speaker_index: int = Field(ge=0)


class SeparationRegionParams(BaseModel):
    start: float = Field(ge=0)
    end: float = Field(gt=0)
    mode: SeparationMode = "spotlight"
    target_speaker_index: int = Field(ge=0)


class SeparationOutputParams(BaseModel):
    format: Literal["wav", "flac", "mp3", "aac", "opus"] = "wav"
    bitrate_kbps: int | None = Field(default=None, ge=64, le=320)


class SeparationParams(BaseModel):
    regions: list[SeparationRegionParams] = Field(min_length=1)
    turns: list[TurnInput] = Field(min_length=1)
    duck_db: float = Field(default=-11.0, ge=-40.0, le=0.0)
    transcribe_stems: bool = False
    transcribe_model: str = "small"
    language: str | None = None
    output: SeparationOutputParams = SeparationOutputParams()


class StemWord(BaseModel):
    text: str
    start: float
    end: float


class RegionReport(BaseModel):
    start: float
    end: float
    mode: SeparationMode
    target_speaker_index: int
    applied: bool
    enrollment_start: float | None = None
    enrollment_end: float | None = None
    words: list[StemWord] | None = None
    detail: str | None = None


class SeparationResult(BaseModel):
    token: str
    output_filename: str
    output_format: str
    device_used: str
    regions: list[RegionReport]
    warnings: list[WarningItem] = []

    def model_post_init(self, __context: Any) -> None:  # keep parity with mastering result shape
        return None


class OverlapRegionIn(BaseModel):
    start: float = Field(ge=0)
    end: float = Field(gt=0)
    speaker_indices: list[int] = Field(min_length=2)


class SoloTracksParams(BaseModel):
    """Auto-prepared per-speaker playback tracks: every overlap region is
    replaced by that speaker's isolated voice, the rest stays original."""

    regions: list[OverlapRegionIn] = Field(min_length=1)
    turns: list[TurnInput] = Field(min_length=1)
    output: SeparationOutputParams = SeparationOutputParams()


class SoloTrackOut(BaseModel):
    speaker_index: int
    token: str
    output_filename: str


class SoloRegionReport(BaseModel):
    start: float
    end: float
    speaker_index: int
    applied: bool
    detail: str | None = None


class SoloTracksResult(BaseModel):
    tracks: list[SoloTrackOut]
    regions: list[SoloRegionReport]
    output_format: str
    device_used: str
    warnings: list[WarningItem] = []
