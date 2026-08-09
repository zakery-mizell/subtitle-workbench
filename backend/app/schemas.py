from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


GuideLabel = Literal["SILENT", "CUT", "REPEAT"]
SpeakerAssignmentMode = Literal["hybrid"]


class SpeakerInput(BaseModel):
    id: int
    name: str


class WordToken(BaseModel):
    id: str
    text: str
    start: float
    end: float
    confidence: float
    low_confidence: bool
    speaker_id: int | None = None
    speaker_name: str | None = None


class Paragraph(BaseModel):
    id: str
    start: float
    end: float
    speaker_id: int | None = None
    speaker_name: str | None = None
    text: str
    word_ids: list[str] = Field(default_factory=list)


class Caption(BaseModel):
    id: str
    start: float
    end: float
    speaker_id: int | None = None
    speaker_name: str | None = None
    lines: list[str] = Field(default_factory=list)
    word_ids: list[str] = Field(default_factory=list)
    blank_after: bool = False


class GuideBlock(BaseModel):
    id: str
    start: float
    end: float
    label: GuideLabel
    reason: str
    skip: bool = True


class WarningItem(BaseModel):
    code: str
    message: str


class WaveformFrame(BaseModel):
    time: float
    min: float
    max: float
    rms: float


class SpeechSpan(BaseModel):
    start: float
    end: float
    peak: float


class CapabilitiesResponse(BaseModel):
    diarization_configured: bool
    speaker_profiles: list[str] = Field(default_factory=list)
    # Changes every time the backend process starts; the frontend uses it to
    # discard stale autosaved sessions after a server restart.
    instance_id: str


class SpeakerProfileResponse(BaseModel):
    name: str
    speaker_profiles: list[str] = Field(default_factory=list)


class WaveformAnalysisResponse(BaseModel):
    duration: float
    sample_rate: int
    frame_duration: float
    threshold: float
    frames: list[WaveformFrame]
    speech_spans: list[SpeechSpan]
    warnings: list[WarningItem] = Field(default_factory=list)


class SpeakerTurnOut(BaseModel):
    """One raw diarization turn; turns of different speakers may overlap."""

    start: float
    end: float
    speaker_index: int


class OverlapRegionOut(BaseModel):
    start: float
    end: float
    speaker_indices: list[int]


class TranscriptResponse(BaseModel):
    audio_filename: str
    duration: float | None = None
    speakers: list[SpeakerInput]
    words: list[WordToken]
    paragraphs: list[Paragraph]
    captions: list[Caption]
    guide_blocks: list[GuideBlock]
    warnings: list[WarningItem] = Field(default_factory=list)
    model: str
    speaker_assignment_mode: SpeakerAssignmentMode = "hybrid"
    language: Literal["en"] = "en"
    gpu_enabled: bool
    speaker_turns: list[SpeakerTurnOut] = Field(default_factory=list)
    overlap_regions: list[OverlapRegionOut] = Field(default_factory=list)


class RetranscribeRangeResponse(BaseModel):
    start: float
    end: float
    words: list[WordToken]
    paragraphs: list[Paragraph]
    captions: list[Caption]
    warnings: list[WarningItem] = Field(default_factory=list)
    model: str
    language: Literal["en"] = "en"
    gpu_enabled: bool
