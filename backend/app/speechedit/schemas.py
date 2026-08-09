from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from ..schemas import WarningItem

SpeechEditMode = Literal["edit", "generate"]
SpeechEditOutputFormat = Literal["wav", "flac", "mp3", "aac", "opus"]

SpeechEditLanguage = Literal["English"]


class PatchEdit(BaseModel):
    """One span to regenerate in EDIT mode."""

    start_s: float = Field(ge=0.0)
    end_s: float = Field(gt=0.0)
    new_text: str = Field(min_length=1)
    # Force the patch to a target length instead of the original span length; lets
    # a mis-timed word be re-timed. Capped at the 20 s per-span edit limit.
    fix_duration_s: float | None = Field(None, gt=0.0, le=20.0)
    # Full target transcript for the whole window, overriding the auto-built one --
    # the escape hatch when degraded audio will not transcribe.
    window_text: str | None = None


class SpeechEditParams(BaseModel):
    """Parameters for an F5-TTS speech-edit ("Patch") job."""

    mode: SpeechEditMode = "edit"
    output_format: SpeechEditOutputFormat = "flac"
    # EDIT mode: the spans to patch (validated non-empty in the service).
    edits: list[PatchEdit] = []
    # GENERATE mode: the text to speak and the reference transcript.
    gen_text: str = ""
    ref_text: str = ""
    auto_ref_text: bool = True
    whisper_model: str = "small"
    language: SpeechEditLanguage = "English"
    # Flow-matching function evaluations; more = higher quality, slower.
    nfe_step: int = Field(32, ge=8, le=64)
    speed: float = Field(1.0, ge=0.5, le=2.0)
    seed: int | None = None


class PatchRegion(BaseModel):
    """What one edit regenerated, surfaced so the UI can show it."""

    start_s: float
    end_s: float
    window_start_s: float
    window_end_s: float
    text_used: str


class SpeechEditResult(BaseModel):
    token: str
    filename: str
    output_format: str
    sample_rate: int
    duration_sec: float
    device_used: str
    mode: SpeechEditMode
    regions: list[PatchRegion] = []
    ref_text_used: str | None = None
    warnings: list[WarningItem] = []
