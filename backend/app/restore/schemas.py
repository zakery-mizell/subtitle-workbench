from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from ..schemas import WarningItem

RestoreOutputFormat = Literal["wav", "flac", "mp3", "aac", "opus"]


class RestoreParams(BaseModel):
    """Parameters for a standalone Diamond speech-restoration job."""

    output_format: RestoreOutputFormat = "flac"
    chunk_sec: float = Field(2.5, ge=0.5, le=10.0)
    overlap_sec: float = Field(0.4, ge=0.0, le=2.0)
    rep_penalty: float = Field(1.3, ge=1.0, le=2.0)


class RestoreResult(BaseModel):
    token: str
    filename: str
    output_format: str
    sample_rate: int
    duration_sec: float
    device_used: str
    warnings: list[WarningItem] = []
