from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from ..schemas import WarningItem

RestoreOutputFormat = Literal["wav", "flac", "mp3", "aac", "opus"]
# Diamond regenerates the voice (44.1 kHz, English); Sidon cleanses SSL features
# (48 kHz, multilingual, ~50x faster). See restore/engine.py and sidon_engine.py.
RestoreEngineName = Literal["sidon", "diamond"]


class RestoreParams(BaseModel):
    """Parameters for a standalone speech-restoration job."""

    engine: RestoreEngineName = "sidon"
    output_format: RestoreOutputFormat = "flac"
    # Diamond only: its internal autoregressive windowing.
    chunk_sec: float = Field(2.5, ge=0.5, le=10.0)
    overlap_sec: float = Field(0.4, ge=0.0, le=2.0)
    rep_penalty: float = Field(1.3, ge=1.0, le=2.0)
    # Sidon only: attention window. Longer = more context, quadratically more
    # memory (15 s ~ 6 GB, 30 s ~ 12 GB, 96 s ~ 24 GB on a 150 s clip).
    sidon_chunk_sec: float = Field(15.0, ge=5.0, le=96.0)


class RestoreResult(BaseModel):
    token: str
    filename: str
    engine: RestoreEngineName
    output_format: str
    sample_rate: int
    duration_sec: float
    device_used: str
    warnings: list[WarningItem] = []
