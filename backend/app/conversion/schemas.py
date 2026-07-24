from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from ..schemas import WarningItem

ConversionOutputFormat = Literal["wav", "flac", "mp3", "aac", "opus"]


class ConversionParams(BaseModel):
    """Parameters for a Seed-VC (V2) zero-shot voice-conversion job."""

    output_format: ConversionOutputFormat = "flac"
    # 30 is the upstream default; 50 trades speed for quality, which suits an
    # offline render.
    diffusion_steps: int = Field(50, ge=10, le=100)
    length_adjust: float = Field(1.0, ge=0.5, le=2.0)
    # Named correctly on our side; the upstream call site keeps its
    # `intelligebility_cfg_rate` misspelling.
    intelligibility_cfg: float = Field(0.7, ge=0.0, le=1.0)
    similarity_cfg: float = Field(0.7, ge=0.0, le=1.0)
    # Engages the AR model so accent/emotion transfer too, not just timbre.
    convert_style: bool = False


class ConversionResult(BaseModel):
    token: str
    filename: str
    output_format: str
    sample_rate: int
    duration_sec: float
    device_used: str
    diffusion_steps: int
    warnings: list[WarningItem] = []
