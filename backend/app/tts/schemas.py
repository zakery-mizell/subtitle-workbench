from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from ..schemas import WarningItem

TtsLanguage = Literal[
    "Auto",
    "Chinese",
    "English",
    "German",
    "Italian",
    "Portuguese",
    "Spanish",
    "Japanese",
    "Korean",
    "French",
    "Russian",
]
TtsModelSize = Literal["1.7b", "0.6b"]
TtsOutputFormat = Literal["wav", "flac", "mp3", "aac", "opus"]


class TtsParams(BaseModel):
    """Parameters for a Qwen3-TTS voice-cloning synthesis job."""

    text: str = Field(min_length=1, max_length=20000)
    language: TtsLanguage = "Auto"
    # A reference transcript unlocks the higher-fidelity ICL clone. When blank,
    # `auto_ref_text` lets the server transcribe the clip for you; when that is
    # off too, the voice is cloned from its speaker signature alone.
    ref_text: str | None = None
    auto_ref_text: bool = True
    whisper_model: str = "small"
    model_size: TtsModelSize = "1.7b"
    output_format: TtsOutputFormat = "flac"


class TtsResult(BaseModel):
    token: str
    filename: str
    output_format: str
    sample_rate: int
    duration_sec: float
    device_used: str
    model_size: str
    # "transcript" = ICL clone (a reference transcript was available);
    # "voice-signature" = x-vector-only clone (no transcript, lower fidelity).
    clone_mode: Literal["transcript", "voice-signature"]
    ref_text_used: str | None = None
    warnings: list[WarningItem] = []
