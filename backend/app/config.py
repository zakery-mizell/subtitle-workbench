from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


ROOT_DIR = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    app_name: str = "Subtitle Workbench"
    cors_origin: str = "http://localhost:5173"
    whisper_cache_dir: str = str(ROOT_DIR / "models" / "whisper")
    temp_upload_dir: str = str(ROOT_DIR / "tmp" / "uploads")
    diarization_auth_token: str | None = None
    diarization_max_duration_seconds: float = 3600.0
    low_confidence_threshold: float = 0.55
    silence_seconds: float = 5.0
    mastering_output_dir: str = str(ROOT_DIR / "tmp" / "mastering")
    mastering_device: str = "auto"
    mastering_job_ttl_seconds: float = 14400.0
    model_cache_dir: str = str(ROOT_DIR / "models")
    speaker_profile_dir: str = str(ROOT_DIR / "models" / "speaker_profiles")
    # Diamond restore device (env RESTORE_DEVICE). None = auto: cuda if
    # available else cpu (never auto-mps; see restore/engine.py device notes).
    restore_device: str | None = None
    # Sidon restore device (env SIDON_DEVICE). None = auto: cuda if available
    # else cpu. mps is rejected outright (see restore/sidon_engine.py notes).
    sidon_device: str | None = None
    # Seed-VC voice-conversion device (env CONVERSION_DEVICE). None = auto: cuda,
    # else mps, else cpu (mps IS auto-picked here; see conversion/engine.py notes).
    conversion_device: str | None = None
    # F5-TTS speech-edit device (env SPEECHEDIT_DEVICE). None = auto: cuda, else
    # mps, else cpu (mps IS auto-picked here; see speechedit/engine.py notes).
    speechedit_device: str | None = None
    # Qwen3-TTS voice-clone device (env TTS_DEVICE). None = auto: cuda, else
    # mps, else cpu (mps IS auto-picked here; see tts/engine.py device notes).
    tts_device: str | None = None

    model_config = SettingsConfigDict(
        env_file=str(ROOT_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()
