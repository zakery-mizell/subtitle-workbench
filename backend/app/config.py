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
    default_language: str | None = None
    mastering_output_dir: str = str(ROOT_DIR / "tmp" / "mastering")
    mastering_device: str = "auto"
    mastering_job_ttl_seconds: float = 14400.0
    model_cache_dir: str = str(ROOT_DIR / "models")

    model_config = SettingsConfigDict(
        env_file=str(ROOT_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()
