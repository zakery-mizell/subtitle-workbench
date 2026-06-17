from __future__ import annotations

import gc
import os
import shutil
from pathlib import Path
from typing import Any

from fastapi import HTTPException

from .config import settings
from .runtime_warnings import suppress_known_audio_stack_warnings
from .schemas import WarningItem


def _emit_progress(message: str) -> None:
    print(f"[WhisperX] {message}", flush=True)


def _install_windows_hf_symlink_fallback() -> None:
    if os.name != "nt":
        return

    import huggingface_hub.file_download as file_download

    if getattr(file_download, "_subtitle_workbench_symlink_patch", False):
        return

    original_create_symlink = file_download._create_symlink

    def patched_create_symlink(src: str, dst: str, new_blob: bool = False) -> None:
        try:
            return original_create_symlink(src, dst, new_blob=new_blob)
        except OSError as exc:
            if getattr(exc, "winerror", None) != 1314:
                raise

            abs_src = os.path.abspath(os.path.expanduser(src))
            abs_dst = os.path.abspath(os.path.expanduser(dst))
            abs_dst_folder = os.path.dirname(abs_dst)
            resolved_dst_folder = str(Path(abs_dst_folder).expanduser().resolve())
            file_download._are_symlinks_supported_in_dir[resolved_dst_folder] = False

            Path(abs_dst_folder).mkdir(parents=True, exist_ok=True)
            try:
                os.remove(abs_dst)
            except OSError:
                pass

            if new_blob:
                shutil.move(abs_src, abs_dst, copy_function=file_download._copy_no_matter_what)
            else:
                shutil.copyfile(abs_src, abs_dst)

    file_download._create_symlink = patched_create_symlink
    file_download._subtitle_workbench_symlink_patch = True


def transcribe_with_whisperx(
    audio_path: str,
    model_name: str,
    requested_language: str | None,
    hotwords: str | None = None,
) -> tuple[dict[str, Any], list[WarningItem], bool]:
    with suppress_known_audio_stack_warnings():
        import torch
        import whisperx

        _install_windows_hf_symlink_fallback()
        Path(settings.whisper_cache_dir).mkdir(parents=True, exist_ok=True)
        os.environ["HF_HOME"] = settings.whisper_cache_dir
        os.environ["XDG_CACHE_HOME"] = settings.whisper_cache_dir

        gpu_enabled = torch.cuda.is_available()
        device = "cuda" if gpu_enabled else "cpu"
        compute_type = "float16" if gpu_enabled else "float32"
        batch_size = 8 if gpu_enabled else 1
        warnings: list[WarningItem] = []
        effective_hotwords = (hotwords or "").strip() or None

        asr_model = None
        align_model = None

        try:
            audio_array = whisperx.load_audio(audio_path)
            _emit_progress(
                f"Loading model '{model_name}' on {device.upper()}"
                + (f" with language '{requested_language}'" if requested_language else " with language auto-detect")
                + "..."
            )

            try:
                asr_model = whisperx.load_model(
                    model_name,
                    device,
                    compute_type=compute_type,
                    language=requested_language,
                    asr_options={"hotwords": effective_hotwords} if effective_hotwords else None,
                    vad_method="silero",
                    download_root=settings.whisper_cache_dir,
                )
                result = asr_model.transcribe(
                    audio_array,
                    batch_size=batch_size,
                    language=requested_language,
                    print_progress=True,
                )
                _emit_progress("Transcription pass complete.")
            except torch.OutOfMemoryError as exc:
                raise HTTPException(
                    status_code=507,
                    detail=(
                        f"WhisperX model '{model_name}' could not fit in GPU memory. Close other GPU-heavy "
                        "applications or choose a smaller model such as medium, small, or base."
                    ),
                ) from exc
            except RuntimeError as exc:
                if gpu_enabled and "out of memory" in str(exc).lower():
                    raise HTTPException(
                        status_code=507,
                        detail=(
                            f"WhisperX model '{model_name}' could not fit in GPU memory. Close other GPU-heavy "
                            "applications or choose a smaller model such as medium, small, or base."
                        ),
                    ) from exc
                raise
            finally:
                if asr_model is not None:
                    del asr_model
                    asr_model = None
                    gc.collect()
                    if gpu_enabled:
                        torch.cuda.empty_cache()

            segments = result.get("segments") or []
            align_language = str(result.get("language") or requested_language or "").strip() or None
            if segments and align_language:
                try:
                    _emit_progress(f"Running word alignment for language '{align_language}'...")
                    align_model, align_metadata = whisperx.load_align_model(
                        language_code=align_language,
                        device=device,
                        model_dir=settings.whisper_cache_dir,
                    )
                    aligned = whisperx.align(
                        segments,
                        align_model,
                        align_metadata,
                        audio_array,
                        device,
                        return_char_alignments=False,
                    )
                    result["segments"] = aligned.get("segments") or segments
                    _emit_progress("Word alignment complete.")
                except Exception as exc:
                    warnings.append(
                        WarningItem(
                            code="alignment_fallback",
                            message=(
                                "WhisperX alignment was unavailable for this file, so missing word timings were "
                                f"estimated from segment boundaries. Details: {exc}"
                            ),
                        )
                    )
                    _emit_progress("Alignment unavailable. Estimating missing word timings from segment boundaries.")

            return result, warnings, gpu_enabled
        finally:
            if align_model is not None:
                del align_model
            gc.collect()
            if 'torch' in locals() and gpu_enabled:
                torch.cuda.empty_cache()
