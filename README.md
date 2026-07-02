# Subtitle Workbench

Local web app for:

- transcribing audio with WhisperX / faster-whisper
- generating editable `.srt` subtitles
- generating a paragraph transcript `.txt`
- creating a second `.srt` edit guide with `SILENT`, `CUT`, and `REPEAT` blocks
- Auphonic-style audio post production ("Master" tab), fully local:
  - AI noise reduction (MossFormer2-SE-48K via ClearerVoice-Studio)
  - hum removal (50/60 Hz + harmonics, auto-detected)
  - adaptive high-pass filtering
  - adaptive speech leveler with soft-knee compression
  - loudness normalization (-16 podcast / -14 streaming / -23 EBU R128 / custom) with a 4x-oversampled true-peak limiter
  - silence and filler-word ("um", "uh") cutting with subtitle timestamp remapping and Audacity label export
  - before/after loudness report (LUFS, LRA, true peak, noise floor)
  - encoded export: wav, flac, mp3, aac, opus

## What is implemented

- drag and drop audio upload
- WhisperX model dropdown, defaulted to `large-v3`, with `turbo` also available
- GPU-first backend setup for NVIDIA CUDA on Windows
- optional speaker diarization path with named speakers
- transcript review view with clickable words and low-confidence highlighting
- subtitle editing view with:
  - Enter to split into a new caption
  - Shift+Enter for multiline captions
  - multiline caption indicator bar on the left
  - blank-line spacing between caption groups
  - extend-caption-to-next action
  - manual `CUT`, `REPEAT`, and `SILENT` guide blocks
- find/replace
- undo/redo
- playback skip toggle for guide blocks
- waveform timeline analysis that draws speech regions and subtitle blocks
- waveform snapping for subtitle start/end edges near detected speech onsets and offsets
- speaker handoff / possible overlap markers on the waveform timeline
- project file export/import for full-session resume, including guide blocks, timing metadata, confidence data, and embedded audio
- export for:
  - `captions.srt`
  - `edit-guide.srt`
  - `transcript.txt`

## Important setup note about multiple speakers

Whisper does not do speaker diarization by itself. The backend uses `pyannote.audio` for diarization when:

- `speaker_count > 1`
- `DIARIZATION_AUTH_TOKEN` is set in `.env`

Without that token, the app still runs, but it falls back to a single-speaker assignment and returns a warning in the UI. That is why speaker names/count alone are not enough to make speakers work.

For `pyannote/speaker-diarization-3.1`, the account also needs gated access to:

- `https://hf.co/pyannote/segmentation-3.0`

The first multi-speaker run also needs internet access to download the gated pyannote models. After those assets are cached locally, later runs can work offline.
For very long files, the backend skips diarization by default and returns a single-speaker transcript with a warning. This avoids a second long GPU-heavy pass after Whisper finishes.

## Audio mastering ("Master" tab)

The mastering pipeline runs entirely locally as a background job with staged progress:

decode (48 kHz float) → hum removal → AI denoise → adaptive high-pass → cutting → adaptive leveler → compressor → loudness normalization → true-peak limit → encode

Notes:

- The MossFormer2 denoiser model (an extra download from Hugging Face on first use) is installed via `pip install clearvoice --no-deps` plus its runtime dependencies; if it is missing, mastering still runs and simply skips denoising with a warning. ClearVoice stores its checkpoints in `./checkpoints` relative to the backend working directory (the repo root when using the run scripts).
- Cut lists are computed on the original timeline. "Apply cuts to subtitles" shifts every caption, word, and guide block to match the shortened audio (undo restores the original timing). "Detect only" mode exports an Audacity label track instead.
- After mastering, the player gets an Original/Mastered A/B toggle.
- Long files: mastering is limited to 4 hours per file.

## Project layout

- `backend/app/main.py`
- `backend/app/text_processing.py`
- `backend/app/mastering/` (audio post production pipeline)
- `backend/app/jobs.py` (in-process job registry for mastering)
- `frontend/src/App.tsx`
- `frontend/src/MasteringPanel.tsx`
- `scripts/install.ps1` (Windows) / `scripts/install.sh` (macOS)

## Configure

1. Copy `.env.example` to `.env`.
2. Set `DIARIZATION_AUTH_TOKEN` if you want automatic multi-speaker diarization.
3. Optional: set `DIARIZATION_MAX_DURATION_SECONDS` if you want to allow multi-speaker diarization on longer files. The default is `3600` seconds.
4. If you want the Whisper model cache elsewhere, update `WHISPER_CACHE_DIR`.
5. Make sure `ffmpeg` is on `PATH`; waveform analysis and range retranscription use it to decode audio/video.

## Requirements

- Windows PowerShell, or macOS with bash/zsh
- Python 3.12
- Node.js and npm
- ffmpeg on `PATH`
- NVIDIA GPU recommended for larger WhisperX models and fast AI denoising (Apple Silicon uses MPS on macOS)

## Install

Windows:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\install.ps1
```

macOS:

```bash
./scripts/install.sh
```

That script will:

1. create `.venv` with Python 3.12, unless `SUBTITLE_WORKBENCH_VENV` or `.venv-path` points elsewhere
2. install CUDA 12.8 PyTorch packages compatible with WhisperX 3.8.6
3. install backend dependencies
4. install frontend dependencies

## Run

Single-click on Windows:

```powershell
.\Run Subtitle Workbench.bat
```

This opens separate backend and frontend PowerShell windows, then opens `http://localhost:5173`.

Backend:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run-backend.ps1
```

Frontend:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run-frontend.ps1
```

Then open `http://localhost:5173`.

macOS equivalents:

```bash
./scripts/run-app.sh        # backend + frontend together
./scripts/run-backend.sh
./scripts/run-frontend.sh
```

## Current limitations

- project export embeds the source audio, so large projects create large JSON files
- transcript and subtitle text are both directly editable in the UI
- diarization quality depends on `pyannote.audio` and the Hugging Face token-backed model access
- `huggingface_hub` must stay below `1.0` for the current `pyannote.audio` version in this project
- large models download on first use and need enough GPU memory; if memory is tight, choose `medium`, `small`, `base`, or `tiny`
- long-form transcription uses Whisper with `condition_on_previous_text=False` to reduce repetition loops on large files

## Tests

Run:

```powershell
.\.venv\Scripts\python.exe -m unittest discover backend/tests
```
