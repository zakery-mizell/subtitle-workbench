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
- AI overlap separation ("Overlaps" tab + automatic solo tracks), fully local:
  - detects regions where speakers talk over each other (raw pyannote diarization)
  - automatically renders per-speaker solo tracks so "Only <name>" playback isolates each voice even inside overlaps
  - spotlight one voice above a ducked mix, or mute a voice entirely, per overlap region
  - recovers words that transcription lost inside overlaps and adds them to the transcript
  - exports simultaneous speech as broadcast-style dialogue subtitles (one line per speaker)

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
- deterministic sentence-aware caption splitting (max 2 lines, 42 chars/line, no sentence carryover) with a "Re-split Captions" action
- instant Original/Mastered A/B playback (both versions stay loaded and in sync; switching is gapless)
- captions auto-scroll with playback (suspends briefly while you scroll manually)
- custom transport (skip/play/speed/mute) with the waveform as the scrubber; waveform analysis runs automatically on load
- caption list rendered as editor rows with clickable timecode gutters
- light and dark themes (toggle in the view-options gear)
- drag a file anywhere in the window to load it; transcription setup lives in a slide-over drawer once a session is open
- speaker handoff / possible overlap markers on the waveform timeline
- project file export/import for full-session resume, including guide blocks, timing metadata, confidence data, and embedded audio
- export for:
  - `captions.srt`
  - `edit-guide.srt`
  - `transcript.txt`

## Important setup note about multiple speakers

Whisper does not do speaker diarization by itself. The backend uses `pyannote.audio` 4.x with the
`pyannote/speaker-diarization-community-1` model when:

- `speaker_count > 1`
- `DIARIZATION_AUTH_TOKEN` is set in `.env`

Without that token, the app still runs, but it falls back to a single-speaker assignment and returns a warning in the UI. That is why speaker names/count alone are not enough to make speakers work.

The token's Hugging Face account needs gated access to:

- `https://hf.co/pyannote/speaker-diarization-community-1`

The first multi-speaker run also needs internet access to download the gated pyannote models. After those assets are cached locally, later runs can work offline.

Diarization uses community-1's exclusive mode (a single most-likely speaker at any moment), which
keeps word-level speaker assignment stable through overlaps and backchannels. Multi-speaker runs
default to the "Word (tighter switches)" speaker timing mode so captions split exactly where the
speaker changes; "Segment (stable)" is still available in the setup drawer.

When a transcript has multiple speakers, the transport bar gains a speaker selector ("All speakers"
/ "Only <name>") that mutes playback outside the chosen speaker's words, so you can audit one voice
at a time. It affects playback only, never exports. With the UniSE overlap engine installed, the
selector goes further: overlap sections play that speaker's AI-separated voice alone (see below).
For very long files, the backend skips diarization by default and returns a single-speaker transcript with a warning. This avoids a second long GPU-heavy pass after Whisper finishes.

## Audio mastering ("Master" tab)

The mastering pipeline runs entirely locally as a background job with staged progress:

decode (48 kHz float) → hum removal → AI denoise → adaptive high-pass → cutting → adaptive leveler → compressor → loudness normalization → true-peak limit → encode

Notes:

- The MossFormer2 denoiser model (an extra download from Hugging Face on first use) is installed via `pip install clearvoice --no-deps` plus its runtime dependencies; if it is missing, mastering still runs and simply skips denoising with a warning. ClearVoice stores its checkpoints in `./checkpoints` relative to the backend working directory (the repo root when using the run scripts).
- Cut lists are computed on the original timeline. "Apply cuts to subtitles" shifts every caption, word, and guide block to match the shortened audio (undo restores the original timing). "Detect only" mode exports an Audacity label track instead.
- After mastering, the player gets an Original/Mastered A/B toggle.
- Long files: mastering is limited to 4 hours per file.

## Overlapping speech ("Overlaps" tab + automatic solo tracks)

When more than one speaker is configured, transcription also returns the raw (non-exclusive)
diarization, so genuine overlaps survive. They show as translucent bands on the waveform and an
"N overlaps to untangle" chip.

Everything below needs the UniSE engine installed first (about 2.8 GB of checkpoints):

```bash
./scripts/install-unise.sh        # macOS
powershell -ExecutionPolicy Bypass -File .\scripts\install-unise.ps1   # Windows
```

What it does:

- **Automatic solo tracks** — after transcription, a background job renders one full-length track
  per speaker in which every overlap is replaced by that speaker's isolated voice (UniSE
  target-speaker extraction, enrollment picked from the speaker's nearest clean solo span). The
  "Only <name>" selector uses these tracks transparently; "All speakers" playback always stays the
  untouched original, so a false-positive overlap can never degrade normal listening. Finished
  tracks persist across page reloads (artifacts are revalidated instead of re-rendered).
- **Overlaps tab** — per-region controls to *spotlight* one voice (duck the original mix ~11 dB and
  lay the separated voice on top) or *mute* one voice (replace the region with the
  everyone-but-X reconstruction). The result plugs into the instant Original/Separated A/B toggle.
- **Word recovery** — spotlighted stems are transcribed with WhisperX; recovered words (usually
  lost, because mixed audio transcribes only the dominant voice) can be added to the transcript as
  a caption for that speaker with one click.
- **Dialogue subtitles** — captions of different speakers that share more than half of the shorter
  one's duration export as a single dialogue cue (one line per speaker, "- " dashes or "NAME: "
  prefixes), so the SRT never contains overlapping timecodes. In the editor those captions stay
  separately editable, marked with an overlap badge, and playback highlights both.

Notes:

- Separation is generative 16 kHz re-synthesis confined to the overlap regions (the rest of the
  audio is untouched, bit-for-bit). Words can occasionally warble — review recovered text before
  trusting it, and treat spotlight/mute renders as listening/repair aids rather than mastering
  output.
- Inference cost scales with total overlap duration x speakers involved, not file length
  (roughly 2.4x real-time per speaker per overlap on Apple Silicon MPS; much faster on CUDA).
- UniSE model code is vendored into `vendor/unified-audio/` and checkpoints into
  `checkpoints/unise/` by the install script; `backend/tools/demo_unise.py` builds a synthetic
  two-voice overlap clip and verifies the engine end to end.

## Speech restoration ("Restore")

[Diamond](https://huggingface.co/nineninesix/diamond-1.0) (`nineninesix/diamond-1.0`) is a
sequence-to-sequence model that resynthesizes clean speech from degraded input. Unlike the
mastering denoiser, it regenerates the voice rather than filtering it, so it can repair heavy
codec artefacts, clipping, and muffling that subtractive tools cannot.

Install it first (python deps plus the model checkpoint, downloaded on install or lazily on first
use):

```bash
./scripts/install-restore.sh        # macOS
powershell -ExecutionPolicy Bypass -File .\scripts\install-restore.ps1   # Windows
```

Two usage modes:

- **Standalone restore** — a background job that restores a whole uploaded file end to end and
  returns a 44.1 kHz artifact (`POST /api/restore`, polled like mastering/separation). No
  transcription or diarization is involved.
- **Restored solo tracks** — pass `"restore": true` in the automatic solo-tracks job and each
  assembled per-speaker track is run through Diamond after the overlaps are stitched in. If the
  engine is unavailable the job degrades gracefully: it keeps the unrestored track and adds a
  warning.

Notes / caveats:

- Diamond is **generative resynthesis**: content above ~8 kHz is generated, not recovered. Treat
  the output as a repair/listening aid rather than a faithful capture.
- It is **English-trained** and can occasionally smear or invent words on badly clipped audio —
  listen before trusting it.
- Slow on CPU (~11x real-time on Apple Silicon; much faster on CUDA). The device auto-policy is
  cuda if available else cpu — it never auto-picks MPS, which measured *slower* than CPU here
  because the tiny autoregressive kernels are launch-bound. Set `RESTORE_DEVICE` to override.

## Voice cloning ("Voice")

[Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) (`Qwen/Qwen3-TTS-12Hz-1.7B-Base`, Apache-2.0) is
an open-weights text-to-speech model with instant voice cloning: give it a few seconds of any
voice and it speaks arbitrary text in that voice. Everything runs locally — the reference clip
and the synthesized audio never leave the machine.

Install it first (python deps plus the 1.7B checkpoint, ~4.5 GB):

```bash
./scripts/install-tts.sh        # macOS
powershell -ExecutionPolicy Bypass -File .\scripts\install-tts.ps1   # Windows
```

How it works (the "Voice" tab):

- Drop in a reference clip (audio or video; a clean 3–10 second sample of one voice works best;
  clips longer than 30 s are trimmed), type the text to speak, and run. The job is polled like
  mastering/restore (`POST /api/tts`).
- Cloning is highest-fidelity in **transcript mode**: the model conditions on the reference
  audio *and* its transcript. The panel auto-transcribes the clip with WhisperX by default, or
  you can type/correct the transcript yourself.
- With no transcript at all (auto-transcribe off or failed), it falls back to **voice-signature
  mode** — cloning from the speaker embedding alone. It still sounds like the speaker, but with
  lower fidelity; the result is labeled accordingly.
- Language select covers the 10 supported languages (Chinese, English, German, Italian,
  Portuguese, Spanish, Japanese, Korean, French, Russian) plus Auto-detect.
- Advanced: a smaller `0.6b` model variant (faster, ~2.5 GB, downloads lazily on first use) and
  the WhisperX model used for reference transcription.

Notes / caveats:

- Long texts are synthesized sentence-by-sentence in chunks, so progress is visible and memory
  stays bounded.
- A typed transcript that does not match the clip can derail ICL generation into a ramble;
  synthesis is token-capped per chunk (~3 codec frames per character), so a bad transcript
  truncates instead of hanging the job. Measured on Apple Silicon MPS: roughly 6x real-time
  for both model sizes with an accurate transcript (a one-minute job for ~7 s of speech,
  including reference transcription); much faster on CUDA.
- Device auto-policy is cuda → mps → cpu (unlike Diamond, this model is compute-bound, so Apple
  Silicon MPS is worth it; voice clone needs float32 on MPS). Set `TTS_DEVICE` to override.
- Clone responsibly: only clone voices you have the right to use.

## Project layout

- `backend/app/main.py`
- `backend/app/text_processing.py`
- `backend/app/mastering/` (audio post production pipeline)
- `backend/app/separation/` (UniSE overlap separation: engine, blending, overlap math)
- `backend/app/restore/` (Diamond speech-restoration engine + job service)
- `backend/app/tts/` (Qwen3-TTS voice-cloning engine + job service)
- `backend/app/jobs.py` (in-process job registry shared by mastering, separation, restore, and TTS)
- `frontend/src/App.tsx`
- `frontend/src/MasteringPanel.tsx`
- `frontend/src/OverlapsPanel.tsx`
- `frontend/src/VoicePanel.tsx`
- `scripts/install.ps1` (Windows) / `scripts/install.sh` (macOS)
- `scripts/install-unise.ps1` / `scripts/install-unise.sh` (optional overlap-separation engine)
- `scripts/install-restore.ps1` / `scripts/install-restore.sh` (optional Diamond restore engine)
- `scripts/install-tts.ps1` / `scripts/install-tts.sh` (optional Qwen3-TTS voice-cloning engine)

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

macOS single-click:

Double-click `Run Subtitle Workbench (Mac).command` in Finder. It runs the installer if needed, starts the backend and frontend, and opens `http://localhost:5173`. Close the Terminal window (or press Ctrl+C) to stop it.

macOS equivalents (from a terminal):

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
