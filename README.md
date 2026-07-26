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

## Voice conversion ("Convert")

[Seed-VC](https://github.com/Plachtaa/seed-vc) (V2, the `hubert-bsqvae-small` model) re-renders a
source performance in the timbre of a reference voice: the words, timing, and delivery come from
the source, the voice from the reference. Unlike restoration (which regenerates the *same* voice),
conversion *swaps* the voice — useful for re-voicing a rough phone recording with a clean clip of
the target speaker. It is zero-shot and speech-only (no singing/f0 mode); output is 22.05 kHz mono.

**It is not audio upscaling, and its input threshold is stricter than your ears.** What it
re-voices is read from the audio (HuBERT features → ASTRAL tokens), never from the transcript, so
there is nothing to fall back on when the audio is ambiguous. A narrowband phone recording can be
perfectly understandable to you and still be unusable here, because the encoder relies on cues
that were never recorded (fricative energy above the phone band, crisp stop releases); the result
is fluent but wrong words, and raising intelligibility CFG only articulates the wrong tokens more
clearly. Denoise/restoration helps with noise and reverb, not with a missing top octave. When a
source is that far gone, replace the *performance* before converting — re-speak it (ADR), or
synthesize each caption in **Patch** with its SRT duration so the original timing survives. See
"Known limitation" under §16 of `IMPLEMENTED_FEATURES.md` for the full pipeline, including the
not-yet-implemented F0/energy contour transfer that would carry the original delivery onto a
synthesized take.

Install it first (the two missing python deps plus the model checkpoints, downloaded on install or
lazily on first use):

```bash
./scripts/install-convert.sh        # macOS
powershell -ExecutionPolicy Bypass -File .\scripts\install-convert.ps1   # Windows
```

Usage: upload a **source recording** and a **target voice reference**, then run
(`POST /api/convert`, polled like mastering/restore/separation). Both clips are decoded with ffmpeg
and handed to the model; the output artifact is served/deleted like the other engines.

Notes / caveats:

- Only the first **25 seconds** of the reference are used (longer clips are trimmed with a warning);
  a clean clip of one voice works best. Long sources are chunked internally (30 s windows, 5 s
  overlap) — no manual splitting needed.
- Advanced knobs: diffusion steps (default 50; higher = better/slower), length adjust, similarity
  and intelligibility CFG, and a "convert style/accent too" toggle that engages the AR model to
  transfer accent and emotion, not just timbre.
- Device auto-policy is cuda → mps → cpu (MPS *is* auto-picked here — the diffusion/AR stack is
  compute-bound, unlike Diamond); float16 on cuda, float32 on mps/cpu. Set `CONVERSION_DEVICE` to
  override.
- Seed-VC model code is vendored into `vendor/seed-vc/` (not pip-installable); checkpoints land in
  the repo's `models/` Hugging Face cache.

## Speech editing ("Patch")

[F5-TTS](https://github.com/SWivid/F5-TTS) (`F5TTS_v1_Base`) is a flow-matching TTS model that works
in the mel domain, which unlocks two modes under one **Patch** tab:

- **Patch words** (the headline feature): upload a full recording, mark one or more time spans whose
  words are garbled or misspoken, type the replacement text per span, and F5-TTS regenerates *only*
  those spans by mel-domain infilling conditioned on the surrounding audio — so the patch inherits
  the surrounding voice and prosody. Audio outside the edited windows stays bit-identical.
- **Generate speech**: zero-shot voice-cloned TTS — upload a reference clip (plus its transcript,
  auto-transcribed with WhisperX if you leave it blank) and synthesize arbitrary text in that voice.

Both modes emit 24 kHz mono. Install it first (python deps plus the checkpoint and vocoder):

```bash
./scripts/install-speechedit.sh        # macOS
powershell -ExecutionPolicy Bypass -File .\scripts\install-speechedit.ps1   # Windows
```

Usage: one upload plus `params_json` (`POST /api/speech-edit`, polled like the other engines). The
upload is the full recording in Patch mode and the reference clip in Generate mode; the output
artifact is served/deleted like everything else.

Notes / caveats:

- Each edit span is patched inside a padded **window** (4 s of context each side, capped at 25 s
  total; spans over 20 s are rejected). The recording is transcribed once with WhisperX; each
  window's target transcript is built by swapping the edited span's words for your replacement text.
  If degraded audio will not transcribe, supply a per-span **window text override**.
- Edits are validated (non-overlapping, in-bounds, sorted) and spliced back-to-front with 0.15 s
  crossfades so earlier patches never shift later timestamps. An optional per-span **target
  duration** re-times a patch instead of matching the original span length.
- Generate mode **requires** a reference transcript — F5-TTS has no x-vector fallback, so a failed
  auto-transcription fails the job with a clear message. The reference is trimmed to 12 s (with a
  warning) so the transcript matches what the model sees.
- Device auto-policy is cuda → mps → cpu (MPS *is* auto-picked — this is a compute-bound
  transformer); F5-TTS enforces fp32 off-cuda itself. Set `SPEECHEDIT_DEVICE` to override.
- `f5-tts` is installed with `--no-deps` (its gradio/bitsandbytes pins would collide); `pip check`
  will flag its own unmet pins, which is expected noise. Checkpoints land in the repo's `models/` HF
  cache.

## Project layout

- `backend/app/main.py`
- `backend/app/text_processing.py`
- `backend/app/mastering/` (audio post production pipeline)
- `backend/app/separation/` (UniSE overlap separation: engine, blending, overlap math)
- `backend/app/restore/` (Diamond speech-restoration engine + job service)
- `backend/app/conversion/` (Seed-VC voice-conversion engine + job service)
- `backend/app/speechedit/` (F5-TTS speech-edit engine + windowing/job service)
- `backend/app/jobs.py` (in-process job registry shared by mastering, separation, restore, conversion, and speech-edit)
- `frontend/src/App.tsx`
- `frontend/src/MasteringPanel.tsx`
- `frontend/src/OverlapsPanel.tsx`
- `frontend/src/ConvertPanel.tsx`
- `frontend/src/PatchPanel.tsx`
- `scripts/install.ps1` (Windows) / `scripts/install.sh` (macOS)
- `scripts/install-unise.ps1` / `scripts/install-unise.sh` (optional overlap-separation engine)
- `scripts/install-restore.ps1` / `scripts/install-restore.sh` (optional Diamond restore engine)
- `scripts/install-convert.ps1` / `scripts/install-convert.sh` (optional Seed-VC voice-conversion engine)
- `scripts/install-speechedit.ps1` / `scripts/install-speechedit.sh` (optional F5-TTS speech-edit engine)

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
