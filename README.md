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

### Remembering speaker identities

Diarization produces anonymous voice clusters; a typed name does not tell the
model what that person sounds like. In the Source panel, **Learn voice** saves a
local voice profile from a clean solo clip or an already isolated track. On
later recordings, entering the same speaker name makes the workbench match the
anonymous clusters to the saved voices automatically. Profiles stay local in
`models/speaker_profiles/` and only need to be learned once.

Without a saved profile, names still fall back to detected speaking order. That
is convenient but inherently fallible when the diarizer briefly invents a
speaker at the beginning or misses someone's first short reply.

The first multi-speaker run also needs internet access to download the gated pyannote models. After those assets are cached locally, later runs can work offline.

Diarization uses community-1's exclusive mode (a single most-likely speaker at any moment). Speaker
assignment always uses one hybrid policy: word timing places precise handoffs, segment timing repairs
a short stray edge when diarization splits one coherent aligned sentence, and the raw targeted-window
UniSE audit can correct clear mistakes without fragmenting a segment that already has a strong speaker
majority. There is no timing-mode selector. All transcription and generated speech is
English-only; language auto-detection and language selectors are intentionally disabled.

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

- **Automatic targeted solo tracks** — after transcription, a background job renders one
  timeline-preserving track per speaker. UniSE target-speaker extraction inspects merged
  windows around detected overlaps and rapid speaker handoffs, but replaces audio **only inside
  genuine overlaps**. Ordinary handoffs keep the untouched recording. UniSE always receives at least its
  native five seconds of surrounding context, while audible corrections remain confined to overlapping speech. Those windows use a nearby clean
  solo voice sample; everywhere else keeps the untouched original sound. The assembled track is
  then muted outside that speaker's speech regions. The
  "Only <name>" selector uses these tracks transparently; "All speakers" playback always stays the
  untouched original, so a false-positive overlap can never degrade normal listening. Finished
  tracks persist across page reloads (artifacts are revalidated instead of re-rendered).
- **Per-speaker transcripts** — every rendered track is also transcribed on its own with WhisperX
  (the *speaker-based* transcript, complementing the word-based mixture transcript). Because each
  stem carries a single voice, its words survive overlaps that the mixture transcript loses. The
  Export tab offers per-speaker subtitles (.srt) and transcript (.txt), timed on the shared
  timeline so they line up with the isolated audio downloads.
- **Automatic handoff auditing** — on every diarized multi-speaker upload, the app compares words
  near rapid speaker changes against the raw UniSE target-speaker window results. It automatically
  reassigns only words that are clearly present in the other speaker's stem and absent from the
  currently assigned stem; ambiguous cases stay unchanged. Confirmed changes re-split the affected
  subtitle boundary and create an undo checkpoint. Each target window is transcribed before it is
  blended or gated, so a mistaken diarization boundary cannot erase the evidence needed to correct it.
  Short pause-bounded utterances (up to three words) are also treated as candidates even when both
  surrounding gaps are too large to count as a fast handoff. Because generated UniSE stems can leak,
  these candidates are assigned by comparing the original utterance against every speaker's clean
  pyannote voice embedding; one speaker must win by conservative absolute and relative margins.
- **Overlaps on the waveform** — overlap regions are highlighted bands on the main waveform
  (scroll to zoom in, click a band to open its controls under the waveform): *spotlight* one voice
  (duck the original mix ~16 dB and lay the separated voice on top) or *mute* one voice (replace
  the region with the everyone-but-X reconstruction). The result plugs into the instant
  Original/Separated A/B toggle.
- **Word recovery** — spotlighted stems are transcribed with WhisperX; recovered words (usually
  lost, because mixed audio transcribes only the dominant voice) can be added to the transcript as
  a caption for that speaker with one click.
- **Dialogue subtitles** — captions of different speakers that share more than half of the shorter
  one's duration export as a single dialogue cue (one line per speaker, "- " dashes or "NAME: "
  prefixes), so the SRT never contains overlapping timecodes. In the editor those captions stay
  separately editable, marked with an overlap badge, and playback highlights both.

Notes:

- Separation is generative 16 kHz re-synthesis confined to overlap and rapid-handoff windows (the
  rest of the audio is untouched). Words can occasionally warble — review recovered text before
  trusting it, and treat spotlight/mute renders as listening/repair aids rather than mastering
  output.
- Automatic separation cost scales with the total duration of merged overlap/handoff windows and
  the relevant speakers in each window, rather than the full recording duration.
- UniSE model code is vendored into `vendor/unified-audio/` and checkpoints into
  `checkpoints/unise/` by the install script; `backend/tools/demo_unise.py` builds a synthetic
  two-voice overlap clip and verifies the engine end to end.

## Speech restoration ("Restore")

Two engines sit behind one panel. Both go further than the mastering denoiser, which only
filters the signal it is given.

- **Sidon** ([sarulab-speech/sidon-v0.1](https://huggingface.co/sarulab-speech/sidon-v0.1),
  ICASSP 2026) — the default. A LoRA-adapted w2v-BERT 2.0 predictor cleanses the SSL features of
  the noisy input and a decoder vocoder resynthesises 48 kHz audio from them. Multilingual,
  identity-preserving, and ~5x FASTER than real time on CPU.
- **Diamond** ([nineninesix/diamond-1.0](https://huggingface.co/nineninesix/diamond-1.0)) — a
  sequence-to-sequence RQ-Transformer that regenerates the voice outright at 44.1 kHz. The more
  aggressive repair, and the more expensive one.

Reach for Sidon first. It handles noise, reverb, and codec damage on any language and is fast
enough to iterate with. Reach for Diamond when the source is damaged past what cleaning can fix
and you are willing to accept invented detail in exchange — it is English-only and roughly 50x
slower.

Install whichever you need (checkpoints download on install, or lazily on first use):

```bash
./scripts/install-sidon.sh          # macOS — no new packages, just ~1 GB of weights
./scripts/install-restore.sh        # macOS — Diamond, plus its python deps
```

```bash
powershell -ExecutionPolicy Bypass -File .\scripts\install-sidon.ps1     # Windows
powershell -ExecutionPolicy Bypass -File .\scripts\install-restore.ps1   # Windows
```

Two usage modes, each taking an engine:

- **Standalone restore** — a background job that restores a whole uploaded file end to end
  (`POST /api/restore` with `"engine": "sidon" | "diamond"`, polled like mastering/separation).
  No transcription or diarization is involved.
- **Restored solo tracks** — pass `"restore": true` (plus an optional `"restore_engine"`) in the
  automatic solo-tracks job and each assembled per-speaker track is restored after the overlaps
  are stitched in. If the engine is unavailable the job degrades gracefully: it keeps the
  unrestored track and adds a warning.

Notes / caveats:

- Sidon **preserves the speaker**, which is why it also suits cleaning a voice-cloning reference
  before Convert or Patch — a generic noise suppressor scrubs away the cues a cloner needs.
- Sidon's memory scales with its context window, measured over a 150 s clip: 15 s of context
  needs ~6 GB, 30 s ~12 GB, 96 s ~24 GB. On CUDA that is VRAM, so the default is 15 s. Longer
  windows sound slightly better (log-mel cosine 0.989 at 15 s, 0.993 at 30 s, against a 96 s
  render). Chunk seams do not accumulate timing drift — measured envelope lag against the input
  stays within 10 ms — so restored audio remains aligned to the workspace timeline.
- Sidon's device policy is cuda if available else cpu, set via `SIDON_DEVICE`. **MPS is not
  supported**: the published checkpoints are `torch.jit.trace` exports with device-pinned
  constants, so `map_location='mps'` hits a float64 constant and moving a CPU-loaded module fails
  with "Passed CPU tensor to MPS op". CPU is fast enough that this costs little.
- Diamond is **generative resynthesis**: content above ~8 kHz is generated, not recovered. Treat
  the output as a repair/listening aid rather than a faithful capture.
- Diamond is **English-trained** and can occasionally smear or invent words on badly clipped
  audio — listen before trusting it.
- Diamond is slow on CPU (~11x real-time on Apple Silicon; much faster on CUDA). Its auto-policy
  is cuda if available else cpu — it never auto-picks MPS, which measured *slower* than CPU here
  because the tiny autoregressive kernels are launch-bound. Set `RESTORE_DEVICE` to override.
- Comparing two restored renders by waveform correlation is meaningless — both vocoders
  regenerate phase, so perceptually identical renders can correlate near zero. Compare log-mels.

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
- Language is fixed to English for synthesis and reference transcription.
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
- `backend/app/restore/` (Sidon + Diamond speech-restoration engines + job service)
- `backend/app/conversion/` (Seed-VC voice-conversion engine + job service)
- `backend/app/speechedit/` (F5-TTS speech-edit engine + windowing/job service)
- `backend/app/tts/` (Qwen3-TTS voice-cloning engine + job service)
- `backend/app/jobs.py` (in-process job registry shared by mastering, separation, restore, conversion, speech-edit, and TTS)
- `frontend/src/App.tsx`
- `frontend/src/MasteringPanel.tsx`
- `frontend/src/OverlapsPanel.tsx`
- `frontend/src/ConvertPanel.tsx`
- `frontend/src/PatchPanel.tsx`
- `frontend/src/VoicePanel.tsx`
- `scripts/install.ps1` (Windows) / `scripts/install.sh` (macOS)
- `scripts/install-unise.ps1` / `scripts/install-unise.sh` (optional overlap-separation engine)
- `scripts/install-sidon.ps1` / `scripts/install-sidon.sh` (Sidon restore engine)
- `scripts/install-restore.ps1` / `scripts/install-restore.sh` (optional Diamond restore engine)
- `scripts/install-convert.ps1` / `scripts/install-convert.sh` (optional Seed-VC voice-conversion engine)
- `scripts/install-speechedit.ps1` / `scripts/install-speechedit.sh` (optional F5-TTS speech-edit engine)
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
