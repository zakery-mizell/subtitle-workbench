# Implemented Features

This file inventories the end-user and API functionality that is clearly implemented in the current codebase.

It is based on the wired-up code in:

- `frontend/src/App.tsx`
- `frontend/src/lib/exporters.ts`
- `frontend/src/lib/glossary.ts`
- `frontend/src/lib/qa.ts`
- `frontend/src/lib/srt.ts`
- `backend/app/main.py`
- `backend/app/text_processing.py`
- `backend/app/whisperx_transcription.py`
- `backend/app/diarization.py`
- `backend/app/schemas.py`
- `backend/app/config.py`

## 1. Input and Transcription

- Drag-and-drop or file-picker upload for audio and video sources.
- Whisper model selection from `tiny` through `large-v3` and `turbo`.
- Configurable speaker count, up to 12 speakers.
- Editable speaker names before and after transcription.
- Speaker timing mode toggle for multi-speaker sessions:
  - `segment`
  - `word`
- Single project glossary / jargon list used as WhisperX hotwords during transcription.
- Optional filler-word / simple-stutter cleanup for full transcription and range retranscription.
- WhisperX transcription with GPU or CPU execution.
- Silero VAD-enabled transcription path in WhisperX.
- Word-level alignment when WhisperX alignment is available.
- Automatic fallback to coarser segment timings when alignment is unavailable, with warning messaging.
- Automatic conversion of transcription results into:
  - word tokens
  - transcript paragraphs
  - subtitle captions
  - guide blocks

## 2. Speaker Handling

- Optional multi-speaker diarization path through WhisperX / pyannote when multiple speakers are requested and a diarization token is configured.
- Fallback to single-speaker labeling when diarization is unavailable, not configured, or fails.
- Duration-based diarization cutoff for long files, with warning messaging instead of attempting an expensive diarization pass.
- Segment-level or word-level speaker assignment after diarization.
- Imported speaker labels from SRT can be normalized into app speaker identities.
- Speaker reassignment from a selected caption forward across the current contiguous speaker run.
- Per-speaker subtitle attribution can be turned on or off for export.

## 3. Transcript and Subtitle Editing

- Two editing views:
  - transcript paragraphs
  - subtitle captions
- Direct transcript editing.
- Direct subtitle editing.
- Transcript edits redistribute text back across the affected captions instead of flattening the whole session.
- Subtitle edits rebuild paragraph text for the affected transcript block.
- Timing-aware text fragments remain clickable back to the audio timeline.
- Caption line breaking uses custom subtitle-focused reflow heuristics.
- Reflow-all action for all captions.
- Blank-gap toggle after a caption.
- Multiline caption editing.
- Subtitle keyboard editing workflow:
  - `Enter` splits into a new caption
  - `Shift+Enter` inserts a new line
  - `Backspace` at the start merges backward
  - `Delete` at the end merges forward
  - arrow-key navigation across caption cards
- Undo / redo history.
- Find / replace over the editable caption text, with transcript paragraphs rebuilt from the result.

## 4. Playback and Review

- Built-in audio player.
- FFmpeg-backed waveform analysis for uploaded audio/video files.
- Canvas waveform timeline with subtitle blocks, detected speech spans, playhead, and speaker handoff markers.
- Subtitle edge snapping from waveform speech onsets/offsets, with conservative overlap prevention.
- Speaker switch / tight handoff / possible overlap markers derived from caption and word-level speaker timing.
- Current-time tracking.
- Click text to seek audio.
- Optional click-to-play autoplay behavior.
- Follow-playback auto-scroll.
- Jump to current transcript item.
- Jump to current subtitle item.
- Active timing highlights while audio plays.
- Optional timing-highlight visibility toggle.
- Low-confidence highlighting.
- Low-confidence highlights can be acknowledged by editing the affected text.
- Optional subtitle line guides.
- Keyboard shortcuts for playback and click-autoplay toggling:
  - `Ctrl+Space` play / pause
  - `Shift+Space` toggle click autoplay when focus is outside text inputs

## 5. Guide Blocks, Glossary, and QA

- Automatic guide-block generation for:
  - `SILENT`
  - `CUT`
  - `REPEAT`
- Silence detection between caption gaps.
- Repeated filler / repeated phrase detection for automatic `CUT` or `REPEAT` ranges.
- Manual guide-block creation from a transcript selection.
- Manual guide-block creation from a subtitle selection.
- Guide-block deletion.
- Per-block skip toggle.
- Playback can skip guide blocks marked as skippable.
- Collapsible side panel with:
  - guide tools
  - jargon / glossary tools
  - QA report
- Glossary / jargon dictionary stored in the working session and project file.
- Automatic jargon candidate detection based on repetition, confidence, capitalization, uncommon words, and hyphenation.
- One-click add of suggested jargon terms into the glossary.
- Exact and fuzzy glossary matching against current captions.
- Batch retranscription of caption ranges that match glossary terms.
- QA report generation for subtitle issues including:
  - more than two lines
  - line length target and hard-cap violations
  - reading-speed target and hard-cap violations
  - very short or very long durations
  - low-confidence words
  - fuzzy glossary mismatches
- QA report export as `.txt`.

## 6. Retranscription and Repair

- Retranscribe the current transcript block.
- Retranscribe the current subtitle block.
- Retranscribe a selected range from transcript text.
- Retranscribe a selected range from subtitle text.
- Glossary terms are automatically included in retranscription requests.
- Backend audio clipping for the requested time span using `ffmpeg`.
- Padding around the requested retranscription range before clipping.
- Replacement of only the selected timed region instead of rebuilding the whole session.
- Re-sync of caption / word assignments after retranscription.
- Warning when retranscription returns no words.

## 7. Import, Resume, and Persistence

- Resume workflow from `audio + .srt`.
- Two resume modes:
  - keep original SRT timing and rematch text to fresh WhisperX words
  - retime the imported subtitles to the uploaded audio
- SRT parsing with support for speaker-label lines like `Speaker Name:`.
- Project save / load as `.subtitle-workbench.json`.
- Project file includes:
  - editor state
  - guide blocks
  - speaker settings
  - timings
  - warnings
  - confidence-related data
  - glossary text
  - key UI toggles and view state
  - embedded audio for playback restoration
- Local autosave in `localStorage`.
- Restore of the most recent autosaved workspace.

## 8. Export

- Subtitle export as `.srt`.
- Transcript export as `.txt`.
- Edit-guide export as `.srt`.
- QA report export as `.txt`.
- Optional "extend subtitles to next caption on export" behavior.
- Optional export-time normalization to 30 fps boundaries.
- Speaker labels can be included in exported captions when needed.
- Speaker attribution can be disabled per speaker during subtitle export.
- Export filenames include the source stem and speaker labels.

## 9. Backend / API Surface

- `GET /api/health`
- `GET /api/capabilities`
- `POST /api/transcribe`
- `POST /api/retranscribe-range`
- `POST /api/transcribe` accepts wired-up request fields for:
  - audio upload
  - model
  - speaker count
  - speakers JSON
  - speaker assignment mode
  - optional language
  - optional hotwords
  - optional disfluency cleanup
- `POST /api/retranscribe-range` accepts wired-up request fields for:
  - audio upload
  - model
  - start / end timestamps
  - optional language
  - optional hotwords
  - optional disfluency cleanup
- Structured API responses for:
  - words
  - paragraphs
  - captions
  - guide blocks
  - warnings
  - language
  - speaker assignment mode
  - GPU usage flag

## 10. Runtime and Processing Details

- Windows Hugging Face symlink fallback patch for Whisper cache handling.
- Configurable Whisper cache directory.
- Configurable diarization cutoff.
- Configurable low-confidence threshold.
- Configurable silence threshold used for `SILENT` guide blocks.
- Configurable default language.
- Audio is normalized to mono 16 kHz WAV when needed for diarization and retranscription clipping.
- GPU out-of-memory handling returns a user-facing error that recommends closing other GPU apps or using a smaller model.
- Warning propagation from backend to UI when alignment or diarization fall back.

## 11. Audio Mastering (Master tab)

Local Auphonic-style post production implemented in `backend/app/mastering/` and `frontend/src/MasteringPanel.tsx`:

- Background job model with staged progress polling (`POST /api/master`, `GET /api/jobs/{id}`).
- Full-quality 48 kHz float processing path, separate from the 16 kHz ASR path; stereo preserved with channel-linked gains.
- AI noise reduction with MossFormer2-SE-48K (ClearerVoice-Studio), chunked with overlap crossfades, dry/wet amount, cuda→mps→cpu device selection, and graceful skip with a warning when the model is unavailable.
- Hum detection (Welch PSD peak analysis, auto 50/60 Hz) and removal via zero-phase IIR notch cascade over the harmonics.
- Adaptive zero-phase high-pass filter tuned below the detected voice fundamental.
- Speech/music/background classification (noise-floor-anchored threshold, envelope-modulation music heuristic).
- Adaptive leveler: segment-local momentary loudness, per-frame gain toward the median speech loudness with tight/moderate/soft clamps, background never boosted, flat conservative gain for music, asymmetric gain smoothing; followed by a soft-knee compressor.
- Loudness normalization to preset targets (-16 podcast, -14 streaming, -23 EBU R128, -19 mono, custom) with measure→gain→limit→re-measure convergence.
- True-peak limiter with 4x oversampling, block lookahead, and exponential release.
- Automatic cutting: silence trimming with kept-pause preservation and filler-word regions from WhisperX word timestamps; modes remove / replace-with-silence / detect-only.
- Cut list served on the original timeline (`GET /api/master/{token}/cut-list`), Audacity label export, and one-click subtitle remapping (captions, words, paragraphs, guide blocks) through the undo history.
- Before/after loudness report: integrated LUFS, loudness range, true peak, noise floor, plus per-stage details.
- Processed master playback via Original/Mastered A/B toggle, waveform refresh from the processed file, and encoded download (wav, flac, mp3, aac, opus).
- Unit tests for loudness math, limiter, hum removal, leveler, classifier, cutting/remap, job registry, and endpoint contracts.
