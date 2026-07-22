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

## 12. Caption Rules, A/B Playback, and Streamlined UI

- Deterministic sentence-first caption segmentation (`backend/app/text_processing.py`): captions never contain a mid-text sentence boundary; whole short sentences ("Yeah. Exactly.") may share one caption; long sentences split at clause boundaries; max 2 lines at 42 chars/line with abbreviation guards (Dr., initials). Constants at the top of the module control every rule.
- `POST /api/rebuild-captions` re-runs the rules on the current words; "Re-split Captions" button in the subtitles toolbar commits the result through undo history.
- Caption editor enforces the 2-line cap (Shift+Enter blocked past 2 lines; pasted extra lines fold into line 2).
- Instant A/B: original and mastered audio render as two persistent elements that co-play in lockstep; the Original/Mastered toggle is a gapless mute swap with drift correction, falling back to cut-list time mapping when the master has a shortened timeline.
- Follow playback is on by default; the caption/transcript list auto-scrolls to the active block and suspends for 3 s after manual scrolling.
- Streamlined layout: one-row transport bar (clock, A/B, view mode, jump, gear popover with view toggles), consolidated waveform strip, sticky player panel, collapsible source/setup rail with summary card, warnings as a collapsible strip, and Export moved into the side panel as its own tab.

## 13. Overlap Separation (Overlaps tab, UniSE)

- Diarization now surfaces the raw (non-exclusive) pyannote annotation alongside the exclusive one; `/api/transcribe` returns `speaker_turns` and `overlap_regions` (spans where 2+ speakers talk at once, ≥0.4 s, merged across ≤0.4 s gaps).
- Overlap regions render as translucent bands on the waveform plus an "N overlaps to untangle" chip that opens the Overlaps side-panel tab.
- Per-region controls: Spotlight one voice (duck the original mix ~11 dB and overlay the AI-separated voice) or Mute one voice (replace the region with the everyone-but-X reconstruction), with a speaker picker per region.
- Separation engine: vendored Alibaba QuarkAudio-UniSE (decoder-only AR-LM over BiCodec speech tokens, WavLM features, 16 kHz, 5 s windows) in `backend/app/separation/`; task modes `se`/`tse`/`rtse`; enrollment audio picked automatically from the target speaker's nearest clean solo span (≥1.5 s, ideally 5 s); MPS→CPU fallback; `transformers_compat.py` bridges the vendored code onto transformers ≥4.5x.
- `POST /api/separate` job (shared job registry/polling with mastering) blends stems back into the full-length original at its native sample rate with equal-power crossfades, RMS matching, and a region peak limiter; artifacts under `tmp/separation/` with audio/waveform/delete endpoints.
- Optional per-region WhisperX transcription of the spotlighted stem returns recovered words; one click adds them to the transcript as a new caption for that speaker (words in overlaps are otherwise usually lost — the mixed audio transcribes only the dominant voice).
- Result playback goes through the existing instant A/B toggle, relabeled Original/Separated; download button for the processed file.
- Install via `scripts/install-unise.sh` / `.ps1` (clones vendor repo, installs `backend/requirements-separation.txt`, downloads ~2.8 GB of checkpoints into `checkpoints/unise/`); `backend/tools/demo_unise.py` builds a synthetic two-voice overlap clip and verifies separation end to end.
- Unit tests for overlap/solo-span math, enrollment picking, and blend rendering (`backend/tests/test_separation.py`).
- Automatic solo tracks: when a transcription reports overlap regions, a background job (`POST /api/separate-solo`) renders one full-length track per involved speaker with every overlap replaced by that speaker's isolated voice (UniSE tse). The "Only <speaker>" transport selector then plays original audio in solo passages and the separated voice through overlaps — no manual steps. While soloing with a ready track, the mute gate opens on diarized turns (not just transcript words) so overlap speech whose words never transcribed is still audible. Main "All speakers" playback never uses these tracks, so false-positive overlaps cannot degrade normal listening. Status chips: "Isolating overlap voices…" / "Solo voices ready".
- Simultaneous-speech subtitles: captions of different speakers that share >50% of the shorter one's duration merge into a single broadcast-style dialogue cue on SRT export (one line per speaker, "- " dashes or "NAME: " prefixes when attribution is on), so exports never contain overlapping timecodes. In the editor those captions get an overlap badge with a left accent border, and playback highlights every caption under the playhead, not just the first. Auto solo tracks are encoded as FLAC to halve disk use.

## 14. Speech Restoration (Restore tab, Diamond)

- Generative speech resynthesis with Diamond (`nineninesix/diamond-1.0`, a sequence-to-sequence RQ-Transformer) in `backend/app/restore/` and `frontend/src/RestorePanel.tsx`: regenerates clean 44.1 kHz speech from degraded input (heavy codec artifacts, clipping, muffling) instead of filtering it, so the output replaces the input rather than blending in. English-trained; content above ~8 kHz is generated, not recovered.
- Standalone restore: `POST /api/restore` background job (shared job registry/polling with mastering and separation) restores a whole uploaded audio/video file end to end — ffmpeg decode to mono 24 kHz, restore, encode to wav/flac/mp3/aac/opus (default FLAC); artifacts under `tmp/restore/` with audio (`GET`/`HEAD /api/restore/{token}/audio`), waveform, and delete endpoints.
- Long audio is restored in 30 s segments with 1 s crossfades applied at the 44.1 kHz output rate; both RNGs are seeded for reproducibility; one global peak normalize at the end, never boosting quiet audio.
- Tunable chunk length, overlap, and repetition penalty with validated ranges, exposed in the panel under Advanced settings.
- Device policy: cuda if available else cpu — MPS is never auto-picked (measured slower than CPU on Apple Silicon because the tiny autoregressive kernels are launch-bound); `RESTORE_DEVICE` env override; cached engine per device with CPU fallback when a GPU load or segment fails.
- Restore side-panel tab: own dropzone that defaults to the workspace audio, staged progress bar with "Restoring m:ss / m:ss" messages, result chips (sample rate, duration, format), inline player, download, and discard (which deletes the server artifact).
- Restored solo tracks: the "Restore voices (Diamond)" checkbox in the Overlaps panel sets `"restore": true` on the automatic solo-tracks job, running each assembled per-speaker track through Diamond after overlap stitching; on failure the job keeps the unrestored track and adds a warning. The restore flag is part of the solo-tracks cache key so raw and restored renders never share tokens.
- Install via `scripts/install-restore.sh` / `.ps1`: an ordered mixed `--no-deps` pip sequence so descript-audiotools cannot downgrade protobuf and break onnxruntime, then the checkpoint download into the `models/` HF cache (also fetched lazily on first use); missing pieces surface as `RestoreUnavailable` errors that point at the install script.

## 15. Voice Cloning (Voice tab, Qwen3-TTS)

- Local instant voice cloning with Qwen3-TTS Base (`Qwen/Qwen3-TTS-12Hz-1.7B-Base`, Apache-2.0) in `backend/app/tts/` and `frontend/src/VoicePanel.tsx`: a few seconds of any voice (audio or video reference clip) reads arbitrary typed text in that voice; all inference is local, nothing leaves the machine.
- `POST /api/tts` background job (shared job registry/polling) with staged progress — decode reference, resolve reference transcript, load model, chunked synthesis, encode; artifacts under `tmp/tts/` with audio (`GET`/`HEAD /api/tts/{token}/audio`), waveform, and delete endpoints.
- Two clone modes: higher-fidelity ICL "transcript" mode when a reference transcript exists — a manually typed transcript wins, else the clip is auto-transcribed with WhisperX (toggleable, selectable model, default `small`) — falling back to x-vector-only "voice-signature" mode (speaker embedding alone, lower fidelity) with a warning when transcription is off, fails, or comes back empty. The result reports the clone mode and the exact transcript used.
- Reference clips are decoded to mono 24 kHz and trimmed to the first 30 s with a truncation warning; the clone prompt is built once per job and reused across chunks.
- Synthesis text up to 20,000 chars is split into ≤300-char chunks at sentence boundaries (over-long sentences hard-split on whitespace), rendered chunk by chunk with per-chunk progress, stitched with 0.12 s gaps, then peak-normalized once; both RNGs seeded for reproducibility.
- 10 languages (Chinese, English, German, Italian, Portuguese, Spanish, Japanese, Korean, French, Russian) plus Auto; model sizes `1.7b` (default) and `0.6b` (faster, downloads lazily on first use); output wav/flac/mp3/aac/opus (default FLAC).
- Device policy: cuda → mps → cpu — MPS IS auto-picked here (unlike Diamond, the transformer is compute-bound); bfloat16 on cuda but float32 on mps/cpu (fp16/bf16 cloning is broken on MPS); flash_attention_2 only on cuda when `flash_attn` imports, else sdpa; `TTS_DEVICE` env override; cached engine per (size, device) with CPU fallback that rebuilds the clone prompt and retries the chunk.
- Voice side-panel tab: reference dropzone, auto-transcribe toggle, editable reference transcript, text-to-speak with character count, language/format selects, Advanced settings (model size, WhisperX model), progress bar, result chips (rate, duration, format, device, clone mode), inline player, "Reference transcript used" disclosure, download, and discard.
- Install via `scripts/install-tts.sh` / `.ps1`: `qwen-tts` installed `--no-deps` so its transformers/gradio pins cannot fight the venv, then unpinned accelerate, then the ~4.5 GB 1.7B checkpoint into the `models/` HF cache.
