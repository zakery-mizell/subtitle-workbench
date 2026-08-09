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
- Fixed hybrid speaker timing for multi-speaker sessions: word-level switch placement with segment-level context/fallback; no user-selectable mode.
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
- Persistent local voice profiles learned from a clean solo clip or isolated track; saved profiles map anonymous diarization clusters to requested speaker names by voice similarity instead of relying on first-appearance order.
- Fallback to single-speaker labeling when diarization is unavailable, not configured, or fails.
- Duration-based diarization cutoff for long files, with warning messaging instead of attempting an expensive diarization pass.
- One universal hybrid speaker assignment policy after diarization: word-level timing places precise switches; segment identity supplies fallback and repairs a ≤0.75 s stray prefix/suffix only when its speaker clearly dominates the rest of that aligned segment; the later raw targeted-window UniSE audit corrects strongly supported handoff mistakes but cannot fragment a segment whose current speaker already has a clear majority. No timing-mode control remains.
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
- `POST /api/convert` (Seed-VC voice conversion; two uploads), `GET`/`HEAD /api/convert/{token}/audio`, `GET /api/convert/{token}/waveform`, `DELETE /api/convert/{token}`
- `POST /api/speech-edit` (F5-TTS speech editing; single upload + `params_json`, edit or generate mode), `GET`/`HEAD /api/speech-edit/{token}/audio`, `GET /api/speech-edit/{token}/waveform`, `DELETE /api/speech-edit/{token}`
- `POST /api/transcribe` accepts wired-up request fields for:
  - audio upload
  - model
  - speaker count
  - speakers JSON
  - legacy speaker assignment field (accepted but ignored; hybrid is always used)
  - legacy language field (accepted but ignored; English is always used)
  - optional hotwords
  - optional disfluency cleanup
- `POST /api/retranscribe-range` accepts wired-up request fields for:
  - audio upload
  - model
  - start / end timestamps
  - legacy language field (accepted but ignored; English is always used)
  - optional hotwords
  - optional disfluency cleanup
- Structured API responses for:
  - words
  - paragraphs
  - captions
  - guide blocks
  - warnings
  - fixed language (`en`)
  - fixed speaker assignment policy (`hybrid`)
  - GPU usage flag

## 10. Runtime and Processing Details

- Windows Hugging Face symlink fallback patch for Whisper cache handling.
- Configurable Whisper cache directory.
- Configurable diarization cutoff.
- Configurable low-confidence threshold.
- Configurable silence threshold used for `SILENT` guide blocks.
- English-only WhisperX alignment/transcription across uploads, range retranscription, UniSE stems, voice references, and speech-edit windows.
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

## 13. Overlap Separation (main waveform, UniSE)

- Diarization now surfaces the raw (non-exclusive) pyannote annotation alongside the exclusive one; `/api/transcribe` returns `speaker_turns` and `overlap_regions` (spans where 2+ speakers talk at once, ≥0.4 s, merged across ≤0.4 s gaps).
- Overlap regions render as labeled clickable bands directly on the main waveform; the "N overlaps to untangle" chip zooms the waveform to the next overlap and selects it. The main waveform zooms with plain scroll (anchored at the cursor, trackpad horizontal / shift+drag pans, double-click or the Fit badge resets) with time ticks while zoomed.
- Per-region controls live in a dock under the waveform (opened by clicking a band or a chip in the dock's numbered overlap strip, which also shows each region's enabled state): Spotlight one voice (duck the original mix ~16 dB and overlay the AI-separated voice) or Mute one voice (replace the region with the everyone-but-X reconstruction), with a speaker picker per region; the Overlaps side-panel tab keeps only the global solo-track settings.
- Separation engine: vendored Alibaba QuarkAudio-UniSE (decoder-only AR-LM over BiCodec speech tokens, WavLM features, 16 kHz, 5 s windows) in `backend/app/separation/`; task modes `se`/`tse`/`rtse`; enrollment audio picked automatically from the target speaker's nearest clean solo span (≥1.5 s, ideally 5 s); MPS→CPU fallback; `transformers_compat.py` bridges the vendored code onto transformers ≥4.5x.
- `POST /api/separate` job (shared job registry/polling with mastering) blends stems back into the full-length original at its native sample rate with equal-power crossfades, RMS matching, and a region peak limiter; artifacts under `tmp/separation/` with audio/waveform/delete endpoints.
- Optional per-region WhisperX transcription of the spotlighted stem returns recovered words; one click adds them to the transcript as a new caption for that speaker (words in overlaps are otherwise usually lost — the mixed audio transcribes only the dominant voice).
- Result playback goes through the existing instant A/B toggle, relabeled Original/Separated; download button for the processed file.
- Install via `scripts/install-unise.sh` / `.ps1` (clones vendor repo, installs `backend/requirements-separation.txt`, downloads ~2.8 GB of checkpoints into `checkpoints/unise/`); `backend/tools/demo_unise.py` builds a synthetic two-voice overlap clip and verifies separation end to end.
- Unit tests for overlap/solo-span math, enrollment picking, and blend rendering (`backend/tests/test_separation.py`).
- Automatic solo tracks: every diarized multi-speaker transcription starts a background job (`POST /api/separate-solo`) that renders one full-length track per speaker. UniSE patches only merged overlap and rapid-handoff windows; the original audio remains elsewhere before the track is gated to that speaker. The "Only <speaker>" transport selector uses these tracks with no manual steps. Main "All speakers" playback never uses them, so a false-positive target window cannot degrade normal listening. Status chips: "Checking overlap and handoff voices…" / "Solo voices ready".
- Simultaneous-speech subtitles: captions of different speakers that share >50% of the shorter one's duration merge into a single broadcast-style dialogue cue on SRT export (one line per speaker, "- " dashes or "NAME: " prefixes when attribution is on), so exports never contain overlapping timecodes. In the editor those captions get an overlap badge with a left accent border, and playback highlights every caption under the playhead, not just the first. Auto solo tracks are encoded as FLAC to halve disk use.
- Targeted voice separation (`mode: "targeted"` on `POST /api/separate-solo`, the default for every diarized multi-speaker recording): rapid word-level speaker boundaries become 2.5-second candidate windows, overlap spans are added, and nearby/overlapping candidates are merged. UniSE TSE runs only for the relevant speakers inside those windows, using the nearest clean solo enrollment. The inference slice always carries at least UniSE's native five seconds of context so a short target does not wrap the outgoing voice into the model input; only the detected core is audited and patched. Each full-length playback track keeps the original audio elsewhere and is then gated to that speaker's speech regions. There is no full-recording UniSE option in the UI.
- Speaker-based transcription (`transcribe: true`, always sent by the frontend): each assembled, gated speaker track is transcribed on the shared timeline for per-speaker SRT/TXT export. In addition, every raw targeted UniSE window is transcribed before blending or gating when handoff auditing is active; these independent words are not erased by the diarization boundary being checked. Results persist across reloads with the track tokens.
- Automatic rapid-handoff audit (default for every diarized multi-speaker upload): raw targeted-window transcripts are compared only against mixture words within 1 second of a tight speaker transition. The service returns conservative corrections when the alternate voice extraction has strong time-aligned lexical evidence while the assigned extraction does not. The frontend applies confirmed word-speaker changes, re-runs the deterministic caption splitter for the connected boundary captions, preserves unrelated or concurrently edited subtitle text, records a warning, and adds one undo checkpoint. Short acknowledgements require a corroborating neighbouring correction; ambiguous/leaky evidence never changes attribution.
- Missed-handoff recovery for isolated replies: a pause-bounded aligned segment of up to three words and 1.25 seconds becomes a targeted candidate even when no adjacent word already carries a different speaker label. All enrolled speakers are included in its UniSE listening window, but attribution uses the original audio rather than generated stems: the cached pyannote Community-1 embedding model compares the reply (with 100 ms silent-side context) against each speaker's nearest clean five-second enrollment. A correction requires similarity ≥0.20, ≥0.10 over the runner-up, and ≥0.15 over the currently assigned speaker. Missing/ambiguous embeddings leave the label unchanged. This catches fully mislabeled one-word replies that ordinary boundary discovery cannot see.

## 14. Speech Restoration (Restore tab: Sidon + Diamond)

- Generative speech resynthesis with Diamond (`nineninesix/diamond-1.0`, a sequence-to-sequence RQ-Transformer) in `backend/app/restore/` and `frontend/src/RestorePanel.tsx`: regenerates clean 44.1 kHz speech from degraded input (heavy codec artifacts, clipping, muffling) instead of filtering it, so the output replaces the input rather than blending in. English-trained; content above ~8 kHz is generated, not recovered.
- Standalone restore: `POST /api/restore` background job (shared job registry/polling with mastering and separation) restores a whole uploaded audio/video file end to end — ffmpeg decode to mono 24 kHz, restore, encode to wav/flac/mp3/aac/opus (default FLAC); artifacts under `tmp/restore/` with audio (`GET`/`HEAD /api/restore/{token}/audio`), waveform, and delete endpoints.
- Long audio is restored in 30 s segments with 1 s crossfades applied at the 44.1 kHz output rate; both RNGs are seeded for reproducibility; one global peak normalize at the end, never boosting quiet audio.
- Tunable chunk length, overlap, and repetition penalty with validated ranges, exposed in the panel under Advanced settings.
- Device policy: cuda if available else cpu — MPS is never auto-picked (measured slower than CPU on Apple Silicon because the tiny autoregressive kernels are launch-bound); `RESTORE_DEVICE` env override; cached engine per device with CPU fallback when a GPU load or segment fails.
- Restore side-panel tab: own dropzone that defaults to the workspace audio, staged progress bar with "Restoring m:ss / m:ss" messages, result chips (sample rate, duration, format), inline player, download, and discard (which deletes the server artifact).
- Restored solo tracks: the "Restore voices" checkbox in the Overlaps panel sets `"restore": true` on the automatic solo-tracks job (with `"restore_engine"` choosing the model), running each assembled per-speaker track through the engine after overlap stitching; on failure the job keeps the unrestored track and adds a warning. Both the flag and the engine are part of the solo-tracks cache key so raw and restored renders — and the two engines' differing sample rates — never share tokens.
- Install via `scripts/install-restore.sh` / `.ps1`: an ordered mixed `--no-deps` pip sequence so descript-audiotools cannot downgrade protobuf and break onnxruntime, then the checkpoint download into the `models/` HF cache (also fetched lazily on first use); missing pieces surface as `RestoreUnavailable` errors that point at the install script.
- Sidon engine (`sarulab-speech/sidon-v0.1`, ICASSP 2026) in `backend/app/restore/sidon_engine.py`, the default: a LoRA-adapted w2v-BERT 2.0 feature predictor cleanses the SSL features of the degraded input and a decoder vocoder resynthesises 48 kHz audio. Multilingual and identity-preserving, so it also suits cleaning a voice-cloning reference before Convert/Patch. Measured RTF ~0.2 on an M1 Max CPU — about 50x cheaper than Diamond.
- Sidon runs the reference pipeline (peak-normalize to 0.9, 50 Hz high-pass, resample to 16 kHz, 1.5 s tail pad, SeamlessM4T log-mel front-end from `facebook/w2v-bert-2.0`), chunked with a one-frame feature carry-over across seams and a matching one-frame output trim, so each chunk after the first contributes exactly `chunk_sec * 48000` samples. Output length is pinned to `round(n / sr * 48000)` and the level is scaled back to the input's original peak, so a quiet recording stays quiet and the result stays aligned to the workspace timeline (measured envelope lag within 10 ms at both 2 and 6 chunks — seams do not accumulate drift). A trailing stub chunk is folded into its predecessor rather than fed to the front-end.
- Sidon context window (`sidon_chunk_sec`, 5-96 s, default 15) trades quality for memory: measured over a 150 s clip, 15 s needs ~6 GB peak RSS, 30 s ~12 GB, 96 s ~24 GB, with log-mel cosines of 0.989 / 0.993 / 1.000 against the 96 s render. The default is sized so a CUDA run fits a mid-range card's VRAM.
- Sidon device policy: cuda if available else cpu, `SIDON_DEVICE` override; the cuda/cpu checkpoint pair are `torch.jit.trace` exports with device-pinned constants, so the artifact matching the device is downloaded and loaded and modules are never moved between devices. MPS is rejected explicitly with a pointed error rather than failing obscurely (`map_location='mps'` hits a float64 constant; a CPU-loaded module raises "Passed CPU tensor to MPS op").
- Install Sidon via `scripts/install-sidon.sh` / `.ps1`: no new packages (torch, torchaudio, transformers and huggingface_hub all ship with the base install, so the script verifies them and stops with a pointed message if any are missing), then downloads only the ~1 GB device variant this machine will load plus the log-mel front-end config.

## 15. Voice Cloning (Voice tab, Qwen3-TTS)

- Local instant voice cloning with Qwen3-TTS Base (`Qwen/Qwen3-TTS-12Hz-1.7B-Base`, Apache-2.0) in `backend/app/tts/` and `frontend/src/VoicePanel.tsx`: a few seconds of any voice (audio or video reference clip) reads arbitrary typed text in that voice; all inference is local, nothing leaves the machine.
- `POST /api/tts` background job (shared job registry/polling) with staged progress — decode reference, resolve reference transcript, load model, chunked synthesis, encode; artifacts under `tmp/tts/` with audio (`GET`/`HEAD /api/tts/{token}/audio`), waveform, and delete endpoints.
- Two clone modes: higher-fidelity ICL "transcript" mode when a reference transcript exists — a manually typed transcript wins, else the clip is auto-transcribed with WhisperX (toggleable, selectable model, default `small`) — falling back to x-vector-only "voice-signature" mode (speaker embedding alone, lower fidelity) with a warning when transcription is off, fails, or comes back empty. The result reports the clone mode and the exact transcript used.
- Reference clips are decoded to mono 24 kHz and trimmed to the first 30 s with a truncation warning; the clone prompt is built once per job and reused across chunks.
- Synthesis text up to 20,000 chars is split into ≤300-char chunks at sentence boundaries (over-long sentences hard-split on whitespace), rendered chunk by chunk with per-chunk progress, stitched with 0.12 s gaps, then peak-normalized once; both RNGs seeded for reproducibility.
- English-only synthesis and reference transcription; model sizes `1.7b` (default) and `0.6b` (faster, downloads lazily on first use); output wav/flac/mp3/aac/opus (default FLAC).
- Device policy: cuda → mps → cpu — MPS IS auto-picked here (unlike Diamond, the transformer is compute-bound); bfloat16 on cuda but float32 on mps/cpu (fp16/bf16 cloning is broken on MPS); flash_attention_2 only on cuda when `flash_attn` imports, else sdpa; `TTS_DEVICE` env override; cached engine per (size, device) with CPU fallback that rebuilds the clone prompt and retries the chunk.
- Voice side-panel tab: reference dropzone, auto-transcribe toggle, editable reference transcript, text-to-speak with character count, format select, Advanced settings (model size, WhisperX model), progress bar, result chips (rate, duration, format, device, clone mode), inline player, "Reference transcript used" disclosure, download, and discard. Language is fixed to English with no selector.
- Install via `scripts/install-tts.sh` / `.ps1`: `qwen-tts` installed `--no-deps` so its transformers/gradio pins cannot fight the venv, then unpinned accelerate, then the ~4.5 GB 1.7B checkpoint into the `models/` HF cache.

## 16. Seed-VC voice conversion (Convert tab)

- Zero-shot voice conversion with Seed-VC V2 (`hubert-bsqvae-small`, "best in suppressing source speaker traits") in `backend/app/conversion/` and `frontend/src/ConvertPanel.tsx`: a source performance (e.g. a rough phone recording) is re-rendered in the timbre of a clean reference clip of the target speaker — words, timing, and delivery from the source, voice from the reference. Speech-only (no singing/f0 mode); output is 22.05 kHz mono; all inference is local.
- `POST /api/convert` background job (shared job registry/polling with mastering, separation, and restore) takes two uploads (`audio` source + `reference` voice) with staged progress — decode both, load model, streamed conversion, encode; artifacts under `tmp/conversion/` with audio (`GET`/`HEAD /api/convert/{token}/audio`), waveform, and delete endpoints.
- Both uploads are ffmpeg-decoded to mono 22.05 kHz wavs (they may be mp4/m4a containers librosa cannot read reliably) and the decoded paths are handed to the wrapper, then cleaned up in a finally. The reference is trimmed to the first 25 s by the model with a `convert_ref_truncated` warning; long sources are chunked internally (30 s windows, 5 s overlap, cosine crossfade) with no external splitting.
- Conversion drives the mandatory streaming generator (`convert_voice_with_streaming`, `stream_output=True`); both RNGs are seeded for reproducibility; progress is estimated from `ceil(source_duration / 25 s)` chunks and capped at 0.97 until the final full-audio yield arrives.
- Tunables: diffusion steps (10–100, default 50; upstream default 30, higher trades speed for quality), length adjust (0.5–2.0), similarity CFG and intelligibility CFG (0.0–1.0, default 0.7), and a "convert style/accent too" toggle (`convert_style`) that engages the AR model for accent/emotion transfer. AR sampling knobs (top_p 0.9, temperature 1.0, repetition penalty 1.0) stay internal. Output wav/flac/mp3/aac/opus (default FLAC).
- Device policy: cuda → mps → cpu — MPS IS auto-picked here (the diffusion/AR stack is compute-bound, unlike Diamond); float16 on cuda but float32 on mps/cpu (the CFM diffusion is forced fp32 internally regardless); `CONVERSION_DEVICE` env override; cached engine per device with CPU fallback. `HF_HOME` is hard-set and `hf_utils.load_custom_model_from_hf` is monkeypatched off its CWD-relative cache so every checkpoint honors the `models/` cache; the V1 `inference.py` module is never imported (it pins `HF_HUB_CACHE` at import time).
- Convert side-panel tab: separate source and target-voice dropzones (source defaults to the workspace audio), a "convert style/accent too" checkbox, output-format select, Advanced settings (diffusion steps, length adjust, similarity CFG, intelligibility CFG), progress bar, result chips (rate, duration, format, device, diffusion steps), inline player, download, and discard.
- Seed-VC model code is vendored into `vendor/seed-vc/` (not pip-installable, gitignored). Install via `scripts/install-convert.sh` / `.ps1`: clone the repo, install only the two missing deps (`hydra-core`, `munch`) — never the upstream `requirements.txt`, which pins torch 2.4 / transformers 4.46 / numpy 1.26 — then download the ~1.6 GB checkpoints (Seed-VC V2, ASTRAL quantizers, campplus, hubert, bigvgan, whisper-small tokenizer) into the `models/` HF cache; missing pieces surface as `ConversionUnavailable` errors that point at the install script.

### Known limitation: conversion is not audio upscaling, and it needs *machine*-intelligible input

Conversion replaces timbre; it does not repair a source. Its content path is entirely
audio-derived — HuBERT features quantized to ASTRAL tokens — so **the transcript cannot be fed
to it** and there is no text to fall back on when the audio is ambiguous. The failure mode that
matters in practice: a band-limited phone recording can be perfectly intelligible to a human
listener and still be too degraded for the content encoder, because the cues the encoder leans
on (fricative/sibilant energy above the phone band, crisp stop releases) were never recorded.
The output is then fluent but wrong — words smeared or replaced — and turning intelligibility
CFG up only sharpens articulation of tokens that were already wrong.

Human intelligibility and encoder intelligibility are therefore different thresholds, and
clearing the first does not clear the second. Restoration/denoise (§14) helps when the problem
is noise or reverb; it does not synthesize a missing top octave, so it does not move the needle
on a truly narrowband source. What does work is replacing the *performance* with a clean one
before converting — see the overdub route below — because that makes the phonetics real rather
than inferred.

Overdub-first pipeline (the reliable route for unusable sources), in increasing automation:
1. **Re-perform (ADR).** Speak the lines again against the original for timing and feeling. The
   re-performance carries emotion natively, and being already the target voice it makes
   conversion optional (use it only to polish timbre toward a reference).
2. **Synthesize per caption, timing locked.** Patch/generate each caption's text in the target
   voice with `fix_duration_s` set from the SRT so the original rhythm is preserved. Clean
   phonetics, original timing — but the emotional contour is the TTS model's invention, not the
   source's.
3. **Transfer the contour with DSP.** After (2), force the synthesized audio's F0 and energy
   envelopes onto the measured contours of the original (word-aligned via the WhisperX word
   timings the legacy load already produces, F0 via pyworld/torchcrepe, pitch rescaled into the
   target speaker's range). This is what recovers "the same delivery" without a human take, and
   it is buildable on what is already installed. Not implemented yet.

Also worth one cheap experiment before any of that: run a dedicated **bandwidth-extension**
model (audio super-resolution, e.g. AudioSR-class) over the source and re-run conversion. If the
content encoder's problem really is the missing high band, restoring plausible high-frequency
detail may lift the source over the encoder's threshold. Unproven here; the engines in this repo
do not do it.

## 17. F5-TTS speech editing (Patch tab)

- Two modes under one **Patch** tab, backed by F5-TTS (`F5TTS_v1_Base`, a mel-domain flow-matching TTS model) in `backend/app/speechedit/` and `frontend/src/PatchPanel.tsx`; both modes emit 24 kHz mono and run fully locally. `POST /api/speech-edit` is one upload plus `params_json` on the shared job registry/polling; artifacts land under `tmp/speechedit/` (token prefix `se_`, filename suffix `.f5tts.<fmt>`) with audio (`GET`/`HEAD /api/speech-edit/{token}/audio`), waveform, and delete endpoints.
- **Patch words** (the headline feature): upload a full recording, mark one or more time spans whose words are garbled/missing, and give the replacement text per span. F5-TTS regenerates ONLY those spans by mel-domain infilling conditioned on the surrounding audio, so a patch inherits the surrounding voice and prosody; audio outside the edited windows stays bit-identical. The single upload is the reference clip in generate mode and the full recording in edit mode.
- **Generate speech**: zero-shot voice-cloned TTS — upload a reference clip (plus transcript, auto-transcribed with WhisperX if blank) and synthesize arbitrary text in that voice. F5-TTS REQUIRES a reference transcript (no x-vector fallback like Qwen had), so a manual transcript wins and a failed/empty auto-transcription fails the job with a clear `patch_ref_transcript_failed` message; the reference is trimmed to 12 s with a warning so the transcript matches what the model sees.
- Windowing layer (edit mode): each edit span is patched inside a padded window — `WINDOW_PAD_SECONDS` 4 s each side, clamped to the audio bounds and capped at `MAX_WINDOW_SECONDS` 25 s total (pad shrinks symmetrically); spans over 20 s are rejected. The recording is transcribed once with WhisperX; per-window target text is built by replacing the words overlapping the edit span (straddling words count as inside) with the span's `new_text`, with a per-edit `window_text` override for when degraded audio will not transcribe (transcription failure without an override fails the job).
- Edits are validated (non-empty, sorted, non-overlapping, in-bounds, span ≤ 20 s) and processed back-to-front (descending start) so earlier splices never shift later spans' sample bounds; each window is spliced back with 0.15 s linear crossfades placed INSIDE the window so everything outside `[w0, w1]` stays bit-identical, and the length delta from an optional per-span `fix_duration_s` (re-timing a patch, ≤ 20 s) is tracked sample-accurately. The result reports per-edit `regions` (`start_s`, `end_s`, `window_start_s`, `window_end_s`, `text_used`) so the UI shows exactly what was regenerated.
- Mel-infill mechanics (mirrors `infer/speech_edit.py`): RMS-normalize quiet windows up to 0.1 (un-scaled on output), build a kept/zero mel with a matching edit mask via the pure `build_edit_arrays` frame plan (`hop_length` 256, `target_sample_rate` 24000, 100 mel channels, frame = round(sec · 24000 / 256)), feed `convert_char_to_pinyin([window_text])`, `ema_model.sample(..., edit_mask=...)` then `vocoder.decode` under `torch.inference_mode()`. Generate mode calls `F5TTS.infer` (target_rms 0.1, cross_fade 0.15, sway −1, cfg 2). NFE steps (8–64, default 32), speed (0.5–2.0), and an optional seed are exposed.
- Windowing/mask/splice math (validation, window computation, word-replacement text, frame math, crossfade splice) is kept in PURE functions so `backend/tests/test_speechedit.py` covers it without loading a model, alongside params validation and mocked-engine endpoint tests for both modes.
- Device policy: cuda → mps → cpu — MPS IS auto-picked here (compute-bound transformer, unlike Diamond); F5-TTS enforces fp32 off-cuda itself (MPS-safe), so no dtype is set; `SPEECHEDIT_DEVICE` env override; cached engine per device with CPU fallback. `HF_HOME` is hard-set to `models/` (not setdefault — the whisper transcription pass repoints it process-wide) alongside the explicit `hf_cache_dir` F5TTS honors.
- Patch side-panel tab: mode toggle (Patch words / Generate speech), file dropzone (edit mode prefilled from the workspace audio), an editable list of edit rows (start, end, replacement text, plus a per-row Advanced disclosure for target duration and window-text override, with add/remove-span buttons), generate-mode reference/transcript/text fields and a speed slider, an Advanced section (NFE steps, seed, transcription model), output-format select, progress bar, result chips (rate, duration, format, device), inline player, a "Patched N spans" list rendering `regions`, download, and discard. Transcription and generation are fixed to English.
- Install via `scripts/install-speechedit.sh` / `.ps1`: `f5-tts` installed `--no-deps` (its gradio/bitsandbytes pins would collide with the app), then the runtime deps traced through real imports (`cached_path hydra-core vocos torchdiffeq pypinyin rjieba tomli ema_pytorch wandb datasets` — the last three are import-time-only landmines that `f5_tts/model/__init__.py` pulls in via the Trainer/dataset modules), then the ~1.35 GB `F5TTS_v1_Base` checkpoint and the ~54 MB `charactr/vocos-mel-24khz` vocoder into the `models/` HF cache. `pip check` flags f5-tts's own skipped pins as expected noise (the script warns rather than fails).
