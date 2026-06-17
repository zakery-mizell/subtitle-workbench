# Waveform Timing Update - 2026-05-28

## Summary

This update added a waveform-assisted subtitle timing workflow to Subtitle Workbench. The goal is to use visual/audio energy cues to tighten subtitle start and end times around actual spoken audio, while also exposing likely speaker switches and overlap points.

## What Changed

- Added `POST /api/analyze-waveform`.
- Added FFmpeg-based audio decoding for any uploaded audio/video file.
- Added waveform frame generation for display.
- Added adaptive speech-span detection from RMS energy.
- Added a canvas waveform timeline under the audio player.
- Added subtitle timing blocks on the waveform.
- Added detected speech regions on the waveform.
- Added a playhead and click-to-seek behavior on the waveform.
- Added speaker switch, tight handoff, and possible overlap markers.
- Added `Analyze waveform` and `Snap subtitle edges` controls.
- Added conservative subtitle edge snapping:
  - start edges snap near detected speech onsets
  - end edges snap near detected speech offsets
  - small start/end padding is applied
  - caption overlap is prevented
  - moves are bounded so a bad waveform detection does not rewrite timing too aggressively
- Added backend unit tests for speech span detection and waveform display downsampling.
- Updated README and implemented-feature inventory.

## Files Added

- `backend/app/waveform_analysis.py`
- `backend/tests/test_waveform_analysis.py`
- `WAVEFORM_TIMING_UPDATE.md`

## Main Files Updated

- `backend/app/main.py`
- `backend/app/schemas.py`
- `frontend/src/App.tsx`
- `frontend/src/types.ts`
- `frontend/src/styles.css`
- `README.md`
- `IMPLEMENTED_FEATURES.md`

## Verification

Passed:

- `.\.venv\Scripts\python.exe -m unittest discover backend/tests`
- `npm run build`
- Synthetic FFmpeg waveform analysis smoke test

Dev server note: the app build passes, but hidden background server startup from the Codex shell did not keep ports `8000` and `5173` bound. Run the existing scripts manually if needed:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run-backend.ps1
powershell -ExecutionPolicy Bypass -File .\scripts\run-frontend.ps1
```

## WhisperX Status

Current project pin: `whisperx==3.8.1`.

Latest checked release: `v3.8.6`, released May 25, 2026.

Is it much better than the previous version? Not generally for raw transcription accuracy. WhisperX still wraps Whisper/faster-whisper for ASR, then adds VAD, forced alignment, and diarization. Recent WhisperX releases are mainly timing/alignment reliability and maintenance updates.

The updates are still relevant for this project because we depend heavily on word timings:

- `v3.8.6`: adds an Indonesian alignment model, fixes `ignore` interpolation handling in `interpolate_nans`, bumps `nltk`, and hardens CI.
- `v3.8.5`: pins `torchvision` and `torchcodec` for Torch 2.8 compatibility.
- `v3.8.4`: adds progress callbacks, fixes a file-handle leak, restores word-level timestamps for unalignable characters, and requires `faster-whisper>=1.2.0`.
- `v3.8.2`: exposed `avg_logprob` and reverted a wildcard alignment change that broke word-level timestamps.

Practical read: upgrading from `3.8.1` to `3.8.6` is probably worth testing for timestamp reliability, especially around numbers, symbols, foreign script, missing/NaN alignment values, and dependency compatibility. It should not be expected to produce a major transcription-text accuracy jump by itself.

Source: https://github.com/m-bain/whisperX/releases

## OpenAI GPT-4o Transcription Status

OpenAI currently documents:

- `gpt-4o-transcribe`
- `gpt-4o-mini-transcribe`
- `gpt-4o-transcribe-diarize`

`gpt-4o-transcribe-diarize` supports `diarized_json`, which returns speaker-aware segments with `speaker`, `start`, and `end` metadata. OpenAI also documents that this diarize model is available through `/v1/audio/transcriptions` only and is not supported in the Realtime API.

Local availability: no. These GPT-4o transcription models are OpenAI API models, not downloadable/local Whisper-style models. For local/offline transcription, this project should continue using WhisperX, Whisper/faster-whisper, pyannote, or other local forced-alignment/diarization tools.

Sources:

- https://developers.openai.com/api/docs/guides/speech-to-text
- https://developers.openai.com/api/docs/models/gpt-4o-transcribe-diarize

