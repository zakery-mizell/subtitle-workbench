"""End-to-end demo of UniSE overlap separation.

Builds a synthetic two-speaker clip with a true overlap (macOS `say` voices),
then runs UniSE target-speaker extraction (tse) and removal (rtse) on the
overlap region and renders the spotlight/mute blends against the 48 kHz
original. Optionally verifies intelligibility by transcribing each stem.

Run from backend/:  ../.venv/bin/python -m tools.demo_unise [--skip-transcribe]
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf

from app.mastering.audio_io import decode_master
from app.separation import blend
from app.separation.unise_engine import UNISE_SAMPLE_RATE, load_engine

ROOT = Path(__file__).resolve().parents[2]
DEMO_DIR = ROOT / "tmp" / "unise-demo"
CLIP_PATH = DEMO_DIR / "demo48k.wav"
TRUTH_PATH = DEMO_DIR / "truth.json"
GAP_SECONDS = 0.4
REGION_PAD_SECONDS = 0.35

SCRIPTS = {
    "a_solo": "The quarterly report shows steady growth across every region, and the numbers from the coastal offices look especially strong this month.",
    "b_solo": "Meanwhile the engineering team finished the database migration over the weekend without any downtime for our customers.",
    "a_overlap": "I really think we should focus on the customer feedback before we commit to anything new this quarter.",
    "b_overlap": "The deadline for the next release is Friday afternoon and we still have twelve open tickets to close.",
}
VOICES = {"a": "Samantha", "b": "Daniel"}


def _say_to_wav(voice: str, text: str, out_path: Path) -> np.ndarray:
    aiff = out_path.with_suffix(".aiff")
    subprocess.run(["say", "-v", voice, "-o", str(aiff), text], check=True, capture_output=True)
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-i", str(aiff), "-ac", "1", "-ar", "48000", str(out_path)],
        check=True,
        capture_output=True,
    )
    aiff.unlink()
    audio, _ = sf.read(out_path, dtype="float32")
    return audio


def build_clip() -> dict:
    if sys.platform != "darwin":
        raise SystemExit("Clip synthesis uses macOS `say`; build the clip on the Mac or supply your own.")
    DEMO_DIR.mkdir(parents=True, exist_ok=True)

    parts = {name: _say_to_wav(VOICES[name[0]], text, DEMO_DIR / f"part_{name}.wav") for name, text in SCRIPTS.items()}
    sr = 48000
    gap = np.zeros(int(GAP_SECONDS * sr), dtype=np.float32)
    overlap_len = max(parts["a_overlap"].size, parts["b_overlap"].size)
    overlap = np.zeros(overlap_len, dtype=np.float32)
    overlap[: parts["a_overlap"].size] += parts["a_overlap"]
    overlap[: parts["b_overlap"].size] += parts["b_overlap"]
    overlap *= 0.72  # keep the summed voices out of clipping range

    pieces = [parts["a_solo"], gap, parts["b_solo"], gap, overlap, gap]
    mix = np.concatenate(pieces)

    cursor = 0.0
    marks: dict[str, list[float]] = {}
    for name, piece in (("a_solo", parts["a_solo"]), ("gap1", gap), ("b_solo", parts["b_solo"]), ("gap2", gap), ("overlap", overlap)):
        marks[name] = [cursor, cursor + piece.size / sr]
        cursor += piece.size / sr

    sf.write(CLIP_PATH, mix, sr)
    truth = {
        "sample_rate": sr,
        "a_solo": marks["a_solo"],
        "b_solo": marks["b_solo"],
        "overlap": marks["overlap"],
        "scripts": SCRIPTS,
        "voices": VOICES,
    }
    TRUTH_PATH.write_text(json.dumps(truth, indent=2))
    print(f"Built {CLIP_PATH.name}: {mix.size / sr:.1f}s, overlap {marks['overlap'][0]:.2f}-{marks['overlap'][1]:.2f}s")
    return truth


def decode_16k(path: Path) -> np.ndarray:
    command = [
        "ffmpeg", "-v", "error", "-i", str(path), "-vn", "-ac", "1",
        "-ar", str(UNISE_SAMPLE_RATE), "-f", "f32le", "-acodec", "pcm_f32le", "pipe:1",
    ]
    completed = subprocess.run(command, check=True, capture_output=True)
    return np.frombuffer(completed.stdout, dtype=np.float32).copy()


def seconds_slice(audio: np.ndarray, sr: int, start: float, end: float) -> np.ndarray:
    return audio[int(start * sr) : int(end * sr)]


def transcribe(path: Path) -> str:
    from app.whisperx_transcription import transcribe_with_whisperx

    result, _, _ = transcribe_with_whisperx(str(path), model_name="small", requested_language="en", hotwords=None)
    return " ".join(segment.get("text", "").strip() for segment in result.get("segments", [])).strip()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rebuild-clip", action="store_true", help="Force re-synthesizing the demo clip")
    parser.add_argument("--skip-transcribe", action="store_true", help="Skip WhisperX verification of the stems")
    args = parser.parse_args()

    if args.rebuild_clip or not (CLIP_PATH.is_file() and TRUTH_PATH.is_file()):
        truth = build_clip()
    else:
        truth = json.loads(TRUTH_PATH.read_text())

    mix16 = decode_16k(CLIP_PATH)
    region_start, region_end = truth["overlap"]
    window_start = max(0.0, region_start - REGION_PAD_SECONDS)
    window_end = min(mix16.size / UNISE_SAMPLE_RATE, region_end + REGION_PAD_SECONDS)
    overlap16 = seconds_slice(mix16, UNISE_SAMPLE_RATE, window_start, window_end)

    enrollments = {
        "a": seconds_slice(mix16, UNISE_SAMPLE_RATE, *truth["a_solo"]),
        "b": seconds_slice(mix16, UNISE_SAMPLE_RATE, *truth["b_solo"]),
    }

    print("Loading UniSE engine...")
    t0 = time.perf_counter()
    engine = load_engine()
    print(f"Engine ready on {engine.device.upper()} in {time.perf_counter() - t0:.1f}s")

    stems: dict[str, np.ndarray] = {}
    for speaker in ("a", "b"):
        enroll = engine.prepare_enrollment(enrollments[speaker])
        for task in ("tse", "rtse"):
            label = f"{task}_{speaker}"
            t0 = time.perf_counter()
            stems[label] = engine.run_task(task, overlap16, enroll)
            elapsed = time.perf_counter() - t0
            out = DEMO_DIR / f"stem_{label}.wav"
            sf.write(out, stems[label], UNISE_SAMPLE_RATE)
            rtf = elapsed / (overlap16.size / UNISE_SAMPLE_RATE)
            print(f"  {label}: {elapsed:.1f}s for {overlap16.size / UNISE_SAMPLE_RATE:.1f}s audio (RTF {rtf:.1f}) -> {out.name}")

    original = decode_master(str(CLIP_PATH))
    renders = {
        "spotlight_a": ("tse_a", blend.apply_spotlight),
        "spotlight_b": ("tse_b", blend.apply_spotlight),
        "mute_a": ("rtse_a", blend.apply_replace),
        "mute_b": ("rtse_b", blend.apply_replace),
    }
    for name, (stem_label, renderer) in renders.items():
        samples = original.samples.copy()
        render = blend.RegionRender(
            start=window_start,
            stem=stems[stem_label],
            region_start=region_start,
            region_end=region_end,
        )
        renderer(samples, original.sample_rate, render, UNISE_SAMPLE_RATE)
        out = DEMO_DIR / f"render_{name}.wav"
        sf.write(out, samples.T, original.sample_rate)
        print(f"  rendered {out.name}")

    if args.skip_transcribe:
        print("Skipped transcription verification.")
        return

    print("\nTranscribing (WhisperX small) for intelligibility check...")
    mix_overlap_path = DEMO_DIR / "overlap_mix.wav"
    sf.write(mix_overlap_path, overlap16, UNISE_SAMPLE_RATE)
    report = {
        "overlap mix (baseline)": mix_overlap_path,
        "tse_a (should be Samantha's line)": DEMO_DIR / "stem_tse_a.wav",
        "tse_b (should be Daniel's line)": DEMO_DIR / "stem_tse_b.wav",
        "rtse_a (should be Daniel's line)": DEMO_DIR / "stem_rtse_a.wav",
        "rtse_b (should be Samantha's line)": DEMO_DIR / "stem_rtse_b.wav",
    }
    print(f"\nGround truth A: {SCRIPTS['a_overlap']}")
    print(f"Ground truth B: {SCRIPTS['b_overlap']}\n")
    for label, path in report.items():
        print(f"[{label}]\n  {transcribe(path) or '(nothing recognized)'}\n")


if __name__ == "__main__":
    main()
