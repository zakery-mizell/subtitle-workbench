import { useEffect, useRef, useState } from "react";

import {
  defaultTtsParams,
  deleteTts,
  fetchTtsJob,
  startTtsJob,
  ttsAudioUrl,
  TTS_LANGUAGES,
} from "./lib/tts";
import type { TtsJobStatus, TtsLanguage, TtsModelSize, TtsOutputFormat, TtsParams, TtsResult } from "./lib/tts";

const JOB_POLL_MS = 1000;

interface VoicePanelProps {
  apiBaseUrl: string;
}

function formatDuration(seconds: number): string {
  const minutes = Math.floor(seconds / 60);
  const rest = Math.round(seconds - minutes * 60);
  return `${minutes}:${String(rest).padStart(2, "0")}`;
}

export function VoicePanel({ apiBaseUrl }: VoicePanelProps) {
  const [file, setFile] = useState<File | null>(null);
  const [params, setParams] = useState<TtsParams>(() => defaultTtsParams());
  const [dragActive, setDragActive] = useState(false);
  const [jobId, setJobId] = useState<string | null>(null);
  const [job, setJob] = useState<TtsJobStatus | null>(null);
  const [result, setResult] = useState<TtsResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const pollRef = useRef<number | null>(null);

  const running = job !== null && (job.status === "queued" || job.status === "running");
  const refTranscript = params.ref_text ?? "";
  const hasRefTranscript = refTranscript.trim().length > 0;

  useEffect(() => {
    if (!jobId) {
      return;
    }
    let cancelled = false;

    async function poll() {
      try {
        const status = await fetchTtsJob(apiBaseUrl, jobId!);
        if (cancelled) {
          return;
        }
        setJob(status);
        if (status.status === "done" && status.result) {
          setResult(status.result);
          setJobId(null);
        } else if (status.status === "error") {
          setError(status.error ?? "Voice cloning failed.");
          setJobId(null);
        }
      } catch (pollError) {
        if (!cancelled) {
          setError(pollError instanceof Error ? pollError.message : "Lost contact with the voice job.");
          setJobId(null);
        }
      }
    }

    void poll();
    pollRef.current = window.setInterval(() => void poll(), JOB_POLL_MS);
    return () => {
      cancelled = true;
      if (pollRef.current !== null) {
        window.clearInterval(pollRef.current);
        pollRef.current = null;
      }
    };
  }, [apiBaseUrl, jobId]);

  function update(mutator: (draft: TtsParams) => void) {
    setParams((current) => {
      const draft = { ...current };
      mutator(draft);
      return draft;
    });
  }

  async function handleRun() {
    if (!file || !params.text.trim()) {
      return;
    }
    setError(null);
    setResult(null);
    setJob(null);
    // A typed transcript wins over auto-transcription; send null when it is blank.
    const payload: TtsParams = { ...params, ref_text: hasRefTranscript ? refTranscript : null };
    try {
      const id = await startTtsJob(apiBaseUrl, file, payload);
      setJobId(id);
      setJob({ id, status: "queued", stage: "queued", progress: 0, message: null, error: null, result: null });
    } catch (startError) {
      setError(startError instanceof Error ? startError.message : "Could not start voice cloning.");
    }
  }

  async function handleDiscard() {
    if (!result) {
      return;
    }
    const token = result.token;
    setResult(null);
    setJob(null);
    setError(null);
    await deleteTts(apiBaseUrl, token);
  }

  return (
    <section className="selection-panel mastering-panel restore-panel">
      <div className="panel-section-heading">
        <p className="eyebrow">Voice</p>
        <h3>Qwen3-TTS voice cloning</h3>
      </div>
      <p className="helper-text">
        Upload a short clip of any voice — a few seconds is enough — and Qwen3-TTS clones it locally to speak any text
        you type.
      </p>
      <p className="helper-text">
        Open weights, runs on this machine — the clip never leaves it. Cloning is sharpest when the clip's transcript is
        known.
      </p>

      <label
        className={`dropzone ${dragActive ? "is-dragging" : ""}`}
        onDragOver={(event) => {
          event.preventDefault();
          setDragActive(true);
        }}
        onDragLeave={() => setDragActive(false)}
        onDrop={(event) => {
          // preventDefault marks the drop as consumed; the window-level handler
          // still clears the global overlay but leaves the workspace file alone.
          event.preventDefault();
          setDragActive(false);
          const dropped = event.dataTransfer.files?.[0];
          if (dropped) {
            setFile(dropped);
          }
        }}
      >
        <input
          type="file"
          accept="audio/*,video/*"
          onChange={(event) => setFile(event.target.files?.[0] ?? null)}
        />
        <span>{file ? file.name : "Drag a reference clip here or click to choose one."}</span>
      </label>
      <p className="helper-text">Only the first 30 seconds are used — a clean 3–10 second clip of one voice works best.</p>

      <label className="toggle-row">
        <input
          type="checkbox"
          checked={params.auto_ref_text}
          onChange={(event) => update((draft) => void (draft.auto_ref_text = event.target.checked))}
        />
        Auto-transcribe reference (WhisperX)
      </label>
      <label>
        Reference transcript
        <textarea
          rows={2}
          value={refTranscript}
          placeholder="Type or paste what the clip says (optional)"
          onChange={(event) => update((draft) => void (draft.ref_text = event.target.value))}
        />
      </label>
      {!params.auto_ref_text && !hasRefTranscript ? (
        <p className="helper-text">
          With no transcript and auto-transcription off, the clone falls back to the voice signature only (lower
          fidelity).
        </p>
      ) : null}

      <label>
        Text to speak
        <textarea
          rows={3}
          value={params.text}
          placeholder="Type the text the cloned voice should say."
          onChange={(event) => update((draft) => void (draft.text = event.target.value))}
        />
      </label>
      {params.text.length > 0 ? <p className="helper-text">{params.text.length} characters</p> : null}

      <label>
        Language
        <select
          value={params.language}
          onChange={(event) => update((draft) => void (draft.language = event.target.value as TtsLanguage))}
        >
          {TTS_LANGUAGES.map((language) => (
            <option key={language} value={language}>
              {language}
            </option>
          ))}
        </select>
      </label>

      <label>
        Output format
        <select
          value={params.output_format}
          onChange={(event) => update((draft) => void (draft.output_format = event.target.value as TtsOutputFormat))}
        >
          <option value="flac">FLAC</option>
          <option value="wav">WAV</option>
          <option value="mp3">MP3</option>
          <option value="aac">AAC</option>
          <option value="opus">Opus</option>
        </select>
      </label>

      <details className="restore-advanced">
        <summary>Advanced settings</summary>
        <div className="mastering-grid-2">
          <label>
            Model quality
            <select
              value={params.model_size}
              onChange={(event) => update((draft) => void (draft.model_size = event.target.value as TtsModelSize))}
            >
              <option value="1.7b">Best quality (default)</option>
              <option value="0.6b">Faster, smaller</option>
            </select>
          </label>
          <label>
            Auto-transcribe model
            <select
              value={params.whisper_model}
              onChange={(event) => update((draft) => void (draft.whisper_model = event.target.value))}
            >
              <option value="base">base</option>
              <option value="small">small</option>
              <option value="medium">medium</option>
              <option value="large-v3">large-v3</option>
            </select>
          </label>
        </div>
      </details>

      <div className="inline-actions">
        <button disabled={!file || !params.text.trim() || running} onClick={() => void handleRun()}>
          {running ? "Cloning..." : "Clone voice"}
        </button>
      </div>
      {!file ? <p className="helper-text">Choose a reference clip first (drag one in or click to browse).</p> : null}

      {running && job ? (
        <div className="mastering-progress">
          <div className="mastering-progress-track">
            <div className="mastering-progress-fill" style={{ width: `${Math.round(job.progress * 100)}%` }} />
          </div>
          <p className="helper-text">
            {Math.round(job.progress * 100)}% – {job.message ?? job.stage}
          </p>
        </div>
      ) : null}

      {error ? <p className="status-text">{error}</p> : null}

      {result ? (
        <>
          <div className="panel-section-heading">
            <p className="eyebrow">Result</p>
            <h3>Cloned speech</h3>
          </div>
          <div className="chip-row">
            <span className="metric-chip">{(result.sample_rate / 1000).toFixed(1)} kHz</span>
            <span className="metric-chip">{formatDuration(result.duration_sec)}</span>
            <span className="metric-chip">{result.output_format.toUpperCase()}</span>
            <span className="metric-chip">{result.device_used}</span>
            <span className="metric-chip">
              {result.clone_mode === "voice-signature" ? "Voice signature (lower fidelity)" : "Transcript clone"}
            </span>
          </div>
          <audio className="restore-player" controls src={ttsAudioUrl(apiBaseUrl, result.token)} preload="auto" />
          {result.ref_text_used !== null ? (
            <details className="restore-advanced">
              <summary>Reference transcript used</summary>
              <p className="helper-text">{result.ref_text_used}</p>
            </details>
          ) : null}
          {result.warnings.map((warning) => (
            <p key={warning.code} className="status-text">
              {warning.message}
            </p>
          ))}
          <div className="inline-actions">
            <a className="button-link" href={ttsAudioUrl(apiBaseUrl, result.token)} download={result.filename}>
              Download cloned audio
            </a>
            <button type="button" onClick={() => void handleDiscard()}>
              Discard
            </button>
          </div>
        </>
      ) : null}
    </section>
  );
}

export default VoicePanel;
