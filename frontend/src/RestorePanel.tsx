import { useEffect, useRef, useState } from "react";

import {
  defaultRestoreParams,
  deleteRestore,
  fetchRestoreJob,
  restoreAudioUrl,
  startRestoreJob,
} from "./lib/restore";
import type { RestoreJobStatus, RestoreOutputFormat, RestoreParams, RestoreResult } from "./lib/restore";

const JOB_POLL_MS = 1000;

interface RestorePanelProps {
  apiBaseUrl: string;
  // The audio already loaded in the workspace, offered as a convenient default.
  audioFile: File | null;
}

function formatDuration(seconds: number): string {
  const minutes = Math.floor(seconds / 60);
  const rest = Math.round(seconds - minutes * 60);
  return `${minutes}:${String(rest).padStart(2, "0")}`;
}

export function RestorePanel({ apiBaseUrl, audioFile }: RestorePanelProps) {
  const [file, setFile] = useState<File | null>(audioFile);
  const [params, setParams] = useState<RestoreParams>(() => defaultRestoreParams());
  const [dragActive, setDragActive] = useState(false);
  const [jobId, setJobId] = useState<string | null>(null);
  const [job, setJob] = useState<RestoreJobStatus | null>(null);
  const [result, setResult] = useState<RestoreResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const pollRef = useRef<number | null>(null);

  const running = job !== null && (job.status === "queued" || job.status === "running");

  useEffect(() => {
    if (!jobId) {
      return;
    }
    let cancelled = false;

    async function poll() {
      try {
        const status = await fetchRestoreJob(apiBaseUrl, jobId!);
        if (cancelled) {
          return;
        }
        setJob(status);
        if (status.status === "done" && status.result) {
          setResult(status.result);
          setJobId(null);
        } else if (status.status === "error") {
          setError(status.error ?? "Restore failed.");
          setJobId(null);
        }
      } catch (pollError) {
        if (!cancelled) {
          setError(pollError instanceof Error ? pollError.message : "Lost contact with the restore job.");
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

  function update(mutator: (draft: RestoreParams) => void) {
    setParams((current) => {
      const draft = { ...current };
      mutator(draft);
      return draft;
    });
  }

  async function handleRun() {
    if (!file) {
      return;
    }
    setError(null);
    setResult(null);
    setJob(null);
    try {
      const id = await startRestoreJob(apiBaseUrl, file, params);
      setJobId(id);
      setJob({ id, status: "queued", stage: "queued", progress: 0, message: null, error: null, result: null });
    } catch (startError) {
      setError(startError instanceof Error ? startError.message : "Could not start restore.");
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
    await deleteRestore(apiBaseUrl, token);
  }

  return (
    <section className="selection-panel mastering-panel restore-panel">
      <div className="panel-section-heading">
        <p className="eyebrow">Restore</p>
        <h3>Diamond speech restoration</h3>
      </div>
      <p className="helper-text">
        Regenerates degraded, muffled, or low-bitrate speech as studio-quality 44.1 kHz audio. No transcription or
        splitting — just pick a file and run.
      </p>
      <p className="helper-text">
        Diamond regenerates speech (English-trained) — roughly 10× slower than real-time on this Mac.
      </p>

      <label
        className={`dropzone ${dragActive ? "is-dragging" : ""}`}
        onDragOver={(event) => {
          event.preventDefault();
          setDragActive(true);
        }}
        onDragLeave={() => setDragActive(false)}
        onDrop={(event) => {
          event.preventDefault();
          event.stopPropagation();
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
        <span>{file ? file.name : "Drag an audio file here or click to choose one."}</span>
      </label>

      <label>
        Output format
        <select
          value={params.output_format}
          onChange={(event) => update((draft) => void (draft.output_format = event.target.value as RestoreOutputFormat))}
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
            Chunk length (s)
            <input
              type="number"
              step={0.1}
              min={1}
              max={10}
              value={params.chunk_sec}
              onChange={(event) => update((draft) => void (draft.chunk_sec = Number(event.target.value)))}
            />
          </label>
          <label>
            Overlap (s)
            <input
              type="number"
              step={0.1}
              min={0}
              max={2}
              value={params.overlap_sec}
              onChange={(event) => update((draft) => void (draft.overlap_sec = Number(event.target.value)))}
            />
          </label>
          <label>
            Repetition penalty
            <input
              type="number"
              step={0.05}
              min={1}
              max={2}
              value={params.rep_penalty}
              onChange={(event) => update((draft) => void (draft.rep_penalty = Number(event.target.value)))}
            />
          </label>
        </div>
      </details>

      <div className="inline-actions">
        <button disabled={!file || running} onClick={() => void handleRun()}>
          {running ? "Restoring..." : "Run restore"}
        </button>
      </div>
      {!file ? <p className="helper-text">Choose an audio file first (drag one in or click to browse).</p> : null}

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
            <h3>Restored audio</h3>
          </div>
          <div className="chip-row">
            <span className="metric-chip">{(result.sample_rate / 1000).toFixed(1)} kHz</span>
            <span className="metric-chip">{formatDuration(result.duration_sec)}</span>
            <span className="metric-chip">{result.output_format.toUpperCase()}</span>
          </div>
          <audio className="restore-player" controls src={restoreAudioUrl(apiBaseUrl, result.token)} preload="auto" />
          {result.warnings.map((warning, index) => (
            <p key={`${index}-${warning}`} className="status-text">
              {warning}
            </p>
          ))}
          <div className="inline-actions">
            <a className="button-link" href={restoreAudioUrl(apiBaseUrl, result.token)} download={result.filename}>
              Download restored audio
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

export default RestorePanel;
