import { useEffect, useRef, useState } from "react";

import {
  conversionAudioUrl,
  defaultConversionParams,
  deleteConversion,
  fetchConversionJob,
  startConversionJob,
} from "./lib/conversion";
import type {
  ConversionJobStatus,
  ConversionOutputFormat,
  ConversionParams,
  ConversionResult,
  ConversionSource,
} from "./lib/conversion";

const JOB_POLL_MS = 1000;

interface IsolatedTrack {
  speakerId: number;
  name: string;
  token: string;
}

interface ConvertPanelProps {
  apiBaseUrl: string;
  // The audio already loaded in the workspace, offered as the source default.
  audioFile: File | null;
  // Rendered per-speaker tracks, offered as the one-click source. Full-length and
  // timeline-preserving, so the conversion of one is a drop-in replacement voice.
  isolatedTracks?: IsolatedTrack[];
  onConvertedVoice?: (speakerId: number, result: { token: string; url: string; filename: string }) => void;
  // Discarding deletes the artifact server-side, so anything registered via
  // onConvertedVoice must be unregistered too or it points at a 404.
  onConvertedVoiceRemoved?: (speakerId: number) => void;
}

function formatDuration(seconds: number): string {
  const minutes = Math.floor(seconds / 60);
  const rest = Math.round(seconds - minutes * 60);
  return `${minutes}:${String(rest).padStart(2, "0")}`;
}

export function ConvertPanel({ apiBaseUrl, audioFile, isolatedTracks, onConvertedVoice, onConvertedVoiceRemoved }: ConvertPanelProps) {
  const [sourceFile, setSourceFile] = useState<File | null>(audioFile);
  const [referenceFile, setReferenceFile] = useState<File | null>(null);
  const [params, setParams] = useState<ConversionParams>(() => defaultConversionParams());
  const [sourceDragActive, setSourceDragActive] = useState(false);
  const [referenceDragActive, setReferenceDragActive] = useState(false);
  const [jobId, setJobId] = useState<string | null>(null);
  const [job, setJob] = useState<ConversionJobStatus | null>(null);
  const [result, setResult] = useState<ConversionResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const pollRef = useRef<number | null>(null);
  // Null until the user picks a side, so tracks that finish rendering minutes
  // after this panel opened still become the default without overriding a
  // deliberate "Upload a file…".
  const [sourceMode, setSourceMode] = useState<"isolated" | "upload" | null>(null);
  const [selectedSpeakerId, setSelectedSpeakerId] = useState<number | null>(null);
  // Which speaker the running job converts, so the finished artifact can be
  // reported upward as that speaker's alternate voice.
  const jobSpeakerRef = useRef<number | null>(null);
  // Which speaker the displayed result was reported for, so a discard can
  // unregister exactly that alternate voice.
  const resultSpeakerRef = useRef<number | null>(null);
  const onConvertedVoiceRef = useRef(onConvertedVoice);
  onConvertedVoiceRef.current = onConvertedVoice;

  const tracks = isolatedTracks ?? [];
  const mode = sourceMode ?? (tracks.length ? "isolated" : "upload");
  const selectedTrack = tracks.find((track) => track.speakerId === selectedSpeakerId) ?? tracks[0] ?? null;
  const sourceReady = mode === "isolated" ? selectedTrack !== null : sourceFile !== null;

  const running = job !== null && (job.status === "queued" || job.status === "running");

  useEffect(() => {
    if (!jobId) {
      return;
    }
    let cancelled = false;

    async function poll() {
      try {
        const status = await fetchConversionJob(apiBaseUrl, jobId!);
        if (cancelled) {
          return;
        }
        setJob(status);
        if (status.status === "done" && status.result) {
          setResult(status.result);
          setJobId(null);
          const speakerId = jobSpeakerRef.current;
          jobSpeakerRef.current = null;
          resultSpeakerRef.current = speakerId;
          if (speakerId !== null) {
            onConvertedVoiceRef.current?.(speakerId, {
              token: status.result.token,
              url: conversionAudioUrl(apiBaseUrl, status.result.token),
              filename: status.result.filename,
            });
          }
        } else if (status.status === "error") {
          setError(status.error ?? "Voice conversion failed.");
          setJobId(null);
        }
      } catch (pollError) {
        if (!cancelled) {
          setError(pollError instanceof Error ? pollError.message : "Lost contact with the conversion job.");
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

  function update(mutator: (draft: ConversionParams) => void) {
    setParams((current) => {
      const draft = { ...current };
      mutator(draft);
      return draft;
    });
  }

  async function handleRun() {
    const source: ConversionSource | null =
      mode === "isolated"
        ? selectedTrack
          ? { token: selectedTrack.token, label: selectedTrack.name }
          : null
        : sourceFile;
    if (!source || !referenceFile) {
      return;
    }
    setError(null);
    setResult(null);
    setJob(null);
    resultSpeakerRef.current = null;
    jobSpeakerRef.current = mode === "isolated" && selectedTrack ? selectedTrack.speakerId : null;
    try {
      const id = await startConversionJob(apiBaseUrl, source, referenceFile, params);
      setJobId(id);
      setJob({ id, status: "queued", stage: "queued", progress: 0, message: null, error: null, result: null });
    } catch (startError) {
      setError(startError instanceof Error ? startError.message : "Could not start voice conversion.");
    }
  }

  async function handleDiscard() {
    if (!result) {
      return;
    }
    const token = result.token;
    const speakerId = resultSpeakerRef.current;
    resultSpeakerRef.current = null;
    setResult(null);
    setJob(null);
    setError(null);
    if (speakerId !== null) {
      onConvertedVoiceRemoved?.(speakerId);
    }
    await deleteConversion(apiBaseUrl, token);
  }

  return (
    <section className="selection-panel mastering-panel restore-panel">
      <div className="panel-section-heading">
        <p className="eyebrow">Convert</p>
        <h3>Seed-VC voice conversion</h3>
      </div>
      <p className="helper-text">
        Re-render a source performance — a bad phone recording, say — in the timbre of a clean reference clip of the
        target voice. The words, timing, and delivery stay; only the voice changes.
      </p>
      <p className="helper-text">
        Zero-shot and fully local — both clips stay on this machine. Speech only; 22.05 kHz output.
      </p>

      <div className="mode-toggle" role="group" aria-label="Conversion source">
        <button type="button" className={mode === "isolated" ? "is-active" : ""} onClick={() => setSourceMode("isolated")}>
          Isolated track
        </button>
        <button type="button" className={mode === "upload" ? "is-active" : ""} onClick={() => setSourceMode("upload")}>
          Upload a file…
        </button>
      </div>

      {mode === "isolated" ? (
        tracks.length ? (
          <>
            <label>
              Source voice
              <select
                value={selectedTrack ? String(selectedTrack.speakerId) : ""}
                onChange={(event) => setSelectedSpeakerId(Number(event.target.value))}
              >
                {tracks.map((track) => (
                  <option key={track.speakerId} value={track.speakerId}>
                    {track.name} — isolated track
                  </option>
                ))}
              </select>
            </label>
            <p className="helper-text">
              The result joins playback as that speaker's alternate voice — switch Original / Converted under the
              transport bar.
            </p>
          </>
        ) : (
          <p className="helper-text">Load audio with speakers to get isolated tracks, or upload a file.</p>
        )
      ) : (
        <label
          className={`dropzone ${sourceDragActive ? "is-dragging" : ""}`}
          onDragOver={(event) => {
            event.preventDefault();
            setSourceDragActive(true);
          }}
          onDragLeave={() => setSourceDragActive(false)}
          onDrop={(event) => {
            // preventDefault marks the drop as consumed; the window-level handler
            // still clears the global overlay but leaves the workspace file alone.
            event.preventDefault();
            setSourceDragActive(false);
            const dropped = event.dataTransfer.files?.[0];
            if (dropped) {
              setSourceFile(dropped);
            }
          }}
        >
          <input
            type="file"
            accept="audio/*,video/*"
            onChange={(event) => setSourceFile(event.target.files?.[0] ?? null)}
          />
          <span>{sourceFile ? sourceFile.name : "Source recording — drag it here or click to choose."}</span>
        </label>
      )}

      <label
        className={`dropzone ${referenceDragActive ? "is-dragging" : ""}`}
        onDragOver={(event) => {
          event.preventDefault();
          setReferenceDragActive(true);
        }}
        onDragLeave={() => setReferenceDragActive(false)}
        onDrop={(event) => {
          event.preventDefault();
          setReferenceDragActive(false);
          const dropped = event.dataTransfer.files?.[0];
          if (dropped) {
            setReferenceFile(dropped);
          }
        }}
      >
        <input
          type="file"
          accept="audio/*,video/*"
          onChange={(event) => setReferenceFile(event.target.files?.[0] ?? null)}
        />
        <span>{referenceFile ? referenceFile.name : "Target voice reference — drag it here or click to choose."}</span>
      </label>
      <p className="helper-text">Only the first 25 seconds of the reference are used — a clean clip of one voice works best.</p>

      <label className="toggle-row">
        <input
          type="checkbox"
          checked={params.convert_style}
          onChange={(event) => update((draft) => void (draft.convert_style = event.target.checked))}
        />
        Convert style/accent too (transfers accent and emotion, not just timbre)
      </label>

      <label>
        Output format
        <select
          value={params.output_format}
          onChange={(event) => update((draft) => void (draft.output_format = event.target.value as ConversionOutputFormat))}
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
            Diffusion steps
            <input
              type="number"
              step={1}
              min={10}
              max={100}
              value={params.diffusion_steps}
              onChange={(event) => update((draft) => void (draft.diffusion_steps = Number(event.target.value)))}
            />
          </label>
          <label>
            Length adjust
            <input
              type="number"
              step={0.05}
              min={0.5}
              max={2}
              value={params.length_adjust}
              onChange={(event) => update((draft) => void (draft.length_adjust = Number(event.target.value)))}
            />
          </label>
          <label>
            Similarity CFG
            <input
              type="number"
              step={0.05}
              min={0}
              max={1}
              value={params.similarity_cfg}
              onChange={(event) => update((draft) => void (draft.similarity_cfg = Number(event.target.value)))}
            />
          </label>
          <label>
            Intelligibility CFG
            <input
              type="number"
              step={0.05}
              min={0}
              max={1}
              value={params.intelligibility_cfg}
              onChange={(event) => update((draft) => void (draft.intelligibility_cfg = Number(event.target.value)))}
            />
          </label>
        </div>
      </details>

      <div className="inline-actions">
        <button disabled={!sourceReady || !referenceFile || running} onClick={() => void handleRun()}>
          {running ? "Converting..." : "Convert voice"}
        </button>
      </div>
      {!sourceReady || !referenceFile ? (
        <p className="helper-text">Choose both a source recording and a target voice reference first.</p>
      ) : null}

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
            <h3>Converted voice</h3>
          </div>
          <div className="chip-row">
            <span className="metric-chip">{(result.sample_rate / 1000).toFixed(1)} kHz</span>
            <span className="metric-chip">{formatDuration(result.duration_sec)}</span>
            <span className="metric-chip">{result.output_format.toUpperCase()}</span>
            <span className="metric-chip">{result.device_used}</span>
            <span className="metric-chip">{result.diffusion_steps} steps</span>
          </div>
          <audio className="restore-player" controls src={conversionAudioUrl(apiBaseUrl, result.token)} preload="auto" />
          {result.warnings.map((warning) => (
            <p key={warning.code} className="status-text">
              {warning.message}
            </p>
          ))}
          <div className="inline-actions">
            <a className="button-link" href={conversionAudioUrl(apiBaseUrl, result.token)} download={result.filename}>
              Download converted audio
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

export default ConvertPanel;
