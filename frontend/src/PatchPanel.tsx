import { useEffect, useRef, useState } from "react";

import {
  defaultPatchEdit,
  defaultSpeechEditParams,
  deleteSpeechEdit,
  fetchSpeechEditJob,
  speechEditAudioUrl,
  startSpeechEditJob,
} from "./lib/speechedit";
import type {
  PatchEdit,
  SpeechEditJobStatus,
  SpeechEditMode,
  SpeechEditOutputFormat,
  SpeechEditParams,
  SpeechEditResult,
} from "./lib/speechedit";

const JOB_POLL_MS = 1000;

interface PatchPanelProps {
  apiBaseUrl: string;
  // The audio already loaded in the workspace, offered as the edit-mode default.
  audioFile: File | null;
}

function formatDuration(seconds: number): string {
  const minutes = Math.floor(seconds / 60);
  const rest = Math.round(seconds - minutes * 60);
  return `${minutes}:${String(rest).padStart(2, "0")}`;
}

export function PatchPanel({ apiBaseUrl, audioFile }: PatchPanelProps) {
  const [file, setFile] = useState<File | null>(audioFile);
  const [params, setParams] = useState<SpeechEditParams>(() => defaultSpeechEditParams());
  const [dragActive, setDragActive] = useState(false);
  const [jobId, setJobId] = useState<string | null>(null);
  const [job, setJob] = useState<SpeechEditJobStatus | null>(null);
  const [result, setResult] = useState<SpeechEditResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const pollRef = useRef<number | null>(null);

  const running = job !== null && (job.status === "queued" || job.status === "running");
  const isEdit = params.mode === "edit";

  useEffect(() => {
    if (!jobId) {
      return;
    }
    let cancelled = false;

    async function poll() {
      try {
        const status = await fetchSpeechEditJob(apiBaseUrl, jobId!);
        if (cancelled) {
          return;
        }
        setJob(status);
        if (status.status === "done" && status.result) {
          setResult(status.result);
          setJobId(null);
        } else if (status.status === "error") {
          setError(status.error ?? "Speech editing failed.");
          setJobId(null);
        }
      } catch (pollError) {
        if (!cancelled) {
          setError(pollError instanceof Error ? pollError.message : "Lost contact with the patch job.");
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

  function update(mutator: (draft: SpeechEditParams) => void) {
    setParams((current) => {
      const draft = { ...current, edits: current.edits.map((edit) => ({ ...edit })) };
      mutator(draft);
      return draft;
    });
  }

  function updateEdit(index: number, mutator: (draft: PatchEdit) => void) {
    update((draft) => {
      const edit = draft.edits[index];
      if (edit) {
        mutator(edit);
      }
    });
  }

  const genTextReady = params.gen_text.trim().length > 0;
  const editsReady = params.edits.some((edit) => edit.end_s > edit.start_s && edit.new_text.trim().length > 0);
  const canRun = Boolean(file) && (isEdit ? editsReady : genTextReady) && !running;

  async function handleRun() {
    if (!file) {
      return;
    }
    setError(null);
    setResult(null);
    setJob(null);
    // Normalize blank optional fields to null so the backend sees real omissions.
    const payload: SpeechEditParams = {
      ...params,
      edits: params.edits.map((edit) => ({
        ...edit,
        window_text: edit.window_text && edit.window_text.trim() ? edit.window_text : null,
        fix_duration_s: edit.fix_duration_s && edit.fix_duration_s > 0 ? edit.fix_duration_s : null,
      })),
    };
    try {
      const id = await startSpeechEditJob(apiBaseUrl, file, payload);
      setJobId(id);
      setJob({ id, status: "queued", stage: "queued", progress: 0, message: null, error: null, result: null });
    } catch (startError) {
      setError(startError instanceof Error ? startError.message : "Could not start speech editing.");
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
    await deleteSpeechEdit(apiBaseUrl, token);
  }

  return (
    <section className="selection-panel mastering-panel restore-panel">
      <div className="panel-section-heading">
        <p className="eyebrow">Patch</p>
        <h3>F5-TTS speech editing</h3>
      </div>
      <p className="helper-text">
        Patch mode regenerates muffled or misspoken words in place — mark the spans, type what should have been said,
        and F5-TTS re-synthesizes only those windows, inheriting the surrounding voice. Everything outside the edits
        stays untouched.
      </p>
      <p className="helper-text">Open weights, fully local — the recording never leaves this machine. 24 kHz mono output.</p>

      <div className="mode-toggle">
        <button
          type="button"
          className={isEdit ? "is-active" : ""}
          onClick={() => update((draft) => void (draft.mode = "edit"))}
        >
          Patch words
        </button>
        <button
          type="button"
          className={!isEdit ? "is-active" : ""}
          onClick={() => update((draft) => void (draft.mode = "generate"))}
        >
          Generate speech
        </button>
      </div>

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
        <span>
          {file
            ? file.name
            : isEdit
              ? "Full recording to patch — drag it here or click to choose."
              : "Reference clip — drag it here or click to choose."}
        </span>
      </label>

      {isEdit ? (
        <>
          <div className="panel-section-heading">
            <p className="eyebrow">Edits</p>
            <h3>Spans to patch</h3>
          </div>
          {params.edits.map((edit, index) => (
            <div key={index} className="patch-edit-row">
              <div className="mastering-grid-2">
                <label>
                  Start (s)
                  <input
                    type="number"
                    step={0.1}
                    min={0}
                    value={edit.start_s}
                    onChange={(event) => updateEdit(index, (draft) => void (draft.start_s = Number(event.target.value)))}
                  />
                </label>
                <label>
                  End (s)
                  <input
                    type="number"
                    step={0.1}
                    min={0}
                    value={edit.end_s}
                    onChange={(event) => updateEdit(index, (draft) => void (draft.end_s = Number(event.target.value)))}
                  />
                </label>
              </div>
              <label>
                Replacement text
                <textarea
                  rows={2}
                  value={edit.new_text}
                  placeholder="What should have been said in this span"
                  onChange={(event) => updateEdit(index, (draft) => void (draft.new_text = event.target.value))}
                />
              </label>
              <details className="restore-advanced">
                <summary>Advanced (this span)</summary>
                <label>
                  Target duration (s, optional)
                  <input
                    type="number"
                    step={0.1}
                    min={0}
                    value={edit.fix_duration_s ?? ""}
                    placeholder="Match span length"
                    onChange={(event) =>
                      updateEdit(index, (draft) => {
                        const value = event.target.value;
                        draft.fix_duration_s = value === "" ? null : Number(value);
                      })
                    }
                  />
                </label>
                <label>
                  Window text override (optional)
                  <textarea
                    rows={2}
                    value={edit.window_text ?? ""}
                    placeholder="Full target transcript for the whole window, if auto-transcription fails"
                    onChange={(event) =>
                      updateEdit(index, (draft) => {
                        const value = event.target.value;
                        draft.window_text = value === "" ? null : value;
                      })
                    }
                  />
                </label>
              </details>
              {params.edits.length > 1 ? (
                <div className="inline-actions">
                  <button
                    type="button"
                    onClick={() => update((draft) => void draft.edits.splice(index, 1))}
                  >
                    Remove span
                  </button>
                </div>
              ) : null}
            </div>
          ))}
          <div className="inline-actions">
            <button type="button" onClick={() => update((draft) => void draft.edits.push(defaultPatchEdit()))}>
              Add span
            </button>
          </div>
        </>
      ) : (
        <>
          <p className="helper-text">Only the first 12 seconds of the reference are used — a clean clip of one voice works best.</p>
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
              value={params.ref_text}
              placeholder="Type or paste what the clip says (required — F5-TTS needs it)"
              onChange={(event) => update((draft) => void (draft.ref_text = event.target.value))}
            />
          </label>
          <label>
            Text to speak
            <textarea
              rows={3}
              value={params.gen_text}
              placeholder="Type the text the cloned voice should say."
              onChange={(event) => update((draft) => void (draft.gen_text = event.target.value))}
            />
          </label>
          <label>
            Speed
            <input
              type="range"
              min={0.5}
              max={2}
              step={0.05}
              value={params.speed}
              onChange={(event) => update((draft) => void (draft.speed = Number(event.target.value)))}
            />
            <span className="helper-text">{params.speed.toFixed(2)}×</span>
          </label>
        </>
      )}

      <label>
        Output format
        <select
          value={params.output_format}
          onChange={(event) => update((draft) => void (draft.output_format = event.target.value as SpeechEditOutputFormat))}
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
            Quality (NFE steps)
            <input
              type="number"
              step={1}
              min={8}
              max={64}
              value={params.nfe_step}
              onChange={(event) => update((draft) => void (draft.nfe_step = Number(event.target.value)))}
            />
          </label>
          <label>
            Seed (optional)
            <input
              type="number"
              step={1}
              value={params.seed ?? ""}
              placeholder="Random"
              onChange={(event) =>
                update((draft) => {
                  const value = event.target.value;
                  draft.seed = value === "" ? null : Number(value);
                })
              }
            />
          </label>
          <label>
            Transcription model
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
        <button disabled={!canRun} onClick={() => void handleRun()}>
          {running ? "Patching..." : isEdit ? "Patch audio" : "Generate speech"}
        </button>
      </div>
      {!file ? (
        <p className="helper-text">
          Choose {isEdit ? "the recording to patch" : "a reference clip"} first (drag one in or click to browse).
        </p>
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
            <h3>{result.mode === "edit" ? "Patched audio" : "Generated speech"}</h3>
          </div>
          <div className="chip-row">
            <span className="metric-chip">{(result.sample_rate / 1000).toFixed(1)} kHz</span>
            <span className="metric-chip">{formatDuration(result.duration_sec)}</span>
            <span className="metric-chip">{result.output_format.toUpperCase()}</span>
            <span className="metric-chip">{result.device_used}</span>
          </div>
          <audio className="restore-player" controls src={speechEditAudioUrl(apiBaseUrl, result.token)} preload="auto" />
          {result.regions.length > 0 ? (
            <details className="restore-advanced" open>
              <summary>Patched {result.regions.length} span{result.regions.length === 1 ? "" : "s"}</summary>
              {result.regions.map((region, index) => (
                <p key={index} className="helper-text">
                  Patched {region.start_s.toFixed(1)}–{region.end_s.toFixed(1)} s — “{region.text_used}”
                </p>
              ))}
            </details>
          ) : null}
          {result.ref_text_used ? (
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
            <a className="button-link" href={speechEditAudioUrl(apiBaseUrl, result.token)} download={result.filename}>
              Download audio
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

export default PatchPanel;
