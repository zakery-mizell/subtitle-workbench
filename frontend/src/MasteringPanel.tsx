import { useEffect, useRef, useState } from "react";

import {
  defaultMasteringParams,
  fetchMasteringJob,
  LOUDNESS_PRESETS,
  masterAudioUrl,
  masterCutListUrl,
  startMasteringJob,
} from "./lib/mastering";
import type {
  LevelerStrength,
  LoudnessPreset,
  MasteringJobStatus,
  MasteringParams,
  MasteringResult,
  OutputFormat,
} from "./lib/mastering";
import type { WordToken } from "./types";

const JOB_POLL_MS = 1000;

interface MasteringPanelProps {
  apiBaseUrl: string;
  audioFile: File | null;
  words: WordToken[];
  onProcessed: (result: MasteringResult, audioUrl: string) => void;
  onApplyCutsToSubtitles: (result: MasteringResult) => void;
  onSeek: (time: number) => void;
}

function formatDb(value: number | null | undefined, unit: string): string {
  if (value === null || value === undefined || !Number.isFinite(value)) {
    return "–";
  }
  return `${value.toFixed(1)} ${unit}`;
}

function formatDuration(seconds: number): string {
  const minutes = Math.floor(seconds / 60);
  const rest = Math.round(seconds - minutes * 60);
  return `${minutes}:${String(rest).padStart(2, "0")}`;
}

export function MasteringPanel({
  apiBaseUrl,
  audioFile,
  words,
  onProcessed,
  onApplyCutsToSubtitles,
  onSeek,
}: MasteringPanelProps) {
  const [params, setParams] = useState<MasteringParams>(() => defaultMasteringParams());
  const [jobId, setJobId] = useState<string | null>(null);
  const [job, setJob] = useState<MasteringJobStatus | null>(null);
  const [result, setResult] = useState<MasteringResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [cutsApplied, setCutsApplied] = useState(false);
  const pollRef = useRef<number | null>(null);

  const running = job !== null && (job.status === "queued" || job.status === "running");

  useEffect(() => {
    if (!jobId) {
      return;
    }
    let cancelled = false;

    async function poll() {
      try {
        const status = await fetchMasteringJob(apiBaseUrl, jobId!);
        if (cancelled) {
          return;
        }
        setJob(status);
        if (status.status === "done" && status.result) {
          setResult(status.result);
          setJobId(null);
          onProcessed(status.result, masterAudioUrl(apiBaseUrl, status.result.token));
        } else if (status.status === "error") {
          setError(status.error ?? "Mastering failed.");
          setJobId(null);
        }
      } catch (pollError) {
        if (!cancelled) {
          setError(pollError instanceof Error ? pollError.message : "Lost contact with the mastering job.");
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
  }, [apiBaseUrl, jobId, onProcessed]);

  function update(mutator: (draft: MasteringParams) => void) {
    setParams((current) => {
      const draft = structuredClone(current);
      mutator(draft);
      return draft;
    });
  }

  async function handleRun() {
    if (!audioFile) {
      return;
    }
    setError(null);
    setResult(null);
    setCutsApplied(false);
    setJob(null);
    try {
      const id = await startMasteringJob(apiBaseUrl, audioFile, params, words.length ? words : null);
      setJobId(id);
      setJob({ id, status: "queued", stage: "queued", progress: 0, message: null, error: null, result: null });
    } catch (startError) {
      setError(startError instanceof Error ? startError.message : "Could not start mastering.");
    }
  }

  const preset = LOUDNESS_PRESETS.find((entry) => entry.id === params.loudness.preset);
  const report = result?.report ?? null;

  return (
    <section className="selection-panel mastering-panel">
      <div className="panel-section-heading">
        <p className="eyebrow">Master</p>
        <h3>Audio post production</h3>
      </div>
      <p className="helper-text">
        Auphonic-style processing, fully local: denoise, hum removal, speech leveling, loudness normalization, and
        silence/filler cutting.
      </p>

      <label className="mastering-toggle">
        <input
          type="checkbox"
          checked={params.denoise.enabled}
          onChange={(event) => update((draft) => void (draft.denoise.enabled = event.target.checked))}
        />
        <span>AI denoise (MossFormer2)</span>
      </label>
      {params.denoise.enabled ? (
        <label>
          Denoise amount: {Math.round(params.denoise.amount)}%
          <input
            type="range"
            min={0}
            max={100}
            value={params.denoise.amount}
            onChange={(event) => update((draft) => void (draft.denoise.amount = Number(event.target.value)))}
          />
        </label>
      ) : null}

      <label className="mastering-toggle">
        <input
          type="checkbox"
          checked={params.hum_removal.enabled}
          onChange={(event) => update((draft) => void (draft.hum_removal.enabled = event.target.checked))}
        />
        <span>Hum removal (50/60 Hz + harmonics)</span>
      </label>

      <label className="mastering-toggle">
        <input
          type="checkbox"
          checked={params.high_pass.enabled}
          onChange={(event) => update((draft) => void (draft.high_pass.enabled = event.target.checked))}
        />
        <span>Adaptive high-pass filter</span>
      </label>

      <label className="mastering-toggle">
        <input
          type="checkbox"
          checked={params.leveler.enabled}
          onChange={(event) => update((draft) => void (draft.leveler.enabled = event.target.checked))}
        />
        <span>Adaptive leveler</span>
      </label>
      {params.leveler.enabled ? (
        <label>
          Leveler strength
          <select
            value={params.leveler.strength}
            onChange={(event) => update((draft) => void (draft.leveler.strength = event.target.value as LevelerStrength))}
          >
            <option value="soft">Soft (±6 dB)</option>
            <option value="moderate">Moderate (±9 dB)</option>
            <option value="tight">Tight (±12 dB)</option>
          </select>
        </label>
      ) : null}

      <label className="mastering-toggle">
        <input
          type="checkbox"
          checked={params.loudness.enabled}
          onChange={(event) => update((draft) => void (draft.loudness.enabled = event.target.checked))}
        />
        <span>Loudness normalization</span>
      </label>
      {params.loudness.enabled ? (
        <>
          <label>
            Loudness target
            <select
              value={params.loudness.preset}
              onChange={(event) =>
                update((draft) => {
                  draft.loudness.preset = event.target.value as LoudnessPreset;
                  const chosen = LOUDNESS_PRESETS.find((entry) => entry.id === draft.loudness.preset);
                  if (chosen?.lufs !== null && chosen?.lufs !== undefined) {
                    draft.loudness.target_lufs = chosen.lufs;
                  }
                })
              }
            >
              {LOUDNESS_PRESETS.map((entry) => (
                <option key={entry.id} value={entry.id}>
                  {entry.label}
                </option>
              ))}
            </select>
          </label>
          {params.loudness.preset === "custom" ? (
            <div className="mastering-grid-2">
              <label>
                Target LUFS
                <input
                  type="number"
                  step={0.5}
                  min={-40}
                  max={-6}
                  value={params.loudness.target_lufs}
                  onChange={(event) => update((draft) => void (draft.loudness.target_lufs = Number(event.target.value)))}
                />
              </label>
              <label>
                True peak dBTP
                <input
                  type="number"
                  step={0.5}
                  min={-9}
                  max={0}
                  value={params.loudness.true_peak_dbtp}
                  onChange={(event) => update((draft) => void (draft.loudness.true_peak_dbtp = Number(event.target.value)))}
                />
              </label>
            </div>
          ) : null}
        </>
      ) : null}

      <label className="mastering-toggle">
        <input
          type="checkbox"
          checked={params.cutting.silence.enabled}
          onChange={(event) => update((draft) => void (draft.cutting.silence.enabled = event.target.checked))}
        />
        <span>Cut long silences</span>
      </label>
      <label className="mastering-toggle">
        <input
          type="checkbox"
          checked={params.cutting.fillers.enabled}
          onChange={(event) => update((draft) => void (draft.cutting.fillers.enabled = event.target.checked))}
        />
        <span>Cut filler words (um, uh)</span>
      </label>
      {params.cutting.fillers.enabled && !words.length ? (
        <p className="helper-text">Filler cutting needs a transcript. Transcribe first, then run mastering.</p>
      ) : null}
      {params.cutting.silence.enabled || params.cutting.fillers.enabled ? (
        <label>
          Cut mode
          <select
            value={params.cutting.apply_mode}
            onChange={(event) =>
              update((draft) => void (draft.cutting.apply_mode = event.target.value as MasteringParams["cutting"]["apply_mode"]))
            }
          >
            <option value="list_only">Detect only (review first)</option>
            <option value="apply">Remove segments</option>
            <option value="silence">Replace with silence</option>
          </select>
        </label>
      ) : null}

      <label>
        Output format
        <select
          value={params.output.format}
          onChange={(event) => update((draft) => void (draft.output.format = event.target.value as OutputFormat))}
        >
          <option value="wav">WAV</option>
          <option value="flac">FLAC</option>
          <option value="mp3">MP3</option>
          <option value="aac">AAC</option>
          <option value="opus">Opus</option>
        </select>
      </label>

      <div className="inline-actions">
        <button disabled={!audioFile || running} onClick={() => void handleRun()}>
          {running ? "Processing..." : "Run mastering"}
        </button>
      </div>
      {!audioFile ? <p className="helper-text">Choose an audio file first (drag one in or use Browse).</p> : null}

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

      {result && report ? (
        <>
          <div className="panel-section-heading">
            <p className="eyebrow">Result</p>
            <h3>
              {preset && params.loudness.preset !== "custom" ? preset.label : `${params.loudness.target_lufs} LUFS`}
            </h3>
          </div>
          <table className="mastering-stats">
            <thead>
              <tr>
                <th></th>
                <th>Before</th>
                <th>After</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td>Loudness</td>
                <td>{formatDb(report.before.integrated_lufs, "LUFS")}</td>
                <td>{formatDb(report.after.integrated_lufs, "LUFS")}</td>
              </tr>
              <tr>
                <td>Loudness range</td>
                <td>{formatDb(report.before.lra, "LU")}</td>
                <td>{formatDb(report.after.lra, "LU")}</td>
              </tr>
              <tr>
                <td>True peak</td>
                <td>{formatDb(report.before.true_peak_dbtp, "dBTP")}</td>
                <td>{formatDb(report.after.true_peak_dbtp, "dBTP")}</td>
              </tr>
              <tr>
                <td>Noise floor</td>
                <td>{formatDb(report.before.noise_floor_dbfs, "dBFS")}</td>
                <td>{formatDb(report.after.noise_floor_dbfs, "dBFS")}</td>
              </tr>
              <tr>
                <td>Duration</td>
                <td>{formatDuration(result.duration_before)}</td>
                <td>{formatDuration(result.duration_after)}</td>
              </tr>
            </tbody>
          </table>
          <div className="chip-row">
            {report.hum.detected ? (
              <span className="metric-chip">Hum {report.hum.base_frequency} Hz ×{report.hum.harmonics_notched}</span>
            ) : null}
            {report.denoise.applied ? <span className="metric-chip">Denoised ({report.denoise.device_used})</span> : null}
            {report.high_pass.applied ? <span className="metric-chip">High-pass {report.high_pass.cutoff_hz} Hz</span> : null}
            {report.leveler.applied ? (
              <span className="metric-chip">
                Leveler +{report.leveler.max_boost_db}/-{report.leveler.max_cut_db} dB
              </span>
            ) : null}
          </div>
          {result.warnings.map((warning) => (
            <p key={warning.code} className="status-text">
              {warning.message}
            </p>
          ))}
          <div className="inline-actions">
            <a className="button-link" href={masterAudioUrl(apiBaseUrl, result.token)} download={result.output_filename}>
              Download master
            </a>
            {result.cut_list.length ? (
              <a
                className="button-link"
                href={masterCutListUrl(apiBaseUrl, result.token, "audacity")}
                download="cut-labels.txt"
              >
                Download Audacity labels
              </a>
            ) : null}
          </div>

          {result.cut_list.length ? (
            <>
              <div className="panel-section-heading">
                <p className="eyebrow">Cuts</p>
                <h3>{result.cut_list.length} region{result.cut_list.length === 1 ? "" : "s"} on the original timeline</h3>
              </div>
              <div className="qa-list mastering-cut-list">
                {result.cut_list.slice(0, 20).map((cut, index) => (
                  <div key={`${cut.start}-${index}`} className="qa-row">
                    <div className="qa-copy">
                      <strong>{cut.label}</strong>
                      <p className="helper-text">
                        {cut.start.toFixed(2)}s – {cut.end.toFixed(2)}s ({cut.reason})
                      </p>
                    </div>
                    <div className="qa-actions">
                      <button onClick={() => onSeek(cut.start)}>Jump</button>
                    </div>
                  </div>
                ))}
              </div>
              {params.cutting.apply_mode === "apply" ? (
                <div className="inline-actions">
                  <button
                    disabled={cutsApplied}
                    onClick={() => {
                      onApplyCutsToSubtitles(result);
                      setCutsApplied(true);
                    }}
                  >
                    {cutsApplied ? "Cuts applied to subtitles" : "Apply cuts to subtitles"}
                  </button>
                </div>
              ) : null}
              {params.cutting.apply_mode === "apply" ? (
                <p className="helper-text">
                  This shifts every caption, word, and guide block to match the shortened audio. Undo restores the
                  original timing.
                </p>
              ) : null}
            </>
          ) : null}
        </>
      ) : null}
    </section>
  );
}

export default MasteringPanel;
