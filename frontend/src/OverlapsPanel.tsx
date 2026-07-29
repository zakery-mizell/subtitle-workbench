import { useEffect, useMemo, useRef, useState } from "react";

import {
  buildSeparationParams,
  fetchSeparationJob,
  separatedAudioUrl,
  startSeparationJob,
} from "./lib/separation";
import type {
  RegionReport,
  SeparationJobStatus,
  SeparationMode,
  SeparationRegionParams,
  SeparationResult,
} from "./lib/separation";
import type { RestoreEngineName } from "./lib/restore";
import { formatClock } from "./lib/time";
import type { OverlapRegion, Speaker, SpeakerTurn } from "./types";

const JOB_POLL_MS = 1000;

export interface RegionChoice {
  enabled: boolean;
  mode: SeparationMode;
  targetIndex: number;
}

interface OverlapsPanelProps {
  apiBaseUrl: string;
  audioFile: File | null;
  regions: OverlapRegion[];
  turns: SpeakerTurn[];
  speakers: Speaker[];
  language: string | null;
  restoreSoloTracks: boolean;
  onRestoreSoloTracksChange: (value: boolean) => void;
  restoreEngine: RestoreEngineName;
  onRestoreEngineChange: (value: RestoreEngineName) => void;
  onProcessed: (result: SeparationResult, audioUrl: string) => void;
  onApplyWords: (report: RegionReport) => void;
  onSeek: (time: number, options?: { play?: boolean }) => void;
}

function regionKey(region: OverlapRegion): string {
  return `${region.start.toFixed(3)}-${region.end.toFixed(3)}`;
}

function defaultChoice(region: OverlapRegion): RegionChoice {
  return { enabled: false, mode: "spotlight", targetIndex: region.speaker_indices[0] ?? 0 };
}

export function OverlapsPanel({
  apiBaseUrl,
  audioFile,
  regions,
  turns,
  speakers,
  language,
  restoreSoloTracks,
  onRestoreSoloTracksChange,
  restoreEngine,
  onRestoreEngineChange,
  onProcessed,
  onApplyWords,
  onSeek,
}: OverlapsPanelProps) {
  const [choices, setChoices] = useState<Record<string, RegionChoice>>({});
  const [transcribeStems, setTranscribeStems] = useState(true);
  const [jobId, setJobId] = useState<string | null>(null);
  const [job, setJob] = useState<SeparationJobStatus | null>(null);
  const [result, setResult] = useState<SeparationResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [appliedWordKeys, setAppliedWordKeys] = useState<Set<string>>(new Set());
  const pollRef = useRef<number | null>(null);

  const running = job !== null && (job.status === "queued" || job.status === "running");

  const speakerNames = useMemo(() => {
    const names = new Map<number, string>();
    speakers.forEach((speaker, index) => names.set(index, speaker.name));
    return names;
  }, [speakers]);

  const speakerName = (index: number) => speakerNames.get(index) ?? `Speaker ${index + 1}`;

  const enabledRegions = regions.filter((region) => (choices[regionKey(region)] ?? defaultChoice(region)).enabled);

  useEffect(() => {
    if (!jobId) {
      return;
    }
    let cancelled = false;

    async function poll() {
      try {
        const status = await fetchSeparationJob(apiBaseUrl, jobId!);
        if (cancelled) {
          return;
        }
        setJob(status);
        if (status.status === "done" && status.result) {
          setResult(status.result);
          setJobId(null);
          onProcessed(status.result, separatedAudioUrl(apiBaseUrl, status.result.token));
        } else if (status.status === "error") {
          setError(status.error ?? "Separation failed.");
          setJobId(null);
        }
      } catch (pollError) {
        if (!cancelled) {
          setError(pollError instanceof Error ? pollError.message : "Lost contact with the separation job.");
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

  function updateChoice(region: OverlapRegion, mutator: (draft: RegionChoice) => void) {
    setChoices((current) => {
      const key = regionKey(region);
      const draft = { ...(current[key] ?? defaultChoice(region)) };
      mutator(draft);
      return { ...current, [key]: draft };
    });
  }

  async function handleRun() {
    if (!audioFile || !enabledRegions.length) {
      return;
    }
    setError(null);
    setResult(null);
    setJob(null);
    setAppliedWordKeys(new Set());
    const regionParams: SeparationRegionParams[] = enabledRegions.map((region) => {
      const choice = choices[regionKey(region)] ?? defaultChoice(region);
      return {
        start: region.start,
        end: region.end,
        mode: choice.mode,
        target_speaker_index: choice.targetIndex,
      };
    });
    try {
      const params = buildSeparationParams(regionParams, turns, {
        transcribe_stems: transcribeStems,
        language,
      });
      const id = await startSeparationJob(apiBaseUrl, audioFile, params);
      setJobId(id);
      setJob({ id, status: "queued", stage: "queued", progress: 0, message: null, error: null, result: null });
    } catch (startError) {
      setError(startError instanceof Error ? startError.message : "Could not start separation.");
    }
  }

  return (
    <section className="selection-panel mastering-panel overlaps-panel">
      <div className="panel-section-heading">
        <p className="eyebrow">Overlaps</p>
        <h3>Untangle simultaneous speech</h3>
      </div>
      <p className="helper-text">
        AI separation (UniSE) for the moments where speakers talk over each other. Spotlight lifts one voice above a
        quieted mix; mute removes a voice entirely. The processed sections are re-synthesized at speech quality, so
        the rest of the recording stays untouched.
      </p>

      {!regions.length ? (
        <p className="helper-text">
          No overlapping speech was detected in this recording. Overlaps are found during transcription when more
          than one speaker is configured, so retranscribe with the right speaker count if something is missing.
        </p>
      ) : null}

      {regions.map((region, index) => {
        const key = regionKey(region);
        const choice = choices[key] ?? defaultChoice(region);
        const report = result?.regions.find(
          (candidate) => Math.abs(candidate.start - region.start) < 0.05 && Math.abs(candidate.end - region.end) < 0.05,
        );
        return (
          <div key={key} className={`overlap-card ${choice.enabled ? "is-enabled" : ""}`}>
            <div className="overlap-card-header">
              <label className="mastering-toggle">
                <input
                  type="checkbox"
                  checked={choice.enabled}
                  onChange={(event) => updateChoice(region, (draft) => void (draft.enabled = event.target.checked))}
                />
                <span>
                  Overlap {index + 1} · {formatClock(region.start)}–{formatClock(region.end)}
                </span>
              </label>
              <button type="button" className="overlap-jump" onClick={() => onSeek(Math.max(0, region.start - 1), { play: true })}>
                Listen
              </button>
            </div>
            <p className="helper-text overlap-speakers">
              {region.speaker_indices.map((speakerIndex) => speakerName(speakerIndex)).join(" and ")} at the same time
              for {(region.end - region.start).toFixed(1)}s
            </p>
            {choice.enabled ? (
              <div className="mastering-grid-2">
                <label>
                  Action
                  <select
                    value={choice.mode}
                    onChange={(event) => updateChoice(region, (draft) => void (draft.mode = event.target.value as SeparationMode))}
                  >
                    <option value="spotlight">Spotlight one voice</option>
                    <option value="mute">Mute one voice</option>
                  </select>
                </label>
                <label>
                  {choice.mode === "spotlight" ? "Voice to keep on top" : "Voice to remove"}
                  <select
                    value={choice.targetIndex}
                    onChange={(event) => updateChoice(region, (draft) => void (draft.targetIndex = Number(event.target.value)))}
                  >
                    {region.speaker_indices.map((speakerIndex) => (
                      <option key={speakerIndex} value={speakerIndex}>
                        {speakerName(speakerIndex)}
                      </option>
                    ))}
                  </select>
                </label>
              </div>
            ) : null}
            {report ? (
              <div className="overlap-report">
                {report.applied ? (
                  <p className="helper-text">
                    {report.mode === "spotlight"
                      ? `${speakerName(report.target_speaker_index)} spotlighted.`
                      : `${speakerName(report.target_speaker_index)} muted.`}
                    {report.enrollment_start !== null
                      ? ` Voice sample: ${formatClock(report.enrollment_start)}–${formatClock(report.enrollment_end ?? report.enrollment_start)}.`
                      : ""}
                  </p>
                ) : (
                  <p className="status-text">{report.detail ?? "This overlap could not be processed."}</p>
                )}
                {report.words?.length ? (
                  <>
                    <p className="overlap-stem-words">“{report.words.map((word) => word.text).join(" ")}”</p>
                    <div className="inline-actions">
                      <button
                        disabled={appliedWordKeys.has(key)}
                        onClick={() => {
                          onApplyWords(report);
                          setAppliedWordKeys((current) => new Set(current).add(key));
                        }}
                      >
                        {appliedWordKeys.has(key)
                          ? "Added to transcript"
                          : `Add ${speakerName(report.target_speaker_index)}'s words to transcript`}
                      </button>
                    </div>
                  </>
                ) : null}
              </div>
            ) : null}
          </div>
        );
      })}

      {regions.length ? (
        <>
          <div className="panel-section-heading">
            <p className="eyebrow">Per-speaker muting</p>
            <h3>Per-speaker solo tracks</h3>
          </div>
          <p className="helper-text">
            The transport bar&apos;s speaker toggles mute one voice at a time, even inside overlaps.
            These tracks are prepared automatically.
          </p>
          <label className="mastering-toggle">
            <input
              type="checkbox"
              checked={restoreSoloTracks}
              onChange={(event) => onRestoreSoloTracksChange(event.target.checked)}
            />
            <span>Restore voices</span>
          </label>
          {restoreSoloTracks ? (
            <label>
              Engine
              <select
                value={restoreEngine}
                onChange={(event) => onRestoreEngineChange(event.target.value as RestoreEngineName)}
              >
                <option value="sidon">Sidon — multilingual, 48 kHz</option>
                <option value="diamond">Diamond — English, 44.1 kHz</option>
              </select>
            </label>
          ) : null}
          <p className="helper-text">
            {restoreEngine === "sidon"
              ? "Cleans each solo track after isolation and returns 48 kHz. Multilingual, and about 4× faster than real-time on this Mac."
              : "Regenerates each solo track as 44.1 kHz speech after isolation. English-trained and roughly 10× slower than real-time on this Mac."}{" "}
            Changing either setting re-renders the solo tracks.
          </p>

          <label className="mastering-toggle">
            <input type="checkbox" checked={transcribeStems} onChange={(event) => setTranscribeStems(event.target.checked)} />
            <span>Transcribe spotlighted voices (recovers words lost in the overlap)</span>
          </label>
          <div className="inline-actions">
            <button disabled={!audioFile || !enabledRegions.length || running} onClick={() => void handleRun()}>
              {running
                ? "Separating..."
                : `Process ${enabledRegions.length || "selected"} overlap${enabledRegions.length === 1 ? "" : "s"}`}
            </button>
          </div>
          {!audioFile ? <p className="helper-text">Choose an audio file first (drag one in or use Browse).</p> : null}
          {!enabledRegions.length && audioFile ? (
            <p className="helper-text">Tick the overlaps you want to process, pick an action for each, then run.</p>
          ) : null}
        </>
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
          <div className="chip-row">
            <span className="metric-chip">Ran on {result.device_used.toUpperCase()}</span>
            <span className="metric-chip">
              {result.regions.filter((region) => region.applied).length}/{result.regions.length} overlaps processed
            </span>
          </div>
          {result.warnings.map((warning) => (
            <p key={warning.code} className="status-text">
              {warning.message}
            </p>
          ))}
          <p className="helper-text">
            Playback switched to the processed audio — use the Original/Processed toggle in the transport bar to
            compare, and download from there when happy.
          </p>
          <div className="inline-actions">
            <a className="button-link" href={separatedAudioUrl(apiBaseUrl, result.token)} download={result.output_filename}>
              Download processed audio
            </a>
          </div>
        </>
      ) : null}
    </section>
  );
}

export default OverlapsPanel;
