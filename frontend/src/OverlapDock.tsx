import { useEffect, useRef, useState } from "react";
import type { Dispatch, SetStateAction } from "react";

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
import { formatClock } from "./lib/time";
import type { OverlapRegion, Speaker, SpeakerTurn } from "./types";

const JOB_POLL_MS = 1000;

export interface RegionChoice {
  enabled: boolean;
  mode: SeparationMode;
  targetIndex: number;
}

export function overlapRegionKey(region: OverlapRegion): string {
  return `${region.start.toFixed(3)}-${region.end.toFixed(3)}`;
}

export function defaultOverlapChoice(region: OverlapRegion): RegionChoice {
  return { enabled: false, mode: "spotlight", targetIndex: region.speaker_indices[0] ?? 0 };
}

interface OverlapDockProps {
  apiBaseUrl: string;
  audioFile: File | null;
  regions: OverlapRegion[];
  turns: SpeakerTurn[];
  speakers: Speaker[];
  choices: Record<string, RegionChoice>;
  // The parent's setter, not a plain callback: every edit here is a functional
  // update, so a second edit can never be written against a stale `choices`.
  onChoicesChange: Dispatch<SetStateAction<Record<string, RegionChoice>>>;
  selectedIndex: number | null;
  onSelectIndex: (index: number | null) => void;
  onProcessed: (result: SeparationResult, audioUrl: string) => void;
  onApplyWords: (report: RegionReport) => void;
  onSeek: (time: number, options?: { play?: boolean }) => void;
}

/**
 * Lives directly under the main waveform: clicking a highlighted overlap band
 * on the waveform selects a region here, where it can be spotlighted or muted.
 * The separation job itself (UniSE) also runs from this dock.
 */
export function OverlapDock({
  apiBaseUrl,
  audioFile,
  regions,
  turns,
  speakers,
  choices,
  onChoicesChange,
  selectedIndex,
  onSelectIndex,
  onProcessed,
  onApplyWords,
  onSeek,
}: OverlapDockProps) {
  const [transcribeStems, setTranscribeStems] = useState(true);
  const [jobId, setJobId] = useState<string | null>(null);
  const [job, setJob] = useState<SeparationJobStatus | null>(null);
  const [result, setResult] = useState<SeparationResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [appliedWordKeys, setAppliedWordKeys] = useState<Set<string>>(new Set());
  const pollRef = useRef<number | null>(null);
  // The poll interval must survive App re-renders (4x a second during
  // playback), so the callback it fires is read from a ref instead of being an
  // effect dependency — rebuilding the interval discards the in-flight poll.
  const onProcessedRef = useRef(onProcessed);
  onProcessedRef.current = onProcessed;

  const running = job !== null && (job.status === "queued" || job.status === "running");

  const speakerName = (index: number) => speakers[index]?.name ?? `Speaker ${index + 1}`;

  const enabledRegions = regions.filter(
    (region) => (choices[overlapRegionKey(region)] ?? defaultOverlapChoice(region)).enabled,
  );

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
          onProcessedRef.current(status.result, separatedAudioUrl(apiBaseUrl, status.result.token));
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
  }, [apiBaseUrl, jobId]);

  function updateChoice(region: OverlapRegion, mutator: (draft: RegionChoice) => void) {
    const key = overlapRegionKey(region);
    onChoicesChange((current) => {
      const draft = { ...(current[key] ?? defaultOverlapChoice(region)) };
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
      const choice = choices[overlapRegionKey(region)] ?? defaultOverlapChoice(region);
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
      });
      const id = await startSeparationJob(apiBaseUrl, audioFile, params);
      setJobId(id);
      setJob({ id, status: "queued", stage: "queued", progress: 0, message: null, error: null, result: null });
    } catch (startError) {
      setError(startError instanceof Error ? startError.message : "Could not start separation.");
    }
  }

  const selectedRegion = selectedIndex !== null ? regions[selectedIndex] ?? null : null;
  const showRunRow = enabledRegions.length > 0 || running || result !== null || error !== null;

  if (!regions.length) {
    return null;
  }

  return (
    <div className="overlap-dock">
      {/* The whole set at a glance — the dock itself only ever shows one
          region, so this is where a batch is assembled and reviewed. */}
      <div className="overlap-chip-strip">
        {regions.map((region, index) => {
          const key = overlapRegionKey(region);
          const enabled = (choices[key] ?? defaultOverlapChoice(region)).enabled;
          return (
            <button
              key={key}
              type="button"
              className={`overlap-chip ${enabled ? "is-enabled" : ""} ${index === selectedIndex ? "is-selected" : ""}`}
              aria-pressed={index === selectedIndex}
              title={`${formatClock(region.start)}–${formatClock(region.end)} · ${
                enabled ? "will be processed" : "not selected for processing"
              }`}
              onClick={() => onSelectIndex(index)}
            >
              {enabled ? <span aria-hidden>✓</span> : null}
              Overlap {index + 1}
            </button>
          );
        })}
      </div>

      {selectedRegion ? (
        (() => {
          const key = overlapRegionKey(selectedRegion);
          const choice = choices[key] ?? defaultOverlapChoice(selectedRegion);
          const report = result?.regions.find(
            (candidate) =>
              Math.abs(candidate.start - selectedRegion.start) < 0.05 &&
              Math.abs(candidate.end - selectedRegion.end) < 0.05,
          );
          return (
            <div className={`overlap-card ${choice.enabled ? "is-enabled" : ""}`}>
              <div className="overlap-card-header">
                <strong>
                  Overlap {(selectedIndex ?? 0) + 1} · {formatClock(selectedRegion.start)}–{formatClock(selectedRegion.end)}
                </strong>
                <div className="overlap-card-tools">
                  <button
                    type="button"
                    className="overlap-jump"
                    onClick={() => onSeek(Math.max(0, selectedRegion.start - 1), { play: true })}
                  >
                    Listen
                  </button>
                  <button
                    type="button"
                    className="overlap-jump"
                    aria-label="Close overlap controls"
                    onClick={() => onSelectIndex(null)}
                  >
                    ✕
                  </button>
                </div>
              </div>
              <p className="helper-text overlap-speakers">
                {selectedRegion.speaker_indices.map((speakerIndex) => speakerName(speakerIndex)).join(" and ")} at the
                same time for {(selectedRegion.end - selectedRegion.start).toFixed(1)}s. Spotlight lifts one voice above
                a quieted mix; mute removes a voice entirely.
              </p>
              <label className="mastering-toggle">
                <input
                  type="checkbox"
                  checked={choice.enabled}
                  onChange={(event) => updateChoice(selectedRegion, (draft) => void (draft.enabled = event.target.checked))}
                />
                <span>Untangle this overlap</span>
              </label>
              {choice.enabled ? (
                <div className="mastering-grid-2">
                  <label>
                    Action
                    <select
                      value={choice.mode}
                      onChange={(event) =>
                        updateChoice(selectedRegion, (draft) => void (draft.mode = event.target.value as SeparationMode))
                      }
                    >
                      <option value="spotlight">Spotlight one voice</option>
                      <option value="mute">Mute one voice</option>
                    </select>
                  </label>
                  <label>
                    {choice.mode === "spotlight" ? "Voice to keep on top" : "Voice to remove"}
                    <select
                      value={choice.targetIndex}
                      onChange={(event) =>
                        updateChoice(selectedRegion, (draft) => void (draft.targetIndex = Number(event.target.value)))
                      }
                    >
                      {selectedRegion.speaker_indices.map((speakerIndex) => (
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
        })()
      ) : null}

      {showRunRow ? (
        <div className="overlap-dock-run">
          <label className="mastering-toggle">
            <input type="checkbox" checked={transcribeStems} onChange={(event) => setTranscribeStems(event.target.checked)} />
            <span>Transcribe spotlighted voices</span>
          </label>
          <button disabled={!audioFile || !enabledRegions.length || running} onClick={() => void handleRun()}>
            {running
              ? "Separating..."
              : `Process ${enabledRegions.length || "selected"} overlap${enabledRegions.length === 1 ? "" : "s"}`}
          </button>
          {!audioFile ? <span className="helper-text">Choose an audio file first.</span> : null}
          {running && job ? (
            <div className="mastering-progress overlap-dock-progress">
              <div className="mastering-progress-track">
                <div className="mastering-progress-fill" style={{ width: `${Math.round(job.progress * 100)}%` }} />
              </div>
              <p className="helper-text">
                {Math.round(job.progress * 100)}% – {job.message ?? job.stage}
              </p>
            </div>
          ) : null}
          {error ? <span className="status-text">{error}</span> : null}
          {result ? (
            <>
              <span className="metric-chip">Ran on {result.device_used.toUpperCase()}</span>
              <span className="metric-chip">
                {result.regions.filter((region) => region.applied).length}/{result.regions.length} overlaps processed
              </span>
              <a className="button-link" href={separatedAudioUrl(apiBaseUrl, result.token)} download={result.output_filename}>
                Download processed audio
              </a>
              <p className="helper-text overlap-dock-hint">
                Playback switched to the processed audio — use the Original/Processed toggle in the transport bar to
                compare.
              </p>
            </>
          ) : null}
          {result?.warnings.map((warning) => (
            <span key={warning.code} className="status-text">
              {warning.message}
            </span>
          )) ?? null}
        </div>
      ) : null}
    </div>
  );
}

export default OverlapDock;
