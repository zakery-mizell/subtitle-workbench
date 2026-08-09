import type { OverlapRegion, SpeakerTurn, WarningItem } from "../types";
import type { RestoreEngineName } from "./restore";

export type SeparationMode = "spotlight" | "mute";

export interface SeparationRegionParams {
  start: number;
  end: number;
  mode: SeparationMode;
  target_speaker_index: number;
}

export interface SeparationParams {
  regions: SeparationRegionParams[];
  turns: Array<{ start: number; end: number; speaker_index: number }>;
  duck_db: number;
  transcribe_stems: boolean;
  transcribe_model: string;
  language: "en";
  output: { format: "wav" | "flac" | "mp3" | "aac" | "opus"; bitrate_kbps: number | null };
}

export interface StemWord {
  text: string;
  start: number;
  end: number;
  confidence?: number | null;
}

export interface HandoffAuditWord {
  id: string;
  text: string;
  start: number;
  end: number;
  speaker_index: number;
}

export interface HandoffCorrection {
  word_id: string;
  from_speaker_index: number;
  to_speaker_index: number;
  confidence: number;
  boundary_time: number;
}

export interface RegionReport {
  start: number;
  end: number;
  mode: SeparationMode;
  target_speaker_index: number;
  applied: boolean;
  enrollment_start: number | null;
  enrollment_end: number | null;
  words: StemWord[] | null;
  detail: string | null;
}

export interface SeparationResult {
  token: string;
  output_filename: string;
  output_format: string;
  device_used: string;
  regions: RegionReport[];
  warnings: WarningItem[];
}

export interface SeparationJobStatus {
  id: string;
  status: "queued" | "running" | "done" | "error";
  stage: string;
  progress: number;
  message: string | null;
  error: string | null;
  result: SeparationResult | null;
}

export function buildSeparationParams(
  regions: SeparationRegionParams[],
  turns: SpeakerTurn[],
  options?: Partial<Pick<SeparationParams, "duck_db" | "transcribe_stems" | "transcribe_model">>,
): SeparationParams {
  return {
    regions,
    turns: turns.map((turn) => ({ start: turn.start, end: turn.end, speaker_index: turn.speaker_index })),
    duck_db: options?.duck_db ?? -16,
    transcribe_stems: options?.transcribe_stems ?? true,
    transcribe_model: options?.transcribe_model ?? "small",
    language: "en",
    output: { format: "wav", bitrate_kbps: null },
  };
}

export async function startSeparationJob(
  apiBaseUrl: string,
  audioFile: File,
  params: SeparationParams,
): Promise<string> {
  const formData = new FormData();
  formData.append("audio", audioFile);
  formData.append("params_json", JSON.stringify(params));
  const response = await fetch(`${apiBaseUrl}/api/separate`, { method: "POST", body: formData });
  if (!response.ok) {
    const detail = await response.json().catch(() => null);
    throw new Error(detail?.detail ?? `Separation request failed (${response.status}).`);
  }
  const payload = (await response.json()) as { job_id: string };
  return payload.job_id;
}

export async function fetchSeparationJob(apiBaseUrl: string, jobId: string): Promise<SeparationJobStatus> {
  const response = await fetch(`${apiBaseUrl}/api/jobs/${jobId}`);
  if (!response.ok) {
    throw new Error(`Could not read separation job status (${response.status}).`);
  }
  return (await response.json()) as SeparationJobStatus;
}

export function separatedAudioUrl(apiBaseUrl: string, token: string): string {
  return `${apiBaseUrl}/api/separate/${token}/audio`;
}

export interface SoloTrackOut {
  speaker_index: number;
  token: string;
  output_filename: string;
  /** False when the track fell back to the gated original (no solo speech to enroll from). */
  separated: boolean;
  /** Word-level transcript of the finished stem (the "speaker-based" transcript). */
  words: StemWord[] | null;
  detail: string | null;
}

export interface SoloRegionReport {
  start: number;
  end: number;
  speaker_index: number;
  applied: boolean;
  detail: string | null;
}

export interface SoloTracksResult {
  tracks: SoloTrackOut[];
  regions: SoloRegionReport[];
  output_format: string;
  device_used: string;
  warnings: WarningItem[];
  handoff_corrections: HandoffCorrection[];
  handoffs_audited: number;
}

export interface SoloTracksJobStatus {
  id: string;
  status: "queued" | "running" | "done" | "error";
  stage: string;
  progress: number;
  message: string | null;
  error: string | null;
  result: SoloTracksResult | null;
}

export interface SpeakerRegionParams {
  start: number;
  end: number;
  speaker_index: number;
}

export type SoloTracksMode = "gated" | "targeted";

export interface SoloTracksOptions {
  /**
   * "targeted" runs UniSE only in merged windows around overlaps and rapid
   * handoffs. "gated" is the legacy overlap-only mode.
   */
  mode?: SoloTracksMode;
  /** Transcribe each finished stem and return its words (speaker-based transcript). */
  transcribe?: boolean;
  transcribeModel?: string;
  /** Mixture words used to find and verify rapid target-speaker windows. */
  auditWords?: HandoffAuditWord[];
}

export async function startSoloTracksJob(
  apiBaseUrl: string,
  audioFile: File,
  regions: OverlapRegion[],
  turns: SpeakerTurn[],
  speakerRegions: SpeakerRegionParams[] = [],
  restore = false,
  restoreEngine: RestoreEngineName = "sidon",
  options: SoloTracksOptions = {},
): Promise<string> {
  const params = {
    mode: options.mode ?? "gated",
    regions,
    turns: turns.map((turn) => ({ start: turn.start, end: turn.end, speaker_index: turn.speaker_index })),
    // Silence-snapped regions per speaker: the backend mutes everything outside
    // them, so each track is timeline-preserving but carries one voice only.
    // Targeted mode uses these to keep each assembled playback track speaker-only.
    speaker_regions: speakerRegions,
    // When set, each per-speaker track is restored after isolation (Sidon at
    // 48 kHz or Diamond at 44.1 kHz); default keeps the raw separated stems.
    restore,
    restore_engine: restoreEngine,
    transcribe: options.transcribe ?? false,
    transcribe_model: options.transcribeModel ?? "small",
    language: "en",
    audit_handoffs: true,
    audit_words: options.auditWords ?? [],
    // FLAC halves the disk footprint of the timeline-preserving speaker tracks.
    output: { format: "flac", bitrate_kbps: null },
  };
  const formData = new FormData();
  formData.append("audio", audioFile);
  formData.append("params_json", JSON.stringify(params));
  const response = await fetch(`${apiBaseUrl}/api/separate-solo`, { method: "POST", body: formData });
  if (!response.ok) {
    const detail = await response.json().catch(() => null);
    throw new Error(detail?.detail ?? `Solo track request failed (${response.status}).`);
  }
  const payload = (await response.json()) as { job_id: string };
  return payload.job_id;
}

export async function fetchSoloTracksJob(apiBaseUrl: string, jobId: string): Promise<SoloTracksJobStatus> {
  const response = await fetch(`${apiBaseUrl}/api/jobs/${jobId}`);
  if (!response.ok) {
    throw new Error(`Could not read solo track job status (${response.status}).`);
  }
  return (await response.json()) as SoloTracksJobStatus;
}

export function describeOverlapSpeakers(region: OverlapRegion, names: Map<number, string>): string {
  return region.speaker_indices.map((index) => names.get(index) ?? `Speaker ${index + 1}`).join(" + ");
}
