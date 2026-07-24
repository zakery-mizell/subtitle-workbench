import type { OverlapRegion, SpeakerTurn, WarningItem } from "../types";

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
  language: string | null;
  output: { format: "wav" | "flac" | "mp3" | "aac" | "opus"; bitrate_kbps: number | null };
}

export interface StemWord {
  text: string;
  start: number;
  end: number;
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
  options?: Partial<Pick<SeparationParams, "duck_db" | "transcribe_stems" | "transcribe_model" | "language">>,
): SeparationParams {
  return {
    regions,
    turns: turns.map((turn) => ({ start: turn.start, end: turn.end, speaker_index: turn.speaker_index })),
    duck_db: options?.duck_db ?? -11,
    transcribe_stems: options?.transcribe_stems ?? true,
    transcribe_model: options?.transcribe_model ?? "small",
    language: options?.language ?? null,
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

export async function startSoloTracksJob(
  apiBaseUrl: string,
  audioFile: File,
  regions: OverlapRegion[],
  turns: SpeakerTurn[],
  speakerRegions: SpeakerRegionParams[] = [],
  restore = false,
): Promise<string> {
  const params = {
    regions,
    turns: turns.map((turn) => ({ start: turn.start, end: turn.end, speaker_index: turn.speaker_index })),
    // Silence-snapped regions per speaker: the backend mutes everything outside
    // them, so each track is timeline-preserving but carries one voice only.
    speaker_regions: speakerRegions,
    // When set, each per-speaker track is regenerated at 44.1 kHz studio quality
    // (Diamond) after isolation; default keeps the raw separated stems.
    restore,
    // FLAC halves the disk footprint of these full-length per-speaker tracks
    // (a 1 h recording is ~600 MB as 48 kHz WAV) and <audio> plays it fine.
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
