import type { WarningItem, WordToken } from "../types";

export type LevelerStrength = "tight" | "moderate" | "soft";
export type CutApplyMode = "apply" | "silence" | "list_only";
export type OutputFormat = "wav" | "flac" | "mp3" | "aac" | "opus";
export type LoudnessPreset = "podcast" | "streaming" | "broadcast" | "mono" | "custom";

export interface MasteringParams {
  denoise: { enabled: boolean; amount: number };
  hum_removal: { enabled: boolean; base_frequency: "auto" | "50" | "60"; max_harmonics: number };
  high_pass: { enabled: boolean; cutoff: "auto" | number };
  leveler: { enabled: boolean; strength: LevelerStrength };
  loudness: { enabled: boolean; preset: LoudnessPreset; target_lufs: number; true_peak_dbtp: number };
  cutting: {
    silence: { enabled: boolean; keep_pause_seconds: number; max_pause_seconds: number };
    fillers: { enabled: boolean; engine: "words" };
    apply_mode: CutApplyMode;
  };
  output: { format: OutputFormat; bitrate_kbps: number | null; downmix_mono: boolean };
}

export const LOUDNESS_PRESETS: Array<{ id: LoudnessPreset; label: string; lufs: number | null }> = [
  { id: "podcast", label: "Podcast -16 LUFS", lufs: -16 },
  { id: "streaming", label: "Streaming -14 LUFS", lufs: -14 },
  { id: "broadcast", label: "Broadcast (EBU R128) -23 LUFS", lufs: -23 },
  { id: "mono", label: "Mono voice -19 LUFS", lufs: -19 },
  { id: "custom", label: "Custom", lufs: null },
];

export function defaultMasteringParams(): MasteringParams {
  return {
    denoise: { enabled: true, amount: 100 },
    hum_removal: { enabled: true, base_frequency: "auto", max_harmonics: 8 },
    high_pass: { enabled: true, cutoff: "auto" },
    leveler: { enabled: true, strength: "moderate" },
    loudness: { enabled: true, preset: "podcast", target_lufs: -16, true_peak_dbtp: -1 },
    cutting: {
      silence: { enabled: false, keep_pause_seconds: 0.6, max_pause_seconds: 1.5 },
      fillers: { enabled: false, engine: "words" },
      apply_mode: "list_only",
    },
    output: { format: "wav", bitrate_kbps: null, downmix_mono: false },
  };
}

export interface LoudnessStats {
  integrated_lufs: number;
  lra: number;
  true_peak_dbtp: number;
  noise_floor_dbfs: number;
}

export interface CutRegion {
  start: number;
  end: number;
  reason: "silence" | "filler";
  label: string;
}

export interface MasteringReport {
  before: LoudnessStats;
  after: LoudnessStats;
  hum: { detected: boolean; base_frequency: number | null; harmonics_notched: number };
  denoise: { applied: boolean; device_used: string | null };
  high_pass: { applied: boolean; cutoff_hz: number | null };
  leveler: { applied: boolean; max_boost_db: number | null; max_cut_db: number | null };
}

export interface MasteringResult {
  token: string;
  output_filename: string;
  output_format: OutputFormat;
  duration_before: number;
  duration_after: number;
  report: MasteringReport;
  cut_list: CutRegion[];
  warnings: WarningItem[];
}

export interface MasteringJobStatus {
  id: string;
  status: "queued" | "running" | "done" | "error";
  stage: string;
  progress: number;
  message: string | null;
  error: string | null;
  result: MasteringResult | null;
}

export async function startMasteringJob(
  apiBaseUrl: string,
  audioFile: File,
  params: MasteringParams,
  words: WordToken[] | null,
): Promise<string> {
  const formData = new FormData();
  formData.append("audio", audioFile);
  formData.append("params_json", JSON.stringify(params));
  if (words && params.cutting.fillers.enabled) {
    formData.append(
      "words_json",
      JSON.stringify(words.map((word) => ({ text: word.text, start: word.start, end: word.end }))),
    );
  }
  const response = await fetch(`${apiBaseUrl}/api/master`, { method: "POST", body: formData });
  if (!response.ok) {
    const detail = await response.json().catch(() => null);
    throw new Error(detail?.detail ?? `Mastering request failed (${response.status}).`);
  }
  const payload = (await response.json()) as { job_id: string };
  return payload.job_id;
}

export async function fetchMasteringJob(apiBaseUrl: string, jobId: string): Promise<MasteringJobStatus> {
  const response = await fetch(`${apiBaseUrl}/api/jobs/${jobId}`);
  if (!response.ok) {
    throw new Error(`Could not read mastering job status (${response.status}).`);
  }
  return (await response.json()) as MasteringJobStatus;
}

export function masterAudioUrl(apiBaseUrl: string, token: string): string {
  return `${apiBaseUrl}/api/master/${token}/audio`;
}

export function masterCutListUrl(apiBaseUrl: string, token: string, format: "json" | "audacity"): string {
  return `${apiBaseUrl}/api/master/${token}/cut-list?format=${format}`;
}
