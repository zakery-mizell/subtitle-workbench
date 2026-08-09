import type { WarningItem } from "../types";

export type SpeechEditMode = "edit" | "generate";
export type SpeechEditOutputFormat = "wav" | "flac" | "mp3" | "aac" | "opus";
export type SpeechEditLanguage = "English";

export interface PatchEdit {
  start_s: number;
  end_s: number;
  new_text: string;
  fix_duration_s: number | null;
  window_text: string | null;
}

export interface SpeechEditParams {
  mode: SpeechEditMode;
  output_format: SpeechEditOutputFormat;
  edits: PatchEdit[];
  gen_text: string;
  ref_text: string;
  auto_ref_text: boolean;
  whisper_model: string;
  language: SpeechEditLanguage;
  nfe_step: number;
  speed: number;
  seed: number | null;
}

export function defaultPatchEdit(): PatchEdit {
  return { start_s: 0, end_s: 0, new_text: "", fix_duration_s: null, window_text: null };
}

export function defaultSpeechEditParams(): SpeechEditParams {
  return {
    mode: "edit",
    output_format: "flac",
    edits: [defaultPatchEdit()],
    gen_text: "",
    ref_text: "",
    auto_ref_text: true,
    whisper_model: "small",
    language: "English",
    nfe_step: 32,
    speed: 1.0,
    seed: null,
  };
}

export interface PatchRegion {
  start_s: number;
  end_s: number;
  window_start_s: number;
  window_end_s: number;
  text_used: string;
}

export interface SpeechEditResult {
  token: string;
  filename: string;
  output_format: string;
  sample_rate: number;
  duration_sec: number;
  device_used: string;
  mode: SpeechEditMode;
  regions: PatchRegion[];
  ref_text_used: string | null;
  warnings: WarningItem[];
}

export interface SpeechEditJobStatus {
  id: string;
  status: "queued" | "running" | "done" | "error";
  stage: string;
  progress: number;
  message: string | null;
  error: string | null;
  result: SpeechEditResult | null;
}

export async function startSpeechEditJob(
  apiBaseUrl: string,
  audioFile: File,
  params: SpeechEditParams,
): Promise<string> {
  const formData = new FormData();
  formData.append("audio", audioFile);
  formData.append("params_json", JSON.stringify(params));
  const response = await fetch(`${apiBaseUrl}/api/speech-edit`, { method: "POST", body: formData });
  if (!response.ok) {
    const detail = await response.json().catch(() => null);
    throw new Error(detail?.detail ?? `Speech-edit request failed (${response.status}).`);
  }
  const payload = (await response.json()) as { job_id: string };
  return payload.job_id;
}

export async function fetchSpeechEditJob(apiBaseUrl: string, jobId: string): Promise<SpeechEditJobStatus> {
  const response = await fetch(`${apiBaseUrl}/api/jobs/${jobId}`);
  if (!response.ok) {
    throw new Error(`Could not read speech-edit job status (${response.status}).`);
  }
  return (await response.json()) as SpeechEditJobStatus;
}

export function speechEditAudioUrl(apiBaseUrl: string, token: string): string {
  return `${apiBaseUrl}/api/speech-edit/${token}/audio`;
}

export function speechEditWaveformUrl(apiBaseUrl: string, token: string): string {
  return `${apiBaseUrl}/api/speech-edit/${token}/waveform`;
}

export async function deleteSpeechEdit(apiBaseUrl: string, token: string): Promise<void> {
  await fetch(`${apiBaseUrl}/api/speech-edit/${token}`, { method: "DELETE" }).catch(() => undefined);
}
