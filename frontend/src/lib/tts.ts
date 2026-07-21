import type { WarningItem } from "../types";

export type TtsLanguage =
  | "Auto"
  | "Chinese"
  | "English"
  | "German"
  | "Italian"
  | "Portuguese"
  | "Spanish"
  | "Japanese"
  | "Korean"
  | "French"
  | "Russian";
export type TtsModelSize = "1.7b" | "0.6b";
export type TtsOutputFormat = "wav" | "flac" | "mp3" | "aac" | "opus";

export const TTS_LANGUAGES: TtsLanguage[] = [
  "Auto",
  "Chinese",
  "English",
  "German",
  "Italian",
  "Portuguese",
  "Spanish",
  "Japanese",
  "Korean",
  "French",
  "Russian",
];

export interface TtsParams {
  text: string;
  language: TtsLanguage;
  ref_text: string | null;
  auto_ref_text: boolean;
  whisper_model: string;
  model_size: TtsModelSize;
  output_format: TtsOutputFormat;
}

export function defaultTtsParams(): TtsParams {
  return {
    text: "",
    language: "Auto",
    ref_text: null,
    auto_ref_text: true,
    whisper_model: "small",
    model_size: "1.7b",
    output_format: "flac",
  };
}

export interface TtsResult {
  token: string;
  filename: string;
  output_format: string;
  sample_rate: number;
  duration_sec: number;
  device_used: string;
  model_size: string;
  clone_mode: "transcript" | "voice-signature";
  ref_text_used: string | null;
  warnings: WarningItem[];
}

export interface TtsJobStatus {
  id: string;
  status: "queued" | "running" | "done" | "error";
  stage: string;
  progress: number;
  message: string | null;
  error: string | null;
  result: TtsResult | null;
}

export async function startTtsJob(apiBaseUrl: string, refFile: File, params: TtsParams): Promise<string> {
  const formData = new FormData();
  formData.append("audio", refFile);
  formData.append("params_json", JSON.stringify(params));
  const response = await fetch(`${apiBaseUrl}/api/tts`, { method: "POST", body: formData });
  if (!response.ok) {
    const detail = await response.json().catch(() => null);
    throw new Error(detail?.detail ?? `Voice cloning request failed (${response.status}).`);
  }
  const payload = (await response.json()) as { job_id: string };
  return payload.job_id;
}

export async function fetchTtsJob(apiBaseUrl: string, jobId: string): Promise<TtsJobStatus> {
  const response = await fetch(`${apiBaseUrl}/api/jobs/${jobId}`);
  if (!response.ok) {
    throw new Error(`Could not read voice job status (${response.status}).`);
  }
  return (await response.json()) as TtsJobStatus;
}

export function ttsAudioUrl(apiBaseUrl: string, token: string): string {
  return `${apiBaseUrl}/api/tts/${token}/audio`;
}

export async function deleteTts(apiBaseUrl: string, token: string): Promise<void> {
  await fetch(`${apiBaseUrl}/api/tts/${token}`, { method: "DELETE" }).catch(() => undefined);
}
