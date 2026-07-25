import type { WarningItem } from "../types";

export type ConversionOutputFormat = "wav" | "flac" | "mp3" | "aac" | "opus";

export interface ConversionParams {
  output_format: ConversionOutputFormat;
  diffusion_steps: number;
  length_adjust: number;
  intelligibility_cfg: number;
  similarity_cfg: number;
  convert_style: boolean;
}

export function defaultConversionParams(): ConversionParams {
  return {
    output_format: "flac",
    diffusion_steps: 50,
    length_adjust: 1.0,
    intelligibility_cfg: 0.7,
    similarity_cfg: 0.7,
    convert_style: false,
  };
}

export interface ConversionResult {
  token: string;
  filename: string;
  output_format: string;
  sample_rate: number;
  duration_sec: number;
  device_used: string;
  diffusion_steps: number;
  warnings: WarningItem[];
}

export interface ConversionJobStatus {
  id: string;
  status: "queued" | "running" | "done" | "error";
  stage: string;
  progress: number;
  message: string | null;
  error: string | null;
  result: ConversionResult | null;
}

/**
 * Either an uploaded file or an already-rendered isolated speaker track, which
 * the backend reads in place from its own artifact store.
 */
export type ConversionSource = File | { token: string; label: string };

export async function startConversionJob(
  apiBaseUrl: string,
  source: ConversionSource,
  referenceFile: File,
  params: ConversionParams,
): Promise<string> {
  const formData = new FormData();
  if (source instanceof File) {
    formData.append("audio", source);
  } else {
    // No `audio` part at all: the endpoint ignores it when a token is present,
    // and an empty part would still be a multi-megabyte no-op upload.
    formData.append("source_token", source.token);
  }
  formData.append("reference", referenceFile);
  formData.append("params_json", JSON.stringify(params));
  const response = await fetch(`${apiBaseUrl}/api/convert`, { method: "POST", body: formData });
  if (!response.ok) {
    const detail = await response.json().catch(() => null);
    throw new Error(detail?.detail ?? `Conversion request failed (${response.status}).`);
  }
  const payload = (await response.json()) as { job_id: string };
  return payload.job_id;
}

export async function fetchConversionJob(apiBaseUrl: string, jobId: string): Promise<ConversionJobStatus> {
  const response = await fetch(`${apiBaseUrl}/api/jobs/${jobId}`);
  if (!response.ok) {
    throw new Error(`Could not read conversion job status (${response.status}).`);
  }
  return (await response.json()) as ConversionJobStatus;
}

export function conversionAudioUrl(apiBaseUrl: string, token: string): string {
  return `${apiBaseUrl}/api/convert/${token}/audio`;
}

export function conversionWaveformUrl(apiBaseUrl: string, token: string): string {
  return `${apiBaseUrl}/api/convert/${token}/waveform`;
}

export async function deleteConversion(apiBaseUrl: string, token: string): Promise<void> {
  await fetch(`${apiBaseUrl}/api/convert/${token}`, { method: "DELETE" }).catch(() => undefined);
}
