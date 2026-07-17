export type RestoreOutputFormat = "flac" | "wav" | "mp3" | "aac" | "opus";

export interface RestoreParams {
  output_format: RestoreOutputFormat;
  chunk_sec: number;
  overlap_sec: number;
  rep_penalty: number;
}

export function defaultRestoreParams(): RestoreParams {
  return { output_format: "flac", chunk_sec: 2.5, overlap_sec: 0.4, rep_penalty: 1.3 };
}

export interface RestoreResult {
  token: string;
  filename: string;
  output_format: string;
  sample_rate: number;
  duration_sec: number;
  warnings: string[];
}

export interface RestoreJobStatus {
  id: string;
  status: "queued" | "running" | "done" | "error";
  stage: string;
  progress: number;
  message: string | null;
  error: string | null;
  result: RestoreResult | null;
}

export async function startRestoreJob(
  apiBaseUrl: string,
  audioFile: File,
  params: RestoreParams,
): Promise<string> {
  const formData = new FormData();
  formData.append("audio", audioFile);
  formData.append("params_json", JSON.stringify(params));
  const response = await fetch(`${apiBaseUrl}/api/restore`, { method: "POST", body: formData });
  if (!response.ok) {
    const detail = await response.json().catch(() => null);
    throw new Error(detail?.detail ?? `Restore request failed (${response.status}).`);
  }
  const payload = (await response.json()) as { job_id: string };
  return payload.job_id;
}

export async function fetchRestoreJob(apiBaseUrl: string, jobId: string): Promise<RestoreJobStatus> {
  const response = await fetch(`${apiBaseUrl}/api/jobs/${jobId}`);
  if (!response.ok) {
    throw new Error(`Could not read restore job status (${response.status}).`);
  }
  return (await response.json()) as RestoreJobStatus;
}

export function restoreAudioUrl(apiBaseUrl: string, token: string): string {
  return `${apiBaseUrl}/api/restore/${token}/audio`;
}

export function restoreWaveformUrl(apiBaseUrl: string, token: string): string {
  return `${apiBaseUrl}/api/restore/${token}/waveform`;
}

export async function deleteRestore(apiBaseUrl: string, token: string): Promise<void> {
  await fetch(`${apiBaseUrl}/api/restore/${token}`, { method: "DELETE" }).catch(() => undefined);
}
