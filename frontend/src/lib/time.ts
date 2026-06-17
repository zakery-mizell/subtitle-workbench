export function formatClock(seconds: number): string {
  if (!Number.isFinite(seconds)) {
    return "00:00.000";
  }
  const totalMs = Math.max(0, Math.round(seconds * 1000));
  const hours = Math.floor(totalMs / 3600000);
  const minutes = Math.floor((totalMs % 3600000) / 60000);
  const secs = Math.floor((totalMs % 60000) / 1000);
  const ms = totalMs % 1000;
  if (hours > 0) {
    return `${pad(hours)}:${pad(minutes)}:${pad(secs)}.${padMs(ms)}`;
  }
  return `${pad(minutes)}:${pad(secs)}.${padMs(ms)}`;
}

export function formatSrtTimestamp(seconds: number): string {
  const totalMs = Math.max(0, Math.round(seconds * 1000));
  const hours = Math.floor(totalMs / 3600000);
  const minutes = Math.floor((totalMs % 3600000) / 60000);
  const secs = Math.floor((totalMs % 60000) / 1000);
  const ms = totalMs % 1000;
  return `${pad(hours)}:${pad(minutes)}:${pad(secs)},${padMs(ms)}`;
}

export function normalizeToFps(seconds: number, fps: number, mode: "nearest" | "floor" | "ceil" = "nearest"): number {
  if (!Number.isFinite(seconds) || !Number.isFinite(fps) || fps <= 0) {
    return seconds;
  }

  const frame = seconds * fps;
  const normalizedFrame =
    mode === "floor"
      ? Math.floor(frame)
      : mode === "ceil"
        ? Math.ceil(frame)
        : Math.round(frame);
  return normalizedFrame / fps;
}

function pad(value: number): string {
  return String(value).padStart(2, "0");
}

function padMs(value: number): string {
  return String(value).padStart(3, "0");
}
