import type { Caption, GuideBlock, Speaker } from "../types";
import { formatSrtTimestamp, normalizeToFps } from "./time";

function resolveSpeakerName(caption: Caption, speakerMap: Map<number, string>): string | null {
  if (caption.speaker_id !== null) {
    return speakerMap.get(caption.speaker_id) ?? caption.speaker_name ?? `Speaker ${caption.speaker_id + 1}`;
  }

  return caption.speaker_name?.trim() || null;
}

function speakerAllowsAttribution(caption: Caption, speakerConfig: Map<number, Speaker>): boolean {
  if (caption.speaker_id === null) {
    return caption.speaker_name?.trim().length ? true : false;
  }

  return speakerConfig.get(caption.speaker_id)?.show_attribution !== false;
}

function shouldIncludeSpeakerLabels(captions: Caption[], speakers: Speaker[]): boolean {
  const speakerMap = new Map(speakers.map((speaker) => [speaker.id, speaker.name]));
  const speakerConfig = new Map(speakers.map((speaker) => [speaker.id, speaker]));
  const labels = new Set(
    captions
      .filter((caption) => caption.lines.some((line) => line.trim()))
      .filter((caption) => speakerAllowsAttribution(caption, speakerConfig))
      .map((caption) => resolveSpeakerName(caption, speakerMap))
      .filter((label): label is string => Boolean(label)),
  );

  return (speakers.length > 1 && labels.size > 0) || labels.size > 1;
}

function exportCaptionEnd(captions: Caption[], index: number, extendToNextOnExport: boolean): number {
  const caption = captions[index];
  const next = captions[index + 1];
  if (!next) {
    return caption.end;
  }

  const targetEnd = extendToNextOnExport ? next.start : caption.end;
  return Math.max(caption.start, Math.min(targetEnd, next.start));
}

function normalizeExportRange(
  start: number,
  end: number,
  nextStart: number | null,
  normalizeTo30Fps: boolean,
): { start: number; end: number } {
  if (!normalizeTo30Fps) {
    return {
      start: Math.max(0, start),
      end: Math.max(start, end),
    };
  }

  const normalizedStart = normalizeToFps(start, 30, "floor");
  const normalizedNextStart = nextStart === null ? null : normalizeToFps(nextStart, 30, "floor");
  let normalizedEnd = normalizeToFps(end, 30, "ceil");

  if (normalizedNextStart !== null) {
    normalizedEnd = Math.min(normalizedEnd, normalizedNextStart);
  }
  if (normalizedEnd < normalizedStart) {
    normalizedEnd = normalizedStart;
  }

  return {
    start: normalizedStart,
    end: normalizedEnd,
  };
}

export function captionsToSrt(
  captions: Caption[],
  speakers: Speaker[],
  extendToNextOnExport = false,
  normalizeTo30Fps = false,
): string {
  const visibleCaptions = captions
    .filter((caption) => caption.lines.some((line) => line.trim()))
    .sort((left, right) => left.start - right.start || left.end - right.end);
  const speakerMap = new Map(speakers.map((speaker) => [speaker.id, speaker.name]));
  const speakerConfig = new Map(speakers.map((speaker) => [speaker.id, speaker]));
  const includeSpeakerLabels = shouldIncludeSpeakerLabels(visibleCaptions, speakers);

  return visibleCaptions
    .map((caption, index) => {
      const body = caption.lines.join("\n").trim();
      const speakerLabel =
        includeSpeakerLabels && speakerAllowsAttribution(caption, speakerConfig)
          ? resolveSpeakerName(caption, speakerMap)
          : null;
      const next = visibleCaptions[index + 1] ?? null;
      const normalizedRange = normalizeExportRange(
        caption.start,
        exportCaptionEnd(visibleCaptions, index, extendToNextOnExport),
        next?.start ?? null,
        normalizeTo30Fps,
      );
      const exportBody = speakerLabel ? `${speakerLabel}:\n${body}` : body;
      return `${index + 1}\n${formatSrtTimestamp(normalizedRange.start)} --> ${formatSrtTimestamp(normalizedRange.end)}\n${exportBody}\n`;
    })
    .join("\n");
}

export function guideToSrt(blocks: GuideBlock[], normalizeTo30Fps = false): string {
  return blocks
    .map((block, index) => {
      const normalizedRange = normalizeExportRange(block.start, block.end, null, normalizeTo30Fps);
      return `${index + 1}\n${formatSrtTimestamp(normalizedRange.start)} --> ${formatSrtTimestamp(normalizedRange.end)}\n${block.label}\n`;
    })
    .join("\n");
}

export function captionsToTranscriptText(captions: Caption[], speakers: Speaker[]): string {
  const speakerMap = new Map(speakers.map((speaker) => [speaker.id, speaker.name]));
  const grouped = captions
    .filter((caption) => caption.lines.some((line) => line.trim()))
    .reduce<
      Array<{
        speakerId: number | null;
        speakerName: string;
        chunks: string[];
      }>
    >((acc, caption) => {
      const speakerName =
        caption.speaker_id !== null
          ? speakerMap.get(caption.speaker_id) ?? caption.speaker_name ?? `Speaker ${caption.speaker_id + 1}`
          : caption.speaker_name ?? "Speaker";
      const text = caption.lines.join(" ").replace(/\s+/g, " ").trim();
      if (!text) {
        return acc;
      }
      const previous = acc[acc.length - 1];
      if (previous && previous.speakerId === caption.speaker_id) {
        previous.chunks.push(text);
      } else {
        acc.push({ speakerId: caption.speaker_id, speakerName, chunks: [text] });
      }
      return acc;
    }, []);

  return grouped.map((entry) => `${entry.speakerName}\n${entry.chunks.join(" ")}`).join("\n\n");
}

function normalizeFilenamePart(value: string, fallback: string, maxLength: number): string {
  const normalized = value
    .normalize("NFKD")
    .replace(/[\u0300-\u036f]/g, "")
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-+|-+$/g, "");

  if (!normalized) {
    return fallback;
  }

  const trimmed = normalized.slice(0, maxLength).replace(/-+$/g, "");
  return trimmed || fallback;
}

function buildSpeakerLabel(speakers: Speaker[]): string {
  const uniqueNames = Array.from(
    new Map(
      speakers
        .map((speaker) => speaker.name.trim())
        .filter(Boolean)
        .map((name) => [name.toLowerCase(), name]),
    ).values(),
  );

  if (!uniqueNames.length) {
    return "speaker";
  }

  const visible = uniqueNames.slice(0, 2).map((name) => normalizeFilenamePart(name, "speaker", 10));
  const extraCount = uniqueNames.length - visible.length;
  return extraCount > 0 ? `${visible.join("-")}-plus${extraCount}` : visible.join("-");
}

export function buildExportFilename(
  audioFilename: string | null | undefined,
  speakers: Speaker[],
  kind: "subtitles" | "edit-guide" | "transcript",
  extension: "srt" | "txt",
): string {
  const stem = normalizeFilenamePart((audioFilename ?? "").replace(/\.[^.]+$/, ""), "audio", 32);
  const speakerLabel = buildSpeakerLabel(speakers);
  return `${stem}__${speakerLabel}__${kind}.${extension}`;
}
