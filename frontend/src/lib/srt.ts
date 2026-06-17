import type { Caption } from "../types";

const TIMESTAMP_RE =
  /(\d{2}):(\d{2}):(\d{2}),(\d{3})\s+-->\s+(\d{2}):(\d{2}):(\d{2}),(\d{3})/;
const SPEAKER_LINE_RE = /^\s*([^:\n]{1,80}?)\s*:\s*$/;

export function parseSrt(text: string): Caption[] {
  const blocks = text
    .replace(/\r\n/g, "\n")
    .trim()
    .split(/\n{2,}/)
    .map((block) => block.trim())
    .filter(Boolean);

  const captions: Caption[] = [];
  for (const [index, block] of blocks.entries()) {
    const lines = block.split("\n");
    const firstLine = lines[0]?.trim() ?? "";
    const secondLine = lines[1]?.trim() ?? "";
    const timestampLine = TIMESTAMP_RE.test(firstLine) ? firstLine : secondLine;
    const bodyLines = timestampLine === firstLine ? lines.slice(1) : TIMESTAMP_RE.test(secondLine) ? lines.slice(2) : lines.slice(1);
    const match = timestampLine.match(TIMESTAMP_RE);
    if (!match) {
      continue;
    }

    const normalizedBodyLines = bodyLines.length ? bodyLines : [""];
    const speakerMatch = normalizedBodyLines.length > 1 ? normalizedBodyLines[0]?.match(SPEAKER_LINE_RE) : null;
    const speakerName = speakerMatch?.[1]?.trim() || null;
    const visibleLines = speakerMatch ? normalizedBodyLines.slice(1) : normalizedBodyLines;

    captions.push({
      id: `import-${index}`,
      start: toSeconds(match[1], match[2], match[3], match[4]),
      end: toSeconds(match[5], match[6], match[7], match[8]),
      speaker_id: null,
      speaker_name: speakerName,
      lines: visibleLines.length ? visibleLines : [""],
      word_ids: [],
      blank_after: false,
    });
  }
  return captions;
}

function toSeconds(hours: string, minutes: string, seconds: string, millis: string): number {
  return (
    Number(hours) * 3600 +
    Number(minutes) * 60 +
    Number(seconds) +
    Number(millis) / 1000
  );
}
