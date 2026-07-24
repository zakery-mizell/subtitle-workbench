import type { Caption } from "../types";

const TIMESTAMP_RE =
  /(\d{2}):(\d{2}):(\d{2}),(\d{3})\s+-->\s+(\d{2}):(\d{2}):(\d{2}),(\d{3})/;
const SPEAKER_LINE_RE = /^\s*([^:\n]{1,80}?)\s*:\s*$/;
// Inline `NAME: dialogue`. The name charset excludes most punctuation so that
// sentences like "Warning: do not open" rarely reach the candidate stage, and the
// qualification pass below throws out the ones that still do.
const INLINE_SPEAKER_RE = /^\s*([A-Za-z0-9 _.'()\-]{1,40}?):\s+(\S.*)$/;

interface ParsedCue {
  start: number;
  end: number;
  speakerName: string | null;
  lines: string[];
  inline: { name: string; remainder: string } | null;
}

export function parseSrt(text: string): Caption[] {
  const cues = parseCues(text);
  const inlineSpeakers = qualifiedInlineSpeakerNames(cues);

  return cues.map((cue, index) => {
    const inline = cue.inline && inlineSpeakers.has(cue.inline.name) ? cue.inline : null;
    const lines = inline ? [inline.remainder, ...cue.lines.slice(1)] : cue.lines;

    return {
      id: `import-${index}`,
      start: cue.start,
      end: cue.end,
      speaker_id: null,
      speaker_name: cue.speakerName ?? inline?.name ?? null,
      lines: lines.length ? lines : [""],
      word_ids: [],
      blank_after: false,
    };
  });
}

function parseCues(text: string): ParsedCue[] {
  const blocks = text
    .replace(/\r\n/g, "\n")
    .trim()
    .split(/\n{2,}/)
    .map((block) => block.trim())
    .filter(Boolean);

  const cues: ParsedCue[] = [];
  for (const block of blocks) {
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
    const inlineMatch = speakerName ? null : visibleLines[0]?.match(INLINE_SPEAKER_RE);
    const inlineName = inlineMatch?.[1]?.trim() || null;

    cues.push({
      start: toSeconds(match[1], match[2], match[3], match[4]),
      end: toSeconds(match[5], match[6], match[7], match[8]),
      speakerName,
      lines: visibleLines.length ? visibleLines : [""],
      inline: inlineName && inlineMatch?.[2] ? { name: inlineName, remainder: inlineMatch[2] } : null,
    });
  }
  return cues;
}

// A lone inline candidate seen once is far more likely to be ordinary dialogue than a
// speaker label, so require repetition or the company of a second distinct name.
function qualifiedInlineSpeakerNames(cues: ParsedCue[]): Set<string> {
  const counts = new Map<string, number>();
  for (const cue of cues) {
    if (cue.inline) {
      counts.set(cue.inline.name, (counts.get(cue.inline.name) ?? 0) + 1);
    }
  }

  if (counts.size >= 2) {
    return new Set(counts.keys());
  }
  return new Set(
    Array.from(counts.entries())
      .filter(([, count]) => count >= 2)
      .map(([name]) => name),
  );
}

function toSeconds(hours: string, minutes: string, seconds: string, millis: string): number {
  return (
    Number(hours) * 3600 +
    Number(minutes) * 60 +
    Number(seconds) +
    Number(millis) / 1000
  );
}
