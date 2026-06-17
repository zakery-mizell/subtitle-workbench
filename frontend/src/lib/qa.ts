import type { Caption, WordToken } from "../types";
import type { CaptionGlossaryMatch } from "./glossary";

export type QaSeverity = "error" | "warning" | "info";

export interface QaIssue {
  id: string;
  captionIndex: number;
  severity: QaSeverity;
  code: string;
  message: string;
  excerpt: string;
  start: number;
  end: number;
}

export interface QaReport {
  summary: {
    captionCount: number;
    issueCount: number;
    errorCount: number;
    warningCount: number;
    infoCount: number;
    flaggedCaptionCount: number;
  };
  issues: QaIssue[];
}

function plainCaptionText(caption: Caption): string {
  return caption.lines.join(" ").replace(/\s+/g, " ").trim();
}

function formatTimestamp(seconds: number): string {
  const safe = Math.max(0, seconds);
  const hours = Math.floor(safe / 3600);
  const minutes = Math.floor((safe % 3600) / 60);
  const wholeSeconds = Math.floor(safe % 60);
  const millis = Math.round((safe - Math.floor(safe)) * 1000);
  const hh = String(hours).padStart(2, "0");
  const mm = String(minutes).padStart(2, "0");
  const ss = String(wholeSeconds).padStart(2, "0");
  const mmm = String(millis).padStart(3, "0");
  return `${hh}:${mm}:${ss},${mmm}`;
}

function truncate(text: string, maxLength = 110): string {
  return text.length <= maxLength ? text : `${text.slice(0, maxLength - 1)}…`;
}

export function buildQaReport(
  captions: Caption[],
  words: WordToken[],
  glossaryMatches: CaptionGlossaryMatch[],
): QaReport {
  const issues: QaIssue[] = [];
  const wordById = new Map(words.map((word) => [word.id, word]));

  captions.forEach((caption, captionIndex) => {
    const text = plainCaptionText(caption);
    const duration = Math.max(0.001, caption.end - caption.start);
    const maxLineLength = Math.max(...caption.lines.map((line) => line.length), 0);
    const cps = text.length / duration;
    const lowConfidenceCount = caption.word_ids.reduce(
      (count, wordId) => count + (wordById.get(wordId)?.low_confidence ? 1 : 0),
      0,
    );
    const glossaryMatch = glossaryMatches[captionIndex];

    if (caption.lines.length > 2) {
      issues.push({
        id: `caption-${captionIndex}-line-count`,
        captionIndex,
        severity: "error",
        code: "line_count",
        message: `Uses ${caption.lines.length} subtitle lines.`,
        excerpt: truncate(text),
        start: caption.start,
        end: caption.end,
      });
    }

    if (maxLineLength > 55) {
      issues.push({
        id: `caption-${captionIndex}-line-hard-cap`,
        captionIndex,
        severity: "error",
        code: "line_length_hard",
        message: `Line length ${maxLineLength} exceeds the hard cap of 55.`,
        excerpt: truncate(text),
        start: caption.start,
        end: caption.end,
      });
    } else if (maxLineLength > 42) {
      issues.push({
        id: `caption-${captionIndex}-line-target`,
        captionIndex,
        severity: "warning",
        code: "line_length_target",
        message: `Line length ${maxLineLength} exceeds the target of 42.`,
        excerpt: truncate(text),
        start: caption.start,
        end: caption.end,
      });
    }

    if (cps > 20) {
      issues.push({
        id: `caption-${captionIndex}-cps-hard`,
        captionIndex,
        severity: "error",
        code: "reading_speed_hard",
        message: `Reading speed ${cps.toFixed(1)} cps exceeds 20.`,
        excerpt: truncate(text),
        start: caption.start,
        end: caption.end,
      });
    } else if (cps > 17) {
      issues.push({
        id: `caption-${captionIndex}-cps-target`,
        captionIndex,
        severity: "warning",
        code: "reading_speed_target",
        message: `Reading speed ${cps.toFixed(1)} cps exceeds the 17 cps target.`,
        excerpt: truncate(text),
        start: caption.start,
        end: caption.end,
      });
    }

    if (duration < 1 && text.length > 18) {
      issues.push({
        id: `caption-${captionIndex}-duration-short`,
        captionIndex,
        severity: "warning",
        code: "duration_short",
        message: `Caption duration ${duration.toFixed(2)}s is very short for this amount of text.`,
        excerpt: truncate(text),
        start: caption.start,
        end: caption.end,
      });
    } else if (duration > 7) {
      issues.push({
        id: `caption-${captionIndex}-duration-long`,
        captionIndex,
        severity: "warning",
        code: "duration_long",
        message: `Caption duration ${duration.toFixed(2)}s is longer than 7 seconds.`,
        excerpt: truncate(text),
        start: caption.start,
        end: caption.end,
      });
    }

    if (lowConfidenceCount > 0) {
      issues.push({
        id: `caption-${captionIndex}-low-confidence`,
        captionIndex,
        severity: "warning",
        code: "low_confidence",
        message: `${lowConfidenceCount} low-confidence word${lowConfidenceCount === 1 ? "" : "s"} in this caption.`,
        excerpt: truncate(text),
        start: caption.start,
        end: caption.end,
      });
    }

    if (glossaryMatch?.fuzzyTerms.length) {
      issues.push({
        id: `caption-${captionIndex}-glossary`,
        captionIndex,
        severity: "warning",
        code: "glossary_mismatch",
        message: `Possible glossary mismatch: ${glossaryMatch.fuzzyTerms.slice(0, 3).join(", ")}.`,
        excerpt: truncate(text),
        start: caption.start,
        end: caption.end,
      });
    }
  });

  const flaggedCaptionCount = new Set(issues.map((issue) => issue.captionIndex)).size;
  const errorCount = issues.filter((issue) => issue.severity === "error").length;
  const warningCount = issues.filter((issue) => issue.severity === "warning").length;
  const infoCount = issues.filter((issue) => issue.severity === "info").length;

  return {
    summary: {
      captionCount: captions.length,
      issueCount: issues.length,
      errorCount,
      warningCount,
      infoCount,
      flaggedCaptionCount,
    },
    issues,
  };
}

export function formatQaReport(report: QaReport, captions: Caption[]): string {
  const lines = [
    "Subtitle QA Report",
    "",
    `Captions: ${report.summary.captionCount}`,
    `Flagged captions: ${report.summary.flaggedCaptionCount}`,
    `Issues: ${report.summary.issueCount}`,
    `Errors: ${report.summary.errorCount}`,
    `Warnings: ${report.summary.warningCount}`,
    `Info: ${report.summary.infoCount}`,
    "",
  ];

  if (!report.issues.length) {
    lines.push("No QA issues found.");
    return lines.join("\n");
  }

  for (const issue of report.issues) {
    const caption = captions[issue.captionIndex];
    lines.push(
      `[${issue.severity.toUpperCase()}] Subtitle ${issue.captionIndex + 1} ${formatTimestamp(caption?.start ?? issue.start)} -> ${formatTimestamp(caption?.end ?? issue.end)}`,
      issue.message,
      issue.excerpt,
      "",
    );
  }

  return lines.join("\n");
}
