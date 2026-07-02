import { memo, useEffect, useMemo, useRef, useState, type CSSProperties, type KeyboardEvent } from "react";

import { buildExportFilename, captionsToSrt, guideToSrt } from "./lib/exporters";
import {
  appendGlossaryTerms,
  detectJargonCandidates,
  findCaptionGlossaryMatches,
  mergeVocabularyTexts,
  parseGlossaryTerms,
} from "./lib/glossary";
import { buildQaReport, formatQaReport } from "./lib/qa";
import { remapCaptions, remapGuideBlocks, remapWords } from "./lib/cuts";
import type { MasteringResult } from "./lib/mastering";
import { parseSrt } from "./lib/srt";
import { formatClock } from "./lib/time";
import MasteringPanel from "./MasteringPanel";
import type {
  BackendCapabilities,
  Caption,
  GuideBlock,
  GuideLabel,
  Paragraph,
  RetranscribeRangeResponse,
  SpeechSpan,
  Speaker,
  SpeakerAssignmentMode,
  TranscriptResponse,
  WaveformAnalysisResponse,
  WarningItem,
  WordToken,
} from "./types";

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL ?? "http://localhost:8000";
const MODEL_OPTIONS = ["tiny", "base", "small", "medium", "large", "large-v2", "large-v3", "turbo"];
const MATCH_LOOKAHEAD = 3;
const REMATCH_LOOKAHEAD = 16;
const REMATCH_SEARCH_WINDOWS_SECONDS = [8, 20, 45];
const STRONG_REMATCH_RATIO = 0.75;
const MIN_REMATCH_RATIO = 0.35;
const TARGET_CAPTION_LINE_LENGTH = 42;
const MAX_CAPTION_LINE_LENGTH = 55;
const ATTRIBUTION_LINE_MAX_LENGTH = 32;
const SENTENCE_END_RE = /[.!?]["')\]]?$/;
const CLAUSE_END_RE = /[,;:]["')\]]?$/;
const TITLECASE_TOKEN_RE = /^[A-Z][A-Za-z'\u2019-]+$/;
const HONORIFIC_TOKEN_RE = /^(mr|mrs|ms|dr|prof|sir|lady|lord|st)\.?$/i;
const DOUBLE_QUOTE_RE = /["\u201C\u201D]/g;
const TITLE_CONNECTORS = new Set(["a", "an", "and", "for", "in", "of", "on", "the", "to"]);
const ENTITY_INTRODUCERS = new Set(["called", "named", "titled"]);
const DISCOURSE_LEAD_INS = new Set([
  "so",
  "well",
  "but",
  "and",
  "now",
  "then",
  "yes",
  "no",
  "okay",
  "ok",
  "look",
  "listen",
  "anyway",
  "actually",
  "basically",
  "still",
  "frankly",
  "honestly",
  "you know",
  "i mean",
]);
const WEAK_LINE_STARTS = new Set([
  "and",
  "but",
  "or",
  "so",
  "because",
  "if",
  "then",
  "than",
  "that",
  "which",
  "who",
  "when",
  "where",
  "to",
  "of",
  "for",
  "with",
  "a",
  "an",
  "the",
]);
const WEAK_LINE_ENDS = new Set([
  "a",
  "an",
  "the",
  "and",
  "but",
  "or",
  "so",
  "to",
  "of",
  "for",
  "with",
  "at",
  "by",
  "from",
  "in",
  "on",
  "if",
  "than",
  "that",
  "which",
  "who",
  "when",
  "where",
  "is",
  "are",
  "was",
  "were",
  "be",
  "been",
  "being",
]);
const SPEAKER_ASSIGNMENT_OPTIONS: Array<{ value: SpeakerAssignmentMode; label: string }> = [
  { value: "segment", label: "Segment (stable)" },
  { value: "word", label: "Word (tighter switches)" },
];
const AUTOSAVE_STORAGE_KEY = "subtitle-workbench:autosave";
const AUTOSAVE_STORAGE_VERSION = 5;
const PROJECT_FILE_VERSION = 1;
const TEXT_EDIT_CHECKPOINT_MS = 800;
const AUTOSAVE_DELAY_MS = 700;
const SIDE_PANEL_TABS = [
  { id: "guide", label: "Guide" },
  { id: "jargon", label: "Jargon" },
  { id: "qa", label: "QA" },
  { id: "master", label: "Master" },
] as const;
const DEFAULT_GUIDE_PANEL_COLLAPSED = true;
const WAVEFORM_START_PAD_SECONDS = 0.03;
const WAVEFORM_END_PAD_SECONDS = 0.08;
const WAVEFORM_MAX_EDGE_MOVE_SECONDS = 1.25;
const WAVEFORM_MIN_CAPTION_SECONDS = 0.18;
const WAVEFORM_MIN_EDGE_DELTA_SECONDS = 0.015;

type ViewMode = "transcript" | "subtitles";
type SelectionKind = "paragraph" | "caption";
type SidePanelTab = (typeof SIDE_PANEL_TABS)[number]["id"];
type LegacyPersistedWorkspace = Partial<PersistedWorkspace> & { hotwords?: unknown; version?: unknown };
type SpeakerTimelineEventKind = "switch" | "tight_handoff" | "overlap";

interface EditorState {
  captions: Caption[];
  guideBlocks: GuideBlock[];
  speakers: Speaker[];
  paragraphs: Paragraph[];
}

interface WorkspaceState {
  editor: EditorState;
  words: WordToken[];
  warnings: WarningItem[];
  language: string | null;
}

interface HistoryState {
  past: WorkspaceState[];
  present: WorkspaceState | null;
  future: WorkspaceState[];
}

interface CommitOptions {
  wordSource?: WordToken[];
  syncCaptionTiming?: boolean;
  transformWords?: (words: WordToken[]) => WordToken[];
  warnings?: WarningItem[];
  language?: string | null;
}

interface CaptionWordSyncOptions {
  mode?: "global" | "time_anchored";
  preserveTiming?: boolean;
}

interface PersistedWorkspace {
  version: number;
  session: TranscriptResponse | null;
  editor: EditorState | null;
  model: string;
  speakerCount: number;
  speakerInputs: Speaker[];
  speakerAssignmentMode: SpeakerAssignmentMode;
  glossaryText: string;
  skipCuts: boolean;
  clickToPlay: boolean;
  followPlayback: boolean;
  showLineGuides: boolean;
  showTimingHighlights: boolean;
  viewMode: ViewMode;
  sidePanelTab: SidePanelTab;
  isGuidePanelCollapsed: boolean;
  extendCaptionsOnExport: boolean;
  normalizeExportTimingTo30Fps: boolean;
  showSpeakerAttributionOptions: boolean;
  removeDisfluencies: boolean;
  acknowledgedLowConfidenceWordIds: string[];
}

interface ProjectAudioPayload {
  name: string;
  type: string;
  data_url: string;
}

interface ProjectFile {
  format: "subtitle-workbench-project";
  version: number;
  workspace: PersistedWorkspace;
  audio: ProjectAudioPayload | null;
}

interface BlockSelection {
  kind: SelectionKind;
  index: number;
  start: number;
  end: number;
  text: string;
}

interface MatchedFragment {
  key: string;
  text: string;
  charStart: number;
  charEnd: number;
  word: WordToken | null;
}

interface RetranscribeTarget {
  start: number;
  end: number;
  label: string;
}

interface FocusRequest {
  token: number;
  caret: number;
}

interface SpeakerTimelineEvent {
  id: string;
  kind: SpeakerTimelineEventKind;
  time: number;
  start: number;
  end: number;
  label: string;
  captionIndex?: number;
}

interface WaveformAlignmentResult {
  captions: Caption[];
  edgeAdjustmentCount: number;
  captionAdjustmentCount: number;
}

interface TimedTextEditorProps {
  value: string;
  wordIds: string[];
  lookup: Map<string, WordToken>;
  currentTime: number;
  showTimingHighlights?: boolean;
  className?: string;
  commitMode?: "immediate" | "blur";
  minHeight?: number;
  showLineGuides?: boolean;
  fallbackTime: number;
  autoPlayOnSeek?: boolean;
  focusRequest?: FocusRequest | null;
  acknowledgedWordIds?: Set<string>;
  onChange: (value: string) => void;
  onSeek: (time: number, options?: { play?: boolean }) => void;
  onSelectionChange: (start: number, end: number) => void;
  onAcknowledgeWords?: (wordIds: string[]) => void;
  onFocus?: () => void;
  onBlur?: () => void;
  onUndo?: () => void;
  onRedo?: () => void;
  onKeyDown?: (event: KeyboardEvent<HTMLTextAreaElement>) => void;
}

function buildDefaultSpeakers(): Speaker[] {
  return [{ id: 0, name: "Speaker 1", show_attribution: true }];
}

function normalizeSpeaker(speaker: Speaker, fallbackId: number): Speaker {
  return {
    id: Number.isFinite(speaker.id) ? speaker.id : fallbackId,
    name: speaker.name || `Speaker ${fallbackId + 1}`,
    show_attribution: speaker.show_attribution !== false,
  };
}

function normalizeSpeakers(speakers: Speaker[]): Speaker[] {
  return speakers.map((speaker, index) => normalizeSpeaker(speaker, index));
}

function persistWorkspace(snapshot: PersistedWorkspace) {
  try {
    window.localStorage.setItem(AUTOSAVE_STORAGE_KEY, JSON.stringify(snapshot));
    return;
  } catch {
    if (!snapshot.session) {
      return;
    }
  }

  try {
    window.localStorage.setItem(
      AUTOSAVE_STORAGE_KEY,
      JSON.stringify({
        ...snapshot,
        session: {
          ...snapshot.session,
          words: [],
        },
      } satisfies PersistedWorkspace),
    );
  } catch {
    // Ignore autosave failures. Manual export still works.
  }
}

function cloneWords(words: WordToken[]): WordToken[] {
  return words.map((word) => ({ ...word }));
}

function cloneWarnings(warnings: WarningItem[]): WarningItem[] {
  return warnings.map((warning) => ({ ...warning }));
}

function cloneEditorState(state: EditorState): EditorState {
  return {
    captions: state.captions.map((caption) => ({ ...caption, lines: [...caption.lines], word_ids: [...caption.word_ids] })),
    guideBlocks: state.guideBlocks.map((block) => ({ ...block })),
    speakers: normalizeSpeakers(state.speakers),
    paragraphs: state.paragraphs.map((paragraph) => ({
      ...paragraph,
      word_ids: [...paragraph.word_ids],
      caption_ids: paragraph.caption_ids ? [...paragraph.caption_ids] : undefined,
    })),
  };
}

function cloneWorkspaceState(state: WorkspaceState): WorkspaceState {
  return {
    editor: cloneEditorState(state.editor),
    words: cloneWords(state.words),
    warnings: cloneWarnings(state.warnings),
    language: state.language,
  };
}

function buildWorkspaceState(
  editor: EditorState,
  words: WordToken[],
  warnings: WarningItem[],
  language: string | null,
): WorkspaceState {
  return {
    editor: normalizeEditorState(editor),
    words: cloneWords(words),
    warnings: cloneWarnings(warnings),
    language,
  };
}

function buildWorkspaceFromSession(session: TranscriptResponse): WorkspaceState {
  return buildWorkspaceState(
    {
      captions: session.captions,
      guideBlocks: session.guide_blocks,
      speakers: normalizeSpeakers(session.speakers),
      paragraphs: session.paragraphs,
    },
    session.words,
    session.warnings,
    session.language,
  );
}

function buildSessionSnapshot(session: TranscriptResponse | null, workspace: WorkspaceState | null): TranscriptResponse | null {
  if (!session) {
    return null;
  }

  if (!workspace) {
    return {
      ...session,
      speakers: normalizeSpeakers(session.speakers),
      words: cloneWords(session.words),
      paragraphs: session.paragraphs.map((paragraph) => ({
        ...paragraph,
        word_ids: [...paragraph.word_ids],
        caption_ids: paragraph.caption_ids ? [...paragraph.caption_ids] : undefined,
      })),
      captions: session.captions.map((caption) => ({ ...caption, lines: [...caption.lines], word_ids: [...caption.word_ids] })),
      guide_blocks: session.guide_blocks.map((block) => ({ ...block })),
      warnings: cloneWarnings(session.warnings),
    };
  }

  return {
    ...session,
    speakers: normalizeSpeakers(workspace.editor.speakers),
    words: cloneWords(workspace.words),
    paragraphs: workspace.editor.paragraphs.map((paragraph) => ({
      ...paragraph,
      word_ids: [...paragraph.word_ids],
      caption_ids: paragraph.caption_ids ? [...paragraph.caption_ids] : undefined,
    })),
    captions: workspace.editor.captions.map((caption) => ({ ...caption, lines: [...caption.lines], word_ids: [...caption.word_ids] })),
    guide_blocks: workspace.editor.guideBlocks.map((block) => ({ ...block })),
    warnings: cloneWarnings(workspace.warnings),
    language: workspace.language,
  };
}

function captionValue(caption: Caption): string {
  return caption.lines.join("\n");
}

function plainCaptionText(caption: Caption): string {
  return caption.lines.join(" ").replace(/\s+/g, " ").trim();
}

function normalizeEditableText(text: string): string {
  return text
    .replace(/\s+([,.;:!?])/g, "$1")
    .replace(/\(\s+/g, "(")
    .replace(/\s+\)/g, ")")
    .replace(/\s+/g, " ")
    .trim();
}

function stripCaptionToken(text: string): string {
  return text
    .replace(/^[\"'(\[\u2018\u201C]+/, "")
    .replace(/[\"')\].,;:!?\u2019\u201D]+$/, "");
}

function looksLikeTitleToken(text: string): boolean {
  return TITLECASE_TOKEN_RE.test(stripCaptionToken(text));
}

function isHonorificToken(text: string): boolean {
  return HONORIFIC_TOKEN_RE.test(stripCaptionToken(text));
}

function splitCrossesQuotedSpan(left: string, right: string): boolean {
  const leftQuotes = left.match(DOUBLE_QUOTE_RE) ?? [];
  const rightQuotes = right.match(DOUBLE_QUOTE_RE) ?? [];
  return leftQuotes.length % 2 === 1 && rightQuotes.length > 0;
}

function startsSentence(text: string): boolean {
  return /^[\"'(\[\u2018\u201C]?[A-Z]/.test(text);
}

function isNameOrTitleBoundary(lastWord: string, firstWord: string): boolean {
  const leftClean = stripCaptionToken(lastWord);
  const rightClean = stripCaptionToken(firstWord);
  const leftNormalized = normalizeToken(leftClean);
  const rightNormalized = normalizeToken(rightClean);

  if (!leftClean || !rightClean) {
    return false;
  }

  if (isHonorificToken(leftClean) && looksLikeTitleToken(rightClean)) {
    return true;
  }

  if (looksLikeTitleToken(leftClean) && looksLikeTitleToken(rightClean)) {
    return true;
  }

  if (looksLikeTitleToken(leftClean) && TITLE_CONNECTORS.has(rightNormalized)) {
    return true;
  }

  if (TITLE_CONNECTORS.has(leftNormalized) && looksLikeTitleToken(rightClean)) {
    return true;
  }

  if (ENTITY_INTRODUCERS.has(leftNormalized) && looksLikeTitleToken(rightClean)) {
    return true;
  }

  return false;
}

function normalizedPhrase(words: string[]): string {
  return words.map((word) => normalizeToken(word)).filter(Boolean).join(" ");
}

function isWeakCommaLeadIn(leftWords: string[], leftEndsClause: boolean): boolean {
  if (!leftEndsClause) {
    return false;
  }

  const normalizedLeft = normalizedPhrase(leftWords);
  if (!normalizedLeft) {
    return false;
  }

  if (leftWords.length <= 2 && DISCOURSE_LEAD_INS.has(normalizedLeft)) {
    return true;
  }

  return leftWords.length <= 2 && DISCOURSE_LEAD_INS.has(normalizeToken(leftWords[0] ?? ""));
}

function captionSplitScore(left: string, right: string, targetLineLength: number, hardCap: number): number {
  const leftLen = left.length;
  const rightLen = right.length;
  const leftWords = left.split(" ");
  const rightWords = right.split(" ");
  const lastWord = leftWords[leftWords.length - 1];
  const rightLeadWord = rightWords[0];
  const firstWord = normalizeToken(rightLeadWord);
  const lastNormalized = normalizeToken(lastWord);
  const leftEndsSentence = SENTENCE_END_RE.test(lastWord) && startsSentence(rightLeadWord);
  const leftEndsClause = CLAUSE_END_RE.test(lastWord);
  const shorter = Math.min(leftLen, rightLen);
  const longer = Math.max(leftLen, rightLen);
  const lineBalanceRatio = longer > 0 ? shorter / longer : 1;
  const weakCommaLeadIn = isWeakCommaLeadIn(leftWords, leftEndsClause);
  const midpoint = (leftLen + rightLen) / 2;
  let score = Math.abs(leftLen - rightLen) * 2.2;

  score += Math.abs(leftWords.length - rightWords.length) * 1.2;
  score += Math.max(0, leftLen - targetLineLength) * 1.7;
  score += Math.max(0, rightLen - targetLineLength) * 1.7;
  score += Math.max(0, leftLen - hardCap) * 6;
  score += Math.max(0, rightLen - hardCap) * 6;

  if (leftLen < 16) {
    score += (16 - leftLen) * 8;
  }
  if (rightLen < 16) {
    score += (16 - rightLen) * 8;
  }
  if (leftWords.length < 3) {
    score += (3 - leftWords.length) * 48;
  }
  if (rightWords.length < 3) {
    score += (3 - rightWords.length) * 28;
  }
  if (lineBalanceRatio < 0.65) {
    score += (0.65 - lineBalanceRatio) * 160;
  }

  score += Math.abs(leftLen - midpoint) * 0.6;

  if (leftEndsSentence) {
    score -= 18;
  } else if (leftEndsClause && leftLen >= 20 && leftWords.length >= 4 && !weakCommaLeadIn) {
    score -= 8;
  }

  if (splitCrossesQuotedSpan(left, right)) {
    score += 44;
  }

  if (isNameOrTitleBoundary(lastWord, rightLeadWord)) {
    score += 52;
  }

  if (WEAK_LINE_ENDS.has(lastNormalized)) {
    score += 16;
  }
  if (WEAK_LINE_STARTS.has(firstWord)) {
    score += 14;
  }

  if (",.;:!?)]}\"'".includes(right[0] ?? "")) {
    score += 40;
  }
  if ("([{\"'".includes(left[left.length - 1] ?? "")) {
    score += 28;
  }

  if (weakCommaLeadIn) {
    score += 160;
  }
  if (leftLen >= 24 && leftLen <= targetLineLength && rightLen >= 24 && rightLen <= targetLineLength) {
    score -= 12;
  }

  return score;
}

function reflowCaptionText(text: string): string[] {
  const normalized = normalizeEditableText(text);
  if (!normalized) {
    return [""];
  }

  if (normalized.length <= TARGET_CAPTION_LINE_LENGTH) {
    return [normalized];
  }

  const words = normalized.split(" ");
  if (words.length < 2) {
    return [normalized];
  }

  let bestLines = [normalized];
  let bestScore = Number.POSITIVE_INFINITY;
  const hardCap = MAX_CAPTION_LINE_LENGTH;
  const candidateIndexes = words.slice(1).map((_, index) => index + 1);

  if (!candidateIndexes.length) {
    return [normalized];
  }

  for (const index of candidateIndexes) {
    const left = words.slice(0, index).join(" ").trim();
    const right = words.slice(index).join(" ").trim();
    if (!left || !right) {
      continue;
    }

    const score = captionSplitScore(left, right, TARGET_CAPTION_LINE_LENGTH, hardCap);

    if (score < bestScore) {
      bestScore = score;
      bestLines = [left, right];
    }
  }

  return bestLines.map((line) => normalizeEditableText(line));
}

function looksLikeStandaloneCaptionLead(line: string): boolean {
  const trimmed = line.trim();
  if (!trimmed || trimmed.length > ATTRIBUTION_LINE_MAX_LENGTH) {
    return false;
  }
  return trimmed.endsWith(":") || /^[(-\u2013\u2014]/.test(trimmed);
}

function normalizeCaptionLines(lines: string[]): string[] {
  const normalizedLines = lines
    .map((line) => normalizeEditableText(line))
    .filter(Boolean);

  if (!normalizedLines.length) {
    return [""];
  }

  if (normalizedLines.length === 1) {
    return reflowCaptionText(normalizedLines[0]);
  }

  const [firstLine, ...restLines] = normalizedLines;
  if (looksLikeStandaloneCaptionLead(firstLine)) {
    const body = normalizeEditableText(restLines.join(" "));
    const bodyLines = body ? reflowCaptionText(body) : [];
    return [firstLine, ...bodyLines].slice(0, 3);
  }

  if (normalizedLines.length <= 3 && normalizedLines.every((line) => line.length <= MAX_CAPTION_LINE_LENGTH)) {
    return normalizedLines;
  }

  return reflowCaptionText(normalizeEditableText(normalizedLines.join(" ")));
}

function paragraphsToTranscriptText(paragraphs: Paragraph[], speakers: Speaker[]): string {
  const speakerMap = new Map(speakers.map((speaker) => [speaker.id, speaker.name]));
  return paragraphs
    .map((paragraph) => {
      const speakerName =
        paragraph.speaker_id !== null
          ? speakerMap.get(paragraph.speaker_id) ?? paragraph.speaker_name ?? `Speaker ${paragraph.speaker_id + 1}`
          : paragraph.speaker_name ?? "Speaker";
      return `${speakerName}\n${paragraph.text.trim()}`;
    })
    .filter((block) => block.trim())
    .join("\n\n");
}

function mergeCaptionLines(left: string[], right: string[]): { lines: string[]; caret: number } {
  const leftLines = left.map((line) => line.trim()).filter(Boolean);
  const rightLines = right.map((line) => line.trim()).filter(Boolean);

  if (!leftLines.length && !rightLines.length) {
    return { lines: [""], caret: 0 };
  }
  if (!leftLines.length) {
    return { lines: rightLines, caret: 0 };
  }
  if (!rightLines.length) {
    return { lines: leftLines, caret: leftLines.join("\n").length };
  }

  const boundaryPrefix = leftLines.slice(0, -1).join("\n");
  const caret = boundaryPrefix.length + (boundaryPrefix ? 1 : 0) + leftLines[leftLines.length - 1].length + 1;
  const mergedLines = [...leftLines];
  mergedLines[mergedLines.length - 1] = normalizeEditableText(`${mergedLines[mergedLines.length - 1]} ${rightLines[0]}`);
  mergedLines.push(...rightLines.slice(1));
  return { lines: mergedLines, caret };
}

function normalizeCaptionEditorLines(value: string): string[] {
  const lines = value
    .replace(/\r\n/g, "\n")
    .split("\n")
    .map((line) => line.trim())
    .filter(Boolean);
  return lines.length ? lines : [""];
}

function isEditableTarget(target: EventTarget | null): boolean {
  if (!(target instanceof HTMLElement)) {
    return false;
  }
  return target.isContentEditable || ["INPUT", "TEXTAREA", "SELECT"].includes(target.tagName);
}

function normalizeToken(text: string): string {
  return text
    .toLowerCase()
    .replace(/[\u2018\u2019]/g, "'")
    .replace(/^[^a-z0-9']+|[^a-z0-9']+$/g, "");
}

function uniqueWordIds(fragments: MatchedFragment[]): string[] {
  return Array.from(new Set(fragments.flatMap((fragment) => (fragment.word ? [fragment.word.id] : []))));
}

function buildMatchedFragments(text: string, wordIds: string[], lookup: Map<string, WordToken>): MatchedFragment[] {
  const sourceWords = wordIds.map((wordId) => lookup.get(wordId)).filter((word): word is WordToken => Boolean(word));
  const tokenMatches = text.match(/\s+|[^\s]+/g) ?? [text || ""];
  const fragments: MatchedFragment[] = [];
  let sourceIndex = 0;
  let charOffset = 0;

  tokenMatches.forEach((token, index) => {
    const charStart = charOffset;
    const charEnd = charStart + token.length;
    charOffset = charEnd;

    if (!token.trim()) {
      fragments.push({ key: `space-${index}-${charStart}`, text: token, charStart, charEnd, word: null });
      return;
    }

    const normalized = normalizeToken(token);
    let matchedWord: WordToken | null = null;
    if (normalized) {
      for (let offset = 0; offset < MATCH_LOOKAHEAD && sourceIndex + offset < sourceWords.length; offset += 1) {
        const candidate = sourceWords[sourceIndex + offset];
        if (normalizeToken(candidate.text) === normalized) {
          matchedWord = candidate;
          sourceIndex += offset + 1;
          break;
        }
      }
    }

    fragments.push({ key: `token-${index}-${charStart}`, text: token, charStart, charEnd, word: matchedWord });
  });

  return fragments;
}

function lowerBoundWordStart(words: WordToken[], time: number): number {
  let low = 0;
  let high = words.length;

  while (low < high) {
    const mid = Math.floor((low + high) / 2);
    if (words[mid].start < time) {
      low = mid + 1;
    } else {
      high = mid;
    }
  }

  return low;
}

function upperBoundWordStart(words: WordToken[], time: number): number {
  let low = 0;
  let high = words.length;

  while (low < high) {
    const mid = Math.floor((low + high) / 2);
    if (words[mid].start <= time) {
      low = mid + 1;
    } else {
      high = mid;
    }
  }

  return low;
}

function normalizedCaptionTokens(text: string): string[] {
  return (text.match(/\s+|[^\s]+/g) ?? [])
    .filter((token) => Boolean(token.trim()))
    .map((token) => normalizeToken(token))
    .filter(Boolean);
}

function rematchCaptionInWindow(
  caption: Caption,
  orderedWords: WordToken[],
  minimumGlobalIndex: number,
  windowPaddingSeconds: number,
): { matchedWords: WordToken[]; ratio: number } {
  const captionText = plainCaptionText(caption);
  const normalizedTokens = normalizedCaptionTokens(captionText);
  if (!normalizedTokens.length) {
    return { matchedWords: [], ratio: 0 };
  }

  const windowStart = Math.max(0, caption.start - windowPaddingSeconds);
  const windowEnd = caption.end + windowPaddingSeconds;
  let candidateStart = Math.max(minimumGlobalIndex, lowerBoundWordStart(orderedWords, windowStart));
  while (candidateStart > minimumGlobalIndex && orderedWords[candidateStart - 1].end >= windowStart) {
    candidateStart -= 1;
  }
  const candidateEnd = Math.max(candidateStart, upperBoundWordStart(orderedWords, windowEnd));
  const windowWords = orderedWords.slice(candidateStart, candidateEnd);
  if (!windowWords.length) {
    return { matchedWords: [], ratio: 0 };
  }

  const matchedWords: WordToken[] = [];
  let sourceIndex = 0;

  for (const token of normalizedTokens) {
    let matchedIndex = -1;
    for (let index = sourceIndex; index < windowWords.length; index += 1) {
      if (normalizeToken(windowWords[index].text) === token) {
        matchedIndex = index;
        break;
      }
    }

    if (matchedIndex < 0) {
      continue;
    }

    matchedWords.push(windowWords[matchedIndex]);
    sourceIndex = matchedIndex + 1;
  }

  return {
    matchedWords,
    ratio: matchedWords.length / normalizedTokens.length,
  };
}

function syncCaptionWordAssignments(captions: Caption[], words: WordToken[], options?: CaptionWordSyncOptions): Caption[] {
  if (!words.length) {
    return captions;
  }

  const orderedWords = [...words].sort((left, right) => left.start - right.start);
  if (options?.mode === "time_anchored") {
    const wordIndexById = new Map(orderedWords.map((word, index) => [word.id, index]));
    let minimumGlobalIndex = 0;

    return captions.map((caption) => {
      const captionText = plainCaptionText(caption);
      if (!captionText) {
        return {
          ...caption,
          word_ids: [],
        };
      }

      let bestMatch: { matchedWords: WordToken[]; ratio: number } = { matchedWords: [], ratio: 0 };
      for (const windowPaddingSeconds of REMATCH_SEARCH_WINDOWS_SECONDS) {
        const nextMatch = rematchCaptionInWindow(caption, orderedWords, minimumGlobalIndex, windowPaddingSeconds);
        if (nextMatch.ratio > bestMatch.ratio || (nextMatch.ratio === bestMatch.ratio && nextMatch.matchedWords.length > bestMatch.matchedWords.length)) {
          bestMatch = nextMatch;
        }
        if (nextMatch.ratio >= STRONG_REMATCH_RATIO) {
          bestMatch = nextMatch;
          break;
        }
      }

      if (!bestMatch.matchedWords.length || bestMatch.ratio < MIN_REMATCH_RATIO) {
        return caption;
      }

      const lastMatchedWord = bestMatch.matchedWords[bestMatch.matchedWords.length - 1];
      const lastMatchedIndex = wordIndexById.get(lastMatchedWord.id) ?? -1;
      if (lastMatchedIndex >= minimumGlobalIndex) {
        minimumGlobalIndex = lastMatchedIndex + 1;
      }

      return {
        ...caption,
        start: options.preserveTiming ? caption.start : bestMatch.matchedWords[0].start,
        end: options.preserveTiming ? caption.end : bestMatch.matchedWords[bestMatch.matchedWords.length - 1].end,
        word_ids: bestMatch.matchedWords.map((word) => word.id),
      };
    });
  }

  let sourceIndex = 0;

  return captions.map((caption) => {
    const captionText = plainCaptionText(caption);
    if (!captionText) {
      return {
        ...caption,
        word_ids: [],
      };
    }

    const tokens = captionText.match(/\s+|[^\s]+/g) ?? [];
    const matchedWords: WordToken[] = [];

    for (const token of tokens) {
      if (!token.trim()) {
        continue;
      }

      const normalized = normalizeToken(token);
      if (!normalized) {
        continue;
      }

      let matchedIndex = -1;
      for (
        let offset = 0;
        offset <= REMATCH_LOOKAHEAD && sourceIndex + offset < orderedWords.length;
        offset += 1
      ) {
        if (normalizeToken(orderedWords[sourceIndex + offset].text) === normalized) {
          matchedIndex = sourceIndex + offset;
          break;
        }
      }

      if (matchedIndex < 0) {
        continue;
      }

      matchedWords.push(orderedWords[matchedIndex]);
      sourceIndex = matchedIndex + 1;
    }

    if (!matchedWords.length) {
      return caption;
    }

    return {
      ...caption,
      start: options?.preserveTiming ? caption.start : matchedWords[0].start,
      end: options?.preserveTiming ? caption.end : matchedWords[matchedWords.length - 1].end,
      word_ids: matchedWords.map((word) => word.id),
    };
  });
}

function timeFromCaret(fragments: MatchedFragment[], offset: number, fallback: number): number {
  const directMatch = fragments.find((fragment) => fragment.word && offset >= fragment.charStart && offset <= fragment.charEnd);
  if (directMatch?.word) {
    return directMatch.word.start;
  }

  const previousMatch = [...fragments].reverse().find((fragment) => fragment.word && fragment.charEnd <= offset);
  if (previousMatch?.word) {
    return previousMatch.word.start;
  }

  const nextMatch = fragments.find((fragment) => fragment.word && fragment.charStart >= offset);
  if (nextMatch?.word) {
    return nextMatch.word.start;
  }

  return fallback;
}

function timeRangeFromSelection(
  fragments: MatchedFragment[],
  start: number,
  end: number,
  fallbackStart: number,
  fallbackEnd: number,
): { start: number; end: number } {
  if (start === end) {
    return { start: fallbackStart, end: fallbackEnd };
  }

  const left = Math.min(start, end);
  const right = Math.max(start, end);
  const selectedWords = fragments.filter(
    (fragment) => fragment.word && fragment.charStart < right && fragment.charEnd > left,
  );

  if (!selectedWords.length) {
    return { start: fallbackStart, end: fallbackEnd };
  }

  return {
    start: selectedWords[0].word?.start ?? fallbackStart,
    end: selectedWords[selectedWords.length - 1].word?.end ?? fallbackEnd,
  };
}

function clampSplitTime(start: number, end: number, proposed: number): number {
  if (end - start <= 0.05) {
    return start + (end - start) / 2;
  }
  return Math.min(end - 0.02, Math.max(start + 0.02, proposed));
}

function clampNumber(value: number, min: number, max: number): number {
  if (max < min) {
    return min;
  }
  return Math.min(max, Math.max(min, value));
}

function roundTiming(value: number): number {
  return Math.round(value * 1000) / 1000;
}

function findNearestTimedIndex(items: Array<{ start: number; end: number }>, time: number): number {
  if (!items.length) {
    return -1;
  }

  let bestIndex = 0;
  let bestDistance = Number.POSITIVE_INFINITY;

  items.forEach((item, index) => {
    const distance = time < item.start ? item.start - time : time > item.end ? time - item.end : 0;
    if (distance < bestDistance) {
      bestDistance = distance;
      bestIndex = index;
    }
  });

  return bestIndex;
}

function getCaptionWordBounds(caption: Caption, lookup: Map<string, WordToken>): { start: number; end: number } | null {
  const words = caption.word_ids
    .map((wordId) => lookup.get(wordId))
    .filter((word): word is WordToken => Boolean(word))
    .sort((left, right) => left.start - right.start);

  if (!words.length) {
    return null;
  }

  return {
    start: words[0].start,
    end: words[words.length - 1].end,
  };
}

function speechSpansInWindow(spans: SpeechSpan[], start: number, end: number): SpeechSpan[] {
  return spans.filter((span) => span.end >= start && span.start <= end);
}

function findStartSpeechSpan(
  spans: SpeechSpan[],
  referenceStart: number,
  hasWordBounds: boolean,
): SpeechSpan | null {
  const afterLimit = hasWordBounds ? 0.22 : WAVEFORM_MAX_EDGE_MOVE_SECONDS;
  const beforeLimit = WAVEFORM_MAX_EDGE_MOVE_SECONDS;
  const candidates = spans
    .filter(
      (span) =>
        span.start >= referenceStart - beforeLimit &&
        span.start <= referenceStart + afterLimit &&
        span.end >= referenceStart - 0.12,
    )
    .sort((left, right) => Math.abs(left.start - referenceStart) - Math.abs(right.start - referenceStart));

  return candidates[0] ?? null;
}

function findEndSpeechSpan(
  spans: SpeechSpan[],
  referenceEnd: number,
  hasWordBounds: boolean,
): SpeechSpan | null {
  const beforeLimit = hasWordBounds ? 0.22 : WAVEFORM_MAX_EDGE_MOVE_SECONDS;
  const afterLimit = WAVEFORM_MAX_EDGE_MOVE_SECONDS;
  const candidates = spans
    .filter(
      (span) =>
        span.end >= referenceEnd - beforeLimit &&
        span.end <= referenceEnd + afterLimit &&
        span.start <= referenceEnd + 0.12,
    )
    .sort((left, right) => Math.abs(left.end - referenceEnd) - Math.abs(right.end - referenceEnd));

  return candidates[0] ?? null;
}

function alignCaptionsToWaveformSpans(
  captions: Caption[],
  speechSpans: SpeechSpan[],
  lookup: Map<string, WordToken>,
  duration: number | null,
): WaveformAlignmentResult {
  if (!captions.length || !speechSpans.length) {
    return { captions, edgeAdjustmentCount: 0, captionAdjustmentCount: 0 };
  }

  const aligned = captions.map((caption) => {
    const wordBounds = getCaptionWordBounds(caption, lookup);
    const referenceStart = wordBounds?.start ?? caption.start;
    const referenceEnd = wordBounds?.end ?? caption.end;
    const searchStart = Math.max(0, Math.min(caption.start, referenceStart) - WAVEFORM_MAX_EDGE_MOVE_SECONDS);
    const searchEnd = Math.max(caption.end, referenceEnd) + WAVEFORM_MAX_EDGE_MOVE_SECONDS;
    const nearbySpans = speechSpansInWindow(speechSpans, searchStart, searchEnd);
    const startSpan = findStartSpeechSpan(nearbySpans, referenceStart, Boolean(wordBounds));
    const endSpan = findEndSpeechSpan(nearbySpans, referenceEnd, Boolean(wordBounds));

    let start = caption.start;
    let end = caption.end;

    if (startSpan) {
      const candidateStart = Math.max(0, startSpan.start - WAVEFORM_START_PAD_SECONDS);
      if (
        Math.abs(candidateStart - caption.start) <= WAVEFORM_MAX_EDGE_MOVE_SECONDS &&
        candidateStart < end - WAVEFORM_MIN_CAPTION_SECONDS
      ) {
        start = candidateStart;
      }
    }

    if (endSpan) {
      const candidateEnd = duration === null ? endSpan.end + WAVEFORM_END_PAD_SECONDS : Math.min(duration, endSpan.end + WAVEFORM_END_PAD_SECONDS);
      if (
        Math.abs(candidateEnd - caption.end) <= WAVEFORM_MAX_EDGE_MOVE_SECONDS &&
        candidateEnd > start + WAVEFORM_MIN_CAPTION_SECONDS
      ) {
        end = candidateEnd;
      }
    }

    return {
      ...caption,
      start: roundTiming(start),
      end: roundTiming(end),
      lines: [...caption.lines],
      word_ids: [...caption.word_ids],
    };
  });

  for (let index = 0; index < aligned.length; index += 1) {
    const caption = aligned[index];
    const previous = aligned[index - 1];
    const next = aligned[index + 1];

    if (previous && caption.start < previous.end + 0.02) {
      caption.start = roundTiming(clampNumber(previous.end + 0.02, caption.start, caption.end - WAVEFORM_MIN_CAPTION_SECONDS));
    }

    if (next && caption.end > next.start - 0.02) {
      caption.end = roundTiming(clampNumber(next.start - 0.02, caption.start + WAVEFORM_MIN_CAPTION_SECONDS, caption.end));
    }

    if (caption.end - caption.start < WAVEFORM_MIN_CAPTION_SECONDS) {
      caption.start = captions[index].start;
      caption.end = captions[index].end;
    }
  }

  let edgeAdjustmentCount = 0;
  let captionAdjustmentCount = 0;
  aligned.forEach((caption, index) => {
    const original = captions[index];
    const startMoved = Math.abs(caption.start - original.start) >= WAVEFORM_MIN_EDGE_DELTA_SECONDS;
    const endMoved = Math.abs(caption.end - original.end) >= WAVEFORM_MIN_EDGE_DELTA_SECONDS;
    edgeAdjustmentCount += (startMoved ? 1 : 0) + (endMoved ? 1 : 0);
    if (startMoved || endMoved) {
      captionAdjustmentCount += 1;
    }
  });

  return { captions: aligned, edgeAdjustmentCount, captionAdjustmentCount };
}

function speakerLabelForItem(item: { speaker_id: number | null; speaker_name: string | null }): string {
  if (item.speaker_name) {
    return item.speaker_name;
  }
  return item.speaker_id === null ? "Speaker" : `Speaker ${item.speaker_id + 1}`;
}

function isSpeechActiveBetween(spans: SpeechSpan[], start: number, end: number): boolean {
  if (end <= start) {
    return true;
  }
  return spans.some((span) => span.start <= end && span.end >= start);
}

function pushTimelineEvent(events: SpeakerTimelineEvent[], event: SpeakerTimelineEvent) {
  const duplicate = events.some(
    (existing) => existing.kind === event.kind && Math.abs(existing.time - event.time) < 0.16,
  );
  if (!duplicate) {
    events.push(event);
  }
}

function detectSpeakerTimelineEvents(
  captions: Caption[],
  words: WordToken[],
  speechSpans: SpeechSpan[],
): SpeakerTimelineEvent[] {
  const events: SpeakerTimelineEvent[] = [];

  for (let index = 0; index < captions.length - 1; index += 1) {
    const current = captions[index];
    const next = captions[index + 1];
    if (
      current.speaker_id === null ||
      next.speaker_id === null ||
      current.speaker_id === next.speaker_id
    ) {
      continue;
    }

    const gap = next.start - current.end;
    const switchStart = Math.min(current.end, next.start);
    const switchEnd = Math.max(current.end, next.start);
    const kind: SpeakerTimelineEventKind =
      gap < 0.12 || isSpeechActiveBetween(speechSpans, switchStart, switchEnd) ? "tight_handoff" : "switch";
    const time = gap >= 0 ? current.end + gap / 2 : Math.max(next.start, current.end + gap / 2);

    pushTimelineEvent(events, {
      id: `caption-${index}-${kind}`,
      kind,
      time,
      start: switchStart,
      end: switchEnd,
      label: `${speakerLabelForItem(current)} -> ${speakerLabelForItem(next)}`,
      captionIndex: index + 1,
    });
  }

  const orderedWords = words
    .filter((word) => word.speaker_id !== null)
    .sort((left, right) => left.start - right.start);

  for (let index = 1; index < orderedWords.length; index += 1) {
    const previous = orderedWords[index - 1];
    const current = orderedWords[index];
    if (previous.speaker_id === current.speaker_id) {
      continue;
    }

    const overlapStart = Math.max(previous.start, current.start);
    const overlapEnd = Math.min(previous.end, current.end);
    if (overlapEnd - overlapStart > 0.02) {
      pushTimelineEvent(events, {
        id: `word-overlap-${index}`,
        kind: "overlap",
        time: overlapStart + (overlapEnd - overlapStart) / 2,
        start: overlapStart,
        end: overlapEnd,
        label: `${speakerLabelForItem(previous)} + ${speakerLabelForItem(current)}`,
      });
      continue;
    }

    const handoffGap = current.start - previous.end;
    if (handoffGap >= 0 && handoffGap < 0.1) {
      pushTimelineEvent(events, {
        id: `word-handoff-${index}`,
        kind: "tight_handoff",
        time: previous.end + handoffGap / 2,
        start: previous.end,
        end: current.start,
        label: `${speakerLabelForItem(previous)} -> ${speakerLabelForItem(current)}`,
      });
    }
  }

  return events.sort((left, right) => left.time - right.time);
}

function buildParagraphsFromCaptions(captions: Caption[]): Paragraph[] {
  const paragraphs: Paragraph[] = [];
  let bucket: Caption[] = [];

  const flush = () => {
    if (!bucket.length) {
      return;
    }
    paragraphs.push({
      id: `p-${bucket[0].id}-${bucket[bucket.length - 1].id}`,
      start: bucket[0].start,
      end: bucket[bucket.length - 1].end,
      speaker_id: bucket[0].speaker_id,
      speaker_name: bucket[0].speaker_name,
      text: normalizeEditableText(bucket.map((caption) => plainCaptionText(caption)).join(" ")),
      word_ids: bucket.flatMap((caption) => caption.word_ids),
      caption_ids: bucket.map((caption) => caption.id),
    });
    bucket = [];
  };

  for (const caption of captions) {
    if (!bucket.length) {
      bucket.push(caption);
      continue;
    }

    const previous = bucket[bucket.length - 1];
    const textSoFar = normalizeEditableText(bucket.map((item) => plainCaptionText(item)).join(" "));
    const shouldBreak =
      previous.speaker_id !== caption.speaker_id ||
      caption.start - previous.end > 1.8 ||
      (textSoFar.length > 380 && SENTENCE_END_RE.test(plainCaptionText(previous)));
    if (shouldBreak) {
      flush();
    }

    bucket.push(caption);
  }

  flush();
  return paragraphs;
}

function rebuildParagraphFromCaptions(
  paragraph: Paragraph,
  captions: Caption[],
  captionIndexes: number[],
): Paragraph {
  const bucket = captionIndexes
    .map((index) => captions[index])
    .filter((caption): caption is Caption => Boolean(caption));
  if (!bucket.length) {
    return paragraph;
  }

  return {
    ...paragraph,
    start: bucket[0].start,
    end: bucket[bucket.length - 1].end,
    speaker_id: bucket[0].speaker_id,
    speaker_name: bucket[0].speaker_name,
    text: normalizeEditableText(bucket.map((caption) => plainCaptionText(caption)).join(" ")),
    word_ids: bucket.flatMap((caption) => caption.word_ids),
    caption_ids: bucket.map((caption) => caption.id),
  };
}

function findParagraphIndexByCaptionId(paragraphs: Paragraph[], captionId: string): number {
  return paragraphs.findIndex((paragraph) => paragraph.caption_ids?.includes(captionId));
}

function normalizeEditorState(state: EditorState): EditorState {
  const next = cloneEditorState(state);
  next.speakers = normalizeSpeakers(next.speakers);
  next.captions = next.captions.map((caption) => ({
    ...caption,
    lines: normalizeCaptionLines(caption.lines),
  }));
  if (!next.paragraphs.length) {
    next.paragraphs = buildParagraphsFromCaptions(next.captions);
  }
  return next;
}

function getParagraphCaptionIndexes(paragraph: Paragraph, captions: Caption[]): number[] {
  if (paragraph.caption_ids?.length) {
    const captionIndexById = new Map(captions.map((caption, index) => [caption.id, index]));
    return paragraph.caption_ids
      .map((captionId) => captionIndexById.get(captionId))
      .filter((index): index is number => index !== undefined);
  }

  const overlapping = captions
    .map((caption, index) => ({ caption, index }))
    .filter(({ caption }) => Math.max(paragraph.start, caption.start) < Math.min(paragraph.end, caption.end))
    .map(({ index }) => index);

  if (overlapping.length) {
    return overlapping;
  }

  const nearest = findNearestTimedIndex(captions, paragraph.start);
  return nearest >= 0 ? [nearest] : [];
}

function buildCaptionRangesFromIndexes(captions: Caption[], indexes: number[]): RetranscribeTarget[] {
  const ordered = [...new Set(indexes)].sort((left, right) => left - right);
  if (!ordered.length) {
    return [];
  }

  const ranges: RetranscribeTarget[] = [];
  let startIndex = ordered[0];
  let endIndex = ordered[0];

  const flush = () => {
    const startCaption = captions[startIndex];
    const endCaption = captions[endIndex];
    if (!startCaption || !endCaption) {
      return;
    }
    ranges.push({
      start: startCaption.start,
      end: endCaption.end,
      label:
        startIndex === endIndex
          ? `subtitle ${startIndex + 1}`
          : `subtitles ${startIndex + 1}-${endIndex + 1}`,
    });
  };

  for (let cursor = 1; cursor < ordered.length; cursor += 1) {
    const currentIndex = ordered[cursor];
    const previousIndex = ordered[cursor - 1];
    const previousCaption = captions[previousIndex];
    const currentCaption = captions[currentIndex];
    const contiguous =
      currentIndex === previousIndex + 1 &&
      previousCaption &&
      currentCaption &&
      currentCaption.start - previousCaption.end <= 0.6;

    if (contiguous) {
      endIndex = currentIndex;
      continue;
    }

    flush();
    startIndex = currentIndex;
    endIndex = currentIndex;
  }

  flush();
  return ranges;
}

function buildCaptionTextSpans(texts: string[]): Array<{ start: number; end: number }> {
  let cursor = 0;
  return texts.map((text, index) => {
    const start = cursor;
    const end = start + text.length;
    cursor = end + (index < texts.length - 1 ? 1 : 0);
    return { start, end };
  });
}

function findCaptionIndexForOffset(
  spans: Array<{ start: number; end: number }>,
  offset: number,
  preferPrevious: boolean,
): number {
  if (spans.length === 1) {
    return 0;
  }

  if (preferPrevious && offset > 0) {
    const previousIndex = spans.findIndex((span) => offset - 1 >= span.start && offset - 1 < span.end);
    if (previousIndex >= 0) {
      return previousIndex;
    }
  }

  const directIndex = spans.findIndex((span) => offset >= span.start && offset < span.end);
  if (directIndex >= 0) {
    return directIndex;
  }

  const previousIndex = [...spans].reverse().findIndex((span) => span.end <= offset);
  if (previousIndex >= 0) {
    return spans.length - 1 - previousIndex;
  }

  return 0;
}

function findWhitespaceSplit(text: string, target: number, min: number, max: number): number {
  let bestIndex = Math.min(max, Math.max(min, target));
  let bestDistance = Number.POSITIVE_INFINITY;

  for (let index = min; index <= max; index += 1) {
    if (text[index] !== " ") {
      continue;
    }
    const distance = Math.abs(index - target);
    if (distance < bestDistance) {
      bestDistance = distance;
      bestIndex = index;
    }
  }

  return bestIndex;
}

function distributeTextAcrossCaptions(text: string, originalTexts: string[]): string[] {
  const normalized = normalizeEditableText(text);
  if (originalTexts.length <= 1) {
    return [normalized];
  }

  if (!normalized) {
    return originalTexts.map(() => "");
  }

  const weights = originalTexts.map((item) => Math.max(item.length, 1));
  const totalWeight = weights.reduce((sum, value) => sum + value, 0);
  const cuts: number[] = [];
  let previousCut = 0;

  for (let boundary = 0; boundary < originalTexts.length - 1; boundary += 1) {
    const remainingBoundaries = originalTexts.length - boundary - 2;
    const cumulativeWeight = weights.slice(0, boundary + 1).reduce((sum, value) => sum + value, 0);
    const target = Math.round((normalized.length * cumulativeWeight) / totalWeight);
    const min = Math.min(normalized.length, previousCut + 1);
    const max = Math.max(min, normalized.length - Math.max(0, remainingBoundaries));
    const splitAt = findWhitespaceSplit(normalized, target, min, max);
    cuts.push(splitAt);
    previousCut = splitAt + 1;
  }

  const slices: string[] = [];
  let start = 0;
  for (const cut of cuts) {
    slices.push(normalized.slice(start, cut).trim());
    start = cut + 1;
  }
  slices.push(normalized.slice(start).trim());
  return slices;
}

function applyParagraphTextToCaptions(draft: EditorState, paragraphIndex: number, nextText: string) {
  const paragraph = draft.paragraphs[paragraphIndex];
  if (!paragraph) {
    return;
  }

  const captionIndexes = getParagraphCaptionIndexes(paragraph, draft.captions);
  if (!captionIndexes.length) {
    return;
  }

  const originalTexts = captionIndexes.map((index) => plainCaptionText(draft.captions[index]));
  const originalParagraphText = normalizeEditableText(originalTexts.join(" "));
  const normalizedNextText = normalizeEditableText(nextText);

  if (originalParagraphText === normalizedNextText) {
    return;
  }

  let start = 0;
  while (
    start < originalParagraphText.length &&
    start < normalizedNextText.length &&
    originalParagraphText[start] === normalizedNextText[start]
  ) {
    start += 1;
  }

  let originalEnd = originalParagraphText.length;
  let nextEnd = normalizedNextText.length;
  while (
    originalEnd > start &&
    nextEnd > start &&
    originalParagraphText[originalEnd - 1] === normalizedNextText[nextEnd - 1]
  ) {
    originalEnd -= 1;
    nextEnd -= 1;
  }

  const spans = buildCaptionTextSpans(originalTexts);
  const firstAffected = findCaptionIndexForOffset(spans, start, true);
  const lastAffected =
    originalEnd > start
      ? findCaptionIndexForOffset(spans, Math.max(start, originalEnd - 1), false)
      : firstAffected;
  const affectedStart = spans[firstAffected]?.start ?? 0;
  const affectedEnd = spans[lastAffected]?.end ?? originalParagraphText.length;
  const originalAffectedText = originalParagraphText.slice(affectedStart, affectedEnd);
  const relativeStart = Math.max(0, start - affectedStart);
  const relativeEnd = Math.max(relativeStart, originalEnd - affectedStart);
  const nextAffectedText =
    originalAffectedText.slice(0, relativeStart) +
    normalizedNextText.slice(start, nextEnd) +
    originalAffectedText.slice(relativeEnd);
  const redistributed = distributeTextAcrossCaptions(
    nextAffectedText,
    originalTexts.slice(firstAffected, lastAffected + 1),
  );

  redistributed.forEach((text, offset) => {
    const caption = draft.captions[captionIndexes[firstAffected + offset]];
    caption.lines = normalizeCaptionLines([text]);
  });
}

function applyCaptionTextEdit(editor: EditorState, index: number, value: string): EditorState {
  const currentCaption = editor.captions[index];
  if (!currentCaption) {
    return editor;
  }

  const nextLines = normalizeCaptionEditorLines(value);
  if (captionValue(currentCaption) === nextLines.join("\n")) {
    return editor;
  }

  const nextCaptions = [...editor.captions];
  nextCaptions[index] = {
    ...currentCaption,
    lines: nextLines,
  };

  const paragraphIndex = findParagraphIndexByCaptionId(editor.paragraphs, currentCaption.id);
  if (paragraphIndex < 0) {
    return {
      ...editor,
      captions: nextCaptions,
    };
  }

  const nextParagraphs = [...editor.paragraphs];
  const paragraph = editor.paragraphs[paragraphIndex];
  const captionIndexes = getParagraphCaptionIndexes(paragraph, nextCaptions);
  nextParagraphs[paragraphIndex] = rebuildParagraphFromCaptions(paragraph, nextCaptions, captionIndexes);

  return {
    ...editor,
    captions: nextCaptions,
    paragraphs: nextParagraphs,
  };
}

function applyParagraphTextEdit(editor: EditorState, paragraphIndex: number, value: string): EditorState {
  const paragraph = editor.paragraphs[paragraphIndex];
  if (!paragraph) {
    return editor;
  }

  if (normalizeEditableText(value) === normalizeEditableText(paragraph.text)) {
    return editor;
  }

  const captionIndexes = getParagraphCaptionIndexes(paragraph, editor.captions);
  if (!captionIndexes.length) {
    return editor;
  }

  const nextCaptions = [...editor.captions];
  captionIndexes.forEach((captionIndex) => {
    nextCaptions[captionIndex] = {
      ...nextCaptions[captionIndex],
      lines: [...nextCaptions[captionIndex].lines],
    };
  });

  const draft: EditorState = {
    ...editor,
    captions: nextCaptions,
    paragraphs: [...editor.paragraphs],
  };
  applyParagraphTextToCaptions(draft, paragraphIndex, value);

  const nextParagraphs = [...editor.paragraphs];
  nextParagraphs[paragraphIndex] = rebuildParagraphFromCaptions(paragraph, nextCaptions, captionIndexes);

  return {
    ...editor,
    captions: nextCaptions,
    paragraphs: nextParagraphs,
  };
}

function normalizeImportedCaptions(captions: Caption[]): { speakers: Speaker[]; captions: Caption[] } {
  const names = Array.from(
    new Map(
      captions
        .map((caption) => caption.speaker_name?.trim())
        .filter((name): name is string => Boolean(name))
        .map((name) => [name.toLowerCase(), name]),
    ).values(),
  );

  if (!names.length) {
    const [defaultSpeaker] = buildDefaultSpeakers();
    return {
      speakers: [defaultSpeaker],
      captions: captions.map((caption) => ({
        ...caption,
        speaker_id: defaultSpeaker.id,
        speaker_name: defaultSpeaker.name,
      })),
    };
  }

  const speakers = normalizeSpeakers(names.map((name, index) => ({ id: index, name })));
  const speakerLookup = new Map(speakers.map((speaker) => [speaker.name.toLowerCase(), speaker]));

  return {
    speakers,
    captions: captions.map((caption) => {
      const name = caption.speaker_name?.trim();
      if (!name) {
        return caption;
      }

      const speaker = speakerLookup.get(name.toLowerCase());
      if (!speaker) {
        return caption;
      }

      return {
        ...caption,
        speaker_id: speaker.id,
        speaker_name: speaker.name,
      };
    }),
  };
}

function buildImportedSession(audioFilename: string, captions: Caption[]): TranscriptResponse {
  const normalized = normalizeImportedCaptions(captions);
  return {
    audio_filename: audioFilename,
    duration: normalized.captions[normalized.captions.length - 1]?.end ?? null,
    speakers: normalized.speakers,
    words: [],
    paragraphs: buildParagraphsFromCaptions(normalized.captions),
    captions: normalized.captions,
    guide_blocks: [],
    warnings: [
      {
        code: "imported_srt",
        message: "Loaded from an existing SRT. Word-level confidence and word-accurate highlighting are only available for fresh Whisper transcriptions.",
      },
    ],
    model: "imported",
    speaker_assignment_mode: "segment",
    language: null,
    gpu_enabled: false,
  };
}

function hasExplicitImportedSpeakerLabels(captions: Caption[]): boolean {
  return captions.some((caption) => Boolean(caption.speaker_name?.trim()));
}

function buildRealignedImportedSession(
  audioFilename: string,
  captions: Caption[],
  baseSession: TranscriptResponse,
  options?: { retimeCaptions?: boolean },
): TranscriptResponse {
  const retimeCaptions = Boolean(options?.retimeCaptions);
  const importedHasSpeakers = hasExplicitImportedSpeakerLabels(captions);
  const normalizedImport = importedHasSpeakers ? normalizeImportedCaptions(captions) : null;
  const speakerSource = importedHasSpeakers ? normalizedImport?.speakers ?? baseSession.speakers : baseSession.speakers;
  const importedCaptions = normalizedImport?.captions ?? captions;
  const seedCaptions = importedCaptions.map((caption) => ({ ...caption, word_ids: [] }));
  const rematchedCaptions = syncCaptionWordAssignments(seedCaptions, baseSession.words, {
    mode: "time_anchored",
    preserveTiming: !retimeCaptions,
  });
  const alignedCaptions = importedHasSpeakers ? rematchedCaptions : applyBlockSpeakers(rematchedCaptions, baseSession.words);
  const paragraphs = applyBlockSpeakers(buildParagraphsFromCaptions(alignedCaptions), baseSession.words);
  const matchedCaptionCount = alignedCaptions.filter((caption) => caption.word_ids.length > 0).length;

  return {
    ...baseSession,
    audio_filename: audioFilename,
    duration: baseSession.duration ?? alignedCaptions[alignedCaptions.length - 1]?.end ?? null,
    speakers: speakerSource.map((speaker) => ({ ...speaker })),
    words: cloneWords(baseSession.words),
    paragraphs,
    captions: alignedCaptions,
    guide_blocks: [],
    warnings: [
      ...cloneWarnings(baseSession.warnings),
      {
        code: "imported_srt_realigned",
        message:
          retimeCaptions
            ? matchedCaptionCount === alignedCaptions.length
              ? "Reloaded from an edited SRT and rebuilt caption timings from a fresh WhisperX pass."
              : `Reloaded from an edited SRT and rebuilt timings for ${matchedCaptionCount} of ${alignedCaptions.length} captions from a fresh WhisperX pass. Added non-spoken text may keep broader imported timing.`
            : matchedCaptionCount === alignedCaptions.length
              ? "Reloaded from an edited SRT, preserved the original SRT timing, and rematched the text to fresh WhisperX words."
              : `Reloaded from an edited SRT, preserved the original SRT timing, and rematched ${matchedCaptionCount} of ${alignedCaptions.length} captions to fresh WhisperX words.`,
      },
    ],
  };
}

function chooseSpeakerForRange(
  start: number,
  end: number,
  sourceCaptions: Array<Pick<Caption, "start" | "end" | "speaker_id" | "speaker_name">>,
): { speaker_id: number | null; speaker_name: string | null } {
  if (!sourceCaptions.length) {
    return { speaker_id: null, speaker_name: null };
  }

  const weighted = new Map<string, { speaker_id: number | null; speaker_name: string | null; weight: number }>();
  for (const caption of sourceCaptions) {
    const overlap = Math.max(0, Math.min(end, caption.end) - Math.max(start, caption.start));
    if (overlap <= 0) {
      continue;
    }
    const key = `${caption.speaker_id ?? "null"}|${caption.speaker_name ?? ""}`;
    const existing = weighted.get(key);
    if (existing) {
      existing.weight += overlap;
    } else {
      weighted.set(key, {
        speaker_id: caption.speaker_id,
        speaker_name: caption.speaker_name,
        weight: overlap,
      });
    }
  }

  if (weighted.size) {
    return [...weighted.values()].sort((left, right) => right.weight - left.weight)[0];
  }

  const midpoint = (start + end) / 2;
  const nearest = [...sourceCaptions].sort((left, right) => {
    const leftDistance = midpoint < left.start ? left.start - midpoint : midpoint > left.end ? midpoint - left.end : 0;
    const rightDistance = midpoint < right.start ? right.start - midpoint : midpoint > right.end ? midpoint - right.end : 0;
    return leftDistance - rightDistance;
  })[0];

  return {
    speaker_id: nearest?.speaker_id ?? null,
    speaker_name: nearest?.speaker_name ?? null,
  };
}

function dominantSpeakerFromWords(words: WordToken[]): { speaker_id: number | null; speaker_name: string | null } {
  const weighted = new Map<string, { speaker_id: number | null; speaker_name: string | null; weight: number }>();
  for (const word of words) {
    const key = `${word.speaker_id ?? "null"}|${word.speaker_name ?? ""}`;
    const weight = Math.max(0.01, word.end - word.start);
    const existing = weighted.get(key);
    if (existing) {
      existing.weight += weight;
    } else {
      weighted.set(key, {
        speaker_id: word.speaker_id,
        speaker_name: word.speaker_name,
        weight,
      });
    }
  }

  if (!weighted.size) {
    return { speaker_id: null, speaker_name: null };
  }

  return [...weighted.values()].sort((left, right) => right.weight - left.weight)[0];
}

function applyBlockSpeakers<T extends { word_ids: string[]; speaker_id: number | null; speaker_name: string | null }>(
  items: T[],
  words: WordToken[],
): T[] {
  const wordLookup = new Map(words.map((word) => [word.id, word]));
  return items.map((item) => {
    const itemWords = item.word_ids.map((wordId) => wordLookup.get(wordId)).filter((word): word is WordToken => Boolean(word));
    if (!itemWords.length) {
      return item;
    }
    return {
      ...item,
      ...dominantSpeakerFromWords(itemWords),
    };
  });
}

function replaceTimedRange<T extends { start: number; end: number }>(
  current: T[],
  replacements: T[],
  start: number,
  end: number,
): T[] {
  return [...current.filter((item) => item.end <= start || item.start >= end), ...replacements].sort((left, right) => left.start - right.start);
}

function prepareRetranscribedRange(
  payload: RetranscribeRangeResponse,
  sourceCaptions: Caption[],
): Pick<RetranscribeRangeResponse, "words" | "captions" | "paragraphs"> {
  const idPrefix = `rt-${Date.now()}-${Math.round(payload.start * 1000)}`;
  const remappedWordIds = new Map<string, string>();

  const words = payload.words.map((word, index) => {
    const nextId = `${idPrefix}-w-${index}`;
    remappedWordIds.set(word.id, nextId);
    return {
      ...word,
      id: nextId,
      ...chooseSpeakerForRange(word.start, word.end, sourceCaptions),
    };
  });

  const captions = applyBlockSpeakers(
    payload.captions.map((caption, index) => ({
      ...caption,
      id: `${idPrefix}-c-${index}`,
      word_ids: caption.word_ids.map((wordId) => remappedWordIds.get(wordId) ?? wordId),
    })),
    words,
  );

  const paragraphs = applyBlockSpeakers(
    payload.paragraphs.map((paragraph, index) => ({
      ...paragraph,
      id: `${idPrefix}-p-${index}`,
      word_ids: paragraph.word_ids.map((wordId) => remappedWordIds.get(wordId) ?? wordId),
    })),
    words,
  );

  return { words, captions, paragraphs };
}

interface WaveformTimelineProps {
  analysis: WaveformAnalysisResponse | null;
  captions: Caption[];
  speakerEvents: SpeakerTimelineEvent[];
  currentTime: number;
  onSeek: (time: number, options?: { play?: boolean }) => void;
}

function drawWaveformTimeline(
  canvas: HTMLCanvasElement,
  props: WaveformTimelineProps,
  width: number,
  height: number,
) {
  const context = canvas.getContext("2d");
  if (!context) {
    return;
  }

  const dpr = window.devicePixelRatio || 1;
  canvas.width = Math.max(1, Math.floor(width * dpr));
  canvas.height = Math.max(1, Math.floor(height * dpr));
  canvas.style.width = `${width}px`;
  canvas.style.height = `${height}px`;
  context.setTransform(dpr, 0, 0, dpr, 0, 0);

  const { analysis, captions, speakerEvents, currentTime } = props;
  const duration = Math.max(
    analysis?.duration ?? 0,
    captions[captions.length - 1]?.end ?? 0,
    currentTime,
    1,
  );
  const xForTime = (time: number) => clampNumber((time / duration) * width, 0, width);

  context.clearRect(0, 0, width, height);
  context.fillStyle = "#ffffff";
  context.fillRect(0, 0, width, height);

  const gridStep =
    duration > 3600 ? 600 : duration > 1200 ? 300 : duration > 420 ? 60 : duration > 120 ? 20 : 5;
  context.strokeStyle = "#e2e8f0";
  context.lineWidth = 1;
  context.fillStyle = "#647184";
  context.font = "11px Segoe UI, sans-serif";
  for (let time = 0; time <= duration; time += gridStep) {
    const x = xForTime(time);
    context.beginPath();
    context.moveTo(x, 0);
    context.lineTo(x, height);
    context.stroke();
    if (time > 0) {
      context.fillText(formatClock(time), x + 4, 12);
    }
  }

  if (analysis) {
    context.fillStyle = "rgba(15, 118, 110, 0.12)";
    analysis.speech_spans.forEach((span) => {
      const x = xForTime(span.start);
      context.fillRect(x, 20, Math.max(1, xForTime(span.end) - x), height - 42);
    });
  }

  context.fillStyle = "rgba(37, 99, 235, 0.2)";
  context.strokeStyle = "rgba(37, 99, 235, 0.55)";
  captions.forEach((caption) => {
    const x = xForTime(caption.start);
    const w = Math.max(1, xForTime(caption.end) - x);
    context.fillRect(x, 18, w, 14);
    context.strokeRect(x, 18, w, 14);
  });

  const centerY = height * 0.62;
  const waveScale = height * 0.31;
  context.strokeStyle = analysis ? "#1d2635" : "#b8c3d4";
  context.lineWidth = 1;
  context.beginPath();
  context.moveTo(0, centerY);
  context.lineTo(width, centerY);
  context.stroke();

  if (analysis?.frames.length) {
    context.strokeStyle = "#334155";
    context.beginPath();
    analysis.frames.forEach((frame) => {
      const x = xForTime(frame.time);
      const yTop = centerY - frame.max * waveScale;
      const yBottom = centerY - frame.min * waveScale;
      context.moveTo(x, yTop);
      context.lineTo(x, yBottom);
    });
    context.stroke();
  } else {
    context.fillStyle = "#647184";
    context.fillText("Analyze waveform", 12, centerY - 10);
  }

  speakerEvents.forEach((event) => {
    const x = xForTime(event.time);
    context.strokeStyle =
      event.kind === "overlap" ? "#c2410c" : event.kind === "tight_handoff" ? "#b7791f" : "#7c3aed";
    context.lineWidth = event.kind === "overlap" ? 2 : 1.5;
    context.beginPath();
    context.moveTo(x, 10);
    context.lineTo(x, height - 8);
    context.stroke();
  });

  const playheadX = xForTime(currentTime);
  context.strokeStyle = "#ef4444";
  context.lineWidth = 2;
  context.beginPath();
  context.moveTo(playheadX, 0);
  context.lineTo(playheadX, height);
  context.stroke();
}

const WaveformTimeline = memo(function WaveformTimeline(props: WaveformTimelineProps) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const wrapperRef = useRef<HTMLDivElement | null>(null);
  const [size, setSize] = useState({ width: 640, height: 132 });

  useEffect(() => {
    const wrapper = wrapperRef.current;
    if (!wrapper) {
      return;
    }

    const updateSize = () => {
      const rect = wrapper.getBoundingClientRect();
      setSize({
        width: Math.max(320, Math.floor(rect.width)),
        height: 132,
      });
    };

    updateSize();
    const observer = new ResizeObserver(updateSize);
    observer.observe(wrapper);
    return () => observer.disconnect();
  }, []);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) {
      return;
    }
    drawWaveformTimeline(canvas, props, size.width, size.height);
  }, [props, size]);

  return (
    <div className="waveform-timeline" ref={wrapperRef}>
      <canvas
        ref={canvasRef}
        className="waveform-canvas"
        onPointerDown={(event) => {
          const canvas = canvasRef.current;
          if (!canvas) {
            return;
          }
          const rect = canvas.getBoundingClientRect();
          const duration = Math.max(
            props.analysis?.duration ?? 0,
            props.captions[props.captions.length - 1]?.end ?? 0,
            1,
          );
          const time = ((event.clientX - rect.left) / rect.width) * duration;
          props.onSeek(time, { play: false });
        }}
      />
    </div>
  );
});

const TimedTextEditor = memo(function TimedTextEditor({
  value,
  wordIds,
  lookup,
  currentTime,
  showTimingHighlights = true,
  className,
  commitMode = "immediate",
  minHeight = 92,
  showLineGuides = false,
  fallbackTime,
  autoPlayOnSeek = true,
  focusRequest,
  acknowledgedWordIds,
  onChange,
  onSeek,
  onSelectionChange,
  onAcknowledgeWords,
  onFocus,
  onBlur,
  onUndo,
  onRedo,
  onKeyDown,
}: TimedTextEditorProps) {
  const [draftValue, setDraftValue] = useState(value);
  const [selectionRange, setSelectionRange] = useState<{ start: number; end: number }>({ start: 0, end: 0 });
  const [isFocused, setIsFocused] = useState(false);
  const effectiveValue = commitMode === "blur" ? draftValue : value;
  const fragments = useMemo(() => buildMatchedFragments(effectiveValue, wordIds, lookup), [effectiveValue, wordIds, lookup]);
  const textareaRef = useRef<HTMLTextAreaElement | null>(null);
  const lastFocusTokenRef = useRef<number | null>(null);

  useEffect(() => {
    if (commitMode === "blur") {
      setDraftValue(value);
    }
  }, [commitMode, value]);

  useEffect(() => {
    if (!focusRequest || lastFocusTokenRef.current === focusRequest.token) {
      return;
    }

    const textarea = textareaRef.current;
    if (!textarea) {
      return;
    }

    lastFocusTokenRef.current = focusRequest.token;
    const nextCaret = Math.max(0, Math.min(focusRequest.caret, effectiveValue.length));
    textarea.focus();
    textarea.setSelectionRange(nextCaret, nextCaret);
    syncSelection(textarea);
  }, [effectiveValue.length, focusRequest]);

  function syncSelection(target: HTMLTextAreaElement) {
    const nextSelection = {
      start: target.selectionStart ?? 0,
      end: target.selectionEnd ?? 0,
    };
    setSelectionRange(nextSelection);
    onSelectionChange(nextSelection.start, nextSelection.end);

    const left = Math.min(nextSelection.start, nextSelection.end);
    const right = Math.max(nextSelection.start, nextSelection.end);
    const acknowledged = fragments
      .filter((fragment) => {
        if (!fragment.word) {
          return false;
        }
        if (left === right) {
          return left >= fragment.charStart && left <= fragment.charEnd;
        }
        return fragment.charStart < right && fragment.charEnd > left;
      })
      .flatMap((fragment) => (fragment.word ? [fragment.word.id] : []));

    if (acknowledged.length) {
      onAcknowledgeWords?.(acknowledged);
    }
  }

  function selectionSuppressesLowConfidence(fragment: MatchedFragment): boolean {
    if (!isFocused || !fragment.word) {
      return false;
    }

    const left = Math.min(selectionRange.start, selectionRange.end);
    const right = Math.max(selectionRange.start, selectionRange.end);
    if (left === right) {
      return left >= fragment.charStart && left <= fragment.charEnd;
    }

    return fragment.charStart < right && fragment.charEnd > left;
  }

  function handleMouseUp(target: HTMLTextAreaElement) {
    syncSelection(target);
    if ((target.selectionStart ?? 0) !== (target.selectionEnd ?? 0)) {
      return;
    }
    onSeek(timeFromCaret(fragments, target.selectionStart ?? 0, fallbackTime), { play: autoPlayOnSeek });
  }

  const editorStyle = {
    minHeight: `${minHeight}px`,
    "--guide-target-column": `${TARGET_CAPTION_LINE_LENGTH}ch`,
    "--guide-hard-column": `${MAX_CAPTION_LINE_LENGTH}ch`,
  } as CSSProperties;

  return (
    <div className={`timed-editor ${className ?? ""} ${showLineGuides ? "show-line-guides" : ""}`} style={editorStyle}>
      <div className="timed-editor-overlay" aria-hidden="true">
        {fragments.map((fragment) => {
          const classes = ["text-fragment"];
          if (
            fragment.word?.low_confidence &&
            !acknowledgedWordIds?.has(fragment.word.id) &&
            !selectionSuppressesLowConfidence(fragment)
          ) {
            classes.push("is-low-confidence");
          }
          if (showTimingHighlights && fragment.word && currentTime >= fragment.word.start && currentTime <= fragment.word.end) {
            classes.push("is-current");
          }
          return (
            <span key={fragment.key} className={classes.join(" ")}>
              {fragment.text}
            </span>
          );
        })}
        {!value ? <span className="text-fragment"> </span> : null}
      </div>
      <textarea
        ref={textareaRef}
        className="timed-editor-input"
        value={effectiveValue}
        spellCheck={false}
        onChange={(event) => {
          if (commitMode === "blur") {
            setDraftValue(event.target.value);
            return;
          }
          onChange(event.target.value);
        }}
        onKeyDown={(event) => {
          const isUndoShortcut =
            !event.altKey &&
            (event.ctrlKey || event.metaKey) &&
            event.key.toLowerCase() === "z";
          if (isUndoShortcut && onUndo) {
            event.preventDefault();
            if (event.shiftKey) {
              onRedo?.();
            } else {
              onUndo();
            }
            return;
          }

          const isRedoShortcut =
            !event.altKey &&
            !event.shiftKey &&
            (event.ctrlKey || event.metaKey) &&
            event.key.toLowerCase() === "y";
          if (isRedoShortcut && onRedo) {
            event.preventDefault();
            onRedo();
            return;
          }

          onKeyDown?.(event);
        }}
        onFocus={(event) => {
          setIsFocused(true);
          syncSelection(event.currentTarget);
          if (commitMode === "blur") {
            setDraftValue(event.currentTarget.value);
          }
          onFocus?.();
        }}
        onBlur={() => {
          setIsFocused(false);
          if (commitMode === "blur" && draftValue !== value) {
            onChange(draftValue);
          }
          onBlur?.();
        }}
        onSelect={(event) => syncSelection(event.currentTarget)}
        onMouseUp={(event) => handleMouseUp(event.currentTarget)}
      />
    </div>
  );
}, (previous, next) => {
  return (
    previous.value === next.value &&
    previous.wordIds === next.wordIds &&
    previous.lookup === next.lookup &&
    previous.currentTime === next.currentTime &&
    previous.showTimingHighlights === next.showTimingHighlights &&
    previous.className === next.className &&
    previous.commitMode === next.commitMode &&
    previous.minHeight === next.minHeight &&
    previous.showLineGuides === next.showLineGuides &&
    previous.fallbackTime === next.fallbackTime &&
    previous.autoPlayOnSeek === next.autoPlayOnSeek &&
    previous.focusRequest?.token === next.focusRequest?.token &&
    previous.acknowledgedWordIds === next.acknowledgedWordIds
  );
});

function App() {
  const [session, setSession] = useState<TranscriptResponse | null>(null);
  const [history, setHistory] = useState<HistoryState>({ past: [], present: null, future: [] });
  const [viewMode, setViewMode] = useState<ViewMode>("subtitles");
  const [loading, setLoading] = useState(false);
  const [dragActive, setDragActive] = useState(false);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [audioUrl, setAudioUrl] = useState<string | null>(null);
  const [speakerCount, setSpeakerCount] = useState(1);
  const [speakerInputs, setSpeakerInputs] = useState<Speaker[]>(() => buildDefaultSpeakers());
  const [model, setModel] = useState("large-v3");
  const [speakerAssignmentMode, setSpeakerAssignmentMode] = useState<SpeakerAssignmentMode>("segment");
  const [glossaryText, setGlossaryText] = useState("");
  const [findText, setFindText] = useState("");
  const [replaceText, setReplaceText] = useState("");
  const [skipCuts, setSkipCuts] = useState(true);
  const [clickToPlay, setClickToPlay] = useState(true);
  const [followPlayback, setFollowPlayback] = useState(false);
  const [showLineGuides, setShowLineGuides] = useState(false);
  const [showTimingHighlights, setShowTimingHighlights] = useState(true);
  const [sidePanelTab, setSidePanelTab] = useState<SidePanelTab>("guide");
  const [isGuidePanelCollapsed, setIsGuidePanelCollapsed] = useState(DEFAULT_GUIDE_PANEL_COLLAPSED);
  const [extendCaptionsOnExport, setExtendCaptionsOnExport] = useState(false);
  const [normalizeExportTimingTo30Fps, setNormalizeExportTimingTo30Fps] = useState(false);
  const [showSpeakerAttributionOptions, setShowSpeakerAttributionOptions] = useState(false);
  const [removeDisfluencies, setRemoveDisfluencies] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);
  const [statusMessage, setStatusMessage] = useState<string | null>(null);
  const [retranscribing, setRetranscribing] = useState(false);
  const [waveformLoading, setWaveformLoading] = useState(false);
  const [waveformAnalysis, setWaveformAnalysis] = useState<WaveformAnalysisResponse | null>(null);
  const [processedAudio, setProcessedAudio] = useState<{ url: string; filename: string; hasCutTimeline: boolean } | null>(null);
  const [playbackSource, setPlaybackSource] = useState<"original" | "processed">("original");
  const [resumeProjectFile, setResumeProjectFile] = useState<File | null>(null);
  const [resumeAudioFile, setResumeAudioFile] = useState<File | null>(null);
  const [resumeSubtitleFile, setResumeSubtitleFile] = useState<File | null>(null);
  const [selection, setSelection] = useState<BlockSelection | null>(null);
  const [captionFocusRequest, setCaptionFocusRequest] = useState<{ index: number; request: FocusRequest } | null>(null);
  const [acknowledgedLowConfidenceWordIds, setAcknowledgedLowConfidenceWordIds] = useState<string[]>([]);
  const [backendCapabilities, setBackendCapabilities] = useState<BackendCapabilities | null>(null);

  const audioRef = useRef<HTMLAudioElement | null>(null);
  const paragraphRefs = useRef<Array<HTMLElement | null>>([]);
  const captionRefs = useRef<Array<HTMLElement | null>>([]);
  const suppressAutoSpeakerModeRef = useRef(false);
  const autosaveReadyRef = useRef(false);
  const autosaveTimerRef = useRef<number | null>(null);
  const lastTextEditRef = useRef<{ kind: SelectionKind; index: number; timestamp: number } | null>(null);
  const activeWorkspace = history.present;
  const activeEditor = activeWorkspace?.editor ?? null;
  const currentAudioFilename = selectedFile?.name ?? session?.audio_filename ?? null;

  useEffect(() => {
    let cancelled = false;

    async function loadCapabilities() {
      try {
        const response = await fetch(`${API_BASE_URL}/api/capabilities`);
        if (!response.ok) {
          return;
        }
        const payload = (await response.json()) as BackendCapabilities;
        if (!cancelled) {
          setBackendCapabilities(payload);
        }
      } catch {
        // Leave capability-driven hints hidden if the backend is unavailable.
      }
    }

    void loadCapabilities();
    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    try {
      const raw = window.localStorage.getItem(AUTOSAVE_STORAGE_KEY);
      if (!raw) {
        return;
      }

      const saved = JSON.parse(raw) as LegacyPersistedWorkspace;
      if (saved.version !== 4 && saved.version !== AUTOSAVE_STORAGE_VERSION) {
        window.localStorage.removeItem(AUTOSAVE_STORAGE_KEY);
        return;
      }

      const restoredEditor = saved.editor ?? null;
      const restoredSession = saved.session ?? null;
      const restoredSpeakers = normalizeSpeakers(
        saved.speakerInputs?.length
          ? saved.speakerInputs
          : restoredEditor?.speakers.length
            ? restoredEditor.speakers
            : restoredSession?.speakers.length
              ? restoredSession.speakers
              : buildDefaultSpeakers(),
      );

      if (restoredSession || restoredEditor) {
        restorePersistedWorkspace(
          {
            version: AUTOSAVE_STORAGE_VERSION,
            session: restoredSession,
            editor: restoredEditor,
            model: typeof saved.model === "string" ? saved.model : model,
            speakerCount: Math.max(1, saved.speakerCount ?? restoredSpeakers.length),
            speakerInputs: normalizeSpeakers(restoredSpeakers),
            speakerAssignmentMode: saved.speakerAssignmentMode === "word" ? "word" : "segment",
            glossaryText: mergeVocabularyTexts(
              typeof saved.glossaryText === "string" ? saved.glossaryText : "",
              typeof saved.hotwords === "string" ? saved.hotwords : "",
            ),
            skipCuts: typeof saved.skipCuts === "boolean" ? saved.skipCuts : true,
            clickToPlay: typeof saved.clickToPlay === "boolean" ? saved.clickToPlay : true,
            followPlayback: typeof saved.followPlayback === "boolean" ? saved.followPlayback : false,
            showLineGuides: typeof saved.showLineGuides === "boolean" ? saved.showLineGuides : false,
            showTimingHighlights: typeof saved.showTimingHighlights === "boolean" ? saved.showTimingHighlights : true,
            viewMode: saved.viewMode === "transcript" ? "transcript" : "subtitles",
            sidePanelTab:
              saved.sidePanelTab === "jargon" || saved.sidePanelTab === "qa" || saved.sidePanelTab === "guide" || saved.sidePanelTab === "master"
                ? saved.sidePanelTab
                : "guide",
            isGuidePanelCollapsed: DEFAULT_GUIDE_PANEL_COLLAPSED,
            extendCaptionsOnExport: typeof saved.extendCaptionsOnExport === "boolean" ? saved.extendCaptionsOnExport : false,
            normalizeExportTimingTo30Fps:
              typeof saved.normalizeExportTimingTo30Fps === "boolean" ? saved.normalizeExportTimingTo30Fps : false,
            showSpeakerAttributionOptions:
              typeof saved.showSpeakerAttributionOptions === "boolean" ? saved.showSpeakerAttributionOptions : false,
            removeDisfluencies: typeof saved.removeDisfluencies === "boolean" ? saved.removeDisfluencies : false,
            acknowledgedLowConfidenceWordIds: Array.isArray(saved.acknowledgedLowConfidenceWordIds)
              ? saved.acknowledgedLowConfidenceWordIds.filter((item): item is string => typeof item === "string")
              : [],
          },
          {
            statusMessage: "Restored the last autosaved workspace. Reattach the audio file if you need playback.",
          },
        );
      }
    } catch {
      window.localStorage.removeItem(AUTOSAVE_STORAGE_KEY);
    } finally {
      autosaveReadyRef.current = true;
    }
  }, []);

  useEffect(() => {
    return () => {
      if (audioUrl) {
        URL.revokeObjectURL(audioUrl);
      }
    };
  }, [audioUrl]);

  useEffect(() => {
    setSpeakerInputs((current) => {
      const next = Array.from(
        { length: speakerCount },
        (_, index) => current[index] ?? { id: index, name: `Speaker ${index + 1}`, show_attribution: true },
      );
      return normalizeSpeakers(next);
    });
  }, [speakerCount]);

  useEffect(() => {
    if (suppressAutoSpeakerModeRef.current) {
      suppressAutoSpeakerModeRef.current = false;
      return;
    }

    if (speakerCount <= 1) {
      setSpeakerAssignmentMode("segment");
      return;
    }

    setSpeakerAssignmentMode("word");
  }, [speakerCount]);

  useEffect(() => {
    if (!selection || !activeEditor) {
      return;
    }
    if (selection.kind === "caption" && selection.index >= activeEditor.captions.length) {
      setSelection(null);
    }
    if (selection.kind === "paragraph" && selection.index >= activeEditor.paragraphs.length) {
      setSelection(null);
    }
  }, [activeEditor, selection]);

  useEffect(() => {
    paragraphRefs.current.length = activeEditor?.paragraphs.length ?? 0;
    captionRefs.current.length = activeEditor?.captions.length ?? 0;
  }, [activeEditor]);

  useEffect(() => {
    if (!autosaveReadyRef.current) {
      return;
    }

    if (autosaveTimerRef.current !== null) {
      window.clearTimeout(autosaveTimerRef.current);
      autosaveTimerRef.current = null;
    }

    autosaveTimerRef.current = window.setTimeout(() => {
      const snapshot = buildPersistedWorkspaceSnapshot();
      if (!snapshot) {
        window.localStorage.removeItem(AUTOSAVE_STORAGE_KEY);
        return;
      }

      persistWorkspace(snapshot);
      autosaveTimerRef.current = null;
    }, AUTOSAVE_DELAY_MS);

    return () => {
      if (autosaveTimerRef.current !== null) {
        window.clearTimeout(autosaveTimerRef.current);
        autosaveTimerRef.current = null;
      }
    };
  }, [acknowledgedLowConfidenceWordIds, activeWorkspace, clickToPlay, extendCaptionsOnExport, followPlayback, glossaryText, isGuidePanelCollapsed, model, normalizeExportTimingTo30Fps, removeDisfluencies, session, showLineGuides, showSpeakerAttributionOptions, showTimingHighlights, sidePanelTab, skipCuts, speakerAssignmentMode, speakerCount, speakerInputs, viewMode]);

  useEffect(() => {
    const audio = audioRef.current;
    if (!audio || !activeEditor) {
      return;
    }

    const onTimeUpdate = () => {
      const time = audio.currentTime;
      setCurrentTime(time);
      if (!skipCuts) {
        return;
      }
      const activeCut = activeEditor.guideBlocks.find((block) => block.skip && time >= block.start && time < block.end);
      if (activeCut) {
        audio.currentTime = Math.min(activeCut.end + 0.02, audio.duration || activeCut.end + 0.02);
      }
    };

    audio.addEventListener("timeupdate", onTimeUpdate);
    return () => audio.removeEventListener("timeupdate", onTimeUpdate);
  }, [activeEditor, skipCuts]);

  const activeWords = activeWorkspace?.words ?? session?.words ?? [];
  const transcriptWords = useMemo(() => new Map(activeWords.map((word) => [word.id, word])), [activeWords]);
  const transcriptionVocabulary = useMemo(() => mergeVocabularyTexts(glossaryText).replace(/\n/g, ", "), [glossaryText]);
  const glossaryTerms = useMemo(() => parseGlossaryTerms(glossaryText), [glossaryText]);
  const glossaryMatches = useMemo(
    () => (activeEditor ? findCaptionGlossaryMatches(activeEditor.captions, glossaryText) : []),
    [activeEditor, glossaryText],
  );
  const glossaryMatchedCaptionCount = useMemo(
    () => glossaryMatches.filter((match) => match.exactTerms.length || match.fuzzyTerms.length).length,
    [glossaryMatches],
  );
  const jargonCandidates = useMemo(
    () => (activeEditor ? detectJargonCandidates(activeWords, activeEditor.captions, glossaryText) : []),
    [activeEditor, activeWords, glossaryText],
  );
  const qaReport = useMemo(
    () => buildQaReport(activeEditor?.captions ?? [], activeWords, glossaryMatches),
    [activeEditor, activeWords, glossaryMatches],
  );
  const sidePanelMeta =
    sidePanelTab === "guide"
      ? {
          eyebrow: "Edit guide",
          title: "Cut blocks",
          detail: `${activeEditor?.guideBlocks.length ?? 0} blocks`,
        }
      : sidePanelTab === "jargon"
        ? {
            eyebrow: "Jargon",
            title: "Glossary",
            detail: `${jargonCandidates.length} candidates`,
          }
        : sidePanelTab === "master"
          ? {
              eyebrow: "Master",
              title: "Audio post production",
              detail: processedAudio ? "Master ready" : "Not processed",
            }
          : {
              eyebrow: "QA",
              title: "Report",
              detail: `${qaReport.summary.issueCount} issues`,
            };
  const collapsedPanelMetaDetail =
    sidePanelTab === "guide"
      ? skipCuts
        ? "Playback skip on"
        : "Playback skip off"
      : sidePanelTab === "jargon"
        ? `${glossaryTerms.length} term${glossaryTerms.length === 1 ? "" : "s"}`
        : sidePanelTab === "master"
          ? processedAudio
            ? "Processed audio loaded"
            : "Local processing"
          : `${qaReport.summary.flaggedCaptionCount} caption${qaReport.summary.flaggedCaptionCount === 1 ? "" : "s"}`;
  const multiSpeaker = (activeEditor?.speakers.length ?? speakerInputs.length) > 1;

  const activeCaptionIndex = useMemo(() => {
    if (!activeEditor) {
      return -1;
    }
    return activeEditor.captions.findIndex((caption) => currentTime >= caption.start && currentTime <= caption.end);
  }, [activeEditor, currentTime]);

  const activeParagraphIndex = useMemo(() => {
    if (!activeEditor) {
      return -1;
    }
    return activeEditor.paragraphs.findIndex((paragraph) => currentTime >= paragraph.start && currentTime <= paragraph.end);
  }, [activeEditor, currentTime]);

  const selectedCaptionIndex = selection?.kind === "caption" ? selection.index : activeCaptionIndex;
  const showPyannoteSetupHint = speakerCount > 1 && backendCapabilities?.diarization_configured === false;
  const activeWarnings = activeWorkspace?.warnings ?? session?.warnings ?? [];
  const speakerTimelineEvents = useMemo(
    () => detectSpeakerTimelineEvents(activeEditor?.captions ?? [], activeWords, waveformAnalysis?.speech_spans ?? []),
    [activeEditor, activeWords, waveformAnalysis],
  );
  const focusTokenRef = useRef(0);
  const lastFollowedBlockRef = useRef<string | null>(null);
  const acknowledgedWordIdSet = useMemo(
    () => new Set(acknowledgedLowConfidenceWordIds),
    [acknowledgedLowConfidenceWordIds],
  );

  useEffect(() => {
    if (!followPlayback || !activeEditor) {
      lastFollowedBlockRef.current = null;
      return;
    }

    const activeIndex = viewMode === "transcript" ? activeParagraphIndex : activeCaptionIndex;
    if (activeIndex < 0) {
      return;
    }

    const key = `${viewMode}:${activeIndex}`;
    if (lastFollowedBlockRef.current === key) {
      return;
    }

    lastFollowedBlockRef.current = key;
    const element = viewMode === "transcript" ? paragraphRefs.current[activeIndex] : captionRefs.current[activeIndex];
    scrollIntoViewCentered(element);
  }, [activeCaptionIndex, activeEditor, activeParagraphIndex, followPlayback, viewMode]);

  function requestCaptionFocus(index: number, caret: number) {
    focusTokenRef.current += 1;
    setCaptionFocusRequest({
      index,
      request: {
        token: focusTokenRef.current,
        caret,
      },
    });
    setSelection({ kind: "caption", index, start: caret, end: caret, text: "" });
  }

  function acknowledgeLowConfidenceWords(wordIds: string[]) {
    setAcknowledgedLowConfidenceWordIds((current) => {
      const next = new Set(current);
      let changed = false;
      wordIds.forEach((wordId) => {
        if (!next.has(wordId)) {
          next.add(wordId);
          changed = true;
        }
      });
      return changed ? Array.from(next) : current;
    });
  }

  useEffect(() => {
    const onKeyDown = (event: globalThis.KeyboardEvent) => {
      const isUndoShortcut =
        !event.altKey &&
        (event.ctrlKey || event.metaKey) &&
        event.key.toLowerCase() === "z";
      if (isUndoShortcut && !isEditableTarget(event.target)) {
        event.preventDefault();
        if (event.shiftKey) {
          redo();
        } else {
          undo();
        }
        return;
      }

      const isRedoShortcut =
        !event.altKey &&
        !event.shiftKey &&
        (event.ctrlKey || event.metaKey) &&
        event.key.toLowerCase() === "y";
      if (isRedoShortcut && !isEditableTarget(event.target)) {
        event.preventDefault();
        redo();
        return;
      }

      if (event.code !== "Space" || event.altKey || event.metaKey) {
        return;
      }

      if (event.ctrlKey && !event.shiftKey) {
        event.preventDefault();
        togglePlayback();
        return;
      }

      if (event.shiftKey && !event.ctrlKey && !isEditableTarget(event.target)) {
        event.preventDefault();
        setClickToPlay((current) => !current);
      }
    };

    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [currentAudioFilename]);

  function buildPersistedWorkspaceSnapshot(): PersistedWorkspace | null {
    if (!session && !activeWorkspace) {
      return null;
    }

    return {
      version: AUTOSAVE_STORAGE_VERSION,
      session: buildSessionSnapshot(session, activeWorkspace),
      editor: activeEditor ? cloneEditorState(activeEditor) : null,
      model,
      speakerCount,
      speakerInputs: normalizeSpeakers(speakerInputs),
      speakerAssignmentMode,
      glossaryText,
      skipCuts,
      clickToPlay,
      followPlayback,
      showLineGuides,
      showTimingHighlights,
      viewMode,
      sidePanelTab,
      isGuidePanelCollapsed,
      extendCaptionsOnExport,
      normalizeExportTimingTo30Fps,
      showSpeakerAttributionOptions,
      removeDisfluencies,
      acknowledgedLowConfidenceWordIds,
    };
  }

  function replaceWorkspace(next: WorkspaceState | null) {
    setHistory({
      past: [],
      present: next ? cloneWorkspaceState(next) : null,
      future: [],
    });
  }

  function restorePersistedWorkspace(
    persisted: PersistedWorkspace | LegacyPersistedWorkspace,
    options?: { statusMessage?: string; audioFile?: File | null },
  ) {
    const legacyHotwords = "hotwords" in persisted && typeof persisted.hotwords === "string" ? persisted.hotwords : "";
    const restoredSpeakerInputs = normalizeSpeakers(
      persisted.speakerInputs?.length
        ? persisted.speakerInputs
        : persisted.editor?.speakers.length
          ? persisted.editor.speakers
          : persisted.session?.speakers.length
            ? persisted.session.speakers
            : buildDefaultSpeakers(),
    );
    const restoredGlossaryText = mergeVocabularyTexts(
      typeof persisted.glossaryText === "string" ? persisted.glossaryText : "",
      legacyHotwords,
    );

    suppressAutoSpeakerModeRef.current = true;
    setSession(
      persisted.session
        ? {
            ...persisted.session,
            speakers: normalizeSpeakers(persisted.session.speakers),
          }
        : null,
    );
    setSpeakerInputs(restoredSpeakerInputs);
    setSpeakerCount(Math.max(1, persisted.speakerCount ?? restoredSpeakerInputs.length));
    setModel(typeof persisted.model === "string" ? persisted.model : "large-v3");
    setSpeakerAssignmentMode(persisted.speakerAssignmentMode === "word" ? "word" : "segment");
    setGlossaryText(restoredGlossaryText);
    setSkipCuts(typeof persisted.skipCuts === "boolean" ? persisted.skipCuts : true);
    setClickToPlay(typeof persisted.clickToPlay === "boolean" ? persisted.clickToPlay : true);
    setFollowPlayback(Boolean(persisted.followPlayback));
    setShowLineGuides(Boolean(persisted.showLineGuides));
    setShowTimingHighlights(Boolean(persisted.showTimingHighlights));
    setViewMode(persisted.viewMode === "transcript" ? "transcript" : "subtitles");
    setSidePanelTab(
      persisted.sidePanelTab === "jargon" || persisted.sidePanelTab === "qa" || persisted.sidePanelTab === "guide" || persisted.sidePanelTab === "master"
        ? persisted.sidePanelTab
        : "guide",
    );
    setIsGuidePanelCollapsed(DEFAULT_GUIDE_PANEL_COLLAPSED);
    setExtendCaptionsOnExport(Boolean(persisted.extendCaptionsOnExport));
    setNormalizeExportTimingTo30Fps(Boolean(persisted.normalizeExportTimingTo30Fps));
    setShowSpeakerAttributionOptions(Boolean(persisted.showSpeakerAttributionOptions));
    setRemoveDisfluencies(Boolean(persisted.removeDisfluencies));
    setAcknowledgedLowConfidenceWordIds(
      Array.isArray(persisted.acknowledgedLowConfidenceWordIds)
        ? persisted.acknowledgedLowConfidenceWordIds.filter((item): item is string => typeof item === "string")
        : [],
    );
    setSelection(null);
    setCurrentTime(0);
    if (options && "audioFile" in options) {
      setAudioFile(options.audioFile ?? null);
    }

    const restoredEditor =
      persisted.editor ??
      (persisted.session
        ? {
            captions: persisted.session.captions,
            guideBlocks: persisted.session.guide_blocks,
            speakers: persisted.session.speakers,
            paragraphs: persisted.session.paragraphs,
          }
        : null);
    const restoredWorkspace =
      restoredEditor || persisted.session
        ? buildWorkspaceState(
            restoredEditor ?? {
              captions: [],
              guideBlocks: [],
              speakers: persisted.session?.speakers ?? buildDefaultSpeakers(),
              paragraphs: [],
            },
            persisted.session?.words ?? [],
            persisted.session?.warnings ?? [],
            persisted.session?.language ?? null,
          )
        : null;
    replaceWorkspace(restoredWorkspace);

    if (options?.statusMessage) {
      setStatusMessage(options.statusMessage);
    }
  }

  function shouldCreateTextEditCheckpoint(kind: SelectionKind, index: number): boolean {
    const now = Date.now();
    const previous = lastTextEditRef.current;
    const createCheckpoint =
      !previous ||
      previous.kind !== kind ||
      previous.index !== index ||
      now - previous.timestamp > TEXT_EDIT_CHECKPOINT_MS;

    lastTextEditRef.current = { kind, index, timestamp: now };
    return createCheckpoint;
  }

  function commitTextEdit(
    kind: SelectionKind,
    index: number,
    updater: (editor: EditorState) => EditorState,
  ) {
    setHistory((current) => {
      if (!current.present) {
        return current;
      }

      const nextEditor = updater(current.present.editor);
      if (nextEditor === current.present.editor) {
        return current;
      }

      const createCheckpoint = shouldCreateTextEditCheckpoint(kind, index);
      return {
        past: createCheckpoint
          ? [...current.past, cloneWorkspaceState(current.present)].slice(-120)
          : current.past,
        present: {
          ...current.present,
          editor: nextEditor,
        },
        future: [],
      };
    });
  }

  function commit(mutator: (draft: EditorState) => void, options?: CommitOptions) {
    lastTextEditRef.current = null;
    setHistory((current) => {
      if (!current.present) {
        return current;
      }

      const next = cloneWorkspaceState(current.present);
      mutator(next.editor);
      if (options?.transformWords) {
        next.words = options.transformWords(next.words);
      }
      if (options?.syncCaptionTiming) {
        next.editor.captions = syncCaptionWordAssignments(next.editor.captions, options?.wordSource ?? next.words);
      }
      next.editor.paragraphs = buildParagraphsFromCaptions(next.editor.captions);
      if (options?.warnings) {
        next.warnings = cloneWarnings(options.warnings);
      }
      if (options?.language !== undefined) {
        next.language = options.language;
      }
      return {
        past: [...current.past, cloneWorkspaceState(current.present)].slice(-120),
        present: next,
        future: [],
      };
    });
  }

  function undo() {
    lastTextEditRef.current = null;
    setHistory((current) => {
      if (!current.past.length || !current.present) {
        return current;
      }
      const previous = current.past[current.past.length - 1];
      return {
        past: current.past.slice(0, -1),
        present: cloneWorkspaceState(previous),
        future: [cloneWorkspaceState(current.present), ...current.future].slice(0, 120),
      };
    });
    setSelection(null);
  }

  function redo() {
    lastTextEditRef.current = null;
    setHistory((current) => {
      if (!current.future.length || !current.present) {
        return current;
      }
      const [next, ...rest] = current.future;
      return {
        past: [...current.past, cloneWorkspaceState(current.present)].slice(-120),
        present: cloneWorkspaceState(next),
        future: rest,
      };
    });
    setSelection(null);
  }

  function setAudioFile(file: File | null) {
    setSelectedFile(file);
    setWaveformAnalysis(null);
    setCurrentTime(0);
    setProcessedAudio(null);
    setPlaybackSource("original");
    if (audioUrl) {
      URL.revokeObjectURL(audioUrl);
    }
    setAudioUrl(file ? URL.createObjectURL(file) : null);
  }

  async function refreshWaveformFromMaster(token: string) {
    try {
      const response = await fetch(`${API_BASE_URL}/api/master/${token}/waveform`);
      if (response.ok) {
        setWaveformAnalysis((await response.json()) as WaveformAnalysisResponse);
      }
    } catch {
      // The old waveform stays; it is only a visual aid.
    }
  }

  function handleMasteringProcessed(result: MasteringResult, url: string) {
    const previousTime = audioRef.current?.currentTime ?? 0;
    const hasCutTimeline = result.duration_after < result.duration_before - 0.01;
    setProcessedAudio({ url, filename: result.output_filename, hasCutTimeline });
    setPlaybackSource("processed");
    if (!hasCutTimeline) {
      // Same timeline, so keep the listening position and refresh the waveform.
      window.setTimeout(() => {
        if (audioRef.current) {
          audioRef.current.currentTime = previousTime;
        }
      }, 150);
      void refreshWaveformFromMaster(result.token);
    }
    setStatusMessage("Mastering finished. Playback now uses the processed audio.");
  }

  function handleApplyCutsToSubtitles(result: MasteringResult) {
    const cuts = [...result.cut_list].sort((a, b) => a.start - b.start);
    if (!cuts.length || !activeWorkspace) {
      return;
    }
    const remappedWords = remapWords(activeWorkspace.words, cuts);
    const keptWordIds = new Set(remappedWords.map((word) => word.id));
    commit(
      (draft) => {
        draft.captions = remapCaptions(draft.captions, cuts, keptWordIds);
        draft.guideBlocks = remapGuideBlocks(draft.guideBlocks, cuts);
      },
      { transformWords: () => remappedWords },
    );
    void refreshWaveformFromMaster(result.token);
    setStatusMessage("Cuts applied. Subtitles now match the processed audio (undo to restore).");
  }

  function buildTranscriptionFormData(audioFile: File): FormData {
    const effectiveSpeakerAssignmentMode: SpeakerAssignmentMode = speakerCount > 1 ? speakerAssignmentMode : "segment";
    const formData = new FormData();
    formData.append("audio", audioFile);
    formData.append("model", model);
    formData.append("speaker_count", String(speakerCount));
    formData.append(
      "speakers_json",
      JSON.stringify(normalizeSpeakers(speakerInputs).map((speaker) => ({ id: speaker.id, name: speaker.name }))),
    );
    formData.append("speaker_assignment_mode", effectiveSpeakerAssignmentMode);
    formData.append("remove_disfluencies", String(removeDisfluencies));
    if (transcriptionVocabulary.trim()) {
      formData.append("hotwords", transcriptionVocabulary.trim());
    }
    return formData;
  }

  async function requestTranscription(audioFile: File): Promise<TranscriptResponse> {
    const response = await fetch(`${API_BASE_URL}/api/transcribe`, {
      method: "POST",
      body: buildTranscriptionFormData(audioFile),
    });

    if (!response.ok) {
      const raw = await response.text();
      let message = raw;
      try {
        const parsed = JSON.parse(raw) as { detail?: string };
        message = parsed.detail ?? raw;
      } catch {
        // keep raw text
      }
      throw new Error(message || `Transcription failed with status ${response.status}`);
    }

    return (await response.json()) as TranscriptResponse;
  }

  async function requestWaveformAnalysis(audioFile: File): Promise<WaveformAnalysisResponse> {
    const formData = new FormData();
    formData.append("audio", audioFile);

    const response = await fetch(`${API_BASE_URL}/api/analyze-waveform`, {
      method: "POST",
      body: formData,
    });

    if (!response.ok) {
      const raw = await response.text();
      let message = raw;
      try {
        const parsed = JSON.parse(raw) as { detail?: string };
        message = parsed.detail ?? raw;
      } catch {
        // keep raw text
      }
      throw new Error(message || `Waveform analysis failed with status ${response.status}`);
    }

    return (await response.json()) as WaveformAnalysisResponse;
  }

  function resetWorkspace() {
    lastTextEditRef.current = null;
    setSession(null);
    setHistory({ past: [], present: null, future: [] });
    setResumeProjectFile(null);
    setResumeAudioFile(null);
    setResumeSubtitleFile(null);
    setSpeakerCount(1);
    setSpeakerInputs(buildDefaultSpeakers());
    setSpeakerAssignmentMode("segment");
    setGlossaryText("");
    setFindText("");
    setReplaceText("");
    setFollowPlayback(false);
    setShowLineGuides(false);
    setShowTimingHighlights(true);
    setSidePanelTab("guide");
    setIsGuidePanelCollapsed(false);
    setExtendCaptionsOnExport(false);
    setNormalizeExportTimingTo30Fps(false);
    setShowSpeakerAttributionOptions(false);
    setRemoveDisfluencies(false);
    setAcknowledgedLowConfidenceWordIds([]);
    setCurrentTime(0);
    setStatusMessage("Workspace reset.");
    setSelection(null);
    setViewMode("subtitles");
    setWaveformAnalysis(null);
    setWaveformLoading(false);
    setAudioFile(null);
    window.localStorage.removeItem(AUTOSAVE_STORAGE_KEY);
  }

  function seekAudio(time: number, options?: { play?: boolean }) {
    const audio = audioRef.current;
    if (!audio) {
      return;
    }
    audio.currentTime = Math.max(0, time);
    if (options?.play ?? true) {
      void audio.play();
    }
  }

  function togglePlayback() {
    const audio = audioRef.current;
    if (!audio || !currentAudioFilename) {
      return;
    }
    if (audio.paused) {
      void audio.play();
      return;
    }
    audio.pause();
  }

  function scrollIntoViewCentered(element: HTMLElement | null) {
    if (!element) {
      return;
    }
    element.scrollIntoView({ behavior: "smooth", block: "center" });
  }

  function jumpToCurrentTranscript() {
    const index = activeEditor ? findNearestTimedIndex(activeEditor.paragraphs, currentTime) : -1;
    if (index < 0) {
      return;
    }
    scrollIntoViewCentered(paragraphRefs.current[index]);
  }

  function jumpToCurrentSubtitle() {
    const index = activeEditor ? findNearestTimedIndex(activeEditor.captions, currentTime) : -1;
    if (index < 0) {
      return;
    }
    scrollIntoViewCentered(captionRefs.current[index]);
  }

  function reflowAllCaptions() {
    commit((draft) => {
      draft.captions = draft.captions.map((caption) => ({
        ...caption,
        lines: normalizeCaptionLines(caption.lines),
      }));
    });
    setStatusMessage("Reflowed caption line breaks.");
  }

  async function handleAnalyzeWaveform() {
    if (!selectedFile) {
      setStatusMessage("Choose an audio file first.");
      return;
    }

    setWaveformLoading(true);
    setStatusMessage("Analyzing waveform with FFmpeg...");
    try {
      const analysis = await requestWaveformAnalysis(selectedFile);
      setWaveformAnalysis(analysis);
      setStatusMessage(
        `Waveform analyzed: ${analysis.speech_spans.length} speech region${analysis.speech_spans.length === 1 ? "" : "s"} detected.`,
      );
    } catch (error) {
      const message = error instanceof Error ? error.message : "Unknown error";
      setStatusMessage(message);
    } finally {
      setWaveformLoading(false);
    }
  }

  function handleAlignCaptionsToWaveform() {
    if (!activeEditor) {
      setStatusMessage("Transcribe a file or load a session first.");
      return;
    }
    if (!waveformAnalysis) {
      setStatusMessage("Analyze the waveform before snapping subtitle timing.");
      return;
    }

    const result = alignCaptionsToWaveformSpans(
      activeEditor.captions,
      waveformAnalysis.speech_spans,
      transcriptWords,
      waveformAnalysis.duration,
    );

    if (!result.edgeAdjustmentCount) {
      setStatusMessage("Waveform snap found no subtitle edges close enough to move safely.");
      return;
    }

    commit((draft) => {
      draft.captions = result.captions.map((caption) => ({
        ...caption,
        lines: [...caption.lines],
        word_ids: [...caption.word_ids],
      }));
    });
    setStatusMessage(
      `Waveform snap adjusted ${result.edgeAdjustmentCount} edge${result.edgeAdjustmentCount === 1 ? "" : "s"} across ${result.captionAdjustmentCount} subtitle${result.captionAdjustmentCount === 1 ? "" : "s"}.`,
    );
  }

  async function handleTranscribe() {
    if (!selectedFile) {
      setStatusMessage("Choose an audio file first.");
      return;
    }

    setLoading(true);
    setStatusMessage("Running WhisperX. Large models on long files will take time even on GPU.");
    try {
      const payload = await requestTranscription(selectedFile);
      const normalizedPayload = {
        ...payload,
        speakers: normalizeSpeakers(payload.speakers),
      };
      suppressAutoSpeakerModeRef.current = true;
      setSession(normalizedPayload);
      setSpeakerCount(normalizedPayload.speakers.length);
      setSpeakerInputs(normalizedPayload.speakers);
      setSpeakerAssignmentMode(normalizedPayload.speaker_assignment_mode);
      replaceWorkspace(buildWorkspaceFromSession(normalizedPayload));
      setAcknowledgedLowConfidenceWordIds([]);
      setViewMode("subtitles");
      setSelection(null);
      setStatusMessage(
        `Transcribed with ${normalizedPayload.model}${normalizedPayload.gpu_enabled ? " on GPU" : " on CPU"} using ${normalizedPayload.speaker_assignment_mode}-level speaker assignment.`,
      );
    } catch (error) {
      const message = error instanceof Error ? error.message : "Unknown error";
      setStatusMessage(message);
    } finally {
      setLoading(false);
    }
  }

  async function handleResumeSession(options?: { retimeCaptions?: boolean }) {
    if (!resumeAudioFile || !resumeSubtitleFile) {
      setStatusMessage("Choose both an audio file and an SRT subtitle file.");
      return;
    }

    const text = await resumeSubtitleFile.text();
    const captions = parseSrt(text);
    if (!captions.length) {
      setStatusMessage("Could not parse any captions from that SRT file.");
      return;
    }

    setLoading(true);
    setStatusMessage(
      options?.retimeCaptions
        ? "Running WhisperX on the uploaded audio and retiming the imported captions."
        : "Running WhisperX on the uploaded audio while preserving the imported SRT timing.",
    );
    try {
      const payload = await requestTranscription(resumeAudioFile);
      const importedSession = buildRealignedImportedSession(resumeAudioFile.name, captions, payload, options);
      const normalizedImportedSession = {
        ...importedSession,
        speakers: normalizeSpeakers(importedSession.speakers),
      };
      restorePersistedWorkspace(
        {
          version: AUTOSAVE_STORAGE_VERSION,
          session: normalizedImportedSession,
          editor: {
            captions: normalizedImportedSession.captions,
            guideBlocks: normalizedImportedSession.guide_blocks,
            speakers: normalizedImportedSession.speakers,
            paragraphs: normalizedImportedSession.paragraphs,
          },
          model,
          speakerCount: normalizedImportedSession.speakers.length,
          speakerInputs: normalizedImportedSession.speakers,
          speakerAssignmentMode: normalizedImportedSession.speaker_assignment_mode,
          glossaryText,
          skipCuts,
          clickToPlay,
          followPlayback,
          showLineGuides,
          showTimingHighlights,
          viewMode: "subtitles",
          sidePanelTab,
          isGuidePanelCollapsed,
          extendCaptionsOnExport,
          normalizeExportTimingTo30Fps,
          showSpeakerAttributionOptions,
          removeDisfluencies,
          acknowledgedLowConfidenceWordIds: [],
        },
        {
          audioFile: resumeAudioFile,
        },
      );
      setStatusMessage(`Loaded ${captions.length} captions from ${resumeSubtitleFile.name} and rebuilt timing from the audio.`);
    } catch (error) {
      const message = error instanceof Error ? error.message : "Unknown error";
      setStatusMessage(message);
    } finally {
      setLoading(false);
    }
  }

  function updateCaptionText(index: number, value: string) {
    if (activeEditor?.captions[index]?.word_ids.length) {
      acknowledgeLowConfidenceWords(activeEditor.captions[index].word_ids);
    }
    commitTextEdit("caption", index, (editor) => applyCaptionTextEdit(editor, index, value));
  }

  function updateParagraphText(index: number, value: string) {
    if (activeEditor?.paragraphs[index]?.word_ids.length) {
      acknowledgeLowConfidenceWords(activeEditor.paragraphs[index].word_ids);
    }
    commitTextEdit("paragraph", index, (editor) => applyParagraphTextEdit(editor, index, value));
  }

  function updateSpeakerName(index: number, name: string) {
    const targetSpeakerId = activeEditor?.speakers[index]?.id ?? speakerInputs[index]?.id ?? index;
    setSpeakerInputs((current) =>
      normalizeSpeakers(current.map((speaker, itemIndex) => (itemIndex === index ? { ...speaker, name } : speaker))),
    );
    setSession((current) => {
      if (!current) {
        return current;
      }
      return {
        ...current,
        speakers: normalizeSpeakers(current.speakers.map((speaker) => (speaker.id === targetSpeakerId ? { ...speaker, name } : speaker))),
      };
    });
    commit((draft) => {
      if (draft.speakers[index]) {
        draft.speakers[index].name = name;
      }
      draft.captions = draft.captions.map((caption) => (caption.speaker_id === targetSpeakerId ? { ...caption, speaker_name: name } : caption));
    }, {
      transformWords: (words) => words.map((word) => (word.speaker_id === targetSpeakerId ? { ...word, speaker_name: name } : word)),
    });
  }

  function updateSpeakerAttribution(index: number, showAttribution: boolean) {
    const targetSpeakerId = activeEditor?.speakers[index]?.id ?? speakerInputs[index]?.id ?? index;
    setSpeakerInputs((current) =>
      normalizeSpeakers(current.map((speaker, itemIndex) => (itemIndex === index ? { ...speaker, show_attribution: showAttribution } : speaker))),
    );
    setSession((current) => {
      if (!current) {
        return current;
      }
      return {
        ...current,
        speakers: normalizeSpeakers(
          current.speakers.map((speaker) =>
            speaker.id === targetSpeakerId ? { ...speaker, show_attribution: showAttribution } : speaker,
          ),
        ),
      };
    });
    commit((draft) => {
      if (draft.speakers[index]) {
        draft.speakers[index].show_attribution = showAttribution;
      }
    });
  }

  function addTermsToGlossary(terms: string[]) {
    setGlossaryText((current) => appendGlossaryTerms(current, terms));
  }

  function speakerAttributionEnabled(speakerId: number | null): boolean {
    if (speakerId === null) {
      return true;
    }
    const speaker = activeEditor?.speakers.find((item) => item.id === speakerId) ?? speakerInputs.find((item) => item.id === speakerId);
    return speaker?.show_attribution !== false;
  }

  function jumpToCaption(index: number) {
    if (!activeEditor?.captions[index]) {
      return;
    }
    setViewMode("subtitles");
    setSelection({
      kind: "caption",
      index,
      start: 0,
      end: 0,
      text: captionValue(activeEditor.captions[index]),
    });
    seekAudio(activeEditor.captions[index].start, { play: false });
    requestCaptionFocus(index, 0);
    scrollIntoViewCentered(captionRefs.current[index]);
  }

  async function requestRetranscribedRange(target: RetranscribeTarget): Promise<RetranscribeRangeResponse> {
    if (!selectedFile) {
      throw new Error("Choose an audio file first.");
    }

    const formData = new FormData();
    formData.append("audio", selectedFile);
    formData.append("model", model);
    formData.append("start_seconds", String(target.start));
    formData.append("end_seconds", String(target.end));
    formData.append("remove_disfluencies", String(removeDisfluencies));
    if (transcriptionVocabulary.trim()) {
      formData.append("hotwords", transcriptionVocabulary.trim());
    }

    const response = await fetch(`${API_BASE_URL}/api/retranscribe-range`, {
      method: "POST",
      body: formData,
    });

    if (!response.ok) {
      const raw = await response.text();
      let message = raw;
      try {
        const parsed = JSON.parse(raw) as { detail?: string };
        message = parsed.detail ?? raw;
      } catch {
        // keep raw text
      }
      throw new Error(message || `Retranscription failed with status ${response.status}`);
    }

    return (await response.json()) as RetranscribeRangeResponse;
  }

  function applyRetranscribedPayload(payload: RetranscribeRangeResponse, target: RetranscribeTarget) {
    setHistory((current) => {
      if (!current.present) {
        return current;
      }

      const prepared = prepareRetranscribedRange(payload, current.present.editor.captions);
      const nextWords = replaceTimedRange(current.present.words, prepared.words, target.start, target.end);
      const nextEditor = cloneEditorState(current.present.editor);
      nextEditor.captions = replaceTimedRange(nextEditor.captions, prepared.captions, target.start, target.end);
      nextEditor.captions = syncCaptionWordAssignments(nextEditor.captions, nextWords);
      nextEditor.paragraphs = buildParagraphsFromCaptions(nextEditor.captions);

      return {
        past: [...current.past, cloneWorkspaceState(current.present)].slice(-120),
        present: {
          editor: nextEditor,
          words: nextWords,
          warnings: [
            ...current.present.warnings.filter((warning) => warning.code !== "retranscribe_empty"),
            ...cloneWarnings(payload.warnings),
          ],
          language: payload.language ?? current.present.language,
        },
        future: [],
      };
    });
  }

  function splitCaption(index: number, caretStart: number, caretEnd: number) {
    if (!activeEditor) {
      return;
    }

    const currentCaption = activeEditor.captions[index];
    const currentText = captionValue(currentCaption);
    const fragments = buildMatchedFragments(currentText, currentCaption.word_ids, transcriptWords);
    const beforeFragments = fragments.filter((fragment) => fragment.word && fragment.charEnd <= caretStart);
    const afterFragments = fragments.filter((fragment) => fragment.word && fragment.charStart >= caretEnd);
    const beforeWordIds = uniqueWordIds(beforeFragments);
    const afterWordIds = uniqueWordIds(afterFragments);
    const lastBefore = beforeFragments[beforeFragments.length - 1]?.word;
    const firstAfter = afterFragments[0]?.word;
    const originalEnd = currentCaption.end;
    const proposedSplit =
      lastBefore && firstAfter
        ? (lastBefore.end + firstAfter.start) / 2
        : lastBefore
          ? lastBefore.end
          : firstAfter
            ? firstAfter.start
            : currentCaption.start + (originalEnd - currentCaption.start) / 2;
    const splitPoint = clampSplitTime(currentCaption.start, originalEnd, proposedSplit);
    const before = currentText.slice(0, caretStart).trimEnd();
    const after = currentText.slice(caretEnd).trimStart();

    commit((draft) => {
      const caption = draft.captions[index];
      caption.lines = normalizeCaptionEditorLines(before);
      caption.end = splitPoint;
      caption.word_ids = beforeWordIds.length ? beforeWordIds : caretStart >= currentText.length ? currentCaption.word_ids : [];

      const nextCaption: Caption = {
        ...caption,
        id: `${caption.id}-split-${Date.now()}`,
        start: splitPoint,
        end: originalEnd,
        lines: normalizeCaptionEditorLines(after),
        word_ids: afterWordIds.length ? afterWordIds : caretStart <= 0 ? currentCaption.word_ids : [],
        blank_after: false,
      };
      draft.captions.splice(index + 1, 0, nextCaption);
    });
    setSelection({ kind: "caption", index: index + 1, start: 0, end: 0, text: "" });
  }

  function mergeWithPrevious(index: number) {
    if (index <= 0) {
      return;
    }
    const previousCaption = activeEditor?.captions[index - 1];
    const currentCaption = activeEditor?.captions[index];
    const mergeResult =
      previousCaption && currentCaption ? mergeCaptionLines(previousCaption.lines, currentCaption.lines) : null;
    commit((draft) => {
      const previous = draft.captions[index - 1];
      const current = draft.captions[index];
      previous.end = current.end;
      previous.lines = mergeResult?.lines ?? mergeCaptionLines(previous.lines, current.lines).lines;
      previous.word_ids = [...previous.word_ids, ...current.word_ids];
      previous.blank_after = current.blank_after;
      draft.captions.splice(index, 1);
    });
    requestCaptionFocus(index - 1, mergeResult?.caret ?? 0);
  }

  function mergeWithNext(index: number) {
    const currentCaption = activeEditor?.captions[index];
    const nextCaption = activeEditor?.captions[index + 1];
    const mergeResult = currentCaption && nextCaption ? mergeCaptionLines(currentCaption.lines, nextCaption.lines) : null;
    commit((draft) => {
      const current = draft.captions[index];
      const next = draft.captions[index + 1];
      if (!current || !next) {
        return;
      }
      current.end = next.end;
      current.lines = mergeResult?.lines ?? mergeCaptionLines(current.lines, next.lines).lines;
      current.word_ids = [...current.word_ids, ...next.word_ids];
      current.blank_after = next.blank_after;
      draft.captions.splice(index + 1, 1);
    });
    requestCaptionFocus(index, mergeResult?.caret ?? 0);
  }

  function toggleBlankAfter(index: number) {
    commit((draft) => {
      draft.captions[index].blank_after = !draft.captions[index].blank_after;
    });
  }

  function applySpeakerFromCaption(index: number, speakerId: number) {
    if (!activeEditor) {
      return;
    }
    const selected = activeEditor.captions[index];
    const nextSpeaker = activeEditor.speakers.find((speaker) => speaker.id === speakerId);
    if (!selected || !nextSpeaker) {
      return;
    }
    const originalSpeakerId = selected.speaker_id;
    const affectedWordIds = new Set<string>();
    commit((draft) => {
      for (let cursor = index; cursor < draft.captions.length; cursor += 1) {
        if (cursor > index && draft.captions[cursor].speaker_id !== originalSpeakerId) {
          break;
        }
        draft.captions[cursor].speaker_id = nextSpeaker.id;
        draft.captions[cursor].speaker_name = nextSpeaker.name;
        draft.captions[cursor].word_ids.forEach((wordId) => affectedWordIds.add(wordId));
      }
    }, {
      transformWords: (words) =>
        words.map((word) =>
          affectedWordIds.has(word.id)
            ? { ...word, speaker_id: nextSpeaker.id, speaker_name: nextSpeaker.name }
            : word,
        ),
    });
  }

  function addGuideBlock(start: number, end: number, label: GuideLabel, reason: string) {
    commit((draft) => {
      draft.guideBlocks.push({
        id: `manual-${Date.now()}-${draft.guideBlocks.length}`,
        start,
        end,
        label,
        reason,
        skip: true,
      });
      draft.guideBlocks.sort((a, b) => a.start - b.start);
    });
  }

  function getSelectionRange(): { start: number; end: number; source: string } | null {
    if (!selection || !activeEditor) {
      return null;
    }

    if (selection.kind === "caption") {
      const caption = activeEditor.captions[selection.index];
      if (!caption) {
        return null;
      }
      const fragments = buildMatchedFragments(captionValue(caption), caption.word_ids, transcriptWords);
      return {
        ...timeRangeFromSelection(fragments, selection.start, selection.end, caption.start, caption.end),
        source: "subtitle",
      };
    }

    const paragraph = activeEditor.paragraphs[selection.index];
    if (!paragraph) {
      return null;
    }
    const fragments = buildMatchedFragments(paragraph.text, paragraph.word_ids, transcriptWords);
    return {
      ...timeRangeFromSelection(fragments, selection.start, selection.end, paragraph.start, paragraph.end),
      source: "transcript",
    };
  }

  function getRetranscribeTarget(): RetranscribeTarget | null {
    if (!activeEditor) {
      return null;
    }

    if (selection) {
      const range = getSelectionRange();
      if (!range) {
        return null;
      }

      return {
        start: range.start,
        end: range.end,
        label: `${selection.kind === "caption" ? "subtitle" : "transcript"} ${selection.index + 1}${selection.start !== selection.end ? " selection" : ""}`,
      };
    }

    if (viewMode === "transcript") {
      const index = activeParagraphIndex >= 0 ? activeParagraphIndex : findNearestTimedIndex(activeEditor.paragraphs, currentTime);
      const paragraph = index >= 0 ? activeEditor.paragraphs[index] : null;
      if (!paragraph) {
        return null;
      }
      return {
        start: paragraph.start,
        end: paragraph.end,
        label: `transcript ${index + 1}`,
      };
    }

    const index = activeCaptionIndex >= 0 ? activeCaptionIndex : findNearestTimedIndex(activeEditor.captions, currentTime);
    const caption = index >= 0 ? activeEditor.captions[index] : null;
    if (!caption) {
      return null;
    }

    return {
      start: caption.start,
      end: caption.end,
      label: `subtitle ${index + 1}`,
    };
  }

  function markSelection(label: GuideLabel) {
    const range = getSelectionRange();
    if (!range) {
      setStatusMessage("Select text in the transcript or subtitles first.");
      return;
    }
    addGuideBlock(range.start, range.end, label, `Marked from ${range.source} selection`);
    setStatusMessage(`${label} block created for ${formatClock(range.start)} to ${formatClock(range.end)}.`);
  }

  function deleteGuideBlock(id: string) {
    commit((draft) => {
      draft.guideBlocks = draft.guideBlocks.filter((block) => block.id !== id);
    });
  }

  function toggleGuideSkip(id: string) {
    commit((draft) => {
      const block = draft.guideBlocks.find((item) => item.id === id);
      if (block) {
        block.skip = !block.skip;
      }
    });
  }

  function runReplaceAll() {
    if (!findText) {
      return;
    }
    commit((draft) => {
      draft.captions = draft.captions.map((caption) => ({
        ...caption,
        lines: caption.lines.map((line) => line.replaceAll(findText, replaceText)),
      }));
    });
  }

  async function handleRetranscribeSelection() {
    if (!selectedFile) {
      setStatusMessage("Choose an audio file first.");
      return;
    }
    if (!activeEditor || !session) {
      setStatusMessage("Transcribe a file or load a session first.");
      return;
    }

    const target = getRetranscribeTarget();
    if (!target) {
      setStatusMessage("Play or select a subtitle or transcript block first.");
      return;
    }

    setRetranscribing(true);
    setStatusMessage(`Retranscribing ${target.label}...`);

    try {
      const payload = await requestRetranscribedRange(target);
      if (!payload.words.length || !payload.captions.length) {
        const warningMessage = payload.warnings[0]?.message ?? "WhisperX did not return replacement text for that range.";
        setStatusMessage(warningMessage);
        return;
      }

      applyRetranscribedPayload(payload, target);
      setSelection(null);
      setStatusMessage(`Retranscribed ${target.label}${transcriptionVocabulary.trim() ? " with glossary terms." : "."}`);
    } catch (error) {
      const message = error instanceof Error ? error.message : "Unknown error";
      setStatusMessage(message);
    } finally {
      setRetranscribing(false);
    }
  }

  async function handleRetranscribeGlossaryMatches() {
    if (!selectedFile) {
      setStatusMessage("Choose an audio file first.");
      return;
    }
    if (!activeEditor) {
      setStatusMessage("Transcribe a file or load a session first.");
      return;
    }
    if (!glossaryTerms.length) {
      setStatusMessage("Add glossary terms first.");
      return;
    }

    const targetIndexes = glossaryMatches
      .filter((match) => match.exactTerms.length || match.fuzzyTerms.length)
      .map((match) => match.captionIndex);
    const ranges = buildCaptionRangesFromIndexes(activeEditor.captions, targetIndexes);
    if (!ranges.length) {
      setStatusMessage("No subtitle ranges currently match the glossary.");
      return;
    }

    setRetranscribing(true);
    let applied = 0;

    try {
      for (const [index, range] of ranges.entries()) {
        setStatusMessage(`Retranscribing jargon range ${index + 1} of ${ranges.length} (${range.label})...`);
        const payload = await requestRetranscribedRange(range);
        if (!payload.words.length || !payload.captions.length) {
          continue;
        }
        applyRetranscribedPayload(payload, range);
        applied += 1;
      }

      setSelection(null);
      setStatusMessage(
        applied
          ? `Retranscribed ${applied} glossary range${applied === 1 ? "" : "s"} with glossary terms.`
          : "Glossary retranscribe completed, but WhisperX did not return replacement text for those ranges.",
      );
    } catch (error) {
      const message = error instanceof Error ? error.message : "Unknown error";
      setStatusMessage(message);
    } finally {
      setRetranscribing(false);
    }
  }

  function downloadText(filename: string, contents: string, mime = "text/plain;charset=utf-8") {
    const blob = new Blob([contents], { type: mime });
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = filename;
    anchor.click();
    URL.revokeObjectURL(url);
  }

  function buildProjectFilename(audioFilename: string | null | undefined): string {
    const stem = (audioFilename ?? "audio").replace(/\.[^.]+$/, "").replace(/[^\w.-]+/g, "-").replace(/^-+|-+$/g, "") || "audio";
    return `${stem}.subtitle-workbench.json`;
  }

  function buildQaFilename(audioFilename: string | null | undefined): string {
    const stem = (audioFilename ?? "audio").replace(/\.[^.]+$/, "").replace(/[^\w.-]+/g, "-").replace(/^-+|-+$/g, "") || "audio";
    return `${stem}__qa-report.txt`;
  }

  function fileToDataUrl(file: File): Promise<string> {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onload = () => {
        if (typeof reader.result === "string") {
          resolve(reader.result);
          return;
        }
        reject(new Error("Could not encode the audio file for project export."));
      };
      reader.onerror = () => reject(reader.error ?? new Error("Could not read the audio file."));
      reader.readAsDataURL(file);
    });
  }

  async function dataUrlToFile(payload: ProjectAudioPayload): Promise<File> {
    const response = await fetch(payload.data_url);
    const blob = await response.blob();
    return new File([blob], payload.name, { type: payload.type || blob.type || "application/octet-stream" });
  }

  async function handleDownloadProject() {
    const snapshot = buildPersistedWorkspaceSnapshot();
    if (!snapshot || !activeEditor) {
      setStatusMessage("Transcribe or load a session before exporting a project file.");
      return;
    }
    if (!selectedFile) {
      setStatusMessage("Attach the audio file before exporting a project so playback can be restored.");
      return;
    }

    try {
      const audio = {
        name: selectedFile.name,
        type: selectedFile.type,
        data_url: await fileToDataUrl(selectedFile),
      };
      const project: ProjectFile = {
        format: "subtitle-workbench-project",
        version: PROJECT_FILE_VERSION,
        workspace: snapshot,
        audio,
      };
      downloadText(buildProjectFilename(currentAudioFilename), JSON.stringify(project, null, 2), "application/json;charset=utf-8");
      setStatusMessage(`Saved project file for ${selectedFile.name}.`);
    } catch (error) {
      const message = error instanceof Error ? error.message : "Could not export the project file.";
      setStatusMessage(message);
    }
  }

  async function handleLoadProject() {
    if (!resumeProjectFile) {
      setStatusMessage("Choose a project file first.");
      return;
    }

    try {
      const raw = await resumeProjectFile.text();
      const parsed = JSON.parse(raw) as Partial<ProjectFile>;
      if (parsed.format !== "subtitle-workbench-project" || parsed.version !== PROJECT_FILE_VERSION || !parsed.workspace) {
        throw new Error("That file is not a supported Subtitle Workbench project.");
      }

      const audioFile = parsed.audio ? await dataUrlToFile(parsed.audio) : null;
      restorePersistedWorkspace(parsed.workspace, {
        audioFile,
        statusMessage: audioFile
          ? `Loaded project ${resumeProjectFile.name}.`
          : `Loaded project ${resumeProjectFile.name}. Reattach the audio file if you need playback.`,
      });
    } catch (error) {
      const message = error instanceof Error ? error.message : "Could not load that project file.";
      setStatusMessage(message);
    }
  }

  return (
    <div className="app-shell">
      <aside className="control-rail">
        <div className="brand-block">
          <p className="eyebrow">Local WhisperX editor</p>
          <h1>Subtitle Workbench</h1>
          <p className="lede">Transcribe, edit directly in place, and export subtitles, transcript text, and an edit guide.</p>
        </div>

        <section className="panel">
          <h2>Source</h2>
          <label
            className={`dropzone ${dragActive ? "is-dragging" : ""}`}
            onDragOver={(event) => {
              event.preventDefault();
              setDragActive(true);
            }}
            onDragLeave={() => setDragActive(false)}
            onDrop={(event) => {
              event.preventDefault();
              setDragActive(false);
              setAudioFile(event.dataTransfer.files?.[0] ?? null);
            }}
          >
            <input type="file" accept="audio/*,video/*" onChange={(event) => setAudioFile(event.target.files?.[0] ?? null)} />
            <span>{selectedFile ? selectedFile.name : "Drag an audio file here or click to choose one."}</span>
          </label>

          <div className="field-grid">
            <label>
              Model
              <select value={model} onChange={(event) => setModel(event.target.value)}>
                {MODEL_OPTIONS.map((option) => (
                  <option key={option} value={option}>{option}</option>
                ))}
              </select>
            </label>
          </div>

          <label>
            Speaker count
            <input type="number" min={1} max={12} value={speakerCount} onChange={(event) => setSpeakerCount(Math.max(1, Number(event.target.value) || 1))} />
          </label>

          <label>
            Speaker timing mode
            <select
              value={speakerAssignmentMode}
              onChange={(event) => setSpeakerAssignmentMode(event.target.value as SpeakerAssignmentMode)}
              disabled={speakerCount <= 1}
            >
              {SPEAKER_ASSIGNMENT_OPTIONS.map((option) => (
                <option key={option.value} value={option.value}>{option.label}</option>
              ))}
            </select>
          </label>

          <div className="speaker-list">
            {speakerInputs.map((speaker, index) => (
              <label key={speaker.id}>
                Speaker {index + 1}
                <input value={speaker.name} onChange={(event) => updateSpeakerName(index, event.target.value)} />
              </label>
            ))}
          </div>

          <label className="toggle-row">
            <input type="checkbox" checked={removeDisfluencies} onChange={(event) => setRemoveDisfluencies(event.target.checked)} />
            Remove filler words / simple stutters
          </label>

          <p className="helper-text">`Word` mode switches speakers using each word timestamp, which is usually tighter around handoffs. The glossary lives in the Jargon tab and acts as the single project vocabulary list for transcription biasing, QA checks, and jargon-only retranscribe.</p>
          {showPyannoteSetupHint ? (
            <p className="helper-text">Multiple speakers need Hugging Face access for `pyannote/speaker-diarization-3.1`. Put `DIARIZATION_AUTH_TOKEN=hf_...` in `.env`.</p>
          ) : null}

          <button className="primary-button" disabled={loading || retranscribing || !selectedFile} onClick={handleTranscribe}>
            {loading ? "Transcribing..." : "Transcribe"}
          </button>
          <div className="panel-divider" />
          <div className="panel-section-heading">
            <p className="eyebrow">Resume</p>
            <h3>Project file</h3>
          </div>
          <p className="helper-text">Project files preserve the full editor state, guide blocks, timings, confidence data, and embedded audio for playback.</p>
          <label>
            Project file
            <input type="file" accept=".json,.subtitle-workbench.json,application/json" onChange={(event) => setResumeProjectFile(event.target.files?.[0] ?? null)} />
          </label>
          <div className="inline-actions">
            <button onClick={handleLoadProject}>Load project</button>
            <button disabled={!activeEditor || !selectedFile} onClick={handleDownloadProject}>Save project</button>
          </div>
          <div className="panel-divider" />
          <div className="panel-section-heading">
          <p className="eyebrow">Legacy</p>
          <h3>Audio + SRT</h3>
          </div>
          <p className="helper-text">Use this when you only have an audio file and an edited `.srt`. Default load preserves the original SRT timing and only rematches text to fresh WhisperX words.</p>
          <p className="helper-text">`Load + retime to audio` is opt-in and rewrites caption timing from the audio. Larger WhisperX models usually rematch better; `tiny` is faster but more error-prone.</p>
          <label>
            Audio file
            <input type="file" accept="audio/*,video/*" onChange={(event) => setResumeAudioFile(event.target.files?.[0] ?? null)} />
          </label>
          <label>
            Subtitle file (.srt)
            <input type="file" accept=".srt,text/plain" onChange={(event) => setResumeSubtitleFile(event.target.files?.[0] ?? null)} />
          </label>
          <div className="inline-actions">
            <button onClick={() => void handleResumeSession()}>Load with SRT timing</button>
            <button onClick={() => void handleResumeSession({ retimeCaptions: true })}>Load + retime to audio</button>
            <button onClick={resetWorkspace}>Reset</button>
          </div>
          {statusMessage ? <p className="status-text">{statusMessage}</p> : null}
        </section>
      </aside>

      <main className="workspace">
        <section className="player-panel">
          <div className="player-meta">
            <div>
              <p className="eyebrow">Playback</p>
              <h2>{selectedFile?.name ?? session?.audio_filename ?? "No file selected"}</h2>
            </div>
            <div className="transport-meta">
              <span>{formatClock(currentTime)}</span>
              {processedAudio ? (
                <div className="mode-toggle">
                  <button
                    className={playbackSource === "original" ? "is-active" : ""}
                    onClick={() => setPlaybackSource("original")}
                  >
                    Original
                  </button>
                  <button
                    className={playbackSource === "processed" ? "is-active" : ""}
                    onClick={() => setPlaybackSource("processed")}
                  >
                    Mastered
                  </button>
                </div>
              ) : null}
              <div className="mode-toggle">
                <button className={viewMode === "transcript" ? "is-active" : ""} onClick={() => setViewMode("transcript")}>Transcript</button>
                <button className={viewMode === "subtitles" ? "is-active" : ""} onClick={() => setViewMode("subtitles")}>Subtitles</button>
              </div>
            </div>
          </div>
          <audio
            ref={audioRef}
            controls
            src={(playbackSource === "processed" && processedAudio ? processedAudio.url : audioUrl) ?? undefined}
            preload="metadata"
          />
          <WaveformTimeline
            analysis={waveformAnalysis}
            captions={activeEditor?.captions ?? []}
            speakerEvents={speakerTimelineEvents}
            currentTime={currentTime}
            onSeek={seekAudio}
          />
          <div className="waveform-actions">
            <div className="inline-actions">
              <button disabled={!selectedFile || waveformLoading} onClick={() => void handleAnalyzeWaveform()}>
                {waveformLoading ? "Analyzing..." : "Analyze waveform"}
              </button>
              <button disabled={!waveformAnalysis || !activeEditor?.captions.length} onClick={handleAlignCaptionsToWaveform}>
                Snap subtitle edges
              </button>
            </div>
            <div className="chip-row">
              {waveformAnalysis ? (
                <>
                  <span className="metric-chip">{waveformAnalysis.speech_spans.length} speech regions</span>
                  <span className="metric-chip">{waveformAnalysis.frames.length} waveform frames</span>
                </>
              ) : null}
              {speakerTimelineEvents.length ? (
                <span className="metric-chip">{speakerTimelineEvents.length} speaker markers</span>
              ) : null}
            </div>
          </div>
          {speakerTimelineEvents.length ? (
            <div className="waveform-event-strip">
              {speakerTimelineEvents.slice(0, 8).map((event) => (
                <button
                  key={event.id}
                  className={`waveform-event-chip event-${event.kind}`}
                  type="button"
                  onClick={() => seekAudio(event.time, { play: false })}
                >
                  {formatClock(event.time)} {event.kind === "overlap" ? "Overlap" : event.kind === "tight_handoff" ? "Tight handoff" : "Switch"} | {event.label}
                </button>
              ))}
            </div>
          ) : null}
          <div className="transport-note">
            <label className="toggle-row">
              <input type="checkbox" checked={clickToPlay} onChange={(event) => setClickToPlay(event.target.checked)} />
              Click text to play
            </label>
            <label className="toggle-row">
              <input type="checkbox" checked={followPlayback} onChange={(event) => setFollowPlayback(event.target.checked)} />
              Follow playback in editor
            </label>
            <label className="toggle-row">
              <input type="checkbox" checked={showLineGuides} onChange={(event) => setShowLineGuides(event.target.checked)} />
              Show line guides
            </label>
            <label className="toggle-row">
              <input type="checkbox" checked={showTimingHighlights} onChange={(event) => setShowTimingHighlights(event.target.checked)} />
              Show timing highlights
            </label>
            <span>Clicks always seek. `Ctrl+Space` play/pause. `Shift+Space` toggles click autoplay.</span>
          </div>
          <div className="inline-actions">
            <button
              disabled={viewMode === "transcript" ? !activeEditor?.paragraphs.length : !activeEditor?.captions.length}
              onClick={viewMode === "transcript" ? jumpToCurrentTranscript : jumpToCurrentSubtitle}
            >
              Jump to Current
            </button>
          </div>
          {activeWarnings.length ? (
            <div className="warning-stack">
              {activeWarnings.map((warning) => (
                <p key={warning.code + warning.message} className="warning-chip">{warning.message}</p>
              ))}
            </div>
          ) : null}
        </section>

        <div className={`editor-grid ${isGuidePanelCollapsed ? "has-collapsed-guide-panel" : ""}`}>
          <section className="editor-panel">
            {viewMode === "transcript" ? (
              <div className="transcript-view">
                <div className="editor-toolbar">
                  <p className="line-mode-hint">
                    Select text to mark it for cutting. Click in the text to seek audio to that point.{clickToPlay ? " Playback starts immediately after the click." : " Playback waits for `Ctrl+Space` when click autoplay is off."}
                  </p>
                  <div className="inline-actions">
                    <button onClick={undo} disabled={!history.past.length}>Undo</button>
                    <button onClick={redo} disabled={!history.future.length}>Redo</button>
                  </div>
                </div>
                {activeEditor ? (
                  activeEditor.paragraphs.map((paragraph, index) => (
                    <article
                      key={paragraph.id}
                      ref={(node) => {
                        paragraphRefs.current[index] = node;
                      }}
                      className={`paragraph-card ${index === activeParagraphIndex ? "is-active" : ""}`}
                    >
                      {multiSpeaker ? (
                        <header>
                          <span className="speaker-pill">{paragraph.speaker_name ?? "Speaker"}</span>
                        </header>
                      ) : null}
                      <TimedTextEditor
                        className="transcript-editor"
                        commitMode="blur"
                        minHeight={128}
                        value={paragraph.text}
                        wordIds={paragraph.word_ids}
                        lookup={transcriptWords}
                        currentTime={currentTime}
                        showTimingHighlights={showTimingHighlights}
                        fallbackTime={paragraph.start}
                        autoPlayOnSeek={clickToPlay}
                        acknowledgedWordIds={acknowledgedWordIdSet}
                        onChange={(value) => updateParagraphText(index, value)}
                        onSeek={seekAudio}
                        onAcknowledgeWords={acknowledgeLowConfidenceWords}
                        onSelectionChange={(start, end) =>
                          setSelection((current) => {
                            if (
                              start === end &&
                              current?.kind === "paragraph" &&
                              current.index === index &&
                              current.start === current.end
                            ) {
                              return current;
                            }

                            return {
                              kind: "paragraph",
                              index,
                              start,
                              end,
                              text: start === end ? "" : paragraph.text.slice(Math.min(start, end), Math.max(start, end)),
                            };
                          })
                        }
                      />
                    </article>
                  ))
                ) : (
                  <p className="empty-state">Transcribe a file or load an existing SRT to populate the transcript.</p>
                )}
              </div>
            ) : (
              <div className="subtitle-view">
                {activeEditor ? (
                  <>
                    <div className="editor-toolbar">
                      <p className="line-mode-hint">
                        Enter creates the next caption. Shift+Enter inserts a new line. Backspace at the start merges backward. Delete at the end merges forward.
                        {clickToPlay ? " Clicks seek and start playback immediately." : " Clicks still seek, but playback waits for `Ctrl+Space`."}
                      </p>
                      <div className="inline-actions">
                        <button onClick={undo} disabled={!history.past.length}>Undo</button>
                        <button onClick={redo} disabled={!history.future.length}>Redo</button>
                        <button disabled={!activeEditor?.captions.length} onClick={reflowAllCaptions}>Reflow Lines</button>
                      </div>
                    </div>
                    <div className="subtitle-sheet">
                      {activeEditor.captions.map((caption, index) => {
                        const showSpeakerBoundary =
                          multiSpeaker &&
                          speakerAttributionEnabled(caption.speaker_id) &&
                          (index === 0 || activeEditor.captions[index - 1].speaker_id !== caption.speaker_id);
                        return (
                          <article
                            ref={(node) => {
                              captionRefs.current[index] = node;
                            }}
                            className={`caption-card ${caption.lines.length > 1 ? "is-multiline" : ""} ${index === activeCaptionIndex ? "is-active" : ""} ${caption.blank_after ? "has-gap" : ""}`}
                            key={caption.id}
                          >
                            {showSpeakerBoundary ? (
                              <div className="caption-topline compact-topline">
                                <span className="speaker-pill">{caption.speaker_name ?? "Speaker"}</span>
                              </div>
                            ) : null}
                            <TimedTextEditor
                              className="subtitle-editor"
                              minHeight={1}
                              value={captionValue(caption)}
                              wordIds={caption.word_ids}
                              lookup={transcriptWords}
                              currentTime={currentTime}
                              showTimingHighlights={showTimingHighlights}
                              fallbackTime={caption.start}
                              showLineGuides={showLineGuides}
                              autoPlayOnSeek={clickToPlay}
                              focusRequest={captionFocusRequest?.index === index ? captionFocusRequest.request : null}
                              acknowledgedWordIds={acknowledgedWordIdSet}
                              onChange={(value) => updateCaptionText(index, value)}
                              onUndo={undo}
                              onRedo={redo}
                              onSeek={seekAudio}
                              onAcknowledgeWords={acknowledgeLowConfidenceWords}
                              onSelectionChange={(start, end) =>
                                setSelection((current) => {
                                  if (
                                    start === end &&
                                    current?.kind === "caption" &&
                                    current.index === index &&
                                    current.start === current.end
                                  ) {
                                    return current;
                                  }

                                  return {
                                    kind: "caption",
                                    index,
                                    start,
                                    end,
                                    text: start === end ? "" : captionValue(caption).slice(Math.min(start, end), Math.max(start, end)),
                                  };
                                })
                              }
                              onKeyDown={(event) => {
                                const target = event.currentTarget;
                                const selectionStart = target.selectionStart;
                                const selectionEnd = target.selectionEnd;
                                const caret = selectionStart ?? 0;
                                const currentLineStart = target.value.lastIndexOf("\n", Math.max(0, caret - 1)) + 1;
                                const currentLineEndIndex = target.value.indexOf("\n", caret);
                                const currentLineEnd = currentLineEndIndex < 0 ? target.value.length : currentLineEndIndex;

                                if (event.key === "Enter" && !event.shiftKey) {
                                  event.preventDefault();
                                  splitCaption(index, selectionStart ?? 0, selectionEnd ?? 0);
                                  return;
                                }
                                if (
                                  event.key === "ArrowLeft" &&
                                  !event.shiftKey &&
                                  !event.ctrlKey &&
                                  !event.altKey &&
                                  !event.metaKey &&
                                  selectionStart === 0 &&
                                  selectionEnd === 0 &&
                                  index > 0
                                ) {
                                  event.preventDefault();
                                  requestCaptionFocus(index - 1, captionValue(activeEditor.captions[index - 1]).length);
                                  return;
                                }
                                if (
                                  event.key === "ArrowRight" &&
                                  !event.shiftKey &&
                                  !event.ctrlKey &&
                                  !event.altKey &&
                                  !event.metaKey &&
                                  selectionStart === target.value.length &&
                                  selectionEnd === target.value.length &&
                                  index < activeEditor.captions.length - 1
                                ) {
                                  event.preventDefault();
                                  requestCaptionFocus(index + 1, 0);
                                  return;
                                }
                                if (
                                  event.key === "ArrowUp" &&
                                  !event.shiftKey &&
                                  !event.ctrlKey &&
                                  !event.altKey &&
                                  !event.metaKey &&
                                  selectionStart === selectionEnd &&
                                  currentLineStart === 0 &&
                                  index > 0
                                ) {
                                  event.preventDefault();
                                  requestCaptionFocus(index - 1, captionValue(activeEditor.captions[index - 1]).length);
                                  return;
                                }
                                if (
                                  event.key === "ArrowDown" &&
                                  !event.shiftKey &&
                                  !event.ctrlKey &&
                                  !event.altKey &&
                                  !event.metaKey &&
                                  selectionStart === selectionEnd &&
                                  currentLineEnd === target.value.length &&
                                  index < activeEditor.captions.length - 1
                                ) {
                                  event.preventDefault();
                                  requestCaptionFocus(index + 1, 0);
                                  return;
                                }
                                if (
                                  event.key === "Backspace" &&
                                  selectionStart === selectionEnd &&
                                  caret > 0 &&
                                  target.value[caret - 1] === "\n"
                                ) {
                                  event.preventDefault();
                                  updateCaptionText(index, `${target.value.slice(0, caret - 1)} ${target.value.slice(caret)}`);
                                  requestCaptionFocus(index, caret);
                                  return;
                                }
                                if (event.key === "Backspace" && selectionStart === 0 && selectionEnd === 0 && index > 0) {
                                  event.preventDefault();
                                  mergeWithPrevious(index);
                                  return;
                                }
                                if (
                                  event.key === "Delete" &&
                                  selectionStart === target.value.length &&
                                  selectionEnd === target.value.length &&
                                  index < activeEditor.captions.length - 1
                                ) {
                                  event.preventDefault();
                                  mergeWithNext(index);
                                }
                              }}
                            />
                          </article>
                        );
                      })}
                    </div>
                  </>
                ) : (
                  <p className="empty-state">Transcribe a file or load an existing SRT to populate the line editor.</p>
                )}
              </div>
            )}

            <section className="export-panel">
              <div className="panel-section-heading">
                <p className="eyebrow">Export</p>
                <h3>Outputs</h3>
              </div>
              <p className="helper-text">Export from the current transcript and subtitle edits. Extending subtitles only changes the exported SRT. Speaker attribution settings here affect subtitle export only.</p>
              <label className="toggle-row">
                <input type="checkbox" checked={extendCaptionsOnExport} onChange={(event) => setExtendCaptionsOnExport(event.target.checked)} />
                Extend subtitles to next on export
              </label>
              <label className="toggle-row">
                <input
                  type="checkbox"
                  checked={normalizeExportTimingTo30Fps}
                  onChange={(event) => setNormalizeExportTimingTo30Fps(event.target.checked)}
                />
                Normalize export timing to 30fps
              </label>
              <label className="toggle-row">
                <input
                  type="checkbox"
                  checked={showSpeakerAttributionOptions}
                  onChange={(event) => setShowSpeakerAttributionOptions(event.target.checked)}
                />
                Customize speaker attribution in subtitle export
              </label>
              {showSpeakerAttributionOptions && activeEditor?.speakers.length ? (
                <div className="speaker-attribution-group">
                  {activeEditor.speakers.map((speaker, index) => (
                    <label key={speaker.id} className="toggle-row compact-toggle">
                      <input
                        type="checkbox"
                        checked={speaker.show_attribution !== false}
                        onChange={(event) => updateSpeakerAttribution(index, event.target.checked)}
                      />
                      {speaker.name || `Speaker ${index + 1}`}
                    </label>
                  ))}
                </div>
              ) : null}
              <div className="inline-actions">
                <button
                  disabled={!activeEditor}
                  onClick={() =>
                    activeEditor &&
                    downloadText(
                      buildExportFilename(currentAudioFilename, activeEditor.speakers, "subtitles", "srt"),
                      captionsToSrt(
                        activeEditor.captions,
                        activeEditor.speakers,
                        extendCaptionsOnExport,
                        normalizeExportTimingTo30Fps,
                      ),
                    )
                  }
                >
                  Download subtitles (.srt)
                </button>
                <button
                  disabled={!activeEditor}
                  onClick={() =>
                    activeEditor &&
                    downloadText(
                      buildExportFilename(currentAudioFilename, activeEditor.speakers, "transcript", "txt"),
                      paragraphsToTranscriptText(activeEditor.paragraphs, activeEditor.speakers),
                    )
                  }
                >
                  Download transcript (.txt)
                </button>
                <button
                  disabled={!activeEditor}
                  onClick={() =>
                    activeEditor &&
                    downloadText(
                      buildExportFilename(currentAudioFilename, activeEditor.speakers, "edit-guide", "srt"),
                      guideToSrt(activeEditor.guideBlocks, normalizeExportTimingTo30Fps),
                    )
                  }
                >
                  Download edit guide (.srt)
                </button>
              </div>
            </section>
          </section>

          <aside className={`guide-panel ${isGuidePanelCollapsed ? "is-collapsed" : ""}`}>
            <div className="guide-header">
              <div>
                <p className="eyebrow">{sidePanelMeta.eyebrow}</p>
                <h2>{sidePanelMeta.title}</h2>
              </div>
              <div className="guide-header-actions">
                {!isGuidePanelCollapsed ? <span>{sidePanelMeta.detail}</span> : null}
                <button className="ghost-button guide-collapse-button" type="button" onClick={() => setIsGuidePanelCollapsed((current) => !current)}>
                  {isGuidePanelCollapsed ? "Open panel" : "Hide panel"}
                </button>
              </div>
            </div>

            {!isGuidePanelCollapsed ? (
              <>
                <div className="mode-toggle panel-tabbar">
                  {SIDE_PANEL_TABS.map((tab) => (
                    <button
                      key={tab.id}
                      className={sidePanelTab === tab.id ? "is-active" : ""}
                      onClick={() => setSidePanelTab(tab.id)}
                    >
                      {tab.label}
                    </button>
                  ))}
                </div>

                {sidePanelTab === "guide" ? (
                  <>
                    <section className="selection-panel">
                      <div>
                        <p className="eyebrow">Selection</p>
                        <h3>{selection ? `${selection.kind === "caption" ? "Subtitle" : "Transcript"} ${selection.index + 1}` : "No text selected"}</h3>
                      </div>
                      <p className="selection-preview">{selection?.text?.trim() || "Select text in the transcript or subtitles, then mark it from here."}</p>
                      <div className="inline-actions">
                        <button disabled={!selection} onClick={() => markSelection("CUT")}>Mark CUT</button>
                        <button disabled={!selection} onClick={() => markSelection("REPEAT")}>Mark REPEAT</button>
                        <button disabled={!selection} onClick={() => markSelection("SILENT")}>Mark SILENT</button>
                        <button disabled={!selectedFile || retranscribing || loading} onClick={handleRetranscribeSelection}>
                          {retranscribing ? "Retranscribing..." : "Retranscribe Selection / Current"}
                        </button>
                      </div>
                      <p className="helper-text">Use this to repair bad timing or text. If nothing is selected, it retranscribes the current transcript or subtitle block at the playback position. Active glossary terms are included automatically.</p>
                      {multiSpeaker && activeEditor && selectedCaptionIndex >= 0 ? (
                        <label>
                          Speaker from here
                          <select
                            value={String(activeEditor.captions[selectedCaptionIndex]?.speaker_id ?? activeEditor.speakers[0]?.id ?? 0)}
                            onChange={(event) => applySpeakerFromCaption(selectedCaptionIndex, Number(event.target.value))}
                          >
                            {activeEditor.speakers.map((speaker) => (
                              <option key={speaker.id} value={speaker.id}>{speaker.name}</option>
                            ))}
                          </select>
                        </label>
                      ) : null}
                      <div className="inline-actions">
                        <button
                          disabled={selectedCaptionIndex < 0 || !activeEditor}
                          onClick={() => selectedCaptionIndex >= 0 && toggleBlankAfter(selectedCaptionIndex)}
                        >
                          Toggle blank gap
                        </button>
                        <button disabled={!selection} onClick={() => setSelection(null)}>Clear selection</button>
                      </div>
                    </section>

                    <section className="selection-panel">
                      <div className="panel-section-heading">
                        <p className="eyebrow">Editing</p>
                        <h3>Tools</h3>
                      </div>
                      <div className="inline-actions">
                        <button onClick={undo} disabled={!history.past.length}>Undo</button>
                        <button onClick={redo} disabled={!history.future.length}>Redo</button>
                      </div>
                      <label>
                        Find
                        <input value={findText} onChange={(event) => setFindText(event.target.value)} />
                      </label>
                      <label>
                        Replace
                        <input value={replaceText} onChange={(event) => setReplaceText(event.target.value)} />
                      </label>
                      <button onClick={runReplaceAll} disabled={!activeEditor || !findText}>Replace transcript + subtitles</button>
                      <label className="toggle-row">
                        <input type="checkbox" checked={skipCuts} onChange={(event) => setSkipCuts(event.target.checked)} />
                        Skip guide blocks during playback
                      </label>
                      <p className="helper-text">`Ctrl+Space` toggles play/pause. `Shift+Space` toggles click autoplay when focus is outside a text field.</p>
                    </section>

                    <div className="guide-list">
                      {activeEditor ? (
                        activeEditor.guideBlocks.map((block) => (
                          <article key={block.id} className={`guide-card label-${block.label.toLowerCase()}`}>
                            <button className="ghost-button align-left" onClick={() => seekAudio(block.start)}>{block.label} | {formatClock(block.start)} - {formatClock(block.end)}</button>
                            <p>{block.reason}</p>
                            <div className="guide-actions">
                              <label className="toggle-row">
                                <input type="checkbox" checked={block.skip} onChange={() => toggleGuideSkip(block.id)} />
                                Skip
                              </label>
                              <button onClick={() => deleteGuideBlock(block.id)}>Delete</button>
                            </div>
                          </article>
                        ))
                      ) : (
                        <p className="empty-state">Guide blocks will appear here.</p>
                      )}
                    </div>
                  </>
                ) : null}

                {sidePanelTab === "jargon" ? (
                  <section className="selection-panel">
                    <div className="panel-section-heading">
                      <p className="eyebrow">Glossary</p>
                      <h3>Project dictionary</h3>
                    </div>
                    <label>
                      Glossary / jargon dictionary
                      <textarea
                        rows={6}
                        placeholder="One term per line. These terms are used for transcription biasing, QA, and jargon-only retranscribe."
                        value={glossaryText}
                        onChange={(event) => setGlossaryText(event.target.value)}
                      />
                    </label>
                    <p className="helper-text">
                      {glossaryTerms.length
                        ? `${glossaryTerms.length} glossary term${glossaryTerms.length === 1 ? "" : "s"} active. ${glossaryMatchedCaptionCount} caption${glossaryMatchedCaptionCount === 1 ? "" : "s"} currently match the glossary.`
                        : "The app scans for likely jargon candidates. Add the useful ones to the glossary and they will be used during transcription, QA, and jargon-only retranscribe."}
                    </p>
                    <div className="inline-actions">
                      <button disabled={!jargonCandidates.length} onClick={() => addTermsToGlossary(jargonCandidates.slice(0, 12).map((candidate) => candidate.display))}>
                        Add top candidates
                      </button>
                      <button
                        disabled={!selectedFile || retranscribing || !glossaryTerms.length || !glossaryMatchedCaptionCount}
                        onClick={() => void handleRetranscribeGlossaryMatches()}
                      >
                        {retranscribing ? "Retranscribing..." : "Retranscribe Glossary Matches"}
                      </button>
                    </div>
                    {jargonCandidates.length ? (
                      <div className="qa-list">
                        {jargonCandidates.slice(0, 10).map((candidate) => (
                          <div key={candidate.normalized} className="qa-row">
                            <div className="qa-copy">
                              <strong>{candidate.display}</strong>
                              <div className="chip-row">
                                <span className="metric-chip">{candidate.count}x</span>
                                {candidate.lowConfidenceCount ? <span className="metric-chip">{candidate.lowConfidenceCount} low-conf</span> : null}
                                <span className="metric-chip">{candidate.reasons.join(", ")}</span>
                              </div>
                            </div>
                            <div className="qa-actions">
                              <button onClick={() => addTermsToGlossary([candidate.display])}>Add</button>
                              <button disabled={!candidate.captionIndexes.length} onClick={() => jumpToCaption(candidate.captionIndexes[0])}>Jump</button>
                            </div>
                          </div>
                        ))}
                      </div>
                    ) : (
                      <p className="helper-text">No likely jargon candidates detected yet.</p>
                    )}
                  </section>
                ) : null}

                {sidePanelTab === "master" ? (
                  <MasteringPanel
                    apiBaseUrl={API_BASE_URL}
                    audioFile={selectedFile}
                    words={activeWorkspace?.words ?? []}
                    onProcessed={handleMasteringProcessed}
                    onApplyCutsToSubtitles={handleApplyCutsToSubtitles}
                    onSeek={(time) => seekAudio(time, { play: true })}
                  />
                ) : null}

                {sidePanelTab === "qa" ? (
                  <section className="selection-panel">
                    <div className="panel-section-heading">
                      <p className="eyebrow">QA</p>
                      <h3>Report</h3>
                    </div>
                    <p className="helper-text">
                      {qaReport.issues.length
                        ? `${qaReport.summary.issueCount} issue${qaReport.summary.issueCount === 1 ? "" : "s"} across ${qaReport.summary.flaggedCaptionCount} caption${qaReport.summary.flaggedCaptionCount === 1 ? "" : "s"}.`
                        : "No QA issues found with the current captions."}
                    </p>
                    <div className="inline-actions">
                      <button
                        disabled={!activeEditor}
                        onClick={() =>
                          activeEditor &&
                          downloadText(
                            buildQaFilename(currentAudioFilename),
                            formatQaReport(qaReport, activeEditor.captions),
                          )
                        }
                      >
                        Download QA report (.txt)
                      </button>
                    </div>
                    {qaReport.issues.length ? (
                      <div className="qa-list">
                        {qaReport.issues.slice(0, 12).map((issue) => (
                          <div key={issue.id} className={`qa-row severity-${issue.severity}`}>
                            <div className="qa-copy">
                              <strong>{issue.message}</strong>
                              <p className="helper-text">Subtitle {issue.captionIndex + 1}: {issue.excerpt}</p>
                            </div>
                            <div className="qa-actions">
                              <button onClick={() => jumpToCaption(issue.captionIndex)}>Jump</button>
                            </div>
                          </div>
                        ))}
                      </div>
                    ) : null}
                  </section>
                ) : null}
              </>
            ) : (
              <div className="guide-collapsed-shell">
                <p className="helper-text guide-collapsed-copy">Guide, glossary, and QA stay tucked away until you need them.</p>
                <div className="guide-collapsed-metrics">
                  <span className="metric-chip">{sidePanelMeta.detail}</span>
                  <span className="metric-chip">{collapsedPanelMetaDetail}</span>
                </div>
                <div className="guide-collapsed-tabs">
                  {SIDE_PANEL_TABS.map((tab) => (
                    <button
                      key={tab.id}
                      type="button"
                      className={sidePanelTab === tab.id ? "is-active" : ""}
                      onClick={() => {
                        setSidePanelTab(tab.id);
                        setIsGuidePanelCollapsed(false);
                      }}
                    >
                      {tab.label}
                    </button>
                  ))}
                </div>
              </div>
            )}
          </aside>
        </div>
      </main>
    </div>
  );
}

export default App;

