import { Fragment, memo, useCallback, useEffect, useMemo, useRef, useState, type CSSProperties, type KeyboardEvent, type ReactNode } from "react";
import {
  AlertTriangle,
  ArrowLeftRight,
  ArrowRightLeft,
  AudioLines,
  AudioWaveform,
  Bandage,
  BookOpen,
  ClipboardCheck,
  Download,
  Gem,
  Info,
  Loader2,
  Pause,
  Play,
  Redo2,
  Scissors,
  Settings2,
  SlidersHorizontal,
  Undo2,
  Volume2,
  VolumeX,
  Wand2,
  X,
} from "lucide-react";

import { buildExportFilename, captionsAreSimultaneous, captionsToSrt, guideToSrt } from "./lib/exporters";
import {
  appendGlossaryTerms,
  detectJargonCandidates,
  findCaptionGlossaryMatches,
  mergeVocabularyTexts,
  parseGlossaryTerms,
} from "./lib/glossary";
import { buildQaReport, formatQaReport } from "./lib/qa";
import { DEFAULT_LOW_CONFIDENCE_THRESHOLD, isLowConfidenceWord } from "./lib/confidence";
import { conversionAudioUrl } from "./lib/conversion";
import { remapCaptions, remapGuideBlocks, remapTime, remapWords, unremapTime } from "./lib/cuts";
import type { CutRegion, MasteringResult } from "./lib/mastering";
import {
  MEDIA_HAVE_METADATA,
  chooseAudibleSet,
  chooseClockSource,
  classifySoloTrack,
  followersShareClockTimeline,
  shouldCorrectFollower,
  unionIntervals,
} from "./lib/playbackSync";
import type { SoloTrackState, SpeakerTrackInput, SpeakerVoice } from "./lib/playbackSync";
import { materializeRegions, regionsToSpeakerMap, sanitizeSpeakerRegions } from "./lib/regions";
import type { RegionMarker } from "./lib/regions";
import { parseSrt } from "./lib/srt";
import { formatClock, formatGutterClock } from "./lib/time";
import MasteringPanel from "./MasteringPanel";
import OverlapsPanel from "./OverlapsPanel";
import RegionsPanel from "./RegionsPanel";
import RestorePanel from "./RestorePanel";
import ConvertPanel from "./ConvertPanel";
import PatchPanel from "./PatchPanel";
import type { RegionReport, SeparationResult } from "./lib/separation";
import { fetchSoloTracksJob, separatedAudioUrl, startSoloTracksJob } from "./lib/separation";
import type {
  BackendCapabilities,
  Caption,
  GuideBlock,
  GuideLabel,
  OverlapRegion,
  Paragraph,
  RetranscribeRangeResponse,
  SpeechSpan,
  Speaker,
  SpeakerAssignmentMode,
  SpeakerRegion,
  SpeakerTurn,
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

// Fallback gate for muted-speaker playback before the rendered tracks exist:
// audio outside the audible speakers' word spans is muted. Spans get edge padding
// so word onsets are not clipped, and nearby spans merge so playback does not
// stutter between words.
const SOLO_SPAN_PADDING_S = 0.15;
const SOLO_SPAN_MERGE_GAP_S = 0.6;

// Snapped speaker regions (preferred over the padding/merge rule above whenever
// the waveform has been analysed). An imported SRT carries speaker identity but
// no silence: measured on a 19-minute three-speaker file, 374 of 375 cues abut
// the previous cue and 85 of 86 speaker handoffs have a zero-width gap, so 31%
// of handoffs fall INSIDE a continuous speech span and the wrong voice bleeds
// into the region. The audio has the missing silence — every handoff there sits
// in a real gap (median 1.24 s, p10 0.58 s) and a boundary needs to move only a
// median 0.35 s to reach it. So identity comes from the captions and boundaries
// come from the VAD speech spans.
//
// Only 25 of 726 spans (2.2 s total) are genuinely shared by two speakers; below
// this minority share a shared span goes wholly to the majority speaker rather
// than being cut mid-speech for a sliver.
const SHARED_SPAN_MIN_MINORITY_S = 0.15;
// Regions open before the first word and close after the last, but stop short of
// anyone else's speech so every transition lands in measured silence.
const REGION_EXTEND_S = 0.25;
const REGION_SILENCE_MARGIN_S = 0.08;
// The handful of shared-span splits do cut mid-speech; ramp the playback gate
// there instead of stepping it, or the cut clicks. Only the fallback gate on the
// clock uses this -- rendered speaker tracks arrive already faded server-side.
const GATE_RAMP_S = 0.04;

interface SoloInterval {
  start: number;
  end: number;
}

function buildSpeakerIntervals(words: WordToken[], speakerId: number): SoloInterval[] {
  const spans = words
    .filter((word) => word.speaker_id === speakerId)
    .map((word) => ({ start: Math.max(0, word.start - SOLO_SPAN_PADDING_S), end: word.end + SOLO_SPAN_PADDING_S }))
    .sort((a, b) => a.start - b.start);

  const merged: SoloInterval[] = [];
  for (const span of spans) {
    const previous = merged[merged.length - 1];
    if (previous && span.start - previous.end <= SOLO_SPAN_MERGE_GAP_S) {
      previous.end = Math.max(previous.end, span.end);
    } else {
      merged.push({ ...span });
    }
  }
  return merged;
}

interface AttributedPiece {
  start: number;
  end: number;
  speakerId: number;
}

/** Last speech end at or before `time`, or null when nobody spoke earlier. */
function speechEndBefore(spans: SoloInterval[], time: number): number | null {
  // Speech spans are sorted and non-overlapping, so ends rise with starts.
  let low = 0;
  let high = spans.length - 1;
  let found: number | null = null;
  while (low <= high) {
    const mid = (low + high) >> 1;
    if (spans[mid].end <= time) {
      found = spans[mid].end;
      low = mid + 1;
    } else {
      high = mid - 1;
    }
  }
  return found;
}

/** First speech start at or after `time`, or null when nobody speaks again. */
function speechStartAfter(spans: SoloInterval[], time: number): number | null {
  let low = 0;
  let high = spans.length - 1;
  let found: number | null = null;
  while (low <= high) {
    const mid = (low + high) >> 1;
    if (spans[mid].start >= time) {
      found = spans[mid].start;
      high = mid - 1;
    } else {
      low = mid + 1;
    }
  }
  return found;
}

/** Attribute one speech span to speakers using the captions that overlap it. */
function attributeSpan(span: SpeechSpan, overlaps: AttributedPiece[]): AttributedPiece[] {
  const shares = new Map<number, number>();
  for (const piece of overlaps) {
    shares.set(piece.speakerId, (shares.get(piece.speakerId) ?? 0) + (piece.end - piece.start));
  }
  if (!shares.size) {
    return []; // no speaker-labelled caption covers this span: it belongs to nobody
  }

  const ranked = [...shares.entries()].sort((left, right) => right[1] - left[1]);
  const total = ranked.reduce((sum, [, share]) => sum + share, 0);
  if (ranked.length === 1 || total - ranked[0][1] < SHARED_SPAN_MIN_MINORITY_S) {
    // Caption edges must never clip inside a span: one owner takes all of it.
    return [{ start: span.start, end: span.end, speakerId: ranked[0][0] }];
  }

  // Genuinely shared span. The caption boundary is the only handoff signal here.
  const pieces: AttributedPiece[] = [];
  for (const piece of overlaps) {
    const previous = pieces[pieces.length - 1];
    if (previous && previous.speakerId === piece.speakerId) {
      previous.end = Math.max(previous.end, piece.end);
    } else {
      pieces.push({ ...piece });
    }
  }
  for (let index = 0; index < pieces.length - 1; index += 1) {
    const boundary =
      pieces[index + 1].start >= pieces[index].end
        ? pieces[index + 1].start
        : (pieces[index].end + pieces[index + 1].start) / 2;
    pieces[index].end = boundary;
    pieces[index + 1].start = boundary;
  }
  pieces[0].start = span.start;
  pieces[pieces.length - 1].end = span.end;
  return pieces.filter((piece) => piece.end > piece.start);
}

/**
 * Speaker regions whose edges sit in measured silence: identity from the
 * captions, boundaries from the waveform's speech spans. Callers pass
 * `waveformAnalysis.speech_spans` (20 ms resolution, computed pre-downsampling —
 * the display `frames` array is too coarse for this). Kept pure and exported so
 * it can be exercised standalone.
 */
export function buildSpeakerRegionsFromSpeech(
  speechSpans: SpeechSpan[],
  captions: Caption[],
  duration: number | null,
): Map<number, SoloInterval[]> {
  const bySpeaker = new Map<number, SoloInterval[]>();
  const labelled = captions
    .filter((caption) => caption.speaker_id !== null && caption.end > caption.start)
    .map((caption) => ({ start: caption.start, end: caption.end, speakerId: caption.speaker_id as number }))
    .sort((left, right) => left.start - right.start);
  if (!speechSpans.length || !labelled.length) {
    return bySpeaker;
  }
  const spans = [...speechSpans].filter((span) => span.end > span.start).sort((left, right) => left.start - right.start);

  const pieces: AttributedPiece[] = [];
  // The spans an extended region edge must stay clear of: the raw spans, but
  // subdivided wherever a span was split between two speakers, so a split edge
  // does not extend over the other half of its own span.
  const bounds: SoloInterval[] = [];
  let cursor = 0;
  for (const span of spans) {
    while (cursor < labelled.length && labelled[cursor].end <= span.start) {
      cursor += 1;
    }
    const overlaps: AttributedPiece[] = [];
    for (let index = cursor; index < labelled.length && labelled[index].start < span.end; index += 1) {
      const start = Math.max(span.start, labelled[index].start);
      const end = Math.min(span.end, labelled[index].end);
      if (end > start) {
        overlaps.push({ start, end, speakerId: labelled[index].speakerId });
      }
    }
    const attributed = attributeSpan(span, overlaps);
    pieces.push(...attributed);
    if (attributed.length) {
      bounds.push(...attributed.map((piece) => ({ start: piece.start, end: piece.end })));
    } else {
      bounds.push({ start: span.start, end: span.end }); // speech nobody is labelled for
    }
  }

  // Merge a speaker's consecutive pieces whenever nobody else speaks between
  // them. No distance cap: bridging measured silence cannot swallow anyone, which
  // is exactly what the blunt SOLO_SPAN_MERGE_GAP_S rule got wrong for short
  // back-channels ("right", "yeah") from another speaker.
  const merged: AttributedPiece[] = [];
  for (const piece of pieces) {
    const previous = merged[merged.length - 1];
    if (previous && previous.speakerId === piece.speakerId) {
      previous.end = Math.max(previous.end, piece.end);
    } else {
      merged.push({ ...piece });
    }
  }

  for (const region of merged) {
    let start = Math.max(0, region.start - REGION_EXTEND_S);
    const previousEnd = speechEndBefore(bounds, region.start);
    if (previousEnd !== null) {
      start = Math.max(start, previousEnd + REGION_SILENCE_MARGIN_S);
    }
    let end = duration === null ? region.end + REGION_EXTEND_S : Math.min(duration, region.end + REGION_EXTEND_S);
    const nextStart = speechStartAfter(bounds, region.end);
    if (nextStart !== null) {
      end = Math.min(end, nextStart - REGION_SILENCE_MARGIN_S);
    }
    // Never shrink below the speech this region owns. The margin math only
    // inverts at a shared-span split, where the neighbouring "speech" is the
    // other half of the very same span.
    if (start > region.start) {
      start = region.start;
    }
    if (end < region.end) {
      end = region.end;
    }

    const existing = bySpeaker.get(region.speakerId);
    if (!existing) {
      bySpeaker.set(region.speakerId, [{ start, end }]);
      continue;
    }
    // timeInIntervals binary-searches these, so keep them sorted and disjoint.
    const previous = existing[existing.length - 1];
    if (start <= previous.end) {
      previous.end = Math.max(previous.end, end);
    } else {
      existing.push({ start, end });
    }
  }
  return bySpeaker;
}

// Bumped whenever the rendering itself changes in a way that makes older
// artifacts unusable, so a reloaded session re-renders instead of adopting them.
// v2: FLAC seektables (pre-v2 gated tracks seek tens of seconds off).
const SOLO_TRACK_RENDER_VERSION = "v2";

function soloTracksSessionKey(
  session: TranscriptResponse,
  restore: boolean,
  regionCount: number,
  renderNonce: number,
): string {
  // The restore flag is part of the key so Diamond-restored tokens are never
  // reused for a raw run (or vice versa) after a reload. The region count stands
  // in for the gating envelope: re-derived regions must re-render the tracks.
  // Hand edits that only move boundaries keep the count, so the nonce carries an
  // explicit re-render request -- rendering on every nudge would launch a
  // minutes-long job per keystroke.
  return `${SOLO_TRACK_RENDER_VERSION}|${session.audio_filename}|${session.duration ?? 0}|${(session.overlap_regions ?? []).length}|${regionCount}|${restore ? "restore" : "raw"}|n${renderNonce}`;
}

// Cheap FNV-1a over the gating envelope, to tell whether the rendered tracks
// still match the regions on screen.
function regionsSignature(regions: { start: number; end: number; speaker_index: number }[]): string {
  let hash = 0x811c9dc5;
  for (const region of regions) {
    const text = `${region.speaker_index}:${region.start.toFixed(3)}:${region.end.toFixed(3)};`;
    for (let index = 0; index < text.length; index++) {
      hash ^= text.charCodeAt(index);
      hash = Math.imul(hash, 0x01000193);
    }
  }
  return (hash >>> 0).toString(36);
}

/** Index of the interval containing `time`, or -1. Requires sorted intervals. */
function intervalIndexAt(time: number, intervals: SoloInterval[]): number {
  let low = 0;
  let high = intervals.length - 1;
  while (low <= high) {
    const mid = (low + high) >> 1;
    if (time < intervals[mid].start) {
      high = mid - 1;
    } else if (time > intervals[mid].end) {
      low = mid + 1;
    } else {
      return mid;
    }
  }
  return -1;
}

function timeInIntervals(time: number, intervals: SoloInterval[]): boolean {
  return intervalIndexAt(time, intervals) >= 0;
}


/**
 * Playback gain for the solo gate: 1 inside an interval, 0 outside, with a short
 * linear ramp just inside each edge. Runs in a requestAnimationFrame loop, so it
 * shares the binary search rather than scanning.
 */
export function intervalGainAt(time: number, intervals: SoloInterval[], rampSeconds: number): number {
  const index = intervalIndexAt(time, intervals);
  if (index < 0) {
    return 0;
  }
  const interval = intervals[index];
  const ramp = Math.min(rampSeconds, (interval.end - interval.start) / 2);
  if (ramp <= 0) {
    return 1;
  }
  const distance = Math.min(time - interval.start, interval.end - time);
  return Math.max(0, Math.min(1, distance / ramp));
}

// A track that has not loaded metadata yet cannot be seeked -- the browser drops
// the assignment -- and solo tracks are mounted the moment their render lands,
// often mid-playback. The position is remembered and applied on loadedmetadata so
// a follower is already time-locked by the time the selector can reach it. Only
// the latest target is kept: an element that is seeked twice while loading must
// not land on the older position.
const pendingSeeks = new WeakMap<HTMLMediaElement, number>();

function seekWhenReady(element: HTMLMediaElement, time: number) {
  if (element.readyState >= MEDIA_HAVE_METADATA) {
    pendingSeeks.delete(element);
    element.currentTime = time;
    return;
  }
  const alreadyWaiting = pendingSeeks.has(element);
  pendingSeeks.set(element, time);
  if (alreadyWaiting) {
    return;
  }
  const onReady = () => {
    element.removeEventListener("loadedmetadata", onReady);
    const target = pendingSeeks.get(element);
    pendingSeeks.delete(element);
    if (target !== undefined) {
      element.currentTime = target;
    }
  };
  element.addEventListener("loadedmetadata", onReady);
}

const FOLLOW_SCROLL_SUSPEND_MS = 3000;
const THEME_STORAGE_KEY = "subtitle-workbench:theme";
const AUTOSAVE_STORAGE_KEY = "subtitle-workbench:autosave";
const BACKEND_INSTANCE_STORAGE_KEY = "subtitle-workbench:backend-instance";
const AUTOSAVE_STORAGE_VERSION = 5;
const PROJECT_FILE_VERSION = 1;
const TEXT_EDIT_CHECKPOINT_MS = 800;
const AUTOSAVE_DELAY_MS = 700;
const SIDE_PANEL_TABS = [
  { id: "guide", label: "Guide" },
  { id: "jargon", label: "Vocab" },
  { id: "qa", label: "QA" },
  { id: "overlaps", label: "Overlaps" },
  { id: "regions", label: "Regions" },
  { id: "restore", label: "Restore" },
  { id: "convert", label: "Convert" },
  { id: "patch", label: "Patch" },
  { id: "master", label: "Master" },
  { id: "export", label: "Export" },
] as const;
const SIDE_PANEL_TAB_ICONS = {
  guide: Scissors,
  jargon: BookOpen,
  qa: ClipboardCheck,
  overlaps: AudioLines,
  regions: AudioWaveform,
  restore: Gem,
  convert: ArrowRightLeft,
  patch: Bandage,
  master: Wand2,
  export: Download,
} as const;
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
type TranscriptionOverrides = { model?: string; speakers?: Speaker[] };

interface EditorState {
  captions: Caption[];
  guideBlocks: GuideBlock[];
  speakers: Speaker[];
  paragraphs: Paragraph[];
  // Manually edited speaker regions. Null (or absent, in a save written before
  // the Regions panel existed) means the audio-derived regions still apply.
  // Living inside the editor state is what gets these onto the undo stack and
  // into both the autosave and the project file for free.
  regionOverrides?: SpeakerRegion[] | null;
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

interface SoloTrackArtifacts {
  key: string;
  tokens: Record<number, string>;
  regionsSignature?: string;
}

// Converted voices survive a reload the way solo tracks do: server tokens only,
// revalidated with HEAD before anything is mounted, plus which voice was playing.
interface ConvertedVoiceArtifacts {
  tokens: Record<number, string>;
  active: Record<number, SpeakerVoice>;
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
  lowConfidenceThreshold: number;
  // Whether the auto solo-tracks were (or should be) Diamond-restored.
  restoreSoloTracks?: boolean;
  // Finished auto solo-track artifacts (speaker index -> server token), keyed
  // to the session so a reload can revalidate them instead of re-running UniSE.
  // regionsSignature is the gating envelope the artifacts were actually rendered
  // against, so a reload can still tell that later hand edits left them stale.
  soloTracks?: SoloTrackArtifacts | null;
  // Bumped by "Re-render speaker tracks". It is part of the solo-track key, so
  // it has to survive a reload -- otherwise the restored key can never match the
  // one the render effect recomputes and every reload re-renders from scratch.
  soloTrackRenderNonce?: number;
  // Converted voices (speaker id -> conversion token) and the voice each speaker
  // was last heard in.
  convertedVoices?: ConvertedVoiceArtifacts | null;
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
  lowConfidenceThreshold?: number;
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

function requestCompletionNotificationPermission() {
  if ("Notification" in window && Notification.permission === "default") {
    void Notification.requestPermission();
  }
}

// System-level notification for long jobs, shown only when the tab is hidden —
// if the user is already looking at the app, the status line is enough.
function notifyWorkFinished(title: string, body: string) {
  if (!("Notification" in window) || Notification.permission !== "granted" || !document.hidden) {
    return;
  }
  try {
    new Notification(title, { body });
  } catch {
    // Notification constructors can throw on platforms that only allow
    // service-worker notifications; the in-app status line still updates.
  }
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

/** Persisted tab ids are validated against the live tab list, not a hand-kept union. */
function coerceSidePanelTab(value: unknown): SidePanelTab {
  return SIDE_PANEL_TABS.some((tab) => tab.id === value) ? (value as SidePanelTab) : "guide";
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
    regionOverrides: state.regionOverrides ? state.regionOverrides.map((region) => ({ ...region })) : null,
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
  if (!lines.length) {
    return [""];
  }
  // Captions are at most two lines; fold any extras into the second line.
  if (lines.length > 2) {
    return [lines[0], normalizeEditableText(lines.slice(1).join(" "))];
  }
  return lines;
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
    // One box per speaker turn: the same speaker never spans consecutive boxes.
    const shouldBreak = previous.speaker_id !== caption.speaker_id;
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
  // Every restore path (autosave, project file, session load) comes through
  // here, so this is where a hand-edited or truncated override gets vetted:
  // malformed entries are dropped, never thrown on.
  next.regionOverrides = sanitizeSpeakerRegions(state.regionOverrides);
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

// Only-speaker playback gates on `word.speaker_id`, so speakers that come from SRT labels
// have to be pushed down onto the words. Text-based caption<->word matching is too fragile
// to carry that: the anchor pass runs on the `tiny` model, whose garbled text matches only
// a fraction of the words (measured 244 of 3216 on a real 19-minute load), which left the
// solo gate closed almost everywhere. The SRT's own clock is the ground truth here, so a
// word belongs to the caption whose time range contains its midpoint. Words outside every
// caption stay unattributed, which the solo gate correctly treats as "not the soloed
// speaker". When captions overlap, the latest-starting one wins.
function retagWordsFromCaptions(words: WordToken[], captions: Caption[]): WordToken[] {
  const ranges = [...captions].filter((caption) => caption.end > caption.start).sort((a, b) => a.start - b.start);
  const starts = ranges.map((caption) => caption.start);
  const maxDuration = ranges.reduce((max, caption) => Math.max(max, caption.end - caption.start), 0);

  const captionAt = (time: number): Caption | null => {
    let low = 0;
    let high = starts.length - 1;
    let latestStarted = -1;
    while (low <= high) {
      const mid = (low + high) >> 1;
      if (starts[mid] <= time) {
        latestStarted = mid;
        low = mid + 1;
      } else {
        high = mid - 1;
      }
    }
    for (let index = latestStarted; index >= 0 && starts[index] > time - maxDuration - 0.001; index--) {
      if (time < ranges[index].end) {
        return ranges[index];
      }
    }
    return null;
  };

  return words.map((word) => {
    const owner = captionAt((word.start + word.end) / 2);
    return {
      ...word,
      speaker_id: owner?.speaker_id ?? null,
      speaker_name: owner?.speaker_name ?? null,
    };
  });
}

// Diarization numbers its speaker slots by order of first appearance in the audio, which has
// no relation to the SRT's label order. Each slot is majority-voted onto the SRT speaker who
// owns most of the words diarization put in that slot, so speaker turns and overlap regions
// (which the Overlaps panel names and separates by index) keep the SRT's identities. Slots
// with no vote keep their index rather than guessing.
function remapDiarizedSpeakerIndices(
  baseSession: TranscriptResponse,
  taggedWords: WordToken[],
): Pick<TranscriptResponse, "speaker_turns" | "overlap_regions"> {
  const votes = new Map<number, Map<number, number>>();
  baseSession.words.forEach((word, index) => {
    const slot = word.speaker_id;
    const srtSpeaker = taggedWords[index]?.speaker_id;
    if (slot === null || slot === undefined || srtSpeaker === null || srtSpeaker === undefined) {
      return;
    }
    const tally = votes.get(slot) ?? new Map<number, number>();
    tally.set(srtSpeaker, (tally.get(srtSpeaker) ?? 0) + 1);
    votes.set(slot, tally);
  });

  const mapping = new Map<number, number>();
  for (const [slot, tally] of votes) {
    let best = -1;
    let bestCount = 0;
    for (const [speakerId, count] of tally) {
      if (count > bestCount) {
        best = speakerId;
        bestCount = count;
      }
    }
    if (best >= 0) {
      mapping.set(slot, best);
    }
  }

  return {
    speaker_turns: baseSession.speaker_turns?.map((turn) => ({
      ...turn,
      speaker_index: mapping.get(turn.speaker_index) ?? turn.speaker_index,
    })),
    overlap_regions: baseSession.overlap_regions?.map((region) => ({
      ...region,
      speaker_indices: [...new Set(region.speaker_indices.map((index) => mapping.get(index) ?? index))],
    })),
  };
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
  const words = importedHasSpeakers ? retagWordsFromCaptions(baseSession.words, alignedCaptions) : cloneWords(baseSession.words);
  const remappedIndices = importedHasSpeakers ? remapDiarizedSpeakerIndices(baseSession, words) : {};

  return {
    ...baseSession,
    ...remappedIndices,
    audio_filename: audioFilename,
    duration: baseSession.duration ?? alignedCaptions[alignedCaptions.length - 1]?.end ?? null,
    speakers: speakerSource.map((speaker) => ({ ...speaker })),
    words,
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
  overlapRegions: OverlapRegion[];
  currentTime: number;
  theme: "light" | "dark";
  onSeek: (time: number, options?: { play?: boolean }) => void;
}

function themeColor(name: string, fallback: string): string {
  const value = getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  return value || fallback;
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

  const { analysis, captions, currentTime } = props;
  const duration = Math.max(
    analysis?.duration ?? 0,
    captions[captions.length - 1]?.end ?? 0,
    currentTime,
    1,
  );

  context.clearRect(0, 0, width, height);

  if (!analysis?.frames.length) {
    context.fillStyle = themeColor("--muted", "#647184");
    context.font = "12px Inter, sans-serif";
    context.fillText("Waveform appears after loading audio", 12, height / 2 + 4);
    return;
  }

  // Overlap regions sit behind the bars as translucent bands.
  if (props.overlapRegions.length) {
    context.fillStyle = themeColor("--overlap-band", "rgba(216, 90, 48, 0.16)");
    for (const region of props.overlapRegions) {
      const left = clampNumber((region.start / duration) * width, 0, width);
      const right = clampNumber((region.end / duration) * width, 0, width);
      if (right - left > 0.5) {
        context.fillRect(left, 1, right - left, height - 2);
      }
    }
  }

  // Amplitude bars, split into played and unplayed at the playhead.
  const barWidth = 3;
  const barGap = 2;
  const step = barWidth + barGap;
  const barCount = Math.max(1, Math.floor(width / step));
  const centerY = height / 2;
  const maxBar = height * 0.86;
  const playedColor = themeColor("--wave-played", "#1d9e75");
  const restColor = themeColor("--wave-rest", "#b4b2a9");
  const playheadX = clampNumber((currentTime / duration) * width, 0, width);

  // One pass over the frames: bucket each into its bar and keep the peak.
  const peaks = new Float32Array(barCount);
  for (const frame of analysis.frames) {
    const barIndex = Math.min(barCount - 1, Math.floor((frame.time / duration) * barCount));
    const amplitude = Math.max(Math.abs(frame.max), Math.abs(frame.min));
    if (amplitude > peaks[barIndex]) {
      peaks[barIndex] = amplitude;
    }
  }

  for (let barIndex = 0; barIndex < barCount; barIndex += 1) {
    const x = barIndex * step;
    const barHeight = Math.max(3, peaks[barIndex] * maxBar);
    context.fillStyle = x + barWidth / 2 <= playheadX ? playedColor : restColor;
    context.beginPath();
    context.roundRect(x, centerY - barHeight / 2, barWidth, barHeight, 1.5);
    context.fill();
  }

  context.strokeStyle = themeColor("--playhead", "#d85a30");
  context.lineWidth = 2;
  context.beginPath();
  context.moveTo(playheadX, 2);
  context.lineTo(playheadX, height - 2);
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
        height: 72,
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

  const scrubbing = useRef(false);

  function seekFromPointer(clientX: number) {
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
    const fraction = clampNumber((clientX - rect.left) / rect.width, 0, 1);
    props.onSeek(fraction * duration, { play: false });
  }

  return (
    <div className="waveform-timeline" ref={wrapperRef}>
      <canvas
        ref={canvasRef}
        className="waveform-canvas"
        onPointerDown={(event) => {
          scrubbing.current = true;
          event.currentTarget.setPointerCapture(event.pointerId);
          seekFromPointer(event.clientX);
        }}
        onPointerMove={(event) => {
          if (scrubbing.current) {
            seekFromPointer(event.clientX);
          }
        }}
        onPointerUp={(event) => {
          scrubbing.current = false;
          event.currentTarget.releasePointerCapture(event.pointerId);
        }}
        onPointerCancel={() => {
          scrubbing.current = false;
        }}
      />
    </div>
  );
});

function InfoTip({ text }: { text: string }) {
  return (
    <span className="info-tip" tabIndex={0}>
      <Info size={13} aria-hidden />
      <span className="info-tip-bubble" role="tooltip">{text}</span>
    </span>
  );
}

const TimedTextEditor = memo(function TimedTextEditor({
  value,
  wordIds,
  lookup,
  currentTime,
  showTimingHighlights = true,
  lowConfidenceThreshold = DEFAULT_LOW_CONFIDENCE_THRESHOLD,
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
            fragment.word &&
            isLowConfidenceWord(fragment.word, lowConfidenceThreshold) &&
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
    previous.lowConfidenceThreshold === next.lowConfidenceThreshold &&
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
  const [speakerAssignmentMode, setSpeakerAssignmentMode] = useState<SpeakerAssignmentMode>("word");
  const [glossaryText, setGlossaryText] = useState("");
  const [findText, setFindText] = useState("");
  const [replaceText, setReplaceText] = useState("");
  const [skipCuts, setSkipCuts] = useState(false);
  const [clickToPlay, setClickToPlay] = useState(true);
  const [followPlayback, setFollowPlayback] = useState(true);
  const [showLineGuides, setShowLineGuides] = useState(false);
  const [showTimingHighlights, setShowTimingHighlights] = useState(true);
  const [lowConfidenceThreshold, setLowConfidenceThreshold] = useState(DEFAULT_LOW_CONFIDENCE_THRESHOLD);
  const [sidePanelTab, setSidePanelTab] = useState<SidePanelTab>("guide");
  const [isGuidePanelCollapsed, setIsGuidePanelCollapsed] = useState(DEFAULT_GUIDE_PANEL_COLLAPSED);
  const [extendCaptionsOnExport, setExtendCaptionsOnExport] = useState(false);
  const [normalizeExportTimingTo30Fps, setNormalizeExportTimingTo30Fps] = useState(false);
  const [showSpeakerAttributionOptions, setShowSpeakerAttributionOptions] = useState(false);
  const [removeDisfluencies, setRemoveDisfluencies] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);
  const [toasts, setToasts] = useState<Array<{ id: number; text: string }>>([]);
  const toastIdRef = useRef(0);

  function setStatusMessage(message: string | null) {
    if (!message) {
      return;
    }
    const id = ++toastIdRef.current;
    setToasts((current) => [...current.slice(-3), { id, text: message }]);
    window.setTimeout(() => {
      setToasts((current) => current.filter((toast) => toast.id !== id));
    }, 6000);
  }

  function dismissToast(id: number) {
    setToasts((current) => current.filter((toast) => toast.id !== id));
  }
  const [retranscribing, setRetranscribing] = useState(false);
  const [waveformLoading, setWaveformLoading] = useState(false);
  const [waveformAnalysis, setWaveformAnalysis] = useState<WaveformAnalysisResponse | null>(null);
  const [processedAudio, setProcessedAudio] = useState<{
    url: string;
    filename: string;
    label: "Mastered" | "Separated";
    hasCutTimeline: boolean;
    cutList: CutRegion[];
  } | null>(null);
  const [playbackSource, setPlaybackSource] = useState<"original" | "processed">("original");
  const [resplitting, setResplitting] = useState(false);
  const [viewOptionsOpen, setViewOptionsOpen] = useState(false);
  const [setupDrawerOpen, setSetupDrawerOpen] = useState(false);
  const [globalDragActive, setGlobalDragActive] = useState(false);
  const [isPlaying, setIsPlaying] = useState(false);
  const [audioDuration, setAudioDuration] = useState(0);
  const [playbackRate, setPlaybackRate] = useState(1);
  const [userMuted, setUserMuted] = useState(false);
  // Audibility is expressed as the set of muted speakers, not a soloed one: with
  // the set empty the clock plays as it always did, and with anything in it every
  // unmuted speaker's server-gated track plays instead. Session-only, never
  // persisted.
  const [mutedSpeakerIds, setMutedSpeakerIds] = useState<Set<number>>(() => new Set());
  // Set by the audibility pass when no usable track set exists yet, so the clock
  // has to stand in gated. The only state in which the frontend touches volume.
  const [fallbackGateActive, setFallbackGateActive] = useState(false);
  // Auto-prepared per-speaker tracks whose overlap sections contain only that
  // speaker's separated voice; keyed by speaker index (appearance order).
  const [soloTracks, setSoloTracks] = useState<{
    status: "idle" | "running" | "done" | "error";
    tracks: Record<number, { url: string; token: string; filename?: string }>;
  }>({ status: "idle", tracks: {} });
  // Conversions of the isolated tracks, by speaker id. A conversion of a
  // server-gated full-length track is itself gated and full-length, so it drops
  // into the mix in that speaker's place with no client-side work.
  const [convertedVoices, setConvertedVoices] = useState<Map<number, { token: string; url: string; filename?: string }>>(
    () => new Map(),
  );
  // Which voice each speaker is heard in. Absent means the original.
  const [activeVoice, setActiveVoice] = useState<Map<number, SpeakerVoice>>(() => new Map());
  // Exported per-speaker tracks are gated to the regions as they were when the
  // render ran; hand edits afterwards leave them stale until re-rendered. The
  // nonce carries an explicit re-render request into the cache key -- rendering
  // on every nudge would launch a minutes-long job per drag. Both are declared
  // here because the autosave effect below persists them.
  const [trackRenderNonce, setTrackRenderNonce] = useState(0);
  const [renderedRegionsSignature, setRenderedRegionsSignature] = useState<string | null>(null);
  // When on, the auto-prepared solo tracks are regenerated at 44.1 kHz studio
  // quality (Diamond) after voice isolation. Baked into the solo-tracks cache
  // key so raw and restored renders never share tokens.
  const [restoreSoloTracks, setRestoreSoloTracks] = useState(false);
  const [themeDark, setThemeDark] = useState(() => window.localStorage.getItem(THEME_STORAGE_KEY) === "dark");
  const [resumeProjectFile, setResumeProjectFile] = useState<File | null>(null);
  const [resumeAudioFile, setResumeAudioFile] = useState<File | null>(null);
  const [resumeSubtitleFile, setResumeSubtitleFile] = useState<File | null>(null);
  const [selection, setSelection] = useState<BlockSelection | null>(null);
  const [captionFocusRequest, setCaptionFocusRequest] = useState<{ index: number; request: FocusRequest } | null>(null);
  const [acknowledgedLowConfidenceWordIds, setAcknowledgedLowConfidenceWordIds] = useState<string[]>([]);
  const [backendCapabilities, setBackendCapabilities] = useState<BackendCapabilities | null>(null);

  // The clock: the element whose currentTime is authoritative. Never repointed by
  // muting -- see the clock effect below.
  const audioRef = useRef<HTMLAudioElement | null>(null);
  const originalAudioRef = useRef<HTMLAudioElement | null>(null);
  const masteredAudioRef = useRef<HTMLAudioElement | null>(null);
  // Every rendered solo track stays mounted, so muting a speaker is a mute swap
  // rather than a media-element swap. Keyed by artifact token, not by speaker
  // index: a re-render replaces one element with another for the same speaker,
  // and the token tells the two apart during the swap.
  const soloTrackElementsRef = useRef(new Map<string, HTMLAudioElement>());
  // Tokens of the speaker tracks currently carrying the sound (empty = the clock
  // is). Read back as the `alreadyAudible` hysteresis input per track.
  const audibleTrackTokensRef = useRef<Set<string>>(new Set());
  const soloTracksAttemptRef = useRef<string | null>(null);
  const persistedSoloTokensRef = useRef<SoloTrackArtifacts | null>(null);
  // Restored converted-voice records, waiting for the HEAD revalidation below.
  const persistedConvertedVoicesRef = useRef<ConvertedVoiceArtifacts | null>(null);
  const [convertedRestoreNonce, setConvertedRestoreNonce] = useState(0);
  const userScrollAtRef = useRef(0);
  const viewOptionsRef = useRef<HTMLDivElement | null>(null);
  const paragraphRefs = useRef<Array<HTMLElement | null>>([]);
  const captionRefs = useRef<Array<HTMLElement | null>>([]);
  const suppressAutoSpeakerModeRef = useRef(false);
  const autosaveReadyRef = useRef(false);
  const autosaveTimerRef = useRef<number | null>(null);
  const lastTextEditRef = useRef<{ kind: SelectionKind; index: number; timestamp: number } | null>(null);
  const activeWorkspace = history.present;
  const activeEditor = activeWorkspace?.editor ?? null;
  const currentAudioFilename = selectedFile?.name ?? session?.audio_filename ?? null;
  const overlapRegions = useMemo<OverlapRegion[]>(() => session?.overlap_regions ?? [], [session]);
  const speakerTurns = useMemo<SpeakerTurn[]>(() => session?.speaker_turns ?? [], [session]);
  const activeWords = activeWorkspace?.words ?? session?.words ?? [];
  // The speakers the mute toggles cover. Only speakers that actually say
  // something: a silent speaker has no regions, so nothing is ever rendered for
  // it, and counting it as unmuted would strand playback on the fallback gate.
  const soloableSpeakers = useMemo(() => {
    if (!session || session.speakers.length < 2) {
      return [];
    }
    const spokenIds = new Set(activeWords.map((word) => word.speaker_id));
    return session.speakers.filter((speaker) => spokenIds.has(speaker.id));
  }, [session, activeWords]);
  const soloTrackEntries = useMemo(
    () =>
      Object.entries(soloTracks.tracks)
        .map(([index, track]) => ({ speakerIndex: Number(index), ...track }))
        .sort((a, b) => a.speakerIndex - b.speakerIndex),
    [soloTracks],
  );
  // Identity of the mounted follower set: the audio effects re-attach their
  // listeners when a track appears or its artifact changes, not on every render.
  const soloTrackMountKey = useMemo(
    () => soloTrackEntries.map((entry) => `${entry.speakerIndex}:${entry.token}`).join("|"),
    [soloTrackEntries],
  );
  // Converted voices in a stable order, for mounting and for the mount key.
  const convertedVoiceEntries = useMemo(
    () =>
      Array.from(convertedVoices.entries())
        .map(([speakerId, voice]) => ({ speakerId, ...voice }))
        .sort((a, b) => a.speakerId - b.speakerId),
    [convertedVoices],
  );
  // Identity of every mounted follower, solo tracks and converted voices alike:
  // a converted voice arriving mid-session has to re-attach the same listeners a
  // new solo track does, or it never joins the clock.
  const trackMountKey = useMemo(
    () => `${soloTrackMountKey}#${convertedVoiceEntries.map((entry) => `${entry.speakerId}:${entry.token}`).join("|")}`,
    [soloTrackMountKey, convertedVoiceEntries],
  );
  // Each mutable speaker paired with the artifact tokens of its rendered track
  // and its converted voice (null while nothing has been rendered for it), plus
  // which of the two is selected. Track records are keyed by speaker index,
  // which is appearance order -- the order of session.speakers.
  const speakerTrackTokens = useMemo(() => {
    const speakers = session?.speakers ?? [];
    return soloableSpeakers.map((speaker) => {
      const index = speakers.findIndex((entry) => entry.id === speaker.id);
      const converted = convertedVoices.get(speaker.id) ?? null;
      return {
        speakerId: speaker.id,
        token: (index >= 0 ? soloTracks.tracks[index]?.token : null) ?? null,
        convertedToken: converted?.token ?? null,
        // A selection with nothing converted behind it means the original: the
        // toggle only exists once an artifact does.
        voice: (converted && activeVoice.get(speaker.id) === "converted" ? "converted" : "original") as SpeakerVoice,
      };
    });
  }, [activeVoice, convertedVoices, session, soloableSpeakers, soloTracks]);
  // Download links for the per-speaker artifacts. The artifact GET routes already
  // set a Content-Disposition name; `download` only makes it a friendly one.
  const speakerAudioExports = useMemo(() => {
    const speakers = session?.speakers ?? [];
    const basename = (currentAudioFilename ?? "audio").replace(/\.[^.]+$/, "");
    const extensionOf = (filename: string | undefined) => /\.([A-Za-z0-9]+)$/.exec(filename ?? "")?.[1] ?? "flac";
    return soloableSpeakers.map((speaker) => {
      const index = speakers.findIndex((entry) => entry.id === speaker.id);
      const isolated = (index >= 0 ? soloTracks.tracks[index] : null) ?? null;
      const converted = convertedVoices.get(speaker.id) ?? null;
      return {
        speakerId: speaker.id,
        name: speaker.name,
        isolated: isolated
          ? { url: isolated.url, download: `${basename} — ${speaker.name}.isolated.${extensionOf(isolated.filename)}` }
          : null,
        converted: converted
          ? { url: converted.url, download: `${basename} — ${speaker.name}.converted.${extensionOf(converted.filename)}` }
          : null,
      };
    });
  }, [convertedVoices, currentAudioFilename, session, soloableSpeakers, soloTracks]);
  // The isolated tracks offered to the Convert panel as one-click sources.
  const isolatedTrackChoices = useMemo(() => {
    const speakers = session?.speakers ?? [];
    return soloableSpeakers.flatMap((speaker) => {
      const index = speakers.findIndex((entry) => entry.id === speaker.id);
      const track = index >= 0 ? soloTracks.tracks[index] : null;
      return track ? [{ speakerId: speaker.id, name: speaker.name, token: track.token }] : [];
    });
  }, [session, soloableSpeakers, soloTracks]);

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
          // A new backend process means a new working session: drop the
          // autosaved workspace so the editor starts clean after a restart.
          if (payload.instance_id) {
            const stored = window.localStorage.getItem(BACKEND_INSTANCE_STORAGE_KEY);
            if (stored && stored !== payload.instance_id) {
              window.localStorage.removeItem(AUTOSAVE_STORAGE_KEY);
              resetWorkspace();
            }
            window.localStorage.setItem(BACKEND_INSTANCE_STORAGE_KEY, payload.instance_id);
          }
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
            speakerAssignmentMode: saved.speakerAssignmentMode === "segment" ? "segment" : "word",
            glossaryText: mergeVocabularyTexts(
              typeof saved.glossaryText === "string" ? saved.glossaryText : "",
              typeof saved.hotwords === "string" ? saved.hotwords : "",
            ),
            skipCuts: typeof saved.skipCuts === "boolean" ? saved.skipCuts : false,
            clickToPlay: typeof saved.clickToPlay === "boolean" ? saved.clickToPlay : true,
            followPlayback: typeof saved.followPlayback === "boolean" ? saved.followPlayback : true,
            showLineGuides: typeof saved.showLineGuides === "boolean" ? saved.showLineGuides : false,
            showTimingHighlights: typeof saved.showTimingHighlights === "boolean" ? saved.showTimingHighlights : true,
            viewMode: saved.viewMode === "transcript" ? "transcript" : "subtitles",
            sidePanelTab: coerceSidePanelTab(saved.sidePanelTab),
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
            lowConfidenceThreshold:
              typeof saved.lowConfidenceThreshold === "number" && Number.isFinite(saved.lowConfidenceThreshold)
                ? saved.lowConfidenceThreshold
                : DEFAULT_LOW_CONFIDENCE_THRESHOLD,
            restoreSoloTracks: typeof saved.restoreSoloTracks === "boolean" ? saved.restoreSoloTracks : false,
            soloTracks:
              saved.soloTracks && typeof saved.soloTracks.key === "string" && saved.soloTracks.tokens
                ? saved.soloTracks
                : null,
            convertedVoices: saved.convertedVoices?.tokens ? saved.convertedVoices : null,
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
  }, [acknowledgedLowConfidenceWordIds, activeVoice, activeWorkspace, clickToPlay, convertedVoices, extendCaptionsOnExport, followPlayback, glossaryText, isGuidePanelCollapsed, lowConfidenceThreshold, model, normalizeExportTimingTo30Fps, removeDisfluencies, restoreSoloTracks, session, showLineGuides, showSpeakerAttributionOptions, showTimingHighlights, sidePanelTab, skipCuts, soloTracks, speakerAssignmentMode, speakerCount, speakerInputs, trackRenderNonce, viewMode]);

  // One stable ref callback per token. An inline arrow would be a fresh function
  // on every render, and React detaches (null) then re-attaches a changed ref
  // callback each time -- churning the element map under the audibility logic.
  const soloTrackRefCallbacksRef = useRef(new Map<string, (element: HTMLAudioElement | null) => void>());
  const soloTrackRefFor = useCallback((token: string) => {
    const cache = soloTrackRefCallbacksRef.current;
    const existing = cache.get(token);
    if (existing) {
      return existing;
    }
    const callback = (element: HTMLAudioElement | null) => {
      if (element) {
        soloTrackElementsRef.current.set(token, element);
      } else {
        soloTrackElementsRef.current.delete(token);
      }
    };
    cache.set(token, callback);
    return callback;
  }, []);

  // Every mounted element, in a stable order. Reads refs only, so it never has to
  // be rebuilt: the effects below use it to attach and clean up listeners.
  const collectTransportElements = useCallback(() => {
    const elements: HTMLAudioElement[] = [];
    if (originalAudioRef.current) {
      elements.push(originalAudioRef.current);
    }
    if (masteredAudioRef.current) {
      elements.push(masteredAudioRef.current);
    }
    for (const element of soloTrackElementsRef.current.values()) {
      elements.push(element);
    }
    return elements;
  }, []);

  // The elements that co-play behind the clock. The A/B peer joins only while it
  // shares the original timeline (a cut master has its own, and switchPlaybackSource
  // remaps across it instead); solo tracks are always rendered on the original
  // timeline, so they follow whenever the clock is on it too.
  const collectClockFollowers = useCallback(
    (clock: HTMLAudioElement) => {
      const followers: HTMLAudioElement[] = [];
      const abPeer = clock === originalAudioRef.current ? masteredAudioRef.current : originalAudioRef.current;
      if (abPeer && abPeer !== clock && processedAudio && !processedAudio.hasCutTimeline) {
        followers.push(abPeer);
      }
      const clockSource = clock === masteredAudioRef.current ? "mastered" : "original";
      if (followersShareClockTimeline(clockSource, processedAudio?.hasCutTimeline ?? false)) {
        for (const element of soloTrackElementsRef.current.values()) {
          if (element !== clock) {
            followers.push(element);
          }
        }
      }
      return followers;
    },
    [processedAudio],
  );

  // audioRef is the clock, and the A/B selection is the only thing that moves it.
  // Soloing must not repoint it: the transcript highlighting, waveform playhead,
  // skip-cuts logic and the solo gate all read audioRef.current.currentTime, so a
  // repoint mid-playback resynchronises the whole UI onto a freshly loaded element
  // -- the jump this design removes.
  useEffect(() => {
    const clockSource = chooseClockSource(playbackSource, masteredAudioRef.current !== null);
    const clock = clockSource === "mastered" ? masteredAudioRef.current : originalAudioRef.current;
    if (!clock) {
      return;
    }
    audioRef.current = clock;
    if (Number.isFinite(clock.duration) && clock.duration > 0) {
      setAudioDuration(clock.duration);
    }
    setIsPlaying(!clock.paused);
  }, [playbackSource, processedAudio, audioUrl]);

  // Audibility is the whole of what a mute toggle changes: no seek, no play(), no
  // unmount, no volume automation on the tracks. Everything keeps playing; the
  // mute flags move. Rendered tracks are already gated server-side, so the sum of
  // the unmuted ones IS the mix minus the muted voices -- re-gating them here is
  // what produced the jitter this replaces.
  const applyAudibility = useCallback(() => {
    const clock = audioRef.current;
    if (!clock) {
      return;
    }
    const previouslyAudible = audibleTrackTokensRef.current;
    // Both of a speaker's tracks are classified the same way -- a converted voice
    // is just another server-gated follower -- so this runs over either token.
    const classifyToken = (token: string | null): SoloTrackState => {
      const element = token ? (soloTrackElementsRef.current.get(token) ?? null) : null;
      const alreadyAudible = element !== null && token !== null && previouslyAudible.has(token);
      // Nudging a candidate here is safe precisely because it is still muted; once
      // it is audible nothing seeks it again.
      if (
        element &&
        !alreadyAudible &&
        shouldCorrectFollower({ clockTime: clock.currentTime, followerTime: element.currentTime, isAudible: false })
      ) {
        seekWhenReady(element, clock.currentTime);
      }
      return classifySoloTrack({
        hasTrack: element !== null,
        readyState: element?.readyState ?? 0,
        offsetFromClock: element ? element.currentTime - clock.currentTime : 0,
        alreadyAudible,
      });
    };

    const speakers: SpeakerTrackInput[] = speakerTrackTokens.map((entry) => ({
      speakerId: entry.speakerId,
      muted: mutedSpeakerIds.has(entry.speakerId),
      track: classifyToken(entry.token),
      voice: entry.voice,
      converted: classifyToken(entry.convertedToken),
    }));

    const decision = chooseAudibleSet({
      playbackSource,
      hasMastered: masteredAudioRef.current !== null,
      masterHasCutTimeline: processedAudio?.hasCutTimeline ?? false,
      speakers,
    });

    const audibleTokens = new Set<string>();
    for (const choice of decision.audibleTracks) {
      const entry = speakerTrackTokens.find((item) => item.speakerId === choice.speakerId);
      const token = choice.voice === "converted" ? entry?.convertedToken : entry?.token;
      if (token) {
        audibleTokens.add(token);
      }
    }
    audibleTrackTokensRef.current = audibleTokens;

    const audibleElements = new Set<HTMLAudioElement>();
    for (const token of audibleTokens) {
      const element = soloTrackElementsRef.current.get(token);
      if (element) {
        audibleElements.add(element);
      }
    }
    if (decision.clockAudible) {
      audibleElements.add(clock);
    }
    for (const element of collectTransportElements()) {
      element.muted = !audibleElements.has(element);
    }
    // Same value re-set on every timeupdate in the steady state; React bails out
    // of those, so this does not re-render per tick.
    setFallbackGateActive(decision.gateClock);
  }, [collectTransportElements, mutedSpeakerIds, playbackSource, processedAudio, speakerTrackTokens]);

  // Re-evaluate audibility when the muted set changes and whenever a track
  // finishes loading, so speakers muted before the tracks were ready move off the
  // gated clock on their own.
  useEffect(() => {
    applyAudibility();
    const elements = collectTransportElements();
    const reapply = () => applyAudibility();
    for (const element of elements) {
      element.addEventListener("loadedmetadata", reapply);
      element.addEventListener("canplay", reapply);
    }
    return () => {
      for (const element of elements) {
        element.removeEventListener("loadedmetadata", reapply);
        element.removeEventListener("canplay", reapply);
      }
    };
  }, [applyAudibility, collectTransportElements, audioUrl, processedAudio, playbackSource, trackMountKey]);

  // A track rendered mid-session mounts while the clock is already running, so it
  // missed the play event: join it in flight, positioned and at the same rate, so
  // it is swap-ready the moment the selector reaches it.
  useEffect(() => {
    const clock = audioRef.current;
    if (!clock) {
      return;
    }
    for (const follower of collectClockFollowers(clock)) {
      if (
        shouldCorrectFollower({
          clockTime: clock.currentTime,
          followerTime: follower.currentTime,
          isAudible: false,
        })
      ) {
        seekWhenReady(follower, clock.currentTime);
      }
      follower.playbackRate = clock.playbackRate;
      if (!clock.paused && follower.paused) {
        void follower.play().catch(() => undefined);
      }
    }
  }, [collectClockFollowers, audioUrl, processedAudio, playbackSource, trackMountKey]);

  // Transport state: play/pause indicator, duration, speed, and user mute. Only
  // the clock's events count -- followers start and stop a beat later and would
  // otherwise flicker the button and the duration readout.
  useEffect(() => {
    const elements = collectTransportElements();
    if (!elements.length) {
      return;
    }

    const onPlayState = (event: Event) => {
      if (event.target === audioRef.current) {
        setIsPlaying(event.type === "play");
      }
    };
    const onMetadata = (event: Event) => {
      const audio = event.target as HTMLAudioElement;
      if (audio === audioRef.current && Number.isFinite(audio.duration)) {
        setAudioDuration(audio.duration);
      }
    };
    for (const element of elements) {
      element.addEventListener("play", onPlayState);
      element.addEventListener("pause", onPlayState);
      element.addEventListener("loadedmetadata", onMetadata);
      element.addEventListener("durationchange", onMetadata);
    }
    return () => {
      for (const element of elements) {
        element.removeEventListener("play", onPlayState);
        element.removeEventListener("pause", onPlayState);
        element.removeEventListener("loadedmetadata", onMetadata);
        element.removeEventListener("durationchange", onMetadata);
      }
    };
  }, [collectTransportElements, audioUrl, processedAudio, playbackSource, trackMountKey]);

  useEffect(() => {
    for (const element of collectTransportElements()) {
      element.playbackRate = playbackRate;
    }
  }, [collectTransportElements, playbackRate, audioUrl, processedAudio, trackMountKey]);

  function skipBy(seconds: number) {
    const audio = audioRef.current;
    if (!audio) {
      return;
    }
    audio.currentTime = Math.max(0, Math.min(audio.currentTime + seconds, audio.duration || Number.POSITIVE_INFINITY));
  }

  useEffect(() => {
    const elements = collectTransportElements();
    if (!elements.length || !activeEditor) {
      return;
    }

    const onTimeUpdate = (event: Event) => {
      const audio = event.target as HTMLAudioElement;
      if (audio !== audioRef.current) {
        return;
      }
      const time = audio.currentTime;
      setCurrentTime(time);
      // Hold every co-playing follower on the clock's position, so making one
      // audible needs no seek at all. The audible element is excluded on purpose:
      // seeking what is being listened to is exactly the glitch being removed, and
      // its own drift stays inaudible because nothing is cross-faded against it.
      for (const follower of collectClockFollowers(audio)) {
        if (
          shouldCorrectFollower({
            clockTime: time,
            followerTime: follower.currentTime,
            isAudible: !follower.muted,
          })
        ) {
          seekWhenReady(follower, time);
        }
        // A follower that never got a play event (mounted mid-playback, or its
        // play() was rejected) would sit still and be useless to swap to.
        if (!audio.paused && follower.paused) {
          void follower.play().catch(() => undefined);
        }
      }
      // A track that has caught up can take over from the gated clock now.
      applyAudibility();
      if (!skipCuts) {
        return;
      }
      const activeCut = activeEditor.guideBlocks.find((block) => block.skip && time >= block.start && time < block.end);
      if (activeCut) {
        audio.currentTime = Math.min(activeCut.end + 0.02, audio.duration || activeCut.end + 0.02);
      }
    };

    for (const element of elements) {
      element.addEventListener("timeupdate", onTimeUpdate);
    }
    return () => {
      for (const element of elements) {
        element.removeEventListener("timeupdate", onTimeUpdate);
      }
    };
  }, [
    activeEditor,
    applyAudibility,
    collectClockFollowers,
    collectTransportElements,
    skipCuts,
    processedAudio,
    audioUrl,
    playbackSource,
    trackMountKey,
  ]);

  // While the timelines match, mirror play/pause/seek/rate from the clock onto
  // every follower, so switching which elements are audible -- Original vs
  // Mastered, or the clock vs the unmuted speakers' tracks -- is a mute swap and
  // nothing more. Speaker tracks are mirrored whether or not a master exists,
  // which is why this no longer bails out without processedAudio.
  useEffect(() => {
    const clock = audioRef.current;
    if (!clock) {
      return;
    }

    const mirror = (event: Event) => {
      const source = event.target as HTMLAudioElement;
      // Only the clock leads. Followers emit the same events as they are driven,
      // and echoing those back would have them seek each other.
      if (source !== audioRef.current) {
        return;
      }
      for (const follower of collectClockFollowers(source)) {
        if (event.type === "pause") {
          follower.pause();
          continue;
        }
        if (event.type === "ratechange") {
          follower.playbackRate = source.playbackRate;
          continue;
        }
        // play and seeked are deliberate transport changes, so realigning even the
        // audible follower is correct here -- but only when it has actually drifted.
        // A redundant seek on the element being heard clicks for no reason.
        if (
          shouldCorrectFollower({
            clockTime: source.currentTime,
            followerTime: follower.currentTime,
            isAudible: false,
          })
        ) {
          seekWhenReady(follower, source.currentTime);
        }
        if (event.type === "play") {
          void follower.play().catch(() => undefined);
        }
      }
    };

    const events = ["play", "pause", "seeked", "ratechange"] as const;
    const elements = collectTransportElements();
    for (const element of elements) {
      for (const type of events) {
        element.addEventListener(type, mirror);
      }
    }
    return () => {
      for (const element of elements) {
        for (const type of events) {
          element.removeEventListener(type, mirror);
        }
      }
    };
  }, [collectClockFollowers, collectTransportElements, processedAudio, audioUrl, playbackSource, trackMountKey]);

  // One shared derivation of the snapped regions: the fallback gate and the solo-track
  // job must gate on exactly the same edges. Null means "not derivable" (no
  // waveform analysis yet, or no speaker-labelled captions) — then the legacy
  // word-span path applies.
  const derivedSpeakerRegions = useMemo(() => {
    const spans = waveformAnalysis?.speech_spans ?? [];
    const captions = activeEditor?.captions ?? [];
    if (!spans.length || !captions.some((caption) => caption.speaker_id !== null)) {
      return null;
    }
    const regions = buildSpeakerRegionsFromSpeech(
      spans,
      captions,
      waveformAnalysis?.duration ?? session?.duration ?? null,
    );
    return regions.size ? regions : null;
  }, [waveformAnalysis, activeEditor, session]);
  const regionOverrides = activeEditor?.regionOverrides ?? null;
  // Manual edits win outright. No diff/merge against the derivation: once the
  // user has drawn a boundary by hand, re-deriving over it would silently undo
  // the fix. "Re-derive from audio" (which clears the override) is the way back.
  const speakerRegions = useMemo(() => {
    if (!regionOverrides) {
      return derivedSpeakerRegions;
    }
    const regions = regionsToSpeakerMap(regionOverrides);
    return regions.size ? regions : null;
  }, [regionOverrides, derivedSpeakerRegions]);
  // What the Regions panel edits: the override when there is one, otherwise the
  // derivation flattened so the first edit materializes it.
  const editableRegions = useMemo<SpeakerRegion[]>(
    () => regionOverrides ?? materializeRegions(derivedSpeakerRegions),
    [regionOverrides, derivedSpeakerRegions],
  );
  // Envelope for the fallback gate only: the union of the regions belonging to the
  // speakers that are still audible. It gates the CLOCK -- the full mixture --
  // while the per-speaker tracks are missing or still catching up, so it is the
  // one place volume automation is still correct. Deliberately not unioned with
  // speakerTurns: diarized turn edges are not silence-aligned, and over the
  // mixture they would let a muted voice back in.
  const fallbackGateIntervals = useMemo(() => {
    if (!mutedSpeakerIds.size) {
      return [];
    }
    const lists = soloableSpeakers
      .filter((speaker) => !mutedSpeakerIds.has(speaker.id))
      .map((speaker) => {
        const snapped = speakerRegions?.get(speaker.id);
        return snapped?.length ? snapped : buildSpeakerIntervals(activeWords, speaker.id);
      });
    return unionIntervals(lists);
  }, [activeWords, mutedSpeakerIds, soloableSpeakers, speakerRegions]);

  // Speakers that no longer exist cannot be unmuted again, so drop them; a fresh
  // speaker roster (new session, re-import, retag) starts everyone audible.
  const soloableSpeakerKey = useMemo(() => soloableSpeakers.map((speaker) => speaker.id).join(","), [soloableSpeakers]);
  const lastSoloableSpeakerKeyRef = useRef(soloableSpeakerKey);
  useEffect(() => {
    if (lastSoloableSpeakerKeyRef.current === soloableSpeakerKey) {
      return;
    }
    lastSoloableSpeakerKeyRef.current = soloableSpeakerKey;
    setMutedSpeakerIds((current) => (current.size ? new Set<number>() : current));
  }, [soloableSpeakerKey]);

  const toggleSpeakerMuted = useCallback((speakerId: number) => {
    setMutedSpeakerIds((current) => {
      const next = new Set(current);
      if (!next.delete(speakerId)) {
        next.add(speakerId);
      }
      return next;
    });
  }, []);

  const setSpeakerVoice = useCallback((speakerId: number, voice: SpeakerVoice) => {
    setActiveVoice((current) => {
      if ((current.get(speakerId) ?? "original") === voice) {
        return current;
      }
      const next = new Map(current);
      next.set(speakerId, voice);
      return next;
    });
  }, []);

  // A finished conversion of an isolated track registers as that speaker's
  // alternate voice and starts playing straight away -- the toggle is how the
  // user gets back to the original.
  const handleConvertedVoice = useCallback(
    (speakerId: number, result: { token: string; url: string; filename: string }) => {
      setConvertedVoices((current) => {
        const next = new Map(current);
        next.set(speakerId, { token: result.token, url: result.url, filename: result.filename });
        return next;
      });
      setSpeakerVoice(speakerId, "converted");
    },
    [setSpeakerVoice],
  );

  // A discarded conversion is deleted server-side, so its registration has to go
  // too: dropping both entries collapses the voice toggle back to nothing and
  // unmounts the element rather than leaving it pending on a 404 URL.
  const handleConvertedVoiceRemoved = useCallback((speakerId: number) => {
    setConvertedVoices((current) => {
      if (!current.has(speakerId)) {
        return current;
      }
      const next = new Map(current);
      next.delete(speakerId);
      return next;
    });
    setActiveVoice((current) => {
      if (!current.has(speakerId)) {
        return current;
      }
      const next = new Map(current);
      next.delete(speakerId);
      return next;
    });
  }, []);

  // Restored converted voices are adopted only while their artifacts are still on
  // the server: mounting a 404 would leave a permanently pending element that the
  // audibility pass keeps waiting for.
  useEffect(() => {
    const pending = persistedConvertedVoicesRef.current;
    if (!pending) {
      return;
    }
    persistedConvertedVoicesRef.current = null;
    let cancelled = false;

    const revalidate = async () => {
      const entries = Object.entries(pending.tokens);
      const alive = await Promise.all(
        entries.map(async ([, token]) => {
          try {
            const response = await fetch(conversionAudioUrl(API_BASE_URL, token), { method: "HEAD" });
            return response.ok;
          } catch {
            return false;
          }
        }),
      );
      if (cancelled) {
        return;
      }
      const voices = new Map<number, { token: string; url: string; filename?: string }>();
      const active = new Map<number, SpeakerVoice>();
      entries.forEach(([key, token], index) => {
        if (!alive[index]) {
          return;
        }
        const speakerId = Number(key);
        voices.set(speakerId, { token, url: conversionAudioUrl(API_BASE_URL, token) });
        if (pending.active[speakerId] === "converted") {
          active.set(speakerId, "converted");
        }
      });
      setConvertedVoices(voices);
      setActiveVoice(active);
    };

    void revalidate();
    return () => {
      cancelled = true;
    };
  }, [convertedRestoreNonce]);

  // The Regions panel still speaks in terms of one soloed lane: exactly one
  // speaker left audible is that lane, anything else is no solo at all.
  const soloSpeakerId = useMemo(() => {
    const unmuted = soloableSpeakers.filter((speaker) => !mutedSpeakerIds.has(speaker.id));
    return mutedSpeakerIds.size && unmuted.length === 1 ? unmuted[0].id : null;
  }, [mutedSpeakerIds, soloableSpeakers]);
  const handleSoloSpeakerChange = useCallback(
    (speakerId: number | null) => {
      setMutedSpeakerIds(
        speakerId === null
          ? new Set<number>()
          : new Set(soloableSpeakers.filter((speaker) => speaker.id !== speakerId).map((speaker) => speaker.id)),
      );
    },
    [soloableSpeakers],
  );

  // The gate reads the clock, not whichever element happens to be audible: the
  // regions are on the clock's timeline, and the audible element can change under
  // it at any moment. Every element gets the same gain so a swap cannot step it.
  // Outside the fallback the gain is a flat 1 (or 0 for the user's own mute):
  // server-gated tracks must not be re-gated client-side.
  const applyPlaybackVolume = useCallback(() => {
    const gateActive = fallbackGateActive && fallbackGateIntervals.length > 0;
    const time = audioRef.current?.currentTime ?? 0;
    const gain = userMuted ? 0 : gateActive ? intervalGainAt(time, fallbackGateIntervals, GATE_RAMP_S) : 1;
    for (const element of collectTransportElements()) {
      element.volume = gain;
    }
  }, [collectTransportElements, userMuted, fallbackGateActive, fallbackGateIntervals]);

  useEffect(() => {
    applyPlaybackVolume();
    const elements = collectTransportElements();
    const reapply = () => applyPlaybackVolume();
    for (const element of elements) {
      element.addEventListener("timeupdate", reapply);
      element.addEventListener("seeked", reapply);
    }
    return () => {
      for (const element of elements) {
        element.removeEventListener("timeupdate", reapply);
        element.removeEventListener("seeked", reapply);
      }
    };
  }, [applyPlaybackVolume, collectTransportElements, audioUrl, processedAudio, playbackSource, trackMountKey]);

  // The same snapped regions, flattened for the backend: it silences everything
  // outside them so each exported per-speaker track stands alone on the original
  // timeline (that is what the Convert engine needs to feed back in place).
  const soloTrackRegions = useMemo(() => {
    if (!speakerRegions || !session) {
      return [];
    }
    const payload: Array<{ start: number; end: number; speaker_index: number }> = [];
    for (const [speakerId, intervals] of speakerRegions) {
      const speakerIndex = session.speakers.findIndex((speaker) => speaker.id === speakerId);
      if (speakerIndex < 0) {
        continue;
      }
      for (const interval of intervals) {
        payload.push({ start: interval.start, end: interval.end, speaker_index: speakerIndex });
      }
    }
    return payload;
  }, [speakerRegions, session]);
  // Every caption edit re-derives the regions into a fresh array, so the render
  // job below keys off the region count only: re-running its effect would cancel
  // an in-flight render. The ref keeps the payload current for whichever attempt
  // does start.
  const soloTrackRegionCount = soloTrackRegions.length;
  const soloTrackRegionsRef = useRef(soloTrackRegions);
  useEffect(() => {
    soloTrackRegionsRef.current = soloTrackRegions;
  }, [soloTrackRegions]);

  const currentRegionsSignature = useMemo(() => regionsSignature(soloTrackRegions), [soloTrackRegions]);
  const soloTracksStale =
    renderedRegionsSignature !== null && renderedRegionsSignature !== currentRegionsSignature;

  // Solo tracks are prepared automatically: as soon as a transcription reports
  // overlap regions — or the captions yield snapped speaker regions — one
  // background job renders a per-speaker version of the recording whose overlaps
  // contain only that speaker's separated voice, gated to that speaker's regions.
  // The mute toggles then just work — no extra buttons. With nobody muted these
  // tracks are never audible.
  useEffect(() => {
    const canSeparate = overlapRegions.length > 0 && speakerTurns.length > 0;
    if (!session || !selectedFile || (!canSeparate && !soloTrackRegionCount)) {
      return;
    }
    const attemptKey = soloTracksSessionKey(session, restoreSoloTracks, soloTrackRegionCount, trackRenderNonce);
    if (soloTracksAttemptRef.current === attemptKey) {
      return;
    }
    soloTracksAttemptRef.current = attemptKey;
    let cancelled = false;

    async function prepare() {
      // Reload path: previously rendered tracks whose artifacts still exist on
      // the server are adopted as-is instead of re-running separation.
      const persisted = persistedSoloTokensRef.current;
      if (persisted && persisted.key === attemptKey && Object.keys(persisted.tokens).length) {
        const entries = Object.entries(persisted.tokens);
        const alive = await Promise.all(
          entries.map(async ([, token]) => {
            try {
              const response = await fetch(separatedAudioUrl(API_BASE_URL, token), { method: "HEAD" });
              return response.ok;
            } catch {
              return false;
            }
          }),
        );
        if (cancelled) {
          return;
        }
        if (alive.length && alive.every(Boolean)) {
          const tracks: Record<number, { url: string; token: string }> = {};
          for (const [index, token] of entries) {
            tracks[Number(index)] = { url: separatedAudioUrl(API_BASE_URL, token), token };
          }
          setSoloTracks({ status: "done", tracks });
          // These artifacts are gated to the regions recorded alongside them,
          // not to whatever is on screen now: stamping the live regions here
          // would re-mark hand-edited tracks as fresh on every reload.
          setRenderedRegionsSignature(persisted.regionsSignature ?? null);
          return;
        }
        persistedSoloTokensRef.current = null; // expired on the server; re-render
      }

      setSoloTracks({ status: "running", tracks: {} });
      try {
        // Freeze the envelope for this attempt: the signature stored with the
        // artifacts has to describe exactly the regions that were sent, or the
        // staleness check compares the tracks against regions they never used.
        const renderedRegions = soloTrackRegionsRef.current;
        const renderedSignature = regionsSignature(renderedRegions);
        const jobId = await startSoloTracksJob(
          API_BASE_URL,
          selectedFile!,
          overlapRegions,
          speakerTurns,
          renderedRegions,
          restoreSoloTracks,
        );
        while (!cancelled) {
          const status = await fetchSoloTracksJob(API_BASE_URL, jobId);
          if (status.status === "done" && status.result) {
            const tracks: Record<number, { url: string; token: string; filename?: string }> = {};
            for (const track of status.result.tracks) {
              tracks[track.speaker_index] = {
                url: separatedAudioUrl(API_BASE_URL, track.token),
                token: track.token,
                // Kept for the export tab's download name only; a reload adopts
                // tokens alone and falls back to the rendered format there.
                filename: track.output_filename,
              };
            }
            if (!cancelled) {
              // Record the artifacts under the very key this attempt ran with.
              // The autosave snapshot writes this record verbatim rather than
              // recomputing the key, so the two can never drift apart.
              persistedSoloTokensRef.current = {
                key: attemptKey,
                tokens: Object.fromEntries(
                  Object.entries(tracks).map(([index, track]) => [index, track.token]),
                ),
                regionsSignature: renderedSignature,
              };
              setSoloTracks({ status: "done", tracks });
              setRenderedRegionsSignature(renderedSignature);
            }
            return;
          }
          if (status.status === "error") {
            if (!cancelled) {
              setSoloTracks({ status: "error", tracks: {} });
            }
            return;
          }
          await new Promise((resolve) => window.setTimeout(resolve, 1500));
        }
      } catch {
        if (!cancelled) {
          setSoloTracks({ status: "error", tracks: {} });
        }
      }
    }

    void prepare();
    return () => {
      cancelled = true;
    };
  }, [session, selectedFile, overlapRegions, speakerTurns, soloTrackRegionCount, restoreSoloTracks, trackRenderNonce]);

  // Only the fallback gate needs per-frame gain: it rides region edges on the
  // clock, which timeupdate's ~4 Hz would step through audibly. Once the speaker
  // tracks carry the sound there is no automation left to run, so the loop stops
  // -- that steadiness is the point. It also pauses in hidden tabs, where the
  // timeupdate listener above keeps gating.
  useEffect(() => {
    if (!fallbackGateActive) {
      return;
    }
    let frame = 0;
    const tick = () => {
      applyPlaybackVolume();
      frame = window.requestAnimationFrame(tick);
    };
    frame = window.requestAnimationFrame(tick);
    return () => window.cancelAnimationFrame(frame);
  }, [fallbackGateActive, applyPlaybackVolume]);

  const transcriptWords = useMemo(() => new Map(activeWords.map((word) => [word.id, word])), [activeWords]);
  // Custom speaker names join the vocabulary sent to WhisperX so unusual or
  // foreign names transcribe correctly; default "Speaker N" names are noise.
  const transcriptionVocabulary = useMemo(() => {
    const customSpeakerNames = speakerInputs
      .map((speaker) => speaker.name.trim())
      .filter((name) => name && !/^speaker \d+$/i.test(name));
    return mergeVocabularyTexts(glossaryText, customSpeakerNames.join("\n")).replace(/\n/g, ", ");
  }, [glossaryText, speakerInputs]);
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
    () => (activeEditor ? detectJargonCandidates(activeWords, activeEditor.captions, glossaryText, lowConfidenceThreshold) : []),
    [activeEditor, activeWords, glossaryText, lowConfidenceThreshold],
  );
  const qaReport = useMemo(
    () => buildQaReport(activeEditor?.captions ?? [], activeWords, glossaryMatches, lowConfidenceThreshold),
    [activeEditor, activeWords, glossaryMatches, lowConfidenceThreshold],
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
            eyebrow: "Vocabulary",
            title: "Words that transcribe wrong",
            detail: `${glossaryTerms.length} term${glossaryTerms.length === 1 ? "" : "s"}, ${jargonCandidates.length} suggested`,
          }
        : sidePanelTab === "overlaps"
          ? {
              eyebrow: "Overlaps",
              title: "Untangle simultaneous speech",
              detail: `${overlapRegions.length} overlap${overlapRegions.length === 1 ? "" : "s"} found`,
            }
          : sidePanelTab === "regions"
            ? {
                eyebrow: "Regions",
                title: "Speaker regions",
                detail: "Zoom in and fix handoffs",
              }
          : sidePanelTab === "restore"
            ? {
                eyebrow: "Restore",
                title: "Diamond speech restoration",
                detail: "Studio-quality 44.1 kHz",
              }
          : sidePanelTab === "convert"
            ? {
                eyebrow: "Convert",
                title: "Seed-VC voice conversion",
                detail: "Re-voice with a reference clip",
              }
          : sidePanelTab === "patch"
            ? {
                eyebrow: "Patch",
                title: "F5-TTS speech editing",
                detail: "Regenerate muffled words in place",
              }
          : sidePanelTab === "master"
            ? {
                eyebrow: "Master",
                title: "Audio post production",
                detail: processedAudio ? "Master ready" : "Not processed",
              }
            : sidePanelTab === "export"
              ? {
                  eyebrow: "Export",
                  title: "Outputs",
                  detail: activeEditor ? `${activeEditor.captions.length} captions` : "No session",
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
        : sidePanelTab === "overlaps"
          ? `${overlapRegions.length} overlap${overlapRegions.length === 1 ? "" : "s"}`
          : sidePanelTab === "regions"
            ? `${editableRegions.length} region${editableRegions.length === 1 ? "" : "s"}${regionOverrides ? " (manual)" : ""}`
          : sidePanelTab === "restore"
            ? "Diamond restoration"
            : sidePanelTab === "convert"
              ? "Seed-VC conversion"
            : sidePanelTab === "patch"
              ? "F5-TTS patching"
            : sidePanelTab === "master"
            ? processedAudio
              ? "Processed audio loaded"
              : "Local processing"
            : sidePanelTab === "export"
              ? "SRT, transcript, edit guide"
              : `${qaReport.summary.flaggedCaptionCount} caption${qaReport.summary.flaggedCaptionCount === 1 ? "" : "s"}`;
  const multiSpeaker = (activeEditor?.speakers.length ?? speakerInputs.length) > 1;

  const activeCaptionIndex = useMemo(() => {
    if (!activeEditor) {
      return -1;
    }
    return activeEditor.captions.findIndex((caption) => currentTime >= caption.start && currentTime <= caption.end);
  }, [activeEditor, currentTime]);

  // Simultaneous speech renders as multiple co-timed captions; highlight all
  // of them, not just the first one the playhead lands in.
  const activeCaptionIds = useMemo(() => {
    const ids = new Set<string>();
    for (const caption of activeEditor?.captions ?? []) {
      if (currentTime >= caption.start && currentTime <= caption.end) {
        ids.add(caption.id);
      }
    }
    return ids;
  }, [activeEditor, currentTime]);

  // Captions that co-occur with a different speaker's caption (they export as
  // one dialogue cue and get a badge in the editor).
  const simultaneousCaptionIds = useMemo(() => {
    const ids = new Set<string>();
    const sorted = [...(activeEditor?.captions ?? [])].sort((a, b) => a.start - b.start);
    for (let i = 0; i < sorted.length; i += 1) {
      for (let j = i + 1; j < sorted.length && sorted[j].start < sorted[i].end; j += 1) {
        if (captionsAreSimultaneous(sorted[i], sorted[j])) {
          ids.add(sorted[i].id);
          ids.add(sorted[j].id);
        }
      }
    }
    return ids;
  }, [activeEditor]);

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
  // The same events as ticks for the Regions strip. Reduced to the fields the
  // panel draws so it never has to import App's editor types.
  const regionMarkers = useMemo<RegionMarker[]>(
    () =>
      speakerTimelineEvents.map((event) => ({
        id: event.id,
        time: event.time,
        kind: event.kind,
        label: event.label,
      })),
    [speakerTimelineEvents],
  );
  // Tight handoffs and overlaps, keyed by the caption they fall inside, so the
  // caption list can flag them in the gutter. Plain switches are normal
  // conversation and would flag every alternation.
  const reviewableSpeakerEvents = useMemo(
    () => speakerTimelineEvents.filter((event) => event.kind !== "switch"),
    [speakerTimelineEvents],
  );
  const captionEventsByIndex = useMemo(() => {
    const map = new Map<number, SpeakerTimelineEvent[]>();
    const captions = activeEditor?.captions ?? [];
    for (const event of speakerTimelineEvents) {
      if (event.kind === "switch") {
        continue;
      }
      let index = event.captionIndex ?? -1;
      if (index < 0 || index >= captions.length) {
        index = captions.findIndex((caption) => event.time >= caption.start && event.time <= caption.end);
      }
      if (index < 0) {
        continue;
      }
      const list = map.get(index) ?? [];
      list.push(event);
      map.set(index, list);
    }
    return map;
  }, [activeEditor, speakerTimelineEvents]);
  const focusTokenRef = useRef(0);
  const lastFollowedBlockRef = useRef<string | null>(null);
  const acknowledgedWordIdSet = useMemo(
    () => new Set(acknowledgedLowConfidenceWordIds),
    [acknowledgedLowConfidenceWordIds],
  );
  // Live count shown next to the sensitivity dial so the user can see how many
  // words each threshold flags before committing to it.
  const flaggedLowConfidenceCount = useMemo(
    () =>
      activeWords.reduce(
        (count, word) =>
          count + (isLowConfidenceWord(word, lowConfidenceThreshold) && !acknowledgedWordIdSet.has(word.id) ? 1 : 0),
        0,
      ),
    [activeWords, lowConfidenceThreshold, acknowledgedWordIdSet],
  );

  useEffect(() => {
    document.documentElement.dataset.theme = themeDark ? "dark" : "light";
    window.localStorage.setItem(THEME_STORAGE_KEY, themeDark ? "dark" : "light");
  }, [themeDark]);

  // Close the view-options popover on outside clicks.
  useEffect(() => {
    if (!viewOptionsOpen) {
      return;
    }
    const onPointerDown = (event: MouseEvent) => {
      if (viewOptionsRef.current && !viewOptionsRef.current.contains(event.target as Node)) {
        setViewOptionsOpen(false);
      }
    };
    document.addEventListener("mousedown", onPointerDown);
    return () => document.removeEventListener("mousedown", onPointerDown);
  }, [viewOptionsOpen]);

  const hasLoadedSession = Boolean(session || activeEditor);

  // Close the setup drawer with Escape.
  useEffect(() => {
    if (!setupDrawerOpen) {
      return;
    }
    const onKeyDown = (event: globalThis.KeyboardEvent) => {
      if (event.key === "Escape") {
        setSetupDrawerOpen(false);
      }
    };
    document.addEventListener("keydown", onKeyDown);
    return () => document.removeEventListener("keydown", onKeyDown);
  }, [setupDrawerOpen]);

  // Accept audio dropped anywhere in the window, not just on the dropzone.
  useEffect(() => {
    let depth = 0;
    const isFileDrag = (event: DragEvent) => Boolean(event.dataTransfer?.types.includes("Files"));
    const onDragEnter = (event: DragEvent) => {
      if (isFileDrag(event)) {
        depth += 1;
        setGlobalDragActive(true);
      }
    };
    const onDragLeave = (event: DragEvent) => {
      if (isFileDrag(event)) {
        depth = Math.max(0, depth - 1);
        if (depth === 0) {
          setGlobalDragActive(false);
        }
      }
    };
    const onDragOver = (event: DragEvent) => {
      if (isFileDrag(event)) {
        event.preventDefault();
      }
    };
    const onDrop = (event: DragEvent) => {
      depth = 0;
      setGlobalDragActive(false);
      if (event.defaultPrevented) {
        return; // an inner dropzone (e.g. Restore) already consumed this file
      }
      const file = event.dataTransfer?.files?.[0];
      if (file && (file.type.startsWith("audio/") || file.type.startsWith("video/"))) {
        event.preventDefault();
        setAudioFile(file);
      }
    };
    window.addEventListener("dragenter", onDragEnter);
    window.addEventListener("dragleave", onDragLeave);
    window.addEventListener("dragover", onDragOver);
    window.addEventListener("drop", onDrop);
    return () => {
      window.removeEventListener("dragenter", onDragEnter);
      window.removeEventListener("dragleave", onDragLeave);
      window.removeEventListener("dragover", onDragOver);
      window.removeEventListener("drop", onDrop);
    };
  }, [audioUrl]);

  // Manual scrolling suspends follow-playback briefly so it never fights the user.
  useEffect(() => {
    const markUserScroll = () => {
      userScrollAtRef.current = Date.now();
    };
    window.addEventListener("wheel", markUserScroll, { passive: true });
    window.addEventListener("touchmove", markUserScroll, { passive: true });
    return () => {
      window.removeEventListener("wheel", markUserScroll);
      window.removeEventListener("touchmove", markUserScroll);
    };
  }, []);

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

    if (Date.now() - userScrollAtRef.current < FOLLOW_SCROLL_SUSPEND_MS) {
      return;
    }

    lastFollowedBlockRef.current = key;
    const element = viewMode === "transcript" ? paragraphRefs.current[activeIndex] : captionRefs.current[activeIndex];
    if (element) {
      const bounds = element.getBoundingClientRect();
      const fullyVisible = bounds.top >= 0 && bounds.bottom <= window.innerHeight;
      if (!fullyVisible) {
        scrollIntoViewCentered(element);
      }
    }
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

      if (
        (event.key === "ArrowLeft" || event.key === "ArrowRight") &&
        !event.ctrlKey &&
        !event.altKey &&
        !event.metaKey &&
        !event.shiftKey &&
        !isEditableTarget(event.target)
      ) {
        event.preventDefault();
        skipBy(event.key === "ArrowLeft" ? -3 : 3);
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
      lowConfidenceThreshold,
      restoreSoloTracks,
      // Written by the solo-track effect when a render lands, so this is the
      // exact key that effect computed. Rebuilding the key here instead is how
      // the persisted and recomputed keys drift apart -- and a key that never
      // matches means every reload re-renders.
      soloTracks: persistedSoloTokensRef.current ?? null,
      soloTrackRenderNonce: trackRenderNonce,
      convertedVoices: convertedVoiceEntries.length
        ? {
            tokens: Object.fromEntries(convertedVoiceEntries.map((entry) => [entry.speakerId, entry.token])),
            active: Object.fromEntries(
              convertedVoiceEntries.map((entry) => [entry.speakerId, activeVoice.get(entry.speakerId) ?? "original"]),
            ),
          }
        : null,
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
    setSpeakerAssignmentMode(persisted.speakerAssignmentMode === "segment" ? "segment" : "word");
    setGlossaryText(restoredGlossaryText);
    setSkipCuts(typeof persisted.skipCuts === "boolean" ? persisted.skipCuts : false);
    setClickToPlay(typeof persisted.clickToPlay === "boolean" ? persisted.clickToPlay : true);
    setFollowPlayback(persisted.followPlayback !== false);
    setShowLineGuides(Boolean(persisted.showLineGuides));
    setShowTimingHighlights(Boolean(persisted.showTimingHighlights));
    setViewMode(persisted.viewMode === "transcript" ? "transcript" : "subtitles");
    setSidePanelTab(coerceSidePanelTab(persisted.sidePanelTab));
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
    setLowConfidenceThreshold(
      typeof persisted.lowConfidenceThreshold === "number" && Number.isFinite(persisted.lowConfidenceThreshold)
        ? Math.min(1, Math.max(0, persisted.lowConfidenceThreshold))
        : DEFAULT_LOW_CONFIDENCE_THRESHOLD,
    );
    setSelection(null);
    setCurrentTime(0);
    if (options && "audioFile" in options) {
      setAudioFile(options.audioFile ?? null);
    }
    setRestoreSoloTracks(Boolean(persisted.restoreSoloTracks));
    // Adopt previously rendered solo tracks (revalidated by the auto-solo
    // effect) instead of re-running separation after every reload.
    persistedSoloTokensRef.current =
      "soloTracks" in persisted && persisted.soloTracks?.key && persisted.soloTracks.tokens
        ? persisted.soloTracks
        : null;
    // Both halves of the freshness bookkeeping have to come back with the
    // tokens: the nonce because it is baked into the key the effect recomputes,
    // the signature because it is what flags region edits made before the
    // reload. Dropping either one strands the tracks -- a lost nonce re-renders
    // unconditionally, a lost signature hides the staleness warning and leaves
    // the re-render button disabled with no way to reach it.
    setTrackRenderNonce(
      typeof persisted.soloTrackRenderNonce === "number" && Number.isFinite(persisted.soloTrackRenderNonce)
        ? persisted.soloTrackRenderNonce
        : 0,
    );
    setRenderedRegionsSignature(persistedSoloTokensRef.current?.regionsSignature ?? null);
    soloTracksAttemptRef.current = null;
    setSoloTracks({ status: "idle", tracks: {} });
    // Same deal for the converted voices, minus the render key: they are keyed by
    // speaker id and revalidated one artifact at a time by the effect the nonce
    // wakes up.
    const restoredConvertedVoices =
      "convertedVoices" in persisted && persisted.convertedVoices?.tokens ? persisted.convertedVoices : null;
    persistedConvertedVoicesRef.current = restoredConvertedVoices
      ? { tokens: restoredConvertedVoices.tokens, active: restoredConvertedVoices.active ?? {} }
      : null;
    setConvertedVoices(new Map());
    setActiveVoice(new Map());
    setConvertedRestoreNonce((nonce) => nonce + 1);

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
    setSoloTracks({ status: "idle", tracks: {} });
    setMutedSpeakerIds(new Set());
    // A converted voice is a conversion of one recording's isolated track, so it
    // means nothing against different audio.
    setConvertedVoices(new Map());
    setActiveVoice(new Map());
    soloTracksAttemptRef.current = null;
    // persistedSoloTokensRef survives on purpose: re-loading the session's own
    // audio after a reload should adopt the finished tracks, and the session
    // key check rejects tokens that belong to a different transcription.
    if (audioUrl) {
      URL.revokeObjectURL(audioUrl);
    }
    setAudioUrl(file ? URL.createObjectURL(file) : null);
    if (file) {
      // The waveform is the scrubber, so analyze as soon as audio arrives.
      void handleAnalyzeWaveform(file, { quiet: true });
    }
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

  function switchPlaybackSource(next: "original" | "processed") {
    if (next === playbackSource) {
      return;
    }
    const from = audioRef.current;
    const to = next === "processed" ? masteredAudioRef.current : originalAudioRef.current;
    if (!from || !to || !processedAudio) {
      setPlaybackSource(next);
      return;
    }

    const wasPlaying = !from.paused;
    if (processedAudio.hasCutTimeline) {
      // Timelines differ; map the position through the cut list.
      const cuts = processedAudio.cutList;
      const mapped = next === "processed" ? remapTime(from.currentTime, cuts) : unremapTime(from.currentTime, cuts);
      from.pause();
      to.currentTime = mapped;
    } else if (to.paused || Math.abs(to.currentTime - from.currentTime) > 0.25) {
      // Not co-playing yet (or drifted); align before the swap.
      to.currentTime = from.currentTime;
    }
    // Mute flags are not touched here: the audibility effect owns them, and the
    // unmuted speakers' tracks -- not the new clock -- may be carrying the sound.
    if (wasPlaying) {
      void to.play().catch(() => undefined);
    }
    setPlaybackSource(next);
  }

  function handleMasteringProcessed(result: MasteringResult, url: string) {
    const previousTime = audioRef.current?.currentTime ?? 0;
    const wasPlaying = audioRef.current ? !audioRef.current.paused : false;
    const hasCutTimeline = result.duration_after < result.duration_before - 0.01;
    setProcessedAudio({ url, filename: result.output_filename, label: "Mastered", hasCutTimeline, cutList: result.cut_list });
    setPlaybackSource("processed");
    // Once the new master is loaded, carry the listening position (mapped
    // through the cuts when the timeline changed) and keep playing.
    window.setTimeout(() => {
      const mastered = masteredAudioRef.current;
      if (!mastered) {
        return;
      }
      mastered.currentTime = hasCutTimeline ? remapTime(previousTime, result.cut_list) : previousTime;
      if (wasPlaying) {
        void mastered.play().catch(() => undefined);
      }
    }, 200);
    if (!hasCutTimeline) {
      void refreshWaveformFromMaster(result.token);
    }
    setStatusMessage("Mastering finished. Playback now uses the processed audio.");
  }

  function handleSeparationProcessed(result: SeparationResult, url: string) {
    const previousTime = audioRef.current?.currentTime ?? 0;
    const wasPlaying = audioRef.current ? !audioRef.current.paused : false;
    // Separation never changes the timeline, so the A/B swap stays instant.
    setProcessedAudio({ url, filename: result.output_filename, label: "Separated", hasCutTimeline: false, cutList: [] });
    setPlaybackSource("processed");
    window.setTimeout(() => {
      const processed = masteredAudioRef.current;
      if (!processed) {
        return;
      }
      processed.currentTime = previousTime;
      if (wasPlaying) {
        void processed.play().catch(() => undefined);
      }
    }, 200);
    const applied = result.regions.filter((region) => region.applied).length;
    setStatusMessage(
      `Separation finished on ${result.device_used.toUpperCase()}: ${applied} overlap${applied === 1 ? "" : "s"} processed. Playback now uses the processed audio.`,
    );
  }

  function applySeparatedWords(report: RegionReport) {
    const stemWords = report.words ?? [];
    if (!stemWords.length) {
      return;
    }
    const speaker = activeEditor?.speakers[report.target_speaker_index] ?? speakerInputs[report.target_speaker_index] ?? null;
    const idPrefix = `sep-${Date.now()}-${Math.round(report.start * 1000)}`;
    const words: WordToken[] = stemWords.map((word, index) => ({
      id: `${idPrefix}-w-${index}`,
      text: word.text,
      start: word.start,
      end: word.end,
      confidence: 1,
      low_confidence: false,
      speaker_id: speaker?.id ?? null,
      speaker_name: speaker?.name ?? null,
    }));

    // The recovered voice coexists with what is already transcribed in the
    // overlap, so the words are added as a new caption instead of replacing
    // the other speaker's text.
    const captionText = stemWords.map((word) => word.text).join(" ");
    const caption: Caption = {
      id: `${idPrefix}-c-0`,
      start: words[0].start,
      end: words[words.length - 1].end,
      speaker_id: speaker?.id ?? null,
      speaker_name: speaker?.name ?? null,
      lines: normalizeCaptionEditorLines(captionText),
      word_ids: words.map((word) => word.id),
      blank_after: false,
    };

    setHistory((current) => {
      if (!current.present) {
        return current;
      }
      const nextWords = [...current.present.words, ...words].sort((left, right) => left.start - right.start);
      const nextEditor = cloneEditorState(current.present.editor);
      nextEditor.captions = [...nextEditor.captions, caption].sort((left, right) => left.start - right.start);
      nextEditor.paragraphs = buildParagraphsFromCaptions(nextEditor.captions);
      return {
        past: [...current.past, cloneWorkspaceState(current.present)].slice(-120),
        present: { ...current.present, editor: nextEditor, words: nextWords },
        future: [],
      };
    });
    setStatusMessage(
      `Added ${words.length} recovered word${words.length === 1 ? "" : "s"} for ${speaker?.name ?? "the spotlighted speaker"} at ${formatClock(report.start)}.`,
    );
  }

  async function handleResplitCaptions() {
    if (!activeWorkspace?.words.length) {
      return;
    }
    setResplitting(true);
    try {
      const response = await fetch(`${API_BASE_URL}/api/rebuild-captions`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ words: activeWorkspace.words }),
      });
      if (!response.ok) {
        throw new Error(`Re-split failed (${response.status}). Is the backend running?`);
      }
      const payload = (await response.json()) as { captions: Caption[] };
      commit((draft) => {
        draft.captions = payload.captions;
      });
      setStatusMessage("Captions re-split with the deterministic rules. Undo restores the previous state.");
    } catch (error) {
      setStatusMessage(error instanceof Error ? error.message : "Re-split failed.");
    } finally {
      setResplitting(false);
    }
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

  // Overrides let a caller request a transcription that differs from the sidebar setup
  // without touching the sidebar state the workspace is persisted with.
  function buildTranscriptionFormData(audioFile: File, overrides?: TranscriptionOverrides): FormData {
    const effectiveSpeakers = normalizeSpeakers(overrides?.speakers ?? speakerInputs);
    const effectiveSpeakerCount = overrides?.speakers ? effectiveSpeakers.length : speakerCount;
    const effectiveSpeakerAssignmentMode: SpeakerAssignmentMode = effectiveSpeakerCount > 1 ? speakerAssignmentMode : "segment";
    const formData = new FormData();
    formData.append("audio", audioFile);
    formData.append("model", overrides?.model ?? model);
    formData.append("speaker_count", String(effectiveSpeakerCount));
    formData.append(
      "speakers_json",
      JSON.stringify(effectiveSpeakers.map((speaker) => ({ id: speaker.id, name: speaker.name }))),
    );
    formData.append("speaker_assignment_mode", effectiveSpeakerAssignmentMode);
    formData.append("remove_disfluencies", String(removeDisfluencies));
    if (transcriptionVocabulary.trim()) {
      formData.append("hotwords", transcriptionVocabulary.trim());
    }
    return formData;
  }

  async function requestTranscription(audioFile: File, overrides?: TranscriptionOverrides): Promise<TranscriptResponse> {
    const response = await fetch(`${API_BASE_URL}/api/transcribe`, {
      method: "POST",
      body: buildTranscriptionFormData(audioFile, overrides),
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
    setSpeakerAssignmentMode("word");
    setMutedSpeakerIds(new Set());
    setGlossaryText("");
    setFindText("");
    setReplaceText("");
    setFollowPlayback(false);
    setSkipCuts(false);
    setShowLineGuides(false);
    setShowTimingHighlights(true);
    setSidePanelTab("guide");
    setIsGuidePanelCollapsed(false);
    setExtendCaptionsOnExport(false);
    setNormalizeExportTimingTo30Fps(false);
    setShowSpeakerAttributionOptions(false);
    setRemoveDisfluencies(false);
    setAcknowledgedLowConfidenceWordIds([]);
    setLowConfidenceThreshold(DEFAULT_LOW_CONFIDENCE_THRESHOLD);
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

  async function handleAnalyzeWaveform(file?: File, options?: { quiet?: boolean }) {
    const source = file ?? selectedFile;
    if (!source) {
      setStatusMessage("Choose an audio file first.");
      return;
    }

    setWaveformLoading(true);
    try {
      const analysis = await requestWaveformAnalysis(source);
      setWaveformAnalysis(analysis);
      if (!options?.quiet) {
        setStatusMessage(
          `Waveform analyzed: ${analysis.speech_spans.length} speech region${analysis.speech_spans.length === 1 ? "" : "s"} detected.`,
        );
      }
    } catch (error) {
      if (!options?.quiet) {
        const message = error instanceof Error ? error.message : "Unknown error";
        setStatusMessage(message);
      }
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

  // Region edits go through the same editor commit as the waveform snap above,
  // which is what puts them on the undo stack and into the autosave.
  function handleRegionsChange(next: SpeakerRegion[]) {
    commit((draft) => {
      draft.regionOverrides = next.map((region) => ({ ...region }));
    });
  }

  function handleRegionsReset() {
    if (!activeEditor?.regionOverrides) {
      return;
    }
    commit((draft) => {
      draft.regionOverrides = null;
    });
    setStatusMessage("Speaker regions are derived from the audio again.");
  }

  async function handleTranscribe() {
    if (!selectedFile) {
      setStatusMessage("Choose an audio file first.");
      return;
    }

    setLoading(true);
    setStatusMessage("Running WhisperX. Large models on long files will take time even on GPU.");
    requestCompletionNotificationPermission();
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
      notifyWorkFinished("Transcription finished", selectedFile.name);
    } catch (error) {
      const message = error instanceof Error ? error.message : "Unknown error";
      setStatusMessage(message);
      notifyWorkFinished("Transcription failed", message);
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

    const importedHasSpeakers = hasExplicitImportedSpeakerLabels(captions);
    const importedSpeakers = importedHasSpeakers ? normalizeImportedCaptions(captions).speakers : null;
    const diarizingForOverlaps = Boolean(importedSpeakers && importedSpeakers.length > 1);

    setLoading(true);
    setStatusMessage(
      options?.retimeCaptions
        ? "Running a quick `tiny` word-timing pass on the uploaded audio and retiming the imported captions."
        : diarizingForOverlaps
          ? "Running a `tiny` word-timing pass plus diarization (for overlap detection) while preserving the imported SRT timing."
          : "Running a quick `tiny` word-timing pass on the uploaded audio while preserving the imported SRT timing.",
    );
    requestCompletionNotificationPermission();
    try {
      const payload = await requestTranscription(resumeAudioFile, {
        // Legacy loads only need word timings to match the imported text onto the audio;
        // no transcribed word is surfaced as content, so the smallest model is enough.
        // A labeled SRT supplies its own speakers, which the load sends along so
        // diarization still runs (overlap detection needs its turns) — but the SRT's
        // labels, not diarization's arbitrary slots, are what re-tag the words.
        model: "tiny",
        speakers: importedSpeakers ?? undefined,
      });
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
      notifyWorkFinished("Transcription finished", resumeAudioFile.name);
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
    // No normalizeSpeakers here: it would snap a cleared field back to
    // "Speaker N" on every keystroke. Blanks are resolved on blur instead.
    setSpeakerInputs((current) =>
      current.map((speaker, itemIndex) => (itemIndex === index ? { ...speaker, name } : speaker)),
    );
    setSession((current) => {
      if (!current) {
        return current;
      }
      return {
        ...current,
        speakers: current.speakers.map((speaker) => (speaker.id === targetSpeakerId ? { ...speaker, name } : speaker)),
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

  function reassignCaptionSpeaker(index: number, speakerId: number) {
    if (!activeEditor) {
      return;
    }
    const selected = activeEditor.captions[index];
    const nextSpeaker = activeEditor.speakers.find((speaker) => speaker.id === speakerId);
    if (!selected || !nextSpeaker || selected.speaker_id === nextSpeaker.id) {
      return;
    }
    const wordIds = new Set(selected.word_ids);
    commit((draft) => {
      draft.captions[index].speaker_id = nextSpeaker.id;
      draft.captions[index].speaker_name = nextSpeaker.name;
    }, {
      transformWords: (words) =>
        words.map((word) =>
          wordIds.has(word.id)
            ? { ...word, speaker_id: nextSpeaker.id, speaker_name: nextSpeaker.name }
            : word,
        ),
    });
  }

  function focusSpeakerEvent(event: SpeakerTimelineEvent) {
    seekAudio(Math.max(0, event.start - 1), { play: false });
    const captions = activeEditor?.captions ?? [];
    let index = event.captionIndex ?? -1;
    if (index < 0 || index >= captions.length) {
      index = captions.findIndex((caption) => event.time >= caption.start && event.time <= caption.end);
    }
    if (index >= 0 && viewMode === "subtitles") {
      captionRefs.current[index]?.scrollIntoView({ block: "center", behavior: "smooth" });
    }
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

  const setupPanel = (
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
                <input
                  value={speaker.name}
                  onChange={(event) => updateSpeakerName(index, event.target.value)}
                  onBlur={(event) => {
                    const trimmed = event.target.value.trim();
                    if (trimmed !== event.target.value || !trimmed) {
                      updateSpeakerName(index, trimmed || `Speaker ${index + 1}`);
                    }
                  }}
                />
              </label>
            ))}
          </div>

          <label className="toggle-row">
            <input type="checkbox" checked={removeDisfluencies} onChange={(event) => setRemoveDisfluencies(event.target.checked)} />
            Remove filler words / simple stutters
          </label>

          <p className="helper-text">`Word` mode switches speakers using each word timestamp, which is usually tighter around handoffs. The project vocabulary lives in the Vocab tab and biases transcription toward names and technical terms Whisper tends to mishear.</p>
          {showPyannoteSetupHint ? (
            <p className="helper-text">Multiple speakers need Hugging Face access for `pyannote/speaker-diarization-3.1`. Put `DIARIZATION_AUTH_TOKEN=hf_...` in `.env`.</p>
          ) : null}

          <button className="primary-button" disabled={loading || retranscribing || !selectedFile} onClick={handleTranscribe}>
            {loading ? "Transcribing..." : "Transcribe"}
          </button>
          <details className="rail-details">
            <summary>Project file (save / resume)</summary>
            <p className="helper-text">Project files preserve the full editor state, guide blocks, timings, confidence data, and embedded audio for playback.</p>
            <label>
              Project file
              <input type="file" accept=".json,.subtitle-workbench.json,application/json" onChange={(event) => setResumeProjectFile(event.target.files?.[0] ?? null)} />
            </label>
            <div className="inline-actions">
              <button onClick={handleLoadProject}>Load project</button>
              <button disabled={!activeEditor || !selectedFile} onClick={handleDownloadProject}>Save project</button>
            </div>
          </details>
          <details className="rail-details">
            <summary>Legacy: audio + SRT</summary>
            <p className="helper-text">Use this when you only have an audio file and an edited `.srt`. The audio gets a quick word-timing pass with the `tiny` model regardless of the model chosen above — those words only anchor the SRT text to the audio and are never shown as content. Default load preserves the original SRT timing and only rematches text to those words.</p>
            <p className="helper-text">When the SRT carries speaker labels (`NAME:` on its own line or as an inline `NAME: dialogue` prefix), those labels become the session speakers, stay editable after load, and drive only-speaker playback. `Load + retime to audio` is opt-in and rewrites caption timing from the audio.</p>
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
          </details>
        </section>
  );

  return (
    <div className={`app-shell ${hasLoadedSession ? "no-rail" : ""}`}>
      {!hasLoadedSession ? (
        <aside className="control-rail">
          <div className="brand-block">
            <p className="eyebrow">Local WhisperX editor</p>
            <h1>Subtitle Workbench</h1>
            <p className="lede">Transcribe, edit directly in place, and export subtitles, transcript text, and an edit guide.</p>
          </div>
          {setupPanel}
        </aside>
      ) : null}

      {setupDrawerOpen ? (
        <div className="drawer-backdrop" onClick={() => setSetupDrawerOpen(false)}>
          <aside className="setup-drawer" onClick={(event) => event.stopPropagation()}>
            <div className="drawer-header">
              <h2>Source and setup</h2>
              <button className="icon-button" aria-label="Close setup" onClick={() => setSetupDrawerOpen(false)}>
                <X size={16} />
              </button>
            </div>
            {setupPanel}
          </aside>
        </div>
      ) : null}

      <main className="workspace">
        <section className="player-panel">
          <div className="transport-bar">
            <div className="transport-cluster">
              <button className="icon-button transport-play" onClick={togglePlayback} disabled={!audioUrl} aria-label={isPlaying ? "Pause" : "Play"}>
                {isPlaying ? <Pause size={18} /> : <Play size={18} />}
              </button>
              <button
                className="icon-button"
                onClick={() => setUserMuted((muted) => !muted)}
                disabled={!audioUrl}
                title={userMuted ? "Unmute" : "Mute"}
                aria-label={userMuted ? "Unmute" : "Mute"}
              >
                {userMuted ? <VolumeX size={16} /> : <Volume2 size={16} />}
              </button>
              <select
                className="transport-speed"
                value={playbackRate}
                onChange={(event) => setPlaybackRate(Number(event.target.value))}
                aria-label="Playback speed"
              >
                <option value={0.75}>0.75×</option>
                <option value={1}>1×</option>
                <option value={1.25}>1.25×</option>
                <option value={1.5}>1.5×</option>
                <option value={2}>2×</option>
              </select>
              <span className="transport-clock">
                {formatClock(currentTime)}
                <span className="transport-clock-total"> / {formatClock(audioDuration)}</span>
              </span>
            </div>
            <span className="transport-file" title={selectedFile?.name ?? session?.audio_filename ?? undefined}>
              {selectedFile?.name ?? session?.audio_filename ?? "No file loaded"}
            </span>
            <div className="transport-cluster">
              {processedAudio ? (
                <div className="mode-toggle">
                  <button
                    className={playbackSource === "original" ? "is-active" : ""}
                    onClick={() => switchPlaybackSource("original")}
                  >
                    Original
                  </button>
                  <button
                    className={playbackSource === "processed" ? "is-active" : ""}
                    onClick={() => switchPlaybackSource("processed")}
                  >
                    {processedAudio.label}
                  </button>
                </div>
              ) : null}
              <div className="mode-toggle">
                <button className={viewMode === "transcript" ? "is-active" : ""} onClick={() => setViewMode("transcript")}>Transcript</button>
                <button className={viewMode === "subtitles" ? "is-active" : ""} onClick={() => setViewMode("subtitles")}>Subtitles</button>
              </div>
              <button
                disabled={viewMode === "transcript" ? !activeEditor?.paragraphs.length : !activeEditor?.captions.length}
                onClick={viewMode === "transcript" ? jumpToCurrentTranscript : jumpToCurrentSubtitle}
              >
                Jump to current
              </button>
              {hasLoadedSession ? (
                <button
                  className="icon-button"
                  title="Source and setup"
                  aria-label="Source and setup"
                  onClick={() => setSetupDrawerOpen(true)}
                >
                  <SlidersHorizontal size={16} />
                </button>
              ) : null}
              <div className="view-options" ref={viewOptionsRef}>
                <button
                  type="button"
                  className={`icon-button ${viewOptionsOpen ? "is-active" : ""}`}
                  title="View options"
                  aria-label="View options"
                  onClick={() => setViewOptionsOpen((open) => !open)}
                >
                  <Settings2 size={16} />
                </button>
                {viewOptionsOpen ? (
                  <div className="view-options-popover">
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
                    <label className="toggle-row">
                      <input type="checkbox" checked={themeDark} onChange={(event) => setThemeDark(event.target.checked)} />
                      Dark theme
                    </label>
                    <label className="slider-row">
                      Low-confidence sensitivity
                      <input
                        type="range"
                        min={0}
                        max={0.95}
                        step={0.05}
                        value={lowConfidenceThreshold}
                        onChange={(event) => setLowConfidenceThreshold(Number(event.target.value))}
                      />
                      <span className="slider-row-meta">
                        {lowConfidenceThreshold <= 0
                          ? "Off — nothing flagged"
                          : `Below ${Math.round(lowConfidenceThreshold * 100)}% — flags ${flaggedLowConfidenceCount} word${flaggedLowConfidenceCount === 1 ? "" : "s"}`}
                      </span>
                    </label>
                    <p className="helper-text">Clicks always seek. `Ctrl+Space` play/pause. `←`/`→` skip 3s. `Shift+Space` toggles click autoplay.</p>
                  </div>
                ) : null}
              </div>
            </div>
          </div>
          {/* Speaker audibility has its own row: inside the transport pill the
              names were squeezed to a few characters, and a converted voice adds
              a second control per speaker. */}
          {soloableSpeakers.length ? (
            <div className="speaker-audibility-row">
              {soloableSpeakers.map((speaker) => {
                const speakerMuted = mutedSpeakerIds.has(speaker.id);
                const converted = convertedVoices.get(speaker.id) ?? null;
                const voice = activeVoice.get(speaker.id) === "converted" ? "converted" : "original";
                return (
                  <div key={speaker.id} className="speaker-audibility-group">
                    <div className="mode-toggle" role="group" aria-label={`Mute ${speaker.name}`}>
                      <button
                        type="button"
                        className={speakerMuted ? "is-muted" : "is-active"}
                        aria-pressed={!speakerMuted}
                        disabled={!audioUrl}
                        title={`${speakerMuted ? "Unmute" : "Mute"} ${speaker.name}`}
                        onClick={() => toggleSpeakerMuted(speaker.id)}
                      >
                        {speaker.name}
                      </button>
                    </div>
                    {converted ? (
                      <div className="mode-toggle" role="group" aria-label={`${speaker.name} voice`}>
                        <button
                          type="button"
                          className={voice === "original" ? "is-active" : ""}
                          disabled={!audioUrl}
                          onClick={() => setSpeakerVoice(speaker.id, "original")}
                        >
                          Original
                        </button>
                        <button
                          type="button"
                          className={voice === "converted" ? "is-active" : ""}
                          disabled={!audioUrl}
                          title="Play the converted voice in this speaker's place"
                          onClick={() => setSpeakerVoice(speaker.id, "converted")}
                        >
                          Converted
                        </button>
                      </div>
                    ) : null}
                  </div>
                );
              })}
            </div>
          ) : null}
          <audio ref={originalAudioRef} className="peer-audio-hidden" src={audioUrl ?? undefined} preload="auto" />
          {processedAudio ? (
            <audio ref={masteredAudioRef} className="peer-audio-hidden" src={processedAudio.url} preload="auto" />
          ) : null}
          {/* Every rendered solo track stays mounted and co-playing. Mounting only
              the selected one would make each switch a load-seek-play cycle, which
              is inherently gappy and lands the new element off position. */}
          {soloTrackEntries.map((entry) => (
            <audio
              key={entry.token}
              ref={soloTrackRefFor(entry.token)}
              className="peer-audio-hidden"
              src={entry.url}
              preload="auto"
            />
          ))}
          {/* Converted voices join the same element map, keyed by their own token:
              to the transport they are followers like any other track, so the
              Original/Converted switch is a mute swap too. */}
          {convertedVoiceEntries.map((entry) => (
            <audio
              key={entry.token}
              ref={soloTrackRefFor(entry.token)}
              className="peer-audio-hidden"
              src={entry.url}
              preload="auto"
            />
          ))}
          <WaveformTimeline
            analysis={waveformAnalysis}
            captions={activeEditor?.captions ?? []}
            speakerEvents={speakerTimelineEvents}
            overlapRegions={overlapRegions}
            currentTime={currentTime}
            theme={themeDark ? "dark" : "light"}
            onSeek={seekAudio}
          />
          <div className="waveform-strip">
            <button disabled={!waveformAnalysis || !activeEditor?.captions.length} onClick={handleAlignCaptionsToWaveform}>
              Snap subtitle edges
            </button>
            {waveformLoading ? (
              <span className="metric-chip">
                <Loader2 size={12} className="spin" aria-hidden /> Analyzing audio…
              </span>
            ) : null}
            {waveformAnalysis ? (
              <span className="metric-chip">{waveformAnalysis.speech_spans.length} speech regions</span>
            ) : null}
            {reviewableSpeakerEvents.length ? (
              <button
                type="button"
                className="waveform-event-chip event-tight_handoff"
                title="Open the speaker events list in the Guide panel"
                onClick={() => {
                  setSidePanelTab("guide");
                  setIsGuidePanelCollapsed(false);
                }}
              >
                <AlertTriangle size={12} aria-hidden />
                &nbsp;{reviewableSpeakerEvents.length} speaker {reviewableSpeakerEvents.length === 1 ? "event" : "events"} to review
              </button>
            ) : null}
            {overlapRegions.length ? (
              <button
                type="button"
                className="waveform-event-chip event-overlap"
                title="Open the Overlaps panel to spotlight or mute a voice"
                onClick={() => {
                  setSidePanelTab("overlaps");
                  setIsGuidePanelCollapsed(false);
                }}
              >
                <AudioLines size={12} aria-hidden />
                &nbsp;{overlapRegions.length} overlap{overlapRegions.length === 1 ? "" : "s"} to untangle
              </button>
            ) : null}
            {soloTracks.status === "running" ? (
              <span className="metric-chip" title="Preparing per-speaker audio so the speaker mute toggles isolate voices inside overlaps">
                <Loader2 size={12} className="spin" aria-hidden /> Isolating overlap voices…
              </span>
            ) : null}
            {soloTracks.status === "done" && Object.keys(soloTracks.tracks).length ? (
              <span
                className="metric-chip"
                title={
                  restoreSoloTracks
                    ? "Muting a speaker drops its Diamond-restored voice, even through overlaps"
                    : "Muting a speaker drops that voice from playback, even through overlaps"
                }
              >
                {restoreSoloTracks ? "Solo voices ready · restored" : "Solo voices ready"}
              </span>
            ) : null}
          </div>
          {activeWarnings.length ? (
            <details className="warning-details">
              <summary>
                {activeWarnings.length === 1
                  ? activeWarnings[0].message.length > 90
                    ? `${activeWarnings[0].message.slice(0, 89)}…`
                    : activeWarnings[0].message
                  : `${activeWarnings.length} warnings`}
              </summary>
              <div className="warning-stack">
                {activeWarnings.map((warning) => (
                  <p key={warning.code + warning.message} className="warning-chip">{warning.message}</p>
                ))}
              </div>
            </details>
          ) : null}
        </section>

        <div className={`editor-grid ${isGuidePanelCollapsed ? "has-collapsed-guide-panel" : ""}`}>
          <section className="editor-panel">
            {viewMode === "transcript" ? (
              <div className="transcript-view">
                <div className="editor-toolbar">
                  <span className="toolbar-title">
                    Transcript
                    <InfoTip text="Select text to mark it for cutting. Click in the text to seek audio to that point. Playback follows your click-to-play setting." />
                  </span>
                  <div className="inline-actions">
                    <button onClick={undo} disabled={!history.past.length}><Undo2 size={14} aria-hidden />&nbsp;Undo</button>
                    <button onClick={redo} disabled={!history.future.length}><Redo2 size={14} aria-hidden />&nbsp;Redo</button>
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
                        lowConfidenceThreshold={lowConfidenceThreshold}
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
                      <span className="toolbar-title">
                        Subtitles
                        <InfoTip text="Enter creates the next caption. Shift+Enter inserts a second line (captions max out at two). Backspace at the start merges backward; Delete at the end merges forward." />
                      </span>
                      <div className="inline-actions">
                        <button onClick={undo} disabled={!history.past.length}><Undo2 size={14} aria-hidden />&nbsp;Undo</button>
                        <button onClick={redo} disabled={!history.future.length}><Redo2 size={14} aria-hidden />&nbsp;Redo</button>
                        <button disabled={!activeEditor?.captions.length} onClick={reflowAllCaptions}>Reflow Lines</button>
                        <button disabled={!activeWorkspace?.words.length || resplitting} onClick={() => void handleResplitCaptions()}>
                          {resplitting ? "Re-splitting..." : "Re-split Captions"}
                        </button>
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
                            className={`caption-card ${caption.lines.length > 1 ? "is-multiline" : ""} ${activeCaptionIds.has(caption.id) ? "is-active" : ""} ${caption.blank_after ? "has-gap" : ""} ${simultaneousCaptionIds.has(caption.id) ? "is-simultaneous" : ""}`}
                            key={caption.id}
                          >
                            {multiSpeaker ? (
                              <div className={`caption-topline compact-topline ${showSpeakerBoundary ? "" : "caption-topline-ghost"}`}>
                                <select
                                  className="speaker-pill speaker-pill-select"
                                  value={caption.speaker_id === null ? "" : String(caption.speaker_id)}
                                  title="Change this caption's speaker"
                                  aria-label="Change this caption's speaker"
                                  onChange={(event) => reassignCaptionSpeaker(index, Number(event.target.value))}
                                >
                                  {caption.speaker_id === null ? (
                                    <option value="" disabled>{caption.speaker_name ?? "Speaker"}</option>
                                  ) : null}
                                  {activeEditor.speakers.map((speaker) => (
                                    <option key={speaker.id} value={speaker.id}>{speaker.name}</option>
                                  ))}
                                </select>
                              </div>
                            ) : null}
                            <div className="caption-row">
                            <div className="caption-gutter">
                              <button
                                type="button"
                                className="caption-time"
                                title="Seek to this caption"
                                onClick={() => seekAudio(caption.start, { play: clickToPlay })}
                              >
                                {formatGutterClock(caption.start)}
                              </button>
                              {(captionEventsByIndex.get(index) ?? []).slice(0, 2).map((event) => (
                                <button
                                  key={event.id}
                                  type="button"
                                  className={`waveform-event-chip caption-event-badge event-${event.kind}`}
                                  title={`${formatClock(event.time)} ${event.kind === "overlap" ? "Possible overlap" : "Tight handoff"} | ${event.label} — click to listen; fix with the speaker pill`}
                                  onClick={() => seekAudio(Math.max(0, event.start - 1), { play: true })}
                                >
                                  {event.kind === "overlap" ? <AlertTriangle size={11} aria-hidden /> : <ArrowLeftRight size={11} aria-hidden />}
                                </button>
                              ))}
                              {simultaneousCaptionIds.has(caption.id) ? (
                                <span
                                  className="caption-simultaneous-badge"
                                  title="Plays at the same time as another speaker's caption. Exports as one dialogue subtitle with a line per speaker."
                                >
                                  <AudioLines size={11} aria-hidden />
                                </span>
                              ) : null}
                            </div>
                            <TimedTextEditor
                              className="subtitle-editor"
                              minHeight={1}
                              value={captionValue(caption)}
                              wordIds={caption.word_ids}
                              lookup={transcriptWords}
                              currentTime={currentTime}
                              showTimingHighlights={showTimingHighlights}
                              lowConfidenceThreshold={lowConfidenceThreshold}
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
                                if (event.key === "Enter" && event.shiftKey && target.value.includes("\n")) {
                                  // Captions are capped at two lines.
                                  event.preventDefault();
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
                            </div>
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
                  {SIDE_PANEL_TABS.map((tab) => {
                    const TabIcon = SIDE_PANEL_TAB_ICONS[tab.id];
                    return (
                      <button
                        key={tab.id}
                        className={sidePanelTab === tab.id ? "is-active" : ""}
                        onClick={() => setSidePanelTab(tab.id)}
                      >
                        <span className="panel-tab-label">
                          <TabIcon size={14} aria-hidden />
                          {tab.label}
                          {tab.id === "qa" && qaReport.summary.issueCount > 0 ? (
                            <span className="tab-badge">{qaReport.summary.issueCount}</span>
                          ) : null}
                        </span>
                      </button>
                    );
                  })}
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

                    {reviewableSpeakerEvents.length ? (
                      <section className="selection-panel">
                        <div className="panel-section-heading">
                          <p className="eyebrow">Speaker events</p>
                          <h3>{reviewableSpeakerEvents.length} to review</h3>
                        </div>
                        <div className="speaker-events-list">
                          {reviewableSpeakerEvents.map((event) => (
                            <button
                              key={event.id}
                              type="button"
                              className={`waveform-event-chip event-${event.kind}`}
                              onClick={() => focusSpeakerEvent(event)}
                            >
                              {formatClock(event.time)} {event.kind === "overlap" ? "Overlap" : "Tight handoff"} | {event.label}
                            </button>
                          ))}
                        </div>
                        <p className="helper-text">
                          Tight handoffs and overlaps are where diarization mislabels speakers most often. Click one to
                          jump there, listen, and fix mistakes with the speaker pill on the caption.
                        </p>
                      </section>
                    ) : null}

                    <section className="selection-panel">
                      <div className="panel-section-heading">
                        <p className="eyebrow">Editing</p>
                        <h3>Tools</h3>
                      </div>
                      <div className="inline-actions">
                        <button onClick={undo} disabled={!history.past.length}><Undo2 size={14} aria-hidden />&nbsp;Undo</button>
                        <button onClick={redo} disabled={!history.future.length}><Redo2 size={14} aria-hidden />&nbsp;Redo</button>
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
                      <p className="eyebrow">Vocabulary</p>
                      <h3>
                        Words that transcribe wrong
                        <InfoTip text="A persistent list of names, technical terms, and acronyms Whisper tends to mishear or misspell (e.g. a name spelled 'Zakery'). It biases the next transcription toward these spellings, flags near-miss captions in QA, and seeds jargon-only retranscribe. It carries across every file, so add a term once and future recordings come out right." />
                      </h3>
                    </div>
                    <label>
                      Project vocabulary
                      <textarea
                        rows={6}
                        placeholder={"One term per line, e.g.\nZakery\nWhisperX\nKubernetes"}
                        value={glossaryText}
                        onChange={(event) => setGlossaryText(event.target.value)}
                      />
                    </label>
                    <p className="helper-text">
                      {glossaryTerms.length
                        ? `${glossaryTerms.length} term${glossaryTerms.length === 1 ? "" : "s"} active. ${glossaryMatchedCaptionCount} caption${glossaryMatchedCaptionCount === 1 ? "" : "s"} currently match.`
                        : "Add names and technical terms before transcribing so they come out right the first time. The list below suggests unusual words found in this transcript."}
                    </p>
                    <div className="inline-actions">
                      <button disabled={!jargonCandidates.length} onClick={() => addTermsToGlossary(jargonCandidates.slice(0, 12).map((candidate) => candidate.display))}>
                        Add top suggestions
                      </button>
                      <button
                        disabled={!selectedFile || retranscribing || !glossaryTerms.length || !glossaryMatchedCaptionCount}
                        onClick={() => void handleRetranscribeGlossaryMatches()}
                      >
                        {retranscribing ? "Retranscribing..." : "Retranscribe Vocabulary Matches"}
                      </button>
                    </div>
                    {jargonCandidates.length ? (
                      <>
                        <p className="helper-text">Suggested from this transcript — add the ones that are real vocabulary:</p>
                        <div className="qa-list">
                          {jargonCandidates.map((candidate) => (
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
                      </>
                    ) : (
                      <p className="helper-text">No unusual words detected in this transcript. Add names and terms manually above.</p>
                    )}
                  </section>
                ) : null}

                {sidePanelTab === "overlaps" ? (
                  <OverlapsPanel
                    apiBaseUrl={API_BASE_URL}
                    audioFile={selectedFile}
                    regions={overlapRegions}
                    turns={speakerTurns}
                    speakers={activeEditor?.speakers ?? speakerInputs}
                    language={activeWorkspace?.language ?? session?.language ?? null}
                    restoreSoloTracks={restoreSoloTracks}
                    onRestoreSoloTracksChange={setRestoreSoloTracks}
                    onProcessed={handleSeparationProcessed}
                    onApplyWords={applySeparatedWords}
                    onSeek={seekAudio}
                  />
                ) : null}

                {sidePanelTab === "regions" ? (
                  <RegionsPanel
                    speakers={activeEditor?.speakers ?? speakerInputs}
                    soloableSpeakerIds={soloableSpeakers.map((speaker) => speaker.id)}
                    regions={editableRegions}
                    overrideActive={regionOverrides !== null}
                    tracksStale={soloTracksStale}
                    tracksRendering={soloTracks.status === "running"}
                    frames={waveformAnalysis?.frames ?? []}
                    speechSpans={waveformAnalysis?.speech_spans ?? []}
                    overlapRegions={overlapRegions}
                    markers={regionMarkers}
                    duration={waveformAnalysis?.duration ?? session?.duration ?? audioDuration}
                    currentTime={currentTime}
                    soloSpeakerId={soloSpeakerId}
                    theme={themeDark ? "dark" : "light"}
                    onChange={handleRegionsChange}
                    onReset={handleRegionsReset}
                    onRerenderTracks={() => setTrackRenderNonce((nonce) => nonce + 1)}
                    onSoloSpeakerChange={handleSoloSpeakerChange}
                    onSeek={seekAudio}
                  />
                ) : null}

                {sidePanelTab === "restore" ? (
                  <RestorePanel apiBaseUrl={API_BASE_URL} audioFile={selectedFile} />
                ) : null}

                {sidePanelTab === "convert" ? (
                  <ConvertPanel
                    apiBaseUrl={API_BASE_URL}
                    audioFile={selectedFile}
                    isolatedTracks={isolatedTrackChoices}
                    onConvertedVoice={handleConvertedVoice}
                    onConvertedVoiceRemoved={handleConvertedVoiceRemoved}
                  />
                ) : null}

                {sidePanelTab === "patch" ? (
                  <PatchPanel apiBaseUrl={API_BASE_URL} audioFile={selectedFile} />
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

                {sidePanelTab === "export" ? (
                  <section className="selection-panel">
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

                    <div className="panel-section-heading">
                      <p className="eyebrow">Speaker audio</p>
                      <h3>Isolated tracks</h3>
                    </div>
                    {speakerAudioExports.some((entry) => entry.isolated || entry.converted) ? (
                      <>
                        <p className="helper-text">
                          Full-length, timeline-preserving audio: each track carries one voice and silence elsewhere.
                        </p>
                        <div className="inline-actions">
                          {speakerAudioExports.map((entry) => (
                            <Fragment key={entry.speakerId}>
                              {entry.isolated ? (
                                <a className="button-link" href={entry.isolated.url} download={entry.isolated.download}>
                                  {entry.name} — isolated
                                </a>
                              ) : null}
                              {entry.converted ? (
                                <a className="button-link" href={entry.converted.url} download={entry.converted.download}>
                                  {entry.name} — converted
                                </a>
                              ) : null}
                            </Fragment>
                          ))}
                        </div>
                      </>
                    ) : (
                      <p className="helper-text">
                        No speaker tracks yet. They render automatically once the audio has speakers; a converted voice
                        shows up here after a conversion in the Convert tab.
                      </p>
                    )}
                  </section>
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
                <p className="helper-text guide-collapsed-copy">Guide, glossary, QA, mastering, and export stay tucked away until you need them.</p>
                <div className="guide-collapsed-metrics">
                  <span className="metric-chip">{sidePanelMeta.detail}</span>
                  <span className="metric-chip">{collapsedPanelMetaDetail}</span>
                </div>
                <div className="guide-collapsed-tabs">
                  {SIDE_PANEL_TABS.map((tab) => {
                    const TabIcon = SIDE_PANEL_TAB_ICONS[tab.id];
                    return (
                      <button
                        key={tab.id}
                        type="button"
                        className={`icon-button ${sidePanelTab === tab.id ? "is-active" : ""}`}
                        title={tab.label}
                        aria-label={tab.label}
                        onClick={() => {
                          setSidePanelTab(tab.id);
                          setIsGuidePanelCollapsed(false);
                        }}
                      >
                        <TabIcon size={16} aria-hidden />
                        {tab.id === "qa" && qaReport.summary.issueCount > 0 ? (
                          <span className="tab-badge">{qaReport.summary.issueCount}</span>
                        ) : null}
                      </button>
                    );
                  })}
                </div>
              </div>
            )}
          </aside>
        </div>
      </main>

      {globalDragActive ? (
        <div className="drop-overlay">
          <div className="drop-overlay-card">Drop the audio or video file to load it</div>
        </div>
      ) : null}

      {toasts.length ? (
        <div className="toast-stack" role="status">
          {toasts.map((toast) => (
            <div key={toast.id} className="toast">
              <span>{toast.text}</span>
              <button className="toast-dismiss" aria-label="Dismiss" onClick={() => dismissToast(toast.id)}>
                <X size={14} />
              </button>
            </div>
          ))}
        </div>
      ) : null}
    </div>
  );
}

export default App;

