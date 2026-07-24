import type { SpeakerRegion, SpeechSpan } from "../types";

export interface RegionInterval {
  start: number;
  end: number;
}

export type RegionEdge = "start" | "end";

/** A tick on the region strip: a handoff, a switch, or a detected overlap. */
export interface RegionMarker {
  id: string;
  time: number;
  kind: string;
  label: string;
}

/** One VAD frame. Anything shorter than this cannot be placed against the audio. */
export const MIN_REGION_S = 0.02;

// Snap tolerances, asymmetric like findStartSpeechSpan/findEndSpeechSpan in
// App.tsx: growing a region outward to swallow a whole speech span is always
// safe, pulling its edge inward clips the voice it is supposed to carry. So a
// candidate that opens the region is accepted from further away than one that
// closes it.
export const SNAP_OUTWARD_S = 0.2;
export const SNAP_INWARD_S = 0.08;

// Ids only have to stay stable while one list is being edited — they key the
// React rows and the selection, and they survive a merge (see normalizeRegions).
// Nothing outside the editor reads them.
const REGION_ID_PREFIX = "region-";

export function regionId(sequence: number): string {
  return `${REGION_ID_PREFIX}${sequence}`;
}

function nextIdSequence(regions: SpeakerRegion[]): number {
  let highest = 0;
  for (const region of regions) {
    if (!region.id.startsWith(REGION_ID_PREFIX)) {
      continue;
    }
    const parsed = Number(region.id.slice(REGION_ID_PREFIX.length));
    if (Number.isFinite(parsed) && parsed > highest) {
      highest = parsed;
    }
  }
  return highest + 1;
}

function roundMs(value: number): number {
  return Math.round(value * 1000) / 1000;
}

function clamp(value: number, min: number, max: number): number {
  return Math.min(max, Math.max(min, value));
}

function upperBound(duration: number | null): number {
  return duration !== null && Number.isFinite(duration) && duration > 0 ? duration : Number.POSITIVE_INFINITY;
}

/**
 * Clamp to [0, duration], drop degenerate regions, then per speaker sort and
 * merge everything that touches or overlaps. Regions of DIFFERENT speakers are
 * left alone — that is how the user expresses genuine simultaneous speech — but
 * same-speaker regions must come out sorted and disjoint because the playback
 * gate (intervalIndexAt) binary-searches them.
 *
 * `preferId` survives a merge, so the region the user is dragging keeps its
 * identity when it swallows a sibling and the selection does not vanish.
 */
export function normalizeRegions(
  regions: SpeakerRegion[],
  duration: number | null,
  preferId?: string,
): SpeakerRegion[] {
  const limit = upperBound(duration);
  const clamped = regions
    .filter(
      (region) =>
        Number.isFinite(region.start) && Number.isFinite(region.end) && Number.isFinite(region.speaker_id),
    )
    .map((region) => {
      const low = roundMs(clamp(Math.min(region.start, region.end), 0, limit));
      const high = roundMs(clamp(Math.max(region.start, region.end), 0, limit));
      return { ...region, start: low, end: high };
    })
    .filter((region) => region.end - region.start >= MIN_REGION_S);

  const bySpeaker = new Map<number, SpeakerRegion[]>();
  for (const region of clamped) {
    const bucket = bySpeaker.get(region.speaker_id);
    if (bucket) {
      bucket.push(region);
    } else {
      bySpeaker.set(region.speaker_id, [region]);
    }
  }

  const merged: SpeakerRegion[] = [];
  for (const bucket of bySpeaker.values()) {
    bucket.sort((left, right) => left.start - right.start || left.end - right.end);
    for (const region of bucket) {
      const previous = merged[merged.length - 1];
      if (previous && previous.speaker_id === region.speaker_id && region.start <= previous.end) {
        previous.end = Math.max(previous.end, region.end);
        if (region.id === preferId) {
          previous.id = region.id;
        }
        continue;
      }
      merged.push({ ...region });
    }
  }

  return merged.sort((left, right) => left.start - right.start || left.speaker_id - right.speaker_id);
}

/**
 * Flatten the derived `Map<speakerId, intervals>` into the editable list. This
 * is the "materialize" step: the first manual edit turns the audio-derived
 * regions into an explicit override that is from then on the only truth.
 */
export function materializeRegions(derived: Map<number, RegionInterval[]> | null): SpeakerRegion[] {
  if (!derived) {
    return [];
  }
  const flat: Array<{ start: number; end: number; speaker_id: number }> = [];
  for (const [speakerId, intervals] of derived) {
    for (const interval of intervals) {
      flat.push({ start: roundMs(interval.start), end: roundMs(interval.end), speaker_id: speakerId });
    }
  }
  flat.sort((left, right) => left.start - right.start || left.speaker_id - right.speaker_id);
  return flat.map((entry, index) => ({ id: regionId(index + 1), ...entry }));
}

/** Group an override list back into the shape the gate and the export expect. */
export function regionsToSpeakerMap(regions: SpeakerRegion[]): Map<number, RegionInterval[]> {
  const bySpeaker = new Map<number, RegionInterval[]>();
  for (const region of normalizeRegions(regions, null)) {
    const bucket = bySpeaker.get(region.speaker_id);
    const interval = { start: region.start, end: region.end };
    if (bucket) {
      bucket.push(interval);
    } else {
      bySpeaker.set(region.speaker_id, [interval]);
    }
  }
  return bySpeaker;
}

/** Every edge belonging to some OTHER speaker — snap targets for a handoff. */
export function collectForeignEdges(regions: SpeakerRegion[], speakerId: number): number[] {
  const edges: number[] = [];
  for (const region of regions) {
    if (region.speaker_id === speakerId) {
      continue;
    }
    edges.push(region.start, region.end);
  }
  return edges;
}

/**
 * Nearest snap target for a dragged edge. Speech-span edges are the primary
 * reference: they are measured at 20 ms from the un-downsampled audio, unlike
 * the display frames, so they are the only thing in the UI a boundary can be
 * placed against precisely.
 */
export function snapRegionEdge(
  time: number,
  edge: RegionEdge,
  speechSpans: SpeechSpan[],
  foreignEdges: number[] = [],
): number {
  const earlierLimit = edge === "start" ? SNAP_OUTWARD_S : SNAP_INWARD_S;
  const laterLimit = edge === "start" ? SNAP_INWARD_S : SNAP_OUTWARD_S;
  let best: number | null = null;
  let bestDistance = Number.POSITIVE_INFINITY;

  const consider = (candidate: number) => {
    const delta = candidate - time;
    if (delta < -earlierLimit || delta > laterLimit) {
      return;
    }
    const distance = Math.abs(delta);
    if (distance < bestDistance) {
      bestDistance = distance;
      best = candidate;
    }
  };

  for (const span of speechSpans) {
    // A region start belongs at a speech onset, an end at a speech offset;
    // snapping a start to an offset would open it over the previous voice.
    consider(edge === "start" ? span.start : span.end);
  }
  for (const candidate of foreignEdges) {
    consider(candidate);
  }
  return best === null ? time : roundMs(best);
}

export interface EdgeMoveOptions {
  duration: number | null;
  speechSpans?: SpeechSpan[];
  /** Alt-drag turns snapping off for the odd boundary that has to sit in speech. */
  snap?: boolean;
}

export function moveRegionEdge(
  regions: SpeakerRegion[],
  id: string,
  edge: RegionEdge,
  time: number,
  options: EdgeMoveOptions,
): SpeakerRegion[] {
  const target = regions.find((region) => region.id === id);
  if (!target) {
    return regions;
  }
  const limit = upperBound(options.duration);
  const requested = clamp(time, 0, limit);
  const snapped =
    options.snap === false
      ? requested
      : snapRegionEdge(
          requested,
          edge,
          options.speechSpans ?? [],
          collectForeignEdges(regions, target.speaker_id),
        );

  const next = regions.map((region) => {
    if (region.id !== id) {
      return region;
    }
    // The dragged edge may not cross its own partner: past it the region would
    // invert, and a zero-width region is not something the user can grab back.
    if (edge === "start") {
      return { ...region, start: Math.min(snapped, region.end - MIN_REGION_S) };
    }
    return { ...region, end: Math.max(snapped, region.start + MIN_REGION_S) };
  });
  return normalizeRegions(next, options.duration, id);
}

/** Split at `time`; the two halves' union is exactly the original region. */
export function splitRegionAt(
  regions: SpeakerRegion[],
  id: string,
  time: number,
  duration: number | null,
): SpeakerRegion[] {
  const target = regions.find((region) => region.id === id);
  if (!target || time <= target.start + MIN_REGION_S || time >= target.end - MIN_REGION_S) {
    return regions;
  }
  const cut = roundMs(time);
  const rightId = regionId(nextIdSequence(regions));
  const next: SpeakerRegion[] = [];
  for (const region of regions) {
    if (region.id !== id) {
      next.push(region);
      continue;
    }
    next.push({ ...region, end: cut });
    next.push({ ...region, id: rightId, start: cut });
  }
  // Deliberately not normalized: the halves touch at `cut`, and merging them
  // straight back would undo the split.
  return next.sort((left, right) => left.start - right.start || left.speaker_id - right.speaker_id);
}

export function deleteRegion(regions: SpeakerRegion[], id: string): SpeakerRegion[] {
  return regions.filter((region) => region.id !== id);
}

export interface AddRegionResult {
  regions: SpeakerRegion[];
  id: string;
}

export function addRegion(
  regions: SpeakerRegion[],
  speakerId: number,
  start: number,
  end: number,
  duration: number | null,
): AddRegionResult {
  const id = regionId(nextIdSequence(regions));
  const added: SpeakerRegion = {
    id,
    start: Math.min(start, end),
    end: Math.max(start, end),
    speaker_id: speakerId,
  };
  return { regions: normalizeRegions([...regions, added], duration, id), id };
}

export function reassignRegion(
  regions: SpeakerRegion[],
  id: string,
  speakerId: number,
  duration: number | null,
): SpeakerRegion[] {
  const next = regions.map((region) => (region.id === id ? { ...region, speaker_id: speakerId } : region));
  return normalizeRegions(next, duration, id);
}

export interface RegionStat {
  count: number;
  seconds: number;
}

export function regionStats(regions: SpeakerRegion[]): Map<number, RegionStat> {
  const stats = new Map<number, RegionStat>();
  for (const region of regions) {
    const current = stats.get(region.speaker_id) ?? { count: 0, seconds: 0 };
    current.count += 1;
    current.seconds += Math.max(0, region.end - region.start);
    stats.set(region.speaker_id, current);
  }
  return stats;
}

/**
 * Spans where this speaker's regions collide with another speaker's. Allowed and
 * meaningful — it is how a genuine overlap is expressed — so the lanes tint it
 * rather than the editor preventing it.
 */
export function crossSpeakerOverlaps(regions: SpeakerRegion[], speakerId: number): RegionInterval[] {
  const mine = regions.filter((region) => region.speaker_id === speakerId);
  const theirs = regions.filter((region) => region.speaker_id !== speakerId);
  const spans: RegionInterval[] = [];
  for (const region of mine) {
    for (const other of theirs) {
      const start = Math.max(region.start, other.start);
      const end = Math.min(region.end, other.end);
      if (end > start) {
        spans.push({ start, end });
      }
    }
  }
  return spans.sort((left, right) => left.start - right.start);
}

/**
 * Restore guard for persisted overrides: keep every well-formed entry, drop the
 * rest rather than throwing, and treat anything that is not an array as "no
 * override" so pre-editor saves keep loading. An array that survives empty stays
 * an override — the user may have deliberately cleared every region.
 */
export function sanitizeSpeakerRegions(value: unknown): SpeakerRegion[] | null {
  if (!Array.isArray(value)) {
    return null;
  }
  const kept: SpeakerRegion[] = [];
  for (const entry of value) {
    if (!entry || typeof entry !== "object") {
      continue;
    }
    const candidate = entry as Partial<SpeakerRegion>;
    if (typeof candidate.id !== "string" || !candidate.id) {
      continue;
    }
    if (typeof candidate.start !== "number" || !Number.isFinite(candidate.start) || candidate.start < 0) {
      continue;
    }
    if (typeof candidate.end !== "number" || !Number.isFinite(candidate.end)) {
      continue;
    }
    if (typeof candidate.speaker_id !== "number" || !Number.isInteger(candidate.speaker_id)) {
      continue;
    }
    if (candidate.end - candidate.start < MIN_REGION_S) {
      continue;
    }
    kept.push({
      id: candidate.id,
      start: candidate.start,
      end: candidate.end,
      speaker_id: candidate.speaker_id,
    });
  }
  return normalizeRegions(kept, null);
}

/**
 * First index whose interval could still reach `time`. Requires sorted, disjoint
 * intervals (ends rise with starts), which is exactly what normalizeRegions and
 * the VAD speech spans guarantee.
 */
export function firstIntervalIndexAtOrAfter(items: RegionInterval[], time: number): number {
  let low = 0;
  let high = items.length - 1;
  let found = items.length;
  while (low <= high) {
    const mid = (low + high) >> 1;
    if (items[mid].end >= time) {
      found = mid;
      high = mid - 1;
    } else {
      low = mid + 1;
    }
  }
  return found;
}

/**
 * The intervals visible in [from, to]. Every canvas pass in the editor goes
 * through this: a 19-minute file holds ~700 speech spans and the panel redraws
 * on every pointer move, so nothing outside the window may be touched.
 */
export function sliceIntervals<T extends RegionInterval>(items: T[], from: number, to: number): T[] {
  const visible: T[] = [];
  for (let index = firstIntervalIndexAtOrAfter(items, from); index < items.length; index += 1) {
    if (items[index].start > to) {
      break;
    }
    visible.push(items[index]);
  }
  return visible;
}

/** Same idea for the display frames, which carry a single timestamp. */
export function firstFrameIndexAtOrAfter(frames: Array<{ time: number }>, time: number): number {
  let low = 0;
  let high = frames.length - 1;
  let found = frames.length;
  while (low <= high) {
    const mid = (low + high) >> 1;
    if (frames[mid].time >= time) {
      found = mid;
      high = mid - 1;
    } else {
      low = mid + 1;
    }
  }
  return found;
}

export interface ViewWindow {
  start: number;
  seconds: number;
}

/** Zoom around an anchor time, keeping the anchor under the cursor. */
export function zoomViewWindow(
  view: ViewWindow,
  factor: number,
  anchorTime: number,
  duration: number,
  minSeconds: number,
): ViewWindow {
  const total = Math.max(minSeconds, duration);
  const seconds = clamp(view.seconds * factor, minSeconds, total);
  const anchorFraction = view.seconds > 0 ? clamp((anchorTime - view.start) / view.seconds, 0, 1) : 0.5;
  const start = clamp(anchorTime - anchorFraction * seconds, 0, Math.max(0, total - seconds));
  return { start: roundMs(start), seconds };
}

export function panViewWindow(view: ViewWindow, deltaSeconds: number, duration: number): ViewWindow {
  const total = Math.max(view.seconds, duration);
  return {
    start: roundMs(clamp(view.start + deltaSeconds, 0, Math.max(0, total - view.seconds))),
    seconds: view.seconds,
  };
}
