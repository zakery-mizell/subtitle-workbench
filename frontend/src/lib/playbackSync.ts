/**
 * Transport decisions for the multi-element player.
 *
 * Every track is mounted at once -- the original, the mastered A/B peer, and one
 * per rendered speaker -- and they all play together; only the mute flags move.
 * Muting a speaker must therefore be a mute swap and nothing else: unmounting an
 * element, seeking one that is audible, or automating its volume is what made
 * hot-swapping jump and jitter.
 *
 * Two constraints run through everything here:
 *   - Rendered per-speaker tracks are already gain-gated server-side (silence
 *     outside that speaker's regions, separated voice inside overlaps, 60 ms
 *     equal-power edge fades). They must never be re-gated client-side: the
 *     element runs tens of milliseconds off the clock that would drive such a
 *     gate, so every region edge would be clipped or double-faded out of phase.
 *   - Nothing audible is ever seeked. Drift correction applies to muted
 *     elements only, which is why they are all held on the clock's position
 *     while silent.
 *
 * These are the decisions that swap makes, kept free of media elements so they
 * can be exercised without a DOM.
 */

/** `HTMLMediaElement.HAVE_METADATA`: duration known, so a seek will be honoured. */
export const MEDIA_HAVE_METADATA = 1;
/** `HTMLMediaElement.HAVE_CURRENT_DATA`: there is audio at the playhead to hear. */
export const MEDIA_HAVE_CURRENT_DATA = 2;

/**
 * How far a muted follower may drift from the clock before it is nudged.
 * Tight on purpose: a follower is corrected only while muted, so this is also
 * the worst-case offset it carries into being unmuted -- and from that moment on
 * it is never seeked again.
 */
export const PEER_SYNC_TOLERANCE_S = 0.04;

/**
 * How far an audible track may drift before it is treated as stalled. Far wider
 * than the follower tolerance because an audible element is never corrected: its
 * offset is allowed to grow, and only a genuine stall (buffer underrun, tab
 * throttling) should hand the sound back to the gated clock.
 */
export const AUDIBLE_STALL_TOLERANCE_S = 0.25;

/** Which element's `currentTime` is authoritative for the whole app. */
export type ClockSource = "original" | "mastered";

/**
 * Whether a speaker's rendered track can carry the sound right now.
 * `pending` covers "still loading", "loaded but not yet time-locked" and
 * "audible but stalled": in every case the clock takes over (gated) rather than
 * the mix going silent.
 */
export type SoloTrackState = "ready" | "pending" | "missing";

export interface TimeInterval {
  start: number;
  end: number;
}

/** One speaker's contribution to the audibility decision. */
export interface SpeakerTrackInput {
  speakerId: number;
  muted: boolean;
  track: SoloTrackState;
}

export interface AudibleSet {
  clock: ClockSource;
  /** The clock element carries the sound. */
  clockAudible: boolean;
  /** Speakers whose rendered track carries the sound, all at once. */
  audibleSpeakerIds: number[];
  /**
   * The clock is standing in for tracks that are not usable yet, so it must be
   * gated to the union of the unmuted speakers' regions. The only case in which
   * frontend gain automation is correct.
   */
  gateClock: boolean;
}

/**
 * The clock is the A/B selection alone. It is deliberately independent of which
 * speakers are muted: `audioRef` points here, and the transcript highlight,
 * waveform playhead, skip-cuts logic and the fallback gate all read it, so
 * repointing it on a mute toggle would resynchronise the entire UI mid-playback.
 */
export function chooseClockSource(playbackSource: "original" | "processed", hasMastered: boolean): ClockSource {
  return playbackSource === "processed" && hasMastered ? "mastered" : "original";
}

/**
 * Speaker tracks are rendered on the original timeline. A mastered file with
 * cuts has its own shorter timeline, so while that is the clock no follower can
 * track it -- the same position means two different moments.
 */
export function followersShareClockTimeline(clock: ClockSource, masterHasCutTimeline: boolean): boolean {
  return clock === "original" || !masterHasCutTimeline;
}

/**
 * Classify one speaker's track. Readiness needs both loaded audio and a
 * position already locked to the clock, because unmuting never seeks.
 *
 * `alreadyAudible` is hysteresis, not an optimisation: an audible element is
 * never drift-corrected, so its offset grows freely. Without this it would be
 * demoted for drift, corrected while muted, promoted again, and oscillate. The
 * stall tolerance bounds that hysteresis so a track that has actually died still
 * hands the sound back.
 *
 * The readyState test comes first so that hysteresis cannot trap a broken track:
 * one that stalls loses its buffered data, falls back to the clock, gets
 * corrected while muted, and takes over again on its own.
 */
export function classifySoloTrack(input: {
  hasTrack: boolean;
  readyState: number;
  offsetFromClock: number;
  alreadyAudible: boolean;
  toleranceSeconds?: number;
  stallToleranceSeconds?: number;
}): SoloTrackState {
  if (!input.hasTrack) {
    return "missing";
  }
  if (input.readyState < MEDIA_HAVE_CURRENT_DATA) {
    return "pending";
  }
  if (input.alreadyAudible) {
    const stallTolerance = input.stallToleranceSeconds ?? AUDIBLE_STALL_TOLERANCE_S;
    return Math.abs(input.offsetFromClock) > stallTolerance ? "pending" : "ready";
  }
  const tolerance = input.toleranceSeconds ?? PEER_SYNC_TOLERANCE_S;
  return Math.abs(input.offsetFromClock) > tolerance ? "pending" : "ready";
}

/**
 * Which of {clock, per-speaker tracks} are audible for a given muted set.
 *
 * Nothing muted is the plain path: the clock alone, ungated. With anything
 * muted the clock goes silent and every unmuted speaker's track plays at once
 * -- their samples are already speaker-gated server-side, so their sum is the
 * mix minus the muted voices, with no client-side gain anywhere. Toggling a
 * speaker then changes `muted` flags and nothing else, which is what makes it
 * jitter-free.
 *
 * Two fallbacks keep sound flowing instead of guessing:
 *   - any needed track missing/pending (still rendering, still loading, stalled)
 *     -> the clock stands in, gated to the union of the unmuted regions;
 *   - a cut-timeline master as the clock -> tracks live on the original
 *     timeline and cannot follow it, so the gated clock is the only option.
 *
 * Every speaker muted is silence, deliberately: that is what "mute everyone"
 * asks for.
 */
export function chooseAudibleSet(input: {
  playbackSource: "original" | "processed";
  hasMastered: boolean;
  masterHasCutTimeline: boolean;
  speakers: SpeakerTrackInput[];
}): AudibleSet {
  const clock = chooseClockSource(input.playbackSource, input.hasMastered);
  const unmuted = input.speakers.filter((speaker) => !speaker.muted);
  if (unmuted.length === input.speakers.length) {
    return { clock, clockAudible: true, audibleSpeakerIds: [], gateClock: false };
  }
  if (!unmuted.length) {
    return { clock, clockAudible: false, audibleSpeakerIds: [], gateClock: false };
  }
  const tracksUsable =
    followersShareClockTimeline(clock, input.masterHasCutTimeline) &&
    unmuted.every((speaker) => speaker.track === "ready");
  if (!tracksUsable) {
    return { clock, clockAudible: true, audibleSpeakerIds: [], gateClock: true };
  }
  return {
    clock,
    clockAudible: false,
    audibleSpeakerIds: unmuted.map((speaker) => speaker.speakerId),
    gateClock: false,
  };
}

/**
 * Drift correction, only ever on silent elements. Correcting the element being
 * listened to is precisely the glitch this design removes, so audibility vetoes
 * the correction outright rather than being weighed against the offset.
 */
export function shouldCorrectFollower(input: {
  clockTime: number;
  followerTime: number;
  isAudible: boolean;
  toleranceSeconds?: number;
}): boolean {
  if (input.isAudible) {
    return false;
  }
  const tolerance = input.toleranceSeconds ?? PEER_SYNC_TOLERANCE_S;
  return Math.abs(input.followerTime - input.clockTime) > tolerance;
}

/**
 * Union of per-speaker region lists, for the fallback gate. The gain lookup
 * binary-searches this, so the result has to be sorted and disjoint: abutting
 * and overlapping spans are merged rather than left adjacent.
 */
export function unionIntervals(lists: TimeInterval[][]): TimeInterval[] {
  const spans = lists
    .flat()
    .filter((span) => span.end > span.start)
    .sort((left, right) => left.start - right.start);
  const merged: TimeInterval[] = [];
  for (const span of spans) {
    const previous = merged[merged.length - 1];
    if (previous && span.start <= previous.end) {
      previous.end = Math.max(previous.end, span.end);
    } else {
      merged.push({ start: span.start, end: span.end });
    }
  }
  return merged;
}
