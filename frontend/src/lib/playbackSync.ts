/**
 * Transport decisions for the multi-element player.
 *
 * Every track is mounted at once -- the original, the mastered A/B peer, one per
 * rendered speaker, and one per converted voice -- and they all play together;
 * only the mute flags move. Muting a speaker or switching it to its converted
 * voice must therefore be a mute swap and nothing else: unmounting an element,
 * seeking one that is audible, or automating its volume is what made
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

/** Which of a speaker's two tracks the listener asked for. */
export type SpeakerVoice = "original" | "converted";

/** One speaker's contribution to the audibility decision. */
export interface SpeakerTrackInput {
  speakerId: number;
  muted: boolean;
  /** The isolated (server-gated) track of this speaker's own voice. */
  track: SoloTrackState;
  /** Which voice is selected; absent means the original. */
  voice?: SpeakerVoice;
  /**
   * The converted re-voicing of `track`, classified the same way. Absent means
   * nothing has been converted for this speaker.
   */
  converted?: SoloTrackState;
}

/** One audible speaker and which of its tracks is carrying the sound. */
export interface AudibleTrackChoice {
  speakerId: number;
  voice: SpeakerVoice;
}

export interface AudibleSet {
  clock: ClockSource;
  /** The clock element carries the sound. */
  clockAudible: boolean;
  /** Speakers whose rendered track carries the sound, all at once. */
  audibleTracks: AudibleTrackChoice[];
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
 * Which of a speaker's tracks would actually carry the sound. A converted voice
 * is only used once it can play: falling back to the speaker's own isolated
 * track keeps that voice present (at the original timbre) instead of dropping it
 * or handing the whole mix back to the clock while the artifact loads.
 */
function chosenVoice(speaker: SpeakerTrackInput): { voice: SpeakerVoice; state: SoloTrackState } {
  if (speaker.voice === "converted" && speaker.converted === "ready") {
    return { voice: "converted", state: "ready" };
  }
  return { voice: "original", state: speaker.track };
}

/**
 * Which of {clock, per-speaker tracks} are audible for a given muted set and
 * per-speaker voice choice.
 *
 * Nothing muted and everyone on their own voice is the plain path: the clock
 * alone, ungated. Anything else -- a muted speaker, or a speaker switched to a
 * converted voice -- silences the clock and plays every unmuted speaker's chosen
 * track at once. Those samples are already speaker-gated server-side, and a
 * conversion of such a track inherits that gating, so their sum is the mix with
 * the muted voices dropped and the converted ones substituted, with no
 * client-side gain anywhere. Toggling then changes mute flags and nothing else,
 * which is what makes it jitter-free.
 *
 * Two fallbacks keep sound flowing instead of guessing:
 *   - any needed track missing/pending (still rendering, still loading, stalled)
 *     -> the clock stands in, gated to the union of the unmuted regions;
 *   - a cut-timeline master as the clock -> tracks live on the original
 *     timeline and cannot follow it, so the gated clock is the only option.
 *
 * Every speaker muted is silence, deliberately: that is what "mute everyone"
 * asks for -- a converted voice on a muted speaker stays silent too.
 */
export function chooseAudibleSet(input: {
  playbackSource: "original" | "processed";
  hasMastered: boolean;
  masterHasCutTimeline: boolean;
  speakers: SpeakerTrackInput[];
}): AudibleSet {
  const clock = chooseClockSource(input.playbackSource, input.hasMastered);
  const unmuted = input.speakers.filter((speaker) => !speaker.muted);
  // A single speaker on a converted voice makes the track-sum the only mix that
  // can express the request, however plain the mute state is.
  const anyConverted = input.speakers.some((speaker) => speaker.voice === "converted");
  if (unmuted.length === input.speakers.length && !anyConverted) {
    return { clock, clockAudible: true, audibleTracks: [], gateClock: false };
  }
  if (!unmuted.length) {
    return { clock, clockAudible: false, audibleTracks: [], gateClock: false };
  }
  const chosen = unmuted.map((speaker) => ({ speakerId: speaker.speakerId, ...chosenVoice(speaker) }));
  const tracksUsable =
    followersShareClockTimeline(clock, input.masterHasCutTimeline) &&
    chosen.every((entry) => entry.state === "ready");
  if (!tracksUsable) {
    return { clock, clockAudible: true, audibleTracks: [], gateClock: true };
  }
  return {
    clock,
    clockAudible: false,
    audibleTracks: chosen.map((entry) => ({ speakerId: entry.speakerId, voice: entry.voice })),
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
