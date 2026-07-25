/**
 * Transport decisions for the multi-element player.
 *
 * Every track is mounted at once -- the original, the mastered A/B peer, and one
 * per rendered solo speaker -- and they all play together; exactly one is
 * audible. Switching the "Only <name>" selector therefore has to be a mute swap
 * and nothing else: unmounting an element, or seeking the one being listened to,
 * is what made hot-swapping jump by tens of seconds.
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
 * Well under a frame at any rate anyone plays back at, so a follower that is
 * within tolerance can be made audible without an audible step.
 */
export const PEER_SYNC_TOLERANCE_S = 0.08;

/** Which element's `currentTime` is authoritative for the whole app. */
export type ClockSource = "original" | "mastered";

/**
 * Whether the soloed speaker's track can carry the sound right now.
 * `pending` covers both "still loading" and "loaded but not yet time-locked":
 * in either case the clock stays audible (gated) rather than going silent.
 */
export type SoloTrackState = "ready" | "pending" | "missing";

export interface AudibleTarget {
  clock: ClockSource;
  audible: "clock" | "solo";
}

/**
 * The clock is the A/B selection alone. It is deliberately independent of the
 * solo selector: `audioRef` points here, and the transcript highlight, waveform
 * playhead, skip-cuts logic and the solo gate all read it, so repointing it when
 * the soloed speaker changes would resynchronise the entire UI mid-playback.
 */
export function chooseClockSource(playbackSource: "original" | "processed", hasMastered: boolean): ClockSource {
  return playbackSource === "processed" && hasMastered ? "mastered" : "original";
}

/**
 * Solo tracks are rendered on the original timeline. A mastered file with cuts
 * has its own shorter timeline, so while that is the clock no follower can track
 * it -- the same position means two different moments.
 */
export function followersShareClockTimeline(clock: ClockSource, masterHasCutTimeline: boolean): boolean {
  return clock === "original" || !masterHasCutTimeline;
}

/**
 * Classify the soloed speaker's track. Readiness needs both loaded audio and a
 * position already locked to the clock, because the swap itself never seeks.
 *
 * `alreadyAudible` is hysteresis, not an optimisation: the audible element is
 * never drift-corrected, so its offset grows freely. Without this it would be
 * demoted for drift, corrected while muted, promoted again, and oscillate.
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
}): SoloTrackState {
  if (!input.hasTrack) {
    return "missing";
  }
  if (input.readyState < MEDIA_HAVE_CURRENT_DATA) {
    return "pending";
  }
  if (input.alreadyAudible) {
    return "ready";
  }
  const tolerance = input.toleranceSeconds ?? PEER_SYNC_TOLERANCE_S;
  return Math.abs(input.offsetFromClock) > tolerance ? "pending" : "ready";
}

/**
 * The only thing the solo selector changes: which mounted element is audible.
 * Falling back to the clock keeps sound flowing (gated to the speaker's regions)
 * while a track is missing or still catching up.
 */
export function chooseAudibleTarget(input: {
  playbackSource: "original" | "processed";
  hasMastered: boolean;
  masterHasCutTimeline: boolean;
  soloSpeakerIndex: number;
  soloTrackState: SoloTrackState;
}): AudibleTarget {
  const clock = chooseClockSource(input.playbackSource, input.hasMastered);
  const soloUsable =
    input.soloSpeakerIndex >= 0 &&
    input.soloTrackState === "ready" &&
    followersShareClockTimeline(clock, input.masterHasCutTimeline);
  return { clock, audible: soloUsable ? "solo" : "clock" };
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
