import type { WordToken } from "../types";

// Matches the backend default (config.low_confidence_threshold). The user can
// tighten or relax this live in the UI; word.low_confidence only serves as a
// fallback for words that predate the numeric confidence field.
export const DEFAULT_LOW_CONFIDENCE_THRESHOLD = 0.55;

export function isLowConfidenceWord(word: WordToken, threshold: number): boolean {
  return Number.isFinite(word.confidence) ? word.confidence < threshold : word.low_confidence;
}
