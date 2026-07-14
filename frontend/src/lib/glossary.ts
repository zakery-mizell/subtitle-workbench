import type { Caption, WordToken } from "../types";
import { DEFAULT_LOW_CONFIDENCE_THRESHOLD, isLowConfidenceWord } from "./confidence";

export interface ParsedGlossaryTerm {
  term: string;
  normalized: string;
  tokens: string[];
}

export interface JargonCandidate {
  display: string;
  normalized: string;
  count: number;
  lowConfidenceCount: number;
  averageConfidence: number;
  captionIndexes: number[];
  reasons: string[];
  score: number;
}

export interface CaptionGlossaryMatch {
  captionIndex: number;
  exactTerms: string[];
  fuzzyTerms: string[];
}

const TOKEN_RE = /[A-Za-z0-9]+(?:['’-][A-Za-z0-9]+)*/g;
const SENTENCE_END_RE = /[.!?]["'’)\]]?$/;
const COMMON_WORDS = new Set([
  "a", "about", "above", "actually", "after", "again", "against", "all", "almost", "already", "also",
  "although", "always", "am", "an", "and", "another", "any", "anyone", "anything", "are", "around", "as",
  "ask", "asked", "at", "away", "back", "bad", "be", "because", "become", "been", "before", "began", "begin",
  "behind", "being", "below", "best", "better", "between", "big", "both", "bring", "but", "by", "call",
  "called", "came", "can", "cannot", "case", "certain", "change", "come", "comes", "coming", "could",
  "couple", "course", "day", "days", "did", "different", "do", "does", "doing", "done", "down", "during",
  "each", "early", "either", "else", "end", "enough", "even", "ever", "every", "everyone", "everything",
  "exactly", "example", "far", "fact", "feel", "felt", "few", "find", "fine", "first", "for", "found", "from",
  "full", "get", "gets", "getting", "give", "given", "go", "goes", "going", "gone", "good", "got", "gotten",
  "great", "guess", "had", "half", "happen", "happened", "hard", "has", "have", "having", "he", "hear",
  "heard", "help", "her", "here", "hey", "high", "him", "himself", "his", "hold", "home", "how", "however",
  "huge", "idea", "if", "important", "in", "into", "is", "it", "its", "itself", "just", "keep", "kind",
  "knew", "know", "known", "large", "last", "late", "later", "least", "leave", "left", "less", "let", "life",
  "like", "little", "live", "long", "look", "looked", "looking", "looks", "lot", "made", "make", "makes",
  "making", "man", "many", "matter", "may", "maybe", "me", "mean", "means", "meant", "might", "mind", "more",
  "morning", "most", "move", "much", "must", "my", "myself", "name", "need", "needs", "never", "new", "next",
  "nice", "night", "no", "nobody", "none", "nor", "not", "nothing", "now", "number", "of", "off", "often",
  "oh", "okay", "old", "on", "once", "one", "only", "onto", "open", "or", "order", "other", "others", "our",
  "out", "over", "own", "part", "people", "perhaps", "person", "place", "point", "possible", "pretty",
  "probably", "problem", "put", "question", "quite", "rather", "reach", "read", "real", "really", "reason",
  "remember", "right", "run", "said", "same", "saw", "say", "saying", "says", "second", "see", "seem",
  "seemed", "seems", "seen", "sense", "set", "several", "she", "should", "show", "side", "similar", "since",
  "small", "so", "some", "someone", "something", "sometimes", "somewhere", "sort", "sound", "sounds", "start",
  "started", "still", "stuff", "such", "sure", "take", "taken", "takes", "talk", "talked", "talking", "tell",
  "than", "that", "the", "their", "them", "themselves", "then", "there", "these", "they", "thing", "things",
  "think", "thinking", "third", "this", "those", "though", "thought", "three", "through", "time", "times",
  "to", "today", "together", "told", "too", "took", "toward", "true", "try", "trying", "turn", "two",
  "under", "understand", "until", "up", "upon", "us", "use", "used", "using", "usually", "very", "wait",
  "want", "wanted", "wants", "was", "way", "ways", "we", "week", "well", "went", "were", "what", "whatever",
  "when", "where", "whether", "which", "while", "who", "whole", "whom", "why", "will", "wish", "with",
  "within", "without", "won", "word", "words", "work", "working", "world", "would", "wrong", "yeah", "year",
  "years", "yes", "yet", "you", "your", "yours", "yourself",
]);

function tokenize(text: string): string[] {
  return text.match(TOKEN_RE) ?? [];
}

export function normalizeGlossaryToken(text: string): string {
  const lower = text.toLowerCase().replace(/[’]/g, "'");
  const match = lower.match(TOKEN_RE)?.[0] ?? "";
  return match.replace(/^[-']+|[-']+$/g, "");
}

function tokenizeNormalized(text: string): string[] {
  return tokenize(text)
    .map((token) => normalizeGlossaryToken(token))
    .filter(Boolean);
}

function normalizePhrase(text: string): string {
  return tokenizeNormalized(text).join(" ");
}

function splitEntries(text: string): string[] {
  return text
    .split(/[\n,;]+/)
    .map((entry) => entry.trim())
    .filter(Boolean);
}

export function splitVocabularyEntries(text: string): string[] {
  return splitEntries(text);
}

function mergeUniqueEntries(entries: string[]): string[] {
  const seen = new Set<string>();
  const merged: string[] = [];
  for (const entry of entries) {
    const normalized = normalizePhrase(entry);
    if (!normalized || seen.has(normalized)) {
      continue;
    }
    seen.add(normalized);
    merged.push(entry.trim());
  }
  return merged;
}

function looksSpecialToken(text: string): boolean {
  return /[-]/.test(text) || /[A-Z]{2,}/.test(text) || /[A-Z][a-z]/.test(text);
}

function editDistance(left: string, right: string): number {
  if (left === right) {
    return 0;
  }
  if (!left.length) {
    return right.length;
  }
  if (!right.length) {
    return left.length;
  }

  const previous = Array.from({ length: right.length + 1 }, (_, index) => index);
  const current = new Array<number>(right.length + 1).fill(0);

  for (let i = 1; i <= left.length; i += 1) {
    current[0] = i;
    for (let j = 1; j <= right.length; j += 1) {
      const substitutionCost = left[i - 1] === right[j - 1] ? 0 : 1;
      current[j] = Math.min(
        previous[j] + 1,
        current[j - 1] + 1,
        previous[j - 1] + substitutionCost,
      );
    }
    for (let j = 0; j <= right.length; j += 1) {
      previous[j] = current[j];
    }
  }

  return previous[right.length];
}

function similarToken(left: string, right: string): boolean {
  if (!left || !right) {
    return false;
  }
  if (left === right) {
    return true;
  }
  const maxDistance = left.length >= 10 || right.length >= 10 ? 3 : left.length >= 6 || right.length >= 6 ? 2 : 1;
  if (Math.abs(left.length - right.length) > maxDistance) {
    return false;
  }
  return editDistance(left, right) <= maxDistance;
}

function stripEdgePunctuation(text: string): string {
  return text.replace(/^[^A-Za-z0-9]+/, "").replace(/[^A-Za-z0-9]+$/, "");
}

function pickDisplay(forms: Map<string, number>): string {
  const ranked = [...forms.entries()].sort((left, right) => {
    if (right[1] !== left[1]) {
      return right[1] - left[1];
    }
    if (looksSpecialToken(right[0]) !== looksSpecialToken(left[0])) {
      return Number(looksSpecialToken(right[0])) - Number(looksSpecialToken(left[0]));
    }
    return right[0].length - left[0].length;
  });
  return stripEdgePunctuation(ranked[0]?.[0] ?? "");
}

export function parseGlossaryTerms(text: string): ParsedGlossaryTerm[] {
  return mergeUniqueEntries(splitEntries(text))
    .map((term) => ({
      term,
      normalized: normalizePhrase(term),
      tokens: tokenizeNormalized(term),
    }))
    .filter((term) => term.normalized && term.tokens.length);
}

export function mergeVocabularyTexts(...texts: string[]): string {
  return mergeUniqueEntries(texts.flatMap((text) => splitEntries(text))).join("\n");
}

export function appendGlossaryTerms(currentText: string, terms: string[]): string {
  return mergeUniqueEntries([...splitEntries(currentText), ...terms]).join("\n");
}

const MAX_JARGON_CANDIDATES = 25;

export function detectJargonCandidates(
  words: WordToken[],
  captions: Caption[],
  glossaryText: string,
  lowConfidenceThreshold: number = DEFAULT_LOW_CONFIDENCE_THRESHOLD,
): JargonCandidate[] {
  const glossaryKeys = new Set(parseGlossaryTerms(glossaryText).map((term) => term.normalized));
  const captionIndexesByWordId = new Map<string, number[]>();
  captions.forEach((caption, index) => {
    for (const wordId of caption.word_ids) {
      const list = captionIndexesByWordId.get(wordId) ?? [];
      list.push(index);
      captionIndexesByWordId.set(wordId, list);
    }
  });

  const stats = new Map<
    string,
    {
      count: number;
      lowConfidenceCount: number;
      confidenceTotal: number;
      midSentenceCount: number;
      midSentenceTitlecaseCount: number;
      uppercaseCount: number;
      mixedCaseCount: number;
      hyphenated: boolean;
      hasDigit: boolean;
      forms: Map<string, number>;
      captionIndexes: Set<number>;
    }
  >();

  // Words arrive in reading order, so we can tell whether a capital letter is
  // just the start of a sentence or a genuine proper noun mid-sentence.
  let atSentenceStart = true;
  for (const word of words) {
    const isSentenceStart = atSentenceStart;
    atSentenceStart = SENTENCE_END_RE.test(word.text.trim());

    const normalized = normalizeGlossaryToken(word.text);
    if (!normalized || normalized.length < 2) {
      continue;
    }

    const current =
      stats.get(normalized) ??
      {
        count: 0,
        lowConfidenceCount: 0,
        confidenceTotal: 0,
        midSentenceCount: 0,
        midSentenceTitlecaseCount: 0,
        uppercaseCount: 0,
        mixedCaseCount: 0,
        hyphenated: false,
        hasDigit: false,
        forms: new Map<string, number>(),
        captionIndexes: new Set<number>(),
      };

    current.count += 1;
    current.confidenceTotal += word.confidence;
    current.lowConfidenceCount += isLowConfidenceWord(word, lowConfidenceThreshold) ? 1 : 0;
    if (!isSentenceStart) {
      current.midSentenceCount += 1;
      current.midSentenceTitlecaseCount += /^[A-Z][a-z]/.test(word.text) ? 1 : 0;
    }
    current.uppercaseCount += /^[A-Z0-9][A-Z0-9-]*[A-Z][A-Z0-9-]*$/.test(word.text) ? 1 : 0;
    // Internal capital (WhisperX, PyTorch, iPhone) that is not a plain acronym.
    current.mixedCaseCount += /[a-z][A-Z]/.test(word.text) ? 1 : 0;
    current.hyphenated ||= /[a-z]-[a-z]/i.test(word.text);
    current.hasDigit ||= /\d/.test(word.text) && /[a-z]/i.test(word.text);
    current.forms.set(word.text, (current.forms.get(word.text) ?? 0) + 1);
    for (const captionIndex of captionIndexesByWordId.get(word.id) ?? []) {
      current.captionIndexes.add(captionIndex);
    }
    stats.set(normalized, current);
  }

  const candidates: JargonCandidate[] = [];
  for (const [normalized, current] of stats.entries()) {
    if (glossaryKeys.has(normalized)) {
      continue;
    }

    const uncommon = !COMMON_WORDS.has(normalized);
    // A name reads as capitalized in most of its mid-sentence appearances;
    // a word that is only ever capitalized at sentence starts does not qualify.
    const consistentlyCapitalized =
      current.midSentenceTitlecaseCount > 0 &&
      current.midSentenceTitlecaseCount >= Math.ceil(current.midSentenceCount * 0.6);
    const isAcronym = current.uppercaseCount > 0;
    const isMixedCase = current.mixedCaseCount > 0;

    // Every candidate must carry a concrete "this is special vocabulary"
    // signal — being merely uncommon or repeated is not enough.
    const shouldKeep =
      current.hyphenated ||
      current.hasDigit ||
      isAcronym ||
      isMixedCase ||
      consistentlyCapitalized ||
      (current.lowConfidenceCount > 0 && uncommon && normalized.length >= 6);

    if (!shouldKeep) {
      continue;
    }

    const score =
      (current.hyphenated ? 4 : 0) +
      (current.hasDigit ? 4 : 0) +
      (isAcronym ? 4 : 0) +
      (isMixedCase ? 4 : 0) +
      (consistentlyCapitalized ? 3 : 0) +
      (current.lowConfidenceCount > 0 ? 2 : 0) +
      (current.count >= 3 ? 2 : current.count >= 2 ? 1 : 0) +
      (normalized.length >= 10 ? 2 : normalized.length >= 7 ? 1 : 0) +
      (uncommon ? 1 : 0);

    const reasons: string[] = [];
    if (isAcronym) {
      reasons.push("acronym");
    } else if (isMixedCase) {
      reasons.push("mixed case");
    } else if (consistentlyCapitalized) {
      reasons.push("name / proper noun");
    }
    if (current.hyphenated) {
      reasons.push("hyphenated");
    }
    if (current.hasDigit) {
      reasons.push("has a number");
    }
    if (current.lowConfidenceCount > 0) {
      reasons.push("low confidence");
    }

    candidates.push({
      display: pickDisplay(current.forms),
      normalized,
      count: current.count,
      lowConfidenceCount: current.lowConfidenceCount,
      averageConfidence: current.confidenceTotal / current.count,
      captionIndexes: [...current.captionIndexes].sort((left, right) => left - right),
      reasons,
      score,
    });
  }

  candidates.sort((left, right) => {
    if (right.score !== left.score) {
      return right.score - left.score;
    }
    if (right.lowConfidenceCount !== left.lowConfidenceCount) {
      return right.lowConfidenceCount - left.lowConfidenceCount;
    }
    if (right.count !== left.count) {
      return right.count - left.count;
    }
    return left.display.localeCompare(right.display);
  });

  return candidates.slice(0, MAX_JARGON_CANDIDATES);
}

function termMatch(tokens: string[], term: ParsedGlossaryTerm): { exact: boolean; fuzzy: boolean } {
  if (!tokens.length || !term.tokens.length) {
    return { exact: false, fuzzy: false };
  }

  const joinedTokens = tokens.join(" ");
  if (joinedTokens.includes(term.normalized)) {
    return { exact: true, fuzzy: true };
  }

  const usedIndexes = new Set<number>();
  let matched = 0;
  for (const termToken of term.tokens) {
    let exactIndex = -1;
    for (let index = 0; index < tokens.length; index += 1) {
      if (usedIndexes.has(index)) {
        continue;
      }
      if (tokens[index] === termToken) {
        exactIndex = index;
        break;
      }
    }
    if (exactIndex >= 0) {
      usedIndexes.add(exactIndex);
      matched += 1;
      continue;
    }

    for (let index = 0; index < tokens.length; index += 1) {
      if (usedIndexes.has(index)) {
        continue;
      }
      if (similarToken(tokens[index], termToken)) {
        usedIndexes.add(index);
        matched += 1;
        break;
      }
    }
  }

  if (term.tokens.length === 1) {
    return { exact: false, fuzzy: matched === 1 };
  }

  return { exact: false, fuzzy: matched / term.tokens.length >= 0.75 && matched >= Math.min(2, term.tokens.length) };
}

export function findCaptionGlossaryMatches(captions: Caption[], glossaryText: string): CaptionGlossaryMatch[] {
  const terms = parseGlossaryTerms(glossaryText);
  if (!terms.length) {
    return [];
  }

  return captions.map((caption, captionIndex) => {
    const tokens = tokenizeNormalized(caption.lines.join(" "));
    const exactTerms: string[] = [];
    const fuzzyTerms: string[] = [];

    for (const term of terms) {
      const match = termMatch(tokens, term);
      if (match.exact) {
        exactTerms.push(term.term);
      } else if (match.fuzzy) {
        fuzzyTerms.push(term.term);
      }
    }

    return {
      captionIndex,
      exactTerms,
      fuzzyTerms,
    };
  });
}
