import type { Caption, WordToken } from "../types";

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
const COMMON_WORDS = new Set([
  "a", "about", "after", "again", "against", "all", "almost", "also", "am", "an", "and", "any", "are",
  "around", "as", "at", "back", "be", "because", "been", "before", "being", "between", "both", "but", "by",
  "can", "come", "could", "day", "did", "do", "does", "doing", "done", "down", "during", "each", "even",
  "every", "few", "find", "first", "for", "from", "get", "give", "go", "going", "good", "got", "had", "has",
  "have", "he", "her", "here", "him", "his", "how", "i", "if", "in", "into", "is", "it", "its", "itself",
  "just", "kind", "know", "like", "little", "long", "look", "lot", "made", "make", "many", "may", "me",
  "mean", "might", "more", "most", "much", "my", "need", "never", "new", "no", "not", "now", "of", "off",
  "often", "okay", "on", "once", "one", "only", "or", "other", "our", "out", "over", "people", "really",
  "right", "said", "same", "say", "see", "seem", "she", "should", "so", "some", "something", "still", "such",
  "take", "than", "that", "the", "their", "them", "then", "there", "these", "they", "thing", "think", "this",
  "those", "through", "time", "to", "too", "two", "up", "us", "use", "very", "want", "was", "way", "we",
  "well", "were", "what", "when", "where", "which", "while", "who", "why", "will", "with", "would", "yeah",
  "yes", "you", "your",
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
  return ranked[0]?.[0] ?? "";
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

export function detectJargonCandidates(
  words: WordToken[],
  captions: Caption[],
  glossaryText: string,
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
      titlecaseCount: number;
      uppercaseCount: number;
      hyphenated: boolean;
      forms: Map<string, number>;
      captionIndexes: Set<number>;
    }
  >();

  for (const word of words) {
    const normalized = normalizeGlossaryToken(word.text);
    if (!normalized || normalized.length < 4) {
      continue;
    }

    const current =
      stats.get(normalized) ??
      {
        count: 0,
        lowConfidenceCount: 0,
        confidenceTotal: 0,
        titlecaseCount: 0,
        uppercaseCount: 0,
        hyphenated: false,
        forms: new Map<string, number>(),
        captionIndexes: new Set<number>(),
      };

    current.count += 1;
    current.confidenceTotal += word.confidence;
    current.lowConfidenceCount += word.low_confidence ? 1 : 0;
    current.titlecaseCount += /^[A-Z][a-z]/.test(word.text) ? 1 : 0;
    current.uppercaseCount += /^[A-Z0-9-]{2,}$/.test(word.text) ? 1 : 0;
    current.hyphenated ||= /-/.test(word.text);
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
    const score =
      (current.hyphenated ? 5 : 0) +
      (current.uppercaseCount > 0 ? 4 : 0) +
      (current.titlecaseCount > 0 ? 3 : 0) +
      (current.lowConfidenceCount > 0 ? 3 : 0) +
      (current.count >= 3 ? 3 : current.count >= 2 ? 2 : 0) +
      (normalized.length >= 10 ? 2 : normalized.length >= 7 ? 1 : 0) +
      (uncommon ? 1 : 0);

    const shouldKeep =
      score >= 5 ||
      current.hyphenated ||
      current.uppercaseCount > 0 ||
      (current.lowConfidenceCount > 0 && uncommon) ||
      (current.count >= 2 && uncommon && normalized.length >= 6);

    if (!shouldKeep) {
      continue;
    }

    const reasons: string[] = [];
    if (current.hyphenated) {
      reasons.push("hyphenated");
    }
    if (current.uppercaseCount > 0) {
      reasons.push("acronym / uppercase");
    } else if (current.titlecaseCount > 0) {
      reasons.push("proper noun");
    }
    if (current.lowConfidenceCount > 0) {
      reasons.push("low confidence");
    }
    if (current.count >= 2) {
      reasons.push("repeated");
    }
    if (uncommon) {
      reasons.push("uncommon");
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

  return candidates.sort((left, right) => {
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
