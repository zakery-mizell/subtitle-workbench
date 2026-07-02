import type { Caption, GuideBlock, Paragraph, WordToken } from "../types";
import type { CutRegion } from "./mastering";

/**
 * Map a timestamp on the original timeline onto the cut timeline.
 * Mirrors backend/app/mastering/cutting.py remap_timestamp.
 */
export function remapTime(t: number, cuts: CutRegion[]): number {
  let removed = 0;
  for (const cut of cuts) {
    if (t >= cut.end) {
      removed += cut.end - cut.start;
    } else if (t > cut.start) {
      removed += t - cut.start;
    } else {
      break;
    }
  }
  return Math.max(0, t - removed);
}

function sortCuts(cuts: CutRegion[]): CutRegion[] {
  return [...cuts].sort((a, b) => a.start - b.start);
}

function isFullyCut(start: number, end: number, cuts: CutRegion[]): boolean {
  return cuts.some((cut) => start >= cut.start && end <= cut.end);
}

export function remapWords(words: WordToken[], cuts: CutRegion[]): WordToken[] {
  const ordered = sortCuts(cuts);
  return words
    .filter((word) => !isFullyCut(word.start, word.end, ordered))
    .map((word) => {
      const start = remapTime(word.start, ordered);
      return { ...word, start, end: Math.max(start + 0.01, remapTime(word.end, ordered)) };
    });
}

export function remapCaptions(captions: Caption[], cuts: CutRegion[], keptWordIds: Set<string>): Caption[] {
  const ordered = sortCuts(cuts);
  return captions
    .filter((caption) => !isFullyCut(caption.start, caption.end, ordered))
    .map((caption) => {
      const start = remapTime(caption.start, ordered);
      return {
        ...caption,
        start,
        end: Math.max(start + 0.05, remapTime(caption.end, ordered)),
        word_ids: caption.word_ids.filter((id) => keptWordIds.has(id)),
      };
    });
}

export function remapParagraphs(paragraphs: Paragraph[], cuts: CutRegion[], keptWordIds: Set<string>): Paragraph[] {
  const ordered = sortCuts(cuts);
  return paragraphs
    .filter((paragraph) => !isFullyCut(paragraph.start, paragraph.end, ordered))
    .map((paragraph) => {
      const start = remapTime(paragraph.start, ordered);
      return {
        ...paragraph,
        start,
        end: Math.max(start + 0.05, remapTime(paragraph.end, ordered)),
        word_ids: paragraph.word_ids.filter((id) => keptWordIds.has(id)),
      };
    });
}

export function remapGuideBlocks(blocks: GuideBlock[], cuts: CutRegion[]): GuideBlock[] {
  const ordered = sortCuts(cuts);
  return blocks
    .filter((block) => !isFullyCut(block.start, block.end, ordered))
    .map((block) => {
      const start = remapTime(block.start, ordered);
      return { ...block, start, end: Math.max(start + 0.05, remapTime(block.end, ordered)) };
    });
}
