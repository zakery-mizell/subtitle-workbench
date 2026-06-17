export type GuideLabel = "SILENT" | "CUT" | "REPEAT";
export type SpeakerAssignmentMode = "segment" | "word";

export interface Speaker {
  id: number;
  name: string;
  show_attribution?: boolean;
}

export interface WordToken {
  id: string;
  text: string;
  start: number;
  end: number;
  confidence: number;
  low_confidence: boolean;
  speaker_id: number | null;
  speaker_name: string | null;
}

export interface Paragraph {
  id: string;
  start: number;
  end: number;
  speaker_id: number | null;
  speaker_name: string | null;
  text: string;
  word_ids: string[];
  caption_ids?: string[];
}

export interface Caption {
  id: string;
  start: number;
  end: number;
  speaker_id: number | null;
  speaker_name: string | null;
  lines: string[];
  word_ids: string[];
  blank_after: boolean;
}

export interface GuideBlock {
  id: string;
  start: number;
  end: number;
  label: GuideLabel;
  reason: string;
  skip: boolean;
}

export interface WarningItem {
  code: string;
  message: string;
}

export interface WaveformFrame {
  time: number;
  min: number;
  max: number;
  rms: number;
}

export interface SpeechSpan {
  start: number;
  end: number;
  peak: number;
}

export interface BackendCapabilities {
  diarization_configured: boolean;
}

export interface WaveformAnalysisResponse {
  duration: number;
  sample_rate: number;
  frame_duration: number;
  threshold: number;
  frames: WaveformFrame[];
  speech_spans: SpeechSpan[];
  warnings: WarningItem[];
}

export interface TranscriptResponse {
  audio_filename: string;
  duration: number | null;
  speakers: Speaker[];
  words: WordToken[];
  paragraphs: Paragraph[];
  captions: Caption[];
  guide_blocks: GuideBlock[];
  warnings: WarningItem[];
  model: string;
  speaker_assignment_mode: SpeakerAssignmentMode;
  language: string | null;
  gpu_enabled: boolean;
}

export interface RetranscribeRangeResponse {
  start: number;
  end: number;
  words: WordToken[];
  paragraphs: Paragraph[];
  captions: Caption[];
  warnings: WarningItem[];
  model: string;
  language: string | null;
  gpu_enabled: boolean;
}
