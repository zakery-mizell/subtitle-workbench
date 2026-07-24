import { useEffect, useMemo, useRef, useState } from "react";
import type { KeyboardEvent as ReactKeyboardEvent, PointerEvent as ReactPointerEvent } from "react";

import {
  MIN_REGION_S,
  addRegion,
  collectForeignEdges,
  crossSpeakerOverlaps,
  deleteRegion,
  firstFrameIndexAtOrAfter,
  moveRegionEdge,
  normalizeRegions,
  panViewWindow,
  reassignRegion,
  regionStats,
  sliceIntervals,
  snapRegionEdge,
  splitRegionAt,
  zoomViewWindow,
} from "./lib/regions";
import type { RegionEdge, RegionInterval, RegionMarker, ViewWindow } from "./lib/regions";
import { formatClock, formatGutterClock } from "./lib/time";
import type { OverlapRegion, Speaker, SpeakerRegion, SpeechSpan, WaveformFrame } from "./types";

const STRIP_HEIGHT = 78;
const LANE_HEIGHT = 36;
const MIN_VIEW_SECONDS = 0.4;
const VIEW_PRESETS = [2, 5, 10, 30] as const;
const DEFAULT_VIEW_SECONDS = 10;
// A grab handle has to be reachable without being wider than the shortest region
// on screen; 6 px is ~40 ms at the 5 s preset in the side rail.
const EDGE_HIT_PX = 6;
const NUDGE_S = 0.01;
const NUDGE_COARSE_S = 0.1;
// A drag shorter than this on empty lane space is a click, not a new region.
const MIN_CREATE_S = 0.05;
const NEW_REGION_S = 0.6;

interface RegionsPanelProps {
  speakers: Speaker[];
  soloableSpeakerIds: number[];
  regions: SpeakerRegion[];
  /** True once manual edits replaced the audio-derived regions. */
  overrideActive: boolean;
  frames: WaveformFrame[];
  speechSpans: SpeechSpan[];
  overlapRegions: OverlapRegion[];
  markers: RegionMarker[];
  duration: number;
  currentTime: number;
  soloSpeakerId: number | null;
  theme: "light" | "dark";
  onChange: (regions: SpeakerRegion[]) => void;
  onReset: () => void;
  onSoloSpeakerChange: (speakerId: number | null) => void;
  onSeek: (time: number, options?: { play?: boolean }) => void;
}

type DragState =
  | { kind: "edge"; pointerId: number; id: string; edge: RegionEdge }
  | { kind: "create"; pointerId: number; speakerId: number; anchor: number }
  | { kind: "seek"; pointerId: number }
  | { kind: "pan"; pointerId: number; anchorX: number; anchorStart: number };

/**
 * Theme colours are resolved once per theme change, not per draw call: the strip
 * plus every lane redraws on each pointer move, and getComputedStyle forces a
 * style resolution each time it is called.
 */
interface Palette {
  bg: string;
  lane: string;
  speech: string;
  rest: string;
  grid: string;
  muted: string;
  accent: string;
  accentStrong: string;
  playhead: string;
  danger: string;
  warning: string;
  collision: string;
}

function themeColor(name: string, fallback: string): string {
  const value = getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  return value || fallback;
}

function readPalette(): Palette {
  return {
    bg: themeColor("--wave-bg", "#ffffff"),
    lane: themeColor("--surface-soft", "#f9fbff"),
    speech: themeColor("--wave-played", "#1d9e75"),
    rest: themeColor("--wave-rest", "#b4b2a9"),
    grid: themeColor("--wave-grid", "#e2e8f0"),
    muted: themeColor("--muted", "#647184"),
    accent: themeColor("--accent", "#2563eb"),
    accentStrong: themeColor("--accent-strong", "#1d4ed8"),
    playhead: themeColor("--playhead", "#d85a30"),
    danger: themeColor("--danger", "#c2410c"),
    warning: themeColor("--warning", "#b7791f"),
    collision: themeColor("--repeat", "#7c3aed"),
  };
}

function prepareCanvas(
  canvas: HTMLCanvasElement,
  width: number,
  height: number,
): CanvasRenderingContext2D | null {
  const context = canvas.getContext("2d");
  if (!context) {
    return null;
  }
  const dpr = window.devicePixelRatio || 1;
  canvas.width = Math.max(1, Math.floor(width * dpr));
  canvas.height = Math.max(1, Math.floor(height * dpr));
  canvas.style.width = `${width}px`;
  canvas.style.height = `${height}px`;
  context.setTransform(dpr, 0, 0, dpr, 0, 0);
  context.clearRect(0, 0, width, height);
  return context;
}

function fillHatched(
  context: CanvasRenderingContext2D,
  x: number,
  y: number,
  width: number,
  height: number,
  color: string,
) {
  context.save();
  context.beginPath();
  context.rect(x, y, width, height);
  context.clip();
  context.strokeStyle = color;
  context.lineWidth = 1;
  for (let offset = -height; offset < width + height; offset += 6) {
    context.beginPath();
    context.moveTo(x + offset, y + height);
    context.lineTo(x + offset + height, y);
    context.stroke();
  }
  context.restore();
}

function markerColor(kind: string, palette: Palette): string {
  if (kind === "overlap") {
    return palette.danger;
  }
  if (kind === "tight_handoff") {
    return palette.warning;
  }
  return palette.muted;
}

interface Viewport {
  view: ViewWindow;
  width: number;
}

function timeToX(time: number, viewport: Viewport): number {
  return ((time - viewport.view.start) / viewport.view.seconds) * viewport.width;
}

function xToTime(x: number, viewport: Viewport): number {
  return viewport.view.start + (x / viewport.width) * viewport.view.seconds;
}

/**
 * Speech blocks, not the RMS silhouette, are the reference the user places a
 * boundary against: `frames` is downsampled to DISPLAY_FRAME_LIMIT (~190 ms
 * apart on a 19-minute file), while `speech_spans` keep the VAD's 20 ms
 * resolution and are what the regions snap to.
 */
function drawSpeechBlocks(
  context: CanvasRenderingContext2D,
  spans: SpeechSpan[],
  viewport: Viewport,
  top: number,
  height: number,
  alpha: number,
  palette: Palette,
) {
  const viewEnd = viewport.view.start + viewport.view.seconds;
  const visible = sliceIntervals(spans, viewport.view.start, viewEnd);
  if (!visible.length) {
    return;
  }
  const fill = palette.speech;
  context.save();
  for (const span of visible) {
    const left = Math.max(0, timeToX(span.start, viewport));
    const right = Math.min(viewport.width, timeToX(span.end, viewport));
    if (right <= left) {
      continue;
    }
    context.globalAlpha = alpha;
    context.fillStyle = fill;
    context.fillRect(left, top, Math.max(1, right - left), height);
    // Full-strength onset/offset rules: at tight zoom these are the pixels a
    // dragged edge snaps to, so they must stay legible under the block tint.
    context.globalAlpha = Math.min(1, alpha * 2.6);
    context.fillRect(left, top, 1, height);
    context.fillRect(Math.max(left, right - 1), top, 1, height);
  }
  context.restore();
}

function drawOverlapBands(
  context: CanvasRenderingContext2D,
  overlaps: OverlapRegion[],
  viewport: Viewport,
  top: number,
  height: number,
  palette: Palette,
) {
  const viewEnd = viewport.view.start + viewport.view.seconds;
  const color = palette.danger;
  for (const overlap of overlaps) {
    if (overlap.end < viewport.view.start || overlap.start > viewEnd) {
      continue;
    }
    const left = Math.max(0, timeToX(overlap.start, viewport));
    const right = Math.min(viewport.width, timeToX(overlap.end, viewport));
    if (right > left) {
      fillHatched(context, left, top, Math.max(1, right - left), height, color);
    }
  }
}

function drawPlayhead(
  context: CanvasRenderingContext2D,
  time: number,
  viewport: Viewport,
  height: number,
  palette: Palette,
) {
  const x = timeToX(time, viewport);
  if (x < 0 || x > viewport.width) {
    return;
  }
  context.strokeStyle = palette.playhead;
  context.lineWidth = 1.5;
  context.beginPath();
  context.moveTo(x, 0);
  context.lineTo(x, height);
  context.stroke();
}

interface StripProps {
  viewport: Viewport;
  frames: WaveformFrame[];
  speechSpans: SpeechSpan[];
  overlapRegions: OverlapRegion[];
  markers: RegionMarker[];
  currentTime: number;
  palette: Palette;
}

function drawStrip(canvas: HTMLCanvasElement, props: StripProps) {
  const { viewport } = props;
  const context = prepareCanvas(canvas, viewport.width, STRIP_HEIGHT);
  if (!context) {
    return;
  }
  const viewEnd = viewport.view.start + viewport.view.seconds;

  context.fillStyle = props.palette.bg;
  context.fillRect(0, 0, viewport.width, STRIP_HEIGHT);

  const bodyTop = 12;
  const bodyHeight = STRIP_HEIGHT - bodyTop - 12;
  drawOverlapBands(context, props.overlapRegions, viewport, bodyTop, bodyHeight, props.palette);
  drawSpeechBlocks(context, props.speechSpans, viewport, bodyTop, bodyHeight, 0.22, props.palette);

  // RMS silhouette, secondary: at tight zoom only a handful of downsampled
  // frames fall inside the window, so it is drawn as a polyline that degrades
  // to a coarse hint rather than as bars pretending to per-pixel accuracy.
  const centerY = bodyTop + bodyHeight / 2;
  const half = bodyHeight / 2 - 1;
  const firstFrame = Math.max(0, firstFrameIndexAtOrAfter(props.frames, viewport.view.start) - 1);
  context.strokeStyle = props.palette.rest;
  context.lineWidth = 1;
  context.beginPath();
  let plotted = 0;
  for (let index = firstFrame; index < props.frames.length; index += 1) {
    const frame = props.frames[index];
    if (frame.time > viewEnd) {
      break;
    }
    const x = timeToX(frame.time, viewport);
    const amplitude = Math.max(Math.abs(frame.max), Math.abs(frame.min));
    const y = centerY - amplitude * half;
    if (plotted === 0) {
      context.moveTo(x, y);
    } else {
      context.lineTo(x, y);
    }
    plotted += 1;
  }
  if (plotted > 1) {
    context.stroke();
  }

  // Time ruler: three labels is enough orientation in a 348 px rail.
  context.fillStyle = props.palette.muted;
  context.font = "10px Inter, sans-serif";
  context.strokeStyle = props.palette.grid;
  for (let step = 0; step <= 2; step += 1) {
    const time = viewport.view.start + (viewport.view.seconds * step) / 2;
    const x = step === 0 ? 2 : step === 2 ? viewport.width - 2 : viewport.width / 2;
    context.beginPath();
    context.moveTo(x, STRIP_HEIGHT - 11);
    context.lineTo(x, STRIP_HEIGHT - 8);
    context.stroke();
    context.textAlign = step === 0 ? "left" : step === 2 ? "right" : "center";
    context.fillText(formatGutterClock(time), x, STRIP_HEIGHT - 1);
  }
  context.textAlign = "left";

  for (const marker of props.markers) {
    if (marker.time < viewport.view.start || marker.time > viewEnd) {
      continue;
    }
    const x = timeToX(marker.time, viewport);
    context.strokeStyle = markerColor(marker.kind, props.palette);
    context.lineWidth = 2;
    context.beginPath();
    context.moveTo(x, 0);
    context.lineTo(x, 9);
    context.stroke();
  }

  drawPlayhead(context, props.currentTime, viewport, STRIP_HEIGHT - 12, props.palette);
}

interface LaneDrawProps {
  viewport: Viewport;
  regions: SpeakerRegion[];
  speakerId: number;
  speechSpans: SpeechSpan[];
  overlapRegions: OverlapRegion[];
  currentTime: number;
  selectedId: string | null;
  selectedEdge: RegionEdge | null;
  pendingSpan: RegionInterval | null;
  palette: Palette;
}

function drawLane(canvas: HTMLCanvasElement, props: LaneDrawProps) {
  const { viewport } = props;
  const context = prepareCanvas(canvas, viewport.width, LANE_HEIGHT);
  if (!context) {
    return;
  }
  const viewEnd = viewport.view.start + viewport.view.seconds;

  context.fillStyle = props.palette.lane;
  context.fillRect(0, 0, viewport.width, LANE_HEIGHT);
  drawSpeechBlocks(context, props.speechSpans, viewport, 2, LANE_HEIGHT - 4, 0.1, props.palette);
  drawOverlapBands(context, props.overlapRegions, viewport, 0, LANE_HEIGHT, props.palette);

  // One linear pass over the (few dozen) regions keeps the draw windowed; the
  // binary-searched sliceIntervals cannot be used across speakers because
  // cross-speaker regions are allowed to overlap.
  const inWindow = props.regions.filter((region) => region.end > viewport.view.start && region.start < viewEnd);
  const visible = inWindow.filter((region) => region.speaker_id === props.speakerId);
  const barTop = 5;
  const barHeight = LANE_HEIGHT - 10;
  const fill = props.palette.accent;
  const stroke = props.palette.accentStrong;

  for (const region of visible) {
    const left = Math.max(-2, timeToX(region.start, viewport));
    const right = Math.min(viewport.width + 2, timeToX(region.end, viewport));
    const width = Math.max(2, right - left);
    const selected = region.id === props.selectedId;

    context.save();
    context.globalAlpha = selected ? 0.42 : 0.24;
    context.fillStyle = fill;
    context.beginPath();
    context.roundRect(left, barTop, width, barHeight, 3);
    context.fill();
    context.restore();

    context.strokeStyle = selected ? stroke : fill;
    context.lineWidth = selected ? 1.6 : 1;
    context.beginPath();
    context.roundRect(left, barTop, width, barHeight, 3);
    context.stroke();

    // Grab handles. The active one is drawn wider so the user can see which
    // edge the arrow keys will nudge.
    context.fillStyle = stroke;
    for (const edge of ["start", "end"] as RegionEdge[]) {
      const x = edge === "start" ? left : right;
      const active = selected && props.selectedEdge === edge;
      const handleWidth = active ? 3 : 2;
      context.globalAlpha = active ? 1 : 0.75;
      context.fillRect(x - handleWidth / 2, barTop, handleWidth, barHeight);
    }
    context.globalAlpha = 1;
  }

  // Where this speaker's region collides with another's, both lanes tint the
  // span: a deliberate cross-speaker overlap, not an error.
  const collisions = crossSpeakerOverlaps(inWindow, props.speakerId);
  if (collisions.length) {
    context.save();
    context.globalAlpha = 0.3;
    context.fillStyle = props.palette.collision;
    for (const span of collisions) {
      const left = Math.max(0, timeToX(span.start, viewport));
      const right = Math.min(viewport.width, timeToX(span.end, viewport));
      if (right > left) {
        context.fillRect(left, barTop, right - left, barHeight);
      }
    }
    context.restore();
  }

  if (props.pendingSpan) {
    const left = Math.max(0, timeToX(props.pendingSpan.start, viewport));
    const right = Math.min(viewport.width, timeToX(props.pendingSpan.end, viewport));
    context.strokeStyle = stroke;
    context.setLineDash([3, 3]);
    context.lineWidth = 1;
    context.strokeRect(left, barTop, Math.max(1, right - left), barHeight);
    context.setLineDash([]);
  }

  drawPlayhead(context, props.currentTime, viewport, LANE_HEIGHT, props.palette);
}

interface LaneProps extends LaneDrawProps {
  onPointerDown: (event: ReactPointerEvent<HTMLCanvasElement>) => void;
  onPointerMove: (event: ReactPointerEvent<HTMLCanvasElement>) => void;
  onPointerUp: (event: ReactPointerEvent<HTMLCanvasElement>) => void;
}

function RegionLane({ onPointerDown, onPointerMove, onPointerUp, ...drawProps }: LaneProps) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (canvas && drawProps.viewport.width > 0) {
      drawLane(canvas, drawProps);
    }
  });

  return (
    <canvas
      ref={canvasRef}
      className="regions-lane-canvas"
      onPointerDown={onPointerDown}
      onPointerMove={onPointerMove}
      onPointerUp={onPointerUp}
      onPointerCancel={onPointerUp}
    />
  );
}

export function RegionsPanel({
  speakers,
  soloableSpeakerIds,
  regions,
  overrideActive,
  frames,
  speechSpans,
  overlapRegions,
  markers,
  duration,
  currentTime,
  soloSpeakerId,
  theme,
  onChange,
  onReset,
  onSoloSpeakerChange,
  onSeek,
}: RegionsPanelProps) {
  const bodyRef = useRef<HTMLDivElement | null>(null);
  // Measured separately from the body: the body carries the padding and border,
  // and the canvases must be exactly as wide as the content box or the drawn
  // time axis and the pointer's time axis drift apart.
  const trackRef = useRef<HTMLDivElement | null>(null);
  const stripRef = useRef<HTMLCanvasElement | null>(null);
  const dragRef = useRef<DragState | null>(null);
  const [width, setWidth] = useState(0);
  const [view, setView] = useState<ViewWindow>({ start: 0, seconds: DEFAULT_VIEW_SECONDS });
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [selectedEdge, setSelectedEdge] = useState<RegionEdge | null>(null);
  // A drag previews locally and commits once, on release: one undo entry per
  // gesture instead of one per pointer move.
  const [preview, setPreview] = useState<SpeakerRegion[] | null>(null);
  const [pending, setPending] = useState<{ speakerId: number; span: RegionInterval } | null>(null);
  const [startDraft, setStartDraft] = useState<string | null>(null);
  const [endDraft, setEndDraft] = useState<string | null>(null);

  const totalDuration = duration > 0 ? duration : Math.max(1, view.start + view.seconds);
  const displayed = preview ?? regions;
  const selected = displayed.find((region) => region.id === selectedId) ?? null;
  const stats = useMemo(() => regionStats(displayed), [displayed]);
  const viewport = useMemo<Viewport>(() => ({ view, width }), [view, width]);
  const palette = useMemo(() => readPalette(), [theme]);

  useEffect(() => {
    const track = trackRef.current;
    if (!track) {
      return;
    }
    const update = () => setWidth(Math.max(120, Math.floor(track.getBoundingClientRect().width)));
    update();
    const observer = new ResizeObserver(update);
    observer.observe(track);
    return () => observer.disconnect();
  }, []);

  useEffect(() => {
    const canvas = stripRef.current;
    if (!canvas || width <= 0) {
      return;
    }
    drawStrip(canvas, {
      viewport,
      frames,
      speechSpans,
      overlapRegions,
      markers,
      currentTime,
      palette,
    });
  }, [viewport, frames, speechSpans, overlapRegions, markers, currentTime, palette, width]);

  // Wheel zoom/pan needs preventDefault, which React's passive wheel listener
  // cannot do, so it is attached natively.
  useEffect(() => {
    const body = bodyRef.current;
    const track = trackRef.current;
    if (!body || !track) {
      return;
    }
    const onWheel = (event: WheelEvent) => {
      if (!event.ctrlKey && !event.metaKey && !event.shiftKey) {
        return;
      }
      event.preventDefault();
      const rect = track.getBoundingClientRect();
      const localWidth = Math.max(1, rect.width);
      setView((current) => {
        if (event.ctrlKey || event.metaKey) {
          const anchor = current.start + ((event.clientX - rect.left) / localWidth) * current.seconds;
          return zoomViewWindow(current, event.deltaY > 0 ? 1.3 : 1 / 1.3, anchor, totalDuration, MIN_VIEW_SECONDS);
        }
        const delta = ((event.deltaX || event.deltaY) / localWidth) * current.seconds;
        return panViewWindow(current, delta, totalDuration);
      });
    };
    body.addEventListener("wheel", onWheel, { passive: false });
    return () => body.removeEventListener("wheel", onWheel);
  }, [totalDuration]);

  // Follow the playhead only when it leaves the window: returning `current`
  // unchanged is a no-op for React, so the user's own panning is left alone
  // while the position stays visible.
  useEffect(() => {
    setView((current) =>
      currentTime >= current.start && currentTime <= current.start + current.seconds
        ? current
        : panViewWindow({ start: currentTime - current.seconds * 0.35, seconds: current.seconds }, 0, totalDuration),
    );
  }, [currentTime, totalDuration]);

  // Typed times belong to whichever region was selected when they were typed.
  useEffect(() => {
    setStartDraft(null);
    setEndDraft(null);
  }, [selectedId]);

  function commitRegions(next: SpeakerRegion[]) {
    setPreview(null);
    onChange(next);
  }

  function setViewSeconds(seconds: number) {
    setView((current) => {
      const center =
        currentTime >= current.start && currentTime <= current.start + current.seconds
          ? currentTime
          : current.start + current.seconds / 2;
      return zoomViewWindow(current, seconds / current.seconds, center, totalDuration, MIN_VIEW_SECONDS);
    });
  }

  function localTime(event: ReactPointerEvent<HTMLCanvasElement>): number {
    const rect = event.currentTarget.getBoundingClientRect();
    return xToTime(event.clientX - rect.left, { view, width: Math.max(1, rect.width) });
  }

  function handleStripPointerDown(event: ReactPointerEvent<HTMLCanvasElement>) {
    event.currentTarget.setPointerCapture(event.pointerId);
    if (event.shiftKey) {
      dragRef.current = {
        kind: "pan",
        pointerId: event.pointerId,
        anchorX: event.clientX,
        anchorStart: view.start,
      };
      return;
    }
    dragRef.current = { kind: "seek", pointerId: event.pointerId };
    onSeek(Math.max(0, localTime(event)), { play: false });
  }

  function handleStripPointerMove(event: ReactPointerEvent<HTMLCanvasElement>) {
    const drag = dragRef.current;
    if (!drag) {
      return;
    }
    if (drag.kind === "seek") {
      onSeek(Math.max(0, localTime(event)), { play: false });
      return;
    }
    if (drag.kind === "pan") {
      const rect = event.currentTarget.getBoundingClientRect();
      const delta = ((drag.anchorX - event.clientX) / Math.max(1, rect.width)) * view.seconds;
      setView(panViewWindow({ start: drag.anchorStart, seconds: view.seconds }, delta, totalDuration));
    }
  }

  function endDrag(event: ReactPointerEvent<HTMLCanvasElement>) {
    if (event.currentTarget.hasPointerCapture(event.pointerId)) {
      event.currentTarget.releasePointerCapture(event.pointerId);
    }
    dragRef.current = null;
  }

  function handleLanePointerDown(event: ReactPointerEvent<HTMLCanvasElement>, speakerId: number) {
    bodyRef.current?.focus();
    event.currentTarget.setPointerCapture(event.pointerId);
    const rect = event.currentTarget.getBoundingClientRect();
    const laneViewport = { view, width: Math.max(1, rect.width) };
    const x = event.clientX - rect.left;
    const time = xToTime(x, laneViewport);
    const viewEnd = view.start + view.seconds;
    const mine = displayed.filter((region) => region.speaker_id === speakerId);

    for (const region of sliceIntervals(mine, view.start, viewEnd)) {
      const left = timeToX(region.start, laneViewport);
      const right = timeToX(region.end, laneViewport);
      if (Math.abs(x - left) <= EDGE_HIT_PX) {
        setSelectedId(region.id);
        setSelectedEdge("start");
        dragRef.current = { kind: "edge", pointerId: event.pointerId, id: region.id, edge: "start" };
        return;
      }
      if (Math.abs(x - right) <= EDGE_HIT_PX) {
        setSelectedId(region.id);
        setSelectedEdge("end");
        dragRef.current = { kind: "edge", pointerId: event.pointerId, id: region.id, edge: "end" };
        return;
      }
      if (x > left && x < right) {
        setSelectedId(region.id);
        setSelectedEdge(null);
        dragRef.current = null;
        return;
      }
    }

    dragRef.current = { kind: "create", pointerId: event.pointerId, speakerId, anchor: time };
    setPending({ speakerId, span: { start: time, end: time } });
  }

  function handleLanePointerMove(event: ReactPointerEvent<HTMLCanvasElement>) {
    const drag = dragRef.current;
    if (!drag) {
      return;
    }
    const time = localTime(event);
    if (drag.kind === "edge") {
      setPreview(
        moveRegionEdge(regions, drag.id, drag.edge, time, {
          duration: duration || null,
          speechSpans,
          snap: !event.altKey,
        }),
      );
      return;
    }
    if (drag.kind === "create") {
      setPending({
        speakerId: drag.speakerId,
        span: { start: Math.min(drag.anchor, time), end: Math.max(drag.anchor, time) },
      });
    }
  }

  function handleLanePointerUp(event: ReactPointerEvent<HTMLCanvasElement>) {
    const drag = dragRef.current;
    endDrag(event);
    if (!drag) {
      return;
    }
    if (drag.kind === "edge") {
      const next = moveRegionEdge(regions, drag.id, drag.edge, localTime(event), {
        duration: duration || null,
        speechSpans,
        snap: !event.altKey,
      });
      commitRegions(next);
      return;
    }
    if (drag.kind === "create") {
      // Taken from the release event rather than the `pending` preview state,
      // which can still be one pointermove behind when the pointer goes up.
      const time = localTime(event);
      const span = { start: Math.min(drag.anchor, time), end: Math.max(drag.anchor, time) };
      setPending(null);
      if (span.end - span.start < MIN_CREATE_S) {
        return;
      }
      const foreign = collectForeignEdges(regions, drag.speakerId);
      const start = event.altKey ? span.start : snapRegionEdge(span.start, "start", speechSpans, foreign);
      const end = event.altKey ? span.end : snapRegionEdge(span.end, "end", speechSpans, foreign);
      const result = addRegion(regions, drag.speakerId, start, end, duration || null);
      setSelectedId(result.id);
      setSelectedEdge(null);
      commitRegions(result.regions);
    }
  }

  function handleAddAtPlayhead(speakerId: number) {
    const foreign = collectForeignEdges(regions, speakerId);
    const start = snapRegionEdge(currentTime, "start", speechSpans, foreign);
    const end = snapRegionEdge(currentTime + NEW_REGION_S, "end", speechSpans, foreign);
    const result = addRegion(regions, speakerId, start, Math.max(end, start + NEW_REGION_S / 2), duration || null);
    setSelectedId(result.id);
    setSelectedEdge(null);
    commitRegions(result.regions);
  }

  function nudge(deltaSeconds: number) {
    if (!selected) {
      return;
    }
    if (selectedEdge) {
      // Nudges are exact by definition; snapping would pull them straight back.
      commitRegions(
        moveRegionEdge(regions, selected.id, selectedEdge, selected[selectedEdge] + deltaSeconds, {
          duration: duration || null,
          snap: false,
        }),
      );
      return;
    }
    const shifted = regions.map((region) =>
      region.id === selected.id
        ? { ...region, start: region.start + deltaSeconds, end: region.end + deltaSeconds }
        : region,
    );
    commitRegions(normalizeRegions(shifted, duration || null, selected.id));
  }

  function handleKeyDown(event: ReactKeyboardEvent<HTMLDivElement>) {
    if (event.key !== "ArrowLeft" && event.key !== "ArrowRight") {
      return;
    }
    if (!selected || event.metaKey || event.ctrlKey) {
      return;
    }
    // The window-level handler skips playback by 3 s on bare arrows; while a
    // region is selected in here the arrows belong to the editor.
    event.preventDefault();
    event.stopPropagation();
    const step = event.shiftKey ? NUDGE_COARSE_S : NUDGE_S;
    nudge(event.key === "ArrowLeft" ? -step : step);
  }

  function commitEdgeInput(edge: RegionEdge, raw: string) {
    if (!selected) {
      return;
    }
    const value = Number(raw);
    if (!Number.isFinite(value)) {
      return;
    }
    commitRegions(
      moveRegionEdge(regions, selected.id, edge, value, { duration: duration || null, snap: false }),
    );
  }

  const laneSpeakers = speakers.length ? speakers : [{ id: 0, name: "Speaker 1" }];
  const canEdit = speechSpans.length > 0 || regions.length > 0;

  return (
    <section className="selection-panel mastering-panel regions-panel">
      <div className="panel-section-heading">
        <p className="eyebrow">Regions</p>
        <h3>Speaker regions</h3>
      </div>
      <p className="helper-text">
        The gate that isolates each voice for the per-speaker exports. Boundaries are derived from the measured speech
        spans, which is right almost everywhere — fix the handoffs it gets wrong here, zoomed in.
      </p>

      {!canEdit ? (
        <p className="helper-text">
          Load the audio first. Regions are placed against the measured speech spans, which are scanned automatically
          once a file is attached, so there is nothing to edit until then.
        </p>
      ) : null}

      <div className="regions-zoom">
        {VIEW_PRESETS.map((seconds) => (
          <button
            key={seconds}
            type="button"
            className={Math.abs(view.seconds - seconds) < 0.01 ? "is-active" : ""}
            onClick={() => setViewSeconds(seconds)}
          >
            {seconds}s
          </button>
        ))}
        <button
          type="button"
          className={Math.abs(view.seconds - totalDuration) < 0.01 ? "is-active" : ""}
          onClick={() => setView({ start: 0, seconds: totalDuration })}
        >
          All
        </button>
        <span className="metric-chip">
          {formatGutterClock(view.start)}–{formatGutterClock(Math.min(totalDuration, view.start + view.seconds))}
        </span>
      </div>

      <div className="regions-body" ref={bodyRef} tabIndex={0} onKeyDown={handleKeyDown}>
        <div className="regions-track" ref={trackRef}>
          <canvas
            ref={stripRef}
            className="regions-strip-canvas"
            onPointerDown={handleStripPointerDown}
            onPointerMove={handleStripPointerMove}
            onPointerUp={endDrag}
            onPointerCancel={endDrag}
          />

          {laneSpeakers.map((speaker) => {
            const stat = stats.get(speaker.id) ?? { count: 0, seconds: 0 };
            const soloed = soloSpeakerId === speaker.id;
            return (
              <div key={speaker.id} className={`regions-lane ${soloed ? "is-soloed" : ""}`}>
                <div className="regions-lane-header">
                  <span className="regions-lane-name">{speaker.name}</span>
                  <span className="regions-lane-stat">
                    {stat.count} · {formatGutterClock(stat.seconds)}
                  </span>
                  <button
                    type="button"
                    className={`regions-lane-solo ${soloed ? "is-active" : ""}`}
                    disabled={!soloableSpeakerIds.includes(speaker.id)}
                    onClick={() => onSoloSpeakerChange(soloed ? null : speaker.id)}
                  >
                    Solo
                  </button>
                  <button
                    type="button"
                    className="regions-lane-add"
                    title="Add a region at the playhead"
                    onClick={() => handleAddAtPlayhead(speaker.id)}
                  >
                    +
                  </button>
                </div>
                <RegionLane
                  viewport={viewport}
                  regions={displayed}
                  speakerId={speaker.id}
                  speechSpans={speechSpans}
                  overlapRegions={overlapRegions}
                  currentTime={currentTime}
                  selectedId={selectedId}
                  selectedEdge={selectedEdge}
                  pendingSpan={pending && pending.speakerId === speaker.id ? pending.span : null}
                  palette={palette}
                  onPointerDown={(event) => handleLanePointerDown(event, speaker.id)}
                  onPointerMove={handleLanePointerMove}
                  onPointerUp={handleLanePointerUp}
                />
              </div>
            );
          })}
        </div>
      </div>

      <p className="helper-text">
        Drag an edge to move it; it snaps to the nearest speech onset/offset and to the other speakers' edges. Hold Alt
        while dragging for a free edge. Drag empty lane space to add a region. ⌘/ctrl+scroll zooms, shift+scroll or
        shift+drag pans, and clicking the strip seeks.
      </p>

      {selected ? (
        <>
          <div className="panel-section-heading">
            <p className="eyebrow">Selected region</p>
            <h3>{formatClock(selected.start)} – {formatClock(selected.end)}</h3>
          </div>
          <div className="mastering-grid-2">
            <label>
              Start (s)
              <input
                type="number"
                step={0.001}
                min={0}
                value={startDraft ?? selected.start.toFixed(3)}
                onChange={(event) => setStartDraft(event.target.value)}
                onBlur={(event) => {
                  commitEdgeInput("start", event.target.value);
                  setStartDraft(null);
                }}
                onKeyDown={(event) => {
                  if (event.key === "Enter") {
                    commitEdgeInput("start", event.currentTarget.value);
                    setStartDraft(null);
                  }
                }}
              />
            </label>
            <label>
              End (s)
              <input
                type="number"
                step={0.001}
                min={0}
                value={endDraft ?? selected.end.toFixed(3)}
                onChange={(event) => setEndDraft(event.target.value)}
                onBlur={(event) => {
                  commitEdgeInput("end", event.target.value);
                  setEndDraft(null);
                }}
                onKeyDown={(event) => {
                  if (event.key === "Enter") {
                    commitEdgeInput("end", event.currentTarget.value);
                    setEndDraft(null);
                  }
                }}
              />
            </label>
          </div>
          <label>
            Speaker
            <select
              value={String(selected.speaker_id)}
              onChange={(event) =>
                commitRegions(reassignRegion(regions, selected.id, Number(event.target.value), duration || null))
              }
            >
              {laneSpeakers.map((speaker) => (
                <option key={speaker.id} value={speaker.id}>
                  {speaker.name}
                </option>
              ))}
            </select>
          </label>
          <div className="inline-actions">
            <button
              type="button"
              disabled={currentTime <= selected.start + MIN_REGION_S || currentTime >= selected.end - MIN_REGION_S}
              onClick={() => commitRegions(splitRegionAt(regions, selected.id, currentTime, duration || null))}
            >
              Split at playhead
            </button>
            <button
              type="button"
              onClick={() => {
                commitRegions(deleteRegion(regions, selected.id));
                setSelectedId(null);
                setSelectedEdge(null);
              }}
            >
              Delete region
            </button>
            <button type="button" onClick={() => onSeek(Math.max(0, selected.start - 0.4), { play: true })}>
              Listen
            </button>
          </div>
          <p className="helper-text">
            Arrow keys nudge the {selectedEdge ? `${selectedEdge} edge` : "whole region"} by 10 ms, with shift by
            100 ms. Click an edge handle first to nudge just that edge.
          </p>
        </>
      ) : (
        <p className="helper-text">Click a region bar to select it, or an edge handle to grab that edge.</p>
      )}

      <div className="chip-row">
        {laneSpeakers.map((speaker) => {
          const stat = stats.get(speaker.id) ?? { count: 0, seconds: 0 };
          return (
            <span key={speaker.id} className="metric-chip">
              {speaker.name}: {stat.count} region{stat.count === 1 ? "" : "s"}, {formatGutterClock(stat.seconds)}
            </span>
          );
        })}
      </div>

      {overrideActive ? (
        <p className="status-text">
          Manual edits — audio-derived regions are no longer being applied. Re-derive to go back to them.
        </p>
      ) : (
        <p className="helper-text">
          Regions are still derived from the audio. The first edit freezes them into a manual list.
        </p>
      )}
      <div className="inline-actions">
        <button
          type="button"
          disabled={!overrideActive}
          onClick={() => {
            setPreview(null);
            setSelectedId(null);
            setSelectedEdge(null);
            onReset();
          }}
        >
          Re-derive from audio
        </button>
      </div>
    </section>
  );
}

export default RegionsPanel;
