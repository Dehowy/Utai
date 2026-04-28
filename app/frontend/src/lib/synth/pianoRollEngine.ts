/** Canvas 2D rendering engine for the piano roll.
 *  Pure functions — no React dependency.
 *  Retro dark theme to match Utai's visual style.
 */

import type { UProject, UInstrumentPart } from '../../types/ustx';
import { RESOLUTION, midiToName, isBlackKey } from '../../types/ustx';

const KEYBOARD_WIDTH = 48;
const HEADER_HEIGHT = 24;

// Track colors — retro neon palette matching Utai
const TRACK_COLORS = ['#ff6b9d', '#5ce0d8', '#e8b84b', '#9b6dff', '#5dba5d', '#e8944b'];

export interface ViewState {
  scrollX: number;
  scrollY: number;
  pixelsPerTick: number;
  pixelsPerSemitone: number;
  snapTicks: number;
  trackLaneHeight: number;
}

export interface RenderContext {
  ctx: CanvasRenderingContext2D;
  width: number;
  height: number;
  project: UProject;
  viewState: ViewState;
  selectedNotes: Set<string>;
  activePartIndex: number;
  hoverMidi: number | null;
  playheadTicks?: number;
  hoverPlayhead?: boolean;
  instrumentPart?: UInstrumentPart | null;
  partEndTick?: number;
  pitchToolActive?: boolean;
  hoverPitchPoint?: { noteIndex: number; pointIndex: number } | null;
  ghostTracks?: number[];
  modelF0Active?: boolean;
}

/** Main render function — call on every frame. */
export function renderPianoRoll(rc: RenderContext) {
  const { ctx, width, height } = rc;

  // Fill entire background first to avoid black gaps
  ctx.fillStyle = '#0d0d1a';
  ctx.fillRect(0, 0, width, height);

  drawGrid(rc);
  drawHoverRow(rc);
  drawNotes(rc);
  drawPitchCurves(rc);
  drawPartBoundary(rc);
  drawKeyboard(rc);
  drawHeader(rc);
  drawPlayhead(rc);
}

function drawPlayhead(rc: RenderContext) {
  const { ctx, height, viewState } = rc;
  const ticks = rc.playheadTicks;
  if (ticks == null) return;
  const x = KEYBOARD_WIDTH + ticks * viewState.pixelsPerTick - viewState.scrollX;
  const hover = rc.hoverPlayhead;

  // Hover glow zone (wider translucent strip behind the line)
  if (hover && x >= KEYBOARD_WIDTH && x <= rc.width) {
    ctx.save();
    ctx.fillStyle = 'rgba(255, 107, 157, 0.08)';
    ctx.fillRect(x - 6, HEADER_HEIGHT, 12, height - HEADER_HEIGHT);
    ctx.restore();
  }

  // Playhead line
  if (x >= KEYBOARD_WIDTH && x <= rc.width) {
    ctx.save();
    ctx.strokeStyle = '#ff6b9d';
    ctx.lineWidth = hover ? 3 : 2;
    ctx.shadowColor = '#ff6b9d';
    ctx.shadowBlur = hover ? 10 : 6;
    ctx.beginPath();
    ctx.moveTo(x, HEADER_HEIGHT);
    ctx.lineTo(x, height);
    ctx.stroke();
    ctx.restore();
  }

  // Down-triangle on ruler header
  if (x >= KEYBOARD_WIDTH - 6 && x <= rc.width + 6) {
    const triSize = hover ? 7 : 5;
    ctx.save();
    ctx.fillStyle = '#ff6b9d';
    ctx.shadowColor = '#ff6b9d';
    ctx.shadowBlur = hover ? 8 : 4;
    ctx.beginPath();
    ctx.moveTo(x - triSize, 2);
    ctx.lineTo(x + triSize, 2);
    ctx.lineTo(x, HEADER_HEIGHT - 2);
    ctx.closePath();
    ctx.fill();
    ctx.restore();
  }
}

function drawGrid(rc: RenderContext) {
  const { ctx, width, height, viewState, project } = rc;
  const { scrollX, scrollY, pixelsPerTick, pixelsPerSemitone } = viewState;

  // Calculate visible MIDI range from scroll position
  const bottomMidi = Math.max(0, 127 - Math.floor((scrollY + height - HEADER_HEIGHT) / pixelsPerSemitone));
  const topMidi = Math.min(127, 127 - Math.floor(scrollY / pixelsPerSemitone) + 1);

  for (let midi = bottomMidi; midi <= topMidi; midi++) {
    const y = HEADER_HEIGHT + (127 - midi) * pixelsPerSemitone - scrollY;

    // Lane background — dark theme
    ctx.fillStyle = isBlackKey(midi) ? '#14142a' : '#181830';
    ctx.fillRect(KEYBOARD_WIDTH, y, width - KEYBOARD_WIDTH, pixelsPerSemitone);

    // Horizontal grid lines — C notes are brighter
    if (midi % 12 === 0) {
      ctx.strokeStyle = '#5a5a7a';
      ctx.lineWidth = 1.5;
    } else {
      ctx.strokeStyle = '#3a3a58';
      ctx.lineWidth = 0.5;
    }
    ctx.beginPath();
    ctx.moveTo(KEYBOARD_WIDTH, y + pixelsPerSemitone);
    ctx.lineTo(width, y + pixelsPerSemitone);
    ctx.stroke();
  }

  // Beat lines (vertical)
  const ts = project.time_signatures[0];
  const beatsPerBar = ts?.beat_per_bar ?? 4;
  const ticksPerBeat = RESOLUTION;
  const ticksPerBar = ticksPerBeat * beatsPerBar;

  const startTick = Math.max(0, Math.floor(scrollX / pixelsPerTick));
  const endTick = Math.ceil((scrollX + width - KEYBOARD_WIDTH) / pixelsPerTick);
  const firstBeat = Math.floor(startTick / ticksPerBeat) * ticksPerBeat;

  for (let tick = firstBeat; tick <= endTick; tick += ticksPerBeat) {
    const x = KEYBOARD_WIDTH + tick * pixelsPerTick - scrollX;
    if (x < KEYBOARD_WIDTH || x > width) continue;

    const isBar = tick % ticksPerBar === 0;
    ctx.strokeStyle = isBar ? '#5a5a7a' : '#3a3a58';
    ctx.lineWidth = isBar ? 1.5 : 0.5;
    ctx.beginPath();
    ctx.moveTo(x, HEADER_HEIGHT);
    ctx.lineTo(x, height);
    ctx.stroke();
  }
}

function drawHoverRow(rc: RenderContext) {
  const { ctx, width, viewState, hoverMidi } = rc;
  if (hoverMidi === null || hoverMidi < 0 || hoverMidi > 127) return;

  const { scrollY, pixelsPerSemitone } = viewState;
  const y = HEADER_HEIGHT + (127 - hoverMidi) * pixelsPerSemitone - scrollY;
  if (y + pixelsPerSemitone < HEADER_HEIGHT || y > rc.height) return;

  // Highlight the entire row with a subtle glow
  ctx.fillStyle = 'rgba(155, 109, 255, 0.12)';
  ctx.fillRect(KEYBOARD_WIDTH, y, width - KEYBOARD_WIDTH, pixelsPerSemitone);

  // Highlight the keyboard key
  ctx.fillStyle = 'rgba(155, 109, 255, 0.25)';
  ctx.fillRect(0, y, KEYBOARD_WIDTH, pixelsPerSemitone);
}

const NOTE_VALUE_TABLE: [number, string][] = [
  [RESOLUTION * 4, '1/1'],
  [RESOLUTION * 2, '1/2'],
  [RESOLUTION,     '1/4'],
  [RESOLUTION / 2, '1/8'],
  [RESOLUTION / 4, '1/16'],
  [RESOLUTION / 8, '1/32'],
  [Math.round(RESOLUTION * 4 / 3), '1/2T'],
  [Math.round(RESOLUTION * 2 / 3), '1/4T'],
  [Math.round(RESOLUTION / 3),     '1/8T'],
  [Math.round(RESOLUTION / 6),     '1/16T'],
  [Math.round(RESOLUTION / 12),    '1/32T'],
];

export function ticksToDurationLabel(ticks: number, snapTicks?: number): string {
  // Find the largest note value that fits
  const sorted = [...NOTE_VALUE_TABLE].sort((a, b) => b[0] - a[0]);
  let remaining = ticks;
  const parts: string[] = [];

  for (const [val, label] of sorted) {
    if (remaining >= val) {
      const count = Math.floor(remaining / val);
      parts.push(count > 1 ? `${count}×${label}` : label);
      remaining -= count * val;
    }
    if (remaining <= 0) break;
  }

  if (remaining > 0 && parts.length > 0) {
    parts.push(`+${remaining}t`);
  }

  return parts.length > 0 ? parts.join('+') : `${ticks}t`;
}

function interpolatePitchY(points: { x: number; y: number; shape: string }[], t: number): number {
  if (points.length === 0) return 0;
  if (t <= points[0].x) return points[0].y;
  if (t >= points[points.length - 1].x) return points[points.length - 1].y;
  for (let i = 0; i < points.length - 1; i++) {
    const p0 = points[i], p1 = points[i + 1];
    if (p0.x <= t && t <= p1.x) {
      const span = p1.x - p0.x;
      let frac = span > 0 ? (t - p0.x) / span : 0;
      const shape = p0.shape || 'io';
      if (shape === 'io') frac = 0.5 - 0.5 * Math.cos(frac * Math.PI);
      else if (shape === 'i') frac = 1 - Math.cos(frac * Math.PI / 2);
      else if (shape === 'o') frac = Math.sin(frac * Math.PI / 2);
      else if (shape === 'sp') frac = 0;
      return p0.y + (p1.y - p0.y) * frac;
    }
  }
  return 0;
}

function drawPitchCurves(rc: RenderContext) {
  const { ctx, width, height, project, viewState, activePartIndex } = rc;
  const { scrollX, scrollY, pixelsPerTick, pixelsPerSemitone } = viewState;
  if (rc.instrumentPart) return;
  if (rc.modelF0Active) return;

  const part = project.voice_parts[activePartIndex];
  if (!part) return;

  const startTick = Math.floor(scrollX / pixelsPerTick);
  const endTick = Math.ceil((scrollX + width - KEYBOARD_WIDTH) / pixelsPerTick);

  ctx.save();

  const bpm = project.tempos[0]?.bpm ?? 120;
  const ticksPerSec = (bpm * RESOLUTION) / 60;
  const isHoveringCurve = rc.hoverPitchPoint;

  for (let ni = 0; ni < part.notes.length; ni++) {
    const note = part.notes[ni];
    const absTick = part.position + note.position;
    const absEnd = absTick + note.duration;
    if (absEnd < startTick || absTick > endTick) continue;

    const pitchData = note.pitch?.data;
    const x0 = KEYBOARD_WIDTH + absTick * pixelsPerTick - scrollX;
    const xEnd = KEYBOARD_WIDTH + absEnd * pixelsPerTick - scrollX;
    const noteBaseY = HEADER_HEIGHT + (127 - note.tone) * pixelsPerSemitone - scrollY;
    const noteDurMs = (note.duration / ticksPerSec) * 1000;
    if (noteDurMs <= 0) continue;
    const basePitchY = noteBaseY + pixelsPerSemitone / 2;

    // Base pitch line (MIDI note center) — dashed in pitch mode, solid otherwise
    if (rc.pitchToolActive || (pitchData && pitchData.length > 0)) {
      ctx.strokeStyle = '#aaaacc';
      ctx.lineWidth = 1;
      ctx.globalAlpha = rc.pitchToolActive ? 0.6 : 0.3;
      if (rc.pitchToolActive) ctx.setLineDash([4, 4]);
      ctx.beginPath();
      ctx.moveTo(Math.max(x0, KEYBOARD_WIDTH), basePitchY);
      ctx.lineTo(Math.min(xEnd, width), basePitchY);
      ctx.stroke();
      if (rc.pitchToolActive) ctx.setLineDash([]);
    }

    if (!pitchData || pitchData.length === 0) continue;

    const noteHovered = isHoveringCurve?.noteIndex === ni;
    const isPitchMode = rc.pitchToolActive;

    // Line width scales slightly with vertical zoom (clamp 1.5–3.5)
    const baseWidth = Math.max(1.5, Math.min(3.5, pixelsPerSemitone / 10));
    ctx.strokeStyle = noteHovered && isPitchMode ? '#8fffd0' : '#5deba5';
    ctx.lineWidth = noteHovered && isPitchMode ? baseWidth + 1 : baseWidth;
    ctx.globalAlpha = isPitchMode ? (noteHovered ? 1.0 : 0.8) : 0.2;
    ctx.beginPath();
    let first = true;
    for (let px = Math.max(x0, KEYBOARD_WIDTH); px <= Math.min(xEnd, width); px += 2) {
      const frac = (px - x0) / (xEnd - x0);
      const ms = frac * noteDurMs;
      const yDev = interpolatePitchY(pitchData, ms);
      const screenY = basePitchY - (yDev / 10) * pixelsPerSemitone;
      if (first) { ctx.moveTo(px, screenY); first = false; }
      else ctx.lineTo(px, screenY);
    }
    ctx.stroke();

    // Draw control points
    ctx.globalAlpha = isPitchMode ? 1 : 0.2;
    for (let pi = 0; pi < pitchData.length; pi++) {
      const pt = pitchData[pi];
      const ptFrac = pt.x / noteDurMs;
      const ptX = x0 + ptFrac * (xEnd - x0);
      if (ptX < KEYBOARD_WIDTH || ptX > width) continue;
      const ptY = basePitchY - (pt.y / 10) * pixelsPerSemitone;
      const isHovered = rc.hoverPitchPoint?.noteIndex === ni && rc.hoverPitchPoint?.pointIndex === pi;
      ctx.fillStyle = isHovered ? '#ffffff' : '#5deba5';
      ctx.beginPath();
      ctx.arc(ptX, ptY, isHovered ? 5 : 3, 0, Math.PI * 2);
      ctx.fill();
    }
  }

  ctx.globalAlpha = 1;
  ctx.restore();
}

export interface PitchHitResult {
  partIndex: number; noteIndex: number; pointIndex: number;
  noteX0: number; noteXEnd: number; noteDurMs: number; baseMidi: number;
}

/** Hit test for pitch control points OR pitch curve line. pointIndex = -1 means on the curve but no point. */
export function hitTestPitchPoint(
  x: number, y: number, project: UProject, viewState: ViewState, activePartIndex: number,
): PitchHitResult | null {
  const { scrollX, scrollY, pixelsPerTick, pixelsPerSemitone } = viewState;
  const part = project.voice_parts[activePartIndex];
  if (!part) return null;

  const bpm = project.tempos[0]?.bpm ?? 120;
  const ticksPerSec = (bpm * RESOLUTION) / 60;
  const hitRadius = 12;

  for (let ni = 0; ni < part.notes.length; ni++) {
    const note = part.notes[ni];
    const pitchData = note.pitch?.data;
    if (!pitchData || pitchData.length === 0) continue;

    const absTick = part.position + note.position;
    const absEnd = absTick + note.duration;
    const x0 = KEYBOARD_WIDTH + absTick * pixelsPerTick - scrollX;
    const xEnd = KEYBOARD_WIDTH + absEnd * pixelsPerTick - scrollX;
    const noteBaseY = HEADER_HEIGHT + (127 - note.tone) * pixelsPerSemitone - scrollY;
    const noteDurMs = (note.duration / ticksPerSec) * 1000;
    if (noteDurMs <= 0) continue;

    // Check control points first (priority)
    for (let pi = 0; pi < pitchData.length; pi++) {
      const pt = pitchData[pi];
      const ptFrac = pt.x / noteDurMs;
      const ptX = x0 + ptFrac * (xEnd - x0);
      const ptY = noteBaseY + pixelsPerSemitone / 2 - (pt.y / 10) * pixelsPerSemitone;
      if (Math.abs(x - ptX) < hitRadius && Math.abs(y - ptY) < hitRadius) {
        return { partIndex: activePartIndex, noteIndex: ni, pointIndex: pi, noteX0: x0, noteXEnd: xEnd, noteDurMs, baseMidi: note.tone };
      }
    }

    // Check curve line proximity (pointIndex = -1)
    if (x >= x0 && x <= xEnd) {
      const frac = (x - x0) / (xEnd - x0);
      const ms = frac * noteDurMs;
      const yDev = interpolatePitchY(pitchData, ms);
      const curveY = noteBaseY + pixelsPerSemitone / 2 - (yDev / 10) * pixelsPerSemitone;
      if (Math.abs(y - curveY) < hitRadius) {
        return { partIndex: activePartIndex, noteIndex: ni, pointIndex: -1, noteX0: x0, noteXEnd: xEnd, noteDurMs, baseMidi: note.tone };
      }
    }
  }
  return null;
}

function drawPartBoundary(rc: RenderContext) {
  if (rc.partEndTick == null) return;
  const { ctx, height, viewState } = rc;
  const x = KEYBOARD_WIDTH + rc.partEndTick * viewState.pixelsPerTick - viewState.scrollX;
  if (x < KEYBOARD_WIDTH || x > rc.width) return;

  // Dim overlay beyond boundary
  ctx.fillStyle = 'rgba(0, 0, 0, 0.4)';
  ctx.fillRect(x, HEADER_HEIGHT, rc.width - x, height - HEADER_HEIGHT);

  // Boundary line
  ctx.save();
  ctx.strokeStyle = '#ff6b9d';
  ctx.lineWidth = 2;
  ctx.setLineDash([6, 4]);
  ctx.beginPath();
  ctx.moveTo(x, HEADER_HEIGHT);
  ctx.lineTo(x, height);
  ctx.stroke();
  ctx.setLineDash([]);
  ctx.restore();
}

function drawNotes(rc: RenderContext) {
  const { ctx, width, height, project, viewState, selectedNotes, activePartIndex } = rc;
  const { scrollX, scrollY, pixelsPerTick, pixelsPerSemitone } = viewState;

  const startTick = Math.floor(scrollX / pixelsPerTick);
  const endTick = Math.ceil((scrollX + width - KEYBOARD_WIDTH) / pixelsPerTick);

  const instMode = !!rc.instrumentPart;
  const ghostSet = new Set(rc.ghostTracks || []);

  // Determine active track index
  const activeTrackNo = instMode
    ? rc.instrumentPart!.track_no
    : (project.voice_parts[activePartIndex]?.track_no ?? -1);

  // Ghost: draw voice parts belonging to ghost tracks
  for (const vp of project.voice_parts) {
    if (vp.track_no === activeTrackNo) continue;
    if (!ghostSet.has(vp.track_no)) continue;
    const color = TRACK_COLORS[vp.track_no % TRACK_COLORS.length];
    for (const note of vp.notes) {
      const absPos = vp.position + note.position;
      if (absPos + note.duration < startTick || absPos > endTick) continue;
      const x = KEYBOARD_WIDTH + absPos * pixelsPerTick - scrollX;
      const y = HEADER_HEIGHT + (127 - note.tone) * pixelsPerSemitone - scrollY;
      const w = note.duration * pixelsPerTick;
      const h = pixelsPerSemitone;
      if (y + h < HEADER_HEIGHT || y > height) continue;
      ctx.fillStyle = color;
      ctx.globalAlpha = 0.2;
      ctx.fillRect(x, y + 1, w - 1, h - 2);
      ctx.globalAlpha = 1;
    }
  }

  // Ghost: draw instrument parts belonging to ghost tracks
  for (const ip of project.instrument_parts) {
    if (ip.track_no === activeTrackNo) continue;
    if (!ghostSet.has(ip.track_no)) continue;
    const color = TRACK_COLORS[ip.track_no % TRACK_COLORS.length];
    for (const note of ip.notes) {
      const absPos = ip.position + note.position;
      if (absPos + note.duration < startTick || absPos > endTick) continue;
      const x = KEYBOARD_WIDTH + absPos * pixelsPerTick - scrollX;
      const y = HEADER_HEIGHT + (127 - note.tone) * pixelsPerSemitone - scrollY;
      const w = note.duration * pixelsPerTick;
      const h = pixelsPerSemitone;
      if (y + h < HEADER_HEIGHT || y > height) continue;
      ctx.fillStyle = color;
      ctx.globalAlpha = 0.2;
      ctx.fillRect(x, y + 1, w - 1, h - 2);
      ctx.globalAlpha = 1;
    }
  }

  // Active voice parts
  project.voice_parts.forEach((part, pi) => {
    const isActivePart = !instMode && pi === activePartIndex;
    if (!isActivePart) return;

    const color = TRACK_COLORS[part.track_no % TRACK_COLORS.length];
    part.notes.forEach((note, ni) => {
      const absPos = part.position + note.position;
      const absEnd = absPos + note.duration;

      if (absEnd < startTick || absPos > endTick) return;

      const x = KEYBOARD_WIDTH + absPos * pixelsPerTick - scrollX;
      const y = HEADER_HEIGHT + (127 - note.tone) * pixelsPerSemitone - scrollY;
      const w = note.duration * pixelsPerTick;
      const h = pixelsPerSemitone;

      if (y + h < HEADER_HEIGHT || y > height) return;

      const noteId = `${pi}-${ni}`;
      const isSelected = selectedNotes.has(noteId);

      ctx.fillStyle = color;
      ctx.globalAlpha = 0.85;
      ctx.fillRect(x, y + 1, w - 1, h - 2);
      ctx.globalAlpha = 1;

      if (isSelected) {
        ctx.strokeStyle = '#f0f0ff';
        ctx.lineWidth = 2;
        ctx.strokeRect(x, y + 1, w - 1, h - 2);
        ctx.strokeStyle = color;
        ctx.lineWidth = 1;
        ctx.strokeRect(x - 1, y, w + 1, h);
      }

      if (w > 15 && h > 10) {
        ctx.fillStyle = isSelected ? '#f0f0ff' : '#d0d0dc';
        ctx.font = `${Math.min(12, h - 4)}px 'DotGothic16', sans-serif`;
        ctx.textBaseline = 'middle';
        ctx.fillText(note.lyric, x + 3, y + h / 2, w - 6);
      }

      // Duration label
      if (note.duration > viewState.snapTicks && w > 40 && h > 8) {
        const label = ticksToDurationLabel(note.duration, viewState.snapTicks);
        ctx.fillStyle = 'rgba(200, 200, 220, 0.5)';
        ctx.font = `${Math.min(9, h - 6)}px 'DotGothic16', sans-serif`;
        ctx.textAlign = 'right';
        ctx.textBaseline = 'middle';
        ctx.fillText(label, x + w - 3, y + h / 2);
        ctx.textAlign = 'left';
      }
    });
  });

  // Instrument notes (when an instrument part is active)
  const instPart = rc.instrumentPart;
  if (instPart) {
    const color = '#e8944b'; // orange for instrument notes
    instPart.notes.forEach((note, ni) => {
      const absPos = instPart.position + note.position;
      const absEnd = absPos + note.duration;
      if (absEnd < startTick || absPos > endTick) return;
      const x = KEYBOARD_WIDTH + absPos * pixelsPerTick - scrollX;
      const y = HEADER_HEIGHT + (127 - note.tone) * pixelsPerSemitone - scrollY;
      const w = note.duration * pixelsPerTick;
      const h = pixelsPerSemitone;
      if (y + h < HEADER_HEIGHT || y > height) return;

      const noteId = `i0-${ni}`;
      const isSelected = selectedNotes.has(noteId);

      // Note body — brightness by velocity
      const velAlpha = 0.5 + (note.velocity / 127) * 0.4;
      ctx.fillStyle = color;
      ctx.globalAlpha = velAlpha;
      ctx.fillRect(x, y + 1, w - 1, h - 2);
      ctx.globalAlpha = 1;

      if (isSelected) {
        ctx.strokeStyle = '#f0f0ff';
        ctx.lineWidth = 2;
        ctx.strokeRect(x, y + 1, w - 1, h - 2);
        ctx.strokeStyle = color;
        ctx.lineWidth = 1;
        ctx.strokeRect(x - 1, y, w + 1, h);
      }

      // Show velocity as small text
      if (w > 20 && h > 10) {
        ctx.fillStyle = '#d0d0dc';
        ctx.font = `${Math.min(10, h - 4)}px 'DotGothic16', sans-serif`;
        ctx.textBaseline = 'middle';
        ctx.fillText(`v${note.velocity}`, x + 3, y + h / 2, w - 6);
      }
    });
  }
}

function drawKeyboard(rc: RenderContext) {
  const { ctx, height, viewState } = rc;
  const { scrollY, pixelsPerSemitone } = viewState;

  // Background
  ctx.fillStyle = '#111125';
  ctx.fillRect(0, HEADER_HEIGHT, KEYBOARD_WIDTH, height - HEADER_HEIGHT);

  for (let midi = 0; midi <= 127; midi++) {
    const y = HEADER_HEIGHT + (127 - midi) * pixelsPerSemitone - scrollY;
    if (y + pixelsPerSemitone < HEADER_HEIGHT || y > height) continue;

    const black = isBlackKey(midi);

    ctx.fillStyle = black ? '#1e1e3a' : '#2a2a45';
    ctx.fillRect(0, y, black ? KEYBOARD_WIDTH * 0.65 : KEYBOARD_WIDTH - 1, pixelsPerSemitone);

    ctx.strokeStyle = '#3a3a5a';
    ctx.lineWidth = 0.5;
    ctx.beginPath();
    ctx.moveTo(0, y + pixelsPerSemitone);
    ctx.lineTo(KEYBOARD_WIDTH, y + pixelsPerSemitone);
    ctx.stroke();

    // Note name on C keys
    if (midi % 12 === 0 && pixelsPerSemitone > 8) {
      ctx.fillStyle = '#8888a0';
      ctx.font = "10px 'VT323', monospace";
      ctx.textBaseline = 'middle';
      ctx.fillText(midiToName(midi), 2, y + pixelsPerSemitone / 2);
    }
  }

  // Right border
  ctx.strokeStyle = '#4a4a6a';
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(KEYBOARD_WIDTH, HEADER_HEIGHT);
  ctx.lineTo(KEYBOARD_WIDTH, height);
  ctx.stroke();
}

function drawHeader(rc: RenderContext) {
  const { ctx, width, viewState, project } = rc;
  const { scrollX, pixelsPerTick } = viewState;

  ctx.fillStyle = '#111125';
  ctx.fillRect(0, 0, width, HEADER_HEIGHT);

  const ts = project.time_signatures[0];
  const beatsPerBar = ts?.beat_per_bar ?? 4;
  const ticksPerBar = RESOLUTION * beatsPerBar;

  const startTick = Math.floor(scrollX / pixelsPerTick);
  const endTick = Math.ceil((scrollX + width - KEYBOARD_WIDTH) / pixelsPerTick);
  const firstBar = Math.floor(startTick / ticksPerBar);

  ctx.fillStyle = '#8888a0';
  ctx.font = "11px 'VT323', monospace";
  ctx.textBaseline = 'middle';

  for (let bar = firstBar; bar * ticksPerBar <= endTick; bar++) {
    const x = KEYBOARD_WIDTH + bar * ticksPerBar * pixelsPerTick - scrollX;
    if (x < KEYBOARD_WIDTH - 20 || x > width) continue;
    ctx.fillText(`${bar + 1}`, x + 4, HEADER_HEIGHT / 2);
  }

  // Keyboard header corner
  ctx.fillStyle = '#0d0d1a';
  ctx.fillRect(0, 0, KEYBOARD_WIDTH, HEADER_HEIGHT);

  // BPM display
  const bpm = project.tempos[0]?.bpm ?? 120;
  ctx.fillStyle = '#e8b84b';
  ctx.font = "bold 10px 'VT323', monospace";
  ctx.fillText(`${bpm}`, 4, HEADER_HEIGHT / 2);

  // Bottom border
  ctx.strokeStyle = '#4a4a6a';
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(0, HEADER_HEIGHT);
  ctx.lineTo(width, HEADER_HEIGHT);
  ctx.stroke();
}

// --- Coordinate conversion ---
export const KEYBOARD_W = KEYBOARD_WIDTH;
export const HEADER_H = HEADER_HEIGHT;

export function screenToTick(x: number, viewState: ViewState): number {
  return (x - KEYBOARD_WIDTH + viewState.scrollX) / viewState.pixelsPerTick;
}

export function screenToMidi(y: number, viewState: ViewState): number {
  return 127 - (y - HEADER_HEIGHT + viewState.scrollY) / viewState.pixelsPerSemitone;
}

export function tickToScreen(tick: number, viewState: ViewState): number {
  return KEYBOARD_WIDTH + tick * viewState.pixelsPerTick - viewState.scrollX;
}

export function midiToScreen(midi: number, viewState: ViewState): number {
  return HEADER_HEIGHT + (127 - midi) * viewState.pixelsPerSemitone - viewState.scrollY;
}

export function snapTick(tick: number, snapTicks: number): number {
  return Math.round(tick / snapTicks) * snapTicks;
}

/** Snap to the grid line at or before the given tick (for note placement). */
export function snapTickFloor(tick: number, snapTicks: number): number {
  return Math.floor(tick / snapTicks) * snapTicks;
}
