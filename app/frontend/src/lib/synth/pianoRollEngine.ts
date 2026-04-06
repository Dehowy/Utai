/** Canvas 2D rendering engine for the piano roll.
 *  Pure functions — no React dependency.
 *  Retro dark theme to match Utai's visual style.
 */

import type { UProject } from '../../types/ustx';
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
  drawKeyboard(rc);
  drawHeader(rc);
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

function drawNotes(rc: RenderContext) {
  const { ctx, width, height, project, viewState, selectedNotes, activePartIndex } = rc;
  const { scrollX, scrollY, pixelsPerTick, pixelsPerSemitone } = viewState;

  const startTick = Math.floor(scrollX / pixelsPerTick);
  const endTick = Math.ceil((scrollX + width - KEYBOARD_WIDTH) / pixelsPerTick);

  project.voice_parts.forEach((part, pi) => {
    const color = TRACK_COLORS[part.track_no % TRACK_COLORS.length];
    const isActivePart = pi === activePartIndex;

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

      // Note body
      ctx.fillStyle = isActivePart ? color : '#555570';
      ctx.globalAlpha = isActivePart ? 0.85 : 0.4;
      ctx.fillRect(x, y + 1, w - 1, h - 2);
      ctx.globalAlpha = 1;

      // Selection border
      if (isSelected) {
        ctx.strokeStyle = '#f0f0ff';
        ctx.lineWidth = 2;
        ctx.strokeRect(x, y + 1, w - 1, h - 2);
        ctx.strokeStyle = color;
        ctx.lineWidth = 1;
        ctx.strokeRect(x - 1, y, w + 1, h);
      }

      // Lyric text
      if (w > 15 && h > 10) {
        ctx.fillStyle = isSelected ? '#f0f0ff' : '#d0d0dc';
        ctx.font = `${Math.min(12, h - 4)}px 'DotGothic16', sans-serif`;
        ctx.textBaseline = 'middle';
        ctx.fillText(note.lyric, x + 3, y + h / 2, w - 6);
      }
    });
  });
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
