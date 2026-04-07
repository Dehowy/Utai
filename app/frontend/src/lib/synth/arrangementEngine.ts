/** Canvas 2D rendering engine for the arrangement (track overview) view.
 *  Pure functions — no React dependency.
 */

import type { UProject, UVoicePart, UWavePart } from '../../types/ustx';
import { RESOLUTION, midiToName } from '../../types/ustx';
import type { ViewState } from './pianoRollEngine';

export const TRACK_HEADER_W = 140;
export const TRACK_LANE_H = 64;
export const RULER_H = 24;

const TRACK_COLORS = ['#ff6b9d', '#5ce0d8', '#e8b84b', '#9b6dff', '#5dba5d', '#e8944b'];

export interface ArrangementRenderContext {
  ctx: CanvasRenderingContext2D;
  width: number;
  height: number;
  project: UProject;
  viewState: ViewState;
  activePartIndex: number;
  playheadTicks?: number;
  peakData?: Map<string, Float32Array>;
  selectedSegment?: { type: 'voice' | 'instrument'; index: number } | null;
}

export function renderArrangement(rc: ArrangementRenderContext) {
  const { ctx, width, height, project, viewState } = rc;

  ctx.fillStyle = '#0d0d1a';
  ctx.fillRect(0, 0, width, height);

  drawRuler(rc);
  drawTrackLanes(rc);
  drawVoiceParts(rc);
  drawInstrumentParts(rc);
  drawWaveParts(rc);
  drawTrackHeaders(rc);
  drawPlayhead(rc); // after headers so triangle is visible on the ruler
}

function drawRuler(rc: ArrangementRenderContext) {
  const { ctx, width, viewState, project } = rc;
  const ts = project.time_signatures[0];
  const beatsPerBar = ts?.beat_per_bar ?? 4;
  const ticksPerBar = RESOLUTION * beatsPerBar;
  const ppt = viewState.pixelsPerTick;

  ctx.fillStyle = '#111125';
  ctx.fillRect(0, 0, width, RULER_H);

  const startTick = Math.floor(viewState.scrollX / ppt);
  const endTick = startTick + Math.ceil(width / ppt);
  const firstBar = Math.floor(startTick / ticksPerBar);
  const lastBar = Math.ceil(endTick / ticksPerBar);

  ctx.font = "10px 'DotGothic16', sans-serif";
  ctx.textBaseline = 'top';

  for (let bar = firstBar; bar <= lastBar; bar++) {
    const tick = bar * ticksPerBar;
    const x = TRACK_HEADER_W + tick * ppt - viewState.scrollX;
    if (x < TRACK_HEADER_W - 20 || x > rc.width + 20) continue;

    ctx.strokeStyle = '#3a3a58';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(x, 0);
    ctx.lineTo(x, RULER_H);
    ctx.stroke();

    ctx.fillStyle = '#888';
    ctx.fillText(`${bar + 1}`, x + 3, 4);
  }

  // Bottom border
  ctx.strokeStyle = '#3a3a58';
  ctx.beginPath();
  ctx.moveTo(0, RULER_H - 0.5);
  ctx.lineTo(width, RULER_H - 0.5);
  ctx.stroke();
}

function drawTrackLanes(rc: ArrangementRenderContext) {
  const { ctx, width, project, viewState } = rc;
  const laneH = viewState.trackLaneHeight ?? TRACK_LANE_H;

  for (let i = 0; i < project.tracks.length; i++) {
    const y = RULER_H + i * laneH;
    // Lane background (alternating)
    ctx.fillStyle = i % 2 === 0 ? '#0f0f22' : '#121228';
    ctx.fillRect(TRACK_HEADER_W, y, width - TRACK_HEADER_W, laneH);

    // Bottom border
    ctx.strokeStyle = '#1e1e3a';
    ctx.beginPath();
    ctx.moveTo(TRACK_HEADER_W, y + laneH - 0.5);
    ctx.lineTo(width, y + laneH - 0.5);
    ctx.stroke();
  }
}

function drawTrackHeaders(rc: ArrangementRenderContext) {
  const { ctx, project, viewState } = rc;
  const laneH = viewState.trackLaneHeight ?? TRACK_LANE_H;

  // Header panel background
  ctx.fillStyle = '#111125';
  ctx.fillRect(0, 0, TRACK_HEADER_W, rc.height);

  // Right border
  ctx.strokeStyle = '#3a3a58';
  ctx.beginPath();
  ctx.moveTo(TRACK_HEADER_W - 0.5, 0);
  ctx.lineTo(TRACK_HEADER_W - 0.5, rc.height);
  ctx.stroke();

  for (let i = 0; i < project.tracks.length; i++) {
    const track = project.tracks[i];
    const y = RULER_H + i * laneH;
    const color = TRACK_COLORS[i % TRACK_COLORS.length];

    // Color stripe
    ctx.fillStyle = color;
    ctx.fillRect(0, y, 4, laneH);

    // Track name
    ctx.font = "11px 'DotGothic16', sans-serif";
    ctx.fillStyle = '#d0d0e8';
    ctx.textBaseline = 'top';
    ctx.fillText(track.track_name, 10, y + 6, TRACK_HEADER_W - 16);

    // Track type label — hide when lane is too short
    if (laneH >= 50) {
      ctx.font = "9px 'DotGothic16', sans-serif";
      ctx.fillStyle = '#6a6a8a';
      const tt = track.track_type || 'vocal';
      const typeLabel = tt === 'vocal' ? (track.singer || 'Vocal')
        : tt === 'instrument' ? (track.plugin_name || 'Instrument')
        : 'Audio';
      ctx.fillText(typeLabel, 10, y + 20, TRACK_HEADER_W - 16);
    }

    // Mute/Solo indicators
    if (track.mute) {
      ctx.fillStyle = '#ff4444';
      ctx.font = "bold 9px sans-serif";
      ctx.fillText('M', TRACK_HEADER_W - 30, y + 6);
    }
    if (track.solo) {
      ctx.fillStyle = '#ffcc00';
      ctx.font = "bold 9px sans-serif";
      ctx.fillText('S', TRACK_HEADER_W - 16, y + 6);
    }

    // Volume bar — proportional position, hide when too short
    if (laneH >= 48) {
      const volY = y + Math.round(laneH * 0.6);
      const volFrac = Math.min(1, Math.max(0, (track.volume + 24) / 24));
      ctx.fillStyle = 'rgba(155, 109, 255, 0.3)';
      ctx.fillRect(10, volY, TRACK_HEADER_W - 24, 6);
      ctx.fillStyle = color;
      ctx.fillRect(10, volY, (TRACK_HEADER_W - 24) * volFrac, 6);
    }

    // Bottom border
    ctx.strokeStyle = '#1e1e3a';
    ctx.beginPath();
    ctx.moveTo(0, y + laneH - 0.5);
    ctx.lineTo(TRACK_HEADER_W, y + laneH - 0.5);
    ctx.stroke();
  }
}

function drawVoiceParts(rc: ArrangementRenderContext) {
  const { ctx, project, viewState, activePartIndex } = rc;
  const ppt = viewState.pixelsPerTick;
  const laneH = viewState.trackLaneHeight ?? TRACK_LANE_H;

  project.voice_parts.forEach((vp, vpIdx) => {
    const track = project.tracks[vp.track_no];
    if (!track) return;
    const color = TRACK_COLORS[vp.track_no % TRACK_COLORS.length];
    const y = RULER_H + vp.track_no * laneH;
    const x = TRACK_HEADER_W + vp.position * ppt - viewState.scrollX;
    const w = vp.duration * ppt;

    if (x + w < TRACK_HEADER_W || x > rc.width) return;

    const isActive = vpIdx === activePartIndex;
    const isSelected = rc.selectedSegment?.type === 'voice' && rc.selectedSegment.index === vpIdx;

    // Part block
    ctx.fillStyle = color;
    ctx.globalAlpha = isSelected ? 0.45 : isActive ? 0.35 : 0.2;
    ctx.fillRect(x, y + 4, w, laneH - 8);
    ctx.globalAlpha = 1;

    // Border
    ctx.strokeStyle = isSelected ? '#f0f0ff' : color;
    ctx.lineWidth = isSelected ? 2 : isActive ? 2 : 1;
    ctx.strokeRect(x, y + 4, w, laneH - 8);

    // Mini note visualization
    if (vp.notes.length > 0) {
      const minTone = Math.min(...vp.notes.map(n => n.tone));
      const maxTone = Math.max(...vp.notes.map(n => n.tone));
      const toneRange = Math.max(1, maxTone - minTone);
      const noteAreaH = laneH - 16;

      ctx.fillStyle = color;
      ctx.globalAlpha = 0.7;
      for (const note of vp.notes) {
        const nx = x + note.position * ppt;
        const nw = Math.max(1, note.duration * ppt);
        const ny = y + 8 + (1 - (note.tone - minTone) / toneRange) * (noteAreaH - 4);
        const nh = Math.max(1, 3);
        if (nx + nw < TRACK_HEADER_W || nx > rc.width) continue;
        ctx.fillRect(nx, ny, nw, nh);
      }
      ctx.globalAlpha = 1;
    }

    // Part name
    ctx.font = "9px 'DotGothic16', sans-serif";
    ctx.fillStyle = '#e0e0ff';
    ctx.textBaseline = 'top';
    ctx.fillText(vp.name, x + 4, y + 6, w - 8);
  });
}

function drawInstrumentParts(rc: ArrangementRenderContext) {
  const { ctx, project, viewState } = rc;
  const ppt = viewState.pixelsPerTick;
  const laneH = viewState.trackLaneHeight ?? TRACK_LANE_H;

  project.instrument_parts.forEach((ip) => {
    const track = project.tracks[ip.track_no];
    if (!track) return;
    const color = '#e8944b'; // orange for instrument tracks
    const y = RULER_H + ip.track_no * laneH;
    const x = TRACK_HEADER_W + ip.position * ppt - viewState.scrollX;
    const w = ip.duration * ppt;

    if (x + w < TRACK_HEADER_W || x > rc.width) return;

    // Part block
    ctx.fillStyle = color;
    ctx.globalAlpha = 0.25;
    ctx.fillRect(x, y + 4, w, laneH - 8);
    ctx.globalAlpha = 1;
    ctx.strokeStyle = color;
    ctx.lineWidth = 1;
    ctx.strokeRect(x, y + 4, w, laneH - 8);

    // Mini note density visualization
    if (ip.notes.length > 0) {
      const minTone = Math.min(...ip.notes.map(n => n.tone));
      const maxTone = Math.max(...ip.notes.map(n => n.tone));
      const toneRange = Math.max(1, maxTone - minTone);
      const noteAreaH = laneH - 16;

      ctx.fillStyle = color;
      ctx.globalAlpha = 0.7;
      for (const note of ip.notes) {
        const nx = x + note.position * ppt;
        const nw = Math.max(1, note.duration * ppt);
        const ny = y + 8 + (1 - (note.tone - minTone) / toneRange) * (noteAreaH - 4);
        if (nx + nw < TRACK_HEADER_W || nx > rc.width) continue;
        ctx.fillRect(nx, ny, nw, Math.max(1, 3));
      }
      ctx.globalAlpha = 1;
    }

    // Part name
    ctx.font = "9px 'DotGothic16', sans-serif";
    ctx.fillStyle = '#e0e0ff';
    ctx.textBaseline = 'top';
    ctx.fillText(ip.name, x + 4, y + 6, w - 8);
  });
}

function drawWaveParts(rc: ArrangementRenderContext) {
  const { ctx, project, viewState, peakData } = rc;
  const ppt = viewState.pixelsPerTick;
  const laneH = viewState.trackLaneHeight ?? TRACK_LANE_H;

  project.wave_parts.forEach((wp) => {
    const track = project.tracks[wp.track_no];
    if (!track) return;
    const color = TRACK_COLORS[wp.track_no % TRACK_COLORS.length];
    const y = RULER_H + wp.track_no * laneH;

    // Convert duration from ms to ticks (approximate using first tempo)
    const bpm = project.tempos[0]?.bpm ?? 120;
    const durationTicks = (wp.file_duration_ms / 1000) * (bpm / 60) * RESOLUTION;
    const x = TRACK_HEADER_W + wp.position * ppt - viewState.scrollX;
    const w = durationTicks * ppt;

    if (x + w < TRACK_HEADER_W || x > rc.width) return;

    // Part block
    ctx.fillStyle = color;
    ctx.globalAlpha = 0.15;
    ctx.fillRect(x, y + 4, w, laneH - 8);
    ctx.globalAlpha = 1;

    ctx.strokeStyle = color;
    ctx.lineWidth = 1;
    ctx.strokeRect(x, y + 4, w, laneH - 8);

    // Waveform thumbnail from peak data
    const peaks = peakData?.get(wp.buffer_id || '');
    if (peaks && peaks.length > 0) {
      const waveH = laneH - 16;
      const midY = y + 8 + waveH / 2;
      ctx.strokeStyle = color;
      ctx.globalAlpha = 0.6;
      ctx.lineWidth = 1;
      ctx.beginPath();
      for (let i = 0; i < Math.min(peaks.length, Math.floor(w)); i++) {
        const px = x + (i / peaks.length) * w;
        const amp = peaks[i] * (waveH / 2);
        if (i === 0) ctx.moveTo(px, midY - amp);
        else ctx.lineTo(px, midY - amp);
      }
      ctx.stroke();
      ctx.globalAlpha = 1;
    }

    // Part name
    ctx.font = "9px 'DotGothic16', sans-serif";
    ctx.fillStyle = '#e0e0ff';
    ctx.textBaseline = 'top';
    ctx.fillText(wp.name, x + 4, y + 6, w - 8);
  });
}

function drawPlayhead(rc: ArrangementRenderContext) {
  const ticks = rc.playheadTicks;
  if (ticks == null) return;
  const { ctx, viewState } = rc;
  const x = TRACK_HEADER_W + ticks * viewState.pixelsPerTick - viewState.scrollX;

  // Playhead line
  if (x >= TRACK_HEADER_W && x <= rc.width) {
    ctx.save();
    ctx.strokeStyle = '#ff6b9d';
    ctx.lineWidth = 2;
    ctx.shadowColor = '#ff6b9d';
    ctx.shadowBlur = 6;
    ctx.beginPath();
    ctx.moveTo(x, 0);
    ctx.lineTo(x, rc.height);
    ctx.stroke();
    ctx.restore();
  }

  // Down-triangle on ruler
  if (x >= TRACK_HEADER_W - 6 && x <= rc.width + 6) {
    ctx.save();
    ctx.fillStyle = '#ff6b9d';
    ctx.shadowColor = '#ff6b9d';
    ctx.shadowBlur = 4;
    ctx.beginPath();
    ctx.moveTo(x - 5, 2);
    ctx.lineTo(x + 5, 2);
    ctx.lineTo(x, RULER_H - 2);
    ctx.closePath();
    ctx.fill();
    ctx.restore();
  }
}
