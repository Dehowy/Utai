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

export interface DragOverride {
  type: 'voice' | 'instrument' | 'wave';
  index: number;
  position?: number;
  trackNo?: number;
  skip?: number;
  duration?: number;
  fileDurationMs?: number;
  notes?: any[];
}

export interface ArrangementRenderContext {
  ctx: CanvasRenderingContext2D;
  width: number;
  height: number;
  project: UProject;
  viewState: ViewState;
  activePartIndex: number;
  playheadTicks?: number;
  peakData?: Map<string, Float32Array>;
  selectedSegment?: { type: 'voice' | 'instrument' | 'wave'; index: number } | null;
  selectedTrackIndex?: number;
  arrScrollY?: number;
  playheadHover?: boolean;
  dragOverride?: DragOverride | null;
}

export function renderArrangement(rc: ArrangementRenderContext) {
  const { ctx, width, height, project, viewState } = rc;

  ctx.fillStyle = '#0d0d1a';
  ctx.fillRect(0, 0, width, height);

  drawRuler(rc);

  // Clip track content to below ruler
  ctx.save();
  ctx.beginPath();
  ctx.rect(0, RULER_H, width, height - RULER_H);
  ctx.clip();

  drawTrackLanes(rc);
  drawVoiceParts(rc);
  drawInstrumentParts(rc);
  drawWaveParts(rc);
  drawTrackHeaders(rc);

  ctx.restore();

  // Redraw ruler on top to cover any overflow
  drawRuler(rc);
  drawPlayhead(rc);
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

  const arrScrollY = rc.arrScrollY ?? 0;

  for (let i = 0; i < project.tracks.length; i++) {
    const y = RULER_H + i * laneH - arrScrollY;
    if (y + laneH < RULER_H || y > rc.height) continue;
    const isSelected = i === rc.selectedTrackIndex;
    ctx.fillStyle = isSelected ? '#1a1a3a' : (i % 2 === 0 ? '#0f0f22' : '#121228');
    ctx.fillRect(TRACK_HEADER_W, Math.max(RULER_H, y), width - TRACK_HEADER_W, laneH);

    // Selected track left accent bar
    if (isSelected) {
      const color = TRACK_COLORS[i % TRACK_COLORS.length];
      ctx.fillStyle = color;
      ctx.fillRect(TRACK_HEADER_W, y, 2, laneH);
    }

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

  const hdrScrollY = rc.arrScrollY ?? 0;
  for (let i = 0; i < project.tracks.length; i++) {
    const track = project.tracks[i];
    const y = RULER_H + i * laneH - hdrScrollY;
    if (y + laneH < RULER_H || y > rc.height) continue;
    const color = TRACK_COLORS[i % TRACK_COLORS.length];

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
    const y = RULER_H + vp.track_no * laneH - (rc.arrScrollY ?? 0);
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

    // Clip name + vol/pan label
    ctx.font = "9px 'DotGothic16', sans-serif";
    ctx.fillStyle = '#e0e0ff';
    ctx.textBaseline = 'top';
    const vpLabel = _clipLabel(vp.name, vp.volume, vp.pan);
    ctx.fillText(vpLabel, x + 4, y + 6, w - 8);
  });
}

function _clipLabel(name: string, vol?: number, pan?: number): string {
  const v = vol ?? 0;
  const p = pan ?? 0;
  if (v === 0 && p === 0) return name;
  const parts = [name];
  if (v !== 0) parts.push(v <= -24 ? '-∞' : `${v > 0 ? '+' : ''}${v.toFixed(0)}dB`);
  if (p !== 0) parts.push(p < 0 ? `L${-p}` : `R${p}`);
  return parts.join('  ');
}

function drawInstrumentParts(rc: ArrangementRenderContext) {
  const { ctx, project, viewState } = rc;
  const ppt = viewState.pixelsPerTick;
  const laneH = viewState.trackLaneHeight ?? TRACK_LANE_H;

  project.instrument_parts.forEach((ip, ipIdx) => {
    const track = project.tracks[ip.track_no];
    if (!track) return;
    const color = '#e8944b';
    const y = RULER_H + ip.track_no * laneH - (rc.arrScrollY ?? 0);
    const x = TRACK_HEADER_W + ip.position * ppt - viewState.scrollX;
    const w = ip.duration * ppt;

    if (x + w < TRACK_HEADER_W || x > rc.width) return;

    const isSelected = rc.selectedSegment?.type === 'instrument' && rc.selectedSegment.index === ipIdx;

    ctx.fillStyle = color;
    ctx.globalAlpha = isSelected ? 0.45 : 0.25;
    ctx.fillRect(x, y + 4, w, laneH - 8);
    ctx.globalAlpha = 1;
    ctx.strokeStyle = isSelected ? '#f0f0ff' : color;
    ctx.lineWidth = isSelected ? 2 : 1;
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
    ctx.fillText(_clipLabel(ip.name, ip.volume, ip.pan), x + 4, y + 6, w - 8);
  });
}

function drawWaveParts(rc: ArrangementRenderContext) {
  const { ctx, project, viewState, peakData } = rc;
  const ppt = viewState.pixelsPerTick;
  const laneH = viewState.trackLaneHeight ?? TRACK_LANE_H;

  project.wave_parts.forEach((wp, wpIdx) => {
    const ov = rc.dragOverride?.type === 'wave' && rc.dragOverride.index === wpIdx ? rc.dragOverride : null;
    const trackNo = ov?.trackNo ?? wp.track_no;
    const wPos = ov?.position ?? wp.position;
    const wSkip = ov?.skip ?? wp.skip;
    const wDurMs = ov?.fileDurationMs ?? wp.file_duration_ms;
    const track = project.tracks[trackNo];
    if (!track) return;
    const color = TRACK_COLORS[trackNo % TRACK_COLORS.length];
    const y = RULER_H + trackNo * laneH - (rc.arrScrollY ?? 0);

    const bpm = project.tempos[0]?.bpm ?? 120;
    const durationTicks = Math.round((wDurMs / 1000) * (bpm / 60) * RESOLUTION);
    const x = TRACK_HEADER_W + wPos * ppt - viewState.scrollX;
    const w = durationTicks * ppt;

    if (x + w < TRACK_HEADER_W || x > rc.width) return;

    const isSelected = rc.selectedSegment?.type === 'wave' && rc.selectedSegment.index === wpIdx;

    ctx.fillStyle = color;
    ctx.globalAlpha = isSelected ? 0.35 : 0.15;
    ctx.fillRect(x, y + 4, w, laneH - 8);
    ctx.globalAlpha = 1;

    ctx.strokeStyle = isSelected ? '#f0f0ff' : color;
    ctx.lineWidth = isSelected ? 2 : 1;
    ctx.strokeRect(x, y + 4, w, laneH - 8);

    // Clip waveform to segment bounds (prevents overflow at start edge during resize)
    ctx.save();
    ctx.beginPath();
    ctx.rect(x, y + 4, w, laneH - 8);
    ctx.clip();

    // Waveform thumbnail — absolute-time positioned peaks (jitter-free during resize)
    const allPeaks = peakData?.get(wp.buffer_id || '');
    if (allPeaks && allPeaks.length > 1) {
      const totalMs = wp._fileTotalMs || Math.max(...project.wave_parts
        .filter(w2 => w2.buffer_id === wp.buffer_id)
        .map(w2 => (w2.skip || 0) + w2.file_duration_ms));
      const msPP = totalMs / allPeaks.length;
      const skip = wSkip || 0;
      const iStart = Math.max(0, Math.floor(skip / msPP));
      const iEnd = Math.min(allPeaks.length - 1, Math.ceil((skip + wDurMs) / msPP));

      if (iEnd > iStart) {
        const waveH = laneH - 16;
        const midY = y + 8 + waveH / 2;
        const pxPerMs = (bpm / 60 * RESOLUTION / 1000) * ppt;
        const maxPts = Math.max(4, Math.ceil(w) * 2);
        const stride = Math.max(1, Math.ceil((iEnd - iStart) / maxPts));

        const pts: { px: number; amp: number }[] = [];
        for (let i = iStart; i <= iEnd; i += stride) {
          const px = x + (i * msPP - skip) * pxPerMs;
          if (px > rc.width + 5) break;
          let ma = 0;
          const jE = Math.min(i + stride, iEnd + 1);
          for (let j = i; j < jE; j++) { if (allPeaks[j] > ma) ma = allPeaks[j]; }
          pts.push({ px, amp: ma * (waveH / 2) });
        }

        if (pts.length > 1) {
          ctx.fillStyle = color;
          ctx.globalAlpha = 0.3;
          ctx.beginPath();
          ctx.moveTo(pts[0].px, midY - pts[0].amp);
          for (let i = 1; i < pts.length; i++) ctx.lineTo(pts[i].px, midY - pts[i].amp);
          for (let i = pts.length - 1; i >= 0; i--) ctx.lineTo(pts[i].px, midY + pts[i].amp);
          ctx.closePath();
          ctx.fill();
          ctx.strokeStyle = color;
          ctx.globalAlpha = 0.6;
          ctx.lineWidth = 1;
          ctx.beginPath();
          ctx.moveTo(pts[0].px, midY - pts[0].amp);
          for (let i = 1; i < pts.length; i++) ctx.lineTo(pts[i].px, midY - pts[i].amp);
          ctx.stroke();
          ctx.beginPath();
          ctx.moveTo(pts[0].px, midY + pts[0].amp);
          for (let i = 1; i < pts.length; i++) ctx.lineTo(pts[i].px, midY + pts[i].amp);
          ctx.stroke();
          ctx.globalAlpha = 1;
        }
      }
    }

    ctx.restore(); // end waveform clip

    // Part name (outside clip so it renders normally)
    ctx.font = "9px 'DotGothic16', sans-serif";
    ctx.fillStyle = '#e0e0ff';
    ctx.textBaseline = 'top';
    ctx.fillText(_clipLabel(wp.name, wp.volume, wp.pan), x + 4, y + 6, w - 8);
  });

  // Crossfade markers for overlapping wave parts on same track
  const bpmG = project.tempos[0]?.bpm ?? 120;
  const tpsG = (bpmG / 60) * RESOLUTION;
  const wpByTrack = new Map<number, typeof project.wave_parts>();
  project.wave_parts.forEach(wp => {
    const arr = wpByTrack.get(wp.track_no) || [];
    arr.push(wp);
    wpByTrack.set(wp.track_no, arr);
  });
  wpByTrack.forEach((parts, trackNo) => {
    if (parts.length < 2) return;
    const sorted = [...parts].sort((a, b) => a.position - b.position);
    const yBase = RULER_H + trackNo * laneH - (rc.arrScrollY ?? 0);
    for (let i = 0; i < sorted.length - 1; i++) {
      const a = sorted[i], b = sorted[i + 1];
      const aEnd = a.position + (a.file_duration_ms / 1000) * tpsG;
      if (aEnd > b.position) {
        // Overlap region
        const oxStart = TRACK_HEADER_W + b.position * ppt - viewState.scrollX;
        const oxEnd = TRACK_HEADER_W + Math.min(aEnd, b.position + (b.file_duration_ms / 1000) * tpsG) * ppt - viewState.scrollX;
        const ow = oxEnd - oxStart;
        if (ow > 0 && oxEnd > TRACK_HEADER_W && oxStart < rc.width) {
          // Crossfade gradient overlay
          const grad = ctx.createLinearGradient(oxStart, 0, oxEnd, 0);
          grad.addColorStop(0, 'rgba(92, 224, 216, 0.3)');
          grad.addColorStop(0.5, 'rgba(92, 224, 216, 0.1)');
          grad.addColorStop(1, 'rgba(92, 224, 216, 0.3)');
          ctx.fillStyle = grad;
          ctx.fillRect(oxStart, yBase + 2, ow, laneH - 4);
          // X marker
          ctx.strokeStyle = '#5ce0d8';
          ctx.lineWidth = 1;
          ctx.globalAlpha = 0.5;
          ctx.beginPath();
          ctx.moveTo(oxStart, yBase + 2);
          ctx.lineTo(oxEnd, yBase + laneH - 2);
          ctx.moveTo(oxEnd, yBase + 2);
          ctx.lineTo(oxStart, yBase + laneH - 2);
          ctx.stroke();
          ctx.globalAlpha = 1;
        }
      }
    }
  });
}

function drawPlayhead(rc: ArrangementRenderContext) {
  const ticks = rc.playheadTicks;
  if (ticks == null) return;
  const { ctx, viewState } = rc;
  const x = TRACK_HEADER_W + ticks * viewState.pixelsPerTick - viewState.scrollX;

  // Playhead line — glow brighter + thicker when hovered
  if (x >= TRACK_HEADER_W && x <= rc.width) {
    const hover = rc.playheadHover;
    ctx.save();
    ctx.strokeStyle = hover ? '#ff8db5' : '#ff6b9d';
    ctx.lineWidth = hover ? 3 : 2;
    ctx.shadowColor = '#ff6b9d';
    ctx.shadowBlur = hover ? 14 : 6;
    ctx.beginPath();
    ctx.moveTo(x, 0);
    ctx.lineTo(x, rc.height);
    ctx.stroke();
    if (hover) {
      ctx.strokeStyle = 'rgba(255, 107, 157, 0.15)';
      ctx.lineWidth = 10;
      ctx.shadowBlur = 0;
      ctx.beginPath();
      ctx.moveTo(x, 0);
      ctx.lineTo(x, rc.height);
      ctx.stroke();
    }
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
