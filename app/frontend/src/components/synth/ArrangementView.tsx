/** Arrangement view — multi-track overview with lanes, part blocks, and waveforms.
 *  Canvas-based, similar pattern to PianoRollCanvas.
 */

import { useRef, useEffect, useState, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { useSynthStore } from '../../store/synthStore';
import { renderArrangement, TRACK_HEADER_W, RULER_H, type DragOverride } from '../../lib/synth/arrangementEngine';
import { RESOLUTION } from '../../types/ustx';
import * as engine from '../../lib/synth/audioEngine';
import TrackMixer from './TrackMixer';

type ArrDragMode = 'none' | 'playhead' | 'move-segment' | 'resize-segment';

// Module-level drag flag — SynthPage checks this to defer backend calls during drag
let _isDragging = false;
export function isDragging() { return _isDragging; }

// Module-level clipboard for segment copy/paste (accessible from SynthPage shortcuts)
export type SegmentClipboard = { type: 'voice' | 'instrument' | 'wave'; data: any } | null;
let _segmentClipboard: SegmentClipboard = null;
export function getSegmentClipboard() { return _segmentClipboard; }
export function setSegmentClipboard(c: SegmentClipboard) { _segmentClipboard = c; }

/** Paste clipboard segment at given position/track. Rejects overlap for voice/instrument. */
export function pasteSegment(clip: NonNullable<SegmentClipboard>, pos: number, trackNo: number): boolean {
  const s = useSynthStore.getState();
  const targetType = s.project.tracks[trackNo]?.track_type || 'vocal';
  const clipTrackType = clip.type === 'voice' ? 'vocal' : clip.type === 'wave' ? 'audio' : 'instrument';
  if (targetType !== clipTrackType) return false;

  const copy = { ...JSON.parse(JSON.stringify(clip.data)), position: pos, track_no: trackNo };
  const copyEnd = pos + (copy.duration ?? copy.file_duration_ms ?? 0);

  // Overlap check for voice/instrument (audio tracks allow overlap)
  if (clip.type !== 'wave') {
    const parts: any[] = clip.type === 'voice' ? s.project.voice_parts : s.project.instrument_parts;
    const overlaps = parts.some((p: any) => p.track_no === trackNo &&
      pos < p.position + p.duration && copyEnd > p.position);
    if (overlaps) return false;
  }

  const project = JSON.parse(JSON.stringify(s.project));
  if (clip.type === 'voice') project.voice_parts.push(copy);
  else if (clip.type === 'wave') project.wave_parts.push(copy);
  else project.instrument_parts.push(copy);
  const stacks = { undoStack: [...s.undoStack, JSON.parse(JSON.stringify(s.project))].slice(-50), redoStack: [] };
  useSynthStore.setState({ ...stacks, project, dirty: true });
  return true;
}

/** Snap a tick position to the nearest bar line. */
function snapToBar(tick: number): number {
  const s = useSynthStore.getState();
  const ts = s.project.time_signatures[0];
  const ticksPerBar = RESOLUTION * (ts?.beat_per_bar ?? 4);
  return Math.round(tick / ticksPerBar) * ticksPerBar;
}

/** Returns true if the currently selected track is an audio track. */
function isAudioTrackSelected(): boolean {
  const s = useSynthStore.getState();
  return s.project.tracks[s.selectedTrackIndex]?.track_type === 'audio';
}

/** Snap to bar, but skip snap if selected track is audio. */
function snapPlayhead(tick: number): number {
  return isAudioTrackSelected() ? Math.max(0, tick) : snapToBar(tick);
}

export default function ArrangementView() {
  const { t } = useTranslation();
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const [size, setSize] = useState({ width: 800, height: 400 });
  const dragMode = useRef<ArrDragMode>('none');
  const rafId = useRef(0);
  const dragSegment = useRef<{ type: 'voice' | 'instrument' | 'wave'; index: number; startTick: number; origTrackNo: number; edge?: 'start' | 'end'; origPos?: number; origSkip?: number; origDur?: number } | null>(null);
  const [clipboard, setClipboardState] = useState<SegmentClipboard>(null);
  const setClipboard = (c: SegmentClipboard) => { setSegmentClipboard(c); setClipboardState(c); };
  const tabHeld = useRef(false);
  const [contextMenu, setContextMenu] = useState<{ x: number; y: number; trackIdx: number; partHit: { type: 'voice' | 'instrument' | 'wave'; index: number } | null } | null>(null);
  const mouseXRef = useRef<number | null>(null);
  const playheadHoverRef = useRef(false);
  const dragOverrideRef = useRef<DragOverride | null>(null);
  const autoScrollRafRef = useRef(0);
  const lastDragXRef = useRef(0);
  const lastDragYRef = useRef(0);
  const setArrScrollY = (fn: (prev: number) => number) => useSynthStore.setState(s => ({ arrScrollY: fn(s.arrScrollY) }));
  const [renameEdit, setRenameEdit] = useState<{ target: 'voice' | 'instrument' | 'wave' | 'track'; index: number; x: number; y: number; value: string } | null>(null);
  const sizeRef = useRef(size);
  sizeRef.current = size;

  // Canvas paint — reads all data from getState() and refs, no React dependency
  const paint = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    const { width, height } = sizeRef.current;
    canvas.width = width * devicePixelRatio;
    canvas.height = height * devicePixelRatio;
    ctx.scale(devicePixelRatio, devicePixelRatio);
    const s = useSynthStore.getState();
    renderArrangement({
      ctx, width, height,
      project: s.project, viewState: s.viewState,
      activePartIndex: s.activePartIndex,
      playheadTicks: s.playbackPositionTicks,
      selectedSegment: s.selectedSegment,
      selectedTrackIndex: s.selectedTrackIndex,
      peakData: new Map(Object.entries(s.peakData)),
      arrScrollY: s.arrScrollY,
      playheadHover: playheadHoverRef.current,
      dragOverride: dragOverrideRef.current,
    });
    const mx = mouseXRef.current;
    if (mx != null && mx > TRACK_HEADER_W) {
      ctx.save();
      ctx.strokeStyle = 'rgba(255, 255, 255, 0.2)';
      ctx.lineWidth = 1;
      ctx.setLineDash([4, 4]);
      ctx.beginPath();
      ctx.moveTo(mx, RULER_H);
      ctx.lineTo(mx, height);
      ctx.stroke();
      ctx.setLineDash([]);
      ctx.restore();
    }
  }, []);

  const scheduleRepaint = useCallback(() => {
    cancelAnimationFrame(rafId.current);
    rafId.current = requestAnimationFrame(paint);
  }, [paint]);

  // ResizeObserver
  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;
    const ro = new ResizeObserver(entries => {
      const { width, height } = entries[0].contentRect;
      setSize({ width: Math.floor(width), height: Math.floor(height) });
    });
    ro.observe(container);
    return () => ro.disconnect();
  }, []);

  // Repaint on size change
  useEffect(() => { scheduleRepaint(); }, [size, scheduleRepaint]);

  // Subscribe to store — repaint on any relevant state change (no React re-render)
  useEffect(() => {
    const unsub = useSynthStore.subscribe(scheduleRepaint);
    scheduleRepaint();
    return () => { unsub(); cancelAnimationFrame(rafId.current); };
  }, [scheduleRepaint]);

  // Auto-scroll during drag when cursor is near canvas edges
  const autoScrollLoop = useCallback(() => {
    if (dragMode.current === 'none') { autoScrollRafRef.current = 0; return; }
    const { width } = sizeRef.current;
    const x = lastDragXRef.current;
    const edgeZone = 50;

    let speed = 0;
    if (x > width - edgeZone) {
      speed = Math.min(25, (x - width + edgeZone) / 2.5);
    } else if (x < TRACK_HEADER_W + edgeZone) {
      speed = -Math.min(25, (TRACK_HEADER_W + edgeZone - x) / 2.5);
    }

    if (speed !== 0) {
      const s = useSynthStore.getState();
      const newScrollX = Math.max(0, s.viewState.scrollX + speed);
      s.setViewState({ scrollX: newScrollX });

      // Clamp effective x to edge zone boundaries — keeps the dragged edge
      // at the zone boundary with preview space instead of hitting the border
      const cx = Math.max(TRACK_HEADER_W + edgeZone, Math.min(width - edgeZone, x));

      // Recalculate wave drag override with new scrollX
      const seg = dragSegment.current;
      if (seg?.type === 'wave') {
        const ppt = s.viewState.pixelsPerTick;
        const wp = s.project.wave_parts[seg.index];
        if (wp) {
          if (dragMode.current === 'move-segment') {
            const currentTick = (cx - TRACK_HEADER_W + newScrollX) / ppt;
            const bpm = s.project.tempos[0]?.bpm ?? 120;
            const msPerTick = 60000 / (bpm * RESOLUTION);
            const wpDurTicks = Math.round(wp.file_duration_ms / msPerTick);
            const totalDelta = currentTick - seg.startTick;
            let newPos = Math.max(0, Math.round(wp.position + totalDelta));
            const snapThreshold = 8 / ppt;
            const edges: number[] = [];
            for (const vp of s.project.voice_parts) { edges.push(vp.position, vp.position + vp.duration); }
            for (const ip of s.project.instrument_parts) { edges.push(ip.position, ip.position + ip.duration); }
            for (const w of s.project.wave_parts) {
              if (w === wp) continue;
              edges.push(w.position, w.position + Math.round(w.file_duration_ms / msPerTick));
            }
            for (const e of edges) {
              if (Math.abs(newPos - e) < snapThreshold) { newPos = e; break; }
              if (Math.abs(newPos + wpDurTicks - e) < snapThreshold) { newPos = e - wpDurTicks; break; }
            }
            const trackNo = dragOverrideRef.current?.trackNo ?? wp.track_no;
            dragOverrideRef.current = { type: 'wave', index: seg.index, position: newPos, trackNo };
          } else if (dragMode.current === 'resize-segment') {
            const rawTick = (cx - TRACK_HEADER_W + newScrollX) / ppt;
            const bpm = s.project.tempos[0]?.bpm ?? 120;
            const msPerTick = 60000 / (bpm * RESOLUTION);
            const oPos = seg.origPos!;
            const oSkip = seg.origSkip!;
            const oDur = seg.origDur!;
            const fileTotalMs = wp._fileTotalMs || (oSkip + oDur);
            const minDurMs = RESOLUTION * msPerTick;

            let edgeTick = rawTick;
            const snapPx = 8 / ppt;
            const allEdges: number[] = [];
            for (const vp of s.project.voice_parts) { allEdges.push(vp.position, vp.position + vp.duration); }
            for (const ip of s.project.instrument_parts) { allEdges.push(ip.position, ip.position + ip.duration); }
            for (const w of s.project.wave_parts) {
              if (w === wp) continue;
              allEdges.push(w.position, w.position + Math.round(w.file_duration_ms / msPerTick));
            }
            for (const e of allEdges) { if (Math.abs(edgeTick - e) < snapPx) { edgeTick = e; break; } }

            if (seg.edge === 'end') {
              const durTicks = Math.max(RESOLUTION, Math.round(edgeTick - oPos));
              const newDurMs = durTicks * msPerTick;
              const maxDurMs = fileTotalMs - oSkip;
              dragOverrideRef.current = { type: 'wave', index: seg.index, position: oPos, skip: oSkip, fileDurationMs: Math.min(maxDurMs, newDurMs) };
            } else if (seg.edge === 'start') {
              const deltaTicks = Math.round(edgeTick - oPos);
              const deltaMs = deltaTicks * msPerTick;
              const newSkip = oSkip + deltaMs;
              const newDur = oDur - deltaMs;
              if (newSkip >= 0 && newSkip < fileTotalMs && newDur >= minDurMs) {
                dragOverrideRef.current = { type: 'wave', index: seg.index, position: oPos + deltaTicks, skip: newSkip, fileDurationMs: newDur };
              }
            }
          }
        }
      }
      scheduleRepaint();
    }
    autoScrollRafRef.current = requestAnimationFrame(autoScrollLoop);
  }, [scheduleRepaint]);

  /** Get track index from screen Y position, accounting for arrScrollY. */
  const getTrackAtY = useCallback((y: number): number => {
    const s = useSynthStore.getState();
    const laneH = s.viewState.trackLaneHeight ?? 64;
    return Math.floor((y - RULER_H + (s.arrScrollY || 0)) / laneH);
  }, []);

  const EDGE_PX = 8;

  // Helper: find which part is at a click position, with edge detection
  const hitTestPart = useCallback((x: number, y: number): { type: 'voice' | 'instrument' | 'wave'; index: number; edge?: 'start' | 'end' } | null => {
    if (x < TRACK_HEADER_W || y < RULER_H) return null;
    const s = useSynthStore.getState();
    const ppt = s.viewState.pixelsPerTick;
    const clickTick = (x - TRACK_HEADER_W + s.viewState.scrollX) / ppt;
    const trackIdx = getTrackAtY(y);
    const bpm = s.project.tempos[0]?.bpm ?? 120;
    const ticksPerSec = bpm / 60 * RESOLUTION;
    const edgeTicks = EDGE_PX / ppt;
    for (let i = 0; i < s.project.voice_parts.length; i++) {
      const vp = s.project.voice_parts[i];
      if (vp.track_no === trackIdx && clickTick >= vp.position && clickTick <= vp.position + vp.duration) {
        const edge = clickTick <= vp.position + edgeTicks ? 'start' as const : clickTick >= vp.position + vp.duration - edgeTicks ? 'end' as const : undefined;
        return { type: 'voice', index: i, edge };
      }
    }
    for (let i = 0; i < s.project.instrument_parts.length; i++) {
      const ip = s.project.instrument_parts[i];
      if (ip.track_no === trackIdx && clickTick >= ip.position && clickTick <= ip.position + ip.duration) {
        const edge = clickTick <= ip.position + edgeTicks ? 'start' as const : clickTick >= ip.position + ip.duration - edgeTicks ? 'end' as const : undefined;
        return { type: 'instrument', index: i, edge };
      }
    }
    for (let i = 0; i < s.project.wave_parts.length; i++) {
      const wp = s.project.wave_parts[i];
      const durTicks = Math.round((wp.file_duration_ms / 1000) * ticksPerSec);
      if (wp.track_no === trackIdx && clickTick >= wp.position && clickTick <= wp.position + durTicks) {
        const edge = clickTick <= wp.position + edgeTicks ? 'start' as const : clickTick >= wp.position + durTicks - edgeTicks ? 'end' as const : undefined;
        return { type: 'wave', index: i, edge };
      }
    }
    return null;
  }, []);

  // Double-click on a vocal part → switch to piano roll
  const handleDoubleClick = useCallback((e: React.MouseEvent) => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    const s = useSynthStore.getState();

    if (x < TRACK_HEADER_W) return;

    // Double-click on ruler → snap playhead to nearest bar
    if (y < RULER_H && x > TRACK_HEADER_W) {
      const clickTick = (x - TRACK_HEADER_W + s.viewState.scrollX) / s.viewState.pixelsPerTick;
      const snapped = snapToBar(clickTick);
      s.setPlaybackPosition(snapped);
      engine.seek(engine.ticksToSeconds(snapped, s.project.tempos));
      return;
    }

    const trackIdx = getTrackAtY(y);
    if (trackIdx < 0 || trackIdx >= s.project.tracks.length) return;

    const clickTick = (x - TRACK_HEADER_W + s.viewState.scrollX) / s.viewState.pixelsPerTick;
    const hit = hitTestPart(x, y);
    const tool = s.activeTool;

    // Draw mode + double-click on segment → switch to select + select it
    if (tool === 'draw' && hit) {
      s.setTool('select');
      s.setSelectedSegment(hit);
      return;
    }

    // Select mode + double-click on segment → enter piano roll
    if (hit) {
      if (hit.type === 'voice') {
        const vpIdx = s.project.voice_parts.findIndex(vp =>
          vp.track_no === trackIdx && clickTick >= vp.position && clickTick < vp.position + vp.duration
        );
        if (vpIdx >= 0) {
          s.setActivePart(vpIdx);
          s.setViewMode('pianoroll');
        }
      } else if (hit.type === 'instrument') {
        const ipIdx = s.project.instrument_parts.findIndex(ip =>
          ip.track_no === trackIdx && clickTick >= ip.position && clickTick < ip.position + ip.duration
        );
        if (ipIdx >= 0) {
          s.setActiveInstrumentPart(ipIdx);
          s.setViewMode('pianoroll');
        }
      } else if (hit.type === 'wave') {
        s.setActiveWavePart(hit.index);
        s.setViewMode('pianoroll');
      }
      return;
    }

    // Double-click empty space → create segment + switch to select (not for audio tracks)
    const tt = s.project.tracks[trackIdx]?.track_type || 'vocal';
    if (tt === 'audio') return;
    const pos = snapToBar(clickTick);
    s.setPlaybackPosition(pos);
    engine.seek(engine.ticksToSeconds(pos, s.project.tempos));
    const overlapping = (tt === 'instrument' ? s.project.instrument_parts : s.project.voice_parts)
      .some((p: any) => p.track_no === trackIdx && pos >= p.position && pos < p.position + p.duration);
    if (!overlapping) {
      if (tt === 'instrument') s.addInstrumentPartOnTrack(trackIdx, pos);
      else s.addVoicePartOnTrack(trackIdx, pos);
      const s2 = useSynthStore.getState();
      const newParts = tt === 'instrument' ? s2.project.instrument_parts : s2.project.voice_parts;
      const newIdx = newParts.findIndex((p: any) => p.track_no === trackIdx && p.position === pos);
      if (newIdx >= 0) {
        s2.setTool('select');
        s2.setSelectedSegment({ type: tt === 'instrument' ? 'instrument' : 'voice', index: newIdx });
      }
    }
  }, [hitTestPart]);

  // Mouse down — ruler click sets playhead, start drag
  const handleMouseDown = useCallback((e: React.MouseEvent) => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    if (e.button !== 0) return;
    const rect = canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;

    // Click on ruler → set playhead and start dragging
    if (y < RULER_H && x > TRACK_HEADER_W) {
      const s = useSynthStore.getState();
      const rawTick = Math.max(0, (x - TRACK_HEADER_W + s.viewState.scrollX) / s.viewState.pixelsPerTick);
      const tick = snapPlayhead(rawTick);
      s.setPlaybackPosition(tick);
      engine.seek(engine.ticksToSeconds(tick, s.project.tempos));
      dragMode.current = 'playhead';
      return;
    }

    // Click near playhead line in content area → drag playhead (priority over segment)
    if (x > TRACK_HEADER_W && y >= RULER_H) {
      const s = useSynthStore.getState();
      const playheadX = TRACK_HEADER_W + s.playbackPositionTicks * s.viewState.pixelsPerTick - s.viewState.scrollX;
      if (Math.abs(x - playheadX) < 8) {
        dragMode.current = 'playhead';
        return;
      }
    }

    if (x <= TRACK_HEADER_W) {
      // Track header clicks handled by TrackMixer overlay (z-index above canvas)
      // This branch is fallback only (e.g. clicks below all tracks)
      const s = useSynthStore.getState();
      s.setSelectedSegment(null);
      const trackIdx2 = getTrackAtY(y);
      if (trackIdx2 >= 0 && trackIdx2 < s.project.tracks.length) {
        const wasAudio = s.project.tracks[s.selectedTrackIndex]?.track_type === 'audio';
        const nowAudio = s.project.tracks[trackIdx2]?.track_type === 'audio';
        s.setSelectedTrackIndex(trackIdx2);
        if (wasAudio && !nowAudio) {
          const snapped = snapToBar(s.playbackPositionTicks);
          s.setPlaybackPosition(snapped);
          engine.seek(engine.ticksToSeconds(snapped, s.project.tempos));
        }
      }
      return;
    }

    const trackIdx2 = getTrackAtY(y);
    const s0 = useSynthStore.getState();
    if (trackIdx2 >= 0 && trackIdx2 < s0.project.tracks.length) {
      const wasAudio = s0.project.tracks[s0.selectedTrackIndex]?.track_type === 'audio';
      const nowAudio = s0.project.tracks[trackIdx2]?.track_type === 'audio';
      s0.setSelectedTrackIndex(trackIdx2);
      if (wasAudio && !nowAudio) {
        const snapped = snapToBar(s0.playbackPositionTicks);
        s0.setPlaybackPosition(snapped);
        engine.seek(engine.ticksToSeconds(snapped, s0.project.tempos));
      }
    }

    const s = useSynthStore.getState();
    const tool = s.activeTool;
    const hit = hitTestPart(x, y);

    if (tool === 'select') {
      s.setSelectedSegment(hit ? { type: hit.type, index: hit.index } : null);
      if (hit) {
        const clickTick = (x - TRACK_HEADER_W + s.viewState.scrollX) / s.viewState.pixelsPerTick;
        const partArr = hit.type === 'voice' ? s.project.voice_parts : hit.type === 'wave' ? s.project.wave_parts : s.project.instrument_parts;
        const origTrackNo = (partArr[hit.index] as any)?.track_no ?? 0;
        if (hit.edge) {
          dragMode.current = 'resize-segment';
          _isDragging = true;
          useSynthStore.getState().pushUndoSnapshot();
          const part = partArr[hit.index] as any;
          dragSegment.current = { type: hit.type, index: hit.index, startTick: clickTick, origTrackNo, edge: hit.edge,
            origPos: part?.position ?? 0, origSkip: part?.skip ?? 0, origDur: part?.duration ?? part?.file_duration_ms ?? 0 };
        } else {
          dragMode.current = 'move-segment';
          _isDragging = true;
          useSynthStore.getState().pushUndoSnapshot();
          dragSegment.current = { type: hit.type, index: hit.index, startTick: clickTick, origTrackNo };
        }
      } else {
        const rawTick = (x - TRACK_HEADER_W + s.viewState.scrollX) / s.viewState.pixelsPerTick;
        const tick = snapPlayhead(rawTick);
        s.setPlaybackPosition(tick);
        engine.seek(engine.ticksToSeconds(tick, s.project.tempos));
        dragMode.current = 'playhead';
      }
    } else if (tool === 'erase' && hit) {
      if (hit.type === 'voice') s.deleteVoicePart(hit.index);
      else if (hit.type === 'instrument') s.deleteInstrumentPart(hit.index);
      else if (hit.type === 'wave') s.deleteWavePart(hit.index);
    } else if (tool === 'draw' && !hit && x > TRACK_HEADER_W && trackIdx2 >= 0 && trackIdx2 < s.project.tracks.length) {
      const tt = s.project.tracks[trackIdx2]?.track_type || 'vocal';
      if (tt === 'audio') return; // audio tracks: no segment creation
      // Draw tool: click empty → create new segment at snapped bar
      const rawTick = (x - TRACK_HEADER_W + s.viewState.scrollX) / s.viewState.pixelsPerTick;
      const pos = snapToBar(rawTick);
      s.setPlaybackPosition(pos);
      engine.seek(engine.ticksToSeconds(pos, s.project.tempos));
      // Find next segment on same track to cap duration
      const parts = tt === 'instrument' ? s.project.instrument_parts : s.project.voice_parts;
      const nextPart = parts
        .filter((p: any) => p.track_no === trackIdx2 && p.position > pos)
        .sort((a: any, b: any) => a.position - b.position)[0];
      // Check not overlapping existing segment
      const overlapping = parts.some((p: any) => p.track_no === trackIdx2 && pos >= p.position && pos < p.position + p.duration);
      if (!overlapping) {
        if (tt === 'instrument') s.addInstrumentPartOnTrack(trackIdx2, pos);
        else s.addVoicePartOnTrack(trackIdx2, pos);
        // Switch to select + select the newly created segment
        const s2 = useSynthStore.getState();
        const newParts = tt === 'instrument' ? s2.project.instrument_parts : s2.project.voice_parts;
        const newIdx = newParts.findIndex((p: any) => p.track_no === trackIdx2 && p.position === pos);
        if (newIdx >= 0) {
          s2.setTool('select');
          s2.setSelectedSegment({ type: tt === 'instrument' ? 'instrument' : 'voice', index: newIdx });
        }
      }
    }
  }, [hitTestPart]);

  // Start auto-scroll loop when drag starts
  const startAutoScroll = useCallback(() => {
    if (autoScrollRafRef.current) return;
    autoScrollRafRef.current = requestAnimationFrame(autoScrollLoop);
  }, [autoScrollLoop]);

  // Mouse move — drag playhead + cursor styling + guide line
  const handleMouseMove = useCallback((e: React.MouseEvent) => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;

    // Save for auto-scroll recalculation
    lastDragXRef.current = x;
    lastDragYRef.current = y;
    if (dragMode.current !== 'none') startAutoScroll();

    // Update guide line (ref — no React re-render)
    mouseXRef.current = x > TRACK_HEADER_W ? x : null;

    // Cursor styling + playhead hover
    let newHover = false;
    if (dragMode.current === 'playhead') {
      canvas.style.cursor = 'col-resize';
    } else if (dragMode.current === 'move-segment') {
      canvas.style.cursor = 'grabbing';
    } else if (y < RULER_H && x > TRACK_HEADER_W) {
      canvas.style.cursor = 'col-resize';
    } else if (x > TRACK_HEADER_W && y >= RULER_H) {
      const s2 = useSynthStore.getState();
      const phX = TRACK_HEADER_W + s2.playbackPositionTicks * s2.viewState.pixelsPerTick - s2.viewState.scrollX;
      const nearPlayhead = Math.abs(x - phX) < 8;
      const hoverHit = hitTestPart(x, y);
      if (hoverHit?.edge) {
        canvas.style.cursor = 'ew-resize';
      } else if (nearPlayhead) {
        canvas.style.cursor = 'col-resize';
        newHover = true;
      } else {
        canvas.style.cursor = 'default';
      }
    } else {
      canvas.style.cursor = 'default';
    }
    playheadHoverRef.current = newHover;

    if (dragMode.current === 'playhead') {
      if (x > TRACK_HEADER_W) {
        const s = useSynthStore.getState();
        const rawTick = Math.max(0, (x - TRACK_HEADER_W + s.viewState.scrollX) / s.viewState.pixelsPerTick);
        const tick = snapPlayhead(rawTick);
        s.setPlaybackPosition(tick);
        engine.seek(engine.ticksToSeconds(tick, s.project.tempos));
      }
    } else if (dragMode.current === 'move-segment' && dragSegment.current && x > TRACK_HEADER_W) {
      const s = useSynthStore.getState();
      const { width: cw } = sizeRef.current;
      const clampedX = Math.max(TRACK_HEADER_W + 50, Math.min(cw - 50, x));
      const currentTick = (x - TRACK_HEADER_W + s.viewState.scrollX) / s.viewState.pixelsPerTick;

      if (dragSegment.current.type === 'wave') {
        // Wave: visual-only override — zero store writes during drag
        const wp = s.project.wave_parts[dragSegment.current.index];
        if (wp) {
          const bpm = s.project.tempos[0]?.bpm ?? 120;
          const msPerTick = 60000 / (bpm * RESOLUTION);
          const wpDurTicks = Math.round(wp.file_duration_ms / msPerTick);
          const waveTick = (clampedX - TRACK_HEADER_W + s.viewState.scrollX) / s.viewState.pixelsPerTick;
          const totalDelta = waveTick - dragSegment.current.startTick;
          let newPos = Math.max(0, Math.round(wp.position + totalDelta));
          const snapThreshold = 8 / s.viewState.pixelsPerTick;
          const edges: number[] = [];
          for (const vp of s.project.voice_parts) { edges.push(vp.position, vp.position + vp.duration); }
          for (const ip of s.project.instrument_parts) { edges.push(ip.position, ip.position + ip.duration); }
          for (const w of s.project.wave_parts) {
            if (w === wp) continue;
            edges.push(w.position, w.position + Math.round(w.file_duration_ms / msPerTick));
          }
          for (const e of edges) {
            if (Math.abs(newPos - e) < snapThreshold) { newPos = e; break; }
            if (Math.abs(newPos + wpDurTicks - e) < snapThreshold) { newPos = e - wpDurTicks; break; }
          }
          // Cross-track
          let trackNo = dragOverrideRef.current?.trackNo ?? wp.track_no;
          const hoverTrack = getTrackAtY(y);
          if (hoverTrack >= 0 && hoverTrack < s.project.tracks.length &&
              s.project.tracks[hoverTrack]?.track_type === 'audio') {
            trackNo = hoverTrack;
          }
          dragOverrideRef.current = { type: 'wave', index: dragSegment.current.index, position: newPos, trackNo };
        }
      } else {
        // Voice/instrument: store-based (unchanged)
        const deltaTick = snapToBar(currentTick) - snapToBar(dragSegment.current.startTick);
        if (Math.abs(deltaTick) > 0) {
          if (dragSegment.current.type === 'voice') s.moveVoicePart(dragSegment.current.index, deltaTick);
          else if (dragSegment.current.type === 'instrument') s.moveInstrumentPart(dragSegment.current.index, deltaTick);
        }
        dragSegment.current.startTick = currentTick;
        const hoverTrack = getTrackAtY(y);
        if (hoverTrack >= 0 && hoverTrack < s.project.tracks.length) {
          const segType = dragSegment.current.type;
          const expectedTrackType = segType === 'voice' ? 'vocal' : 'instrument';
          if (s.project.tracks[hoverTrack]?.track_type === expectedTrackType) {
            const partArr = segType === 'voice' ? s.project.voice_parts : s.project.instrument_parts;
            const curTrack = (partArr[dragSegment.current.index] as any)?.track_no;
            if (curTrack !== undefined && curTrack !== hoverTrack) {
              s.moveSegmentToTrack(dragSegment.current, hoverTrack);
            }
          }
        }
      }
      canvas.style.cursor = 'grabbing';
    } else if (dragMode.current === 'resize-segment' && dragSegment.current && x > TRACK_HEADER_W) {
      const s = useSynthStore.getState();
      const ppt = s.viewState.pixelsPerTick;
      const rawTick = (x - TRACK_HEADER_W + s.viewState.scrollX) / ppt;
      const seg = dragSegment.current;
      const isWave = seg.type === 'wave';
      const snappedTick = isWave ? rawTick : snapToBar(rawTick);
      const bpm = s.project.tempos[0]?.bpm ?? 120;
      const tps = bpm / 60 * RESOLUTION;
      const msPerTick = 60000 / (bpm * RESOLUTION);

      if (seg.type === 'voice') {
        const part = s.project.voice_parts[seg.index];
        if (part) {
          const p = { ...part };
          if (seg.edge === 'end') {
            p.duration = Math.max(RESOLUTION, snappedTick - p.position);
          } else if (seg.edge === 'start') {
            const delta = snappedTick - p.position;
            if (p.duration - delta >= RESOLUTION) {
              p.position = snappedTick;
              p.duration -= delta;
              p.notes = part.notes.filter((n: any) => n.position >= delta).map((n: any) => ({ ...n, position: n.position - delta }));
            }
          }
          const newParts = [...s.project.voice_parts];
          newParts[seg.index] = p;
          useSynthStore.setState({ project: { ...s.project, voice_parts: newParts }, dirty: true });
        }
      } else if (seg.type === 'instrument') {
        const part = s.project.instrument_parts[seg.index];
        if (part) {
          const p = { ...part };
          if (seg.edge === 'end') {
            p.duration = Math.max(RESOLUTION, snappedTick - p.position);
          } else if (seg.edge === 'start') {
            const delta = snappedTick - p.position;
            if (p.duration - delta >= RESOLUTION) {
              p.position = snappedTick;
              p.duration -= delta;
              p.notes = part.notes.filter((n: any) => n.position >= delta).map((n: any) => ({ ...n, position: n.position - delta }));
            }
          }
          const newParts = [...s.project.instrument_parts];
          newParts[seg.index] = p;
          useSynthStore.setState({ project: { ...s.project, instrument_parts: newParts }, dirty: true });
        }
      }
      else if (seg.type === 'wave') {
        // Wave resize: visual-only override — zero store writes during drag
        // Clamp to edge zone so the edge stays with preview space during auto-scroll
        const { width: cw2 } = sizeRef.current;
        const waveRawTick = (Math.max(TRACK_HEADER_W + 50, Math.min(cw2 - 50, x)) - TRACK_HEADER_W + s.viewState.scrollX) / ppt;
        const wp = s.project.wave_parts[seg.index];
        if (wp) {
          const oPos = seg.origPos!;
          const oSkip = seg.origSkip!;
          const oDur = seg.origDur!;
          const fileTotalMs = wp._fileTotalMs || (oSkip + oDur);
          const minDurMs = RESOLUTION * msPerTick;

          // Cross-segment edge snap (8px threshold)
          let edgeTick = waveRawTick;
          const snapPx = 8 / ppt;
          const allEdges: number[] = [];
          for (const vp of s.project.voice_parts) { allEdges.push(vp.position, vp.position + vp.duration); }
          for (const ip of s.project.instrument_parts) { allEdges.push(ip.position, ip.position + ip.duration); }
          for (const w of s.project.wave_parts) {
            if (w === wp) continue;
            allEdges.push(w.position, w.position + Math.round(w.file_duration_ms / msPerTick));
          }
          for (const e of allEdges) { if (Math.abs(edgeTick - e) < snapPx) { edgeTick = e; break; } }

          if (seg.edge === 'end') {
            const durTicks = Math.max(RESOLUTION, Math.round(edgeTick - oPos));
            const newDurMs = durTicks * msPerTick;
            const maxDurMs = fileTotalMs - oSkip;
            dragOverrideRef.current = {
              type: 'wave', index: seg.index,
              position: oPos, skip: oSkip,
              fileDurationMs: Math.min(maxDurMs, newDurMs),
            };
          } else if (seg.edge === 'start') {
            const deltaTicks = Math.round(edgeTick - oPos);
            const deltaMs = deltaTicks * msPerTick;
            const newSkip = oSkip + deltaMs;
            const newDur = oDur - deltaMs;
            if (newSkip >= 0 && newSkip < fileTotalMs && newDur >= minDurMs) {
              dragOverrideRef.current = {
                type: 'wave', index: seg.index,
                position: oPos + deltaTicks, skip: newSkip,
                fileDurationMs: newDur,
              };
            }
          }
        }
      }
      canvas.style.cursor = 'ew-resize';
    }
    scheduleRepaint();
  }, [scheduleRepaint, startAutoScroll]);

  // Mouse up — end drag (window-level to catch release outside canvas)
  const handleMouseUp = useCallback(() => {
    const wasDragging = _isDragging;
    const override = dragOverrideRef.current;
    dragMode.current = 'none';
    dragSegment.current = null;
    _isDragging = false;
    dragOverrideRef.current = null;
    if (autoScrollRafRef.current) { cancelAnimationFrame(autoScrollRafRef.current); autoScrollRafRef.current = 0; }

    // Commit wave part visual override to store (single write on mouseup)
    if (override?.type === 'wave') {
      const s = useSynthStore.getState();
      const wp = s.project.wave_parts[override.index];
      if (wp) {
        const updated = { ...wp };
        if (override.position !== undefined) updated.position = override.position;
        if (override.trackNo !== undefined) updated.track_no = override.trackNo;
        if (override.skip !== undefined) updated.skip = override.skip;
        if (override.fileDurationMs !== undefined) updated.file_duration_ms = override.fileDurationMs;
        const newParts = [...s.project.wave_parts];
        newParts[override.index] = updated;
        useSynthStore.setState({ project: { ...s.project, wave_parts: newParts }, dirty: true });
      }
    }

    if (wasDragging) {
      scheduleRepaint();
      window.dispatchEvent(new CustomEvent('utai-drag-end'));
    }
  }, [scheduleRepaint]);

  useEffect(() => {
    window.addEventListener('mouseup', handleMouseUp);
    return () => window.removeEventListener('mouseup', handleMouseUp);
  }, [handleMouseUp]);

  useEffect(() => {
    return () => { _isDragging = false; dragMode.current = 'none'; };
  }, []);

  // Track cursor outside canvas during drag (for auto-scroll near edges)
  useEffect(() => {
    const onWindowMove = (e: MouseEvent) => {
      if (dragMode.current === 'none') return;
      const canvas = canvasRef.current;
      if (!canvas) return;
      const rect = canvas.getBoundingClientRect();
      lastDragXRef.current = e.clientX - rect.left;
      lastDragYRef.current = e.clientY - rect.top;
    };
    window.addEventListener('mousemove', onWindowMove);
    return () => window.removeEventListener('mousemove', onWindowMove);
  }, []);

  // Click handler — no-op (all logic moved to handleMouseDown for correct drag timing)
  const handleClick = useCallback(() => {}, []);

  // Right-click context menu
  const handleContextMenu = useCallback((e: React.MouseEvent) => {
    e.preventDefault();
    const canvas = canvasRef.current;
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;

    const trackIdx = y >= RULER_H ? getTrackAtY(y) : -1;
    const validTrack = trackIdx >= 0 && trackIdx < useSynthStore.getState().project.tracks.length;

    const s = useSynthStore.getState();
    const partHit = hitTestPart(x, y);

    // Select segment under cursor on right-click (without moving playhead)
    if (partHit) {
      s.setSelectedSegment(partHit);
      const hitTrackNo = partHit.type === 'voice' ? s.project.voice_parts[partHit.index]?.track_no ?? 0
        : partHit.type === 'instrument' ? s.project.instrument_parts[partHit.index]?.track_no ?? 0
        : s.project.wave_parts[partHit.index]?.track_no ?? 0;
      const wasAudio = s.project.tracks[s.selectedTrackIndex]?.track_type === 'audio';
      const nowAudio = s.project.tracks[hitTrackNo]?.track_type === 'audio';
      s.setSelectedTrackIndex(hitTrackNo);
      if (wasAudio && !nowAudio) {
        const snapped = snapToBar(s.playbackPositionTicks);
        s.setPlaybackPosition(snapped);
        engine.seek(engine.ticksToSeconds(snapped, s.project.tempos));
      }
    } else if (validTrack) {
      const wasAudio = s.project.tracks[s.selectedTrackIndex]?.track_type === 'audio';
      const nowAudio = s.project.tracks[trackIdx]?.track_type === 'audio';
      s.setSelectedTrackIndex(trackIdx);
      if (wasAudio && !nowAudio) {
        const snapped = snapToBar(s.playbackPositionTicks);
        s.setPlaybackPosition(snapped);
        engine.seek(engine.ticksToSeconds(snapped, s.project.tempos));
      }
    }
    setContextMenu({
      x: e.clientX,
      y: e.clientY,
      trackIdx: validTrack ? trackIdx : -1,
      partHit,
    });
  }, []);

  // Wheel scroll
  // Native wheel handler (passive:false needed to preventDefault on Ctrl+wheel)
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const handler = (e: WheelEvent) => {
      e.preventDefault();
      const s = useSynthStore.getState();

      const zoomTrackHeight = () => {
        const factor = e.deltaY < 0 ? 1.15 : 1 / 1.15;
        const cur = s.viewState.trackLaneHeight ?? 64;
        s.setViewState({ trackLaneHeight: Math.round(Math.max(40, Math.min(128, cur * factor))) });
      };

      if (e.ctrlKey && e.shiftKey) {
        zoomTrackHeight();
      } else if (e.ctrlKey) {
        const rect = canvas.getBoundingClientRect();
        const mx = e.clientX - rect.left;
        if (mx < TRACK_HEADER_W) {
          zoomTrackHeight();
        } else {
          const delta = e.deltaY > 0 ? 0.9 : 1.1;
          s.setViewState({ pixelsPerTick: Math.max(0.03, Math.min(5, s.viewState.pixelsPerTick * delta)) });
        }
      } else if (tabHeld.current) {
        zoomTrackHeight();
      } else {
        const rect = canvas.getBoundingClientRect();
        const mx = e.clientX - rect.left;
        const totalH = s.project.tracks.length * (s.viewState.trackLaneHeight ?? 64);
        const maxSY = Math.max(0, totalH - (rect.height - RULER_H));
        if (e.shiftKey || mx < TRACK_HEADER_W) {
          // Shift+scroll or scroll over track header → vertical scroll
          setArrScrollY(prev => Math.max(0, Math.min(maxSY, prev + (e.deltaX || e.deltaY))));
        } else {
          s.setViewState({ scrollX: Math.max(0, s.viewState.scrollX + e.deltaX + e.deltaY) });
        }
      }
    };
    canvas.addEventListener('wheel', handler, { passive: false });
    return () => canvas.removeEventListener('wheel', handler);
  }, []);

  useEffect(() => {
    const down = (e: KeyboardEvent) => { if (e.key === 'Tab') { e.preventDefault(); tabHeld.current = true; } };
    const up = (e: KeyboardEvent) => { if (e.key === 'Tab') tabHeld.current = false; };
    window.addEventListener('keydown', down, true);
    window.addEventListener('keyup', up, true);
    return () => { window.removeEventListener('keydown', down, true); window.removeEventListener('keyup', up, true); };
  }, []);

  // Track management buttons
  const addVocalTrack = () => useSynthStore.getState().addTrack('vocal');
  const addAudioTrack = () => {
    const input = document.createElement('input');
    input.type = 'file';
    input.accept = 'audio/*,.wav,.mp3,.flac,.ogg,.m4a';
    input.onchange = async () => {
      const file = input.files?.[0];
      if (!file) return;
      try {
        const { uploadAudio, getAudioURL } = await import('../../api/client');
        const result = await uploadAudio(file);
        const bufferId = result.buffer_id || result.id;
        if (!bufferId) return;

        // Generate peaks from audio for waveform display
        const audioCtx = new AudioContext();
        let decoded: AudioBuffer;
        try {
          const arrayBuf = await file.arrayBuffer();
          decoded = await audioCtx.decodeAudioData(arrayBuf);
        } catch (decodeErr) {
          audioCtx.close();
          throw decodeErr;
        }
        const channel = decoded.getChannelData(0);
        const peakCount = Math.min(8000, Math.ceil(decoded.duration * 100));
        const blockSize = Math.floor(channel.length / peakCount);
        const peaks = new Float32Array(peakCount);
        for (let i = 0; i < peakCount; i++) {
          let max = 0;
          for (let j = 0; j < blockSize; j++) {
            const v = Math.abs(channel[i * blockSize + j] || 0);
            if (v > max) max = v;
          }
          peaks[i] = max;
        }
        audioCtx.close();

        const durationMs = decoded.duration * 1000;
        // Batch all store updates to avoid race conditions
        useSynthStore.setState(state => {
          const newTracks = [...state.project.tracks];
          const newWaveParts = [...state.project.wave_parts];
          const idx = newTracks.length;
          const colors = ['Blue', 'Pink', 'Green', 'Yellow', 'Purple', 'Orange'];
          newTracks.push({
            track_name: file.name.replace(/\.[^.]+$/, ''),
            track_color: colors[idx % colors.length],
            track_type: 'audio',
            mute: false, solo: false, volume: 0, pan: 0,
            voice_color_names: [''],
          });
          newWaveParts.push({
            name: file.name,
            track_no: idx,
            position: 0,
            relative_path: '',
            file_duration_ms: durationMs,
            _fileTotalMs: durationMs,
            skip: 0, trim: 0, fadein: 0, fadeout: 0,
            buffer_id: bufferId,
          });
          return {
            project: { ...state.project, tracks: newTracks, wave_parts: newWaveParts },
            dirty: true,
            selectedTrackIndex: idx,
            selectedSegment: null,
            peakData: { ...state.peakData, [bufferId]: peaks },
            trackAudioUrls: { ...state.trackAudioUrls, [idx]: getAudioURL(bufferId) },
          };
        });
      } catch (err) {
        console.error('Audio import failed:', err);
      }
    };
    input.click();
  };
  const addInstrumentTrack = () => useSynthStore.getState().addTrack('instrument');

  // Close context menu
  const closeContextMenu = useCallback(() => setContextMenu(null), []);

  // Context menu actions
  const ctxAddVocal = useCallback(() => { useSynthStore.getState().addTrack('vocal'); closeContextMenu(); }, [closeContextMenu]);
  const ctxAddAudio = useCallback(() => { closeContextMenu(); addAudioTrack(); }, [closeContextMenu]);
  const ctxAddInstrument = useCallback(() => { useSynthStore.getState().addTrack('instrument'); closeContextMenu(); }, [closeContextMenu]);
  const ctxDeleteTrack = useCallback(() => {
    if (contextMenu && contextMenu.trackIdx >= 0) {
      useSynthStore.getState().removeTrack(contextMenu.trackIdx);
    }
    closeContextMenu();
  }, [contextMenu, closeContextMenu]);

  return (
    <div style={{ display: 'flex', flexDirection: 'column', flex: 1, overflow: 'hidden' }}>
      {/* Track action bar */}
      <div className="utai-synth-arrangement__actions">
        <button className="utai-synth-arrangement__action-btn" onClick={addVocalTrack}
          style={{ color: 'var(--accent-pink)', borderColor: 'var(--accent-pink)' }}>
          + {t('synth.track.addVocal')}
        </button>
        <button className="utai-synth-arrangement__action-btn" onClick={addAudioTrack}
          style={{ color: 'var(--accent-cyan)', borderColor: 'var(--accent-cyan)' }}>
          + {t('synth.track.addAudio')}
        </button>
        <button className="utai-synth-arrangement__action-btn" onClick={addInstrumentTrack}
          style={{ color: 'var(--accent-orange)', borderColor: 'var(--accent-orange)' }}>
          + {t('synth.track.addInstrument')}
        </button>
      </div>
      {/* Canvas + mixer overlay */}
      <div ref={containerRef} style={{ flex: 1, overflow: 'hidden', position: 'relative' }}>
        <TrackMixer />
        <canvas
          ref={canvasRef}
          style={{ width: size.width, height: size.height, display: 'block', cursor: dragMode.current === 'playhead' ? 'col-resize' : undefined }}
          onClick={handleClick}
          onMouseDown={handleMouseDown}
          onMouseMove={handleMouseMove}
          onMouseUp={handleMouseUp}
          onMouseLeave={() => { if (dragMode.current === 'none') { mouseXRef.current = null; scheduleRepaint(); } }}
          onDoubleClick={handleDoubleClick}
          onContextMenu={handleContextMenu}
        />
        {/* Rename input */}
        {renameEdit && (
          <input
            autoFocus
            defaultValue={renameEdit.value}
            onKeyDown={e => {
              if (e.key === 'Enter') {
                const v = (e.target as HTMLInputElement).value.trim();
                if (v) {
                  const st = useSynthStore.getState();
                  if (renameEdit.target === 'voice') st.renameVoicePart(renameEdit.index, v);
                  else if (renameEdit.target === 'wave') st.renameWavePart(renameEdit.index, v);
                  else if (renameEdit.target === 'track') st.renameTrack(renameEdit.index, v);
                }
                setRenameEdit(null);
              } else if (e.key === 'Escape') {
                setRenameEdit(null);
              }
            }}
            onBlur={e => {
              const v = e.target.value.trim();
              if (v) {
                const st = useSynthStore.getState();
                if (renameEdit.target === 'voice') st.renameVoicePart(renameEdit.index, v);
                else if (renameEdit.target === 'wave') st.renameWavePart(renameEdit.index, v);
                else if (renameEdit.target === 'track') st.renameTrack(renameEdit.index, v);
              }
              setRenameEdit(null);
            }}
            style={{
              position: 'absolute', left: renameEdit.x, top: renameEdit.y,
              zIndex: 20, width: 120, height: 18,
              background: 'var(--bg-deep)', color: 'var(--text-primary)',
              border: '1px solid var(--accent-cyan)', outline: 'none',
              fontSize: 10, fontFamily: 'var(--font-pixel)', padding: '0 4px',
            }}
          />
        )}
        {/* Context menu */}
        {contextMenu && (
          <>
            <div className="utai-modal-backdrop" onClick={closeContextMenu} onContextMenu={e => { e.preventDefault(); closeContextMenu(); }} style={{ zIndex: 800, background: 'transparent' }} />
            <div
              className="utai-synth-ctx-menu"
              ref={el => {
                if (!el) return;
                const rect = el.getBoundingClientRect();
                const overflow = rect.bottom - window.innerHeight;
                if (overflow > 0) el.style.top = `${contextMenu.y - overflow - 4}px`;
              }}
              style={{
                position: 'fixed',
                left: contextMenu.x,
                top: contextMenu.y,
                zIndex: 801,
              }}
            >
              <button className="utai-synth-ctx-menu__item" onClick={ctxAddVocal}>
                + {t('synth.track.addVocal')}
              </button>
              <button className="utai-synth-ctx-menu__item" onClick={ctxAddAudio}>
                + {t('synth.track.addAudio')}
              </button>
              <button className="utai-synth-ctx-menu__item" onClick={ctxAddInstrument}>
                + {t('synth.track.addInstrument')}
              </button>
              {/* Segment operations — uses selectedSegment from store, not click position */}
              {(() => {
                const seg = useSynthStore.getState().selectedSegment;
                if (!seg) return null;
                return (
                  <>
                    <div className="utai-synth-ctx-menu__sep" />
                    <button className="utai-synth-ctx-menu__item" onClick={() => {
                      const s = useSynthStore.getState();
                      if (seg.type === 'wave') {
                        s.splitWavePart(seg.index, s.playbackPositionTicks);
                      } else {
                        const ts = s.project.time_signatures[0];
                        const ticksPerBar = RESOLUTION * (ts?.beat_per_bar ?? 4);
                        const splitTick = Math.round(s.playbackPositionTicks / ticksPerBar) * ticksPerBar;
                        if (seg.type === 'voice') s.splitVoicePart(seg.index, splitTick);
                        else s.splitInstrumentPart(seg.index, splitTick);
                      }
                      closeContextMenu();
                    }}>
                      {t('synth.segment.split')}
                    </button>
                    <button className="utai-synth-ctx-menu__item" onClick={() => {
                      const s = useSynthStore.getState();
                      const ppt = s.viewState.pixelsPerTick;
                      const laneH = s.viewState.trackLaneHeight ?? 64;
                      const parts = seg.type === 'voice' ? s.project.voice_parts
                        : seg.type === 'wave' ? s.project.wave_parts : s.project.instrument_parts;
                      const part = parts[seg.index];
                      const curName = (part as any)?.name || '';
                      const trackNo = (part as any)?.track_no ?? 0;
                      const px = TRACK_HEADER_W + ((part as any)?.position ?? 0) * ppt - s.viewState.scrollX;
                      const py = RULER_H + trackNo * laneH - (s.arrScrollY ?? 0);
                      setRenameEdit({ target: seg.type, index: seg.index, x: Math.max(TRACK_HEADER_W, px), y: py, value: curName });
                      closeContextMenu();
                    }}>
                      Rename Clip
                    </button>
                    <button className="utai-synth-ctx-menu__item" onClick={() => {
                      const s = useSynthStore.getState();
                      const parts = seg.type === 'voice' ? s.project.voice_parts
                        : seg.type === 'wave' ? s.project.wave_parts : s.project.instrument_parts;
                      const part = parts[seg.index];
                      if (part) setClipboard({ type: seg.type, data: JSON.parse(JSON.stringify(part)) });
                      closeContextMenu();
                    }}>
                      Copy
                    </button>
                    <button className="utai-synth-ctx-menu__item utai-synth-ctx-menu__item--danger" onClick={() => {
                      const st = useSynthStore.getState();
                      if (seg.type === 'voice') st.deleteVoicePart(seg.index);
                      else if (seg.type === 'instrument') st.deleteInstrumentPart(seg.index);
                      else if (seg.type === 'wave') st.deleteWavePart(seg.index);
                      st.setSelectedSegment(null);
                      closeContextMenu();
                    }}>
                      {t('synth.segment.delete')}
                    </button>
                  </>
                );
              })()}
              {/* Track operations */}
              {contextMenu.trackIdx >= 0 && (
                <>
                  <div className="utai-synth-ctx-menu__sep" />
                  {/* Paste — only when clipboard has content and target track matches type */}
                  {clipboard && (() => {
                    const targetType = useSynthStore.getState().project.tracks[contextMenu.trackIdx]?.track_type || 'vocal';
                    const clipTrackType = clipboard.type === 'voice' ? 'vocal' : clipboard.type === 'wave' ? 'audio' : 'instrument';
                    if (targetType !== clipTrackType) return null;
                    return (
                      <button className="utai-synth-ctx-menu__item" onClick={() => {
                        pasteSegment(clipboard, useSynthStore.getState().playbackPositionTicks, contextMenu.trackIdx);
                        closeContextMenu();
                      }}>
                        Paste at Playhead
                      </button>
                    );
                  })()}
                  <button className="utai-synth-ctx-menu__item" onClick={() => {
                    const s = useSynthStore.getState();
                    const track = s.project.tracks[contextMenu.trackIdx];
                    const laneH = s.viewState.trackLaneHeight ?? 64;
                    const py = RULER_H + contextMenu.trackIdx * laneH - (s.arrScrollY ?? 0);
                    setRenameEdit({ target: 'track', index: contextMenu.trackIdx, x: 4, y: py, value: track?.track_name || '' });
                    closeContextMenu();
                  }}>
                    Rename Track
                  </button>
                  {useSynthStore.getState().project.tracks.length > 1 && (
                    <button className="utai-synth-ctx-menu__item utai-synth-ctx-menu__item--danger" onClick={ctxDeleteTrack}>
                      {t('synth.track.delete')}
                    </button>
                  )}
                </>
              )}
            </div>
          </>
        )}
      </div>
    </div>
  );
}
