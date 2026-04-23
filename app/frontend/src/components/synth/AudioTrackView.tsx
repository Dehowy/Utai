/** Audio track editor — waveform display, bar lines, timestamps, segment splitting. */

import { useRef, useEffect, useState, useCallback } from 'react';
import { useSynthStore } from '../../store/synthStore';
import { RESOLUTION } from '../../types/ustx';
import { getAudioURL } from '../../api/client';
import * as engine from '../../lib/synth/audioEngine';

const RULER_H = 24;

export default function AudioTrackView() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const project = useSynthStore(s => s.project);
  const viewState = useSynthStore(s => s.viewState);
  const playbackPositionTicks = useSynthStore(s => s.playbackPositionTicks);
  const activeWavePartIndex = useSynthStore(s => s.activeWavePartIndex);
  const storePeakData = useSynthStore(s => s.peakData);
  const [size, setSize] = useState({ width: 800, height: 400 });
  const dragRafId = useRef<number | null>(null);

  const wavePartIdx = activeWavePartIndex;
  const wavePart = project.wave_parts[wavePartIdx];
  const peakData = wavePart?.buffer_id ? storePeakData[wavePart.buffer_id] : undefined;

  // Cleanup rAF on unmount
  useEffect(() => {
    return () => { if (dragRafId.current != null) cancelAnimationFrame(dragRafId.current); };
  }, []);

  // Resize observer
  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;
    const obs = new ResizeObserver(entries => {
      const { width, height } = entries[0].contentRect;
      setSize({ width: Math.floor(width), height: Math.floor(height) });
    });
    obs.observe(container);
    return () => obs.disconnect();
  }, []);

  // Render
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    const dpr = window.devicePixelRatio || 1;
    canvas.width = size.width * dpr;
    canvas.height = size.height * dpr;
    ctx.scale(dpr, dpr);

    const { width, height } = size;
    const vs = viewState;
    const bpm = project.tempos[0]?.bpm ?? 120;
    const ts = project.time_signatures[0];
    const beatsPerBar = ts?.beat_per_bar ?? 4;
    const ticksPerBar = RESOLUTION * beatsPerBar;
    const ticksPerSec = bpm / 60 * RESOLUTION;

    // Background
    ctx.fillStyle = '#0d0d1a';
    ctx.fillRect(0, 0, width, height);

    // Ruler
    ctx.fillStyle = '#111125';
    ctx.fillRect(0, 0, width, RULER_H);

    const startTick = Math.floor(vs.scrollX / vs.pixelsPerTick);
    const endTick = Math.ceil((vs.scrollX + width) / vs.pixelsPerTick);

    // Adaptive bar line interval: skip bars when too dense
    const barWidthPx = ticksPerBar * vs.pixelsPerTick;
    const barStep = barWidthPx < 20 ? 8 : barWidthPx < 40 ? 4 : barWidthPx < 80 ? 2 : 1;

    const firstBar = Math.floor(startTick / ticksPerBar / barStep) * barStep;
    ctx.font = "10px 'VT323', monospace";
    ctx.textBaseline = 'middle';
    for (let bar = firstBar; bar * ticksPerBar <= endTick; bar += barStep) {
      const tick = bar * ticksPerBar;
      const x = tick * vs.pixelsPerTick - vs.scrollX;
      if (x < -20 || x > width + 20) continue;

      ctx.strokeStyle = '#3a3a5a';
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(x, RULER_H);
      ctx.lineTo(x, height);
      ctx.stroke();

      // Bar number
      ctx.fillStyle = '#8888a0';
      ctx.fillText(`${bar + 1}`, x + 3, RULER_H / 2 - 4);

      // Timestamp
      const sec = tick / ticksPerSec;
      const mm = Math.floor(sec / 60);
      const ss = Math.floor(sec % 60);
      const ms = Math.floor((sec % 1) * 100);
      ctx.fillStyle = '#5a5a7a';
      ctx.fillText(`${mm}:${String(ss).padStart(2, '0')}.${String(ms).padStart(2, '0')}`, x + 3, RULER_H / 2 + 6);
    }

    // Beat sub-lines (only when zoomed in enough)
    if (barWidthPx >= 40) {
      const beatStep = barWidthPx < 80 ? RESOLUTION * 2 : RESOLUTION;
      const firstBeat = Math.floor(startTick / beatStep) * beatStep;
      for (let tick = firstBeat; tick <= endTick; tick += beatStep) {
        if (tick % (ticksPerBar * barStep) === 0) continue;
        const x = tick * vs.pixelsPerTick - vs.scrollX;
        if (x < 0 || x > width) continue;
        ctx.strokeStyle = '#1e1e3a';
        ctx.lineWidth = 0.5;
        ctx.beginPath();
        ctx.moveTo(x, RULER_H);
        ctx.lineTo(x, height);
        ctx.stroke();
      }
    }

    // Draw wave parts on this track
    if (wavePart) {
      const trackParts = project.wave_parts.filter(wp => wp.track_no === wavePart.track_no);
      for (const wp of trackParts) {
        const wpPeaks = wp.buffer_id ? storePeakData[wp.buffer_id] : undefined;
        const durTicks = Math.round((wp.file_duration_ms / 1000) * ticksPerSec);
        const x0 = wp.position * vs.pixelsPerTick - vs.scrollX;
        const w = durTicks * vs.pixelsPerTick;
        // Height scales with pixelsPerSemitone — always centered in viewport
        const waveScale = vs.pixelsPerSemitone / 20;
        const viewH = height - RULER_H - 16;
        const waveH = Math.max(20, viewH * waveScale);
        const viewMidY = RULER_H + (height - RULER_H) / 2;
        const midY = viewMidY - (waveH <= viewH ? 0 : (vs.scrollY || 0));
        const isActive = project.wave_parts.indexOf(wp) === wavePartIdx;

        // Background block
        ctx.fillStyle = isActive ? '#1a1a3a' : '#121228';
        ctx.fillRect(Math.max(0, x0), RULER_H, Math.min(w, width - Math.max(0, x0)), height - RULER_H);

        // Segment boundary
        if (x0 > 0) {
          ctx.strokeStyle = '#5ce0d8';
          ctx.lineWidth = 1;
          ctx.setLineDash([3, 3]);
          ctx.beginPath();
          ctx.moveTo(x0, RULER_H);
          ctx.lineTo(x0, height);
          ctx.stroke();
          ctx.setLineDash([]);
        }

        // Waveform — slice peaks to match this segment's skip/duration
        if (wpPeaks && wpPeaks.length > 1 && w > 0) {
          const totalMs = wp._fileTotalMs || Math.max(...project.wave_parts
            .filter(w2 => w2.buffer_id === wp.buffer_id)
            .map(w2 => (w2.skip || 0) + w2.file_duration_ms));
          const si = Math.round(((wp.skip || 0) / totalMs) * wpPeaks.length);
          const ei = Math.min(wpPeaks.length, Math.round((((wp.skip || 0) + wp.file_duration_ms) / totalMs) * wpPeaks.length));
          const peaks = wpPeaks.subarray(si, ei);
          if (peaks.length > 1) {
          const color = isActive ? '#5ce0d8' : '#3a6a6a';
          ctx.fillStyle = color;
          ctx.globalAlpha = 0.25;
          ctx.beginPath();
          for (let i = 0; i < peaks.length; i++) {
            const px = x0 + (i / (peaks.length - 1)) * w;
            const amp = peaks[i] * (waveH / 2);
            if (i === 0) ctx.moveTo(px, midY - amp);
            else ctx.lineTo(px, midY - amp);
          }
          for (let i = peaks.length - 1; i >= 0; i--) {
            const px = x0 + (i / (peaks.length - 1)) * w;
            const amp = peaks[i] * (waveH / 2);
            ctx.lineTo(px, midY + amp);
          }
          ctx.closePath();
          ctx.fill();

          ctx.strokeStyle = color;
          ctx.globalAlpha = 0.6;
          ctx.lineWidth = 1;
          ctx.beginPath();
          for (let i = 0; i < peaks.length; i++) {
            const px = x0 + (i / (peaks.length - 1)) * w;
            const amp = peaks[i] * (waveH / 2);
            if (i === 0) ctx.moveTo(px, midY - amp);
            else ctx.lineTo(px, midY - amp);
          }
          ctx.stroke();
          ctx.beginPath();
          for (let i = 0; i < peaks.length; i++) {
            const px = x0 + (i / (peaks.length - 1)) * w;
            const amp = peaks[i] * (waveH / 2);
            if (i === 0) ctx.moveTo(px, midY + amp);
            else ctx.lineTo(px, midY + amp);
          }
          ctx.stroke();
          ctx.globalAlpha = 1;
          }
        }

        // Part name
        ctx.fillStyle = '#d0d0e8';
        ctx.font = "10px 'DotGothic16', sans-serif";
        ctx.textBaseline = 'top';
        ctx.fillText(wp.name, Math.max(x0 + 4, 4), RULER_H + 4, w - 8);
      }
    }

    // Redraw ruler background to clip any waveform overflow
    ctx.fillStyle = '#111125';
    ctx.fillRect(0, 0, width, RULER_H);

    // Redraw bar numbers + timestamps on top of clipped ruler
    const firstBar2 = Math.floor(startTick / ticksPerBar / barStep) * barStep;
    ctx.font = "10px 'VT323', monospace";
    ctx.textBaseline = 'middle';
    for (let bar = firstBar2; bar * ticksPerBar <= endTick; bar += barStep) {
      const tick = bar * ticksPerBar;
      const bx = tick * vs.pixelsPerTick - vs.scrollX;
      if (bx < -20 || bx > width + 20) continue;
      ctx.fillStyle = '#8888a0';
      ctx.fillText(`${bar + 1}`, bx + 3, RULER_H / 2 - 4);
      const sec = tick / ticksPerSec;
      ctx.fillStyle = '#5a5a7a';
      ctx.fillText(`${Math.floor(sec / 60)}:${String(Math.floor(sec % 60)).padStart(2, '0')}.${String(Math.floor((sec % 1) * 100)).padStart(2, '0')}`, bx + 3, RULER_H / 2 + 6);
    }

    // Ruler bottom border
    ctx.strokeStyle = '#4a4a6a';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(0, RULER_H);
    ctx.lineTo(width, RULER_H);
    ctx.stroke();

    // Playhead line + triangle
    const playX = playbackPositionTicks * vs.pixelsPerTick - vs.scrollX;
    if (playX >= 0 && playX <= width) {
      ctx.save();
      ctx.strokeStyle = '#ff6b9d';
      ctx.lineWidth = 2;
      ctx.shadowColor = '#ff6b9d';
      ctx.shadowBlur = 6;
      ctx.beginPath();
      ctx.moveTo(playX, RULER_H);
      ctx.lineTo(playX, height);
      ctx.stroke();
      ctx.restore();
      // Triangle on ruler
      ctx.fillStyle = '#ff6b9d';
      ctx.beginPath();
      ctx.moveTo(playX - 5, 0);
      ctx.lineTo(playX + 5, 0);
      ctx.lineTo(playX, RULER_H - 2);
      ctx.closePath();
      ctx.fill();
    }

  }, [size, project, viewState, playbackPositionTicks, activeWavePartIndex, storePeakData]);

  // Playhead drag state
  const draggingPlayhead = useRef(false);

  // Context menu state
  const [ctxMenu, setCtxMenu] = useState<{ x: number; y: number; tick: number } | null>(null);

  // Find which wave part a tick belongs to
  const hitWavePart = useCallback((tick: number): number => {
    const s = useSynthStore.getState();
    const wp0 = s.project.wave_parts[s.activeWavePartIndex];
    if (!wp0) return -1;
    const bpm = s.project.tempos[0]?.bpm ?? 120;
    const tps = bpm / 60 * RESOLUTION;
    return s.project.wave_parts.findIndex(wp =>
      wp.track_no === wp0.track_no && tick >= wp.position && tick < wp.position + (wp.file_duration_ms / 1000) * tps
    );
  }, []);

  const handleMouseDown = useCallback((e: React.MouseEvent) => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    if (e.button !== 0) return;
    setCtxMenu(null);
    const rect = canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    const s = useSynthStore.getState();
    const tick = Math.max(0, (x + s.viewState.scrollX) / s.viewState.pixelsPerTick);

    // Click on content area: select segment only, don't move playhead
    if (y > RULER_H) {
      const hitIdx = hitWavePart(tick);
      if (hitIdx >= 0 && hitIdx !== s.activeWavePartIndex) {
        useSynthStore.setState({ activeWavePartIndex: hitIdx });
      }
      return;
    }

    // Click on ruler: playhead drag
    s.setPlaybackPosition(tick);
    draggingPlayhead.current = true;

    const wasPlaying = useSynthStore.getState().isPlaying;
    if (wasPlaying) engine.pause();

    // Continuous auto-scroll via rAF
    let scrollSpeed = 0;
    let mouseX = x;
    const scrollFrame = () => {
      if (!draggingPlayhead.current) return;
      const st = useSynthStore.getState();
      if (scrollSpeed !== 0) {
        st.setViewState({ scrollX: Math.max(0, st.viewState.scrollX + scrollSpeed) });
        // Update playhead to track the scroll
        const r = canvas.getBoundingClientRect();
        const clampedX = Math.max(0, Math.min(r.width, mouseX));
        const st2 = useSynthStore.getState();
        const t = Math.max(0, (clampedX + st2.viewState.scrollX) / st2.viewState.pixelsPerTick);
        st2.setPlaybackPosition(t);
      }
      dragRafId.current = requestAnimationFrame(scrollFrame);
    };
    dragRafId.current = requestAnimationFrame(scrollFrame);

    const onMove = (ev: MouseEvent) => {
      const r = canvas.getBoundingClientRect();
      const mx = ev.clientX - r.left;
      const w = r.width;
      mouseX = mx;

      // Set scroll speed based on edge proximity
      if (mx > w - 40) {
        scrollSpeed = Math.min(15, (mx - w + 40) / 3);
      } else if (mx < 40) {
        scrollSpeed = -Math.min(15, (40 - mx) / 3);
      } else {
        scrollSpeed = 0;
      }

      // Update playhead position
      const clampedX = Math.max(0, Math.min(w, mx));
      const st = useSynthStore.getState();
      const t = Math.max(0, (clampedX + st.viewState.scrollX) / st.viewState.pixelsPerTick);
      st.setPlaybackPosition(t);
    };
    const onUp = () => {
      draggingPlayhead.current = false;
      if (dragRafId.current != null) { cancelAnimationFrame(dragRafId.current); dragRafId.current = null; }
      window.removeEventListener('mousemove', onMove);
      window.removeEventListener('mouseup', onUp);
      const st = useSynthStore.getState();
      engine.seek(engine.ticksToSeconds(st.playbackPositionTicks, st.project.tempos));
      if (wasPlaying) {
        const playBtn = document.querySelector('.utai-synth-transport__btn') as HTMLButtonElement | null;
        if (playBtn) playBtn.click();
      }
    };
    window.addEventListener('mousemove', onMove);
    window.addEventListener('mouseup', onUp);
  }, [hitWavePart]);

  const handleContextMenu = useCallback((e: React.MouseEvent) => {
    e.preventDefault();
    const canvas = canvasRef.current;
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const s = useSynthStore.getState();
    const tick = Math.max(0, (x + s.viewState.scrollX) / s.viewState.pixelsPerTick);

    // Select segment under cursor, don't move playhead
    const hitIdx = hitWavePart(tick);
    if (hitIdx >= 0 && hitIdx !== s.activeWavePartIndex) {
      useSynthStore.setState({ activeWavePartIndex: hitIdx });
    }

    setCtxMenu({ x: e.clientX - rect.left, y: e.clientY - rect.top, tick });
  }, [hitWavePart]);

  // Tab key held state
  const tabHeld = useRef(false);
  useEffect(() => {
    const down = (e: KeyboardEvent) => { if (e.key === 'Tab') tabHeld.current = true; };
    const up = (e: KeyboardEvent) => { if (e.key === 'Tab') tabHeld.current = false; };
    window.addEventListener('keydown', down, true);
    window.addEventListener('keyup', up, true);
    return () => { window.removeEventListener('keydown', down, true); window.removeEventListener('keyup', up, true); };
  }, []);

  // Wheel: scroll = horiz, shift+scroll = vert, ctrl = horiz zoom, tab = vert zoom
  useEffect(() => {
    const canvas = containerRef.current;
    if (!canvas) return;
    const handler = (e: WheelEvent) => {
      e.preventDefault();
      const s = useSynthStore.getState();
      if (e.ctrlKey) {
        // Horizontal zoom anchored at mouse position
        const rect = canvas.getBoundingClientRect();
        const mx = e.clientX - rect.left;
        const factor = e.deltaY < 0 ? 1.15 : 1 / 1.15;
        const oldPPT = s.viewState.pixelsPerTick;
        const newPPT = Math.max(0.005, Math.min(5, oldPPT * factor));
        // Keep the tick under the mouse stationary
        const tickUnderMouse = (mx + s.viewState.scrollX) / oldPPT;
        const newScrollX = Math.max(0, tickUnderMouse * newPPT - mx);
        s.setViewState({ pixelsPerTick: newPPT, scrollX: newScrollX });
      } else if (tabHeld.current) {
        // Vertical zoom (wave height) — center-anchored
        const factor = e.deltaY < 0 ? 1.15 : 1 / 1.15;
        const newPPS = Math.max(2, Math.min(80, s.viewState.pixelsPerSemitone * factor));
        // Reset scrollY to keep centered after zoom
        const waveScale = newPPS / 20;
        const viewH = size.height - RULER_H - 16;
        const waveH = viewH * waveScale;
        const scrollY = waveH <= viewH ? 0 : Math.min(s.viewState.scrollY, (waveH - viewH) / 2);
        s.setViewState({ pixelsPerSemitone: newPPS, scrollY });
      } else if (e.shiftKey) {
        // Vertical scroll — clamp to waveform bounds
        const waveScale = s.viewState.pixelsPerSemitone / 20;
        const viewH = size.height - RULER_H - 16;
        const waveH = viewH * waveScale;
        if (waveH <= viewH) return; // fits in screen, no scroll needed
        const maxScroll = (waveH - viewH) / 2;
        const newY = Math.max(-maxScroll, Math.min(maxScroll, (s.viewState.scrollY || 0) + (e.deltaX || e.deltaY)));
        s.setViewState({ scrollY: newY });
      } else {
        // Horizontal scroll
        s.setViewState({ scrollX: Math.max(0, s.viewState.scrollX + (e.deltaX || e.deltaY)) });
      }
    };
    canvas.addEventListener('wheel', handler, { passive: false });
    return () => canvas.removeEventListener('wheel', handler);
  }, [size.height]);

  const closeCtxMenu = useCallback(() => setCtxMenu(null), []);

  return (
    <div ref={containerRef} style={{ flex: 1, overflow: 'hidden', position: 'relative' }}>
      <canvas
        ref={canvasRef}
        style={{ width: size.width, height: size.height, display: 'block' }}
        onMouseDown={handleMouseDown}
        onContextMenu={handleContextMenu}
      />
      {ctxMenu && (
        <>
          <div style={{ position: 'fixed', inset: 0, zIndex: 800, background: 'transparent' }}
            onClick={closeCtxMenu} onContextMenu={e => { e.preventDefault(); closeCtxMenu(); }} />
          <div style={{
            position: 'absolute', left: ctxMenu.x, top: ctxMenu.y, zIndex: 801,
            background: 'var(--bg-panel)', border: '1px solid var(--border-mid)',
            padding: 4, minWidth: 120,
          }}>
            <button className="utai-synth-ctx-menu__item" onClick={() => {
              const s = useSynthStore.getState();
              s.splitWavePart(s.activeWavePartIndex, s.playbackPositionTicks);
              closeCtxMenu();
            }}>
              Split at Playhead (Ctrl+K)
            </button>
            <button className="utai-synth-ctx-menu__item utai-synth-ctx-menu__item--danger" onClick={() => {
              const s = useSynthStore.getState();
              const idx = s.activeWavePartIndex;
              if (idx >= 0) {
                s.deleteWavePart(idx);
                useSynthStore.setState({ activeWavePartIndex: Math.max(0, idx - 1) });
              }
              closeCtxMenu();
            }}>
              Delete Clip
            </button>
          </div>
        </>
      )}
    </div>
  );
}
