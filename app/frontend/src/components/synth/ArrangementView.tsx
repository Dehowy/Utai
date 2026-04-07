/** Arrangement view — multi-track overview with lanes, part blocks, and waveforms.
 *  Canvas-based, similar pattern to PianoRollCanvas.
 */

import { useRef, useEffect, useState, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { useSynthStore } from '../../store/synthStore';
import { renderArrangement, TRACK_HEADER_W, TRACK_LANE_H, RULER_H } from '../../lib/synth/arrangementEngine';
import { RESOLUTION } from '../../types/ustx';
import * as engine from '../../lib/synth/audioEngine';
import TrackMixer from './TrackMixer';

type ArrDragMode = 'none' | 'playhead';

/** Snap a tick position to the nearest bar line. */
function snapToBar(tick: number): number {
  const s = useSynthStore.getState();
  const ts = s.project.time_signatures[0];
  const ticksPerBar = RESOLUTION * (ts?.beat_per_bar ?? 4);
  return Math.round(tick / ticksPerBar) * ticksPerBar;
}

export default function ArrangementView() {
  const { t } = useTranslation();
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const store = useSynthStore();
  const [size, setSize] = useState({ width: 800, height: 400 });
  const dragMode = useRef<ArrDragMode>('none');
  const [contextMenu, setContextMenu] = useState<{ x: number; y: number; trackIdx: number; partHit: { type: 'voice' | 'instrument'; index: number } | null } | null>(null);
  const [mouseX, setMouseX] = useState<number | null>(null);

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

  // Render
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    canvas.width = size.width * devicePixelRatio;
    canvas.height = size.height * devicePixelRatio;
    ctx.scale(devicePixelRatio, devicePixelRatio);
    renderArrangement({
      ctx, width: size.width, height: size.height,
      project: store.project, viewState: store.viewState,
      activePartIndex: store.activePartIndex,
      playheadTicks: store.playbackPositionTicks,
      selectedSegment: store.selectedSegment,
    });

    // Draw mouse guide line (thin white vertical line at mouse position)
    if (mouseX != null && mouseX > TRACK_HEADER_W) {
      ctx.save();
      ctx.strokeStyle = 'rgba(255, 255, 255, 0.2)';
      ctx.lineWidth = 1;
      ctx.setLineDash([4, 4]);
      ctx.beginPath();
      ctx.moveTo(mouseX, RULER_H);
      ctx.lineTo(mouseX, size.height);
      ctx.stroke();
      ctx.setLineDash([]);
      ctx.restore();
    }
  });

  // Double-click on a vocal part → switch to piano roll
  const handleDoubleClick = useCallback((e: React.MouseEvent) => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;

    if (x < TRACK_HEADER_W) return;

    // Double-click on ruler → snap playhead to nearest bar
    if (y < RULER_H && x > TRACK_HEADER_W) {
      const s = useSynthStore.getState();
      const clickTick = (x - TRACK_HEADER_W + s.viewState.scrollX) / s.viewState.pixelsPerTick;
      const snapped = snapToBar(clickTick);
      s.setPlaybackPosition(snapped);
      engine.seek(engine.ticksToSeconds(snapped, s.project.tempos));
      return;
    }

    const trackIdx = Math.floor((y - RULER_H) / TRACK_LANE_H);
    if (trackIdx < 0 || trackIdx >= store.project.tracks.length) return;

    // Find voice part or instrument part for this track
    const vpIdx = store.project.voice_parts.findIndex(vp => vp.track_no === trackIdx);
    if (vpIdx >= 0) {
      useSynthStore.getState().setActivePart(vpIdx);
      useSynthStore.getState().setViewMode('pianoroll');
      return;
    }
    const ipIdx = store.project.instrument_parts.findIndex(ip => ip.track_no === trackIdx);
    if (ipIdx >= 0) {
      useSynthStore.getState().setActiveInstrumentPart(ipIdx);
      useSynthStore.getState().setViewMode('pianoroll');
    }
  }, [store.project]);

  // Mouse down — ruler click sets playhead, start drag
  const handleMouseDown = useCallback((e: React.MouseEvent) => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    if (e.button !== 0) return;
    const rect = canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;

    // Click on ruler → set playhead (snapped to bar) and start dragging
    if (y < RULER_H && x > TRACK_HEADER_W) {
      const s = useSynthStore.getState();
      const rawTick = Math.max(0, (x - TRACK_HEADER_W + s.viewState.scrollX) / s.viewState.pixelsPerTick);
      const tick = snapToBar(rawTick);
      s.setPlaybackPosition(tick);
      engine.seek(engine.ticksToSeconds(tick, s.project.tempos));
      dragMode.current = 'playhead';
      return;
    }
  }, []);

  // Mouse move — drag playhead + cursor styling + guide line
  const handleMouseMove = useCallback((e: React.MouseEvent) => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;

    // Update guide line
    setMouseX(x > TRACK_HEADER_W ? x : null);

    // Cursor styling
    if (dragMode.current === 'playhead') {
      canvas.style.cursor = 'col-resize';
    } else if (y < RULER_H && x > TRACK_HEADER_W) {
      canvas.style.cursor = 'col-resize';
    } else {
      canvas.style.cursor = 'default';
    }

    if (dragMode.current !== 'playhead') return;
    if (x > TRACK_HEADER_W) {
      const s = useSynthStore.getState();
      const rawTick = Math.max(0, (x - TRACK_HEADER_W + s.viewState.scrollX) / s.viewState.pixelsPerTick);
      const tick = snapToBar(rawTick);
      s.setPlaybackPosition(tick);
      engine.seek(engine.ticksToSeconds(tick, s.project.tempos));
    }
  }, []);

  // Mouse up — end drag
  const handleMouseUp = useCallback(() => {
    dragMode.current = 'none';
  }, []);

  // Helper: find which part (voice or instrument) is at a click position
  const hitTestPart = useCallback((x: number, y: number): { type: 'voice' | 'instrument'; index: number } | null => {
    if (x < TRACK_HEADER_W || y < RULER_H) return null;
    const s = useSynthStore.getState();
    const ppt = s.viewState.pixelsPerTick;
    const clickTick = (x - TRACK_HEADER_W + s.viewState.scrollX) / ppt;
    const trackIdx = Math.floor((y - RULER_H) / TRACK_LANE_H);

    for (let i = 0; i < s.project.voice_parts.length; i++) {
      const vp = s.project.voice_parts[i];
      if (vp.track_no === trackIdx && clickTick >= vp.position && clickTick <= vp.position + vp.duration) {
        return { type: 'voice', index: i };
      }
    }
    for (let i = 0; i < s.project.instrument_parts.length; i++) {
      const ip = s.project.instrument_parts[i];
      if (ip.track_no === trackIdx && clickTick >= ip.position && clickTick <= ip.position + ip.duration) {
        return { type: 'instrument', index: i };
      }
    }
    return null;
  }, []);

  // Click handler — behavior depends on active tool
  const handleClick = useCallback((e: React.MouseEvent) => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;

    if (y < RULER_H) return;

    const trackIdx = Math.floor((y - RULER_H) / TRACK_LANE_H);

    // Header click: mute/solo
    if (x < TRACK_HEADER_W && trackIdx >= 0 && trackIdx < store.project.tracks.length) {
      if (e.altKey) {
        useSynthStore.getState().setTrackSolo(trackIdx, !store.project.tracks[trackIdx].solo);
      } else if (x > TRACK_HEADER_W - 35) {
        useSynthStore.getState().setTrackMute(trackIdx, !store.project.tracks[trackIdx].mute);
      }
      return;
    }

    if (x <= TRACK_HEADER_W) return;

    const s = useSynthStore.getState();
    const tool = s.activeTool;
    const hit = hitTestPart(x, y);

    if (tool === 'select') {
      // Select segment
      s.setSelectedSegment(hit);
    } else if (tool === 'erase' && hit) {
      // Delete segment
      if (hit.type === 'voice') s.deleteVoicePart(hit.index);
      else s.deleteInstrumentPart(hit.index);
    }
    // 'draw' (create) is handled via context menu / split button
  }, [store.project, hitTestPart]);

  // Right-click context menu
  const handleContextMenu = useCallback((e: React.MouseEvent) => {
    e.preventDefault();
    const canvas = canvasRef.current;
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;

    // Determine which track was right-clicked
    const trackIdx = y >= RULER_H ? Math.floor((y - RULER_H) / TRACK_LANE_H) : -1;
    const validTrack = trackIdx >= 0 && trackIdx < useSynthStore.getState().project.tracks.length;

    const partHit = hitTestPart(x, y);
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
      if (e.ctrlKey) {
        const delta = e.deltaY > 0 ? 0.9 : 1.1;
        s.setViewState({ pixelsPerTick: Math.max(0.03, Math.min(5, s.viewState.pixelsPerTick * delta)) });
      } else if (e.shiftKey) {
        s.setViewState({ scrollX: Math.max(0, s.viewState.scrollX + e.deltaY) });
      } else {
        s.setViewState({ scrollX: Math.max(0, s.viewState.scrollX + e.deltaX + e.deltaY) });
      }
    };
    canvas.addEventListener('wheel', handler, { passive: false });
    return () => canvas.removeEventListener('wheel', handler);
  }, []);

  // Track management buttons
  const addVocalTrack = () => store.addTrack('vocal');
  const addAudioTrack = () => store.addTrack('audio');
  const addInstrumentTrack = () => store.addTrack('instrument');

  // Close context menu
  const closeContextMenu = useCallback(() => setContextMenu(null), []);

  // Context menu actions
  const ctxAddVocal = useCallback(() => { store.addTrack('vocal'); closeContextMenu(); }, [store, closeContextMenu]);
  const ctxAddAudio = useCallback(() => { store.addTrack('audio'); closeContextMenu(); }, [store, closeContextMenu]);
  const ctxAddInstrument = useCallback(() => { store.addTrack('instrument'); closeContextMenu(); }, [store, closeContextMenu]);
  const ctxDeleteTrack = useCallback(() => {
    if (contextMenu && contextMenu.trackIdx >= 0) {
      store.removeTrack(contextMenu.trackIdx);
    }
    closeContextMenu();
  }, [store, contextMenu, closeContextMenu]);

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
          onMouseLeave={() => { handleMouseUp(); setMouseX(null); }}
          onDoubleClick={handleDoubleClick}
          onContextMenu={handleContextMenu}
        />
        {/* Context menu */}
        {contextMenu && (
          <>
            <div className="utai-modal-backdrop" onClick={closeContextMenu} onContextMenu={e => { e.preventDefault(); closeContextMenu(); }} style={{ zIndex: 800, background: 'transparent' }} />
            <div
              className="utai-synth-ctx-menu"
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
              {/* Segment operations */}
              {contextMenu.partHit && (
                <>
                  <div className="utai-synth-ctx-menu__sep" />
                  <button className="utai-synth-ctx-menu__item" onClick={() => {
                    const s = useSynthStore.getState();
                    const ts = s.project.time_signatures[0];
                    const ticksPerBar = RESOLUTION * (ts?.beat_per_bar ?? 4);
                    const splitTick = Math.round(s.playbackPositionTicks / ticksPerBar) * ticksPerBar;
                    if (contextMenu.partHit!.type === 'voice') s.splitVoicePart(contextMenu.partHit!.index, splitTick);
                    else s.splitInstrumentPart(contextMenu.partHit!.index, splitTick);
                    closeContextMenu();
                  }}>
                    {t('synth.segment.split')}
                  </button>
                  <button className="utai-synth-ctx-menu__item utai-synth-ctx-menu__item--danger" onClick={() => {
                    if (contextMenu.partHit!.type === 'voice') store.deleteVoicePart(contextMenu.partHit!.index);
                    else store.deleteInstrumentPart(contextMenu.partHit!.index);
                    closeContextMenu();
                  }}>
                    {t('synth.segment.delete')}
                  </button>
                </>
              )}
              {!contextMenu.partHit && contextMenu.trackIdx >= 0 && (
                <>
                  <div className="utai-synth-ctx-menu__sep" />
                  <button className="utai-synth-ctx-menu__item" onClick={() => {
                    const s = useSynthStore.getState();
                    const ts = s.project.time_signatures[0];
                    const ticksPerBar = RESOLUTION * (ts?.beat_per_bar ?? 4);
                    const pos = Math.round(s.playbackPositionTicks / ticksPerBar) * ticksPerBar;
                    const tt = store.project.tracks[contextMenu.trackIdx]?.track_type || 'vocal';
                    if (tt === 'instrument') s.addInstrumentPartOnTrack(contextMenu.trackIdx, pos);
                    else s.addVoicePartOnTrack(contextMenu.trackIdx, pos);
                    closeContextMenu();
                  }}>
                    {t('synth.segment.create')}
                  </button>
                </>
              )}
              {contextMenu.trackIdx >= 0 && store.project.tracks.length > 1 && (
                <>
                  <div className="utai-synth-ctx-menu__sep" />
                  <button className="utai-synth-ctx-menu__item utai-synth-ctx-menu__item--danger" onClick={ctxDeleteTrack}>
                    {t('synth.track.delete')}
                  </button>
                </>
              )}
            </div>
          </>
        )}
      </div>
    </div>
  );
}
