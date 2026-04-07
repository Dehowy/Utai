/** Arrangement view — multi-track overview with lanes, part blocks, and waveforms.
 *  Canvas-based, similar pattern to PianoRollCanvas.
 */

import { useRef, useEffect, useState, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { useSynthStore } from '../../store/synthStore';
import { renderArrangement, TRACK_HEADER_W, TRACK_LANE_H, RULER_H } from '../../lib/synth/arrangementEngine';
import * as engine from '../../lib/synth/audioEngine';

type ArrDragMode = 'none' | 'playhead';

export default function ArrangementView() {
  const { t } = useTranslation();
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const store = useSynthStore();
  const [size, setSize] = useState({ width: 800, height: 400 });
  const dragMode = useRef<ArrDragMode>('none');
  const [contextMenu, setContextMenu] = useState<{ x: number; y: number; trackIdx: number } | null>(null);

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
    });
  });

  // Double-click on a vocal part → switch to piano roll
  const handleDoubleClick = useCallback((e: React.MouseEvent) => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;

    if (x < TRACK_HEADER_W || y < RULER_H) return;

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

    // Click on ruler → set playhead and start dragging
    if (y < RULER_H && x > TRACK_HEADER_W) {
      const s = useSynthStore.getState();
      const tick = Math.max(0, (x - TRACK_HEADER_W + s.viewState.scrollX) / s.viewState.pixelsPerTick);
      s.setPlaybackPosition(tick);
      engine.seek(engine.ticksToSeconds(tick, s.project.tempos));
      dragMode.current = 'playhead';
      return;
    }
  }, []);

  // Mouse move — drag playhead + cursor styling
  const handleMouseMove = useCallback((e: React.MouseEvent) => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;

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
      const tick = Math.max(0, (x - TRACK_HEADER_W + s.viewState.scrollX) / s.viewState.pixelsPerTick);
      s.setPlaybackPosition(tick);
      engine.seek(engine.ticksToSeconds(tick, s.project.tempos));
    }
  }, []);

  // Mouse up — end drag
  const handleMouseUp = useCallback(() => {
    dragMode.current = 'none';
  }, []);

  // Click on track header → select track
  const handleClick = useCallback((e: React.MouseEvent) => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;

    if (y < RULER_H) return;

    const trackIdx = Math.floor((y - RULER_H) / TRACK_LANE_H);
    if (trackIdx < 0 || trackIdx >= store.project.tracks.length) return;

    // Header click: toggle mute
    if (x < TRACK_HEADER_W) {
      if (e.altKey) {
        // Alt+click: toggle solo
        useSynthStore.getState().setTrackSolo(trackIdx, !store.project.tracks[trackIdx].solo);
      } else if (x > TRACK_HEADER_W - 35) {
        // Click near right edge of header: toggle mute
        useSynthStore.getState().setTrackMute(trackIdx, !store.project.tracks[trackIdx].mute);
      }
    }
  }, [store.project]);

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

    setContextMenu({
      x: e.clientX,
      y: e.clientY,
      trackIdx: validTrack ? trackIdx : -1,
    });
  }, []);

  // Wheel scroll
  const handleWheel = useCallback((e: React.WheelEvent) => {
    e.preventDefault();
    if (e.ctrlKey) {
      // Zoom
      const delta = e.deltaY > 0 ? 0.9 : 1.1;
      const newPPT = Math.max(0.05, Math.min(5, store.viewState.pixelsPerTick * delta));
      useSynthStore.getState().setViewState({ pixelsPerTick: newPPT });
    } else if (e.shiftKey) {
      // Horizontal scroll
      useSynthStore.getState().setViewState({ scrollX: Math.max(0, store.viewState.scrollX + e.deltaY) });
    } else {
      // Vertical scroll (for many tracks)
      // No vertical scroll needed for now — tracks fit
    }
  }, [store.viewState]);

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
        <button className="utai-synth-arrangement__action-btn" onClick={addVocalTrack}>
          + {t('synth.track.addVocal')}
        </button>
        <button className="utai-synth-arrangement__action-btn" onClick={addAudioTrack}>
          + {t('synth.track.addAudio')}
        </button>
        <button className="utai-synth-arrangement__action-btn" onClick={addInstrumentTrack} style={{ color: 'var(--accent-orange)', borderColor: 'var(--accent-orange)' }}>
          + {t('synth.track.addInstrument')}
        </button>
      </div>
      {/* Canvas */}
      <div ref={containerRef} style={{ flex: 1, overflow: 'hidden', position: 'relative' }}>
        <canvas
          ref={canvasRef}
          style={{ width: size.width, height: size.height, display: 'block', cursor: dragMode.current === 'playhead' ? 'col-resize' : undefined }}
          onClick={handleClick}
          onMouseDown={handleMouseDown}
          onMouseMove={handleMouseMove}
          onMouseUp={handleMouseUp}
          onMouseLeave={handleMouseUp}
          onDoubleClick={handleDoubleClick}
          onContextMenu={handleContextMenu}
          onWheel={handleWheel}
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
