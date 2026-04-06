/** Arrangement view — multi-track overview with lanes, part blocks, and waveforms.
 *  Canvas-based, similar pattern to PianoRollCanvas.
 */

import { useRef, useEffect, useState, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { useSynthStore } from '../../store/synthStore';
import { renderArrangement, TRACK_HEADER_W, TRACK_LANE_H, RULER_H } from '../../lib/synth/arrangementEngine';

export default function ArrangementView() {
  const { t } = useTranslation();
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const store = useSynthStore();
  const [size, setSize] = useState({ width: 800, height: 400 });

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
      playheadTicks: store.isPlaying || store.playbackPositionTicks > 0 ? store.playbackPositionTicks : undefined,
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

    // Find voice part for this track
    const vpIdx = store.project.voice_parts.findIndex(vp => vp.track_no === trackIdx);
    if (vpIdx >= 0) {
      useSynthStore.getState().setActivePart(vpIdx);
      useSynthStore.getState().setViewMode('pianoroll');
    }
  }, [store.project]);

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

  // Right-click context actions
  const handleContextMenu = useCallback((e: React.MouseEvent) => {
    e.preventDefault();
    // Could add context menu here in the future
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
      </div>
      {/* Canvas */}
      <div ref={containerRef} style={{ flex: 1, overflow: 'hidden', position: 'relative' }}>
        <canvas
          ref={canvasRef}
          style={{ width: size.width, height: size.height, display: 'block' }}
          onClick={handleClick}
          onDoubleClick={handleDoubleClick}
          onContextMenu={handleContextMenu}
          onWheel={handleWheel}
        />
      </div>
    </div>
  );
}
