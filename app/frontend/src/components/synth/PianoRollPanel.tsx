import { useCallback, useRef, useEffect, useState } from 'react';
import { useSynthStore } from '../../store/synthStore';
import PianoRollToolbar from './PianoRollToolbar';
import PianoRollCanvas from './PianoRollCanvas';
import ArrangementView, { isDragging } from './ArrangementView';
import AudioTrackView from './AudioTrackView';
import TransportBar from './TransportBar';
import { KEYBOARD_W } from '../../lib/synth/pianoRollEngine';

const MIN_BARS = 8;
const EXTEND_BARS = 8;

export default function PianoRollPanel() {
  const selectedNotes = useSynthStore(s => s.selectedNotes);
  const setTool = useSynthStore(s => s.setTool);
  const undo = useSynthStore(s => s.undo);
  const redo = useSynthStore(s => s.redo);
  const deleteNotes = useSynthStore(s => s.deleteNotes);
  const selectAll = useSynthStore(s => s.selectAll);
  const scrollX = useSynthStore(s => s.viewState.scrollX);
  const pixelsPerTick = useSynthStore(s => s.viewState.pixelsPerTick);
  const setViewState = useSynthStore(s => s.setViewState);
  const project = useSynthStore(s => s.project);
  const isPlaying = useSynthStore(s => s.isPlaying);
  const viewMode = useSynthStore(s => s.viewMode);
  const activeWaveMode = useSynthStore(s => s.activeWavePartIndex) >= 0;
  const activePartIndex = useSynthStore(s => s.activePartIndex);

  const activePart = project.voice_parts[activePartIndex];
  const hasNextSegment = activePart ? project.voice_parts.some(
    vp => vp.track_no === activePart.track_no && vp.position > activePart.position
  ) : false;

  const scrollbarRef = useRef<HTMLDivElement>(null);
  const isUserScrolling = useRef(false);
  const extendingRef = useRef(false);

  const ts = project.time_signatures[0];
  const ticksPerBar = 480 * (ts?.beat_per_bar ?? 4);

  // Scrollbar range depends on mode
  const partStartTick = activePart?.position ?? 0;
  const partEndTick = activePart ? activePart.position + activePart.duration : 0;
  const partDurationTicks = partEndTick - partStartTick;
  const partBars = Math.ceil(partDurationTicks / ticksPerBar) + (hasNextSegment ? 0 : 2);
  const [extraBars, setExtraBars] = useState(MIN_BARS);
  const bpmVal = project.tempos[0]?.bpm ?? 120;
  const tpsVal = bpmVal / 60 * 480;

  // Arrangement: cover all parts; Piano roll: cover active part only
  const allPartsEndTick = Math.max(
    0,
    ...(project.voice_parts.length ? project.voice_parts.map(vp => vp.position + vp.duration) : [0]),
    ...(project.instrument_parts.length ? project.instrument_parts.map(ip => ip.position + ip.duration) : [0]),
    ...(project.wave_parts.length ? project.wave_parts.map(wp => wp.position + Math.round((wp.file_duration_ms / 1000) * tpsVal)) : [0]),
  );
  const arrBars = Math.ceil(allPartsEndTick / ticksPerBar) + 4;

  // Audio mode: calculate from all wave parts on the active track
  const activeWpIdx = useSynthStore(s => s.activeWavePartIndex);
  let wpEndTick = 0;
  if (activeWpIdx >= 0) {
    const wp0 = project.wave_parts[activeWpIdx];
    if (wp0) {
      for (const wp of project.wave_parts) {
        if (wp.track_no === wp0.track_no) {
          const endT = wp.position + (wp.file_duration_ms / 1000) * tpsVal;
          if (endT > wpEndTick) wpEndTick = endT;
        }
      }
    }
  }
  const wpBars = Math.ceil(wpEndTick / ticksPerBar) + 1;

  const isArr = viewMode === 'arrangement';
  const visibleBars = isArr
    ? Math.max(arrBars, extraBars)
    : activeWaveMode
      ? Math.max(MIN_BARS, wpBars)
      : (hasNextSegment ? partBars : Math.max(partBars, extraBars));
  const totalWidth = visibleBars * ticksPerBar * pixelsPerTick;
  const partStartPx = isArr || activeWaveMode ? 0 : partStartTick * pixelsPerTick;

  // Sync store scrollX → scrollbar position
  useEffect(() => {
    const el = scrollbarRef.current;
    if (!el || isUserScrolling.current) return;
    const noOffset = viewMode === 'arrangement' || activeWaveMode;
    const target = noOffset ? scrollX : scrollX - partStartPx;
    if (Math.abs(el.scrollLeft - target) > 1) {
      el.scrollLeft = target;
    }
  }, [scrollX, partStartPx, viewMode]);

  // Extend when canvas wheel-scroll reaches near the end
  useEffect(() => {
    if (extendingRef.current) return;
    const maxScrollX = totalWidth - (scrollbarRef.current?.clientWidth ?? 0);
    if (maxScrollX > 0 && scrollX >= maxScrollX - 20 && !hasNextSegment && !activeWaveMode && !isArr) {
      extendingRef.current = true;
      setExtraBars(prev => prev + EXTEND_BARS);
      setTimeout(() => { extendingRef.current = false; }, 350);
    }
  }, [scrollX, totalWidth]);

  // Scrollbar scroll → update store + extend when near the end
  const onScroll = useCallback(() => {
    const el = scrollbarRef.current;
    if (!el) return;
    if (isDragging()) return;
    isUserScrolling.current = true;
    const ss = useSynthStore.getState();
    const noOffset = ss.viewMode === 'arrangement' || ss.activeWavePartIndex >= 0;
    setViewState({ scrollX: noOffset ? el.scrollLeft : el.scrollLeft + partStartPx });
    // If scrolled within 20px of the right edge: break drag → wait for mouseup → extend
    const nearEnd = el.scrollWidth - el.scrollLeft - el.clientWidth < 20;
    if (nearEnd && !extendingRef.current && !hasNextSegment && useSynthStore.getState().activeWavePartIndex < 0 && useSynthStore.getState().viewMode !== 'arrangement') {
      extendingRef.current = true;
      const savedScroll = el.scrollLeft;
      // 1) Break drag immediately
      el.style.overflowX = 'hidden';
      requestAnimationFrame(() => {
        el.style.overflowX = 'scroll';
        el.scrollLeft = savedScroll;
      });
      // 2) Wait for mouseup then extend
      const onUp = () => {
        window.removeEventListener('mouseup', onUp);
        setExtraBars(prev => prev + EXTEND_BARS);
        requestAnimationFrame(() => {
          el.scrollLeft = savedScroll;
          extendingRef.current = false;
        });
      };
      window.addEventListener('mouseup', onUp, { once: true });
    }
    requestAnimationFrame(() => { isUserScrolling.current = false; });
  }, [setViewState, partStartPx, hasNextSegment]);

  const onKeyDown = useCallback((e: React.KeyboardEvent) => {
    if (e.key === '1') setTool('select');
    if (e.key === '2') setTool('draw');
    if (e.key === '3') setTool('erase');
    if (e.key === 'Home') {
      e.preventDefault();
      import('../../lib/synth/audioEngine').then(eng => { eng.stop(); });
      useSynthStore.getState().setPlaying(false);
      useSynthStore.getState().setPlaybackPosition(0);
    }
    if (e.ctrlKey && e.key.toLowerCase() === 'z' && !e.shiftKey) { e.preventDefault(); undo(); }
    if (e.ctrlKey && (e.key === 'y' || (e.key.toLowerCase() === 'z' && e.shiftKey))) { e.preventDefault(); redo(); }
    if ((e.key === 'Delete' || e.key === 'Backspace') && selectedNotes.size > 0) {
      e.preventDefault();
      deleteNotes([...selectedNotes]);
    }
    if (e.ctrlKey && e.key === 'a') { e.preventDefault(); selectAll(); }
  }, [selectedNotes, setTool, undo, redo, deleteNotes, selectAll]);

  return (
    <div className="utai-synth-piano" tabIndex={0} onKeyDown={onKeyDown}>
      <div className="utai-synth-piano__toolbar-row">
        <PianoRollToolbar />
        {/* Horizontal scrollbar — offset to align with piano roll grid area */}
        <div
          ref={scrollbarRef}
          className="utai-synth-hscroll"
          style={{ marginLeft: viewMode === 'pianoroll' ? KEYBOARD_W : 0 }}
          onScroll={onScroll}
        >
          <div style={{ width: totalWidth, height: 1 }} />
        </div>
      </div>
      {viewMode === 'arrangement' ? <ArrangementView />
        : activeWaveMode ? <AudioTrackView />
        : <PianoRollCanvas />}
      <TransportBar />
    </div>
  );
}
