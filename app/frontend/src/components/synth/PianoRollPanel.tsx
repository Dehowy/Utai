import { useCallback, useRef, useEffect, useState } from 'react';
import { useSynthStore } from '../../store/synthStore';
import PianoRollToolbar from './PianoRollToolbar';
import PianoRollCanvas from './PianoRollCanvas';
import ArrangementView from './ArrangementView';
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

  const scrollbarRef = useRef<HTMLDivElement>(null);
  const isUserScrolling = useRef(false);
  const extendingRef = useRef(false);

  // Dynamic bar count — starts at MIN_BARS, extends by EXTEND_BARS when scrolled to end
  const ts = project.time_signatures[0];
  const ticksPerBar = 480 * (ts?.beat_per_bar ?? 4);
  const contentMaxTick = project.voice_parts.reduce((max, p) => {
    const partEnd = p.position + p.duration;
    return Math.max(max, partEnd);
  }, 0);
  // At least cover content + a few bars of breathing room
  const contentBars = Math.ceil(contentMaxTick / ticksPerBar) + 2;
  const [extraBars, setExtraBars] = useState(MIN_BARS);
  const visibleBars = Math.max(contentBars, extraBars);
  const totalWidth = visibleBars * ticksPerBar * pixelsPerTick;

  // Sync store scrollX → scrollbar position
  useEffect(() => {
    const el = scrollbarRef.current;
    if (!el || isUserScrolling.current) return;
    el.scrollLeft = scrollX;
  }, [scrollX]);

  // Extend when canvas wheel-scroll reaches near the end
  useEffect(() => {
    if (extendingRef.current) return;
    const maxScrollX = totalWidth - (scrollbarRef.current?.clientWidth ?? 0);
    if (maxScrollX > 0 && scrollX >= maxScrollX - 20) {
      extendingRef.current = true;
      setExtraBars(prev => prev + EXTEND_BARS);
      setTimeout(() => { extendingRef.current = false; }, 350);
    }
  }, [scrollX, totalWidth]);

  // Scrollbar scroll → update store + extend when near the end
  const onScroll = useCallback(() => {
    const el = scrollbarRef.current;
    if (!el) return;
    isUserScrolling.current = true;
    setViewState({ scrollX: el.scrollLeft });
    // If scrolled within 20px of the right edge: break drag → wait for mouseup → extend
    const nearEnd = el.scrollWidth - el.scrollLeft - el.clientWidth < 20;
    if (nearEnd && !extendingRef.current) {
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
  }, [setViewState]);

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
    if (e.ctrlKey && e.key === 'z') { e.preventDefault(); undo(); }
    if (e.ctrlKey && e.key === 'y') { e.preventDefault(); redo(); }
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
          style={{ marginLeft: KEYBOARD_W }}
          onScroll={onScroll}
        >
          <div style={{ width: totalWidth, height: 1 }} />
        </div>
      </div>
      {viewMode === 'pianoroll' ? <PianoRollCanvas /> : <ArrangementView />}
      <TransportBar />
    </div>
  );
}
