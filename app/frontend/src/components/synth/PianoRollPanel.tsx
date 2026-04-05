import { useCallback, useRef, useEffect } from 'react';
import { useSynthStore } from '../../store/synthStore';
import PianoRollToolbar from './PianoRollToolbar';
import PianoRollCanvas from './PianoRollCanvas';

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

  const scrollbarRef = useRef<HTMLDivElement>(null);
  const isUserScrolling = useRef(false);

  // Calculate total content width based on project duration
  const maxTick = project.voice_parts.reduce((max, p) => {
    const partEnd = p.position + p.duration;
    return Math.max(max, partEnd);
  }, 480 * 32); // minimum 8 bars of 4/4
  const totalWidth = maxTick * pixelsPerTick;

  // Sync store scrollX → scrollbar position
  useEffect(() => {
    const el = scrollbarRef.current;
    if (!el || isUserScrolling.current) return;
    el.scrollLeft = scrollX;
  }, [scrollX]);

  // Scrollbar scroll → update store
  const onScroll = useCallback(() => {
    const el = scrollbarRef.current;
    if (!el) return;
    isUserScrolling.current = true;
    setViewState({ scrollX: el.scrollLeft });
    // Reset flag after a tick
    requestAnimationFrame(() => { isUserScrolling.current = false; });
  }, [setViewState]);

  const onKeyDown = useCallback((e: React.KeyboardEvent) => {
    if (e.key === '1') setTool('select');
    if (e.key === '2') setTool('draw');
    if (e.key === '3') setTool('erase');
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
      <PianoRollToolbar />
      <div style={{ flex: 1, minHeight: 0, display: 'flex', flexDirection: 'column' }}>
        <div style={{ flex: 1, minHeight: 0 }}>
          <PianoRollCanvas />
        </div>
        {/* Horizontal scrollbar */}
        <div
          ref={scrollbarRef}
          className="utai-synth-hscroll"
          onScroll={onScroll}
        >
          <div style={{ width: totalWidth, height: 1 }} />
        </div>
      </div>
    </div>
  );
}
