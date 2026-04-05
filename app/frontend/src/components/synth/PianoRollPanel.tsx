import { useCallback } from 'react';
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
      <PianoRollCanvas />
    </div>
  );
}
