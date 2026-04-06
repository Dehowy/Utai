import { useRef, useEffect, useCallback, useState } from 'react';
import { useSynthStore } from '../../store/synthStore';
import { defaultPitch, defaultVibrato } from '../../types/ustx';
import { renderPianoRoll, screenToTick, screenToMidi, snapTick, snapTickFloor, tickToScreen, midiToScreen, HEADER_H, KEYBOARD_W } from '../../lib/synth/pianoRollEngine';
import { hitTest } from '../../lib/synth/noteInteraction';
import { playNote } from '../../lib/synth/pitchPreview';
import LyricEditor from './LyricEditor';

interface DragState {
  mode: 'none' | 'move' | 'resize' | 'draw' | 'select-rect' | 'scroll';
  startX: number; startY: number; startTick: number; startMidi: number;
  noteId?: string; partIndex?: number; noteIndex?: number;
  origScrollX?: number; origScrollY?: number;
}

/** Ghost note preview position (snapped tick + midi) shown when hovering empty area. */
interface GhostNote { tick: number; midi: number; }

export default function PianoRollCanvas() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const store = useSynthStore();
  const [size, setSize] = useState({ width: 800, height: 600 });
  const dragRef = useRef<DragState>({ mode: 'none', startX: 0, startY: 0, startTick: 0, startMidi: 0 });
  const [lyricEdit, setLyricEdit] = useState<{ partIndex: number; noteIndex: number; value: string } | null>(null);
  const [hoverMidi, setHoverMidi] = useState<number | null>(null);
  const [ghostNote, setGhostNote] = useState<GhostNote | null>(null);
  // Track whether lyric editor is open via ref (readable in stale closures)
  const lyricEditRef = useRef(false);

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

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    canvas.width = size.width * devicePixelRatio;
    canvas.height = size.height * devicePixelRatio;
    ctx.scale(devicePixelRatio, devicePixelRatio);
    renderPianoRoll({
      ctx, width: size.width, height: size.height,
      project: store.project, viewState: store.viewState,
      selectedNotes: store.selectedNotes, activePartIndex: store.activePartIndex,
      hoverMidi,
      playheadTicks: store.isPlaying || store.playbackPositionTicks > 0 ? store.playbackPositionTicks : undefined,
    });
    // Draw ghost note preview
    if (ghostNote && !lyricEdit) {
      const vs = store.viewState;
      const partPos = store.project.voice_parts[store.activePartIndex]?.position ?? 0;
      const gx = tickToScreen(partPos + ghostNote.tick, vs);
      const gy = midiToScreen(ghostNote.midi, vs);
      const gw = vs.snapTicks * vs.pixelsPerTick;
      const gh = vs.pixelsPerSemitone;
      ctx.fillStyle = 'rgba(155, 109, 255, 0.25)';
      ctx.fillRect(gx, gy + 1, gw - 1, gh - 2);
      ctx.strokeStyle = 'rgba(155, 109, 255, 0.5)';
      ctx.lineWidth = 1;
      ctx.setLineDash([3, 3]);
      ctx.strokeRect(gx, gy + 1, gw - 1, gh - 2);
      ctx.setLineDash([]);
    }
  }, [size, store.project, store.viewState, store.selectedNotes, store.activePartIndex, hoverMidi, ghostNote, lyricEdit]);

  // Convert mouse event to canvas-local CSS-pixel coordinates.
  // Use nativeEvent.offsetX/Y which the browser computes directly,
  // avoiding getBoundingClientRect rounding issues on high-DPI / scaled displays.
  const canvasXY = useCallback((e: React.MouseEvent<HTMLCanvasElement>) => {
    const ne = e.nativeEvent;
    // offsetX/Y is relative to the target element — if somehow the target
    // isn't the canvas (shouldn't happen), fall back to rect calculation.
    if (ne.target === canvasRef.current) {
      return { x: ne.offsetX, y: ne.offsetY };
    }
    const rect = canvasRef.current!.getBoundingClientRect();
    return { x: e.clientX - rect.left, y: e.clientY - rect.top };
  }, []);

  const onMouseDown = useCallback((e: React.MouseEvent<HTMLCanvasElement>) => {
    // If lyric editor is open, this click should only close it (blur will handle confirm).
    // Swallow the event so no note is created/deleted.
    if (lyricEditRef.current) {
      return;
    }

    const { x, y } = canvasXY(e);
    const s = useSynthStore.getState();
    const hit = hitTest(x, y, s.project, s.viewState, s.activePartIndex);

    if (hit.type === 'keyboard') {
      const midi = Math.ceil(screenToMidi(y, s.viewState));
      if (midi >= 0 && midi <= 127) playNote(midi, 0.3);
      return;
    }

    // Right-click: delete note under cursor
    if (e.button === 2) {
      if (hit.type === 'note') {
        s.deleteNotes([hit.noteId]);
      }
      return;
    }

    // Middle-click: scroll
    if (e.button === 1) {
      dragRef.current = { mode: 'scroll', startX: e.clientX, startY: e.clientY, startTick: 0, startMidi: 0, origScrollX: s.viewState.scrollX, origScrollY: s.viewState.scrollY };
      return;
    }

    // --- Smart tool behavior ---
    if (s.activeTool === 'erase') {
      if (hit.type === 'note') {
        s.deleteNotes([hit.noteId]);
      } else if (hit.type === 'empty') {
        s.setTool('draw');
      }
    } else if (hit.type === 'note') {
      if (s.activeTool !== 'select') s.setTool('select');
      const clickedNote = s.project.voice_parts[hit.partIndex]?.notes[hit.noteIndex];
      if (clickedNote) playNote(clickedNote.tone);
      if (e.shiftKey) {
        s.selectNotes([hit.noteId], true);
      } else if (!s.selectedNotes.has(hit.noteId)) {
        s.selectNotes([hit.noteId]);
      }
      dragRef.current = { mode: 'move', startX: x, startY: y, startTick: screenToTick(x, s.viewState), startMidi: Math.round(screenToMidi(y, s.viewState)), noteId: hit.noteId };
    } else if (hit.type === 'resize') {
      if (s.activeTool !== 'select') s.setTool('select');
      s.selectNotes([hit.noteId]);
      dragRef.current = { mode: 'resize', startX: x, startY: y, startTick: screenToTick(x, s.viewState), startMidi: 0, partIndex: hit.partIndex, noteIndex: hit.noteIndex };
    } else if (hit.type === 'empty') {
      if (s.activeTool !== 'draw') s.setTool('draw');
      const partPos = s.project.voice_parts[s.activePartIndex]?.position ?? 0;
      const rawTick = hit.tick - partPos;
      const part = s.project.voice_parts[s.activePartIndex];
      let snappedTick = snapTickFloor(rawTick, s.viewState.snapTicks);
      if (part && part.notes.length > 0) {
        const tolerance = s.viewState.snapTicks * 0.6;
        for (const n of part.notes) {
          const noteEnd = n.position + n.duration;
          if (Math.abs(rawTick - noteEnd) < tolerance) {
            snappedTick = noteEnd;
            break;
          }
        }
      }
      const midi = Math.max(0, Math.min(127, hit.midi));
      playNote(midi);
      const notePos = Math.max(0, snappedTick);
      s.addNote(s.activePartIndex, {
        position: notePos, duration: s.viewState.snapTicks, tone: midi, lyric: 'a',
        pitch: defaultPitch(), vibrato: defaultVibrato(), tuning: 0, phoneme_expressions: [], phoneme_overrides: [],
      });
      // Auto-open lyric editor on the newly added note.
      // Delay so the mousedown event fully settles (browser focus handling)
      // before the input mounts and grabs focus.
      const partIdx = s.activePartIndex;
      requestAnimationFrame(() => {
        const updatedPart = useSynthStore.getState().project.voice_parts[partIdx];
        if (updatedPart) {
          const ni = updatedPart.notes.findIndex(n => n.position === notePos && n.tone === midi);
          if (ni >= 0) {
            lyricEditRef.current = true;
            setLyricEdit({ partIndex: partIdx, noteIndex: ni, value: 'a' });
          }
        }
      });
    }
  }, []);

  const onMouseMove = useCallback((e: React.MouseEvent<HTMLCanvasElement>) => {
    const { x, y } = canvasXY(e);
    const s = useSynthStore.getState();

    // Always update hover position for row highlight
    const midi = Math.ceil(screenToMidi(y, s.viewState));
    if (midi >= 0 && midi <= 127 && x > 0) {
      setHoverMidi(midi);
    } else {
      setHoverMidi(null);
    }

    // Ghost note preview when hovering empty area (no drag, not over keyboard/header)
    const drag = dragRef.current;
    if (drag.mode === 'none' && x > KEYBOARD_W && y > HEADER_H) {
      const hit = hitTest(x, y, s.project, s.viewState, s.activePartIndex);
      if (hit.type === 'empty') {
        const partPos = s.project.voice_parts[s.activePartIndex]?.position ?? 0;
        const rawTick = hit.tick - partPos;
        let snappedTick = snapTickFloor(rawTick, s.viewState.snapTicks);
        // Snap to note-end if close
        const part = s.project.voice_parts[s.activePartIndex];
        if (part && part.notes.length > 0) {
          const tolerance = s.viewState.snapTicks * 0.6;
          for (const n of part.notes) {
            const noteEnd = n.position + n.duration;
            if (Math.abs(rawTick - noteEnd) < tolerance) { snappedTick = noteEnd; break; }
          }
        }
        setGhostNote({ tick: Math.max(0, snappedTick), midi: Math.max(0, Math.min(127, hit.midi)) });
      } else {
        setGhostNote(null);
      }
    } else {
      setGhostNote(null);
    }

    if (drag.mode === 'none') return;

    if (drag.mode === 'scroll') {
      const dx = e.clientX - drag.startX;
      const dy = e.clientY - drag.startY;
      const maxScrollY = Math.max(0, 128 * s.viewState.pixelsPerSemitone - (size.height - HEADER_H));
      s.setViewState({
        scrollX: Math.max(0, (drag.origScrollX ?? 0) - dx),
        scrollY: Math.min(maxScrollY, Math.max(0, (drag.origScrollY ?? 0) - dy)),
      });
    } else if (drag.mode === 'move' && s.selectedNotes.size > 0) {
      const currentTick = screenToTick(x, s.viewState);
      const currentMidi = Math.round(screenToMidi(y, s.viewState));
      const deltaTick = snapTick(currentTick - drag.startTick, s.viewState.snapTicks);
      const deltaTone = currentMidi - drag.startMidi;
      if (deltaTick !== 0 || deltaTone !== 0) {
        s.moveNotes([...s.selectedNotes], deltaTick, deltaTone);
        drag.startTick += deltaTick;
        drag.startMidi += deltaTone;
        // Play pitch preview when dragging to a new row
        if (deltaTone !== 0) {
          const newMidi = Math.max(0, Math.min(127, currentMidi));
          playNote(newMidi);
        }
      }
    } else if (drag.mode === 'resize' && drag.partIndex !== undefined && drag.noteIndex !== undefined) {
      const currentTick = screenToTick(x, s.viewState);
      const part = s.project.voice_parts[drag.partIndex];
      const note = part?.notes[drag.noteIndex];
      if (note && part) {
        const newDuration = snapTick(currentTick - (part.position + note.position), s.viewState.snapTicks);
        if (newDuration > 0) s.resizeNote(drag.partIndex, drag.noteIndex, newDuration);
      }
    }
  }, []);

  const onMouseUp = useCallback(() => {
    dragRef.current = { mode: 'none', startX: 0, startY: 0, startTick: 0, startMidi: 0 };
  }, []);

  const onDoubleClick = useCallback((e: React.MouseEvent<HTMLCanvasElement>) => {
    const { x, y } = canvasXY(e);
    const s = useSynthStore.getState();
    const hit = hitTest(x, y, s.project, s.viewState, s.activePartIndex);
    if (hit.type === 'note') {
      const note = s.project.voice_parts[hit.partIndex]?.notes[hit.noteIndex];
      if (note) {
        lyricEditRef.current = true;
        setLyricEdit({ partIndex: hit.partIndex, noteIndex: hit.noteIndex, value: note.lyric });
      }
    }
  }, []);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const handler = (e: WheelEvent) => {
      e.preventDefault();
      const s = useSynthStore.getState();
      if (e.ctrlKey) {
        const factor = e.deltaY < 0 ? 1.15 : 1 / 1.15;
        s.setViewState({ pixelsPerTick: Math.max(0.05, Math.min(5, s.viewState.pixelsPerTick * factor)) });
      } else {
        const maxScrollY = Math.max(0, 128 * s.viewState.pixelsPerSemitone - (size.height - HEADER_H));
        s.setViewState({
          scrollX: Math.max(0, s.viewState.scrollX + e.deltaX + (e.shiftKey ? e.deltaY : 0)),
          scrollY: Math.min(maxScrollY, Math.max(0, s.viewState.scrollY + (e.shiftKey ? 0 : e.deltaY))),
        });
      }
    };
    canvas.addEventListener('wheel', handler, { passive: false });
    return () => canvas.removeEventListener('wheel', handler);
  }, [size.height]);

  const closeLyricEditor = useCallback(() => {
    lyricEditRef.current = false;
    setLyricEdit(null);
  }, []);

  const onLyricConfirm = useCallback((value: string) => {
    if (lyricEdit) useSynthStore.getState().setLyric(lyricEdit.partIndex, lyricEdit.noteIndex, value);
    closeLyricEditor();
  }, [lyricEdit, closeLyricEditor]);

  // Calculate lyric editor position from the note's coordinates
  const lyricPos = (() => {
    if (!lyricEdit) return { x: 0, y: 0 };
    const s = useSynthStore.getState();
    const part = s.project.voice_parts[lyricEdit.partIndex];
    const note = part?.notes[lyricEdit.noteIndex];
    if (!note || !part) return { x: 0, y: 0 };
    const absTick = part.position + note.position;
    return {
      x: tickToScreen(absTick, s.viewState),
      y: midiToScreen(note.tone, s.viewState),
    };
  })();

  return (
    <div ref={containerRef} className="utai-synth-piano__canvas-wrap">
      <canvas
        ref={canvasRef}
        style={{ width: size.width, height: size.height, display: 'block' }}
        onMouseDown={onMouseDown} onMouseMove={onMouseMove} onMouseUp={onMouseUp}
        onMouseLeave={() => { onMouseUp(); setHoverMidi(null); setGhostNote(null); }} onDoubleClick={onDoubleClick}
        onContextMenu={e => e.preventDefault()}
      />
      {lyricEdit && <LyricEditor x={lyricPos.x} y={lyricPos.y} value={lyricEdit.value} onConfirm={onLyricConfirm} onCancel={closeLyricEditor} />}
    </div>
  );
}
