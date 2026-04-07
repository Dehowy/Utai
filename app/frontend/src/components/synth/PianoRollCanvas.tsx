import { useRef, useEffect, useCallback, useState } from 'react';
import { useSynthStore } from '../../store/synthStore';
import { defaultPitch, defaultVibrato } from '../../types/ustx';
import { renderPianoRoll, screenToTick, screenToMidi, snapTick, snapTickFloor, tickToScreen, midiToScreen, HEADER_H, KEYBOARD_W } from '../../lib/synth/pianoRollEngine';
import { hitTest } from '../../lib/synth/noteInteraction';
import { playNote } from '../../lib/synth/pitchPreview';
import * as engine from '../../lib/synth/audioEngine';
import { SNAP_OPTIONS } from './PianoRollToolbar';
import LyricEditor from './LyricEditor';

interface DragState {
  mode: 'none' | 'move' | 'resize' | 'draw' | 'select-rect' | 'scroll' | 'playhead';
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
  const [selectRect, setSelectRect] = useState<{ x1: number; y1: number; x2: number; y2: number } | null>(null);
  const [hoverPlayhead, setHoverPlayhead] = useState(false);
  // Track whether lyric editor is open via ref (readable in stale closures)
  const lyricEditRef = useRef(false);
  // Auto-scroll state for playhead edge-dragging (continuous rAF loop)
  const autoScrollRef = useRef<{ speed: number; mouseX: number; rafId: number | null }>({ speed: 0, mouseX: 0, rafId: null });
  // Track right-click on empty area for double-right-click → select tool
  const lastRightClickRef = useRef<{ time: number; x: number; y: number }>({ time: 0, x: 0, y: 0 });

  /** Start playhead drag with window-level mouse capture (survives canvas exit). */
  const startPlayheadDrag = useCallback((initialX: number, initialY: number) => {
    const s = useSynthStore.getState();
    const tick = Math.max(0, screenToTick(initialX, s.viewState));
    s.setPlaybackPosition(tick);
    engine.seek(engine.ticksToSeconds(tick, s.project.tempos));
    dragRef.current = { mode: 'playhead', startX: initialX, startY: initialY, startTick: tick, startMidi: 0 };
    autoScrollRef.current.mouseX = initialX;

    const canvas = canvasRef.current;
    if (!canvas) return;
    const edgeMargin = 40;

    const onMove = (ev: MouseEvent) => {
      const rect = canvas.getBoundingClientRect();
      const mx = ev.clientX - rect.left;
      autoScrollRef.current.mouseX = mx;

      // Update playhead position
      const st = useSynthStore.getState();
      if (mx > KEYBOARD_W) {
        const t = Math.max(0, (mx - KEYBOARD_W + st.viewState.scrollX) / st.viewState.pixelsPerTick);
        st.setPlaybackPosition(t);
        engine.seek(engine.ticksToSeconds(t, st.project.tempos));
      }

      // Determine edge-scroll speed (works even when mouse is outside canvas)
      let speed = 0;
      const w = rect.width;
      if (mx > w - edgeMargin) {
        speed = Math.min(5, (mx - (w - edgeMargin)) / edgeMargin + 1);
      } else if (mx < KEYBOARD_W + edgeMargin && st.viewState.scrollX > 0) {
        speed = -Math.min(5, (KEYBOARD_W + edgeMargin - mx) / edgeMargin + 1);
      }
      autoScrollRef.current.speed = speed;

      // Start rAF loop if at edge
      if (speed !== 0 && autoScrollRef.current.rafId == null) {
        const frame = () => {
          const as = autoScrollRef.current;
          if (as.speed === 0 || dragRef.current.mode !== 'playhead') { as.rafId = null; return; }
          const cur = useSynthStore.getState();
          const newScrollX = Math.max(0, cur.viewState.scrollX + as.speed * 6);
          cur.setViewState({ scrollX: newScrollX });
          const newTick = Math.max(0, (as.mouseX - KEYBOARD_W + newScrollX) / cur.viewState.pixelsPerTick);
          cur.setPlaybackPosition(newTick);
          engine.seek(engine.ticksToSeconds(newTick, cur.project.tempos));
          as.rafId = requestAnimationFrame(frame);
        };
        autoScrollRef.current.rafId = requestAnimationFrame(frame);
      } else if (speed === 0 && autoScrollRef.current.rafId != null) {
        cancelAnimationFrame(autoScrollRef.current.rafId);
        autoScrollRef.current.rafId = null;
      }
    };

    const onUp = () => {
      window.removeEventListener('mousemove', onMove);
      window.removeEventListener('mouseup', onUp);
      if (autoScrollRef.current.rafId != null) {
        cancelAnimationFrame(autoScrollRef.current.rafId);
        autoScrollRef.current.rafId = null;
      }
      autoScrollRef.current.speed = 0;
      dragRef.current = { mode: 'none', startX: 0, startY: 0, startTick: 0, startMidi: 0 };
    };

    window.addEventListener('mousemove', onMove);
    window.addEventListener('mouseup', onUp);
  }, []);

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
    const instIdx = store.activeInstrumentPartIndex;
    const instPart = instIdx >= 0 ? store.project.instrument_parts[instIdx] : null;
    renderPianoRoll({
      ctx, width: size.width, height: size.height,
      project: store.project, viewState: store.viewState,
      selectedNotes: store.selectedNotes, activePartIndex: store.activePartIndex,
      hoverMidi, hoverPlayhead,
      playheadTicks: store.playbackPositionTicks,
      instrumentPart: instPart,
    });
    // Draw selection rectangle
    if (selectRect) {
      const rx = Math.min(selectRect.x1, selectRect.x2);
      const ry = Math.min(selectRect.y1, selectRect.y2);
      const rw = Math.abs(selectRect.x2 - selectRect.x1);
      const rh = Math.abs(selectRect.y2 - selectRect.y1);
      ctx.fillStyle = 'rgba(155, 109, 255, 0.12)';
      ctx.fillRect(rx, ry, rw, rh);
      ctx.strokeStyle = 'rgba(155, 109, 255, 0.5)';
      ctx.lineWidth = 1;
      ctx.setLineDash([4, 3]);
      ctx.strokeRect(rx, ry, rw, rh);
      ctx.setLineDash([]);
    }

    // Draw ghost note preview
    if (ghostNote && !lyricEdit) {
      const vs = store.viewState;
      const instPartActive = store.activeInstrumentPartIndex >= 0;
      const partPos = instPartActive
        ? (store.project.instrument_parts[store.activeInstrumentPartIndex]?.position ?? 0)
        : (store.project.voice_parts[store.activePartIndex]?.position ?? 0);
      const gx = tickToScreen(partPos + ghostNote.tick, vs);
      const gy = midiToScreen(ghostNote.midi, vs);
      const gw = vs.snapTicks * vs.pixelsPerTick;
      const gh = vs.pixelsPerSemitone;
      ctx.fillStyle = instPartActive ? 'rgba(232, 148, 75, 0.25)' : 'rgba(155, 109, 255, 0.25)';
      ctx.fillRect(gx, gy + 1, gw - 1, gh - 2);
      ctx.strokeStyle = instPartActive ? 'rgba(232, 148, 75, 0.5)' : 'rgba(155, 109, 255, 0.5)';
      ctx.lineWidth = 1;
      ctx.setLineDash([3, 3]);
      ctx.strokeRect(gx, gy + 1, gw - 1, gh - 2);
      ctx.setLineDash([]);
    }
  }, [size, store.project, store.viewState, store.selectedNotes, store.activePartIndex, store.activeInstrumentPartIndex, store.playbackPositionTicks, hoverMidi, hoverPlayhead, ghostNote, lyricEdit, selectRect]);

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

    // Click/drag on ruler header → set playhead position (window-captured drag)
    if (y < HEADER_H && x > KEYBOARD_W && e.button === 0) {
      startPlayheadDrag(x, y);
      return;
    }

    // Click near playhead line on the grid area → drag playhead (not create note)
    if (x > KEYBOARD_W && y >= HEADER_H && e.button === 0) {
      const playheadX = KEYBOARD_W + s.playbackPositionTicks * s.viewState.pixelsPerTick - s.viewState.scrollX;
      if (Math.abs(x - playheadX) < 8) {
        startPlayheadDrag(x, y);
        return;
      }
    }

    // ── Instrument mode: simplified note handling ──
    const instIdx = s.activeInstrumentPartIndex;
    if (instIdx >= 0) {
      const instPart = s.project.instrument_parts[instIdx];
      if (!instPart) return;

      if (x <= KEYBOARD_W) {
        const midi = Math.ceil(screenToMidi(y, s.viewState));
        if (midi >= 0 && midi <= 127) playNote(midi, 0.3);
        return;
      }

      // Right-click: delete instrument note
      if (e.button === 2) {
        // Check if clicking on an instrument note
        const partPos = instPart.position;
        const clickTick = screenToTick(x, s.viewState);
        const clickMidi = Math.ceil(screenToMidi(y, s.viewState));
        const hitNoteIdx = instPart.notes.findIndex(n =>
          clickTick >= partPos + n.position && clickTick <= partPos + n.position + n.duration &&
          clickMidi === n.tone
        );
        if (hitNoteIdx >= 0) {
          s.deleteInstrumentNotes([`i${instIdx}-${hitNoteIdx}`]);
        }
        return;
      }

      if (e.button === 1) {
        dragRef.current = { mode: 'scroll', startX: e.clientX, startY: e.clientY, startTick: 0, startMidi: 0, origScrollX: s.viewState.scrollX, origScrollY: s.viewState.scrollY };
        return;
      }

      // Left click on instrument area
      if (e.button === 0 && x > KEYBOARD_W && y > HEADER_H) {
        const partPos = instPart.position;
        const clickTick = screenToTick(x, s.viewState);
        const clickMidi = Math.ceil(screenToMidi(y, s.viewState));

        // Check if clicking on an existing instrument note → select it
        const hitIdx = instPart.notes.findIndex(n =>
          clickTick >= partPos + n.position &&
          clickTick <= partPos + n.position + n.duration &&
          clickMidi === n.tone
        );
        if (hitIdx >= 0) {
          const noteId = `i${instIdx}-${hitIdx}`;
          s.selectNotes([noteId]);
          playNote(clickMidi);
          dragRef.current = { mode: 'move', startX: x, startY: y, startTick: screenToTick(x, s.viewState), startMidi: clickMidi, noteId };
          return;
        }

        // Empty area → create new note
        const rawTick = clickTick - partPos;
        const snappedTick = snapTickFloor(rawTick, s.viewState.snapTicks);
        playNote(clickMidi);
        s.addInstrumentNote(instIdx, {
          position: Math.max(0, snappedTick),
          duration: s.viewState.snapTicks,
          tone: Math.max(0, Math.min(127, clickMidi)),
          velocity: 100,
        });
      }
      return;
    }

    // ── Vocal mode (existing behavior) ──
    const hit = hitTest(x, y, s.project, s.viewState, s.activePartIndex);

    if (hit.type === 'keyboard') {
      const midi = Math.ceil(screenToMidi(y, s.viewState));
      if (midi >= 0 && midi <= 127) playNote(midi, 0.3);
      return;
    }

    // Right-click: delete note under cursor, or double-right-click empty → select tool
    if (e.button === 2) {
      if (hit.type === 'note') {
        s.deleteNotes([hit.noteId]);
        lastRightClickRef.current = { time: 0, x: 0, y: 0 };
      } else if (hit.type === 'empty' && x > KEYBOARD_W && y > HEADER_H) {
        const now = Date.now();
        const prev = lastRightClickRef.current;
        if (now - prev.time < 400 && Math.abs(x - prev.x) < 20 && Math.abs(y - prev.y) < 20) {
          s.setTool('select');
          s.clearSelection();
          lastRightClickRef.current = { time: 0, x: 0, y: 0 };
        } else {
          lastRightClickRef.current = { time: now, x, y };
        }
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
    } else if (s.activeTool === 'select' && hit.type === 'empty') {
      // Selection rectangle drag (multi-select)
      if (!e.shiftKey) s.clearSelection();
      dragRef.current = { mode: 'select-rect', startX: x, startY: y, startTick: screenToTick(x, s.viewState), startMidi: Math.round(screenToMidi(y, s.viewState)) };
      setSelectRect({ x1: x, y1: y, x2: x, y2: y });
      return;
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
    const canvas = canvasRef.current;

    // Always update hover position for row highlight
    const midi = Math.ceil(screenToMidi(y, s.viewState));
    if (midi >= 0 && midi <= 127 && x > 0) {
      setHoverMidi(midi);
    } else {
      setHoverMidi(null);
    }

    // Playhead hover detection
    const drag = dragRef.current;
    const playheadX = KEYBOARD_W + s.playbackPositionTicks * s.viewState.pixelsPerTick - s.viewState.scrollX;
    const nearPlayhead = x > KEYBOARD_W && Math.abs(x - playheadX) < 8;
    setHoverPlayhead(nearPlayhead);

    // Instrument mode flag
    const instIdx = s.activeInstrumentPartIndex;
    const isInstMode = instIdx >= 0;

    // Cursor styling based on hover area
    if (canvas) {
      if (drag.mode === 'playhead' || drag.mode === 'scroll') {
        canvas.style.cursor = drag.mode === 'playhead' ? 'col-resize' : 'grabbing';
      } else if (drag.mode !== 'none') {
        canvas.style.cursor = drag.mode === 'resize' ? 'col-resize' : 'default';
      } else if (y < HEADER_H && x > KEYBOARD_W) {
        canvas.style.cursor = 'col-resize';
      } else if (nearPlayhead) {
        canvas.style.cursor = 'col-resize';
      } else if (x <= KEYBOARD_W) {
        canvas.style.cursor = 'pointer';
      } else if (isInstMode) {
        canvas.style.cursor = 'crosshair';
      } else {
        const hit = hitTest(x, y, s.project, s.viewState, s.activePartIndex);
        if (hit.type === 'resize') canvas.style.cursor = 'col-resize';
        else if (hit.type === 'note') canvas.style.cursor = 'move';
        else canvas.style.cursor = s.activeTool === 'draw' ? 'crosshair' : 'default';
      }
    }

    // Ghost note preview when hovering empty area (no drag, not over keyboard/header)
    if (drag.mode === 'none' && x > KEYBOARD_W && y > HEADER_H) {
      if (isInstMode) {
        // Instrument mode: simple snap, no voice-part hitTest
        const instPart = s.project.instrument_parts[instIdx];
        const partPos = instPart?.position ?? 0;
        const rawTick = screenToTick(x, s.viewState) - partPos;
        const snappedTick = snapTickFloor(rawTick, s.viewState.snapTicks);
        const gMidi = Math.max(0, Math.min(127, Math.ceil(screenToMidi(y, s.viewState))));
        setGhostNote({ tick: Math.max(0, snappedTick), midi: gMidi });
      } else {
        const hit = hitTest(x, y, s.project, s.viewState, s.activePartIndex);
        if (hit.type === 'empty') {
          const partPos = s.project.voice_parts[s.activePartIndex]?.position ?? 0;
          const rawTick = hit.tick - partPos;
          let snappedTick = snapTickFloor(rawTick, s.viewState.snapTicks);
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
      }
    } else {
      setGhostNote(null);
    }

    if (drag.mode === 'none') return;

    // Playhead drag is handled by window-level listeners (startPlayheadDrag),
    // so skip it here to avoid duplicate handling.
    if (drag.mode === 'playhead') return;

    if (drag.mode === 'select-rect') {
      setSelectRect(prev => prev ? { ...prev, x2: x, y2: y } : null);
      return;
    }

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
    // Stop auto-scroll
    if (autoScrollRef.current.rafId != null) {
      cancelAnimationFrame(autoScrollRef.current.rafId);
      autoScrollRef.current.rafId = null;
    }
    autoScrollRef.current.speed = 0;

    const drag = dragRef.current;
    if (drag.mode === 'select-rect' && selectRect) {
      // Find notes within rectangle
      const s = useSynthStore.getState();
      const minX = Math.min(selectRect.x1, selectRect.x2);
      const maxX = Math.max(selectRect.x1, selectRect.x2);
      const minY = Math.min(selectRect.y1, selectRect.y2);
      const maxY = Math.max(selectRect.y1, selectRect.y2);

      const minTick = screenToTick(minX, s.viewState);
      const maxTick = screenToTick(maxX, s.viewState);
      const topMidi = Math.ceil(screenToMidi(minY, s.viewState));
      const botMidi = Math.floor(screenToMidi(maxY, s.viewState));

      const ids: string[] = [];
      s.project.voice_parts.forEach((part, pi) => {
        part.notes.forEach((note, ni) => {
          const absPos = part.position + note.position;
          const absEnd = absPos + note.duration;
          if (absEnd >= minTick && absPos <= maxTick && note.tone >= botMidi && note.tone <= topMidi) {
            ids.push(`${pi}-${ni}`);
          }
        });
      });
      if (ids.length > 0) s.selectNotes(ids);
      setSelectRect(null);
    }
    dragRef.current = { mode: 'none', startX: 0, startY: 0, startTick: 0, startMidi: 0 };
  }, [selectRect]);

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
    } else if (hit.type === 'empty' && y > HEADER_H && x > KEYBOARD_W) {
      // Double-click empty area in select mode → switch to draw and create note
      s.setTool('draw');
      const partPos = s.project.voice_parts[s.activePartIndex]?.position ?? 0;
      const rawTick = hit.tick - partPos;
      const snappedTick = snapTickFloor(rawTick, s.viewState.snapTicks);
      const midi = Math.max(0, Math.min(127, hit.midi));
      playNote(midi);
      const notePos = Math.max(0, snappedTick);
      s.addNote(s.activePartIndex, {
        position: notePos, duration: s.viewState.snapTicks, tone: midi, lyric: 'a',
        pitch: defaultPitch(), vibrato: defaultVibrato(), tuning: 0, phoneme_expressions: [], phoneme_overrides: [],
      });
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

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const handler = (e: WheelEvent) => {
      e.preventDefault();
      const s = useSynthStore.getState();
      if (e.altKey) {
        // Alt+wheel → cycle note type (snap)
        const curIdx = SNAP_OPTIONS.findIndex(o => o.ticks === s.viewState.snapTicks);
        const dir = e.deltaY > 0 ? 1 : -1;
        const newIdx = Math.max(0, Math.min(SNAP_OPTIONS.length - 1, (curIdx === -1 ? 2 : curIdx) + dir));
        s.setViewState({ snapTicks: SNAP_OPTIONS[newIdx].ticks });
      } else if (e.ctrlKey) {
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
        onMouseLeave={() => {
          // Don't cancel playhead drag on leave — it's handled by window listeners
          if (dragRef.current.mode !== 'playhead') onMouseUp();
          setHoverMidi(null); setGhostNote(null); setHoverPlayhead(false); setSelectRect(null);
        }} onDoubleClick={onDoubleClick}
        onContextMenu={e => e.preventDefault()}
      />
      {lyricEdit && <LyricEditor x={lyricPos.x} y={lyricPos.y} value={lyricEdit.value} onConfirm={onLyricConfirm} onCancel={closeLyricEditor} />}
    </div>
  );
}
