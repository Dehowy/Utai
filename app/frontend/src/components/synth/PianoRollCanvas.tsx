import { useRef, useEffect, useCallback, useState } from 'react';
import { useSynthStore } from '../../store/synthStore';
import { defaultPitch, defaultVibrato, RESOLUTION } from '../../types/ustx';
import { renderPianoRoll, screenToTick, screenToMidi, snapTick, snapTickFloor, tickToScreen, midiToScreen, ticksToDurationLabel, hitTestPitchPoint, HEADER_H, KEYBOARD_W } from '../../lib/synth/pianoRollEngine';
import { hitTest } from '../../lib/synth/noteInteraction';
import { playNote } from '../../lib/synth/pitchPreview';
import * as engine from '../../lib/synth/audioEngine';
import { SNAP_OPTIONS, SNAP_ATOMS_ASC } from './PianoRollToolbar';
import { renderVST } from '../../api/client';
import LyricEditor from './LyricEditor';

interface DragState {
  mode: 'none' | 'move' | 'resize' | 'draw' | 'select-rect' | 'erase-rect' | 'scroll' | 'playhead' | 'pitch-point';
  pitchHit?: { partIndex: number; noteIndex: number; pointIndex: number; noteX0: number; noteXEnd: number; noteDurMs: number; baseMidi: number };
  startX: number; startY: number; startTick: number; startMidi: number;
  noteId?: string; partIndex?: number; noteIndex?: number;
  origScrollX?: number; origScrollY?: number;
}

/** Ghost note preview position (snapped tick + midi) shown when hovering empty area. */
interface GhostNote { tick: number; midi: number; }

export default function PianoRollCanvas() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const project = useSynthStore(s => s.project);
  const viewState = useSynthStore(s => s.viewState);
  const selectedNotes = useSynthStore(s => s.selectedNotes);
  const activePartIndex = useSynthStore(s => s.activePartIndex);
  const activeInstrumentPartIndex = useSynthStore(s => s.activeInstrumentPartIndex);
  const playbackPositionTicks = useSynthStore(s => s.playbackPositionTicks);
  const activeTool = useSynthStore(s => s.activeTool);
  const [size, setSize] = useState({ width: 800, height: 600 });
  const dragRef = useRef<DragState>({ mode: 'none', startX: 0, startY: 0, startTick: 0, startMidi: 0 });
  const [lyricEdit, setLyricEdit] = useState<{ partIndex: number; noteIndex: number; value: string } | null>(null);
  const [hoverMidi, setHoverMidi] = useState<number | null>(null);
  const [ghostNote, setGhostNote] = useState<GhostNote | null>(null);
  const [hoverPitchPt, setHoverPitchPt] = useState<{ noteIndex: number; pointIndex: number } | null>(null);
  const [selectRect, setSelectRect] = useState<{ x1: number; y1: number; x2: number; y2: number } | null>(null);
  const [hoverPlayhead, setHoverPlayhead] = useState(false);
  // Track whether lyric editor is open via ref (readable in stale closures)
  const lyricEditRef = useRef(false);
  // Track Tab key held state for Tab+wheel vertical zoom
  const tabHeld = useRef(false);
  // Auto-scroll state for playhead edge-dragging (continuous rAF loop)
  const autoScrollRef = useRef<{ speed: number; mouseX: number; rafId: number | null }>({ speed: 0, mouseX: 0, rafId: null });
  // Track right-click on empty area for double-right-click → select tool
  const lastRightClickRef = useRef<{ time: number; x: number; y: number }>({ time: 0, x: 0, y: 0 });

  /** Preview a note via VST backend render (instrument tracks).
   *  Empty deps OK — reads state via getState() at call time. */
  const playVSTPreview = useCallback((midi: number, durationSec?: number) => {
    const s = useSynthStore.getState();
    const instIdx = s.activeInstrumentPartIndex;
    if (instIdx < 0) return;
    const instPart = s.project.instrument_parts[instIdx];
    const track = instPart ? s.project.tracks[instPart.track_no] : null;
    if (!track?.plugin_name) { playNote(midi, durationSec); return; }
    const bpm = s.project.tempos[0]?.bpm ?? 120;
    const dur = durationSec ?? 0.3;
    renderVST({ plugin_name: track.plugin_name, notes: [{ tone: midi, velocity: 100, start: 0, duration: dur }], bpm })
      .then(blob => { const url = URL.createObjectURL(blob); const a = new Audio(url); a.play(); a.onended = () => URL.revokeObjectURL(url); })
      .catch(() => playNote(midi, durationSec));
  }, []);

  /** Get max relative tick for active part (Infinity if last segment on track). */
  const getPartMaxRelTick = useCallback(() => {
    const s = useSynthStore.getState();
    if (s.activePartIndex < 0) return Infinity;
    const part = s.project.voice_parts[s.activePartIndex];
    if (!part) return Infinity;
    const next = s.project.voice_parts.find(
      vp => vp.track_no === part.track_no && vp.position > part.position
    );
    return next ? next.position - part.position : Infinity;
  }, []);

  /** Start playhead drag with window-level mouse capture (survives canvas exit). */
  const startPlayheadDrag = useCallback((initialX: number, initialY: number) => {
    const s = useSynthStore.getState();
    const partMin = s.project.voice_parts[s.activePartIndex]?.position ?? 0;
    const tick = Math.max(partMin, screenToTick(initialX, s.viewState));
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

      // Update playhead position (clamped to active part, no dead zone)
      const st = useSynthStore.getState();
      const pMin = st.project.voice_parts[st.activePartIndex]?.position ?? 0;
      const effectiveX = Math.max(KEYBOARD_W, mx);
      const t = Math.max(pMin, (effectiveX - KEYBOARD_W + st.viewState.scrollX) / st.viewState.pixelsPerTick);
      st.setPlaybackPosition(t);
      engine.seek(engine.ticksToSeconds(t, st.project.tempos));

      // Determine edge-scroll speed (works even when mouse is outside canvas)
      let speed = 0;
      const w = rect.width;
      if (mx > w - edgeMargin) {
        speed = Math.min(5, (mx - (w - edgeMargin)) / edgeMargin + 1);
      } else if (mx < KEYBOARD_W + edgeMargin) {
        const pMinPx = (st.project.voice_parts[st.activePartIndex]?.position ?? 0) * st.viewState.pixelsPerTick;
        if (st.viewState.scrollX > pMinPx) {
          speed = -Math.min(5, (KEYBOARD_W + edgeMargin - mx) / edgeMargin + 1);
        }
      }
      autoScrollRef.current.speed = speed;

      // Start rAF loop if at edge
      if (speed !== 0 && autoScrollRef.current.rafId == null) {
        const frame = () => {
          const as = autoScrollRef.current;
          if (as.speed === 0 || dragRef.current.mode !== 'playhead') { as.rafId = null; return; }
          const cur = useSynthStore.getState();
          const pMinPx2 = (cur.project.voice_parts[cur.activePartIndex]?.position ?? 0) * cur.viewState.pixelsPerTick;
          const pMinTick = cur.project.voice_parts[cur.activePartIndex]?.position ?? 0;
          const newScrollX = Math.max(pMinPx2, cur.viewState.scrollX + as.speed * 6);
          cur.setViewState({ scrollX: newScrollX });
          const newTick = Math.max(pMinTick, (as.mouseX - KEYBOARD_W + newScrollX) / cur.viewState.pixelsPerTick);
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
    const instIdx = activeInstrumentPartIndex;
    const instPart = instIdx >= 0 ? project.instrument_parts[instIdx] : null;
    // Compute part boundary if next segment exists on same track
    const activePart = project.voice_parts[activePartIndex];
    let partEndTick: number | undefined;
    if (activePart) {
      const nextPart = project.voice_parts.find(
        vp => vp.track_no === activePart.track_no && vp.position > activePart.position
      );
      if (nextPart) partEndTick = nextPart.position;
    }

    renderPianoRoll({
      ctx, width: size.width, height: size.height,
      project: project, viewState: viewState,
      selectedNotes: selectedNotes, activePartIndex: activePartIndex,
      hoverMidi, hoverPlayhead,
      playheadTicks: playbackPositionTicks,
      instrumentPart: instPart,
      partEndTick,
      pitchToolActive: activeTool === 'pitch',
      hoverPitchPoint: hoverPitchPt,
    });
    // Draw selection rectangle
    if (selectRect) {
      const rx = Math.min(selectRect.x1, selectRect.x2);
      const ry = Math.min(selectRect.y1, selectRect.y2);
      const rw = Math.abs(selectRect.x2 - selectRect.x1);
      const rh = Math.abs(selectRect.y2 - selectRect.y1);
      const isEraseRect = activeTool === 'erase';
      ctx.fillStyle = isEraseRect ? 'rgba(255, 107, 157, 0.15)' : 'rgba(155, 109, 255, 0.12)';
      ctx.fillRect(rx, ry, rw, rh);
      ctx.strokeStyle = isEraseRect ? 'rgba(255, 107, 157, 0.5)' : 'rgba(155, 109, 255, 0.5)';
      ctx.lineWidth = 1;
      ctx.setLineDash([4, 3]);
      ctx.strokeRect(rx, ry, rw, rh);
      ctx.setLineDash([]);
    }

    // Draw ghost note preview
    if (ghostNote && !lyricEdit) {
      const vs = viewState;
      const instPartActive = activeInstrumentPartIndex >= 0;
      const partPos = instPartActive
        ? (project.instrument_parts[activeInstrumentPartIndex]?.position ?? 0)
        : (project.voice_parts[activePartIndex]?.position ?? 0);
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
      // Duration label on ghost note
      if (gw > 20) {
        const label = ticksToDurationLabel(vs.snapTicks);
        ctx.fillStyle = instPartActive ? 'rgba(232, 148, 75, 0.7)' : 'rgba(155, 109, 255, 0.7)';
        ctx.font = `${Math.min(10, gh - 4)}px 'DotGothic16', sans-serif`;
        ctx.textBaseline = 'middle';
        ctx.textAlign = 'center';
        ctx.fillText(label, gx + gw / 2, gy + gh / 2, gw - 4);
        ctx.textAlign = 'left';
      }
    }
  }, [size, project, viewState, selectedNotes, activePartIndex, activeInstrumentPartIndex, playbackPositionTicks, activeTool, hoverMidi, hoverPlayhead, ghostNote, lyricEdit, selectRect, hoverPitchPt]);

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
        if (midi >= 0 && midi <= 127) playVSTPreview(midi, 0.3);
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
          dragRef.current = { mode: 'move', startX: x, startY: y, startTick: screenToTick(x, s.viewState), startMidi: clickMidi, noteId };
          // VST preview
          const bpm = s.project.tempos[0]?.bpm ?? 120;
          const durSec = instPart.notes[hitIdx].duration / (480 * bpm / 60);
          playVSTPreview(clickMidi, durSec);
          return;
        }

        // Empty area → create new note
        const rawTick = clickTick - partPos;
        const snappedTick = snapTickFloor(rawTick, Math.min(s.viewState.snapTicks, RESOLUTION));
        playVSTPreview(clickMidi);
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

    // Right-click in pitch mode → delete pitch point only
    if (e.button === 2 && s.activeTool === 'pitch') {
      const pitchHit2 = hitTestPitchPoint(x, y, s.project, s.viewState, s.activePartIndex);
      if (pitchHit2 && pitchHit2.pointIndex >= 0) {
        s.deletePitchPoint(pitchHit2.partIndex, pitchHit2.noteIndex, pitchHit2.pointIndex);
      }
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

    // --- Alt+left-click → toggle pitch mode ---
    if (e.altKey && e.button === 0 && !e.shiftKey && !e.ctrlKey) {
      s.setTool(s.activeTool === 'pitch' ? 'select' : 'pitch');
      return;
    }

    // --- Ctrl+click in pitch mode → create/delete notes without leaving pitch mode ---
    if (e.ctrlKey && s.activeTool === 'pitch') {
      if (e.button === 0 && x > KEYBOARD_W && y > HEADER_H) {
        // Ctrl+left-click → create note at position
        const partPos = s.project.voice_parts[s.activePartIndex]?.position ?? 0;
        const rawTick = screenToTick(x, s.viewState) - partPos;
        const maxRelTick = getPartMaxRelTick();
        if (rawTick < maxRelTick) {
          const snappedTick = snapTickFloor(rawTick, Math.min(s.viewState.snapTicks, RESOLUTION));
          const midi = Math.max(0, Math.min(127, Math.ceil(screenToMidi(y, s.viewState))));
          const notePos = Math.max(0, snappedTick);
          s.addNote(s.activePartIndex, {
            position: notePos, duration: s.viewState.snapTicks, tone: midi,
            lyric: useSynthStore.getState().language === 'ja' ? 'あ' : '啊',
            pitch: defaultPitch(s.viewState.snapTicks / ((s.project.tempos[0]?.bpm ?? 120) * 480 / 60) * 1000), vibrato: defaultVibrato(), tuning: 0,
            phoneme_expressions: [], phoneme_overrides: [],
          });
        }
      } else if (e.button === 2) {
        // Ctrl+right-click → delete note under cursor
        if (hit.type === 'note') s.deleteNotes([hit.noteId]);
      }
      return;
    }

    // --- Pitch tool mode: ALL left-click interactions are pitch-only ---
    if (s.activeTool === 'pitch' && e.button === 0) {
      const pitchHit = hitTestPitchPoint(x, y, s.project, s.viewState, s.activePartIndex);
      if (pitchHit && pitchHit.pointIndex >= 0) {
        // Left-click on control point → drag
        dragRef.current = { mode: 'pitch-point', startX: x, startY: y, startTick: 0, startMidi: 0, pitchHit };
      } else if (pitchHit && pitchHit.pointIndex === -1) {
        // Left-click on curve line → add point here
        const frac = Math.max(0, Math.min(1, (x - pitchHit.noteX0) / (pitchHit.noteXEnd - pitchHit.noteX0)));
        const ptX = Math.round(frac * pitchHit.noteDurMs);
        const noteBaseY = HEADER_H + (127 - pitchHit.baseMidi) * s.viewState.pixelsPerSemitone - s.viewState.scrollY;
        const ptY = Math.round(-((y - noteBaseY - s.viewState.pixelsPerSemitone / 2) / s.viewState.pixelsPerSemitone) * 100) / 10;
        s.addPitchPoint(pitchHit.partIndex, pitchHit.noteIndex, ptX, ptY);
      } else {
        // Left-click on note without curve → add first point
        const noteHit = hitTest(x, y, s.project, s.viewState, s.activePartIndex);
        if (noteHit.type === 'note' && noteHit.partIndex !== undefined && noteHit.noteIndex !== undefined) {
          const part = s.project.voice_parts[noteHit.partIndex];
          const note = part?.notes[noteHit.noteIndex];
          if (note && part) {
            const bpm = s.project.tempos[0]?.bpm ?? 120;
            const ticksPerSec = (bpm * 480) / 60;
            const absTick = part.position + note.position;
            const noteX0 = KEYBOARD_W + absTick * s.viewState.pixelsPerTick - s.viewState.scrollX;
            const noteXEnd = noteX0 + note.duration * s.viewState.pixelsPerTick;
            const noteDurMs = (note.duration / ticksPerSec) * 1000;
            const frac = Math.max(0, Math.min(1, (x - noteX0) / (noteXEnd - noteX0)));
            const ptX = Math.round(frac * noteDurMs);
            const noteBaseY = HEADER_H + (127 - note.tone) * s.viewState.pixelsPerSemitone - s.viewState.scrollY;
            const ptY = Math.round(-((y - noteBaseY - s.viewState.pixelsPerSemitone / 2) / s.viewState.pixelsPerSemitone) * 100) / 10;
            s.addPitchPoint(noteHit.partIndex, noteHit.noteIndex, ptX, ptY);
          }
        }
      }
      return; // block everything else in pitch mode
    }
    if (s.activeTool === 'pitch') return; // block right-click note operations too

    // --- Smart tool behavior ---
    if (s.activeTool === 'erase') {
      if (x > KEYBOARD_W && y > HEADER_H) {
        dragRef.current = { mode: 'erase-rect', startX: x, startY: y, startTick: screenToTick(x, s.viewState), startMidi: Math.round(screenToMidi(y, s.viewState)) };
        setSelectRect({ x1: x, y1: y, x2: x, y2: y });
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
      const maxRelTick = getPartMaxRelTick();
      if (rawTick >= maxRelTick) return; // beyond segment boundary
      const part = s.project.voice_parts[s.activePartIndex];
      let snappedTick = snapTickFloor(rawTick, Math.min(s.viewState.snapTicks, RESOLUTION));
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
        position: notePos, duration: s.viewState.snapTicks, tone: midi, lyric: useSynthStore.getState().language === 'ja' ? 'あ' : '啊',
        pitch: defaultPitch(s.viewState.snapTicks / ((s.project.tempos[0]?.bpm ?? 120) * 480 / 60) * 1000), vibrato: defaultVibrato(), tuning: 0, phoneme_expressions: [], phoneme_overrides: [],
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
            setLyricEdit({ partIndex: partIdx, noteIndex: ni, value: useSynthStore.getState().language === 'ja' ? 'あ' : '啊' });
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
        // Check if hovering an instrument note
        const instPart = s.project.instrument_parts[instIdx];
        if (instPart && x > KEYBOARD_W && y > HEADER_H) {
          const clickTick = screenToTick(x, s.viewState);
          const clickMidi = Math.ceil(screenToMidi(y, s.viewState));
          const onNote = instPart.notes.some(n =>
            clickTick >= instPart.position + n.position &&
            clickTick <= instPart.position + n.position + n.duration &&
            clickMidi === n.tone
          );
          canvas.style.cursor = onNote ? 'move' : 'crosshair';
        } else {
          canvas.style.cursor = 'crosshair';
        }
      } else {
        const pitchPt = hitTestPitchPoint(x, y, s.project, s.viewState, s.activePartIndex);
        setHoverPitchPt(pitchPt ? { noteIndex: pitchPt.noteIndex, pointIndex: pitchPt.pointIndex } : null);
        if (s.activeTool === 'pitch') {
          canvas.style.cursor = pitchPt ? (pitchPt.pointIndex >= 0 ? 'grab' : 'copy') : 'default';
        } else {
          const hit = hitTest(x, y, s.project, s.viewState, s.activePartIndex);
          if (hit.type === 'resize') canvas.style.cursor = 'col-resize';
          else if (hit.type === 'note') canvas.style.cursor = 'move';
          else canvas.style.cursor = (s.activeTool === 'draw' || s.activeTool === 'erase') ? 'crosshair' : 'default';
        }
      }
    }

    // Ghost note preview when hovering empty area (no drag, not over keyboard/header)
    if (drag.mode === 'none' && x > KEYBOARD_W && y > HEADER_H) {
      if (s.activeTool === 'erase') {
        setGhostNote(null);
      } else if (isInstMode) {
        // Instrument mode: simple snap, no voice-part hitTest
        const instPart = s.project.instrument_parts[instIdx];
        const partPos = instPart?.position ?? 0;
        const rawTick = screenToTick(x, s.viewState) - partPos;
        const snappedTick = snapTickFloor(rawTick, Math.min(s.viewState.snapTicks, RESOLUTION));
        const gMidi = Math.max(0, Math.min(127, Math.ceil(screenToMidi(y, s.viewState))));
        setGhostNote({ tick: Math.max(0, snappedTick), midi: gMidi });
      } else {
        const hit = hitTest(x, y, s.project, s.viewState, s.activePartIndex);
        if (hit.type === 'empty') {
          const partPos = s.project.voice_parts[s.activePartIndex]?.position ?? 0;
          const rawTick = hit.tick - partPos;
          let snappedTick = snapTickFloor(rawTick, Math.min(s.viewState.snapTicks, RESOLUTION));
          const part = s.project.voice_parts[s.activePartIndex];
          if (part && part.notes.length > 0) {
            const tolerance = s.viewState.snapTicks * 0.6;
            for (const n of part.notes) {
              const noteEnd = n.position + n.duration;
              if (Math.abs(rawTick - noteEnd) < tolerance) { snappedTick = noteEnd; break; }
            }
          }
          const maxRelTick = getPartMaxRelTick();
          if (snappedTick >= maxRelTick) {
            setGhostNote(null);
          } else {
            setGhostNote({ tick: Math.max(0, snappedTick), midi: Math.max(0, Math.min(127, hit.midi)) });
          }
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

    if (drag.mode === 'pitch-point' && drag.pitchHit) {
      const ph = drag.pitchHit;
      const frac = Math.max(0, Math.min(1, (x - ph.noteX0) / (ph.noteXEnd - ph.noteX0)));
      const newX = frac * ph.noteDurMs;
      const noteBaseY = HEADER_H + (127 - ph.baseMidi) * s.viewState.pixelsPerSemitone - s.viewState.scrollY;
      let newY = -((y - noteBaseY - s.viewState.pixelsPerSemitone / 2) / s.viewState.pixelsPerSemitone) * 10;

      if (useSynthStore.getState().pitchSnap) {
        // Snap to base pitch (y=0) if within 1/4 semitone
        if (Math.abs(newY) < 2.5) newY = 0;

        // Snap to adjacent point's y within same note (horizontal line snap)
        const part0 = s.project.voice_parts[ph.partIndex];
        const curData = part0?.notes[ph.noteIndex]?.pitch?.data;
        if (curData) {
          const pi = ph.pointIndex;
          if (pi > 0 && Math.abs(newY - curData[pi - 1].y) < 2.5) newY = curData[pi - 1].y;
          else if (pi < curData.length - 1 && Math.abs(newY - curData[pi + 1].y) < 2.5) newY = curData[pi + 1].y;
        }

        // Snap to adjacent note's curve endpoint
        const part = s.project.voice_parts[ph.partIndex];
        const note = part?.notes[ph.noteIndex];
        if (note && part) {
          const noteIdx = ph.noteIndex;
          // Check previous note's last pitch point
          if (noteIdx > 0) {
            const prevNote = part.notes[noteIdx - 1];
            const prevData = prevNote?.pitch?.data;
            if (prevData && prevData.length > 0) {
              const lastPt = prevData[prevData.length - 1];
              // Convert prev note's y to this note's coordinate: account for MIDI difference
              const midiDiff = (prevNote.tone - note.tone) * 10; // in 0.1 semitones
              const adjY = lastPt.y + midiDiff;
              if (Math.abs(newY - adjY) < 2.5) newY = adjY;
            }
          }
          // Check next note's first pitch point
          if (noteIdx < part.notes.length - 1) {
            const nextNote = part.notes[noteIdx + 1];
            const nextData = nextNote?.pitch?.data;
            if (nextData && nextData.length > 0) {
              const firstPt = nextData[0];
              const midiDiff = (nextNote.tone - note.tone) * 10;
              const adjY = firstPt.y + midiDiff;
              if (Math.abs(newY - adjY) < 2.5) newY = adjY;
            }
          }
        }
      }

      s.setPitchPoint(ph.partIndex, ph.noteIndex, ph.pointIndex, Math.round(newX), Math.round(newY * 10) / 10);
      return;
    }

    if (drag.mode === 'select-rect' || drag.mode === 'erase-rect') {
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
        const noteIds = [...s.selectedNotes];
        const isInst = noteIds.some(id => id.startsWith('i'));
        if (isInst) {
          s.moveInstrumentNotes(noteIds, deltaTick, deltaTone);
        } else {
          s.moveNotes(noteIds, deltaTick, deltaTone);
        }
        drag.startTick += deltaTick;
        drag.startMidi += deltaTone;
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
        // Snap resize to smallest atom (1/32T) for fine control
        const fineSnap = SNAP_ATOMS_ASC[0].ticks;
        const newDuration = snapTick(currentTick - (part.position + note.position), fineSnap);
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
    if (drag.mode === 'erase-rect' && selectRect) {
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
      if (ids.length > 0) s.deleteNotes(ids);
      setSelectRect(null);
    } else if (drag.mode === 'select-rect' && selectRect) {
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
    if (lyricEditRef.current) return;
    const s = useSynthStore.getState();
    if (s.activeTool === 'pitch' || s.activeTool === 'erase') return;
    const { x, y } = canvasXY(e);
    const hit = hitTest(x, y, s.project, s.viewState, s.activePartIndex);
    if (hit.type === 'note') {
      const note = s.project.voice_parts[hit.partIndex]?.notes[hit.noteIndex];
      if (note) {
        lyricEditRef.current = true;
        setLyricEdit({ partIndex: hit.partIndex, noteIndex: hit.noteIndex, value: note.lyric });
      }
    } else if (hit.type === 'empty' && y > HEADER_H && x > KEYBOARD_W) {
      // Double-click empty area in select mode → switch to draw and create note
      const partPos = s.project.voice_parts[s.activePartIndex]?.position ?? 0;
      const rawTick = hit.tick - partPos;
      const maxRelTick = getPartMaxRelTick();
      if (rawTick >= maxRelTick) return; // beyond segment boundary
      s.setTool('draw');
      const snappedTick = snapTickFloor(rawTick, Math.min(s.viewState.snapTicks, RESOLUTION));
      const midi = Math.max(0, Math.min(127, hit.midi));
      playNote(midi);
      const notePos = Math.max(0, snappedTick);
      s.addNote(s.activePartIndex, {
        position: notePos, duration: s.viewState.snapTicks, tone: midi, lyric: useSynthStore.getState().language === 'ja' ? 'あ' : '啊',
        pitch: defaultPitch(s.viewState.snapTicks / ((s.project.tempos[0]?.bpm ?? 120) * 480 / 60) * 1000), vibrato: defaultVibrato(), tuning: 0, phoneme_expressions: [], phoneme_overrides: [],
      });
      const partIdx = s.activePartIndex;
      requestAnimationFrame(() => {
        const updatedPart = useSynthStore.getState().project.voice_parts[partIdx];
        if (updatedPart) {
          const ni = updatedPart.notes.findIndex(n => n.position === notePos && n.tone === midi);
          if (ni >= 0) {
            lyricEditRef.current = true;
            setLyricEdit({ partIndex: partIdx, noteIndex: ni, value: useSynthStore.getState().language === 'ja' ? 'あ' : '啊' });
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
        // Alt+wheel → cycle base values; beyond 1/1 repeat the sequence offset by N×1/1
        // Sequence: ...1/2, 1/1, 1/1+1/32T, 1/1+1/32, ..., 1/1+1/1, 2×1/1+1/32T, ...
        const cur = s.viewState.snapTicks;
        const wholeNote = SNAP_ATOMS_ASC[SNAP_ATOMS_ASC.length - 1].ticks;
        const base = cur % wholeNote || wholeNote; // current position within the cycle
        const floor = cur - base; // N × wholeNote below
        const idx = SNAP_ATOMS_ASC.findIndex(a => a.ticks === base);
        if (e.deltaY > 0) {
          // shorter
          if (idx > 0) {
            s.setViewState({ snapTicks: floor + SNAP_ATOMS_ASC[idx - 1].ticks });
          } else if (floor > 0) {
            // step down one whole-note tier: N×1/1 → (N-1)×1/1 + 1/1
            s.setViewState({ snapTicks: floor });
          }
          // else already at minimum (1/32T)
        } else {
          // longer
          if (idx >= 0 && idx < SNAP_ATOMS_ASC.length - 1) {
            s.setViewState({ snapTicks: floor + SNAP_ATOMS_ASC[idx + 1].ticks });
          } else {
            // wrap into next tier: current tier's 1/1 → next tier's 1/32T
            s.setViewState({ snapTicks: floor + wholeNote + SNAP_ATOMS_ASC[0].ticks });
          }
        }
      } else if (e.ctrlKey && e.shiftKey) {
        const factor = e.deltaY < 0 ? 1.15 : 1 / 1.15;
        const newPPS = Math.max(6, Math.min(80, s.viewState.pixelsPerSemitone * factor));
        const rect = canvas.getBoundingClientRect();
        const mouseY = e.clientY - rect.top;
        const midMidi = 127 - (s.viewState.scrollY + mouseY - HEADER_H) / s.viewState.pixelsPerSemitone;
        const newScrollY = (127 - midMidi) * newPPS - (mouseY - HEADER_H);
        const maxScrollY = Math.max(0, 128 * newPPS - (size.height - HEADER_H));
        s.setViewState({
          pixelsPerSemitone: newPPS,
          scrollY: Math.min(maxScrollY, Math.max(0, newScrollY)),
        });
      } else if (e.ctrlKey) {
        const rect = canvas.getBoundingClientRect();
        const mx = e.clientX - rect.left;
        const factor = e.deltaY < 0 ? 1.15 : 1 / 1.15;
        if (mx < KEYBOARD_W) {
          // Ctrl+wheel over keyboard → vertical zoom (pixelsPerSemitone)
          const newPPS = Math.max(6, Math.min(80, s.viewState.pixelsPerSemitone * factor));
          // Keep the mouse pitch row centered after zoom
          const mouseY = e.clientY - rect.top;
          const midMidi = 127 - (s.viewState.scrollY + mouseY - HEADER_H) / s.viewState.pixelsPerSemitone;
          const newScrollY = (127 - midMidi) * newPPS - (mouseY - HEADER_H);
          const maxScrollY = Math.max(0, 128 * newPPS - (size.height - HEADER_H));
          s.setViewState({
            pixelsPerSemitone: newPPS,
            scrollY: Math.min(maxScrollY, Math.max(0, newScrollY)),
          });
        } else {
          // Ctrl+wheel over canvas → horizontal zoom (pixelsPerTick)
          s.setViewState({ pixelsPerTick: Math.max(0.05, Math.min(5, s.viewState.pixelsPerTick * factor)) });
        }
      } else if (tabHeld.current) {
        // Tab+wheel → vertical zoom (pixelsPerSemitone) anywhere
        const factor = e.deltaY < 0 ? 1.15 : 1 / 1.15;
        const newPPS = Math.max(6, Math.min(80, s.viewState.pixelsPerSemitone * factor));
        const rect = canvas.getBoundingClientRect();
        const mouseY = e.clientY - rect.top;
        const midMidi = 127 - (s.viewState.scrollY + mouseY - HEADER_H) / s.viewState.pixelsPerSemitone;
        const newScrollY = (127 - midMidi) * newPPS - (mouseY - HEADER_H);
        const maxScrollY = Math.max(0, 128 * newPPS - (size.height - HEADER_H));
        s.setViewState({
          pixelsPerSemitone: newPPS,
          scrollY: Math.min(maxScrollY, Math.max(0, newScrollY)),
        });
      } else {
        const ap = s.project.voice_parts[s.activePartIndex];
        const hasNext = ap ? s.project.voice_parts.some(
          vp => vp.track_no === ap.track_no && vp.position > ap.position
        ) : false;
        const minScrollX = ap ? ap.position * s.viewState.pixelsPerTick : 0;
        const contentW = ap ? (ap.position + ap.duration) * s.viewState.pixelsPerTick : 0;
        const viewportW = size.width - KEYBOARD_W;
        const maxScrollX = hasNext ? Math.max(minScrollX, contentW - viewportW) : Infinity;
        const clampX = (v: number) => Math.max(minScrollX, Math.min(maxScrollX, v));
        if (e.shiftKey) {
          s.setViewState({ scrollX: clampX(s.viewState.scrollX + (e.deltaX || e.deltaY)) });
        } else {
          const maxScrollY = Math.max(0, 128 * s.viewState.pixelsPerSemitone - (size.height - HEADER_H));
          s.setViewState({
            scrollX: clampX(s.viewState.scrollX + e.deltaX),
            scrollY: Math.min(maxScrollY, Math.max(0, s.viewState.scrollY + e.deltaY)),
          });
        }
      }
    };
    canvas.addEventListener('wheel', handler, { passive: false });
    return () => canvas.removeEventListener('wheel', handler);
  }, [size.height]);

  useEffect(() => {
    const down = (e: KeyboardEvent) => { if (e.key === 'Tab') { e.preventDefault(); tabHeld.current = true; } };
    const up = (e: KeyboardEvent) => { if (e.key === 'Tab') tabHeld.current = false; };
    window.addEventListener('keydown', down, true);
    window.addEventListener('keyup', up, true);
    return () => { window.removeEventListener('keydown', down, true); window.removeEventListener('keyup', up, true); };
  }, []);

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
