/** Hit-testing and interaction logic for piano roll notes. */

import type { UProject } from '../../types/ustx';
import type { ViewState } from './pianoRollEngine';
import { screenToTick, screenToMidi, KEYBOARD_W, HEADER_H } from './pianoRollEngine';

const RESIZE_HANDLE_PX = 6;

export type HitResult =
  | { type: 'empty'; tick: number; midi: number }
  | { type: 'note'; partIndex: number; noteIndex: number; noteId: string }
  | { type: 'resize'; partIndex: number; noteIndex: number; noteId: string }
  | { type: 'keyboard' }
  | { type: 'header' };

/** Test what's under the cursor at (canvasX, canvasY). */
export function hitTest(
  x: number,
  y: number,
  project: UProject,
  viewState: ViewState,
  activePartIndex: number,
): HitResult {
  if (x < KEYBOARD_W) return { type: 'keyboard' };
  if (y < HEADER_H) return { type: 'header' };

  const tick = screenToTick(x, viewState);
  const midi = screenToMidi(y, viewState);
  const midiRounded = Math.floor(midi + 0.5);

  const partOrder = [activePartIndex, ...project.voice_parts.map((_, i) => i).filter(i => i !== activePartIndex)];

  for (const pi of partOrder) {
    const part = project.voice_parts[pi];
    if (!part) continue;

    for (let ni = part.notes.length - 1; ni >= 0; ni--) {
      const note = part.notes[ni];
      const absPos = part.position + note.position;
      const absEnd = absPos + note.duration;

      if (tick >= absPos && tick <= absEnd && midiRounded === note.tone) {
        const noteId = `${pi}-${ni}`;

        const rightEdgeTick = absEnd;
        const rightEdgePx = rightEdgeTick * viewState.pixelsPerTick - viewState.scrollX + KEYBOARD_W;
        if (Math.abs(x - rightEdgePx) < RESIZE_HANDLE_PX) {
          return { type: 'resize', partIndex: pi, noteIndex: ni, noteId };
        }

        return { type: 'note', partIndex: pi, noteIndex: ni, noteId };
      }
    }
  }

  return { type: 'empty', tick: Math.max(0, tick), midi: Math.max(0, Math.min(127, midiRounded)) };
}
