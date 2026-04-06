/** Synth page state management (Zustand).
 *  Manages USTX project, undo/redo, selection, view state, and render state.
 */

import { create } from 'zustand';
import type { UProject, UNote, UWavePart } from '../types/ustx';
import type { TrackEffect } from '../lib/synth/effects';
import { RESOLUTION } from '../types/ustx';
import { createDefaultProject } from '../lib/synth/ustx';
import type { ViewState } from '../lib/synth/pianoRollEngine';

export type Tool = 'select' | 'draw' | 'erase';

export interface RVCSettings {
  f0_up_key: number;
  f0_method: string;
  index_rate: number;
  filter_radius: number;
  resample_sr: number;
  rms_mix_rate: number;
  protect: number;
}

export const defaultRVCSettings: RVCSettings = {
  f0_up_key: 0, f0_method: 'rmvpe', index_rate: 0.75, filter_radius: 3,
  resample_sr: 0, rms_mix_rate: 0.25, protect: 0.33,
};

const MAX_UNDO = 50;

function cloneProject(p: UProject): UProject {
  return JSON.parse(JSON.stringify(p));
}

interface SynthState {
  // Project
  project: UProject;
  dirty: boolean;

  // Undo/Redo
  undoStack: UProject[];
  redoStack: UProject[];

  // Selection
  selectedNotes: Set<string>;
  activeTool: Tool;
  activePartIndex: number;

  // View
  viewState: ViewState;

  // Models
  singer: string;
  rvcModel: string;
  rvcSettings: RVCSettings;

  // Render
  renderStatus: 'idle' | 'rendering' | 'converting' | 'done' | 'error';
  renderedUrl: string | null;
  finalUrl: string | null;
  renderError: string;

  // Transport
  viewMode: 'pianoroll' | 'arrangement';
  isPlaying: boolean;
  playbackPositionTicks: number;
  loopEnabled: boolean;
  loopStart: number;
  loopEnd: number;
  trackAudioUrls: Record<number, string>;  // trackIndex → rendered blob URL
  trackEffects: Record<number, TrackEffect[]>;  // trackIndex → effect chain

  // Actions - Project
  loadProject: (project: UProject) => void;
  newProject: () => void;
  markClean: () => void;

  // Actions - Notes
  addNote: (partIndex: number, note: UNote) => void;
  updateNote: (partIndex: number, noteIndex: number, changes: Partial<UNote>) => void;
  deleteNotes: (noteIds: string[]) => void;
  moveNotes: (noteIds: string[], deltaTick: number, deltaTone: number) => void;
  resizeNote: (partIndex: number, noteIndex: number, newDuration: number) => void;
  setLyric: (partIndex: number, noteIndex: number, lyric: string) => void;

  // Actions - Project settings
  setTempo: (bpm: number) => void;
  setTimeSignature: (beatPerBar: number, beatUnit: number) => void;
  setSinger: (trackIndex: number, singerId: string, phonemizer?: string) => void;

  // Actions - UI
  setTool: (tool: Tool) => void;
  setViewState: (partial: Partial<ViewState>) => void;
  selectNotes: (noteIds: string[], add?: boolean) => void;
  selectAll: () => void;
  clearSelection: () => void;
  setActivePart: (partIndex: number) => void;

  // Actions - Models
  setSingerModel: (singer: string) => void;
  setRvcModel: (model: string) => void;
  setRvcSettings: (settings: Partial<RVCSettings>) => void;

  // Actions - Render
  setRenderStatus: (status: SynthState['renderStatus']) => void;
  setRenderedUrl: (url: string | null) => void;
  setFinalUrl: (url: string | null) => void;
  setRenderError: (error: string) => void;

  // Actions - Transport
  setViewMode: (mode: 'pianoroll' | 'arrangement') => void;
  setPlaying: (playing: boolean) => void;
  setPlaybackPosition: (ticks: number) => void;
  toggleLoop: () => void;
  setLoopRange: (start: number, end: number) => void;
  setTrackAudioUrl: (trackIndex: number, url: string) => void;

  // Actions - Track management
  addTrack: (type: 'vocal' | 'audio', name?: string) => void;
  removeTrack: (trackIndex: number) => void;
  setTrackMute: (trackIndex: number, mute: boolean) => void;
  setTrackSolo: (trackIndex: number, solo: boolean) => void;
  setTrackVolume: (trackIndex: number, volume: number) => void;
  setTrackPan: (trackIndex: number, pan: number) => void;
  addWavePart: (trackNo: number, wavePart: UWavePart) => void;
  moveWavePart: (partIndex: number, newPosition: number) => void;

  // Actions - Effects
  addEffect: (trackIndex: number, effect: TrackEffect) => void;
  removeEffect: (trackIndex: number, effectIndex: number) => void;
  updateEffectParams: (trackIndex: number, effectIndex: number, params: Record<string, number>) => void;

  // Actions - Undo/Redo
  undo: () => void;
  redo: () => void;
}

function pushUndo(state: Pick<SynthState, 'project' | 'undoStack'>): { undoStack: UProject[]; redoStack: UProject[] } {
  return {
    undoStack: [...state.undoStack, cloneProject(state.project)].slice(-MAX_UNDO),
    redoStack: [],
  };
}

export const useSynthStore = create<SynthState>((set, get) => ({
  // Initial state
  project: createDefaultProject(),
  dirty: false,
  undoStack: [],
  redoStack: [],
  selectedNotes: new Set<string>(),
  activeTool: 'draw' as Tool,
  activePartIndex: 0,
  viewState: {
    scrollX: 0,
    scrollY: 60 * 20 - 200,
    pixelsPerTick: 0.5,
    pixelsPerSemitone: 20,
    snapTicks: RESOLUTION / 4,
  },
  singer: '',
  rvcModel: '',
  rvcSettings: { ...defaultRVCSettings },
  renderStatus: 'idle',
  renderedUrl: null,
  finalUrl: null,
  renderError: '',
  viewMode: 'pianoroll',
  isPlaying: false,
  playbackPositionTicks: 0,
  loopEnabled: false,
  loopStart: 0,
  loopEnd: 0,
  trackAudioUrls: {},
  trackEffects: {},

  // Project actions
  loadProject: (project) => set({
    project,
    dirty: false,
    undoStack: [],
    redoStack: [],
    selectedNotes: new Set(),
  }),

  newProject: () => set({
    project: createDefaultProject(),
    dirty: false,
    undoStack: [],
    redoStack: [],
    selectedNotes: new Set(),
    activePartIndex: 0,
    renderedUrl: null,
    finalUrl: null,
    renderError: '',
    renderStatus: 'idle',
  }),

  markClean: () => set({ dirty: false }),

  // Note actions
  addNote: (partIndex, note) => set((state) => {
    const stacks = pushUndo(state);
    const project = cloneProject(state.project);
    const part = project.voice_parts[partIndex];
    if (!part) return state;
    part.notes.push(note);
    part.notes.sort((a, b) => a.position - b.position);
    const noteEnd = note.position + note.duration;
    if (noteEnd > part.duration) part.duration = noteEnd + RESOLUTION;
    return { ...stacks, project, dirty: true };
  }),

  updateNote: (partIndex, noteIndex, changes) => set((state) => {
    const stacks = pushUndo(state);
    const project = cloneProject(state.project);
    const note = project.voice_parts[partIndex]?.notes[noteIndex];
    if (!note) return state;
    Object.assign(note, changes);
    return { ...stacks, project, dirty: true };
  }),

  deleteNotes: (noteIds) => set((state) => {
    if (noteIds.length === 0) return state;
    const stacks = pushUndo(state);
    const project = cloneProject(state.project);
    const byPart = new Map<number, Set<number>>();
    for (const id of noteIds) {
      const [pi, ni] = id.split('-').map(Number);
      if (!byPart.has(pi)) byPart.set(pi, new Set());
      byPart.get(pi)!.add(ni);
    }
    for (const [pi, indices] of byPart) {
      const part = project.voice_parts[pi];
      if (!part) continue;
      part.notes = part.notes.filter((_, i) => !indices.has(i));
    }
    return { ...stacks, project, dirty: true, selectedNotes: new Set() };
  }),

  moveNotes: (noteIds, deltaTick, deltaTone) => set((state) => {
    if (noteIds.length === 0) return state;
    const stacks = pushUndo(state);
    const project = cloneProject(state.project);
    for (const id of noteIds) {
      const [pi, ni] = id.split('-').map(Number);
      const note = project.voice_parts[pi]?.notes[ni];
      if (!note) continue;
      note.position = Math.max(0, note.position + deltaTick);
      note.tone = Math.max(0, Math.min(127, note.tone + deltaTone));
    }
    const affectedParts = new Set(noteIds.map(id => Number(id.split('-')[0])));
    for (const pi of affectedParts) {
      project.voice_parts[pi]?.notes.sort((a, b) => a.position - b.position);
    }
    return { ...stacks, project, dirty: true };
  }),

  resizeNote: (partIndex, noteIndex, newDuration) => set((state) => {
    const stacks = pushUndo(state);
    const project = cloneProject(state.project);
    const note = project.voice_parts[partIndex]?.notes[noteIndex];
    if (!note) return state;
    note.duration = Math.max(10, newDuration);
    return { ...stacks, project, dirty: true };
  }),

  setLyric: (partIndex, noteIndex, lyric) => set((state) => {
    const stacks = pushUndo(state);
    const project = cloneProject(state.project);
    const note = project.voice_parts[partIndex]?.notes[noteIndex];
    if (!note) return state;
    note.lyric = lyric;
    return { ...stacks, project, dirty: true };
  }),

  setTempo: (bpm) => set((state) => {
    const stacks = pushUndo(state);
    const project = cloneProject(state.project);
    if (project.tempos.length > 0) project.tempos[0].bpm = bpm;
    return { ...stacks, project, dirty: true };
  }),

  setTimeSignature: (beatPerBar, beatUnit) => set((state) => {
    const stacks = pushUndo(state);
    const project = cloneProject(state.project);
    if (project.time_signatures.length > 0) {
      project.time_signatures[0].beat_per_bar = beatPerBar;
      project.time_signatures[0].beat_unit = beatUnit;
    }
    return { ...stacks, project, dirty: true };
  }),

  setSinger: (trackIndex, singerId, phonemizer) => set((state) => {
    const stacks = pushUndo(state);
    const project = cloneProject(state.project);
    const track = project.tracks[trackIndex];
    if (track) {
      track.singer = singerId;
      if (phonemizer) track.phonemizer = phonemizer;
    }
    return { ...stacks, project, dirty: true };
  }),

  // UI actions
  setTool: (tool) => set({ activeTool: tool }),
  setViewState: (partial) => set((state) => ({
    viewState: { ...state.viewState, ...partial },
  })),
  selectNotes: (noteIds, add) => set((state) => ({
    selectedNotes: add
      ? new Set([...state.selectedNotes, ...noteIds])
      : new Set(noteIds),
  })),
  selectAll: () => set((state) => {
    const all = new Set<string>();
    state.project.voice_parts.forEach((part, pi) => {
      part.notes.forEach((_, ni) => all.add(`${pi}-${ni}`));
    });
    return { selectedNotes: all };
  }),
  clearSelection: () => set({ selectedNotes: new Set() }),
  setActivePart: (partIndex) => set({ activePartIndex: partIndex }),

  // Model actions
  setSingerModel: (singer) => set({ singer }),
  setRvcModel: (model) => set({ rvcModel: model }),
  setRvcSettings: (partial) => set((state) => ({
    rvcSettings: { ...state.rvcSettings, ...partial },
  })),

  // Render actions
  setRenderStatus: (status) => set({ renderStatus: status }),
  setRenderedUrl: (url) => set({ renderedUrl: url }),
  setFinalUrl: (url) => set({ finalUrl: url }),
  setRenderError: (error) => set({ renderError: error }),

  // Transport actions
  setViewMode: (mode) => set({ viewMode: mode }),
  setPlaying: (playing) => set({ isPlaying: playing }),
  setPlaybackPosition: (ticks) => set({ playbackPositionTicks: ticks }),
  toggleLoop: () => set((state) => ({ loopEnabled: !state.loopEnabled })),
  setLoopRange: (start, end) => set({ loopStart: start, loopEnd: end }),
  setTrackAudioUrl: (trackIndex, url) => set((state) => ({
    trackAudioUrls: { ...state.trackAudioUrls, [trackIndex]: url },
  })),

  // Track management
  addTrack: (type, name) => set((state) => {
    const stacks = pushUndo(state);
    const project = cloneProject(state.project);
    const idx = project.tracks.length;
    const colors = ['Blue', 'Pink', 'Green', 'Yellow', 'Purple', 'Orange'];
    project.tracks.push({
      track_name: name || `Track ${idx + 1}`,
      track_color: colors[idx % colors.length],
      track_type: type,
      mute: false, solo: false, volume: 0, pan: 0,
      voice_color_names: [''],
    });
    if (type === 'vocal') {
      project.voice_parts.push({
        name: `Part ${idx + 1}`,
        comment: '',
        track_no: idx,
        position: 0,
        duration: RESOLUTION * 16,
        notes: [],
        curves: [],
      });
    }
    return { ...stacks, project, dirty: true };
  }),

  removeTrack: (trackIndex) => set((state) => {
    if (state.project.tracks.length <= 1) return state;
    const stacks = pushUndo(state);
    const project = cloneProject(state.project);
    project.tracks.splice(trackIndex, 1);
    project.voice_parts = project.voice_parts.filter(vp => vp.track_no !== trackIndex)
      .map(vp => ({ ...vp, track_no: vp.track_no > trackIndex ? vp.track_no - 1 : vp.track_no }));
    project.wave_parts = project.wave_parts.filter(wp => wp.track_no !== trackIndex)
      .map(wp => ({ ...wp, track_no: wp.track_no > trackIndex ? wp.track_no - 1 : wp.track_no }));
    const newPartIdx = Math.min(state.activePartIndex, project.voice_parts.length - 1);
    return { ...stacks, project, dirty: true, activePartIndex: Math.max(0, newPartIdx) };
  }),

  setTrackMute: (trackIndex, mute) => set((state) => {
    const project = cloneProject(state.project);
    if (project.tracks[trackIndex]) project.tracks[trackIndex].mute = mute;
    return { project };
  }),

  setTrackSolo: (trackIndex, solo) => set((state) => {
    const project = cloneProject(state.project);
    if (project.tracks[trackIndex]) project.tracks[trackIndex].solo = solo;
    return { project };
  }),

  setTrackVolume: (trackIndex, volume) => set((state) => {
    const project = cloneProject(state.project);
    if (project.tracks[trackIndex]) project.tracks[trackIndex].volume = volume;
    return { project };
  }),

  setTrackPan: (trackIndex, pan) => set((state) => {
    const project = cloneProject(state.project);
    if (project.tracks[trackIndex]) project.tracks[trackIndex].pan = pan;
    return { project };
  }),

  addWavePart: (trackNo, wavePart) => set((state) => {
    const stacks = pushUndo(state);
    const project = cloneProject(state.project);
    project.wave_parts.push({ ...wavePart, track_no: trackNo });
    return { ...stacks, project, dirty: true };
  }),

  moveWavePart: (partIndex, newPosition) => set((state) => {
    const stacks = pushUndo(state);
    const project = cloneProject(state.project);
    if (project.wave_parts[partIndex]) {
      project.wave_parts[partIndex].position = Math.max(0, newPosition);
    }
    return { ...stacks, project, dirty: true };
  }),

  // Effects
  addEffect: (trackIndex, effect) => set((state) => ({
    trackEffects: {
      ...state.trackEffects,
      [trackIndex]: [...(state.trackEffects[trackIndex] || []), effect],
    },
  })),

  removeEffect: (trackIndex, effectIndex) => set((state) => {
    const chain = [...(state.trackEffects[trackIndex] || [])];
    chain.splice(effectIndex, 1);
    return { trackEffects: { ...state.trackEffects, [trackIndex]: chain } };
  }),

  updateEffectParams: (trackIndex, effectIndex, params) => set((state) => {
    const chain = [...(state.trackEffects[trackIndex] || [])];
    if (!chain[effectIndex]) return state;
    // Handle _enabled toggle
    if ('_enabled' in params) {
      chain[effectIndex] = { ...chain[effectIndex], enabled: params._enabled !== 0 };
    } else {
      chain[effectIndex] = { ...chain[effectIndex], params: { ...chain[effectIndex].params, ...params } };
    }
    return { trackEffects: { ...state.trackEffects, [trackIndex]: chain } };
  }),

  // Undo/Redo
  undo: () => set((state) => {
    if (state.undoStack.length === 0) return state;
    const undoStack = [...state.undoStack];
    const prev = undoStack.pop()!;
    const redoStack = [...state.redoStack, cloneProject(state.project)].slice(-MAX_UNDO);
    return { project: prev, undoStack, redoStack, dirty: true, selectedNotes: new Set() };
  }),

  redo: () => set((state) => {
    if (state.redoStack.length === 0) return state;
    const redoStack = [...state.redoStack];
    const next = redoStack.pop()!;
    const undoStack = [...state.undoStack, cloneProject(state.project)].slice(-MAX_UNDO);
    return { project: next, undoStack, redoStack, dirty: true, selectedNotes: new Set() };
  }),
}));
