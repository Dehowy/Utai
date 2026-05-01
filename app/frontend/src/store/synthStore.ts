/** Synth page state management (Zustand).
 *  Manages USTX project, undo/redo, selection, view state, and render state.
 */

import { create } from 'zustand';
import type { UProject, UNote, UWavePart, UInstrumentNote, UInstrumentPart } from '../types/ustx';
import type { TrackEffect } from '../lib/synth/effects';
import { RESOLUTION } from '../types/ustx';
import { createDefaultProject } from '../lib/synth/ustx';
import type { ViewState } from '../lib/synth/pianoRollEngine';
import i18n from '../i18n/i18n';

function getDefaultSynthLang(): 'zh' | 'ja' {
  const uiLang = i18n.language?.substring(0, 2);
  if (uiLang === 'zh') return 'zh';
  return 'ja'; // ja and en both default to ja
}

export type Tool = 'select' | 'draw' | 'erase' | 'pitch';

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
  activeInstrumentPartIndex: number;  // -1 = no instrument part active, >= 0 = editing instrument part
  activeWavePartIndex: number;  // -1 = no wave part active, >= 0 = editing audio track

  // View
  viewState: ViewState;

  // Models
  singer: string;
  rvcModel: string;
  trackRvcModel: Record<number, string>;  // per-track RVC model override
  rvcSettings: RVCSettings;
  language: 'zh' | 'ja';
  trackLanguage: Record<number, string>;      // per-track language override
  trackIndexRate: Record<number, number>;     // per-track index rate override
  trackF0Offset: Record<number, number>;      // per-track pitch offset (semitones)
  muteStrategy: 'auto' | 'raw' | 'inline' | 'both';
  trackMuteStrategy: Record<number, string>;  // per-track mute strategy override
  globalF0Offset: number;
  pitchSnap: boolean;
  affineTransform: boolean;                    // global affine pitch transform toggle
  trackAffineTransform: Record<number, boolean>; // per-track override
  cosineIndex: boolean;                         // global cosine-normalized index retrieval
  modelF0: boolean;                             // global model-predicted F0 toggle
  trackModelF0: Record<number, boolean>;        // per-track override
  ghostTracks: number[];  // track indices to show as reference overlay in piano roll

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
  arrScrollY: number;  // arrangement vertical scroll (separate from pianoroll scrollY)
  trackAudioUrls: Record<number, string>;  // trackIndex → rendered blob URL (post-effects)
  trackRawAudioUrls: Record<number, string>;  // trackIndex → raw S2H output (pre-effects)
  waveFxUrls: Record<string, string>;  // buffer_id → effected blob URL (audio tracks)
  audioFxRendering: Record<number, boolean>;  // trackIndex → true while effects are processing
  peakData: Record<string, Float32Array>;  // buffer_id → peak array
  trackEffects: Record<number, TrackEffect[]>;  // trackIndex → effect chain
  trackRenderedSnapshot: Record<number, string>;  // trackIndex → fingerprint at render time
  vstStateVersion: Record<number, number>;  // trackIndex → bumped when VST editor saves state

  // Project persistence
  projectPath: string | null;
  isSaving: boolean;
  setProjectPath: (path: string | null) => void;
  setIsSaving: (saving: boolean) => void;

  // Actions - Project
  pushUndoSnapshot: () => void;
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
  setPitchPoint: (partIndex: number, noteIndex: number, pointIndex: number, x: number, y: number) => void;
  addPitchPoint: (partIndex: number, noteIndex: number, x: number, y: number, shape?: string) => void;
  deletePitchPoint: (partIndex: number, noteIndex: number, pointIndex: number) => void;

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
  setActiveInstrumentPart: (partIndex: number) => void;
  setActiveWavePart: (partIndex: number) => void;
  splitWavePart: (partIndex: number, splitTick: number) => void;

  // Track selection (arrangement view — for effects editing)
  selectedTrackIndex: number;
  setSelectedTrackIndex: (idx: number) => void;

  // Actions - Segments (arrangement view)
  selectedSegment: { type: 'voice' | 'instrument' | 'wave'; index: number } | null;
  setSelectedSegment: (seg: { type: 'voice' | 'instrument' | 'wave'; index: number } | null) => void;
  renameVoicePart: (partIndex: number, name: string) => void;
  renameWavePart: (partIndex: number, name: string) => void;
  setClipVolume: (type: 'voice' | 'instrument' | 'wave', index: number, volume: number) => void;
  setClipPan: (type: 'voice' | 'instrument' | 'wave', index: number, pan: number) => void;
  splitVoicePart: (partIndex: number, splitTick: number) => void;
  deleteVoicePart: (partIndex: number) => void;
  moveVoicePart: (partIndex: number, deltaTicks: number) => void;
  addVoicePartOnTrack: (trackNo: number, position: number) => void;
  splitInstrumentPart: (partIndex: number, splitTick: number) => void;
  deleteInstrumentPart: (partIndex: number) => void;
  moveInstrumentPart: (partIndex: number, deltaTicks: number) => void;
  addInstrumentPartOnTrack: (trackNo: number, position: number) => void;
  deleteWavePart: (partIndex: number) => void;

  // Actions - Models
  setSingerModel: (singer: string) => void;
  setRvcModel: (model: string) => void;
  setTrackRvcModel: (trackIndex: number, model: string) => void;
  setRvcSettings: (settings: Partial<RVCSettings>) => void;
  setLanguage: (lang: 'zh' | 'ja') => void;
  setTrackLanguage: (trackIndex: number, lang: string) => void;
  setTrackIndexRate: (trackIndex: number, rate: number) => void;
  setTrackF0Offset: (trackIndex: number, offset: number) => void;
  setMuteStrategy: (strategy: 'auto' | 'raw' | 'inline' | 'both') => void;
  setTrackMuteStrategy: (trackIndex: number, strategy: string) => void;
  setGlobalF0Offset: (offset: number) => void;
  togglePitchSnap: () => void;
  setAffineTransform: (enabled: boolean) => void;
  setTrackAffineTransform: (trackIndex: number, enabled: boolean) => void;
  setCosineIndex: (enabled: boolean) => void;
  setModelF0: (enabled: boolean) => void;
  setTrackModelF0: (trackIndex: number, enabled: boolean) => void;
  toggleGhostTrack: (trackIndex: number) => void;
  rvcModelListVersion: number;
  bumpRvcModelList: () => void;

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
  setTrackRawAudioUrl: (trackIndex: number, url: string) => void;
  setPeakData: (bufferId: string, peaks: Float32Array) => void;

  // Actions - Track management
  addTrack: (type: 'vocal' | 'audio' | 'instrument', name?: string) => void;
  removeTrack: (trackIndex: number) => void;
  renameTrack: (trackIndex: number, name: string) => void;
  setTrackMute: (trackIndex: number, mute: boolean) => void;
  setTrackSolo: (trackIndex: number, solo: boolean) => void;
  setTrackVolume: (trackIndex: number, volume: number) => void;
  setTrackPan: (trackIndex: number, pan: number) => void;
  reorderTrack: (fromIndex: number, toIndex: number) => void;
  moveSegmentToTrack: (seg: { type: 'voice' | 'instrument' | 'wave'; index: number }, newTrackNo: number) => void;
  addWavePart: (trackNo: number, wavePart: UWavePart) => void;
  moveWavePart: (partIndex: number, newPosition: number) => void;

  // Actions - Instrument parts
  addInstrumentNote: (partIndex: number, note: UInstrumentNote) => void;
  updateInstrumentNote: (partIndex: number, noteIndex: number, changes: Partial<UInstrumentNote>) => void;
  deleteInstrumentNotes: (noteIds: string[]) => void;
  moveInstrumentNotes: (noteIds: string[], deltaTick: number, deltaTone: number) => void;
  resizeInstrumentNote: (partIndex: number, noteIndex: number, newDuration: number) => void;
  setTrackPlugin: (trackIndex: number, pluginName: string) => void;

  // Actions - Effects
  addEffect: (trackIndex: number, effect: TrackEffect) => void;
  removeEffect: (trackIndex: number, effectIndex: number) => void;
  updateEffectParams: (trackIndex: number, effectIndex: number, params: Record<string, number>) => void;

  // Render snapshots
  setTrackRenderedSnapshot: (trackIndex: number, snapshot: string) => void;
  bumpVstStateVersion: (trackIndex: number) => void;

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

/** Shift effect scope keys after a part array splice (insert or delete).
 *  prefix: 'v' | 'w' | 'i';  atIndex: the index being inserted/deleted */
function adjustEffectScopes(
  effects: Record<number, TrackEffect[]>,
  prefix: string,
  atIndex: number,
  mode: 'insert' | 'delete',
): Record<number, TrackEffect[]> {
  const result: Record<number, TrackEffect[]> = {};
  for (const [k, arr] of Object.entries(effects)) {
    let list = arr;
    if (mode === 'delete') {
      list = list.filter(fx => fx.scope !== `${prefix}${atIndex}`);
    }
    result[Number(k)] = list.map(fx => {
      if (typeof fx.scope !== 'string' || !fx.scope.startsWith(prefix)) return fx;
      const idx = parseInt(fx.scope.slice(prefix.length), 10);
      if (isNaN(idx) || idx <= atIndex) return fx;
      return { ...fx, scope: `${prefix}${idx + (mode === 'insert' ? 1 : -1)}` };
    });
  }
  return result;
}

export const useSynthStore = create<SynthState>((set, get) => ({
  // Initial state
  project: createDefaultProject(),
  dirty: false,
  undoStack: [],
  redoStack: [],
  selectedNotes: new Set<string>(),
  activeTool: 'select' as Tool,
  activePartIndex: 0,
  activeInstrumentPartIndex: -1,
  activeWavePartIndex: -1,
  selectedTrackIndex: 0,
  selectedSegment: null,
  viewState: {
    scrollX: 0,
    scrollY: 60 * 20 - 200,
    pixelsPerTick: 0.15,
    pixelsPerSemitone: 20,
    snapTicks: RESOLUTION / 4,
    trackLaneHeight: 112,
  },
  singer: '',
  rvcModel: '',
  trackRvcModel: {},
  rvcSettings: { ...defaultRVCSettings },
  language: getDefaultSynthLang(),
  trackLanguage: {},
  trackIndexRate: {},
  trackF0Offset: {},
  muteStrategy: 'auto',
  trackMuteStrategy: {},
  globalF0Offset: 0,
  pitchSnap: true,
  affineTransform: false,
  trackAffineTransform: {},
  cosineIndex: true,
  modelF0: false,
  trackModelF0: {},
  ghostTracks: [],
  renderStatus: 'idle',
  renderedUrl: null,
  finalUrl: null,
  renderError: '',
  viewMode: 'arrangement',
  isPlaying: false,
  playbackPositionTicks: 0,
  loopEnabled: false,
  loopStart: 0,
  loopEnd: 0,
  arrScrollY: 0,
  trackAudioUrls: {},
      trackRawAudioUrls: {},
      waveFxUrls: {},
      audioFxRendering: {},
  peakData: {},
  trackEffects: {},
  trackRenderedSnapshot: {},
  vstStateVersion: {},
  projectPath: null,
  isSaving: false,
  setProjectPath: (path) => set({ projectPath: path }),
  setIsSaving: (saving) => set({ isSaving: saving }),

  // Snapshot undo state without modifying project (call from slider mousedown)
  pushUndoSnapshot: () => set((state) => pushUndo(state)),

  // Project actions
  loadProject: (project) => set((state) => {
    for (const url of Object.values(state.trackAudioUrls)) { if (url) URL.revokeObjectURL(url); }
    if (state.finalUrl) URL.revokeObjectURL(state.finalUrl);
    const trackLanguage: Record<number, string> = {};
    for (const vp of project.voice_parts || []) {
      if (trackLanguage[vp.track_no]) continue;
      const allLyrics = vp.notes.map((n: any) => n.lyric || '').join('');
      if (allLyrics.length > 0) {
        const jaCount = [...allLyrics].filter(c => {
          const cp = c.codePointAt(0) || 0;
          return (cp >= 0x3040 && cp <= 0x309F) || (cp >= 0x30A0 && cp <= 0x30FF);
        }).length;
        trackLanguage[vp.track_no] = jaCount / allLyrics.length > 0.5 ? 'ja' : 'zh';
      }
    }
    return {
      project,
      dirty: true,
      undoStack: [],
      redoStack: [],
      selectedNotes: new Set(),
      activePartIndex: 0,
      activeInstrumentPartIndex: -1,
      activeWavePartIndex: -1,
      selectedTrackIndex: 0,
      selectedSegment: null,
      viewMode: 'arrangement' as const,
      isPlaying: false,
      playbackPositionTicks: 0,
      loopEnabled: false,
      loopStart: 0,
      loopEnd: 0,
      arrScrollY: 0,
      trackAudioUrls: {},
      trackRawAudioUrls: {},
      waveFxUrls: {},
      audioFxRendering: {},
      peakData: {},
      trackEffects: {},
      trackRenderedSnapshot: {},
      vstStateVersion: {},
      trackRvcModel: {},
      trackLanguage,
      trackIndexRate: {},
      trackF0Offset: {},
      trackMuteStrategy: {},
      trackAffineTransform: {},
      cosineIndex: true,
      modelF0: false,
      trackModelF0: {},
      ghostTracks: [],
      affineTransform: false,
      globalF0Offset: 0,
      projectPath: null,
      renderedUrl: null,
      finalUrl: null,
      renderError: '',
      renderStatus: 'idle' as const,
    };
  }),

  newProject: () => set((state) => {
    for (const url of Object.values(state.trackAudioUrls)) { if (url) URL.revokeObjectURL(url); }
    if (state.finalUrl) URL.revokeObjectURL(state.finalUrl);
    return {
      project: createDefaultProject(),
      dirty: false,
      undoStack: [],
      redoStack: [],
      selectedNotes: new Set(),
      activePartIndex: 0,
      activeInstrumentPartIndex: -1,
      activeWavePartIndex: -1,
      selectedTrackIndex: 0,
      selectedSegment: null,
      viewMode: 'arrangement' as const,
      isPlaying: false,
      playbackPositionTicks: 0,
      loopEnabled: false,
      loopStart: 0,
      loopEnd: 0,
      arrScrollY: 0,
      trackAudioUrls: {},
      trackRawAudioUrls: {},
      waveFxUrls: {},
      audioFxRendering: {},
      peakData: {},
      trackEffects: {},
      trackRenderedSnapshot: {},
      vstStateVersion: {},
      rvcModel: '',
      trackRvcModel: {},
      trackLanguage: {},
      trackIndexRate: {},
      trackF0Offset: {},
      trackMuteStrategy: {},
      trackAffineTransform: {},
      cosineIndex: true,
      modelF0: false,
      trackModelF0: {},
      ghostTracks: [],
      affineTransform: false,
      globalF0Offset: 0,
      projectPath: null,
      renderedUrl: null,
      finalUrl: null,
      renderError: '',
      renderStatus: 'idle' as const,
    };
  }),

  markClean: () => set({ dirty: false }),

  // Note actions
  addNote: (partIndex, note) => set((state) => {
    const stacks = pushUndo(state);
    const project = cloneProject(state.project);
    const part = project.voice_parts[partIndex];
    if (!part) return state;
    // Check if this is the last part on its track (no next part after it)
    const nextPartOnTrack = project.voice_parts.find(
      vp => vp.track_no === part.track_no && vp.position > part.position
    );
    const maxDuration = nextPartOnTrack ? nextPartOnTrack.position - part.position : Infinity;
    // Clamp note to not exceed part boundary if there's a next segment
    if (note.position + note.duration > maxDuration) {
      note.duration = Math.max(10, maxDuration - note.position);
    }
    if (note.position >= maxDuration) return state; // can't place note beyond boundary
    part.notes.push(note);
    part.notes.sort((a, b) => a.position - b.position);
    const noteEnd = note.position + note.duration;
    // Only auto-expand if this is the last segment on the track
    if (noteEnd > part.duration && !nextPartOnTrack) part.duration = noteEnd + RESOLUTION;
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
      const part = project.voice_parts[pi];
      const note = part?.notes[ni];
      if (!note || !part) continue;
      const nextPart = project.voice_parts.find(
        vp => vp.track_no === part.track_no && vp.position > part.position
      );
      const maxPos = nextPart ? nextPart.position - part.position - note.duration : Infinity;
      note.position = Math.max(0, Math.min(maxPos, note.position + deltaTick));
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
    const part = project.voice_parts[partIndex];
    const note = part?.notes[noteIndex];
    if (!note || !part) return state;
    const nextPart = project.voice_parts.find(
      vp => vp.track_no === part.track_no && vp.position > part.position
    );
    const maxDur = nextPart ? nextPart.position - part.position - note.position : Infinity;
    note.duration = Math.max(10, Math.min(maxDur, newDuration));
    // Only auto-expand if last segment
    if (!nextPart && note.position + note.duration > part.duration) {
      part.duration = note.position + note.duration + RESOLUTION;
    }
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

  setPitchPoint: (partIndex, noteIndex, pointIndex, x, y) => set((state) => {
    const stacks = pushUndo(state);
    const project = cloneProject(state.project);
    const note = project.voice_parts[partIndex]?.notes[noteIndex];
    if (!note?.pitch?.data?.[pointIndex]) return state;
    note.pitch.data[pointIndex].x = x;
    note.pitch.data[pointIndex].y = y;
    return { ...stacks, project, dirty: true };
  }),

  addPitchPoint: (partIndex, noteIndex, x, y, shape) => set((state) => {
    const stacks = pushUndo(state);
    const project = cloneProject(state.project);
    const note = project.voice_parts[partIndex]?.notes[noteIndex];
    if (!note) return state;
    if (!note.pitch) note.pitch = { snap_first: true, data: [] };
    note.pitch.data.push({ x, y, shape: (shape || 'io') as any });
    note.pitch.data.sort((a: any, b: any) => a.x - b.x);
    return { ...stacks, project, dirty: true };
  }),

  deletePitchPoint: (partIndex, noteIndex, pointIndex) => set((state) => {
    const stacks = pushUndo(state);
    const project = cloneProject(state.project);
    const note = project.voice_parts[partIndex]?.notes[noteIndex];
    if (!note?.pitch?.data?.[pointIndex]) return state;
    note.pitch.data.splice(pointIndex, 1);
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
    state.project.instrument_parts.forEach((part, pi) => {
      part.notes.forEach((_, ni) => all.add(`i${pi}-${ni}`));
    });
    return { selectedNotes: all };
  }),
  clearSelection: () => set({ selectedNotes: new Set() }),
  setActivePart: (partIndex) => set((state) => {
    const part = state.project.voice_parts[partIndex];
    const scrollX = part ? part.position * state.viewState.pixelsPerTick : 0;
    const trackNo = part?.track_no ?? -1;
    const trackLang = trackNo >= 0 ? state.trackLanguage[trackNo] : undefined;
    const trackMute = trackNo >= 0 ? state.trackMuteStrategy[trackNo] : undefined;
    return {
      activePartIndex: partIndex,
      activeInstrumentPartIndex: -1,
      activeWavePartIndex: -1,
      viewState: { ...state.viewState, scrollX },
      ...(trackLang ? { language: trackLang as any } : {}),
      ...(trackMute ? { muteStrategy: trackMute as any } : {}),
    };
  }),
  setActiveInstrumentPart: (partIndex) => set({ activeInstrumentPartIndex: partIndex, activePartIndex: -1, activeWavePartIndex: -1 }),
  setActiveWavePart: (partIndex) => set((state) => {
    const wp = state.project.wave_parts[partIndex];
    const scrollX = wp ? wp.position * state.viewState.pixelsPerTick : 0;
    return { activeWavePartIndex: partIndex, activeInstrumentPartIndex: -1, viewState: { ...state.viewState, scrollX } };
  }),
  splitWavePart: (_, splitTick) => set((state) => {
    const stacks = pushUndo(state);
    const project = cloneProject(state.project);
    const bpm = project.tempos[0]?.bpm ?? 120;
    const ticksPerSec = bpm / 60 * RESOLUTION;

    // Find the wave part that actually contains splitTick
    const wpIdx = project.wave_parts.findIndex(wp => {
      const endTick = wp.position + Math.round((wp.file_duration_ms / 1000) * ticksPerSec);
      return splitTick > wp.position && splitTick < endTick;
    });
    if (wpIdx < 0) return state;
    const wp = project.wave_parts[wpIdx];

    const relSplit = splitTick - wp.position;
    const splitMs = relSplit * 60000 / (bpm * RESOLUTION);
    const origSkip = wp.skip || 0;
    const wpB: typeof wp = {
      ...wp,
      name: wp.name.replace(/\s*#\d+$/, '') + ' #' + (project.wave_parts.filter(w => w.track_no === wp.track_no).length + 1),
      position: splitTick,
      skip: origSkip + splitMs,
      file_duration_ms: wp.file_duration_ms - splitMs,
    };
    wp.file_duration_ms = splitMs;
    wp.trim = 0;
    project.wave_parts.splice(wpIdx + 1, 0, wpB);
    const curActive = state.activeWavePartIndex;
    const newActive = curActive === wpIdx ? wpIdx : curActive > wpIdx ? curActive + 1 : curActive;
    // Shift existing scope keys for wave parts after the split point, then inherit
    let newEffects = adjustEffectScopes({ ...state.trackEffects }, 'w', wpIdx, 'insert');
    const fxKey = `w${wpIdx}`;
    const newFxKey = `w${wpIdx + 1}`;
    const clipFx = (newEffects[wp.track_no] || []).filter(fx => fx.scope === fxKey);
    if (clipFx.length > 0) {
      const inherited = clipFx.map(fx => ({ ...fx, scope: newFxKey }));
      newEffects[wp.track_no] = [...(newEffects[wp.track_no] || []), ...inherited];
    }
    return { ...stacks, project, dirty: true, activeWavePartIndex: newActive, trackEffects: newEffects };
  }),

  // Segment management
  setSelectedTrackIndex: (idx) => set({ selectedTrackIndex: idx }),
  setSelectedSegment: (seg) => set({ selectedSegment: seg }),

  renameVoicePart: (partIndex, name) => set((state) => {
    const stacks = pushUndo(state);
    const project = cloneProject(state.project);
    const part = project.voice_parts[partIndex];
    if (part) part.name = name;
    return { ...stacks, project, dirty: true };
  }),

  renameWavePart: (partIndex, name) => set((state) => {
    const stacks = pushUndo(state);
    const project = cloneProject(state.project);
    const part = project.wave_parts[partIndex];
    if (part) part.name = name;
    return { ...stacks, project, dirty: true };
  }),

  setClipVolume: (type, index, volume) => set((state) => {
    const project = cloneProject(state.project);
    const arr = type === 'voice' ? project.voice_parts : type === 'wave' ? project.wave_parts : project.instrument_parts;
    if (arr[index]) (arr[index] as any).volume = volume;
    return { project, dirty: true };
  }),
  setClipPan: (type, index, pan) => set((state) => {
    const project = cloneProject(state.project);
    const arr = type === 'voice' ? project.voice_parts : type === 'wave' ? project.wave_parts : project.instrument_parts;
    if (arr[index]) (arr[index] as any).pan = pan;
    return { project, dirty: true };
  }),

  splitVoicePart: (partIndex, splitTick) => set((state) => {
    const stacks = pushUndo(state);
    const project = cloneProject(state.project);
    const part = project.voice_parts[partIndex];
    if (!part) return state;
    const relSplit = splitTick - part.position;
    if (relSplit <= 0 || relSplit >= part.duration) return state;

    // Part A: notes before split
    const notesA = part.notes.filter(n => n.position < relSplit);
    // Part B: notes at/after split (positions shifted)
    const notesB = part.notes.filter(n => n.position >= relSplit)
      .map(n => ({ ...n, position: n.position - relSplit }));

    // Name: strip existing trailing number, assign sequential
    const baseName = part.name.replace(/\s*#\d+$/, '');
    const siblingCount = project.voice_parts.filter(vp => vp.track_no === part.track_no).length;
    part.name = baseName + ' #' + siblingCount;
    const partB = {
      name: baseName + ' #' + (siblingCount + 1),
      comment: '',
      track_no: part.track_no,
      position: splitTick,
      duration: part.duration - relSplit,
      notes: notesB,
      curves: [],
      volume: part.volume,
      pan: part.pan,
    };
    part.notes = notesA;
    part.duration = relSplit;
    project.voice_parts.splice(partIndex + 1, 0, partB);
    // Shift existing scope keys for voice parts after the split point, then inherit
    let newEffects = adjustEffectScopes({ ...state.trackEffects }, 'v', partIndex, 'insert');
    const fxKey = `v${partIndex}`;
    const newFxKey = `v${partIndex + 1}`;
    const clipFx = (newEffects[part.track_no] || []).filter(fx => fx.scope === fxKey);
    if (clipFx.length > 0) {
      const inherited = clipFx.map(fx => ({ ...fx, scope: newFxKey }));
      newEffects[part.track_no] = [...(newEffects[part.track_no] || []), ...inherited];
    }
    return { ...stacks, project, dirty: true, trackEffects: newEffects };
  }),

  deleteVoicePart: (partIndex) => set((state) => {
    const stacks = pushUndo(state);
    const project = cloneProject(state.project);
    project.voice_parts.splice(partIndex, 1);
    const newActive = Math.min(state.activePartIndex, project.voice_parts.length - 1);
    const newEffects = adjustEffectScopes({ ...state.trackEffects }, 'v', partIndex, 'delete');
    return { ...stacks, project, dirty: true, activePartIndex: Math.max(0, newActive), selectedSegment: null, selectedNotes: new Set(), trackEffects: newEffects };
  }),

  moveVoicePart: (partIndex, deltaTicks) => set((state) => {
    const stacks = pushUndo(state);
    const project = cloneProject(state.project);
    const part = project.voice_parts[partIndex];
    if (!part) return state;
    const newPos = Math.max(0, part.position + deltaTicks);
    const newEnd = newPos + part.duration;
    // Check collision with other parts on same track
    const siblings = project.voice_parts.filter((vp, i) => i !== partIndex && vp.track_no === part.track_no);
    for (const sib of siblings) {
      const sibEnd = sib.position + sib.duration;
      if (newPos < sibEnd && newEnd > sib.position) return state; // overlap → reject
    }
    part.position = newPos;
    return { ...stacks, project, dirty: true };
  }),

  addVoicePartOnTrack: (trackNo, position) => set((state) => {
    const stacks = pushUndo(state);
    const project = cloneProject(state.project);
    const overlapping = project.voice_parts.some(vp =>
      vp.track_no === trackNo && position >= vp.position && position < vp.position + vp.duration
    );
    if (overlapping) return state;
    const nextPart = project.voice_parts
      .filter(vp => vp.track_no === trackNo && vp.position > position)
      .sort((a, b) => a.position - b.position)[0];
    const defaultDur = RESOLUTION * 32;
    const duration = nextPart ? Math.min(defaultDur, nextPart.position - position) : defaultDur;
    // Preserve scrollX (prevent view jump in arrangement mode)
    const viewState = { ...state.viewState };
    project.voice_parts.push({
      name: `Part ${project.voice_parts.length + 1}`,
      comment: '',
      track_no: trackNo,
      position,
      duration,
      notes: [],
      curves: [],
    });
    return { ...stacks, project, viewState, dirty: true };
  }),

  splitInstrumentPart: (partIndex, splitTick) => set((state) => {
    const stacks = pushUndo(state);
    const project = cloneProject(state.project);
    const part = project.instrument_parts[partIndex];
    if (!part) return state;
    const relSplit = splitTick - part.position;
    if (relSplit <= 0 || relSplit >= part.duration) return state;

    const notesA = part.notes.filter(n => n.position < relSplit);
    const notesB = part.notes.filter(n => n.position >= relSplit)
      .map(n => ({ ...n, position: n.position - relSplit }));

    const baseName2 = part.name.replace(/\s*#\d+$/, '');
    const sibCount2 = project.instrument_parts.filter(ip => ip.track_no === part.track_no).length;
    part.name = baseName2 + ' #' + sibCount2;
    const partB = { name: baseName2 + ' #' + (sibCount2 + 1), track_no: part.track_no, position: splitTick, duration: part.duration - relSplit, notes: notesB, volume: part.volume, pan: part.pan };
    part.notes = notesA;
    part.duration = relSplit;
    project.instrument_parts.splice(partIndex + 1, 0, partB);
    let newEffects = adjustEffectScopes({ ...state.trackEffects }, 'i', partIndex, 'insert');
    const fxKey = `i${partIndex}`;
    const newFxKey = `i${partIndex + 1}`;
    const clipFx = (newEffects[part.track_no] || []).filter(fx => fx.scope === fxKey);
    if (clipFx.length > 0) {
      const inherited = clipFx.map(fx => ({ ...fx, scope: newFxKey }));
      newEffects[part.track_no] = [...(newEffects[part.track_no] || []), ...inherited];
    }
    return { ...stacks, project, dirty: true, trackEffects: newEffects };
  }),

  deleteInstrumentPart: (partIndex) => set((state) => {
    const stacks = pushUndo(state);
    const project = cloneProject(state.project);
    project.instrument_parts.splice(partIndex, 1);
    const newEffects = adjustEffectScopes({ ...state.trackEffects }, 'i', partIndex, 'delete');
    return { ...stacks, project, dirty: true, activeInstrumentPartIndex: -1, selectedSegment: null, selectedNotes: new Set(), trackEffects: newEffects };
  }),

  moveInstrumentPart: (partIndex, deltaTicks) => set((state) => {
    const stacks = pushUndo(state);
    const project = cloneProject(state.project);
    const part = project.instrument_parts[partIndex];
    if (part) part.position = Math.max(0, part.position + deltaTicks);
    return { ...stacks, project, dirty: true };
  }),

  addInstrumentPartOnTrack: (trackNo, position) => set((state) => {
    const stacks = pushUndo(state);
    const project = cloneProject(state.project);
    project.instrument_parts.push({
      name: `Inst ${project.instrument_parts.length + 1}`,
      track_no: trackNo,
      position,
      duration: RESOLUTION * 32,
      notes: [],
    });
    return { ...stacks, project, dirty: true };
  }),

  // Model actions
  setSingerModel: (singer) => set({ singer }),
  setRvcModel: (model) => set((state) => {
    const part = state.project.voice_parts[state.activePartIndex];
    let trackNo = part?.track_no ?? -1;
    if (trackNo < 0 && state.selectedTrackIndex >= 0) {
      const sel = state.project.tracks[state.selectedTrackIndex];
      if (sel?.track_type === 'vocal') trackNo = state.selectedTrackIndex;
    }
    if (trackNo < 0) {
      const idx = state.project.tracks.findIndex(t => t.track_type === 'vocal');
      if (idx >= 0) trackNo = idx;
    }
    return {
      dirty: true,
      trackRvcModel: trackNo >= 0
        ? { ...state.trackRvcModel, [trackNo]: model }
        : state.trackRvcModel,
    };
  }),
  setTrackRvcModel: (trackIndex, model) => set((state) => ({
    dirty: true, trackRvcModel: { ...state.trackRvcModel, [trackIndex]: model },
  })),
  setRvcSettings: (partial) => set((state) => ({
    rvcSettings: { ...state.rvcSettings, ...partial },
  })),
  setLanguage: (lang) => set((state) => {
    const part = state.project.voice_parts[state.activePartIndex];
    let trackNo = part?.track_no ?? -1;
    if (trackNo < 0 && state.selectedTrackIndex >= 0) {
      const sel = state.project.tracks[state.selectedTrackIndex];
      if (sel?.track_type === 'vocal') trackNo = state.selectedTrackIndex;
    }
    if (trackNo < 0) {
      const idx = state.project.tracks.findIndex(t => t.track_type === 'vocal');
      if (idx >= 0) trackNo = idx;
    }
    return {
      dirty: true,
      language: lang,
      trackLanguage: trackNo >= 0
        ? { ...state.trackLanguage, [trackNo]: lang }
        : state.trackLanguage,
    };
  }),
  setTrackLanguage: (trackIndex: number, lang: string) => set((state) => ({
    dirty: true, trackLanguage: { ...state.trackLanguage, [trackIndex]: lang },
  })),
  setTrackIndexRate: (trackIndex: number, rate: number) => set((state) => ({
    dirty: true, trackIndexRate: { ...state.trackIndexRate, [trackIndex]: rate },
  })),
  setTrackF0Offset: (trackIndex: number, offset: number) => set((state) => ({
    dirty: true, trackF0Offset: { ...state.trackF0Offset, [trackIndex]: offset },
  })),
  setMuteStrategy: (strategy) => set((state) => {
    const part = state.project.voice_parts[state.activePartIndex];
    let trackNo = part?.track_no ?? -1;
    if (trackNo < 0 && state.selectedTrackIndex >= 0) {
      const sel = state.project.tracks[state.selectedTrackIndex];
      if (sel?.track_type === 'vocal') trackNo = state.selectedTrackIndex;
    }
    if (trackNo < 0) {
      const idx = state.project.tracks.findIndex(t => t.track_type === 'vocal');
      if (idx >= 0) trackNo = idx;
    }
    return {
      dirty: true,
      muteStrategy: strategy,
      trackMuteStrategy: trackNo >= 0
        ? { ...state.trackMuteStrategy, [trackNo]: strategy }
        : state.trackMuteStrategy,
    };
  }),
  setTrackMuteStrategy: (trackIndex, strategy) => set((state) => ({
    dirty: true, trackMuteStrategy: { ...state.trackMuteStrategy, [trackIndex]: strategy },
  })),
  setGlobalF0Offset: (offset) => set({ globalF0Offset: offset }),
  togglePitchSnap: () => set((s) => ({ pitchSnap: !s.pitchSnap })),
  setAffineTransform: (enabled) => set((s) => {
    const newTrack: Record<number, boolean> = {};
    s.project.tracks.forEach((t, i) => {
      if (t.track_type === 'vocal') newTrack[i] = enabled;
    });
    return { affineTransform: enabled, trackAffineTransform: newTrack, dirty: true };
  }),
  setTrackAffineTransform: (trackIndex, enabled) => set((s) => {
    const next = { ...s.trackAffineTransform, [trackIndex]: enabled };
    const allOn = s.project.tracks.every((t, i) =>
      t.track_type !== 'vocal' || next[i] !== false);
    return { trackAffineTransform: next, affineTransform: enabled ? allOn : false, dirty: true };
  }),
  setCosineIndex: (enabled) => set({ cosineIndex: enabled, dirty: true }),
  setModelF0: (enabled) => set((s) => {
    const newTrack: Record<number, boolean> = {};
    s.project.tracks.forEach((t, i) => {
      if (t.track_type === 'vocal') newTrack[i] = enabled;
    });
    return { modelF0: enabled, trackModelF0: newTrack, dirty: true };
  }),
  setTrackModelF0: (trackIndex, enabled) => set((s) => {
    const next = { ...s.trackModelF0, [trackIndex]: enabled };
    const allOn = s.project.tracks.every((t, i) =>
      t.track_type !== 'vocal' || next[i] !== false);
    return { trackModelF0: next, modelF0: enabled ? allOn : false, dirty: true };
  }),
  toggleGhostTrack: (trackIndex) => set((s) => {
    const cur = s.ghostTracks;
    const next = cur.includes(trackIndex)
      ? cur.filter(i => i !== trackIndex)
      : [...cur, trackIndex];
    return { ghostTracks: next };
  }),
  rvcModelListVersion: 0,
  bumpRvcModelList: () => set((s) => ({ rvcModelListVersion: s.rvcModelListVersion + 1 })),

  // Render actions
  setRenderStatus: (status) => set({ renderStatus: status }),
  setRenderedUrl: (url) => set({ renderedUrl: url }),
  setFinalUrl: (url) => set({ finalUrl: url }),
  setRenderError: (error) => set({ renderError: error }),

  // Transport actions
  setViewMode: (mode) => set((state) => {
    if (mode === 'arrangement') {
      const ppt = Math.max(0.03, Math.min(2, state.viewState.pixelsPerTick));
      const pps = Math.max(8, Math.min(40, state.viewState.pixelsPerSemitone));
      return { viewMode: mode, viewState: { ...state.viewState, scrollX: 0, pixelsPerTick: ppt, pixelsPerSemitone: pps }, arrScrollY: 0, playbackPositionTicks: 0 };
    }
    return { viewMode: mode };
  }),
  setPlaying: (playing) => set({ isPlaying: playing }),
  setPlaybackPosition: (ticks) => set({ playbackPositionTicks: ticks }),
  toggleLoop: () => set((state) => ({ loopEnabled: !state.loopEnabled })),
  setLoopRange: (start, end) => set({ loopStart: start, loopEnd: end }),
  setTrackAudioUrl: (trackIndex, url) => set((state) => {
    const oldUrl = state.trackAudioUrls[trackIndex];
    if (oldUrl && oldUrl !== url && oldUrl !== state.finalUrl && oldUrl !== state.renderedUrl
        && oldUrl !== state.trackRawAudioUrls[trackIndex]) URL.revokeObjectURL(oldUrl);
    return { trackAudioUrls: { ...state.trackAudioUrls, [trackIndex]: url } };
  }),
  setTrackRawAudioUrl: (trackIndex: number, url: string) => set((state) => {
    const oldUrl = state.trackRawAudioUrls[trackIndex];
    if (oldUrl && oldUrl !== url) URL.revokeObjectURL(oldUrl);
    return { trackRawAudioUrls: { ...state.trackRawAudioUrls, [trackIndex]: url } };
  }),
  setPeakData: (bufferId, peaks) => set((state) => ({
    peakData: { ...state.peakData, [bufferId]: peaks },
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
        duration: RESOLUTION * 32,
        notes: [],
        curves: [],
      });
    } else if (type === 'instrument') {
      project.instrument_parts.push({
        name: `Inst ${idx + 1}`,
        track_no: idx,
        position: 0,
        duration: RESOLUTION * 32,
        notes: [],
      });
    }
    const wasAudio = state.project.tracks[state.selectedTrackIndex]?.track_type === 'audio';
    const nowAudio = type === 'audio';
    let playbackTicks = state.playbackPositionTicks;
    if (wasAudio && !nowAudio) {
      const ts = project.time_signatures[0];
      const ticksPerBar = RESOLUTION * (ts?.beat_per_bar ?? 4);
      playbackTicks = Math.round(playbackTicks / ticksPerBar) * ticksPerBar;
    }
    return { ...stacks, project, dirty: true, selectedTrackIndex: idx, selectedSegment: null, playbackPositionTicks: playbackTicks };
  }),

  renameTrack: (trackIndex, name) => set((state) => {
    const stacks = pushUndo(state);
    const project = cloneProject(state.project);
    if (project.tracks[trackIndex]) project.tracks[trackIndex].track_name = name;
    return { ...stacks, project, dirty: true };
  }),

  reorderTrack: (fromIndex, toIndex) => set((state) => {
    if (fromIndex === toIndex) return state;
    const stacks = pushUndo(state);
    const project = cloneProject(state.project);
    const [moved] = project.tracks.splice(fromIndex, 1);
    project.tracks.splice(toIndex, 0, moved);
    const remap = (old: number) => {
      if (old === fromIndex) return toIndex;
      if (fromIndex < toIndex) return old > fromIndex && old <= toIndex ? old - 1 : old;
      return old >= toIndex && old < fromIndex ? old + 1 : old;
    };
    for (const vp of project.voice_parts) vp.track_no = remap(vp.track_no);
    for (const wp of project.wave_parts) wp.track_no = remap(wp.track_no);
    for (const ip of project.instrument_parts) ip.track_no = remap(ip.track_no);
    // Remap trackAudioUrls + trackEffects
    const newUrls: Record<number, string> = {};
    for (const [k, v] of Object.entries(state.trackAudioUrls)) newUrls[remap(Number(k))] = v;
    const newFx: Record<number, import('../lib/synth/effects').TrackEffect[]> = {};
    for (const [k, v] of Object.entries(state.trackEffects)) newFx[remap(Number(k))] = v;
    return { ...stacks, project, dirty: true, selectedTrackIndex: toIndex, trackAudioUrls: newUrls, trackEffects: newFx };
  }),

  moveSegmentToTrack: (seg, newTrackNo) => set((state) => {
    if (newTrackNo < 0 || newTrackNo >= state.project.tracks.length) return state;
    const expectedType = seg.type === 'voice' ? 'vocal' : seg.type === 'wave' ? 'audio' : 'instrument';
    if ((state.project.tracks[newTrackNo]?.track_type || 'vocal') !== expectedType) return state;
    const stacks = pushUndo(state);
    const project = cloneProject(state.project);
    if (seg.type === 'voice') { const p = project.voice_parts[seg.index]; if (p) p.track_no = newTrackNo; }
    else if (seg.type === 'instrument') { const p = project.instrument_parts[seg.index]; if (p) p.track_no = newTrackNo; }
    else if (seg.type === 'wave') { const p = project.wave_parts[seg.index]; if (p) p.track_no = newTrackNo; }
    return { ...stacks, project, dirty: true };
  }),

  removeTrack: (trackIndex) => set((state) => {
    if (state.project.tracks.length <= 1) return state;
    const stacks = pushUndo(state);
    const project = cloneProject(state.project);
    // Collect buffer_ids of wave parts being deleted (for peakData cleanup)
    const deletedBufferIds = state.project.wave_parts
      .filter(wp => wp.track_no === trackIndex)
      .map(wp => wp.buffer_id)
      .filter(Boolean);
    project.tracks.splice(trackIndex, 1);
    project.voice_parts = project.voice_parts.filter(vp => vp.track_no !== trackIndex)
      .map(vp => ({ ...vp, track_no: vp.track_no > trackIndex ? vp.track_no - 1 : vp.track_no }));
    project.wave_parts = project.wave_parts.filter(wp => wp.track_no !== trackIndex)
      .map(wp => ({ ...wp, track_no: wp.track_no > trackIndex ? wp.track_no - 1 : wp.track_no }));
    project.instrument_parts = project.instrument_parts.filter(ip => ip.track_no !== trackIndex)
      .map(ip => ({ ...ip, track_no: ip.track_no > trackIndex ? ip.track_no - 1 : ip.track_no }));
    const newPartIdx = Math.min(state.activePartIndex, project.voice_parts.length - 1);
    const newSelectedTrack = Math.min(state.selectedTrackIndex, project.tracks.length - 1);
    // Clean up trackAudioUrls: remove deleted track, shift higher indices down
    const newTrackUrls: Record<number, string> = {};
    for (const [k, v] of Object.entries(state.trackAudioUrls)) {
      const idx = Number(k);
      if (idx === trackIndex) { if (v) URL.revokeObjectURL(v); continue; }
      const newIdx = idx > trackIndex ? idx - 1 : idx;
      newTrackUrls[newIdx] = v;
    }
    // Clean up trackEffects: remove deleted track, shift higher indices down
    const newTrackEffects: Record<number, import('../lib/synth/effects').TrackEffect[]> = {};
    for (const [k, v] of Object.entries(state.trackEffects)) {
      const idx = Number(k);
      if (idx === trackIndex) continue;
      const newIdx = idx > trackIndex ? idx - 1 : idx;
      newTrackEffects[newIdx] = v;
    }
    // Clean up peakData for deleted wave parts
    const newPeakData = { ...state.peakData };
    for (const bid of deletedBufferIds) if (bid) delete newPeakData[bid];
    const newInstIdx = state.activeInstrumentPartIndex >= project.instrument_parts.length ? -1 : state.activeInstrumentPartIndex;
    const newWaveIdx = state.activeWavePartIndex >= project.wave_parts.length ? -1 : state.activeWavePartIndex;
    return {
      ...stacks, project, dirty: true,
      activePartIndex: Math.max(0, newPartIdx),
      activeInstrumentPartIndex: newInstIdx,
      activeWavePartIndex: newWaveIdx,
      selectedTrackIndex: Math.max(0, newSelectedTrack),
      trackAudioUrls: newTrackUrls,
      trackEffects: newTrackEffects,
      peakData: newPeakData,
    };
  }),

  setTrackMute: (trackIndex, mute) => set((state) => {
    const stacks = pushUndo(state);
    const project = cloneProject(state.project);
    if (project.tracks[trackIndex]) project.tracks[trackIndex].mute = mute;
    return { ...stacks, project, dirty: true };
  }),

  setTrackSolo: (trackIndex, solo) => set((state) => {
    const stacks = pushUndo(state);
    const project = cloneProject(state.project);
    if (project.tracks[trackIndex]) project.tracks[trackIndex].solo = solo;
    return { ...stacks, project, dirty: true };
  }),

  setTrackVolume: (trackIndex, volume) => set((state) => {
    const project = cloneProject(state.project);
    if (project.tracks[trackIndex]) project.tracks[trackIndex].volume = volume;
    return { project, dirty: true };
  }),

  setTrackPan: (trackIndex, pan) => set((state) => {
    const project = cloneProject(state.project);
    if (project.tracks[trackIndex]) project.tracks[trackIndex].pan = pan;
    return { project, dirty: true };
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

  deleteWavePart: (partIndex) => set((state) => {
    const stacks = pushUndo(state);
    const project = cloneProject(state.project);
    project.wave_parts.splice(partIndex, 1);
    const newEffects = adjustEffectScopes({ ...state.trackEffects }, 'w', partIndex, 'delete');
    const newActive = state.activeWavePartIndex >= project.wave_parts.length ? -1 : state.activeWavePartIndex;
    return { ...stacks, project, dirty: true, activeWavePartIndex: newActive, selectedSegment: null, trackEffects: newEffects };
  }),

  // Instrument parts
  addInstrumentNote: (partIndex, note) => set((state) => {
    const stacks = pushUndo(state);
    const project = cloneProject(state.project);
    const part = project.instrument_parts[partIndex];
    if (part) {
      part.notes.push(note);
      part.notes.sort((a, b) => a.position - b.position);
      part.duration = Math.max(part.duration, note.position + note.duration + RESOLUTION * 4);
    }
    return { ...stacks, project, dirty: true };
  }),

  updateInstrumentNote: (partIndex, noteIndex, changes) => set((state) => {
    const stacks = pushUndo(state);
    const project = cloneProject(state.project);
    const part = project.instrument_parts[partIndex];
    if (part?.notes[noteIndex]) {
      Object.assign(part.notes[noteIndex], changes);
    }
    return { ...stacks, project, dirty: true };
  }),

  deleteInstrumentNotes: (noteIds) => set((state) => {
    const stacks = pushUndo(state);
    const project = cloneProject(state.project);
    const toDelete = new Set(noteIds);
    project.instrument_parts.forEach((part, pi) => {
      part.notes = part.notes.filter((_, ni) => !toDelete.has(`i${pi}-${ni}`));
    });
    return { ...stacks, project, dirty: true, selectedNotes: new Set<string>() };
  }),

  moveInstrumentNotes: (noteIds, deltaTick, deltaTone) => set((state) => {
    const stacks = pushUndo(state);
    const project = cloneProject(state.project);
    for (const id of noteIds) {
      const m = id.match(/^i(\d+)-(\d+)$/);
      if (!m) continue;
      const [, pi, ni] = m.map(Number);
      const note = project.instrument_parts[pi]?.notes[ni];
      if (note) {
        note.position = Math.max(0, note.position + deltaTick);
        note.tone = Math.max(0, Math.min(127, note.tone + deltaTone));
      }
    }
    return { ...stacks, project, dirty: true };
  }),

  resizeInstrumentNote: (partIndex, noteIndex, newDuration) => set((state) => {
    const stacks = pushUndo(state);
    const project = cloneProject(state.project);
    const note = project.instrument_parts[partIndex]?.notes[noteIndex];
    if (note) note.duration = Math.max(10, newDuration);
    return { ...stacks, project, dirty: true };
  }),

  setTrackPlugin: (trackIndex, pluginName) => set((state) => {
    const stacks = pushUndo(state);
    const project = cloneProject(state.project);
    if (project.tracks[trackIndex]) {
      project.tracks[trackIndex].plugin_name = pluginName;
    }
    return { ...stacks, project, dirty: true };
  }),

  // Effects
  addEffect: (trackIndex, effect) => set((state) => {
    const stacks = pushUndo(state);
    return {
      ...stacks, dirty: true,
      trackEffects: {
        ...state.trackEffects,
        [trackIndex]: [...(state.trackEffects[trackIndex] || []), effect],
      },
    };
  }),

  removeEffect: (trackIndex, effectIndex) => set((state) => {
    const stacks = pushUndo(state);
    const chain = [...(state.trackEffects[trackIndex] || [])];
    chain.splice(effectIndex, 1);
    return { ...stacks, dirty: true, trackEffects: { ...state.trackEffects, [trackIndex]: chain } };
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
    return { dirty: true, trackEffects: { ...state.trackEffects, [trackIndex]: chain } };
  }),

  // Render snapshots
  setTrackRenderedSnapshot: (trackIndex, snapshot) => set((state) => ({
    trackRenderedSnapshot: { ...state.trackRenderedSnapshot, [trackIndex]: snapshot },
  })),
  bumpVstStateVersion: (trackIndex) => set((state) => ({
    vstStateVersion: { ...state.vstStateVersion, [trackIndex]: (state.vstStateVersion[trackIndex] || 0) + 1 },
  })),

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

// Sync synth language when UI language changes — only update global default,
// never overwrite per-track languages (those are explicitly set by user or auto-detected)
i18n.on('languageChanged', (lng: string) => {
  const lang = lng.substring(0, 2) === 'zh' ? 'zh' as const : 'ja' as const;
  const s = useSynthStore.getState();
  if (Object.keys(s.trackLanguage).length === 0) {
    useSynthStore.setState({ language: lang });
  }
});
