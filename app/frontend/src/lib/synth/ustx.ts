/** USTX YAML parser and serializer.
 *  Handles reading/writing OpenUtau project files in the browser.
 */

import yaml from 'js-yaml';
import type {
  UProject, UVoicePart, UNote,
  UTrack, UExpressionDescriptor,
} from '../../types/ustx';
import { defaultPitch, defaultVibrato } from '../../types/ustx';

// --- Default expressions (matches Ustx.cs AddDefaultExpressions) ---
const DEFAULT_EXPRESSIONS: Record<string, UExpressionDescriptor> = {
  dyn: { name: 'dynamics (curve)', abbr: 'dyn', type: 2, min: -240, max: 120, default_value: 0 },
  pitd: { name: 'pitch deviation (curve)', abbr: 'pitd', type: 2, min: -1200, max: 1200, default_value: 0 },
  clr: { name: 'voice color', abbr: 'clr', type: 1, min: 0, max: 0, default_value: 0, options: [''] },
  eng: { name: 'resampler engine', abbr: 'eng', type: 1, min: 0, max: 0, default_value: 0, options: [''] },
  vel: { name: 'velocity', abbr: 'vel', type: 0, min: 0, max: 200, default_value: 100 },
  vol: { name: 'volume', abbr: 'vol', type: 0, min: 0, max: 200, default_value: 100 },
  atk: { name: 'attack', abbr: 'atk', type: 0, min: 0, max: 200, default_value: 100 },
  dec: { name: 'decay', abbr: 'dec', type: 0, min: 0, max: 100, default_value: 0 },
  gen: { name: 'gender', abbr: 'gen', type: 0, min: -100, max: 100, default_value: 0 },
  bre: { name: 'breath', abbr: 'bre', type: 0, min: 0, max: 100, default_value: 0 },
};

/** Parse a USTX YAML string into a UProject. */
export function parseUstx(yamlStr: string): UProject {
  const raw = yaml.load(yamlStr) as Record<string, any>;
  if (!raw || typeof raw !== 'object') {
    throw new Error('Invalid USTX: not a YAML object');
  }

  const project: UProject = {
    name: raw.name ?? 'Untitled',
    comment: raw.comment ?? '',
    output_dir: raw.output_dir ?? 'Vocal',
    cache_dir: raw.cache_dir ?? 'UCache',
    ustx_version: String(raw.ustx_version ?? '0.9'),
    key: raw.key ?? 0,
    expressions: raw.expressions ?? { ...DEFAULT_EXPRESSIONS },
    exp_selectors: raw.exp_selectors ?? Object.keys(DEFAULT_EXPRESSIONS),
    exp_primary: raw.exp_primary ?? 0,
    exp_secondary: raw.exp_secondary ?? 1,
    time_signatures: raw.time_signatures ?? [{ bar_position: 0, beat_per_bar: 4, beat_unit: 4 }],
    tempos: raw.tempos ?? [{ position: 0, bpm: 120 }],
    tracks: (raw.tracks ?? []).map(parseTrack),
    voice_parts: (raw.voice_parts ?? []).map(parseVoicePart),
    wave_parts: raw.wave_parts ?? [],
    instrument_parts: raw.instrument_parts ?? [],
  };

  // Version migration: pre-0.6 had scalar bpm/beatPerBar/beatUnit
  if (raw.bpm && project.tempos.length === 0) {
    project.tempos = [{ position: 0, bpm: raw.bpm }];
  }
  if (raw.beat_per_bar && project.time_signatures.length === 0) {
    project.time_signatures = [{
      bar_position: 0,
      beat_per_bar: raw.beat_per_bar,
      beat_unit: raw.beat_unit ?? 4,
    }];
  }

  return project;
}

function parseTrack(raw: any): UTrack {
  return {
    track_name: raw.track_name ?? 'Track1',
    track_color: raw.track_color ?? 'Blue',
    track_type: raw.track_type ?? 'vocal',
    singer: raw.singer ?? undefined,
    phonemizer: raw.phonemizer ?? undefined,
    mute: raw.mute ?? false,
    solo: raw.solo ?? false,
    volume: raw.volume ?? 0,
    pan: raw.pan ?? 0,
    voice_color_names: raw.voice_color_names ?? [''],
  };
}

const SUSTAIN_LYRICS = new Set(['+', '+~', '++']);

function parseVoicePart(raw: any): UVoicePart {
  const allNotes: any[] = raw.notes ?? [];
  const parsed = allNotes.map(parseNote);
  let prevLyric = 'a';
  for (const n of parsed) {
    if (SUSTAIN_LYRICS.has(n.lyric)) {
      n.lyric = prevLyric;
    } else {
      prevLyric = n.lyric;
    }
  }
  return {
    name: raw.name ?? 'Part1',
    comment: raw.comment ?? '',
    track_no: raw.track_no ?? 0,
    position: raw.position ?? 0,
    duration: raw.duration ?? 1920,
    notes: parsed,
    curves: raw.curves ?? [],
  };
}

function parseNote(raw: any): UNote {
  return {
    position: raw.position ?? 0,
    duration: raw.duration ?? 480,
    tone: raw.tone ?? 60,
    lyric: raw.lyric ?? 'a',
    pitch: raw.pitch ?? defaultPitch(),
    vibrato: raw.vibrato ?? defaultVibrato(),
    tuning: raw.tuning ?? 0,
    phoneme_expressions: raw.phoneme_expressions ?? [],
    phoneme_overrides: raw.phoneme_overrides ?? [],
  };
}

/** Serialize a UProject to USTX YAML string. */
export function serializeUstx(project: UProject): string {
  const obj: Record<string, any> = {
    name: project.name,
    comment: project.comment,
    output_dir: project.output_dir,
    cache_dir: project.cache_dir,
    ustx_version: project.ustx_version,
    resolution: 480,
    key: project.key,
    expressions: project.expressions,
    exp_selectors: project.exp_selectors,
    exp_primary: project.exp_primary,
    exp_secondary: project.exp_secondary,
    time_signatures: project.time_signatures,
    tempos: project.tempos,
    tracks: project.tracks,
    voice_parts: project.voice_parts.map(serializeVoicePart),
    wave_parts: project.wave_parts,
  };

  return yaml.dump(obj, {
    lineWidth: -1,
    noRefs: true,
    sortKeys: false,
  });
}

function serializeVoicePart(part: UVoicePart): any {
  return {
    ...part,
    notes: part.notes.map(n => ({ ...n })).sort((a, b) => a.position - b.position),
  };
}

/** Create a new default project. */
export function createDefaultProject(): UProject {
  return {
    name: 'New Project',
    comment: '',
    output_dir: 'Vocal',
    cache_dir: 'UCache',
    ustx_version: '0.9',
    key: 0,
    expressions: { ...DEFAULT_EXPRESSIONS },
    exp_selectors: Object.keys(DEFAULT_EXPRESSIONS),
    exp_primary: 0,
    exp_secondary: 1,
    time_signatures: [{ bar_position: 0, beat_per_bar: 4, beat_unit: 4 }],
    tempos: [{ position: 0, bpm: 120 }],
    tracks: [{
      track_name: 'Track1',
      track_color: 'Blue',
      track_type: 'vocal',
      mute: false,
      solo: false,
      volume: 0,
      pan: 0,
      voice_color_names: [''],
    }],
    voice_parts: [{
      name: 'Part1',
      comment: '',
      track_no: 0,
      position: 0,
      duration: 480 * 32,
      notes: [],
      curves: [],
    }],
    wave_parts: [],
    instrument_parts: [],
  };
}
