/** USTX format TypeScript interfaces.
 *  Mirrors OpenUtau.Core/Ustx/ data model.
 *  YAML uses snake_case keys (YamlDotNet UnderscoredNamingConvention).
 */

export const RESOLUTION = 480; // ticks per quarter note (constant)

export type PitchPointShape = 'io' | 'l' | 'i' | 'o' | 'sp';

export interface PitchPoint {
  x: number;  // ms relative to note start
  y: number;  // 0.1 semitones
  shape: PitchPointShape;
}

export interface UPitch {
  snap_first: boolean;
  data: PitchPoint[];
}

export interface UVibrato {
  length: number;   // % of note length (0-100)
  period: number;   // cycle ms (5-500)
  depth: number;    // cents (5-200)
  in: number;       // fade-in % of vibrato
  out: number;      // fade-out %
  shift: number;    // phase shift % (0-100)
  drift: number;    // pitch shift (-100 to 100)
  vol_link: number; // volume modulation (-100 to 100)
}

export interface UExpression {
  index?: number;
  abbr: string;
  value: number;
}

export interface UPhonemeOverride {
  index?: number;
  phoneme?: string;
  offset?: number;
}

export interface UNote {
  position: number;    // ticks relative to part start
  duration: number;    // ticks (min 10)
  tone: number;        // MIDI note 0-127
  lyric: string;
  pitch: UPitch;
  vibrato: UVibrato;
  tuning: number;      // cents
  phoneme_expressions: UExpression[];
  phoneme_overrides: UPhonemeOverride[];
}

export interface UCurve {
  abbr: string;
  xs: number[];
  ys: number[];
}

export interface UVoicePart {
  name: string;
  comment: string;
  track_no: number;
  position: number;   // ticks from project start
  duration: number;   // ticks
  notes: UNote[];
  curves: UCurve[];
}

export interface UWavePart {
  name: string;
  track_no: number;
  position: number;
  relative_path: string;
  file_duration_ms: number;
  skip: number;
  trim: number;
  fadein: number;
  fadeout: number;
  buffer_id?: string;  // backend audio buffer reference for imported audio
}

export interface UTempo {
  position: number;   // ticks
  bpm: number;
}

export interface UTimeSignature {
  bar_position: number;
  beat_per_bar: number;
  beat_unit: number;
}

export interface UExpressionDescriptor {
  name: string;
  abbr: string;
  type: number;       // 0=Numerical, 1=Options, 2=Curve
  min: number;
  max: number;
  default_value: number;
  flag?: string;
  options?: string[];
}

export interface UTrack {
  track_name: string;
  track_color: string;
  track_type?: 'vocal' | 'audio';  // defaults to 'vocal' for backward compat
  singer?: string;
  phonemizer?: string;
  mute: boolean;
  solo: boolean;
  volume: number;
  pan: number;
  voice_color_names?: string[];
}

export interface UProject {
  name: string;
  comment: string;
  output_dir: string;
  cache_dir: string;
  ustx_version: string;
  resolution?: number;  // always 480, sometimes omitted
  key: number;
  expressions: Record<string, UExpressionDescriptor>;
  exp_selectors: string[];
  exp_primary: number;
  exp_secondary: number;
  time_signatures: UTimeSignature[];
  tempos: UTempo[];
  tracks: UTrack[];
  voice_parts: UVoicePart[];
  wave_parts: UWavePart[];
  [key: string]: unknown;
}

// Helper: MIDI note number to note name
export function midiToName(midi: number): string {
  const names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'];
  const octave = Math.floor(midi / 12) - 1;
  return `${names[midi % 12]}${octave}`;
}

// Helper: is a MIDI note a black key?
export function isBlackKey(midi: number): boolean {
  const n = midi % 12;
  return n === 1 || n === 3 || n === 6 || n === 8 || n === 10;
}

// Default vibrato for new notes
export function defaultVibrato(): UVibrato {
  return { length: 0, period: 175, depth: 35, in: 10, out: 10, shift: 0, drift: 0, vol_link: 0 };
}

// Default pitch for new notes (must have at least 2 points when snap_first is true)
export function defaultPitch(): UPitch {
  return {
    snap_first: true,
    data: [
      { x: -40, y: 0, shape: 'io' as PitchPointShape },
      { x: 40, y: 0, shape: 'io' as PitchPointShape },
    ],
  };
}
