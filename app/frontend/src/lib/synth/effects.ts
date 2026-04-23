/** Audio effects — two layers:
 *  1. Web Audio API effects for real-time playback (gain, eq3band)
 *  2. dawdreamer/WORLD effects for offline rendering (pitch, reverb, compressor, etc.)
 */

// ── Render-time effects (dawdreamer + WORLD) ──

export type RenderEffectType = 'world_pitch' | 'warp_pitch' | 'compressor' | 'reverb' | 'filter' | 'delay' | 'panner';

export interface TrackEffect {
  type: EffectType | RenderEffectType;
  params: Record<string, number>;
  enabled: boolean;
  scope?: 'track' | number | string;  // 'track' = whole track, number = voice part index, 'w0'/'w1' = wave part
}

// All render effects with their parameter definitions
export const RENDER_EFFECTS: Record<RenderEffectType, {
  label: string;
  category: string;
  params: Record<string, { default: number; min: number; max: number; step: number; label: string }>;
}> = {
  world_pitch: {
    label: 'WORLD Pitch/Speed',
    category: 'pitch',
    params: {
      semitones: { default: 0, min: -24, max: 24, step: 1, label: 'Semitones' },
      formant_shift: { default: 1.0, min: 0.5, max: 2.0, step: 0.05, label: 'Formant' },
      speed: { default: 1.0, min: 0.25, max: 4.0, step: 0.05, label: 'Speed' },
    },
  },
  warp_pitch: {
    label: 'Pitch / Speed',
    category: 'pitch',
    params: {
      semitones: { default: 0, min: -24, max: 24, step: 1, label: 'Semitones' },
      speed: { default: 1.0, min: 0.25, max: 4.0, step: 0.05, label: 'Speed' },
    },
  },
  compressor: {
    label: 'Compressor',
    category: 'dynamics',
    params: {
      threshold: { default: -10, min: -60, max: 0, step: 1, label: 'Threshold' },
      ratio: { default: 4, min: 1, max: 20, step: 0.5, label: 'Ratio' },
      attack: { default: 5, min: 0.1, max: 100, step: 0.1, label: 'Attack' },
      release: { default: 100, min: 5, max: 2000, step: 5, label: 'Release' },
    },
  },
  reverb: {
    label: 'Reverb',
    category: 'space',
    params: {
      room_size: { default: 0.5, min: 0, max: 1, step: 0.01, label: 'Room' },
      damping: { default: 0.5, min: 0, max: 1, step: 0.01, label: 'Damp' },
      wet: { default: 0.3, min: 0, max: 1, step: 0.01, label: 'Wet' },
      dry: { default: 0.7, min: 0, max: 1, step: 0.01, label: 'Dry' },
      width: { default: 1.0, min: 0, max: 1, step: 0.01, label: 'Width' },
    },
  },
  filter: {
    label: 'Filter',
    category: 'eq',
    params: {
      mode: { default: 0, min: 0, max: 2, step: 1, label: 'Mode' },
      frequency: { default: 1000, min: 20, max: 20000, step: 10, label: 'Freq' },
      q: { default: 0.707, min: 0.1, max: 10, step: 0.01, label: 'Q' },
      gain: { default: 1.0, min: 0, max: 4, step: 0.1, label: 'Gain' },
    },
  },
  delay: {
    label: 'Delay',
    category: 'space',
    params: {
      time_ms: { default: 250, min: 1, max: 2000, step: 1, label: 'Time' },
      wet: { default: 0.3, min: 0, max: 1, step: 0.01, label: 'Wet' },
    },
  },
  panner: {
    label: 'Panner',
    category: 'spatial',
    params: {
      pan: { default: 0, min: -1, max: 1, step: 0.01, label: 'Pan' },
    },
  },
};

// ── Playback-time effects (Web Audio API) ──

export type EffectType = 'gain' | 'eq3band';

export const EFFECT_DEFAULTS: Record<EffectType, Record<string, number>> = {
  gain: { gain: 0 },
  eq3band: { low: 0, mid: 0, high: 0 },
};

export const EFFECT_LABELS: Record<EffectType, string> = {
  gain: 'Gain',
  eq3band: '3-Band EQ',
};

/** Create a chain of AudioNodes for real-time playback effects. */
export function createEffectNodes(
  ctx: AudioContext,
  effect: TrackEffect,
): { input: AudioNode; output: AudioNode } | null {
  if (!effect.enabled) return null;

  switch (effect.type) {
    case 'gain': {
      const gain = ctx.createGain();
      gain.gain.value = Math.pow(10, (effect.params.gain || 0) / 20);
      return { input: gain, output: gain };
    }
    case 'eq3band': {
      const low = ctx.createBiquadFilter();
      low.type = 'lowshelf';
      low.frequency.value = 320;
      low.gain.value = effect.params.low || 0;
      const mid = ctx.createBiquadFilter();
      mid.type = 'peaking';
      mid.frequency.value = 1000;
      mid.Q.value = 0.5;
      mid.gain.value = effect.params.mid || 0;
      const high = ctx.createBiquadFilter();
      high.type = 'highshelf';
      high.frequency.value = 3200;
      high.gain.value = effect.params.high || 0;
      low.connect(mid);
      mid.connect(high);
      return { input: low, output: high };
    }
    default:
      return null;
  }
}

/** Create default params for a render effect type. */
export function createDefaultEffect(type: RenderEffectType, scope: 'track' | number | string = 'track'): TrackEffect {
  const def = RENDER_EFFECTS[type];
  const params: Record<string, number> = {};
  for (const [k, v] of Object.entries(def.params)) {
    params[k] = v.default;
  }
  return { type, params, enabled: true, scope };
}
