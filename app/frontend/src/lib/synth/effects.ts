/** Built-in audio effects using Web Audio API.
 *  Each effect type maps to standard AudioNode chains.
 */

export type EffectType = 'gain' | 'eq3band' | 'reverb' | 'compressor';

export interface TrackEffect {
  type: EffectType;
  params: Record<string, number>;
  enabled: boolean;
}

export const EFFECT_DEFAULTS: Record<EffectType, Record<string, number>> = {
  gain: { gain: 0 },                    // dB
  eq3band: { low: 0, mid: 0, high: 0 }, // dB per band
  reverb: { mix: 0.3, decay: 2 },       // wet mix 0-1, decay seconds
  compressor: { threshold: -24, ratio: 4, attack: 0.003, release: 0.25 },
};

export const EFFECT_LABELS: Record<EffectType, string> = {
  gain: 'Gain',
  eq3band: '3-Band EQ',
  reverb: 'Reverb',
  compressor: 'Compressor',
};

/** Create a chain of AudioNodes for an effect. Returns [input, output] endpoints. */
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
    case 'compressor': {
      const comp = ctx.createDynamicsCompressor();
      comp.threshold.value = effect.params.threshold ?? -24;
      comp.ratio.value = effect.params.ratio ?? 4;
      comp.attack.value = effect.params.attack ?? 0.003;
      comp.release.value = effect.params.release ?? 0.25;
      return { input: comp, output: comp };
    }
    case 'reverb': {
      // Simple reverb using ConvolverNode with generated impulse
      const convolver = ctx.createConvolver();
      const mix = effect.params.mix ?? 0.3;
      const decay = effect.params.decay ?? 2;

      // Generate impulse response
      const sampleRate = ctx.sampleRate;
      const length = sampleRate * decay;
      const impulse = ctx.createBuffer(2, length, sampleRate);
      for (let ch = 0; ch < 2; ch++) {
        const data = impulse.getChannelData(ch);
        for (let i = 0; i < length; i++) {
          data[i] = (Math.random() * 2 - 1) * Math.pow(1 - i / length, 2);
        }
      }
      convolver.buffer = impulse;

      // Dry/wet mix
      const dry = ctx.createGain();
      dry.gain.value = 1 - mix;
      const wet = ctx.createGain();
      wet.gain.value = mix;
      const merger = ctx.createGain();

      // Input → dry → merger
      // Input → convolver → wet → merger
      const splitter = ctx.createGain(); // input node
      splitter.connect(dry);
      splitter.connect(convolver);
      convolver.connect(wet);
      dry.connect(merger);
      wet.connect(merger);

      return { input: splitter, output: merger };
    }
    default:
      return null;
  }
}
