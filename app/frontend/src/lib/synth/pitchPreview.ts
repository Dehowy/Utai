/** Simple pitch preview using Web Audio API oscillator.
 *  Plays a short sine tone at the given MIDI note frequency.
 *  Uses the shared AudioContext from audioEngine.
 */

import { getAudioContext, resumeContext } from './audioEngine';

let activeOsc: OscillatorNode | null = null;
let activeGain: GainNode | null = null;

function midiToFreq(midi: number): number {
  return 440 * Math.pow(2, (midi - 69) / 12);
}

export function playNote(midi: number, duration?: number) {
  const ctx = getAudioContext();
  resumeContext();
  stopNote();

  const osc = ctx.createOscillator();
  const gain = ctx.createGain();

  osc.type = 'triangle';
  osc.frequency.value = midiToFreq(midi);

  const vol = Math.max(0.35, Math.min(0.55, 0.45 + (60 - midi) * 0.004));
  const dur = duration ?? Math.max(0.15, Math.min(0.4, 0.15 + (60 - midi) * 0.005));

  gain.gain.setValueAtTime(0, ctx.currentTime);
  gain.gain.linearRampToValueAtTime(vol, ctx.currentTime + 0.01);
  gain.gain.linearRampToValueAtTime(vol * 0.6, ctx.currentTime + dur * 0.7);
  gain.gain.linearRampToValueAtTime(0, ctx.currentTime + dur);

  osc.connect(gain);
  gain.connect(ctx.destination);

  osc.start(ctx.currentTime);
  osc.stop(ctx.currentTime + dur + 0.05);

  activeOsc = osc;
  activeGain = gain;

  osc.onended = () => {
    if (activeOsc === osc) {
      activeOsc = null;
      activeGain = null;
    }
  };
}

export function stopNote() {
  if (activeOsc && activeGain) {
    try {
      const ctx = getAudioContext();
      activeGain.gain.cancelScheduledValues(ctx.currentTime);
      activeGain.gain.linearRampToValueAtTime(0, ctx.currentTime + 0.02);
      activeOsc.stop(ctx.currentTime + 0.03);
    } catch {
      // Already stopped
    }
    activeOsc = null;
    activeGain = null;
  }
}
