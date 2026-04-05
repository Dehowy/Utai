/** Simple pitch preview using Web Audio API oscillator.
 *  Plays a short sine tone at the given MIDI note frequency.
 */

let audioCtx: AudioContext | null = null;
let activeOsc: OscillatorNode | null = null;
let activeGain: GainNode | null = null;

function getAudioContext(): AudioContext {
  if (!audioCtx) {
    audioCtx = new AudioContext();
  }
  return audioCtx;
}

function midiToFreq(midi: number): number {
  return 440 * Math.pow(2, (midi - 69) / 12);
}

export function playNote(midi: number, duration = 0.15) {
  const ctx = getAudioContext();
  stopNote();

  const osc = ctx.createOscillator();
  const gain = ctx.createGain();

  osc.type = 'sine';
  osc.frequency.value = midiToFreq(midi);

  gain.gain.setValueAtTime(0, ctx.currentTime);
  gain.gain.linearRampToValueAtTime(0.3, ctx.currentTime + 0.01);
  gain.gain.linearRampToValueAtTime(0, ctx.currentTime + duration);

  osc.connect(gain);
  gain.connect(ctx.destination);

  osc.start(ctx.currentTime);
  osc.stop(ctx.currentTime + duration + 0.05);

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
