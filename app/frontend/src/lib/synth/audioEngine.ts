/** Shared Web Audio engine for DAW playback.
 *  Singleton AudioContext + multi-track mixer with per-track gain/pan.
 *  Provides play/pause/stop/seek and 60fps position updates.
 */

import type { UTempo } from '../../types/ustx';
import { RESOLUTION } from '../../types/ustx';

// ── Shared AudioContext ──

let _ctx: AudioContext | null = null;

export function getAudioContext(): AudioContext {
  if (!_ctx) _ctx = new AudioContext();
  return _ctx;
}

export function resumeContext() {
  const ctx = getAudioContext();
  if (ctx.state === 'suspended') ctx.resume();
}

// ── Tempo utilities ──

/** Convert tick position to seconds using the tempo map. */
export function ticksToSeconds(ticks: number, tempos: UTempo[]): number {
  if (tempos.length === 0) return ticks / RESOLUTION * 0.5; // fallback 120 BPM
  let seconds = 0;
  let prevTick = 0;
  let bpm = tempos[0].bpm;
  for (const t of tempos) {
    if (t.position >= ticks) break;
    seconds += (Math.min(ticks, t.position) - prevTick) / RESOLUTION * (60 / bpm);
    prevTick = t.position;
    bpm = t.bpm;
  }
  seconds += (ticks - prevTick) / RESOLUTION * (60 / bpm);
  return seconds;
}

/** Convert seconds to tick position using the tempo map. */
export function secondsToTicks(seconds: number, tempos: UTempo[]): number {
  if (tempos.length === 0) return seconds * 2 * RESOLUTION; // fallback 120 BPM
  let remainingSec = seconds;
  let prevTick = 0;
  let bpm = tempos[0].bpm;
  for (let i = 1; i < tempos.length; i++) {
    const segTicks = tempos[i].position - prevTick;
    const segSec = segTicks / RESOLUTION * (60 / bpm);
    if (remainingSec <= segSec) break;
    remainingSec -= segSec;
    prevTick = tempos[i].position;
    bpm = tempos[i].bpm;
  }
  return prevTick + remainingSec / (60 / bpm) * RESOLUTION;
}

// ── Audio buffer loading ──

const _bufferCache = new Map<string, AudioBuffer>();

export async function loadAudioBuffer(url: string): Promise<AudioBuffer> {
  const cached = _bufferCache.get(url);
  if (cached) return cached;
  const ctx = getAudioContext();
  const res = await fetch(url);
  const arrayBuf = await res.arrayBuffer();
  const audioBuf = await ctx.decodeAudioData(arrayBuf);
  _bufferCache.set(url, audioBuf);
  return audioBuf;
}

export function clearBufferCache() {
  _bufferCache.clear();
}

// ── Track mixer node ──

export interface TrackNode {
  gain: GainNode;
  pan: StereoPannerNode;
  source?: AudioBufferSourceNode;
}

// ── Transport state ──

let _playing = false;
let _startCtxTime = 0;    // audioCtx.currentTime when playback started
let _startOffsetSec = 0;  // seconds offset into the project at start
let _activeSources: AudioBufferSourceNode[] = [];
let _masterGain: GainNode | null = null;
let _rafId = 0;
let _positionCallback: ((seconds: number) => void) | null = null;

export function onPositionUpdate(cb: (seconds: number) => void) {
  _positionCallback = cb;
}

function _tick() {
  if (!_playing) return;
  const ctx = getAudioContext();
  const elapsed = ctx.currentTime - _startCtxTime;
  const posSec = _startOffsetSec + elapsed;
  if (_positionCallback) _positionCallback(posSec);
  _rafId = requestAnimationFrame(_tick);
}

export function isPlaying(): boolean {
  return _playing;
}

export function getCurrentSeconds(): number {
  if (!_playing) return _startOffsetSec;
  const ctx = getAudioContext();
  return _startOffsetSec + (ctx.currentTime - _startCtxTime);
}

// ── Playback control ──

export interface PlayTrackConfig {
  url: string;          // blob URL or backend URL
  startOffsetSec: number; // when this track starts playing (from project start)
  volume: number;       // dB
  pan: number;          // -1 to 1
  muted: boolean;
}

function dbToLinear(db: number): number {
  return Math.pow(10, db / 20);
}

export async function play(
  tracks: PlayTrackConfig[],
  fromSeconds: number = 0,
) {
  stop();
  const ctx = getAudioContext();
  resumeContext();

  _masterGain = ctx.createGain();
  _masterGain.gain.value = 1;
  _masterGain.connect(ctx.destination);

  _startCtxTime = ctx.currentTime;
  _startOffsetSec = fromSeconds;
  _playing = true;

  for (const track of tracks) {
    if (track.muted) continue;
    try {
      const buf = await loadAudioBuffer(track.url);
      const source = ctx.createBufferSource();
      source.buffer = buf;

      const gain = ctx.createGain();
      gain.gain.value = dbToLinear(track.volume);

      const pan = ctx.createStereoPanner();
      pan.pan.value = Math.max(-1, Math.min(1, track.pan / 100));

      source.connect(gain);
      gain.connect(pan);
      pan.connect(_masterGain!);

      // Calculate offset into the track's audio
      const trackOffsetInProject = track.startOffsetSec;
      const playFrom = fromSeconds - trackOffsetInProject;

      if (playFrom >= 0 && playFrom < buf.duration) {
        source.start(0, playFrom);
      } else if (playFrom < 0) {
        // Track starts later — schedule with delay
        source.start(ctx.currentTime + (-playFrom), 0);
      }
      // else: playFrom >= buf.duration → track already ended, skip

      _activeSources.push(source);
      source.onended = () => {
        const idx = _activeSources.indexOf(source);
        if (idx >= 0) _activeSources.splice(idx, 1);
      };
    } catch {
      // Skip tracks that fail to load
    }
  }

  _rafId = requestAnimationFrame(_tick);
}

export function pause() {
  if (!_playing) return;
  const ctx = getAudioContext();
  _startOffsetSec += ctx.currentTime - _startCtxTime;
  _playing = false;
  cancelAnimationFrame(_rafId);
  for (const src of _activeSources) {
    try { src.stop(); } catch { /* already stopped */ }
  }
  _activeSources = [];
  if (_masterGain) {
    _masterGain.disconnect();
    _masterGain = null;
  }
}

export function stop() {
  _startOffsetSec = 0;
  if (_playing) {
    _playing = false;
    cancelAnimationFrame(_rafId);
    for (const src of _activeSources) {
      try { src.stop(); } catch { /* already stopped */ }
    }
    _activeSources = [];
  }
  if (_masterGain) {
    _masterGain.disconnect();
    _masterGain = null;
  }
  if (_positionCallback) _positionCallback(0);
}

export function seek(seconds: number) {
  _startOffsetSec = seconds;
  if (!_playing && _positionCallback) _positionCallback(seconds);
}
