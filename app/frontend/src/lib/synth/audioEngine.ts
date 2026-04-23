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
const MAX_BUFFER_CACHE = 64;

export async function loadAudioBuffer(url: string): Promise<AudioBuffer> {
  const cached = _bufferCache.get(url);
  if (cached) return cached;
  const ctx = getAudioContext();
  const res = await fetch(url);
  const arrayBuf = await res.arrayBuffer();
  const audioBuf = await ctx.decodeAudioData(arrayBuf);
  if (_bufferCache.size >= MAX_BUFFER_CACHE) {
    const oldest = _bufferCache.keys().next().value;
    if (oldest !== undefined) _bufferCache.delete(oldest);
  }
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
let _lastPlayTracks: PlayTrackConfig[] = [];

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
  skipSec?: number;     // offset into the audio buffer (for split wave parts)
  durationSec?: number; // max playback duration (for split wave parts)
}

function dbToLinear(db: number): number {
  if (db <= -24) return 0;
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

  _startOffsetSec = fromSeconds;
  _lastPlayTracks = tracks;

  // Pre-load all buffers before starting playback (avoids timing drift)
  const loaded: { buf: AudioBuffer; track: PlayTrackConfig }[] = [];
  for (const track of tracks) {
    if (track.muted) continue;
    try {
      const buf = await loadAudioBuffer(track.url);
      loaded.push({ buf, track });
    } catch { /* skip */ }
  }

  // Set timing anchor and playing flag AFTER all loading is done
  _startCtxTime = ctx.currentTime;
  _playing = true;

  for (const { buf, track } of loaded) {
    const source = ctx.createBufferSource();
    source.buffer = buf;

    const gain = ctx.createGain();
    gain.gain.value = dbToLinear(track.volume);

    const pan = ctx.createStereoPanner();
    pan.pan.value = Math.max(-1, Math.min(1, track.pan / 100));

    source.connect(gain);
    gain.connect(pan);
    pan.connect(_masterGain!);

    const skip = Math.max(0, Math.min(track.skipSec ?? 0, buf.duration));
    const clipDur = Math.max(0, Math.min(track.durationSec ?? (buf.duration - skip), buf.duration - skip));
    if (clipDur <= 0) continue;
    const playFrom = fromSeconds - track.startOffsetSec;

    if (playFrom >= 0) {
      const bufOffset = skip + playFrom;
      const remaining = clipDur - playFrom;
      if (bufOffset >= buf.duration || remaining <= 0) continue;
      source.start(0, bufOffset, remaining);
    } else {
      const delay = -playFrom;
      if (delay >= clipDur) continue;
      source.start(ctx.currentTime + delay, skip, clipDur);
    }

    _activeSources.push(source);
    source.onended = () => {
      const idx = _activeSources.indexOf(source);
      if (idx >= 0) _activeSources.splice(idx, 1);
    };
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
  // Don't reset _startOffsetSec here — play() and seek() manage it
}

let _seekDebounce: ReturnType<typeof setTimeout> | null = null;

export function seek(seconds: number) {
  _startOffsetSec = seconds;
  if (_playing) {
    const ctx = getAudioContext();
    _startCtxTime = ctx.currentTime;
    // Debounced restart — only restart audio after dragging settles
    if (_lastPlayTracks.length > 0) {
      if (_seekDebounce) clearTimeout(_seekDebounce);
      _seekDebounce = setTimeout(() => {
        _seekDebounce = null;
        if (_playing) play(_lastPlayTracks, _startOffsetSec);
      }, 150);
    }
  }
  if (_positionCallback) _positionCallback(seconds);
}
