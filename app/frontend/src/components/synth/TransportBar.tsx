/** Transport bar: play/pause, position display, loop toggle, reset, waveform strip. */

import { useEffect, useRef, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { useSynthStore } from '../../store/synthStore';
import { RESOLUTION } from '../../types/ustx';
import * as engine from '../../lib/synth/audioEngine';
import {
  getActiveTrackIndex,
  trackNeedsRender,
  getDirtyTrackIndices,
  renderTracks,
  createRenderAbort,
} from '../../lib/synth/renderEngine';
import WaveSurfer from 'wavesurfer.js';

function getLoopRange(): { startTick: number; endTick: number } {
  const s = useSynthStore.getState();
  const bpm = s.project.tempos[0]?.bpm ?? 120;
  const tps = bpm / 60 * RESOLUTION;

  // Audio track mode: range = all wave parts on this track
  if (s.activeWavePartIndex >= 0) {
    const wp0 = s.project.wave_parts[s.activeWavePartIndex];
    if (wp0) {
      let endTick = 0;
      for (const wp of s.project.wave_parts) {
        if (wp.track_no === wp0.track_no) {
          endTick = Math.max(endTick, wp.position + Math.round((wp.file_duration_ms / 1000) * tps));
        }
      }
      return { startTick: 0, endTick: endTick || Infinity };
    }
  }

  if (s.viewMode === 'arrangement') {
    let endTick = 0;
    for (const vp of s.project.voice_parts) {
      for (const n of vp.notes) {
        endTick = Math.max(endTick, vp.position + n.position + n.duration);
      }
    }
    for (const ip of s.project.instrument_parts) {
      for (const n of ip.notes) {
        endTick = Math.max(endTick, ip.position + n.position + n.duration);
      }
    }
    for (const wp of s.project.wave_parts) {
      endTick = Math.max(endTick, wp.position + Math.round((wp.file_duration_ms / 1000) * tps));
    }
    return { startTick: 0, endTick: endTick || Infinity };
  } else {
    // Pianoroll mode — scope to active part
    if (s.activeInstrumentPartIndex >= 0) {
      const part = s.project.instrument_parts[s.activeInstrumentPartIndex];
      if (!part) return { startTick: 0, endTick: RESOLUTION * 4 };
      let lastNoteEnd = part.position;
      for (const n of part.notes) {
        lastNoteEnd = Math.max(lastNoteEnd, part.position + n.position + n.duration);
      }
      return { startTick: part.position, endTick: lastNoteEnd || part.position + part.duration };
    }
    const part = s.project.voice_parts[s.activePartIndex];
    if (!part) return { startTick: 0, endTick: RESOLUTION * 4 };
    const hasNext = s.project.voice_parts.some(
      vp => vp.track_no === part.track_no && vp.position > part.position
    );
    if (hasNext) {
      return { startTick: part.position, endTick: part.position + part.duration };
    }
    let lastNoteEnd = part.position;
    for (const n of part.notes) {
      lastNoteEnd = Math.max(lastNoteEnd, part.position + n.position + n.duration);
    }
    return { startTick: part.position, endTick: lastNoteEnd || part.position + part.duration };
  }
}

export default function TransportBar() {
  const { t } = useTranslation();
  const project = useSynthStore(s => s.project);
  const isPlaying = useSynthStore(s => s.isPlaying);
  const setPlaying = useSynthStore(s => s.setPlaying);
  const playbackPositionTicks = useSynthStore(s => s.playbackPositionTicks);
  const setPlaybackPosition = useSynthStore(s => s.setPlaybackPosition);
  const loopEnabled = useSynthStore(s => s.loopEnabled);
  const toggleLoop = useSynthStore(s => s.toggleLoop);
  const renderedUrl = useSynthStore(s => s.renderedUrl);
  const finalUrl = useSynthStore(s => s.finalUrl);
  const trackAudioUrls = useSynthStore(s => s.trackAudioUrls);
  const waveFxSnapshot = useSynthStore(s => Object.keys(s.waveFxUrls).join(','));
  const renderError = useSynthStore(s => s.renderError);
  const renderStatus = useSynthStore(s => s.renderStatus);

  const waveRef = useRef<HTMLDivElement>(null);
  const wsRef = useRef<WaveSurfer | null>(null);

  const bpm = project.tempos[0]?.bpm ?? 120;
  const ts = project.time_signatures[0];
  const beatsPerBar = ts?.beat_per_bar ?? 4;

  const totalBeats = playbackPositionTicks / RESOLUTION;
  const bar = Math.floor(totalBeats / beatsPerBar) + 1;
  const beat = Math.floor(totalBeats % beatsPerBar) + 1;
  const tick = Math.floor(playbackPositionTicks % RESOLUTION);
  const posStr = `${bar}:${beat}:${String(tick).padStart(3, '0')}`;

  const posSec = engine.ticksToSeconds(playbackPositionTicks, project.tempos);
  const mm = Math.floor(posSec / 60);
  const ss = Math.floor(posSec % 60);
  const ms = Math.floor((posSec % 1) * 100);
  const timeStr = `${mm}:${String(ss).padStart(2, '0')}.${String(ms).padStart(2, '0')}`;

  // Position update + loop/stop logic + auto-scroll
  useEffect(() => {
    engine.onPositionUpdate((seconds) => {
      const s = useSynthStore.getState();
      const ticks = engine.secondsToTicks(seconds, s.project.tempos);
      s.setPlaybackPosition(ticks);

      const { startTick, endTick } = getLoopRange();
      if (ticks >= endTick) {
        if (s.loopEnabled) {
          const loopSec = engine.ticksToSeconds(startTick, s.project.tempos);
          s.setPlaybackPosition(startTick);
          engine.seek(loopSec);
        } else {
          engine.pause();
          s.setPlaying(false);
          s.setPlaybackPosition(endTick);
        }
      }
    });
    return () => engine.onPositionUpdate(() => {});
  }, []);

  // WaveSurfer init
  useEffect(() => {
    if (!waveRef.current) return;
    const ws = WaveSurfer.create({
      container: waveRef.current,
      height: 28,
      waveColor: '#5a4a8a',
      progressColor: '#9b6dff',
      cursorColor: '#ff6b9d',
      cursorWidth: 1,
      barWidth: 1,
      barGap: 1,
      interact: false,
      normalize: true,
      backend: 'WebAudio',
    });
    ws.on('click', (progress) => {
      const dur = ws.getDuration();
      if (!dur || !isFinite(dur) || dur <= 0) return;
      const s = useSynthStore.getState();
      const sec = progress * dur;
      let ticks = engine.secondsToTicks(sec, s.project.tempos);
      if (s.viewMode === 'arrangement') {
        const ts = s.project.time_signatures[0];
        const ticksPerBar = RESOLUTION * (ts?.beat_per_bar ?? 4);
        ticks = Math.round(ticks / ticksPerBar) * ticksPerBar;
      }
      ticks = Math.max(0, ticks);
      s.setPlaybackPosition(ticks);
      engine.seek(engine.ticksToSeconds(ticks, s.project.tempos));
    });
    wsRef.current = ws;
    return () => { ws.destroy(); wsRef.current = null; };
  }, []);

  // Build mixed waveform for bottom preview (ALL tracks combined)
  const wavePartsFingerprint = project.wave_parts.map(wp => `${wp.buffer_id}:${wp.position}:${wp.skip}:${wp.file_duration_ms}`).join('|');
  const mixedBlobUrl = useRef<string | null>(null);
  const mixAbort = useRef(0);

  useEffect(() => {
    const ws = wsRef.current;
    if (!ws) return;

    const gen = ++mixAbort.current;
    const bpm = project.tempos[0]?.bpm ?? 120;
    const tps = bpm / 60 * RESOLUTION;

    const sources: { url: string; startSec: number; skipSec?: number; durSec?: number }[] = [];

    const vocalTrackNo = project.voice_parts[0]?.track_no ?? -1;
    for (let idx = 0; idx < project.tracks.length; idx++) {
      const track = project.tracks[idx];
      if (track.track_type === 'audio') continue;
      const url = trackAudioUrls[idx] || (idx === vocalTrackNo ? (finalUrl || renderedUrl) : null);
      if (!url) continue;
      const vp = project.voice_parts.find(p => p.track_no === idx);
      const startSec = vp ? vp.position / tps : 0;
      sources.push({ url, startSec });
    }

    for (const wp of project.wave_parts) {
      if (!wp.buffer_id) continue;
      const fxUrl = useSynthStore.getState().waveFxUrls[wp.buffer_id];
      sources.push({
        url: fxUrl || `/api/audio/${wp.buffer_id}`,
        startSec: wp.position / tps,
        skipSec: fxUrl ? 0 : (wp.skip || 0) / 1000,
        durSec: wp.file_duration_ms / 1000,
      });
    }

    if (sources.length === 0) {
      ws.empty();
      ws.setOptions({ interact: false });
      return;
    }

    (async () => {
      try {
        const sr = 44100;
        let maxSample = 0;
        const buffers: { buf: AudioBuffer; startSample: number; skipSample: number; durSample: number }[] = [];

        for (const src of sources) {
          if (gen !== mixAbort.current) return;
          const buf = await engine.loadAudioBuffer(src.url);
          const startSample = Math.round(src.startSec * sr);
          const skipSec = src.skipSec ?? 0;
          const durSec = src.durSec ?? (buf.duration - skipSec);
          const skipSample = Math.round(skipSec * buf.sampleRate);
          const durSample = Math.round(durSec * buf.sampleRate);
          const endSample = startSample + Math.round(durSec * sr);
          if (endSample > maxSample) maxSample = endSample;
          buffers.push({ buf, startSample, skipSample, durSample });
        }

        if (maxSample === 0 || gen !== mixAbort.current) return;

        const offline = new OfflineAudioContext(1, maxSample, sr);
        for (const { buf, startSample, skipSample, durSample } of buffers) {
          const source = offline.createBufferSource();
          source.buffer = buf;
          source.connect(offline.destination);
          source.start(startSample / sr, skipSample / buf.sampleRate, durSample / buf.sampleRate);
        }
        const rendered = await offline.startRendering();
        if (gen !== mixAbort.current) return;

        const wavData = rendered.getChannelData(0);
        const wavBuf = new ArrayBuffer(44 + wavData.length * 2);
        const view = new DataView(wavBuf);
        const writeStr = (o: number, s: string) => { for (let i = 0; i < s.length; i++) view.setUint8(o + i, s.charCodeAt(i)); };
        writeStr(0, 'RIFF'); view.setUint32(4, 36 + wavData.length * 2, true); writeStr(8, 'WAVE');
        writeStr(12, 'fmt '); view.setUint32(16, 16, true); view.setUint16(20, 1, true); view.setUint16(22, 1, true);
        view.setUint32(24, sr, true); view.setUint32(28, sr * 2, true); view.setUint16(32, 2, true); view.setUint16(34, 16, true);
        writeStr(36, 'data'); view.setUint32(40, wavData.length * 2, true);
        for (let i = 0; i < wavData.length; i++) {
          const s = Math.max(-1, Math.min(1, wavData[i]));
          view.setInt16(44 + i * 2, s < 0 ? s * 0x8000 : s * 0x7FFF, true);
        }
        const blob = new Blob([wavBuf], { type: 'audio/wav' });
        if (mixedBlobUrl.current) URL.revokeObjectURL(mixedBlobUrl.current);
        mixedBlobUrl.current = URL.createObjectURL(blob);
        ws.load(mixedBlobUrl.current);
        ws.setOptions({ interact: true });
      } catch {
        if (gen === mixAbort.current) ws.empty();
      }
    })();
  }, [finalUrl, renderedUrl, trackAudioUrls, wavePartsFingerprint, waveFxSnapshot]);

  // Sync position → WaveSurfer
  useEffect(() => {
    const ws = wsRef.current;
    if (!ws || !ws.getDuration()) return;
    const totalSec = ws.getDuration();
    if (totalSec <= 0) return;
    const progress = posSec / totalSec;
    if (progress >= 0 && progress <= 1 && isFinite(progress)) {
      ws.seekTo(progress);
    }
  }, [posSec]);

  const handlePlay = useCallback(async () => {
    if (isPlaying) {
      engine.pause();
      setPlaying(false);
      return;
    }

    const state = useSynthStore.getState();
    if (state.renderStatus === 'rendering') return;
    state.setRenderError('');
    const inPianoRoll = state.viewMode === 'pianoroll';

    // Auto-render dirty tracks before playing
    let tracksToRender: number[] = [];
    if (inPianoRoll) {
      const activeTrack = getActiveTrackIndex();
      if (activeTrack >= 0 && trackNeedsRender(activeTrack)) {
        tracksToRender = [activeTrack];
      }
    } else {
      tracksToRender = getDirtyTrackIndices();
    }

    // Block if any vocal track is missing RVC model
    const missingModel = tracksToRender.some(i => {
      const track = state.project.tracks[i];
      return track?.track_type === 'vocal' && !state.trackRvcModel[i];
    });
    if (missingModel) {
      state.setRenderError(t('synth.needRvcModel'));
      state.setRenderStatus('error');
      return;
    }

    if (tracksToRender.length > 0 && (state.renderStatus as string) !== 'rendering') {
      const ac = createRenderAbort();
      state.setRenderStatus('rendering');
      try {
        await renderTracks(tracksToRender, ac.signal);
        useSynthStore.getState().setRenderError('');
        useSynthStore.getState().setRenderStatus('done');
      } catch (err: any) {
        if (err.name === 'AbortError') return;
        useSynthStore.getState().setRenderStatus('error');
        useSynthStore.getState().setRenderError(`Auto-render: ${err.message}`);
        return;
      }
    }

    // Collect tracks for playback
    const state2 = useSynthStore.getState();
    const tracks: engine.PlayTrackConfig[] = [];
    const hasSolo = state2.project.tracks.some(t => t.solo);

    if (inPianoRoll) {
      // Play only the active track
      const activeTrackIdx = getActiveTrackIndex();
      if (activeTrackIdx >= 0) {
        const track = state2.project.tracks[activeTrackIdx];
        if (track) {
          if (track.track_type === 'audio') {
            state2.project.wave_parts.forEach(wp => {
              if (wp.track_no !== activeTrackIdx || !wp.buffer_id) return;
              const fxUrl = state2.waveFxUrls[wp.buffer_id];
              tracks.push({
                url: fxUrl || `/api/audio/${wp.buffer_id}`,
                startOffsetSec: engine.ticksToSeconds(wp.position, state2.project.tempos),
                volume: track.volume + (wp.volume ?? 0),
                pan: Math.max(-100, Math.min(100, track.pan + (wp.pan ?? 0))),
                muted: false,
                skipSec: fxUrl ? 0 : (wp.skip || 0) / 1000,
                durationSec: fxUrl ? undefined : (wp.file_duration_ms || 0) / 1000,
              });
            });
          } else {
            const url = state2.trackAudioUrls[activeTrackIdx];
            if (url) {
              const vp = state2.project.voice_parts.find(p => p.track_no === activeTrackIdx);
              const ip = state2.project.instrument_parts.find(p => p.track_no === activeTrackIdx);
              const startPart = vp || ip;
              const startSec = startPart ? engine.ticksToSeconds(startPart.position, state2.project.tempos) : 0;
              const clipVol = (vp?.volume ?? ip?.volume ?? 0) as number;
              const clipPan = (vp?.pan ?? ip?.pan ?? 0) as number;
              tracks.push({
                url,
                startOffsetSec: startSec,
                volume: track.volume + clipVol,
                pan: Math.max(-100, Math.min(100, track.pan + clipPan)),
                muted: false,
              });
            }
          }
        }
      }
    } else {
      // Arrangement: play all tracks
      state2.project.tracks.forEach((track, idx) => {
        if (track.track_type === 'audio') return;
        const muted = hasSolo ? !track.solo : track.mute;
        const url = state2.trackAudioUrls[idx] || (state2.finalUrl || state2.renderedUrl);
        if (!url) return;
        // Only use finalUrl/renderedUrl fallback for the first vocal track
        if (!state2.trackAudioUrls[idx]) {
          const vocalTrackNo = state2.project.voice_parts[0]?.track_no ?? -1;
          if (idx !== vocalTrackNo) return;
        }
        const vp = state2.project.voice_parts.find(p => p.track_no === idx);
        const ip = state2.project.instrument_parts.find(p => p.track_no === idx);
        const startPart = vp || ip;
        const startSec = startPart ? engine.ticksToSeconds(startPart.position, state2.project.tempos) : 0;
        const clipVol = (vp?.volume ?? 0) as number;
        const clipPan = (vp?.pan ?? 0) as number;
        tracks.push({ url, startOffsetSec: startSec, volume: track.volume + clipVol, pan: Math.max(-100, Math.min(100, track.pan + clipPan)), muted });
      });

      state2.project.wave_parts.forEach((wp) => {
        const track = state2.project.tracks[wp.track_no];
        if (!track || !wp.buffer_id) return;
        const muted = hasSolo ? !track.solo : track.mute;
        const clipVol = wp.volume ?? 0;
        const clipPan = wp.pan ?? 0;
        const fxUrl = state2.waveFxUrls[wp.buffer_id];
        tracks.push({
          url: fxUrl || `/api/audio/${wp.buffer_id}`,
          startOffsetSec: engine.ticksToSeconds(wp.position, state2.project.tempos),
          volume: track.volume + clipVol, pan: Math.max(-100, Math.min(100, track.pan + clipPan)), muted,
          skipSec: fxUrl ? 0 : (wp.skip || 0) / 1000,
          durationSec: fxUrl ? undefined : (wp.file_duration_ms || 0) / 1000,
        });
      });
    }

    if (tracks.length === 0) {
      const s3 = useSynthStore.getState();
      const hasVocalNotes = s3.project.voice_parts.some(vp => vp.notes.length > 0);
      const hasInstNotes = s3.project.instrument_parts.some(ip => ip.notes.length > 0);
      const hasModel = s3.project.tracks.some((tr, i) =>
        tr.track_type === 'vocal' && s3.trackRvcModel[i]);
      const hasPlugin = s3.project.tracks.some(tr =>
        tr.track_type === 'instrument' && tr.plugin_name);
      if (hasVocalNotes && !hasModel) {
        s3.setRenderError(t('synth.needRvcModel'));
        s3.setRenderStatus('error');
      } else if (hasInstNotes && !hasPlugin) {
        s3.setRenderError(t('synth.needVstPlugin'));
        s3.setRenderStatus('error');
      } else if (!hasVocalNotes && !hasInstNotes) {
        s3.setRenderError(t('synth.noNotes', 'No notes to play. Add notes first.'));
        s3.setRenderStatus('error');
      }
      return;
    }
    const fromSec = engine.ticksToSeconds(state2.playbackPositionTicks, state2.project.tempos);
    engine.play(tracks, fromSec);
    setPlaying(true);
  }, [isPlaying, setPlaying]);

  const handleReset = useCallback(() => {
    const s = useSynthStore.getState();
    const partStart = s.project.voice_parts[s.activePartIndex]?.position ?? 0;
    const partStartPx = partStart * s.viewState.pixelsPerTick;
    setPlaybackPosition(partStart);
    engine.seek(engine.ticksToSeconds(partStart, s.project.tempos));
    s.setViewState({ scrollX: partStartPx });
  }, [setPlaybackPosition]);

  return (
    <div className="utai-synth-transport">
      <div className="utai-synth-transport__controls">
        <button
          className={`utai-synth-transport__btn ${isPlaying ? 'utai-synth-transport__btn--active' : ''}`}
          onClick={handlePlay}
          disabled={renderStatus === 'rendering' && !isPlaying}
          title={renderStatus === 'rendering' ? t('synth.transport.rendering', 'Rendering…') : isPlaying ? t('synth.transport.pause') : t('synth.transport.play')}
        >
          {renderStatus === 'rendering' && !isPlaying ? '⟳' : isPlaying ? '║║' : '▶'}
        </button>
        <button
          className="utai-synth-transport__btn"
          onClick={handleReset}
          title={t('synth.transport.reset')}
        >
          ◀◀
        </button>
        <button
          className={`utai-synth-transport__btn ${loopEnabled ? 'utai-synth-transport__btn--active' : ''}`}
          onClick={toggleLoop}
          title={t('synth.transport.loop')}
        >
          ↻
        </button>
      </div>

      <div className="utai-synth-transport__position">
        <span className="utai-synth-transport__pos-main">{posStr}</span>
        <span className="utai-synth-transport__pos-time">{timeStr}</span>
      </div>

      <div className="utai-synth-transport__bpm">
        ♩={bpm}
      </div>

      <div className="utai-synth-transport__wave" ref={waveRef} />

    </div>
  );
}
