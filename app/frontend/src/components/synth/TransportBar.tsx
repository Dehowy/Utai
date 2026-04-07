/** Transport bar: play/pause/stop, position display, loop toggle, waveform strip. */

import { useEffect, useRef, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { useSynthStore } from '../../store/synthStore';
import { RESOLUTION } from '../../types/ustx';
import * as engine from '../../lib/synth/audioEngine';
import WaveSurfer from 'wavesurfer.js';

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

  const waveRef = useRef<HTMLDivElement>(null);
  const wsRef = useRef<WaveSurfer | null>(null);

  // BPM and time signature
  const bpm = project.tempos[0]?.bpm ?? 120;
  const ts = project.time_signatures[0];
  const beatsPerBar = ts?.beat_per_bar ?? 4;

  // Position display: bar:beat:tick
  const totalBeats = playbackPositionTicks / RESOLUTION;
  const bar = Math.floor(totalBeats / beatsPerBar) + 1;
  const beat = Math.floor(totalBeats % beatsPerBar) + 1;
  const tick = Math.floor(playbackPositionTicks % RESOLUTION);
  const posStr = `${bar}:${beat}:${String(tick).padStart(3, '0')}`;

  // Seconds display
  const posSec = engine.ticksToSeconds(playbackPositionTicks, project.tempos);
  const mm = Math.floor(posSec / 60);
  const ss = Math.floor(posSec % 60);
  const ms = Math.floor((posSec % 1) * 100);
  const timeStr = `${mm}:${String(ss).padStart(2, '0')}.${String(ms).padStart(2, '0')}`;

  // Position update callback from audio engine
  useEffect(() => {
    engine.onPositionUpdate((seconds) => {
      const ticks = engine.secondsToTicks(seconds, useSynthStore.getState().project.tempos);
      useSynthStore.getState().setPlaybackPosition(ticks);
    });
    return () => engine.onPositionUpdate(() => {});
  }, []);

  // Initialize WaveSurfer for waveform display
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
      interact: true,
      normalize: true,
      backend: 'WebAudio',
    });
    ws.on('click', (progress) => {
      if (!ws.getDuration()) return;
      const sec = progress * ws.getDuration();
      const ticks = engine.secondsToTicks(sec, useSynthStore.getState().project.tempos);
      useSynthStore.getState().setPlaybackPosition(ticks);
      engine.seek(sec);
    });
    wsRef.current = ws;
    return () => { ws.destroy(); wsRef.current = null; };
  }, []);

  // Load audio into WaveSurfer when render completes
  // Show audio from: RVC final → DiffSinger render → any track that has audio
  const anyTrackUrl = Object.values(trackAudioUrls).find(u => !!u) || null;
  const audioUrl = finalUrl || renderedUrl || anyTrackUrl;
  useEffect(() => {
    if (wsRef.current && audioUrl) {
      wsRef.current.load(audioUrl);
    } else if (wsRef.current) {
      wsRef.current.empty();
    }
  }, [audioUrl]);

  // Sync playback position → WaveSurfer progress (bidirectional)
  useEffect(() => {
    const ws = wsRef.current;
    if (!ws || !ws.getDuration()) return;
    const totalSec = ws.getDuration();
    if (totalSec <= 0) return;
    const progress = posSec / totalSec;
    // seekTo expects 0-1 and doesn't trigger click events
    if (progress >= 0 && progress <= 1 && isFinite(progress)) {
      ws.seekTo(progress);
    }
  }, [posSec]);

  const handlePlay = useCallback(() => {
    if (isPlaying) {
      engine.pause();
      setPlaying(false);
      return;
    }
    // Collect playable tracks
    const state = useSynthStore.getState();
    const tracks: engine.PlayTrackConfig[] = [];
    const hasSolo = state.project.tracks.some(t => t.solo);

    state.project.tracks.forEach((track, idx) => {
      const muted = hasSolo ? !track.solo : track.mute;
      const url = state.trackAudioUrls[idx] || (idx === 0 ? (state.finalUrl || state.renderedUrl) : null);
      if (!url) return;
      // Vocal tracks: find the voice part's position
      const vp = state.project.voice_parts.find(p => p.track_no === idx);
      const startSec = vp ? engine.ticksToSeconds(vp.position, state.project.tempos) : 0;
      tracks.push({
        url,
        startOffsetSec: startSec,
        volume: track.volume,
        pan: track.pan,
        muted,
      });
    });

    // Also add wave parts (audio tracks)
    state.project.wave_parts.forEach((wp) => {
      const track = state.project.tracks[wp.track_no];
      if (!track || !wp.buffer_id) return;
      const muted = hasSolo ? !track.solo : track.mute;
      tracks.push({
        url: `/api/audio/${wp.buffer_id}`,
        startOffsetSec: engine.ticksToSeconds(wp.position, state.project.tempos),
        volume: track.volume,
        pan: track.pan,
        muted,
      });
    });

    if (tracks.length === 0) return;
    const fromSec = engine.ticksToSeconds(state.playbackPositionTicks, state.project.tempos);
    engine.play(tracks, fromSec);
    setPlaying(true);
  }, [isPlaying, setPlaying]);

  const handleStop = useCallback(() => {
    engine.stop();
    setPlaying(false);
    setPlaybackPosition(0);
  }, [setPlaying, setPlaybackPosition]);

  const handleReset = useCallback(() => {
    setPlaybackPosition(0);
    engine.seek(0);
    useSynthStore.getState().setViewState({ scrollX: 0 });
  }, [setPlaybackPosition]);

  return (
    <div className="utai-synth-transport">
      <div className="utai-synth-transport__controls">
        <button
          className={`utai-synth-transport__btn ${isPlaying ? 'utai-synth-transport__btn--active' : ''}`}
          onClick={handlePlay}
          title={isPlaying ? t('synth.transport.pause') : t('synth.transport.play')}
        >
          {isPlaying ? '⏸' : '▶'}
        </button>
        <button
          className="utai-synth-transport__btn"
          onClick={handleStop}
          title={t('synth.transport.stop')}
        >
          ■
        </button>
        <button
          className="utai-synth-transport__btn"
          onClick={handleReset}
          title={t('synth.transport.reset')}
        >
          ⏮
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
