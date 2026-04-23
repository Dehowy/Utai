import { useSynthStore } from '../../store/synthStore';
import { RESOLUTION } from '../../types/ustx';
import { renderVST, applyEffectsUpload } from '../../api/client';
import * as engine from './audioEngine';

export function getTrackFingerprint(trackIdx: number): string {
  const s = useSynthStore.getState();
  const track = s.project.tracks[trackIdx];
  if (!track) return '';
  if (track.track_type === 'vocal') {
    const parts = s.project.voice_parts.filter(vp => vp.track_no === trackIdx);
    const rvcModel = s.trackRvcModel[trackIdx] || s.rvcModel;
    const lang = s.trackLanguage[trackIdx] || s.language;
    const idxRate = s.trackIndexRate[trackIdx] ?? s.rvcSettings.index_rate;
    const f0Off = (s.trackF0Offset[trackIdx] ?? 0) + (s.globalF0Offset || 0);
    const muteStrat = s.trackMuteStrategy[trackIdx] || s.muteStrategy;
    return JSON.stringify({
      p: parts.map(p => ({ pos: p.position, n: p.notes })),
      rvc: rvcModel, f0: s.rvcSettings.f0_up_key, gf0: f0Off, idx: idxRate, lang, mute: muteStrat,
    });
  }
  if (track.track_type === 'instrument') {
    const parts = s.project.instrument_parts.filter(ip => ip.track_no === trackIdx);
    return JSON.stringify({
      p: parts.map(p => ({ pos: p.position, n: p.notes })),
      plug: track.plugin_name,
    });
  }
  return '';
}

export function trackNeedsRender(trackIdx: number): boolean {
  const s = useSynthStore.getState();
  const track = s.project.tracks[trackIdx];
  if (!track || track.track_type === 'audio') return false;

  if (track.track_type === 'vocal') {
    const hasNotes = s.project.voice_parts.some(vp => vp.track_no === trackIdx && vp.notes.length > 0);
    if (!hasNotes) return false;
  } else if (track.track_type === 'instrument') {
    const hasNotes = s.project.instrument_parts.some(ip => ip.track_no === trackIdx && ip.notes.length > 0);
    if (!hasNotes) return false;
    if (!track.plugin_name) return false;
  }

  return getTrackFingerprint(trackIdx) !== (s.trackRenderedSnapshot[trackIdx] ?? '');
}

export function getDirtyTrackIndices(): number[] {
  const s = useSynthStore.getState();
  const indices: number[] = [];
  for (let i = 0; i < s.project.tracks.length; i++) {
    if (trackNeedsRender(i)) indices.push(i);
  }
  return indices;
}

export function getActiveTrackIndex(): number {
  const s = useSynthStore.getState();
  if (s.activeWavePartIndex >= 0)
    return s.project.wave_parts[s.activeWavePartIndex]?.track_no ?? -1;
  if (s.activeInstrumentPartIndex >= 0)
    return s.project.instrument_parts[s.activeInstrumentPartIndex]?.track_no ?? -1;
  if (s.activePartIndex >= 0)
    return s.project.voice_parts[s.activePartIndex]?.track_no ?? -1;
  return -1;
}

export async function renderSingleTrack(
  trackIdx: number,
  signal?: AbortSignal,
): Promise<boolean> {
  const s = useSynthStore.getState();
  const track = s.project.tracks[trackIdx];
  if (!track) return false;

  const bpm = s.project.tempos[0]?.bpm ?? 120;

  if (track.track_type === 'vocal') {
    const trackParts = s.project.voice_parts.filter(vp => vp.track_no === trackIdx);
    const allNotes = trackParts.flatMap(vp =>
      vp.notes.map(n => ({
        position: vp.position + n.position,
        duration: n.duration,
        tone: n.tone,
        lyric: n.lyric,
        pitch: n.pitch?.data?.length ? { data: n.pitch.data } : undefined,
      }))
    );

    if (allNotes.length === 0) return true;
    const rvcModel = s.trackRvcModel[trackIdx] || s.rvcModel;
    if (!rvcModel) return false;

    const form = new FormData();
    form.append('notes', JSON.stringify(allNotes));
    form.append('bpm', String(bpm));
    form.append('language', s.trackLanguage[trackIdx] || s.language);
    form.append('rvc_model', rvcModel);
    form.append('f0_up_key', String((s.trackF0Offset[trackIdx] ?? 0) + (s.globalF0Offset || 0)));
    form.append('index_rate', String(s.trackIndexRate[trackIdx] ?? s.rvcSettings.index_rate));
    form.append('mute_strategy', s.trackMuteStrategy[trackIdx] || s.muteStrategy);

    const res = await fetch('/api/synth/s2h/render', { method: 'POST', body: form, signal });
    if (!res.ok) {
      let msg = `HTTP ${res.status}`;
      try { const j = await res.json(); msg = j.detail || j.error || msg; } catch { msg = await res.text(); }
      throw new Error(msg);
    }

    const rawBlob = await res.blob();
    if (rawBlob.size < 100) return false;

    const rawUrl = URL.createObjectURL(rawBlob);
    useSynthStore.getState().setTrackRawAudioUrl(trackIdx, rawUrl);

    // Apply track effects if any are enabled
    const finalUrl = await applyVocalTrackEffects(trackIdx, rawBlob);

    useSynthStore.getState().setTrackAudioUrl(trackIdx, finalUrl || rawUrl);
    useSynthStore.getState().setFinalUrl(useSynthStore.getState().trackAudioUrls[trackIdx]);

  } else if (track.track_type === 'instrument') {
    const trackParts = s.project.instrument_parts.filter(ip => ip.track_no === trackIdx);
    for (const ip of trackParts) {
      if (!track.plugin_name || ip.notes.length === 0) continue;
      const notes = ip.notes.map(n => ({
        tone: n.tone,
        velocity: n.velocity,
        start: engine.ticksToSeconds(ip.position + n.position, s.project.tempos),
        duration: engine.ticksToSeconds(n.duration, [{ position: 0, bpm }]),
      }));
      const blob = await renderVST({ plugin_name: track.plugin_name, notes, bpm });
      const prevUrl = useSynthStore.getState().trackAudioUrls[trackIdx];
      if (prevUrl) URL.revokeObjectURL(prevUrl);
      useSynthStore.getState().setTrackAudioUrl(trackIdx, URL.createObjectURL(blob));
    }
  } else {
    return true;
  }

  // Apply track effects
  const latestUrl = useSynthStore.getState().trackAudioUrls[trackIdx];
  const effects = s.trackEffects[trackIdx];
  if (effects && effects.length > 0 && latestUrl) {
    const renderFx = effects.filter(e => e.enabled && e.type in {
      world_pitch: 1, warp_pitch: 1, compressor: 1, reverb: 1, filter: 1, delay: 1, panner: 1,
    });
    if (renderFx.length > 0) {
      try {
        const audioBlob = await fetch(latestUrl).then(r => r.blob());
        const processed = await applyEffectsUpload(audioBlob, renderFx);
        const prevUrl2 = useSynthStore.getState().trackAudioUrls[trackIdx];
        if (prevUrl2) URL.revokeObjectURL(prevUrl2);
        const fxUrl = URL.createObjectURL(processed);
        useSynthStore.getState().setTrackAudioUrl(trackIdx, fxUrl);
        if (track.track_type === 'vocal') useSynthStore.getState().setFinalUrl(fxUrl);
      } catch (err: any) {
        console.error(`Track ${trackIdx} effects failed:`, err);
      }
    }
  }

  // Save fingerprint so we know this track is up-to-date
  useSynthStore.getState().setTrackRenderedSnapshot(trackIdx, getTrackFingerprint(trackIdx));
  return true;
}

async function applyVocalTrackEffects(
  trackIdx: number, rawBlob: Blob,
): Promise<string | null> {
  const s = useSynthStore.getState();
  const effects = (s.trackEffects[trackIdx] || [])
    .filter(fx => fx.enabled)
    .map(fx => ({ type: fx.type, params: fx.params, enabled: true }));
  if (effects.length === 0) return null;

  try {
    const processedBlob = await applyEffectsUpload(rawBlob, effects);
    return URL.createObjectURL(processedBlob);
  } catch (err: any) {
    console.error(`Vocal track ${trackIdx} effects failed:`, err);
    return null;
  }
}

export async function renderTracks(
  trackIndices: number[],
  signal?: AbortSignal,
  onProgress?: (msg: string) => void,
): Promise<void> {
  for (let i = 0; i < trackIndices.length; i++) {
    if (signal?.aborted) return;
    const trackIdx = trackIndices[i];
    const trackName = useSynthStore.getState().project.tracks[trackIdx]?.track_name ?? `Track ${trackIdx + 1}`;
    onProgress?.(`${trackName} (${i + 1}/${trackIndices.length})`);
    await renderSingleTrack(trackIdx, signal);
  }
}
