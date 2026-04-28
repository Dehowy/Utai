import { useSynthStore } from '../../store/synthStore';
import { RESOLUTION } from '../../types/ustx';
import { renderVST, applyEffectsUpload } from '../../api/client';
import * as engine from './audioEngine';

let _globalAbort: AbortController | null = null;

export function abortRender() {
  _globalAbort?.abort();
  _globalAbort = null;
  fetch('/api/synth/stop', { method: 'POST' }).catch(() => {});
  useSynthStore.getState().setRenderStatus('idle');
  useSynthStore.getState().setRenderError('');
}

export function createRenderAbort(): AbortController {
  _globalAbort?.abort();
  const ac = new AbortController();
  _globalAbort = ac;
  return ac;
}

export function getTrackFingerprint(trackIdx: number): string {
  const s = useSynthStore.getState();
  const track = s.project.tracks[trackIdx];
  if (!track) return '';
  if (track.track_type === 'vocal') {
    const parts = s.project.voice_parts.filter(vp => vp.track_no === trackIdx);
    const rvcModel = s.trackRvcModel[trackIdx] || '';
    const lang = s.trackLanguage[trackIdx] || s.language;
    const idxRate = s.trackIndexRate[trackIdx] ?? s.rvcSettings.index_rate;
    const f0Off = (s.trackF0Offset[trackIdx] ?? 0) + (s.globalF0Offset || 0);
    const muteStrat = s.trackMuteStrategy[trackIdx] || s.muteStrategy;
    const affine = s.trackAffineTransform[trackIdx] ?? s.affineTransform;
    return JSON.stringify({
      p: parts.map(p => ({ pos: p.position, n: p.notes })),
      rvc: rvcModel, gf0: f0Off, idx: idxRate,
      lang, mute: muteStrat, affine,
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
    const rvcModel = s.trackRvcModel[trackIdx] || '';
    if (!rvcModel) return false;

    const affine = s.trackAffineTransform[trackIdx] ?? s.affineTransform;
    const useModelF0 = s.trackModelF0[trackIdx] ?? s.modelF0;

    const form = new FormData();
    form.append('notes', JSON.stringify(allNotes));
    form.append('bpm', String(bpm));
    form.append('language', s.trackLanguage[trackIdx] || s.language);
    form.append('rvc_model', rvcModel);
    form.append('f0_up_key', String((s.trackF0Offset[trackIdx] ?? 0) + (s.globalF0Offset || 0)));
    form.append('index_rate', String(s.trackIndexRate[trackIdx] ?? s.rvcSettings.index_rate));
    form.append('mute_strategy', s.trackMuteStrategy[trackIdx] || s.muteStrategy);
    form.append('affine_transform', affine ? 'true' : 'false');
    form.append('cosine_index', s.cosineIndex ? 'true' : 'false');
    form.append('model_f0', useModelF0 ? 'true' : 'false');

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
    useSynthStore.getState().setTrackAudioUrl(trackIdx, rawUrl);
    useSynthStore.getState().setFinalUrl(rawUrl);

  } else if (track.track_type === 'instrument') {
    const trackParts = s.project.instrument_parts.filter(ip => ip.track_no === trackIdx);
    if (!track.plugin_name) return false;
    const firstPartPos = trackParts.length > 0
      ? Math.min(...trackParts.map(ip => ip.position)) : 0;
    const offsetSec = engine.ticksToSeconds(firstPartPos, s.project.tempos);
    const allNotes = trackParts.flatMap(ip =>
      ip.notes.map(n => ({
        tone: n.tone,
        velocity: n.velocity,
        start: engine.ticksToSeconds(ip.position + n.position, s.project.tempos) - offsetSec,
        duration: engine.ticksToSeconds(n.duration, [{ position: 0, bpm }]),
      }))
    );
    if (allNotes.length === 0) return true;
    const blob = await renderVST({ plugin_name: track.plugin_name, notes: allNotes, bpm });
    const rawUrl = URL.createObjectURL(blob);
    useSynthStore.getState().setTrackRawAudioUrl(trackIdx, rawUrl);
    useSynthStore.getState().setTrackAudioUrl(trackIdx, rawUrl);
  } else {
    return true;
  }

  // Apply effects to the just-rendered raw audio
  const s2 = useSynthStore.getState();
  const effects = (s2.trackEffects[trackIdx] || [])
    .filter(fx => fx.enabled)
    .map(fx => ({ type: fx.type, params: fx.params, enabled: true }));
  if (effects.length > 0) {
    try {
      const rawUrl = s2.trackRawAudioUrls[trackIdx];
      if (rawUrl) {
        const rawBlob2 = await fetch(rawUrl).then(r => r.blob());
        const fxBlob = await applyEffectsUpload(rawBlob2, effects);
        const fxUrl = URL.createObjectURL(fxBlob);
        useSynthStore.getState().setTrackAudioUrl(trackIdx, fxUrl);
        if (track.track_type === 'vocal') useSynthStore.getState().setFinalUrl(fxUrl);
      }
    } catch (err: any) {
      console.error(`Track ${trackIdx} effects failed:`, err);
    }
  }

  useSynthStore.getState().setTrackRenderedSnapshot(trackIdx, getTrackFingerprint(trackIdx));
  return true;
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
