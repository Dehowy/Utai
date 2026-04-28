/** Synth project menu — dropdown with save/load/export/new/import operations. */

import { useRef, useState } from 'react';
import { useTranslation } from 'react-i18next';
import { useSynthStore } from '../../store/synthStore';
import { parseUstx } from '../../lib/synth/ustx';
import { parseMidi } from '../../lib/synth/midi';
import { utaiAlert, utaiConfirm } from '../UtaiAlert';
import { showToast } from '../Toast';

function getSettings() {
  const s = useSynthStore.getState();
  return { language: s.language, trackLanguage: s.trackLanguage, rvcModel: s.rvcModel, trackRvcModel: s.trackRvcModel, rvcSettings: s.rvcSettings, muteStrategy: s.muteStrategy, trackMuteStrategy: s.trackMuteStrategy, singer: s.singer, pitchSnap: s.pitchSnap, globalF0Offset: s.globalF0Offset, affineTransform: s.affineTransform, trackAffineTransform: s.trackAffineTransform, trackIndexRate: s.trackIndexRate, trackF0Offset: s.trackF0Offset, cosineIndex: s.cosineIndex, modelF0: s.modelF0, trackModelF0: s.trackModelF0, projectPath: s.projectPath };
}

export { getSettings as getSynthSettings };

function getProjectPayload() {
  const s = useSynthStore.getState();
  return { project: JSON.parse(JSON.stringify(s.project)), trackEffects: s.trackEffects };
}

async function confirmIfDirty(t: (k: string) => string): Promise<boolean> {
  const s = useSynthStore.getState();
  if (s.dirty) return await utaiConfirm(t('synth.discardChanges'));
  return true;
}

export async function saveProject(saveAs = false) {
  const s = useSynthStore.getState();
  if (s.isSaving) { showToast('Save in progress, please wait...', 'info'); return; }
  s.setIsSaving(true);
  try {
    const payload = { ...getProjectPayload(), settings: await getSettings() };
    const path = saveAs ? '' : (s.projectPath || '');

    if (path) {
      const res = await fetch('/api/synth/project/save', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ ...payload, path }),
      });
      if (!res.ok) throw new Error(await res.text());
      const data = await res.json();
      if (data.success) {
        s.markClean();
        showToast(`Saved to ${data.path}`, 'success');
      } else {
        throw new Error(data.detail || 'Save failed');
      }
    } else {
      const isElectron = !!(window as any).electronAPI;
      let res: Response;
      if (isElectron) {
        const savePath = await (window as any).electronAPI.saveFile({
          title: 'Save Synth Project',
          filters: [{ name: 'Utai Synth Project', extensions: ['utaiSynth'] }],
          defaultPath: `${s.project.name || 'project'}.utaiSynth`,
        });
        if (!savePath) { s.setIsSaving(false); return; }
        res = await fetch('/api/synth/project/save', {
          method: 'POST', headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ ...payload, path: savePath }),
        });
      } else {
        res = await fetch('/api/synth/project/save-dialog', {
          method: 'POST', headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(payload),
        });
      }
      if (!res.ok) throw new Error(await res.text());
      const data = await res.json();
      if (data.cancelled) { s.setIsSaving(false); return; }
      if (data.success) {
        if (!saveAs) s.setProjectPath(data.path);
        s.markClean();
        showToast(`Saved to ${data.path}`, 'success');
      } else {
        throw new Error(data.detail || 'Save failed');
      }
    }
    // Flush autosave with current state (includes projectPath) so it survives reload
    const cur = useSynthStore.getState();
    fetch('/api/synth/project/autosave', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ project: cur.project, trackEffects: cur.trackEffects, settings: await getSettings() }),
    }).catch(() => {});
  } catch (err: any) {
    showToast(`Save failed: ${err.message}`, 'error');
  } finally {
    s.setIsSaving(false);
  }
}

export default function SynthProjectMenu() {
  const { t } = useTranslation();
  const dirty = useSynthStore(s => s.dirty);
  const projectPath = useSynthStore(s => s.projectPath);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const ustxInputRef = useRef<HTMLInputElement>(null);
  const [open, setOpen] = useState(false);

  const handleNew = async () => {
    setOpen(false);
    if (!(await confirmIfDirty(t))) return;
    useSynthStore.getState().newProject();
    await fetch('/api/synth/project/autosave', { method: 'DELETE' });
    fetch('/api/synth/logs', { method: 'DELETE' }).catch(() => {});
    showToast('New project created', 'info');
  };

  const handleOpen = async () => {
    setOpen(false);
    if (!(await confirmIfDirty(t))) return;
    try {
      const isElectron = !!(window as any).electronAPI;
      let res: Response;
      if (isElectron) {
        const filePath = await (window as any).electronAPI.openFile({
          title: 'Open Project',
          filters: [{ name: 'Utai Project', extensions: ['utaiSynth', 'utai', 'utaiTr'] }],
        });
        if (!filePath) return;
        if (filePath.endsWith('.utai') || filePath.endsWith('.utaiTr')) {
          const { importWorkflowAuto } = await import('../../api/client');
          const data = await (await fetch('/api/workflows/import-path', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ path: filePath }) })).json();
          const { useWorkflowStore } = await import('../../store/workflowStore');
          const { useTrainingStore } = await import('../../store/trainingStore');
          const rfNodes = (data.nodes || []).map((n: any) => ({ id: n.id, type: n.type, position: n.position, data: { params: n.params || {} } }));
          const rfEdges = (data.connections || []).map((c: any, i: number) => ({ id: `e-${i}`, source: c.from_node, sourceHandle: c.from_port, target: c.to_node, targetHandle: c.to_port, type: 'smoothstep', animated: true }));
          if (data.project_type === 'training') {
            useTrainingStore.getState().setWorkflow(data.name || 'Imported', rfNodes, rfEdges);
            window.location.hash = '#/train';
            window.location.href = '/train';
          } else {
            useWorkflowStore.getState().setWorkflow(data.name || 'Imported', rfNodes, rfEdges);
            window.location.href = '/';
          }
          showToast('Project opened', 'success');
          return;
        }
        res = await fetch('/api/synth/project/open-path', {
          method: 'POST', headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ path: filePath }),
        });
      } else {
        res = await fetch('/api/synth/project/open-dialog', { method: 'POST' });
      }
      if (!res.ok) throw new Error(await res.text());
      const data = await res.json();
      if (data.cancelled) return;
      _applyOpenedProject(data);
    } catch (err: any) {
      utaiAlert(`Open failed: ${err.message}`);
    }
  };

  const handleSave = () => { setOpen(false); saveProject(); };
  const handleSaveAs = () => { setOpen(false); saveProject(true); };

  const handleImportScore = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    const ext = file.name.toLowerCase().split('.').pop();

    if (ext === 'mid' || ext === 'midi') {
      const reader = new FileReader();
      reader.onload = () => {
        try {
          const name = file.name.replace(/\.[^.]+$/, '');
          const proj = parseMidi(reader.result as ArrayBuffer, name);
          useSynthStore.getState().loadProject(proj);
          showToast(`Imported ${file.name}`, 'success');
        } catch (err) {
          utaiAlert(`Failed to load MIDI: ${err}`);
        }
      };
      reader.readAsArrayBuffer(file);
    } else {
      const reader = new FileReader();
      reader.onload = () => {
        try {
          const proj = parseUstx(reader.result as string);
          useSynthStore.getState().loadProject(proj);
          showToast('Imported .ustx', 'success');
        } catch (err) {
          utaiAlert(`Failed to load .ustx: ${err}`);
        }
      };
      reader.readAsText(file);
    }
    e.target.value = '';
    setOpen(false);
  };

  const handleExportAudio = async () => {
    setOpen(false);
    const s = useSynthStore.getState();
    try {
      const form = new FormData();
      const trackBlobs: Record<number, { position_ticks: number; volume: number; pan: number }> = {};

      // Collect vocal/instrument track blobs
      for (let idx = 0; idx < s.project.tracks.length; idx++) {
        const track = s.project.tracks[idx];
        if (track.track_type === 'audio') continue;
        const blobUrl = s.trackAudioUrls[idx];
        if (!blobUrl) continue;

        // Find earliest part position for this track
        let posTicks = 0;
        let clipVol = 0;
        let clipPan = 0;
        if (track.track_type === 'vocal') {
          const vp = s.project.voice_parts.find((p: any) => p.track_no === idx);
          if (vp) { posTicks = vp.position; clipVol = vp.volume ?? 0; clipPan = vp.pan ?? 0; }
        } else if (track.track_type === 'instrument') {
          const ip = s.project.instrument_parts.find((p: any) => p.track_no === idx);
          if (ip) { posTicks = ip.position; clipVol = ip.volume ?? 0; clipPan = ip.pan ?? 0; }
        }

        const blob = await fetch(blobUrl).then(r => r.blob());
        form.append('files', blob, `track_${idx}.wav`);
        trackBlobs[idx] = { position_ticks: posTicks, volume: clipVol, pan: clipPan };
      }

      const projectMeta = {
        project: s.project,
        trackBlobs,
        format: 'wav',
        sample_rate: 44100,
        bit_depth: 24,
      };
      form.append('project_json', JSON.stringify(projectMeta));

      const res = await fetch('/api/synth/project/export-mix', { method: 'POST', body: form });
      if (!res.ok) throw new Error(await res.text());
      const data = await res.json();
      if (data.cancelled) return;
      if (data.success) showToast(`Exported to ${data.path}`, 'success');
    } catch (err: any) {
      utaiAlert(`Export failed: ${err.message}`);
    }
  };

  const menuItems = [
    { label: t('synth.project.new'), action: handleNew, shortcut: '' },
    { label: t('synth.project.open'), action: handleOpen, shortcut: '' },
    { sep: true },
    { label: t('synth.project.save'), action: handleSave, shortcut: 'Ctrl+S', disabled: !dirty && !!projectPath },
    { label: t('synth.project.saveAs'), action: handleSaveAs, shortcut: '' },
    { sep: true },
    { label: t('synth.project.importScore'), action: () => { setOpen(false); ustxInputRef.current?.click(); }, shortcut: '' },
    { label: t('synth.project.exportAudio'), action: handleExportAudio, shortcut: '' },
  ];

  return (
    <div style={{ position: 'relative' }}>
      <button className="utai-btn" onClick={() => setOpen(!open)}
        style={{ fontSize: 11, borderColor: dirty ? 'var(--accent-pink)' : 'var(--accent-cyan)', color: dirty ? 'var(--accent-pink)' : 'var(--accent-cyan)' }}>
        {t('synth.project.menu')}
      </button>
      {open && (
        <>
          <div style={{ position: 'fixed', inset: 0, zIndex: 900 }} onClick={() => setOpen(false)} />
          <div style={{
            position: 'absolute', left: 0, top: '100%', zIndex: 901, marginTop: 2,
            background: 'var(--bg-panel)', border: '1px solid var(--border-mid)',
            minWidth: 200, padding: 4,
          }}>
            {menuItems.map((item, i) =>
              'sep' in item ? <div key={i} style={{ height: 1, background: 'var(--border-dim)', margin: '3px 0' }} />
              : (
                <button key={i} className="utai-synth-ctx-menu__item"
                  disabled={(item as any).disabled}
                  onClick={(item as any).action}
                  style={{ display: 'flex', justifyContent: 'space-between', width: '100%', opacity: (item as any).disabled ? 0.4 : 1 }}>
                  <span>{(item as any).label}</span>
                  {(item as any).shortcut && <span style={{ fontSize: 9, color: 'var(--text-muted)', marginLeft: 16 }}>{(item as any).shortcut}</span>}
                </button>
              )
            )}
          </div>
        </>
      )}
      <input ref={ustxInputRef} type="file" accept=".ustx,.ust,.mid,.midi" onChange={handleImportScore} style={{ display: 'none' }} />
    </div>
  );
}

function _applyOpenedProject(data: any) {
  const s = useSynthStore.getState();
  // Revoke old URLs
  for (const url of Object.values(s.trackAudioUrls)) { if (url) URL.revokeObjectURL(url); }
  if (s.finalUrl) URL.revokeObjectURL(s.finalUrl);

  // Detect track language from lyrics (same logic as loadProject)
  const trackLanguage: Record<number, string> = {};
  for (const vp of (data.project.voice_parts || [])) {
    if (trackLanguage[vp.track_no]) continue;
    const allLyrics = vp.notes.map((n: any) => n.lyric || '').join('');
    if (allLyrics.length > 0) {
      const jaCount = [...allLyrics].filter(c => {
        const cp = c.codePointAt(0) || 0;
        return (cp >= 0x3040 && cp <= 0x309F) || (cp >= 0x30A0 && cp <= 0x30FF);
      }).length;
      trackLanguage[vp.track_no] = jaCount / allLyrics.length > 0.5 ? 'ja' : 'zh';
    }
  }

  // Build the FULL new state in one shot — dirty never transitions to true,
  // so the autosave subscriber won't fire and overwrite saved data.
  const newState: Record<string, any> = {
    project: data.project,
    dirty: false,
    undoStack: [], redoStack: [],
    selectedNotes: new Set(),
    activePartIndex: 0, activeInstrumentPartIndex: -1, activeWavePartIndex: -1,
    selectedTrackIndex: 0, selectedSegment: null,
    viewMode: 'arrangement' as const,
    isPlaying: false, playbackPositionTicks: 0,
    loopEnabled: false, loopStart: 0, loopEnd: 0, arrScrollY: 0,
    trackAudioUrls: {}, trackRawAudioUrls: {}, waveFxUrls: {},
    audioFxRendering: {}, peakData: {},
    trackEffects: data.trackEffects || {},
    trackRenderedSnapshot: {},
    trackRvcModel: {}, trackLanguage,
    trackIndexRate: {}, trackF0Offset: {},
    trackMuteStrategy: {}, trackAffineTransform: {},
    cosineIndex: true, modelF0: false, trackModelF0: {},
    ghostTracks: [],
    affineTransform: false, globalF0Offset: 0,
    projectPath: null,
    renderedUrl: null, finalUrl: null,
    renderError: '', renderStatus: 'idle' as const,
  };

  // Overlay saved settings (null checks preserve false/0/{} values)
  if (data.settings) {
    const st = data.settings;
    if (st.language != null) newState.language = st.language;
    if (st.trackLanguage != null) newState.trackLanguage = st.trackLanguage;
    if (st.rvcModel != null) newState.rvcModel = st.rvcModel;
    if (st.trackRvcModel != null) newState.trackRvcModel = st.trackRvcModel;
    if (st.rvcSettings != null) newState.rvcSettings = st.rvcSettings;
    if (st.muteStrategy != null) newState.muteStrategy = st.muteStrategy;
    if (st.trackMuteStrategy != null) newState.trackMuteStrategy = st.trackMuteStrategy;
    if (st.singer != null) newState.singer = st.singer;
    if (st.pitchSnap != null) newState.pitchSnap = st.pitchSnap;
    if (st.globalF0Offset != null) newState.globalF0Offset = st.globalF0Offset;
    if (st.affineTransform != null) newState.affineTransform = st.affineTransform;
    if (st.trackAffineTransform != null) newState.trackAffineTransform = st.trackAffineTransform;
    if (st.trackIndexRate != null) newState.trackIndexRate = st.trackIndexRate;
    if (st.trackF0Offset != null) newState.trackF0Offset = st.trackF0Offset;
    if (st.cosineIndex != null) newState.cosineIndex = st.cosineIndex;
    if (st.modelF0 != null) newState.modelF0 = st.modelF0;
    if (st.trackModelF0 != null) newState.trackModelF0 = st.trackModelF0;
    if (st.projectPath != null) newState.projectPath = st.projectPath;
  }
  if (data.path) newState.projectPath = data.path;

  useSynthStore.setState(newState);

  for (const wp of data.project.wave_parts || []) {
    if (wp.buffer_id) {
      fetch(`/api/audio/${wp.buffer_id}/peaks?width=4000`).then(r => r.json()).then(d => {
        if (d.peaks) useSynthStore.getState().setPeakData(wp.buffer_id, new Float32Array(d.peaks));
      }).catch(() => {});
    }
  }
  showToast('Project loaded', 'success');
}

export { _applyOpenedProject as applyOpenedProject };
