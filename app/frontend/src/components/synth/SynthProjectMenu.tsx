/** Synth project menu — dropdown with save/load/export/new/import operations. */

import { useRef, useState } from 'react';
import { useTranslation } from 'react-i18next';
import { useSynthStore } from '../../store/synthStore';
import { parseUstx } from '../../lib/synth/ustx';
import { utaiAlert, utaiConfirm } from '../UtaiAlert';
import { showToast } from '../Toast';

async function getSettings() {
  const s = useSynthStore.getState();
  return { language: s.language, rvcModel: s.rvcModel, trackRvcModel: s.trackRvcModel, muteStrategy: s.muteStrategy, trackMuteStrategy: s.trackMuteStrategy, singer: s.singer, pitchSnap: s.pitchSnap, rvcSettings: s.rvcSettings, globalF0Offset: s.globalF0Offset };
}

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
          body: JSON.stringify(filePath),
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

  const handleImportUstx = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
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
    e.target.value = '';
    setOpen(false);
  };

  const handleExportAudio = async () => {
    setOpen(false);
    const s = useSynthStore.getState();
    try {
      const res = await fetch('/api/synth/project/export-audio-dialog', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ project: s.project, trackAudioUrls: s.trackAudioUrls, format: 'wav', sample_rate: 44100, bit_depth: 24 }),
      });
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
    { label: t('synth.project.importUstx'), action: () => { setOpen(false); ustxInputRef.current?.click(); }, shortcut: '' },
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
      <input ref={ustxInputRef} type="file" accept=".ustx,.ust" onChange={handleImportUstx} style={{ display: 'none' }} />
    </div>
  );
}

function _applyOpenedProject(data: any) {
  const s = useSynthStore.getState();
  s.loadProject(data.project);
  if (data.trackEffects) useSynthStore.setState({ trackEffects: data.trackEffects });
  if (data.settings) {
    const st = data.settings;
    if (st.language) useSynthStore.setState({ language: st.language });
    if (st.rvcModel) useSynthStore.setState({ rvcModel: st.rvcModel });
    if (st.trackRvcModel) useSynthStore.setState({ trackRvcModel: st.trackRvcModel });
    if (st.muteStrategy) useSynthStore.setState({ muteStrategy: st.muteStrategy });
    if (st.trackMuteStrategy) useSynthStore.setState({ trackMuteStrategy: st.trackMuteStrategy });
    if (st.singer) useSynthStore.setState({ singer: st.singer });
    if (st.pitchSnap != null) useSynthStore.setState({ pitchSnap: st.pitchSnap });
    if (st.rvcSettings) useSynthStore.setState({ rvcSettings: st.rvcSettings });
    if (st.globalF0Offset != null) useSynthStore.setState({ globalF0Offset: st.globalF0Offset });
  }
  if (data.path) useSynthStore.setState({ projectPath: data.path });
  // Regenerate peak data for restored audio buffers
  for (const wp of data.project.wave_parts || []) {
    if (wp.buffer_id) {
      fetch(`/api/audio/${wp.buffer_id}/peaks?width=4000`).then(r => r.json()).then(d => {
        if (d.peaks) useSynthStore.getState().setPeakData(wp.buffer_id, new Float32Array(d.peaks));
      }).catch(() => {});
    }
  }
  useSynthStore.setState({ dirty: false });
  showToast('Project loaded', 'success');
}

export { _applyOpenedProject as applyOpenedProject };
