import { useEffect, useState, useCallback, useRef } from 'react';
import { useTranslation } from 'react-i18next';
import { useSynthStore } from '../../store/synthStore';
import { getVSTPlugins, openVSTEditor, closeVSTEditor } from '../../api/client';
import { onModelsChanged } from '../../utils/modelEvents';

const API_BASE = (import.meta as any).env?.VITE_API_URL || '';

export default function VSTPluginSelector() {
  const { t } = useTranslation();
  const project = useSynthStore(s => s.project);
  const activeInstIdx = useSynthStore(s => s.activeInstrumentPartIndex);
  const setTrackPlugin = useSynthStore(s => s.setTrackPlugin);
  const [plugins, setPlugins] = useState<any[]>([]);
  const [editorOpen, setEditorOpen] = useState(false);
  const prevEditorOpen = useRef(false);
  const bumpVstStateVersion = useSynthStore(s => s.bumpVstStateVersion);
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const instPart = activeInstIdx >= 0 ? project.instrument_parts[activeInstIdx] : null;
  const trackNo = instPart?.track_no ?? -1;
  const track = trackNo >= 0 ? project.tracks[trackNo] : null;

  useEffect(() => {
    getVSTPlugins().then(setPlugins).catch(console.error);
    return onModelsChanged(() => {
      getVSTPlugins().then(setPlugins).catch(console.error);
    });
  }, []);

  const checkStatus = useCallback(() => {
    if (!track?.plugin_name) return;
    fetch(`${API_BASE}/api/vst/editor/status/${encodeURIComponent(track.plugin_name)}`)
      .then(r => r.json()).then(d => setEditorOpen(d.open)).catch(() => setEditorOpen(false));
  }, [track?.plugin_name]);

  useEffect(() => { checkStatus(); }, [track?.plugin_name]);

  useEffect(() => {
    if (prevEditorOpen.current && !editorOpen && trackNo >= 0) {
      bumpVstStateVersion(trackNo);
    }
    prevEditorOpen.current = editorOpen;
  }, [editorOpen, trackNo, bumpVstStateVersion]);

  useEffect(() => {
    if (!editorOpen || !track?.plugin_name) {
      if (pollRef.current) { clearInterval(pollRef.current); pollRef.current = null; }
      return;
    }
    pollRef.current = setInterval(checkStatus, 1500);
    return () => { if (pollRef.current) { clearInterval(pollRef.current); pollRef.current = null; } };
  }, [editorOpen, checkStatus]);

  const handleToggleEditor = useCallback(async () => {
    if (!track?.plugin_name) return;
    try {
      if (editorOpen) {
        await closeVSTEditor(track.plugin_name);
        setEditorOpen(false);
      } else {
        const result = await openVSTEditor(track.plugin_name);
        setEditorOpen(result.status === 'opened' || result.status === 'already_open');
      }
    } catch (e: any) {
      console.error('Editor toggle failed:', e);
      checkStatus();
    }
  }, [track?.plugin_name, editorOpen, checkStatus]);

  if (!track) return null;

  return (
    <div style={{ padding: '6px 10px' }}>
      <div style={{ fontFamily: 'var(--font-pixel)', fontSize: 11, color: 'var(--accent-orange)', marginBottom: 6 }}>
        {t('synth.vstPlugin')}
      </div>
      <select
        value={track.plugin_name || ''}
        onChange={e => setTrackPlugin(trackNo, e.target.value)}
        style={{
          width: '100%',
          background: 'var(--bg-deep)',
          color: 'var(--text-primary)',
          border: '1px solid var(--accent-orange)',
          fontFamily: 'var(--font-mono)',
          fontSize: 11,
          padding: '4px 6px',
        }}
      >
        <option value="">{t('synth.selectPlugin')}</option>
        {plugins.map(p => (
          <option key={p.name} value={p.name}>{p.name}</option>
        ))}
      </select>

      {plugins.length === 0 && (
        <div style={{ fontSize: 9, color: 'var(--text-dim)', marginTop: 4 }}>
          {t('synth.noPlugins')}
        </div>
      )}

      {track.plugin_name && (
        <button
          className="utai-btn"
          style={{
            marginTop: 6,
            fontSize: 10,
            padding: '3px 10px',
            borderColor: 'var(--accent-orange)',
            color: editorOpen ? '#1a1a2e' : 'var(--accent-orange)',
            background: editorOpen ? 'var(--accent-orange)' : 'transparent',
          }}
          onClick={handleToggleEditor}
        >
          {editorOpen ? t('synth.closeEditor') : t('synth.openEditor')}
        </button>
      )}
    </div>
  );
}
