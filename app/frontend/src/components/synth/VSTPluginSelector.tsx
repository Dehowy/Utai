import { useEffect, useState, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { useSynthStore } from '../../store/synthStore';
import { getVSTPlugins, openVSTEditor } from '../../api/client';
import { onModelsChanged } from '../../utils/modelEvents';

export default function VSTPluginSelector() {
  const { t } = useTranslation();
  const project = useSynthStore(s => s.project);
  const activeInstIdx = useSynthStore(s => s.activeInstrumentPartIndex);
  const setTrackPlugin = useSynthStore(s => s.setTrackPlugin);
  const [plugins, setPlugins] = useState<any[]>([]);

  const instPart = activeInstIdx >= 0 ? project.instrument_parts[activeInstIdx] : null;
  const trackNo = instPart?.track_no ?? -1;
  const track = trackNo >= 0 ? project.tracks[trackNo] : null;

  useEffect(() => {
    getVSTPlugins().then(setPlugins).catch(console.error);
    return onModelsChanged(() => {
      getVSTPlugins().then(setPlugins).catch(console.error);
    });
  }, []);

  const handleOpenEditor = useCallback(async () => {
    if (!track?.plugin_name) return;
    try {
      await openVSTEditor(track.plugin_name);
    } catch (e: any) {
      console.error('Failed to open editor:', e);
    }
  }, [track?.plugin_name]);

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
            color: 'var(--accent-orange)',
          }}
          onClick={handleOpenEditor}
        >
          {t('synth.openEditor')}
        </button>
      )}
    </div>
  );
}
