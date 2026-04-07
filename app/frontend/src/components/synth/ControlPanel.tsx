import { useTranslation } from 'react-i18next';
import { useSynthStore } from '../../store/synthStore';
import ProjectInfo from './ProjectInfo';
import SingerSelector from './SingerSelector';
import RVCModelSelector from './RVCModelSelector';
import RVCControls from './RVCControls';
import RenderButton from './RenderButton';
import VSTPluginSelector from './VSTPluginSelector';
import TrackEffectsPanel from './TrackEffectsPanel';
import AudioPreview from './AudioPreview';
import LogViewer from './LogViewer';
import G2pTool from './G2pTool';

export default function ControlPanel() {
  const { t } = useTranslation();
  const renderedUrl = useSynthStore(s => s.renderedUrl);
  const finalUrl = useSynthStore(s => s.finalUrl);
  const project = useSynthStore(s => s.project);
  const activePartIndex = useSynthStore(s => s.activePartIndex);
  const activeInstPartIndex = useSynthStore(s => s.activeInstrumentPartIndex);

  // Determine active track type
  let activeTrackType: 'vocal' | 'audio' | 'instrument' = 'vocal';
  if (activeInstPartIndex >= 0) {
    activeTrackType = 'instrument';
  } else {
    const vp = project.voice_parts[activePartIndex];
    const track = vp ? project.tracks[vp.track_no] : null;
    activeTrackType = (track?.track_type as any) || 'vocal';
  }

  return (
    <div className="utai-synth-control">
      <ProjectInfo />

      {activeTrackType === 'vocal' && (
        <>
          <G2pTool />
          <SingerSelector />
          <RVCModelSelector />
          <RVCControls />
        </>
      )}

      {activeTrackType === 'instrument' && (
        <>
          <VSTPluginSelector />
        </>
      )}

      <TrackEffectsPanel />

      <hr style={{ border: 'none', borderTop: '1px dashed var(--border-dim)' }} />

      <RenderButton />

      <hr style={{ border: 'none', borderTop: '1px dashed var(--border-dim)' }} />

      {activeTrackType === 'vocal' && (
        <>
          <AudioPreview label={t('synth.audioIntermediate')} audioUrl={renderedUrl} />
          <AudioPreview label={t('synth.audioFinal')} audioUrl={finalUrl} />
        </>
      )}

      <div style={{ marginTop: 'auto' }}>
        <LogViewer />
      </div>
    </div>
  );
}
