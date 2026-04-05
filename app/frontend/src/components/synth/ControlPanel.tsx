import { useState } from 'react';
import { useTranslation } from 'react-i18next';
import { useSynthStore } from '../../store/synthStore';
import ProjectInfo from './ProjectInfo';
import SingerSelector from './SingerSelector';
import RVCModelSelector from './RVCModelSelector';
import RVCControls, { type RVCSettings, defaultRVCSettings } from './RVCControls';
import RenderButton from './RenderButton';
import AudioPreview from './AudioPreview';
import LogViewer from './LogViewer';
import G2pTool from './G2pTool';

export default function ControlPanel() {
  const { t } = useTranslation();
  const renderedUrl = useSynthStore(s => s.renderedUrl);
  const finalUrl = useSynthStore(s => s.finalUrl);
  const [rvcSettings, setRvcSettings] = useState<RVCSettings>(defaultRVCSettings);

  return (
    <div className="utai-synth-control">
      <ProjectInfo />
      <SingerSelector />
      <RVCModelSelector />
      <RVCControls settings={rvcSettings} onChange={setRvcSettings} />

      <hr style={{ border: 'none', borderTop: '1px dashed var(--border-dim)' }} />

      <RenderButton rvcSettings={rvcSettings} />

      <hr style={{ border: 'none', borderTop: '1px dashed var(--border-dim)' }} />

      <AudioPreview label={t('synth.audioIntermediate')} audioUrl={renderedUrl} />
      <AudioPreview label={t('synth.audioFinal')} audioUrl={finalUrl} />

      <div style={{ marginTop: 'auto' }}>
        <G2pTool />
        <LogViewer />
      </div>
    </div>
  );
}
