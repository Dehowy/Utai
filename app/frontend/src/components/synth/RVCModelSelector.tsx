import { useEffect, useState, useRef } from 'react';
import { useTranslation } from 'react-i18next';
import { useSynthStore } from '../../store/synthStore';
import { getRVCModels, detectRVCRange, getRVCRangeStatus } from '../../api/client';

interface RangeInfo {
  range_low: number; range_high: number;
  comfort_low: number; comfort_high: number;
  range_label: string; comfort_label: string;
}
interface Model {
  name: string; filename: string; range?: RangeInfo;
}

export default function RVCModelSelector() {
  const { t } = useTranslation();
  const trackRvcModel = useSynthStore(s => s.trackRvcModel);
  const activePartIndex = useSynthStore(s => s.activePartIndex);
  const project = useSynthStore(s => s.project);
  const setRvcModel = useSynthStore(s => s.setRvcModel);
  const listVersion = useSynthStore(s => s.rvcModelListVersion);
  const [models, setModels] = useState<Model[]>([]);
  const [detecting, setDetecting] = useState<string | null>(null);
  const pollRef = useRef<number | null>(null);
  const activeTrackNo = project.voice_parts[activePartIndex]?.track_no ?? -1;
  const displayModel = activeTrackNo >= 0 ? (trackRvcModel[activeTrackNo] || '') : '';

  useEffect(() => {
    getRVCModels().then(data => {
      const rvcModels = (data || []).map((m: any) => ({
        name: m.name,
        filename: m.filename || m.name + '.pth',
        range: m.range || undefined,
      }));
      setModels(rvcModels);
    }).catch(() => setModels([]));
  }, [listVersion]);

  useEffect(() => () => { if (pollRef.current) clearInterval(pollRef.current); }, []);

  const handleDetectRange = async () => {
    const selected = models.find(m => m.filename === displayModel);
    if (!selected) return;
    setDetecting(selected.name);
    try {
      await detectRVCRange(selected.name);
      pollRef.current = window.setInterval(async () => {
        try {
          const status = await getRVCRangeStatus(selected.name);
          if (status.status === 'done') {
            if (pollRef.current) clearInterval(pollRef.current);
            pollRef.current = null;
            setDetecting(null);
            useSynthStore.getState().bumpRvcModelList();
          }
        } catch { /* keep polling */ }
      }, 2000);
    } catch { setDetecting(null); }
  };

  const selectedModel = models.find(m => m.filename === displayModel);

  return (
    <div className="utai-synth-section">
      <div className="utai-synth-section__title">{t('synth.rvcModel')}</div>
      <select value={displayModel} onChange={e => setRvcModel(e.target.value)}>
        <option value="">{t('synth.selectRvcModel')}</option>
        {models.map(m => (
          <option key={m.filename} value={m.filename}>
            {m.name}{m.range ? ` [${m.range.comfort_label}]` : ''}
          </option>
        ))}
      </select>
      {selectedModel?.range && (
        <div style={{ fontSize: 9, color: '#8a8ab0', fontFamily: 'var(--font-pixel)', marginTop: 2 }}>
          Comfort: {selectedModel.range.comfort_label} | Usable: {selectedModel.range.range_label}
        </div>
      )}
      {displayModel && !selectedModel?.range && (
        <button
          onClick={handleDetectRange}
          disabled={!!detecting}
          style={{
            marginTop: 4, fontSize: 9, fontFamily: 'var(--font-pixel)',
            background: 'rgba(155,109,255,0.15)', border: '1px solid var(--accent-purple)',
            color: detecting ? '#5a5a7a' : 'var(--accent-purple)',
            padding: '2px 8px', cursor: detecting ? 'wait' : 'pointer', borderRadius: 3,
          }}
        >
          {detecting ? t('synth.detectingRange') : t('synth.detectRange')}
        </button>
      )}
      {models.length === 0 && <small style={{ color: 'var(--text-muted)', fontSize: 10, display: 'block', marginTop: 4 }}>{t('synth.noRvcModels')}</small>}
    </div>
  );
}
