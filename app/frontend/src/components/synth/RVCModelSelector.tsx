import { useEffect, useState } from 'react';
import { useTranslation } from 'react-i18next';
import { useSynthStore } from '../../store/synthStore';
import { getRVCModels } from '../../api/client';

interface Model { name: string; filename: string; }

export default function RVCModelSelector() {
  const { t } = useTranslation();
  const rvcModel = useSynthStore(s => s.rvcModel);
  const setRvcModel = useSynthStore(s => s.setRvcModel);
  const [models, setModels] = useState<Model[]>([]);

  useEffect(() => {
    getRVCModels().then(data => {
      const rvcModels = (data || []).map((m: any) => ({
        name: m.name, filename: m.filename || m.name + '.pth',
      }));
      setModels(rvcModels);
    }).catch(() => setModels([]));
  }, []);

  return (
    <div className="utai-synth-section">
      <div className="utai-synth-section__title">{t('synth.rvcModel')}</div>
      <select value={rvcModel} onChange={e => setRvcModel(e.target.value)}>
        <option value="">{t('synth.selectRvcModel')}</option>
        {models.map(m => (
          <option key={m.filename} value={m.filename}>{m.name}</option>
        ))}
      </select>
      {models.length === 0 && <small style={{ color: 'var(--text-muted)', fontSize: 10, display: 'block', marginTop: 4 }}>{t('synth.noRvcModels')}</small>}
    </div>
  );
}
