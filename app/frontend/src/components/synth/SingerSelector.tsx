import { useEffect, useState } from 'react';
import { useTranslation } from 'react-i18next';
import { useSynthStore } from '../../store/synthStore';
interface Singer { id: string; name?: string; type?: string; }

export default function SingerSelector() {
  const { t } = useTranslation();
  const singer = useSynthStore(s => s.singer);
  const setSingerModel = useSynthStore(s => s.setSingerModel);
  const setSinger = useSynthStore(s => s.setSinger);
  const [singers, setSingers] = useState<Singer[]>([]);

  useEffect(() => {
    fetch('/api/synth/singers')
      .then(r => { if (!r.ok) throw new Error(); return r.json(); })
      .then(data => { if (Array.isArray(data)) setSingers(data); })
      .catch(() => setSingers([]));
  }, []);

  const handleChange = (singerId: string) => {
    setSingerModel(singerId);
    if (singerId) {
      setSinger(0, singerId, 'OpenUtau.Core.DiffSinger.DiffSingerPhonemizer');
    }
  };

  return (
    <div className="utai-synth-section">
      <div className="utai-synth-section__title">{t('synth.singer')}</div>
      <select value={singer} onChange={e => handleChange(e.target.value)}>
        <option value="">{t('synth.selectSinger')}</option>
        {singers.map(s => (
          <option key={s.id} value={s.id}>{s.name || s.id}</option>
        ))}
      </select>
      {singers.length === 0 && <small style={{ color: 'var(--text-muted)', fontSize: 10, display: 'block', marginTop: 4 }}>{t('synth.noSingers')}</small>}
    </div>
  );
}
