import { useState } from 'react';
import { useTranslation } from 'react-i18next';
import { useSynthStore } from '../../store/synthStore';
import type { RVCSettings } from '../../store/synthStore';

export type { RVCSettings };
export { defaultRVCSettings } from '../../store/synthStore';

export default function RVCControls() {
  const { t } = useTranslation();
  const [open, setOpen] = useState(false);
  const settings = useSynthStore(s => s.rvcSettings);
  const setRvcSettings = useSynthStore(s => s.setRvcSettings);

  const update = (key: keyof RVCSettings, value: number | string) => {
    setRvcSettings({ [key]: value });
  };

  return (
    <div className="utai-synth-section">
      <button
        className="utai-synth-section__title"
        onClick={() => setOpen(!open)}
        style={{ background: 'none', border: 'none', cursor: 'pointer', width: '100%', textAlign: 'left', padding: 0, font: 'inherit', color: 'inherit' }}
      >
        {open ? '▼' : '▶'} {t('synth.rvcParams')}
      </button>

      {open && (
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 6, marginTop: 6 }}>
          <div className="utai-synth-slider">
            <div className="utai-synth-slider__label">{t('synth.pitchShift')} <span>{settings.f0_up_key}</span></div>
            <input type="range" min={-12} max={12} step={1} value={settings.f0_up_key} onChange={e => update('f0_up_key', Number(e.target.value))} />
          </div>

          <div>
            <label>{t('synth.f0Method')}</label>
            <select value={settings.f0_method} onChange={e => update('f0_method', e.target.value)}>
              <option value="rmvpe">RMVPE</option>
              <option value="pm">PM</option>
              <option value="harvest">Harvest</option>
              <option value="crepe">Crepe</option>
            </select>
          </div>

          <div className="utai-synth-slider">
            <div className="utai-synth-slider__label">{t('synth.indexRate')} <span>{settings.index_rate.toFixed(2)}</span></div>
            <input type="range" min={0} max={1} step={0.05} value={settings.index_rate} onChange={e => update('index_rate', Number(e.target.value))} />
          </div>

          <div className="utai-synth-slider">
            <div className="utai-synth-slider__label">{t('synth.filterRadius')} <span>{settings.filter_radius}</span></div>
            <input type="range" min={0} max={7} step={1} value={settings.filter_radius} onChange={e => update('filter_radius', Number(e.target.value))} />
          </div>

          <div className="utai-synth-slider">
            <div className="utai-synth-slider__label">{t('synth.rmsMixRate')} <span>{settings.rms_mix_rate.toFixed(2)}</span></div>
            <input type="range" min={0} max={1} step={0.05} value={settings.rms_mix_rate} onChange={e => update('rms_mix_rate', Number(e.target.value))} />
          </div>

          <div className="utai-synth-slider">
            <div className="utai-synth-slider__label">{t('synth.protect')} <span>{settings.protect.toFixed(2)}</span></div>
            <input type="range" min={0} max={0.5} step={0.01} value={settings.protect} onChange={e => update('protect', Number(e.target.value))} />
          </div>
        </div>
      )}
    </div>
  );
}
