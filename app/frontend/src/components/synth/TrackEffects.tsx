/** Per-track effect chain editor. */

import { useState } from 'react';
import { useTranslation } from 'react-i18next';
import { useSynthStore } from '../../store/synthStore';
import { EFFECT_DEFAULTS, EFFECT_LABELS, type EffectType, type TrackEffect } from '../../lib/synth/effects';

interface Props {
  trackIndex: number;
}

const EFFECT_TYPES: EffectType[] = ['gain', 'eq3band', 'reverb', 'compressor'];

export default function TrackEffects({ trackIndex }: Props) {
  const { t } = useTranslation();
  const effects = useSynthStore(s => s.trackEffects[trackIndex] || []);
  const addEffect = useSynthStore(s => s.addEffect);
  const removeEffect = useSynthStore(s => s.removeEffect);
  const updateEffectParams = useSynthStore(s => s.updateEffectParams);
  const [open, setOpen] = useState(false);

  const handleAdd = (type: EffectType) => {
    addEffect(trackIndex, {
      type,
      params: { ...EFFECT_DEFAULTS[type] },
      enabled: true,
    });
  };

  if (!open) {
    return (
      <button
        className="utai-synth-arrangement__action-btn"
        onClick={() => setOpen(true)}
        style={{ fontSize: 9 }}
      >
        FX ({effects.length})
      </button>
    );
  }

  return (
    <div className="utai-synth-effects">
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 6 }}>
        <span style={{ fontFamily: 'var(--font-pixel)', fontSize: 11, color: 'var(--accent-purple)' }}>
          FX Chain — Track {trackIndex + 1}
        </span>
        <button className="utai-btn" onClick={() => setOpen(false)} style={{ fontSize: 10, padding: '1px 6px' }}>
          ×
        </button>
      </div>

      {effects.map((fx, i) => (
        <div key={i} className="utai-synth-effects__slot">
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <span style={{ fontSize: 10, color: fx.enabled ? '#e0e0ff' : '#666' }}>
              {EFFECT_LABELS[fx.type]}
            </span>
            <div style={{ display: 'flex', gap: 4 }}>
              <button
                className="utai-btn"
                onClick={() => updateEffectParams(trackIndex, i, { _enabled: fx.enabled ? 0 : 1 } as any)}
                style={{ fontSize: 9, padding: '0 4px', color: fx.enabled ? 'var(--accent-cyan)' : '#666' }}
              >
                {fx.enabled ? 'ON' : 'OFF'}
              </button>
              <button
                className="utai-btn"
                onClick={() => removeEffect(trackIndex, i)}
                style={{ fontSize: 9, padding: '0 4px', color: 'var(--accent-red)' }}
              >
                ×
              </button>
            </div>
          </div>
          {fx.enabled && (
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 4, marginTop: 4 }}>
              {Object.entries(fx.params).map(([key, val]) => (
                <label key={key} style={{ fontSize: 9, color: 'var(--text-muted)' }}>
                  {key}
                  <input
                    type="range"
                    min={key === 'mix' ? 0 : key === 'gain' || key === 'low' || key === 'mid' || key === 'high' ? -24 : key === 'threshold' ? -60 : 0}
                    max={key === 'mix' ? 1 : key === 'gain' || key === 'low' || key === 'mid' || key === 'high' ? 24 : key === 'ratio' ? 20 : key === 'decay' ? 5 : 1}
                    step={key === 'mix' || key === 'attack' || key === 'release' ? 0.01 : 0.5}
                    value={val}
                    onChange={e => updateEffectParams(trackIndex, i, { [key]: Number(e.target.value) })}
                    style={{ width: '100%' }}
                  />
                </label>
              ))}
            </div>
          )}
        </div>
      ))}

      <select
        onChange={e => { if (e.target.value) { handleAdd(e.target.value as EffectType); e.target.value = ''; } }}
        defaultValue=""
        style={{ width: '100%', marginTop: 4, background: 'var(--bg-deep)', color: 'var(--text-muted)', border: '1px solid var(--border-dim)', fontSize: 10, padding: '2px 4px' }}
      >
        <option value="">+ Add Effect</option>
        {EFFECT_TYPES.map(t => (
          <option key={t} value={t}>{EFFECT_LABELS[t]}</option>
        ))}
      </select>
    </div>
  );
}
