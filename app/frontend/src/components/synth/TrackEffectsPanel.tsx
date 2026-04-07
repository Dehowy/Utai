/** Track effects chain editor — add/remove/configure render-time effects per track. */

import { useState } from 'react';
import { useTranslation } from 'react-i18next';
import { useSynthStore } from '../../store/synthStore';
import { RENDER_EFFECTS, type RenderEffectType, createDefaultEffect } from '../../lib/synth/effects';

export default function TrackEffectsPanel() {
  const { t } = useTranslation();
  const project = useSynthStore(s => s.project);
  const activePartIndex = useSynthStore(s => s.activePartIndex);
  const activeInstIdx = useSynthStore(s => s.activeInstrumentPartIndex);
  const trackEffects = useSynthStore(s => s.trackEffects);
  const addEffect = useSynthStore(s => s.addEffect);
  const removeEffect = useSynthStore(s => s.removeEffect);
  const updateEffectParams = useSynthStore(s => s.updateEffectParams);

  const [addMenuOpen, setAddMenuOpen] = useState(false);

  // Determine active track
  let trackIdx = 0;
  if (activeInstIdx >= 0) {
    const ip = project.instrument_parts[activeInstIdx];
    if (ip) trackIdx = ip.track_no;
  } else {
    const vp = project.voice_parts[activePartIndex];
    if (vp) trackIdx = vp.track_no;
  }

  const effects = trackEffects[trackIdx] || [];

  const handleAdd = (type: RenderEffectType) => {
    addEffect(trackIdx, createDefaultEffect(type));
    setAddMenuOpen(false);
  };

  const handleParamChange = (fxIdx: number, param: string, value: number) => {
    const fx = effects[fxIdx];
    if (!fx) return;
    updateEffectParams(trackIdx, fxIdx, { ...fx.params, [param]: value });
  };

  return (
    <div style={{ padding: '6px 10px' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 6 }}>
        <span style={{ fontFamily: 'var(--font-pixel)', fontSize: 11, color: 'var(--accent-cyan)' }}>
          {t('synth.effects')}
        </span>
        <div style={{ position: 'relative' }}>
          <button
            className="utai-btn"
            style={{ fontSize: 9, padding: '1px 8px', borderColor: 'var(--accent-cyan)', color: 'var(--accent-cyan)' }}
            onClick={() => setAddMenuOpen(!addMenuOpen)}
          >
            + {t('synth.addEffect')}
          </button>
          {addMenuOpen && (
            <>
              <div style={{ position: 'fixed', inset: 0, zIndex: 900 }} onClick={() => setAddMenuOpen(false)} />
              <div style={{
                position: 'absolute', right: 0, top: '100%', zIndex: 901,
                background: 'var(--bg-panel)', border: '1px solid var(--border-mid)',
                minWidth: 150, padding: 4,
              }}>
                {(Object.entries(RENDER_EFFECTS) as [RenderEffectType, any][])
                  .filter(([type]) => type !== 'panner')
                  .map(([type, def]) => (
                  <button
                    key={type}
                    className="utai-btn"
                    style={{ display: 'block', width: '100%', textAlign: 'left', fontSize: 10, padding: '3px 8px', marginBottom: 2, borderColor: 'var(--border-dim)' }}
                    onClick={() => handleAdd(type)}
                  >
                    {def.label}
                  </button>
                ))}
              </div>
            </>
          )}
        </div>
      </div>

      {effects.length === 0 && (
        <div style={{ fontSize: 9, color: 'var(--text-dim)', textAlign: 'center', padding: 8 }}>
          {t('synth.noEffects')}
        </div>
      )}

      {effects.map((fx, fxIdx) => {
        const def = RENDER_EFFECTS[fx.type as RenderEffectType];
        if (!def) return null;
        return (
          <div key={fxIdx} style={{
            border: '1px solid var(--border-dim)',
            padding: '4px 6px',
            marginBottom: 4,
            background: fx.enabled ? 'rgba(92,224,216,0.03)' : 'transparent',
          }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 3 }}>
              <span style={{ fontSize: 10, color: fx.enabled ? 'var(--accent-cyan)' : 'var(--text-dim)', fontFamily: 'var(--font-pixel)' }}>
                {def.label}
              </span>
              <div style={{ display: 'flex', gap: 4 }}>
                <button
                  className="utai-btn"
                  style={{ fontSize: 8, padding: '0 4px', borderColor: 'var(--border-dim)', color: fx.enabled ? 'var(--accent-green)' : 'var(--text-dim)' }}
                  onClick={() => updateEffectParams(trackIdx, fxIdx, { ...fx.params, __enabled: fx.enabled ? 0 : 1 })}
                  title="Toggle"
                >
                  {fx.enabled ? 'ON' : 'OFF'}
                </button>
                <button
                  className="utai-btn"
                  style={{ fontSize: 8, padding: '0 4px', borderColor: 'var(--accent-pink)', color: 'var(--accent-pink)' }}
                  onClick={() => removeEffect(trackIdx, fxIdx)}
                >
                  x
                </button>
              </div>
            </div>
            {Object.entries(def.params).map(([paramKey, paramDef]) => (
              <div key={paramKey} style={{ display: 'flex', alignItems: 'center', gap: 4, marginBottom: 2 }}>
                <span style={{ fontSize: 9, color: 'var(--text-dim)', width: 55, flexShrink: 0 }}>{paramDef.label}</span>
                <input
                  type="range"
                  min={paramDef.min}
                  max={paramDef.max}
                  step={paramDef.step}
                  value={fx.params[paramKey] ?? paramDef.default}
                  onChange={e => handleParamChange(fxIdx, paramKey, parseFloat(e.target.value))}
                  style={{ flex: 1, height: 12 }}
                />
                <span style={{ fontSize: 9, color: 'var(--text-secondary)', width: 35, textAlign: 'right', fontFamily: 'var(--font-mono)' }}>
                  {(fx.params[paramKey] ?? paramDef.default).toFixed(paramDef.step < 1 ? 2 : 0)}
                </span>
              </div>
            ))}
          </div>
        );
      })}
    </div>
  );
}
