/** Per-track effect chain editor (legacy — used in arrangement view).
 *  Delegates to the shared RENDER_EFFECTS definitions. */

import { useState } from 'react';
import { useSynthStore } from '../../store/synthStore';
import { RENDER_EFFECTS, type RenderEffectType, createDefaultEffect } from '../../lib/synth/effects';

interface Props {
  trackIndex: number;
}

const QUICK_EFFECTS: RenderEffectType[] = ['reverb', 'compressor', 'filter', 'delay'];

export default function TrackEffects({ trackIndex }: Props) {
  const effects = useSynthStore(s => s.trackEffects[trackIndex] || []);
  const addEffect = useSynthStore(s => s.addEffect);
  const removeEffect = useSynthStore(s => s.removeEffect);
  const [open, setOpen] = useState(false);

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
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 4 }}>
        <span style={{ fontSize: 10, color: 'var(--accent-purple)' }}>FX — Track {trackIndex + 1}</span>
        <button className="utai-btn" onClick={() => setOpen(false)} style={{ fontSize: 9, padding: '0 4px' }}>x</button>
      </div>
      {effects.map((fx, i) => (
        <div key={i} style={{ fontSize: 9, display: 'flex', justifyContent: 'space-between', padding: '1px 0' }}>
          <span style={{ color: fx.enabled ? 'var(--accent-cyan)' : 'var(--text-dim)' }}>
            {RENDER_EFFECTS[fx.type as RenderEffectType]?.label || fx.type}
          </span>
          <button style={{ background: 'none', border: 'none', color: 'var(--accent-pink)', cursor: 'pointer', fontSize: 9 }}
            onClick={() => removeEffect(trackIndex, i)}>x</button>
        </div>
      ))}
      <select
        onChange={e => { if (e.target.value) { addEffect(trackIndex, createDefaultEffect(e.target.value as RenderEffectType)); e.target.value = ''; } }}
        defaultValue=""
        style={{ width: '100%', marginTop: 2, background: 'var(--bg-deep)', color: 'var(--text-muted)', border: '1px solid var(--border-dim)', fontSize: 9, padding: '1px 2px' }}
      >
        <option value="">+ Add</option>
        {QUICK_EFFECTS.map(t => (
          <option key={t} value={t}>{RENDER_EFFECTS[t].label}</option>
        ))}
      </select>
    </div>
  );
}
