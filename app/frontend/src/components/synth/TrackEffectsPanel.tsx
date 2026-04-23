/** Track effects chain editor — add/remove/configure render-time effects per track.
 *  Supports scope filtering: Track (whole track) vs Clip (per voice part).
 */

import { useState, useEffect } from 'react';
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
  const [scopeFilter, setScopeFilter] = useState<'track' | string>('track');
  const setClipVolume = useSynthStore(s => s.setClipVolume);
  const setClipPan = useSynthStore(s => s.setClipPan);
  const pushUndoSnapshot = useSynthStore(s => s.pushUndoSnapshot);

  const viewMode = useSynthStore(s => s.viewMode);
  const selectedTrackIndex = useSynthStore(s => s.selectedTrackIndex);
  const activeWaveIdx = useSynthStore(s => s.activeWavePartIndex);
  const selectedSegment = useSynthStore(s => s.selectedSegment);

  let trackIdx = 0;
  if (viewMode === 'arrangement') {
    trackIdx = selectedTrackIndex;
  } else if (activeWaveIdx >= 0) {
    const wp = project.wave_parts[activeWaveIdx];
    if (wp) trackIdx = wp.track_no;
  } else if (activeInstIdx >= 0) {
    const ip = project.instrument_parts[activeInstIdx];
    if (ip) trackIdx = ip.track_no;
  } else {
    const vp = project.voice_parts[activePartIndex];
    if (vp) trackIdx = vp.track_no;
  }

  // Auto-sync scope filter when segment selection changes
  useEffect(() => {
    if (selectedSegment) {
      if (selectedSegment.type === 'voice') setScopeFilter(`v${selectedSegment.index}`);
      else if (selectedSegment.type === 'wave') setScopeFilter(`w${selectedSegment.index}`);
      else setScopeFilter('all');
    } else if (viewMode === 'arrangement') {
      setScopeFilter('track');
    }
  }, [selectedSegment, viewMode]);

  // Auto-sync when entering audio editor via double-click
  useEffect(() => {
    if (activeWaveIdx >= 0) setScopeFilter(`w${activeWaveIdx}`);
  }, [activeWaveIdx]);

  const allEffects = trackEffects[trackIdx] || [];
  const track = project.tracks[trackIdx];
  const setTrackVolume = useSynthStore(s => s.setTrackVolume);
  const setTrackPan = useSynthStore(s => s.setTrackPan);

  // Parts on this track (for clip scope options) — voice + wave parts
  const voicePartsOnTrack = project.voice_parts
    .map((vp, i) => ({ name: vp.name, _key: `v${i}`, _label: vp.name || `Clip ${i + 1}` }))
    .filter((_, i) => project.voice_parts[i].track_no === trackIdx);
  const wavePartsOnTrack = project.wave_parts
    .map((wp, i) => ({ name: wp.name, _key: `w${i}`, _label: wp.name || `Audio ${i + 1}` }))
    .filter((_, i) => project.wave_parts[i].track_no === trackIdx);
  const partsOnTrack = [...voicePartsOnTrack, ...wavePartsOnTrack];

  // Match scope: support legacy number (voice idx), 'v0', 'w0' string keys
  const matchScope = (fxScope: 'track' | number | string | undefined, filter: string): boolean => {
    const s = fxScope ?? 'track';
    if (filter === 'track') return s === 'track';
    // Legacy number scope = voice part index
    if (typeof s === 'number') return filter === `v${s}`;
    return s === filter;
  };

  const filteredEffects = allEffects.map((fx, i) => ({ fx, idx: i })).filter(({ fx }) =>
    matchScope(fx.scope, scopeFilter)
  );

  const handleAdd = (type: RenderEffectType) => {
    const scope = scopeFilter === 'all' ? 'track' : scopeFilter;
    addEffect(trackIdx, createDefaultEffect(type, scope));
    setAddMenuOpen(false);
  };

  const handleParamChange = (fxIdx: number, param: string, value: number) => {
    const fx = allEffects[fxIdx];
    if (!fx) return;
    updateEffectParams(trackIdx, fxIdx, { ...fx.params, [param]: value });
  };

  const tabStyle = (active: boolean) => ({
    background: 'none', border: `1px solid ${active ? 'var(--accent-cyan)' : 'var(--border-dim)'}`,
    color: active ? 'var(--accent-cyan)' : 'var(--text-dim)',
    fontSize: 9, padding: '1px 6px', cursor: 'pointer', fontFamily: 'var(--font-pixel)',
  });

  return (
    <div style={{ padding: '6px 10px' }}>
      {/* Scope filter tabs */}
      <div style={{ display: 'flex', gap: 3, flexWrap: 'wrap', marginBottom: 6 }}>
        <button style={tabStyle(scopeFilter === 'track')} onClick={() => setScopeFilter('track')}>Track</button>
        {partsOnTrack.map(p => (
            <button key={p._key} style={{ ...tabStyle(scopeFilter === p._key), maxWidth: 80, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap', display: 'inline-block' }}
              onClick={() => setScopeFilter(p._key)} title={p._label}>
              {p._label}
            </button>
        ))}
      </div>

      {/* Vol / Pan — track-level when scope=track, clip-level when clip selected */}
      {(() => {
        const isClip = scopeFilter !== 'track';
        const clipType = isClip ? (scopeFilter.startsWith('w') ? 'wave' : scopeFilter.startsWith('v') ? 'voice' : 'instrument') as 'voice' | 'instrument' | 'wave' : null;
        const clipIdx = isClip ? parseInt(scopeFilter.slice(1)) : -1;
        const clipPart = clipType === 'voice' ? project.voice_parts[clipIdx]
          : clipType === 'wave' ? project.wave_parts[clipIdx]
          : clipType === 'instrument' ? project.instrument_parts[clipIdx] : null;
        const vol = isClip ? ((clipPart as any)?.volume ?? 0) : (track?.volume ?? 0);
        const pan = isClip ? ((clipPart as any)?.pan ?? 0) : (track?.pan ?? 0);
        const volDisplay = vol <= -24 ? '-∞' : `${vol > 0 ? '+' : ''}${vol.toFixed(1)} dB`;
        const panDisplay = pan === 0 ? 'C' : pan < 0 ? `L${-pan}` : `R${pan}`;
        const onVolChange = (v: number) => isClip && clipType ? setClipVolume(clipType, clipIdx, v) : setTrackVolume(trackIdx, v);
        const onPanChange = (v: number) => isClip && clipType ? setClipPan(clipType, clipIdx, Math.round(v)) : setTrackPan(trackIdx, Math.round(v));
        const onReset = () => { onVolChange(0); onPanChange(0); };
        return (
          <div style={{ marginBottom: 8, borderBottom: '1px solid var(--border-dim)', paddingBottom: 6 }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 4 }}>
              <span style={{ fontSize: 10, color: 'var(--text-dim)', fontFamily: 'var(--font-pixel)' }}>{isClip ? 'Clip' : 'Track'} Vol / Pan</span>
              <button onClick={onReset} title="Reset" style={{ background: 'none', border: '1px solid var(--border-dim)', color: 'var(--text-dim)', fontSize: 8, padding: '0 4px', cursor: 'pointer', lineHeight: '14px', fontFamily: 'var(--font-mono)' }}>R</button>
            </div>
            <div className="utai-synth-slider" style={{ marginBottom: 4 }}>
              <div className="utai-synth-slider__label" style={{ fontSize: 12, display: 'flex', justifyContent: 'space-between' }}>
                <span style={{ color: 'var(--text-dim)', fontFamily: 'var(--font-pixel)' }}>Vol</span>
                <span style={{ color: '#c8c8e8', fontFamily: 'var(--font-mono)', fontSize: 13 }}>{volDisplay}</span>
              </div>
              <input type="range" min={-24} max={20} step={0.1} value={vol} onMouseDown={pushUndoSnapshot} onChange={e => onVolChange(parseFloat(e.target.value))} />
            </div>
            <div className="utai-synth-slider">
              <div className="utai-synth-slider__label" style={{ fontSize: 12, display: 'flex', justifyContent: 'space-between' }}>
                <span style={{ color: 'var(--text-dim)', fontFamily: 'var(--font-pixel)' }}>Pan</span>
                <span style={{ color: '#c8c8e8', fontFamily: 'var(--font-mono)', fontSize: 13 }}>{panDisplay}</span>
              </div>
              <input type="range" min={-100} max={100} step={1} value={pan} onMouseDown={pushUndoSnapshot} onChange={e => onPanChange(parseInt(e.target.value))} />
            </div>
          </div>
        );
      })()}

      {/* Effects header + add button */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 6 }}>
        <span style={{ fontFamily: 'var(--font-pixel)', fontSize: 11, color: 'var(--accent-cyan)' }}>
          {t('synth.effects')}
          <span style={{ fontSize: 9, color: 'var(--text-dim)', marginLeft: 4 }}>
            ({filteredEffects.length}/{allEffects.length})
          </span>
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

      {filteredEffects.length === 0 && (
        <div style={{ fontSize: 9, color: 'var(--text-dim)', textAlign: 'center', padding: 8 }}>
          {t('synth.noEffects')}
        </div>
      )}

      {filteredEffects.map(({ fx, idx: fxIdx }) => {
        const def = RENDER_EFFECTS[fx.type as RenderEffectType];
        if (!def) return null;
        const rawScopeLabel = fx.scope === 'track' || fx.scope == null ? 'T'
          : typeof fx.scope === 'number' ? (project.voice_parts[fx.scope]?.name || `C${fx.scope + 1}`)
          : typeof fx.scope === 'string' && fx.scope.startsWith('w')
            ? (project.wave_parts[parseInt(fx.scope.slice(1))]?.name || `A${parseInt(fx.scope.slice(1)) + 1}`)
            : typeof fx.scope === 'string' && fx.scope.startsWith('v')
              ? (project.voice_parts[parseInt(fx.scope.slice(1))]?.name || `C${parseInt(fx.scope.slice(1)) + 1}`)
              : 'T';
        const scopeLabel = rawScopeLabel.length > 6 ? rawScopeLabel.slice(0, 5) + '…' : rawScopeLabel;
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
                <span style={{ fontSize: 8, color: 'var(--text-muted)', marginLeft: 4, padding: '0 3px', border: '1px solid var(--border-dim)' }}>
                  {scopeLabel}
                </span>
              </span>
              <div style={{ display: 'flex', gap: 4 }}>
                <button
                  className="utai-btn"
                  style={{ fontSize: 8, padding: '0 4px', borderColor: 'var(--border-dim)', color: fx.enabled ? 'var(--accent-green)' : 'var(--text-dim)' }}
                  onClick={() => { pushUndoSnapshot(); updateEffectParams(trackIdx, fxIdx, { _enabled: fx.enabled ? 0 : 1 }); }}
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
                <span style={{ fontSize: 10, color: 'var(--text-dim)', width: 55, flexShrink: 0, fontFamily: 'var(--font-pixel)' }}>{paramDef.label}</span>
                <input
                  type="range"
                  min={paramDef.min}
                  max={paramDef.max}
                  step={paramDef.step}
                  value={fx.params[paramKey] ?? paramDef.default}
                  onMouseDown={pushUndoSnapshot}
                  onChange={e => handleParamChange(fxIdx, paramKey, parseFloat(e.target.value))}
                  style={{ flex: 1, height: 14 }}
                />
                <span style={{ fontSize: 12, color: '#c8c8e8', width: 44, textAlign: 'right', fontFamily: 'var(--font-mono)', letterSpacing: -0.5 }}>
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
