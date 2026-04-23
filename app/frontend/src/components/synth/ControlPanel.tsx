import React, { useState, useEffect } from 'react';
import { useTranslation } from 'react-i18next';
import { useSynthStore } from '../../store/synthStore';
import ProjectInfo from './ProjectInfo';
import RVCModelSelector from './RVCModelSelector';
import RenderButton from './RenderButton';
import VSTPluginSelector from './VSTPluginSelector';
import TrackEffectsPanel from './TrackEffectsPanel';
import AudioPreview from './AudioPreview';
import LogViewer from './LogViewer';
import G2pTool from './G2pTool';

function LanguageSelector() {
  const { t } = useTranslation();
  const language = useSynthStore(s => s.language);
  const setLanguage = useSynthStore(s => s.setLanguage);
  return (
    <div className="utai-synth-section">
      <div className="utai-synth-section__title">{t('synth.language', 'Language')}</div>
      <select value={language} onChange={e => setLanguage(e.target.value as 'zh' | 'ja')}>
        <option value="zh">中文 Chinese</option>
        <option value="ja">日本語 Japanese</option>
      </select>
    </div>
  );
}

function PitchShiftControl() {
  const { t } = useTranslation();
  const settings = useSynthStore(s => s.rvcSettings);
  const setRvcSettings = useSynthStore(s => s.setRvcSettings);
  return (
    <div className="utai-synth-section">
      <div className="utai-synth-slider">
        <div className="utai-synth-slider__label">
          {t('synth.pitchShift', 'Pitch')} <span>{settings.f0_up_key > 0 ? '+' : ''}{settings.f0_up_key}</span>
        </div>
        <input
          type="range" min={-12} max={12} step={1}
          value={settings.f0_up_key}
          onChange={e => setRvcSettings({ f0_up_key: Number(e.target.value) })}
        />
      </div>
    </div>
  );
}

function MuteStrategySelector() {
  const { t } = useTranslation();
  const muteStrategy = useSynthStore(s => s.muteStrategy);
  const setMuteStrategy = useSynthStore(s => s.setMuteStrategy);
  return (
    <div className="utai-synth-section">
      <div className="utai-synth-section__title">{t('synth.muteStrategy', 'Mute')}</div>
      <select value={muteStrategy} onChange={e => setMuteStrategy(e.target.value as any)}>
        <option value="auto">Auto</option>
        <option value="raw">Raw</option>
        <option value="inline">Inline</option>
        <option value="both">Both</option>
      </select>
    </div>
  );
}

function GlobalPitchOffset() {
  const { t } = useTranslation();
  const offset = useSynthStore(s => s.globalF0Offset);
  const setOffset = useSynthStore(s => s.setGlobalF0Offset);
  return (
    <div className="utai-synth-section">
      <div className="utai-synth-slider">
        <div className="utai-synth-slider__label">
          {t('synth.globalPitch', 'Global Pitch')} <span>{offset > 0 ? '+' : ''}{offset}</span>
        </div>
        <input
          type="range" min={-24} max={24} step={1}
          value={offset}
          onChange={e => setOffset(Number(e.target.value))}
        />
      </div>
    </div>
  );
}

function VocalTrackInspector() {
  const project = useSynthStore(s => s.project);
  const idx = useSynthStore(s => s.selectedTrackIndex);
  const track = project.tracks[idx];

  const trackLanguage = useSynthStore(s => s.trackLanguage);
  const language = useSynthStore(s => s.language);
  const setTrackLanguage = useSynthStore(s => s.setTrackLanguage);
  const trackMuteStrategy = useSynthStore(s => s.trackMuteStrategy);
  const setTrackMuteStrategy = useSynthStore(s => s.setTrackMuteStrategy);
  const trackIndexRate = useSynthStore(s => s.trackIndexRate);
  const defaultIndexRate = useSynthStore(s => s.rvcSettings.index_rate);
  const setTrackIndexRate = useSynthStore(s => s.setTrackIndexRate);
  const trackF0Offset = useSynthStore(s => s.trackF0Offset);
  const setTrackF0Offset = useSynthStore(s => s.setTrackF0Offset);

  if (!track || track.track_type !== 'vocal' || idx < 0) return null;

  const selStyle: React.CSSProperties = {
    flex: 1, fontSize: 10, fontFamily: 'var(--font-pixel)',
    color: '#b0b0d0', background: 'var(--bg-deep)', border: '1px solid var(--border-dim)',
    borderRadius: 3, padding: '2px 4px', height: 20, outline: 'none', cursor: 'pointer',
  };
  const lbl: React.CSSProperties = {
    fontSize: 9, fontFamily: 'var(--font-pixel)', color: 'var(--text-dim)',
    width: 30, flexShrink: 0, textAlign: 'right',
  };
  const row: React.CSSProperties = { display: 'flex', alignItems: 'center', gap: 6 };

  const idxRate = trackIndexRate[idx] ?? defaultIndexRate;
  const f0Off = trackF0Offset[idx] ?? 0;

  return (
    <div style={{ padding: '6px 10px', borderBottom: '1px solid var(--border-dim)', display: 'flex', flexDirection: 'column', gap: 4 }}>
      <div style={row}>
        <span style={lbl}>Lang</span>
        <select value={trackLanguage[idx] || language} onChange={e => setTrackLanguage(idx, e.target.value)} style={selStyle}>
          <option value="zh">中文 Chinese</option>
          <option value="ja">日本語 Japanese</option>
        </select>
      </div>
      <div style={row}>
        <span style={lbl}>Mute</span>
        <select value={trackMuteStrategy[idx] || 'auto'} onChange={e => setTrackMuteStrategy(idx, e.target.value)} style={selStyle}>
          <option value="auto">Auto</option>
          <option value="raw">Raw</option>
          <option value="inline">Inline</option>
          <option value="both">Both</option>
        </select>
      </div>
      <div style={row}>
        <span style={lbl}>Idx</span>
        <input type="range" min={0} max={1} step={0.05} value={idxRate}
          onChange={e => setTrackIndexRate(idx, Number(e.target.value))}
          style={{ flex: 1 }} />
        <span style={{ fontSize: 9, fontFamily: 'var(--font-mono)', color: '#b0b0d0', width: 28, textAlign: 'right' }}>
          {(idxRate * 100).toFixed(0)}%
        </span>
      </div>
      <div style={row}>
        <span style={lbl}>Key</span>
        <input type="range" min={-24} max={24} step={1} value={f0Off}
          onChange={e => setTrackF0Offset(idx, Number(e.target.value))}
          style={{ flex: 1 }} />
        <span style={{ fontSize: 9, fontFamily: 'var(--font-mono)', color: '#b0b0d0', width: 28, textAlign: 'right' }}>
          {f0Off > 0 ? '+' : ''}{f0Off}
        </span>
      </div>
    </div>
  );
}

function SelectedTrackLabel() {
  const project = useSynthStore(s => s.project);
  const selectedTrackIndex = useSynthStore(s => s.selectedTrackIndex);
  const renameTrack = useSynthStore(s => s.renameTrack);
  const [editing, setEditing] = useState(false);
  useEffect(() => { setEditing(false); }, [selectedTrackIndex]);
  const track = project.tracks[selectedTrackIndex];
  if (!track) return null;
  return (
    <div style={{ padding: '4px 10px', fontSize: 11, fontFamily: 'var(--font-pixel)', color: 'var(--accent-cyan)', borderBottom: '1px solid var(--border-dim)', display: 'flex', alignItems: 'center', gap: 4 }}>
      {editing ? (
        <input autoFocus defaultValue={track.track_name}
          onKeyDown={e => {
            if (e.key === 'Enter') { const v = (e.target as HTMLInputElement).value.trim(); if (v) renameTrack(selectedTrackIndex, v); setEditing(false); }
            else if (e.key === 'Escape') setEditing(false);
          }}
          onBlur={e => { const v = e.target.value.trim(); if (v) renameTrack(selectedTrackIndex, v); setEditing(false); }}
          style={{ flex: 1, fontSize: 11, fontFamily: 'var(--font-pixel)', color: 'var(--accent-cyan)', background: 'var(--bg-deep)', border: '1px solid var(--accent-cyan)', outline: 'none', padding: '0 4px' }}
        />
      ) : (
        <>
          <button onClick={() => setEditing(true)} title="Rename" style={{ background: 'none', border: '1px solid var(--border-dim)', color: 'var(--text-dim)', fontSize: 8, padding: '0 4px', cursor: 'pointer', lineHeight: '14px', fontFamily: 'var(--font-mono)' }}>✎</button>
          <span style={{ flex: 1 }}>Track {selectedTrackIndex + 1}: {track.track_name}</span>
        </>
      )}
    </div>
  );
}

export default function ControlPanel() {
  const { t } = useTranslation();
  const finalUrl = useSynthStore(s => s.finalUrl);
  const project = useSynthStore(s => s.project);
  const viewMode = useSynthStore(s => s.viewMode);
  const activePartIndex = useSynthStore(s => s.activePartIndex);
  const activeInstPartIndex = useSynthStore(s => s.activeInstrumentPartIndex);
  const activeWavePartIndex = useSynthStore(s => s.activeWavePartIndex);

  const isArrangement = viewMode === 'arrangement';
  const isAudioMode = activeWavePartIndex >= 0;

  let activeTrackType: 'vocal' | 'audio' | 'instrument' = 'vocal';
  if (isAudioMode) {
    activeTrackType = 'audio';
  } else if (activeInstPartIndex >= 0) {
    activeTrackType = 'instrument';
  } else {
    const vp = project.voice_parts[activePartIndex];
    const track = vp ? project.tracks[vp.track_no] : null;
    activeTrackType = (track?.track_type as any) || 'vocal';
  }

  return (
    <div className="utai-synth-control">
      {isArrangement && <RenderButton renderAll />}
      {isArrangement && <GlobalPitchOffset />}

      <ProjectInfo />

      {isArrangement ? (
        <>
          <SelectedTrackLabel />
          <VocalTrackInspector />
          <TrackEffectsPanel />
        </>
      ) : (
        <>
          {activeTrackType === 'vocal' && (
            <>
              <G2pTool />
              <LanguageSelector />
              <RVCModelSelector />
              <PitchShiftControl />
              <MuteStrategySelector />
            </>
          )}

          {activeTrackType === 'instrument' && (
            <VSTPluginSelector />
          )}

          {activeTrackType === 'audio' && null /* effects only, shown below */}

          <TrackEffectsPanel />
        </>
      )}

      {!isArrangement && (
        <>
          <hr style={{ border: 'none', borderTop: '1px dashed var(--border-dim)' }} />
          <RenderButton />
          <hr style={{ border: 'none', borderTop: '1px dashed var(--border-dim)' }} />
          {activeTrackType === 'vocal' && finalUrl && (
            <AudioPreview label={t('synth.audioFinal', 'Output')} audioUrl={finalUrl} />
          )}
        </>
      )}

      <LogViewer />
    </div>
  );
}
