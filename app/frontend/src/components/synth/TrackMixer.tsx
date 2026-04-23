/** Track mixer — volume and pan per track, overlays on arrangement header area. */

import { useState, useRef, useEffect } from 'react';
import { useSynthStore } from '../../store/synthStore';
import { RULER_H, TRACK_LANE_H, TRACK_HEADER_W } from '../../lib/synth/arrangementEngine';
import { RESOLUTION } from '../../types/ustx';
import * as engine from '../../lib/synth/audioEngine';
import { getRVCModels } from '../../api/client';

const COLORS = ['#ff6b9d', '#5ce0d8', '#e8b84b', '#9b6dff', '#5dba5d', '#e8944b'];

function MixerSlider({ value, min, max, step, onChange, onDragStart, color, label, displayValue }: {
  value: number; min: number; max: number; step: number;
  onChange: (v: number) => void; onDragStart?: () => void; color: string; label: string; displayValue: string;
}) {
  const [dragging, setDragging] = useState(false);
  const pct = ((value - min) / (max - min)) * 100;
  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: 2 }}>
      <span style={{ fontSize: 8, color: '#5a5a7a', width: 18, flexShrink: 0 }}>{label}</span>
      <div style={{ flex: 1, position: 'relative', height: 10 }}>
        <div style={{ position: 'absolute', top: 4, left: 0, right: 0, height: 2, background: '#2a2a4a', borderRadius: 1 }} />
        <div style={{
          position: 'absolute', top: 4, left: 0,
          width: `${pct}%`, height: 2, background: color, borderRadius: 1, opacity: 0.6,
        }} />
        <div style={{
          position: 'absolute', top: 1,
          left: `calc(${pct}% - 3px)`,
          width: 6, height: 8, background: color, borderRadius: 2,
          boxShadow: `0 0 4px ${color}`, pointerEvents: 'none',
        }} />
        {dragging && (
          <div style={{
            position: 'absolute', bottom: 14, left: `${pct}%`, transform: 'translateX(-50%)',
            background: '#222244', border: `1px solid ${color}`, borderRadius: 3,
            padding: '1px 5px', fontSize: 11, fontFamily: 'var(--font-mono)',
            color, whiteSpace: 'nowrap', zIndex: 20, pointerEvents: 'none',
          }}>
            {displayValue}
          </div>
        )}
        <input type="range" min={min} max={max} step={step}
          value={value}
          onChange={e => onChange(parseFloat(e.target.value))}
          onMouseDown={() => { setDragging(true); onDragStart?.(); }}
          onMouseUp={() => setDragging(false)}
          onBlur={() => setDragging(false)}
          style={{ position: 'absolute', top: -2, left: 0, width: '100%', height: 14, opacity: 0, cursor: 'ew-resize' }}
        />
      </div>
      <span style={{ fontSize: 12, color: '#b0b0d0', width: 36, textAlign: 'right', fontFamily: 'var(--font-mono)', flexShrink: 0 }}>
        {displayValue}
      </span>
    </div>
  );
}

export default function TrackMixer() {
  const project = useSynthStore(s => s.project);
  const setTrackVolume = useSynthStore(s => s.setTrackVolume);
  const setTrackPan = useSynthStore(s => s.setTrackPan);
  const pushUndoSnapshot = useSynthStore(s => s.pushUndoSnapshot);
  const setTrackMute = useSynthStore(s => s.setTrackMute);
  const setTrackSolo = useSynthStore(s => s.setTrackSolo);
  const selectedTrackIndex = useSynthStore(s => s.selectedTrackIndex);
  const setSelectedTrackIndex = useSynthStore(s => s.setSelectedTrackIndex);
  const setSelectedSegment = useSynthStore(s => s.setSelectedSegment);
  const renameTrack = useSynthStore(s => s.renameTrack);
  const reorderTrack = useSynthStore(s => s.reorderTrack);
  const trackLaneHeight = useSynthStore(s => s.viewState.trackLaneHeight) ?? TRACK_LANE_H;
  const scrollY = useSynthStore(s => s.arrScrollY);
  const trackRvcModel = useSynthStore(s => s.trackRvcModel);
  const rvcModel = useSynthStore(s => s.rvcModel);
  const setTrackRvcModel = useSynthStore(s => s.setTrackRvcModel);
  const trackLanguage = useSynthStore(s => s.trackLanguage);
  const language = useSynthStore(s => s.language);
  const setTrackLanguage = useSynthStore(s => s.setTrackLanguage);
  const trackMuteStrategy = useSynthStore(s => s.trackMuteStrategy);
  const muteStrategy = useSynthStore(s => s.muteStrategy);
  const setTrackMuteStrategy = useSynthStore(s => s.setTrackMuteStrategy);
  const trackIndexRate = useSynthStore(s => s.trackIndexRate);
  const defaultIndexRate = useSynthStore(s => s.rvcSettings.index_rate);
  const setTrackIndexRate = useSynthStore(s => s.setTrackIndexRate);
  const trackF0Offset = useSynthStore(s => s.trackF0Offset);
  const setTrackF0Offset = useSynthStore(s => s.setTrackF0Offset);
  const rvcListVersion = useSynthStore(s => s.rvcModelListVersion);
  const [editingIdx, setEditingIdx] = useState<number | null>(null);
  const dragFrom = useRef(-1);
  const [rvcModels, setRvcModels] = useState<{ name: string; filename: string }[]>([]);

  useEffect(() => {
    getRVCModels().then(data => {
      setRvcModels((data || []).map((m: any) => ({
        name: m.name, filename: m.filename || m.name + '.pth',
      })));
    }).catch(() => setRvcModels([]));
  }, [rvcListVersion]);

  useEffect(() => {
    const up = () => {
      dragFrom.current = -1;
      document.querySelectorAll<HTMLElement>('[data-track-header]').forEach(el => {
        el.style.cursor = 'move'; el.style.opacity = '1'; el.style.boxShadow = '';
      });
    };
    window.addEventListener('mouseup', up);
    return () => window.removeEventListener('mouseup', up);
  }, []);

  return (
    <div style={{
      position: 'absolute', left: 0, top: RULER_H,
      width: TRACK_HEADER_W, bottom: 0, zIndex: 5,
      background: '#111125', overflow: 'hidden',
    }} onWheel={e => {
      e.preventDefault();
      // Check if Tab is held for vertical zoom (track height)
      const canvas = e.currentTarget.parentElement?.querySelector('canvas');
      if (canvas) {
        canvas.dispatchEvent(new WheelEvent('wheel', {
          deltaX: e.deltaX, deltaY: e.deltaY,
          ctrlKey: e.ctrlKey, shiftKey: e.shiftKey, altKey: e.altKey,
          clientX: e.clientX, clientY: e.clientY,
          bubbles: true, cancelable: true,
        }));
      }
    }} onContextMenu={e => {
      // Let right-click pass through to canvas for context menu
      e.preventDefault();
      const canvas = e.currentTarget.parentElement?.querySelector('canvas');
      if (canvas) canvas.dispatchEvent(new MouseEvent('contextmenu', {
        clientX: e.clientX, clientY: e.clientY, bubbles: true,
      }));
    }}>
      <div style={{ marginTop: -scrollY }}>
      {project.tracks.map((track, idx) => {
        const c = COLORS[idx % COLORS.length];
        const tt = track.track_type || 'vocal';
        const typeLabel = tt === 'vocal' ? 'Vocal'
          : tt === 'instrument' ? 'Inst' : 'Audio';

        const isSelected = idx === selectedTrackIndex;
        const volDisplay = track.volume <= -24 ? '-∞' : `${track.volume > 0 ? '+' : ''}${track.volume.toFixed(0)}dB`;
        const panDisplay = track.pan === 0 ? 'C' : track.pan < 0 ? `L${-track.pan}` : `R${track.pan}`;

        return (
          <div key={idx} data-track-header onClick={() => {
              const s = useSynthStore.getState();
              const wasAudio = s.project.tracks[s.selectedTrackIndex]?.track_type === 'audio';
              const nowAudio = track.track_type === 'audio';
              setSelectedTrackIndex(idx);
              setSelectedSegment(null);
              if (wasAudio && !nowAudio) {
                const ts = s.project.time_signatures[0];
                const ticksPerBar = RESOLUTION * (ts?.beat_per_bar ?? 4);
                const snapped = Math.round(s.playbackPositionTicks / ticksPerBar) * ticksPerBar;
                useSynthStore.getState().setPlaybackPosition(snapped);
                engine.seek(engine.ticksToSeconds(snapped, s.project.tempos));
              }
            }}
            onMouseDown={e => {
              if (e.button === 0 && editingIdx == null) {
                dragFrom.current = idx;
                (e.currentTarget as HTMLElement).style.cursor = 'grabbing';
                (e.currentTarget as HTMLElement).style.opacity = '0.7';
                (e.currentTarget as HTMLElement).style.boxShadow = '0 0 8px rgba(92,224,216,0.3)';
              }
            }}
            onMouseUp={e => {
              (e.currentTarget as HTMLElement).style.cursor = 'move';
              (e.currentTarget as HTMLElement).style.opacity = '1';
              (e.currentTarget as HTMLElement).style.boxShadow = '';
            }}
            onMouseEnter={() => {
              if (editingIdx != null) return;
              if (dragFrom.current >= 0 && dragFrom.current !== idx) {
                reorderTrack(dragFrom.current, idx);
                dragFrom.current = idx;
              }
            }}
            style={{
            height: trackLaneHeight,
            borderBottom: '1px solid #1e1e3a',
            padding: '3px 4px 3px 7px',
            cursor: 'move',
            display: 'flex', flexDirection: 'column', justifyContent: 'space-between',
            position: 'relative', overflow: 'hidden',
            background: isSelected ? 'rgba(92, 224, 216, 0.06)' : undefined,
          }}>
            {/* Color stripe */}
            <div style={{ position: 'absolute', left: 0, top: 0, bottom: 0, width: 3, background: c }} />

            {/* Row 1: name + type + M/S */}
            <div style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
              {editingIdx === idx ? (
                <input autoFocus defaultValue={track.track_name}
                  onClick={e => e.stopPropagation()}
                  onKeyDown={e => {
                    if (e.key === 'Enter') { const v = (e.target as HTMLInputElement).value.trim(); if (v) renameTrack(idx, v); setEditingIdx(null); }
                    else if (e.key === 'Escape') setEditingIdx(null);
                  }}
                  onBlur={e => { const v = e.target.value.trim(); if (v) renameTrack(idx, v); setEditingIdx(null); }}
                  style={{
                    fontSize: 10, color: '#d0d0e8', fontFamily: 'var(--font-pixel)',
                    flex: 1, width: 0, background: 'var(--bg-deep)',
                    border: '1px solid var(--accent-cyan)', outline: 'none', padding: '0 2px',
                  }}
                />
              ) : (
                <span onDoubleClick={e => { e.stopPropagation(); setEditingIdx(idx); }} style={{
                  fontSize: 10, color: '#d0d0e8', fontFamily: 'var(--font-pixel)',
                  flex: 1, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap',
                }}>
                  {track.track_name}
                </span>
              )}
              <span style={{ fontSize: 8, color: '#5a5a7a' }}>{typeLabel}</span>
              <button onClick={e => { e.stopPropagation(); setTrackMute(idx, !track.mute); }} style={{
                background: track.mute ? 'rgba(255,68,68,0.2)' : 'none',
                border: `1px solid ${track.mute ? '#ff4444' : '#3a3a5a'}`,
                color: track.mute ? '#ff4444' : '#5a5a7a',
                fontSize: 7, padding: '0 3px', cursor: 'pointer', lineHeight: '14px',
              }}>M</button>
              <button onClick={e => { e.stopPropagation(); setTrackSolo(idx, !track.solo); }} style={{
                background: track.solo ? 'rgba(255,204,0,0.2)' : 'none',
                border: `1px solid ${track.solo ? '#ffcc00' : '#3a3a5a'}`,
                color: track.solo ? '#ffcc00' : '#5a5a7a',
                fontSize: 7, padding: '0 3px', cursor: 'pointer', lineHeight: '14px',
              }}>S</button>
            </div>

            {/* Row 2: RVC model + Mute strategy (vocal tracks only) */}
            {tt === 'vocal' && trackLaneHeight >= 56 && (
              <div style={{ display: 'flex', alignItems: 'center', gap: 2 }}>
                <select
                  value={trackRvcModel[idx] || ''}
                  onClick={e => e.stopPropagation()}
                  onChange={e => { e.stopPropagation(); setTrackRvcModel(idx, e.target.value); }}
                  title="RVC Model"
                  style={{
                    flex: 1, width: 0,
                    fontSize: 8, fontFamily: 'var(--font-pixel)',
                    color: trackRvcModel[idx] ? '#b0b0d0' : '#5a5a7a',
                    background: '#1a1a32', border: '1px solid #2a2a4a',
                    borderRadius: 2, padding: '0 2px', height: 14,
                    outline: 'none', cursor: 'pointer',
                  }}
                >
                  <option value="">RVC --</option>
                  {rvcModels.map(m => (
                    <option key={m.filename} value={m.filename}>{m.name}</option>
                  ))}
                </select>
                <select
                  value={trackMuteStrategy[idx] || 'auto'}
                  onClick={e => e.stopPropagation()}
                  onChange={e => { e.stopPropagation(); setTrackMuteStrategy(idx, e.target.value); }}
                  title="Mute Strategy"
                  style={{
                    width: 38, flexShrink: 0,
                    fontSize: 8, fontFamily: 'var(--font-pixel)',
                    color: '#b0b0d0',
                    background: '#1a1a32', border: '1px solid #2a2a4a',
                    borderRadius: 2, padding: '0 1px', height: 14,
                    outline: 'none', cursor: 'pointer',
                  }}
                >
                  <option value="auto">Auto</option>
                  <option value="raw">Raw</option>
                  <option value="inline">Inline</option>
                  <option value="both">Both</option>
                </select>
              </div>
            )}

            {/* Row 2b: Language (vocal tracks only) */}
            {tt === 'vocal' && trackLaneHeight >= 56 && (
              <div style={{ display: 'flex', alignItems: 'center', gap: 2 }}>
                <select
                  value={trackLanguage[idx] || language}
                  onClick={e => e.stopPropagation()}
                  onChange={e => { e.stopPropagation(); setTrackLanguage(idx, e.target.value); }}
                  title="Language"
                  style={{
                    flex: 1, width: 0,
                    fontSize: 8, fontFamily: 'var(--font-pixel)',
                    color: '#b0b0d0',
                    background: '#1a1a32', border: '1px solid #2a2a4a',
                    borderRadius: 2, padding: '0 2px', height: 14,
                    outline: 'none', cursor: 'pointer',
                  }}
                >
                  <option value="zh">中文</option>
                  <option value="ja">日本語</option>
                </select>
              </div>
            )}

            {/* Row 2c: Index Rate (vocal only) */}
            {tt === 'vocal' && trackLaneHeight >= 72 && (
              <MixerSlider
                value={trackIndexRate[idx] ?? defaultIndexRate} min={0} max={1} step={0.05}
                onChange={v => setTrackIndexRate(idx, v)}
                color={c} label="Idx" displayValue={`${((trackIndexRate[idx] ?? defaultIndexRate) * 100).toFixed(0)}%`}
              />
            )}
            {/* Row 2d: Pitch offset (vocal only) */}
            {tt === 'vocal' && trackLaneHeight >= 86 && (
              <MixerSlider
                value={trackF0Offset[idx] ?? 0} min={-24} max={24} step={1}
                onChange={v => setTrackF0Offset(idx, v)}
                color={c} label="Key" displayValue={`${(trackF0Offset[idx] ?? 0) > 0 ? '+' : ''}${trackF0Offset[idx] ?? 0}`}
              />
            )}

            {/* Row 3: Volume */}
            <MixerSlider
              value={track.volume} min={-24} max={20} step={0.1}
              onChange={v => setTrackVolume(idx, v)}
              onDragStart={pushUndoSnapshot}
              color={c} label="Vol" displayValue={volDisplay}
            />

            {/* Row 4: Pan — hide when track is too short */}
            {trackLaneHeight >= 52 && (
              <MixerSlider
                value={track.pan} min={-100} max={100} step={1}
                onChange={v => setTrackPan(idx, Math.round(v))}
                onDragStart={pushUndoSnapshot}
                color={c} label="Pan" displayValue={panDisplay}
              />
            )}
          </div>
        );
      })}
      </div>
    </div>
  );
}
