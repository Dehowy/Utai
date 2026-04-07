/** Track mixer — volume and pan per track, overlays on arrangement header area. */

import { useSynthStore } from '../../store/synthStore';
import { RULER_H, TRACK_LANE_H, TRACK_HEADER_W } from '../../lib/synth/arrangementEngine';

const COLORS = ['#ff6b9d', '#5ce0d8', '#e8b84b', '#9b6dff', '#5dba5d', '#e8944b'];

export default function TrackMixer() {
  const project = useSynthStore(s => s.project);
  const setTrackVolume = useSynthStore(s => s.setTrackVolume);
  const setTrackPan = useSynthStore(s => s.setTrackPan);
  const setTrackMute = useSynthStore(s => s.setTrackMute);
  const setTrackSolo = useSynthStore(s => s.setTrackSolo);

  return (
    <div style={{
      position: 'absolute', left: 0, top: RULER_H,
      width: TRACK_HEADER_W, bottom: 0, zIndex: 5,
      background: '#111125',
    }}>
      {project.tracks.map((track, idx) => {
        const c = COLORS[idx % COLORS.length];
        const tt = track.track_type || 'vocal';
        const typeLabel = tt === 'vocal' ? (track.singer || 'Vocal')
          : tt === 'instrument' ? (track.plugin_name || 'Inst') : 'Audio';

        return (
          <div key={idx} style={{
            height: TRACK_LANE_H,
            borderBottom: '1px solid #1e1e3a',
            padding: '4px 4px 4px 7px',
            display: 'flex', flexDirection: 'column', justifyContent: 'space-between',
            position: 'relative', overflow: 'hidden',
          }}>
            {/* Color stripe */}
            <div style={{ position: 'absolute', left: 0, top: 0, bottom: 0, width: 3, background: c }} />

            {/* Row 1: name + type + M/S */}
            <div style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
              <span style={{
                fontSize: 10, color: '#d0d0e8', fontFamily: 'var(--font-pixel)',
                flex: 1, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap',
              }}>
                {track.track_name}
              </span>
              <span style={{ fontSize: 8, color: '#5a5a7a' }}>{typeLabel}</span>
              <button onClick={() => setTrackMute(idx, !track.mute)} style={{
                background: track.mute ? 'rgba(255,68,68,0.2)' : 'none',
                border: `1px solid ${track.mute ? '#ff4444' : '#3a3a5a'}`,
                color: track.mute ? '#ff4444' : '#5a5a7a',
                fontSize: 7, padding: '0 3px', cursor: 'pointer', lineHeight: '14px',
              }}>M</button>
              <button onClick={() => setTrackSolo(idx, !track.solo)} style={{
                background: track.solo ? 'rgba(255,204,0,0.2)' : 'none',
                border: `1px solid ${track.solo ? '#ffcc00' : '#3a3a5a'}`,
                color: track.solo ? '#ffcc00' : '#5a5a7a',
                fontSize: 7, padding: '0 3px', cursor: 'pointer', lineHeight: '14px',
              }}>S</button>
            </div>

            {/* Row 2: Volume */}
            <div style={{ display: 'flex', alignItems: 'center', gap: 2 }}>
              <span style={{ fontSize: 7, color: '#4a4a6a', width: 16 }}>Vol</span>
              <div style={{ flex: 1, position: 'relative', height: 6 }}>
                <div style={{ position: 'absolute', top: 2, left: 0, right: 0, height: 2, background: '#2a2a4a', borderRadius: 1 }} />
                <div style={{
                  position: 'absolute', top: 2, left: 0,
                  width: `${Math.min(100, ((track.volume + 24) / 30) * 100)}%`,
                  height: 2, background: c, borderRadius: 1, opacity: 0.6,
                }} />
                <input type="range" min={-24} max={6} step={0.5}
                  value={track.volume}
                  onChange={e => setTrackVolume(idx, parseFloat(e.target.value))}
                  style={{ position: 'absolute', top: -4, left: 0, width: '100%', height: 14, opacity: 0, cursor: 'pointer' }}
                />
              </div>
              <span style={{ fontSize: 7, color: '#6a6a8a', width: 22, textAlign: 'right', fontFamily: 'var(--font-mono)' }}>
                {track.volume > 0 ? '+' : ''}{track.volume.toFixed(0)}dB
              </span>
            </div>

            {/* Row 3: Pan */}
            <div style={{ display: 'flex', alignItems: 'center', gap: 2 }}>
              <span style={{ fontSize: 7, color: '#4a4a6a', width: 16 }}>Pan</span>
              <div style={{ flex: 1, position: 'relative', height: 6 }}>
                <div style={{ position: 'absolute', top: 2, left: 0, right: 0, height: 2, background: '#2a2a4a', borderRadius: 1 }} />
                {/* Pan indicator from center */}
                <div style={{
                  position: 'absolute', top: 2,
                  left: track.pan >= 0 ? '50%' : `${50 + track.pan / 2}%`,
                  width: `${Math.abs(track.pan) / 2}%`,
                  height: 2, background: c, borderRadius: 1, opacity: 0.6,
                }} />
                <input type="range" min={-100} max={100} step={1}
                  value={track.pan}
                  onChange={e => setTrackPan(idx, parseInt(e.target.value))}
                  style={{ position: 'absolute', top: -4, left: 0, width: '100%', height: 14, opacity: 0, cursor: 'pointer' }}
                />
              </div>
              <span style={{ fontSize: 7, color: '#6a6a8a', width: 22, textAlign: 'right', fontFamily: 'var(--font-mono)' }}>
                {track.pan === 0 ? 'C' : track.pan < 0 ? `L${-track.pan}` : `R${track.pan}`}
              </span>
            </div>
          </div>
        );
      })}
    </div>
  );
}
