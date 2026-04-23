import { useState } from 'react';
import { useTranslation } from 'react-i18next';
import { useSynthStore } from '../../store/synthStore';

export default function ProjectInfo() {
  const { t } = useTranslation();
  const project = useSynthStore(s => s.project);
  const dirty = useSynthStore(s => s.dirty);
  const setTempo = useSynthStore(s => s.setTempo);
  const setTimeSignature = useSynthStore(s => s.setTimeSignature);
  const bpm = project.tempos[0]?.bpm ?? 120;
  const ts = project.time_signatures[0];
  const beatPerBar = ts?.beat_per_bar ?? 4;
  const beatUnit = ts?.beat_unit ?? 4;
  const [editBpm, setEditBpm] = useState<string | null>(null);

  const commitBpm = () => {
    if (editBpm !== null) {
      const v = parseInt(editBpm, 10);
      if (!isNaN(v) && v >= 20 && v <= 300) setTempo(v);
      setEditBpm(null);
    }
  };

  return (
    <div className="utai-synth-section" style={{ display: 'flex', gap: 8, alignItems: 'center', flexWrap: 'wrap' }}>
      <label style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
        BPM:
        <input
          type="text"
          inputMode="numeric"
          value={editBpm !== null ? editBpm : bpm}
          onFocus={() => setEditBpm(String(bpm))}
          onChange={e => setEditBpm(e.target.value.replace(/[^0-9]/g, ''))}
          onBlur={commitBpm}
          onKeyDown={e => {
            if (e.key === 'Enter') { commitBpm(); (e.target as HTMLInputElement).blur(); }
            else if (e.key === 'Escape') { setEditBpm(null); (e.target as HTMLInputElement).blur(); }
          }}
          className="utai-synth-no-spin"
          style={{ width: 45, textAlign: 'center' }}
        />
      </label>

      <span style={{ color: 'var(--text-muted)' }}>|</span>

      <label style={{ display: 'flex', alignItems: 'center', gap: 2 }}>
        <select
          value={beatPerBar}
          onChange={e => setTimeSignature(Number(e.target.value), beatUnit)}
          style={{ width: 36, textAlign: 'center', background: 'var(--bg-deep)', color: 'var(--text-primary)', border: '1px solid var(--border-mid)', fontFamily: 'var(--font-mono)', fontSize: 12, padding: '2px' }}
        >
          {[2, 3, 4, 5, 6, 7, 8, 9, 12].map(n => (
            <option key={n} value={n}>{n}</option>
          ))}
        </select>
        <span style={{ fontFamily: 'var(--font-mono)', color: 'var(--text-secondary)', fontSize: 14 }}>/</span>
        <select
          value={beatUnit}
          onChange={e => setTimeSignature(beatPerBar, Number(e.target.value))}
          style={{ width: 36, textAlign: 'center', background: 'var(--bg-deep)', color: 'var(--text-primary)', border: '1px solid var(--border-mid)', fontFamily: 'var(--font-mono)', fontSize: 12, padding: '2px' }}
        >
          {[2, 4, 8, 16].map(n => (
            <option key={n} value={n}>{n}</option>
          ))}
        </select>
      </label>

      {dirty && <span style={{ color: 'var(--accent-orange)', fontWeight: 'bold', fontFamily: 'var(--font-pixel)' }} title="Unsaved">*</span>}
    </div>
  );
}
