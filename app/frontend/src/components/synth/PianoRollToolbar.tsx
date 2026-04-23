import { useTranslation } from 'react-i18next';
import { useSynthStore, type Tool } from '../../store/synthStore';
import { RESOLUTION } from '../../types/ustx';
import { ticksToDurationLabel } from '../../lib/synth/pianoRollEngine';

export const SNAP_OPTIONS = [
  { label: '1/1', ticks: RESOLUTION * 4, group: 'std' },
  { label: '1/2', ticks: RESOLUTION * 2, group: 'std' },
  { label: '1/4', ticks: RESOLUTION, group: 'std' },
  { label: '1/8', ticks: RESOLUTION / 2, group: 'std' },
  { label: '1/16', ticks: RESOLUTION / 4, group: 'std' },
  { label: '1/32', ticks: RESOLUTION / 8, group: 'std' },
  { label: '1/2T', ticks: Math.round(RESOLUTION * 4 / 3), group: 'tri' },
  { label: '1/4T', ticks: Math.round(RESOLUTION * 2 / 3), group: 'tri' },
  { label: '1/8T', ticks: Math.round(RESOLUTION / 3), group: 'tri' },
  { label: '1/16T', ticks: Math.round(RESOLUTION / 6), group: 'tri' },
  { label: '1/32T', ticks: Math.round(RESOLUTION / 12), group: 'tri' },
];

// All atomic durations sorted ascending — used by Alt+scroll cycling
export const SNAP_ATOMS_ASC = [...SNAP_OPTIONS].sort((a, b) => a.ticks - b.ticks);

export default function PianoRollToolbar() {
  const { t } = useTranslation();
  const activeTool = useSynthStore(s => s.activeTool);
  const viewState = useSynthStore(s => s.viewState);
  const viewMode = useSynthStore(s => s.viewMode);
  const undoStack = useSynthStore(s => s.undoStack);
  const redoStack = useSynthStore(s => s.redoStack);
  const pitchSnap = useSynthStore(s => s.pitchSnap);
  const togglePitchSnap = useSynthStore(s => s.togglePitchSnap);
  const isInstMode = useSynthStore(s => s.activeInstrumentPartIndex) >= 0;
  const setTool = useSynthStore(s => s.setTool);
  const setViewState = useSynthStore(s => s.setViewState);
  const setViewMode = useSynthStore(s => s.setViewMode);
  const undo = useSynthStore(s => s.undo);
  const redo = useSynthStore(s => s.redo);

  const isArr = viewMode === 'arrangement';

  // View mode button style
  const viewBtnStyle = (active: boolean) => ({
    fontSize: 13,
    padding: '2px 8px',
    borderColor: active ? 'var(--accent-purple)' : 'var(--border-dim)',
    color: active ? 'var(--accent-purple)' : 'var(--text-dim)',
    background: active ? 'rgba(155, 109, 255, 0.12)' : 'transparent',
  });

  // Tool button style — different highlight per tool
  const toolColors: Record<Tool, string> = {
    select: '#5ce0d8',  // cyan
    draw: '#9b6dff',    // purple
    erase: '#ff6b9d',   // pink
    pitch: '#5deba5',   // green
  };
  const toolBtnStyle = (id: Tool, active: boolean) => ({
    padding: '2px 8px',
    fontSize: 11,
    borderColor: active ? toolColors[id] : 'var(--border-dim)',
    color: active ? toolColors[id] : 'var(--text-dim)',
    background: active ? `${toolColors[id]}18` : 'transparent',
    opacity: 1,
  });

  return (
    <div className="utai-synth-piano__toolbar">
      {/* View mode toggle */}
      <button className="utai-btn" onClick={() => setViewMode('pianoroll')} title={t('synth.view.pianoroll')} style={viewBtnStyle(!isArr)}>
        ♫
      </button>
      <button className="utai-btn" onClick={() => setViewMode('arrangement')} title={t('synth.view.arrangement')} style={viewBtnStyle(isArr)}>
        ≡
      </button>

      <span style={{ color: 'var(--border-dim)', margin: '0 2px' }}>|</span>

      {/* Tools — labels change based on view mode */}
      {(['select', 'draw', 'erase'] as Tool[]).map((id, i) => (
        <button
          key={id}
          className="utai-btn"
          onClick={() => setTool(id)}
          title={`(${i + 1})`}
          style={toolBtnStyle(id, activeTool === id)}
        >
          {t(isArr
            ? (id === 'select' ? 'synth.arr.select' : id === 'draw' ? 'synth.arr.create' : 'synth.arr.delete')
            : `synth.tool.${id}`
          )}
        </button>
      ))}
      {!isArr && !isInstMode && (
        <button
          className="utai-btn"
          onClick={() => setTool('pitch')}
          title="Pitch Curve (4)"
          style={toolBtnStyle('pitch', activeTool === 'pitch')}
        >
          ♪
        </button>
      )}
      {!isArr && !isInstMode && activeTool === 'pitch' && (
        <button
          className="utai-btn"
          onClick={togglePitchSnap}
          title="Snap to base pitch"
          style={{
            fontSize: 10, padding: '2px 6px',
            borderColor: pitchSnap ? '#5deba5' : 'var(--border-dim)',
            color: pitchSnap ? '#5deba5' : 'var(--text-dim)',
            background: pitchSnap ? 'rgba(93,235,165,0.12)' : 'transparent',
          }}
        >
          Snap
        </button>
      )}

      <span style={{ color: 'var(--border-dim)', margin: '0 2px' }}>|</span>

      {/* Note type (only in piano roll) */}
      {!isArr && (
        <>
          <label style={{ display: 'flex', alignItems: 'center', gap: 4, color: 'var(--text-dim)', fontFamily: 'var(--font-pixel)', fontSize: 11 }}>
            {t('synth.noteType')}:
            <select
              value={viewState.snapTicks}
              onChange={e => setViewState({ snapTicks: Number(e.target.value) })}
              style={{ background: 'var(--bg-deep)', color: 'var(--text-primary)', border: '1px solid var(--border-dim)', fontFamily: 'var(--font-mono)', fontSize: 11, padding: '1px 4px' }}
            >
              <optgroup label="Standard">
                {SNAP_OPTIONS.filter(s => s.group === 'std').map(s => (
                  <option key={s.ticks} value={s.ticks}>{s.label}</option>
                ))}
              </optgroup>
              <optgroup label="Triplet">
                {SNAP_OPTIONS.filter(s => s.group === 'tri').map(s => (
                  <option key={s.ticks} value={s.ticks}>{s.label}</option>
                ))}
              </optgroup>
            </select>
          </label>
          {/* Show resolved label when Alt+scroll goes beyond dropdown presets */}
          {!SNAP_OPTIONS.some(o => o.ticks === viewState.snapTicks) && (
            <span style={{ color: 'var(--accent-purple)', fontFamily: 'var(--font-mono)', fontSize: 10, marginLeft: 2 }}>
              {ticksToDurationLabel(viewState.snapTicks)}
            </span>
          )}
          <span style={{ color: 'var(--border-dim)', margin: '0 2px' }}>|</span>
        </>
      )}

      <button className="utai-btn" onClick={undo} disabled={undoStack.length === 0} title="Undo (Ctrl+Z)"
        style={{ fontSize: 11, padding: '2px 8px', borderColor: 'var(--border-dim)', color: undoStack.length ? 'var(--text-secondary)' : 'var(--text-dim)' }}>
        {t('synth.undo')}
      </button>
      <button className="utai-btn" onClick={redo} disabled={redoStack.length === 0} title="Redo (Ctrl+Y)"
        style={{ fontSize: 11, padding: '2px 8px', borderColor: 'var(--border-dim)', color: redoStack.length ? 'var(--text-secondary)' : 'var(--text-dim)' }}>
        {t('synth.redo')}
      </button>
    </div>
  );
}
