import { useTranslation } from 'react-i18next';
import { useSynthStore, type Tool } from '../../store/synthStore';
import { RESOLUTION } from '../../types/ustx';

const SNAP_OPTIONS = [
  // Standard subdivisions
  { label: '1/1', ticks: RESOLUTION * 4, group: 'std' },
  { label: '1/2', ticks: RESOLUTION * 2, group: 'std' },
  { label: '1/4', ticks: RESOLUTION, group: 'std' },
  { label: '1/8', ticks: RESOLUTION / 2, group: 'std' },
  { label: '1/16', ticks: RESOLUTION / 4, group: 'std' },
  { label: '1/32', ticks: RESOLUTION / 8, group: 'std' },
  // Triplet subdivisions
  { label: '1/2T', ticks: Math.round(RESOLUTION * 4 / 3), group: 'tri' },
  { label: '1/4T', ticks: Math.round(RESOLUTION * 2 / 3), group: 'tri' },
  { label: '1/8T', ticks: Math.round(RESOLUTION / 3), group: 'tri' },
  { label: '1/16T', ticks: Math.round(RESOLUTION / 6), group: 'tri' },
  { label: '1/32T', ticks: Math.round(RESOLUTION / 12), group: 'tri' },
];

const TOOLS: { id: Tool; key: string; labelKey: string }[] = [
  { id: 'select', key: '1', labelKey: 'synth.tool.select' },
  { id: 'draw', key: '2', labelKey: 'synth.tool.draw' },
  { id: 'erase', key: '3', labelKey: 'synth.tool.erase' },
];

export default function PianoRollToolbar() {
  const { t } = useTranslation();
  const activeTool = useSynthStore(s => s.activeTool);
  const viewState = useSynthStore(s => s.viewState);
  const undoStack = useSynthStore(s => s.undoStack);
  const redoStack = useSynthStore(s => s.redoStack);
  const setTool = useSynthStore(s => s.setTool);
  const setViewState = useSynthStore(s => s.setViewState);
  const undo = useSynthStore(s => s.undo);
  const redo = useSynthStore(s => s.redo);

  return (
    <div className="utai-synth-piano__toolbar">
      {TOOLS.map(tool => (
        <button
          key={tool.id}
          className={`utai-btn ${activeTool === tool.id ? 'utai-btn--active' : ''}`}
          onClick={() => setTool(tool.id)}
          title={`${t(tool.labelKey)} (${tool.key})`}
        >
          {t(tool.labelKey)}
        </button>
      ))}

      <span style={{ color: 'var(--text-muted)' }}>|</span>

      <label style={{ display: 'flex', alignItems: 'center', gap: 4, color: 'var(--text-secondary)', fontFamily: 'var(--font-pixel)', fontSize: 11 }}>
        {t('synth.snap')}:
        <select
          value={viewState.snapTicks}
          onChange={e => setViewState({ snapTicks: Number(e.target.value) })}
          style={{ background: 'var(--bg-deep)', color: 'var(--text-primary)', border: '1px solid var(--border-mid)', fontFamily: 'var(--font-mono)', fontSize: 11, padding: '1px 4px' }}
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

      <span style={{ color: 'var(--text-muted)' }}>|</span>

      <button
        className="utai-btn"
        onClick={undo}
        disabled={undoStack.length === 0}
        title="Undo (Ctrl+Z)"
      >
        {t('synth.undo')}
      </button>
      <button
        className="utai-btn"
        onClick={redo}
        disabled={redoStack.length === 0}
        title="Redo (Ctrl+Y)"
      >
        {t('synth.redo')}
      </button>
    </div>
  );
}
