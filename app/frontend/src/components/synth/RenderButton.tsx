import { useRef } from 'react';
import { useTranslation } from 'react-i18next';
import { useSynthStore } from '../../store/synthStore';
import {
  renderTracks,
  getDirtyTrackIndices,
  getActiveTrackIndex,
  trackNeedsRender,
} from '../../lib/synth/renderEngine';

interface Props {
  renderAll?: boolean;
}

export default function RenderButton({ renderAll }: Props) {
  const { t } = useTranslation();
  const dirty = useSynthStore(s => s.dirty);
  const viewMode = useSynthStore(s => s.viewMode);
  const renderStatus = useSynthStore(s => s.renderStatus);
  const renderError = useSynthStore(s => s.renderError);
  const setRenderStatus = useSynthStore(s => s.setRenderStatus);
  const setRenderError = useSynthStore(s => s.setRenderError);
  const abortRef = useRef<AbortController | null>(null);

  const isArrangement = renderAll || viewMode === 'arrangement';

  const handleStop = async () => {
    abortRef.current?.abort();
    abortRef.current = null;
    try { await fetch('/api/synth/stop', { method: 'POST' }); } catch {}
    setRenderStatus('idle');
    setRenderError('');
  };

  const handleRender = async () => {
    setRenderError('');

    const ac = new AbortController();
    abortRef.current = ac;
    setRenderStatus('rendering');

    try {
      let trackIndices: number[];

      if (isArrangement) {
        trackIndices = getDirtyTrackIndices();
        if (trackIndices.length === 0) {
          const s = useSynthStore.getState();
          trackIndices = s.project.tracks
            .map((t, i) => (t.track_type !== 'audio' ? i : -1))
            .filter(i => i >= 0);
        }
      } else {
        const activeTrack = getActiveTrackIndex();
        if (activeTrack < 0) {
          setRenderError(t('synth.noNotes'));
          setRenderStatus('error');
          return;
        }
        trackIndices = [activeTrack];
      }

      const s = useSynthStore.getState();
      const voiceTracksWithoutModel = trackIndices.filter(i => {
        const track = s.project.tracks[i];
        return track?.track_type === 'vocal' && !s.trackRvcModel[i] && !s.rvcModel;
      });
      if (voiceTracksWithoutModel.length > 0) {
        setRenderError(t('synth.needRvcModel', 'Please select an RVC model for vocal tracks first'));
        trackIndices = trackIndices.filter(i => !voiceTracksWithoutModel.includes(i));
        if (trackIndices.length === 0) {
          setRenderStatus('error');
          return;
        }
      }

      await renderTracks(trackIndices, ac.signal, (msg) => {
        setRenderError(msg);
      });

      abortRef.current = null;
      setRenderError('');
      setRenderStatus('done');
    } catch (err: any) {
      if (err.name === 'AbortError') return;
      setRenderError(`Render: ${err.message || 'Unknown error'}`);
      setRenderStatus('error');
    }
  };

  const busy = renderStatus === 'rendering' || renderStatus === 'converting';
  const label = renderAll
    ? t('synth.renderAll', 'Render Voices & Instruments')
    : busy
      ? t('synth.rendering', 'Rendering...')
      : dirty
        ? `${t('synth.runPipeline', 'Render')} *`
        : t('synth.runPipeline', 'Render');

  const flatStyle = renderAll ? { padding: '3px 8px', fontSize: 11 } : {};

  return (
    <div style={renderAll ? { padding: '4px 10px' } : undefined}>
      <div style={{ display: 'flex', gap: 6 }}>
        <button
          className={`utai-synth-render-btn ${dirty ? 'utai-synth-render-btn--dirty' : ''}`}
          onClick={handleRender}
          disabled={busy}
          style={{ flex: 1, ...flatStyle }}
        >
          {busy && renderAll ? t('synth.rendering', 'Rendering...') : label}
        </button>
        {busy && (
          <button
            className="utai-synth-render-btn"
            onClick={handleStop}
            style={{ flex: 'none', width: 36, padding: 0, background: 'rgba(255,80,80,0.15)', color: 'var(--accent-red)', ...flatStyle }}
            title={t('synth.stop', 'Stop')}
          >
            ■
          </button>
        )}
      </div>
      {renderError && (
        <div style={{ color: 'var(--accent-red)', fontSize: 11, marginTop: 4, fontFamily: 'var(--font-pixel)' }}>
          {renderError}
        </div>
      )}
    </div>
  );
}
