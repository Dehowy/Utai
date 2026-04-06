import { useRef } from 'react';
import { useTranslation } from 'react-i18next';
import { useSynthStore } from '../../store/synthStore';
import { serializeUstx } from '../../lib/synth/ustx';

export default function RenderButton() {
  const { t } = useTranslation();
  const dirty = useSynthStore(s => s.dirty);
  const project = useSynthStore(s => s.project);
  const singer = useSynthStore(s => s.singer);
  const rvcModel = useSynthStore(s => s.rvcModel);
  const rvcSettings = useSynthStore(s => s.rvcSettings);
  const renderStatus = useSynthStore(s => s.renderStatus);
  const renderError = useSynthStore(s => s.renderError);
  const setRenderStatus = useSynthStore(s => s.setRenderStatus);
  const setRenderedUrl = useSynthStore(s => s.setRenderedUrl);
  const setFinalUrl = useSynthStore(s => s.setFinalUrl);
  const setRenderError = useSynthStore(s => s.setRenderError);
  const markClean = useSynthStore(s => s.markClean);
  const abortRef = useRef<AbortController | null>(null);

  const hasNotes = project.voice_parts.some(p => p.notes.length > 0);

  const handleStop = async () => {
    abortRef.current?.abort();
    abortRef.current = null;
    try { await fetch('/api/synth/stop', { method: 'POST' }); } catch {}
    setRenderStatus('idle');
    setRenderError('');
  };

  const handleRender = async () => {
    setRenderError('');

    if (!singer) { setRenderError(t('synth.selectSingerFirst')); return; }
    if (!hasNotes) { setRenderError(t('synth.noNotes')); return; }

    const ac = new AbortController();
    abortRef.current = ac;
    const yaml = serializeUstx(project);

    // Step 1: DiffSinger render
    setRenderStatus('rendering');
    let renderedBlob: Blob;
    try {
      const form = new FormData();
      form.append('content', yaml);
      form.append('filename', 'project.ustx');
      const res = await fetch('/api/synth/render/content', { method: 'POST', body: form, signal: ac.signal });
      if (!res.ok) throw new Error(await res.text());
      renderedBlob = await res.blob();
      if (renderedBlob.size < 100) {
        setRenderError(t('synth.renderEmpty'));
        setRenderStatus('error');
        return;
      }
      const prevUrl = useSynthStore.getState().renderedUrl;
      if (prevUrl) URL.revokeObjectURL(prevUrl);
      setRenderedUrl(URL.createObjectURL(renderedBlob));
    } catch (err: any) {
      if (err.name === 'AbortError') return;
      setRenderError(`${t('synth.renderFailed')}: ${err.message || 'Unknown error'}`);
      setRenderStatus('error');
      return;
    }

    // Step 2: RVC convert (if model selected)
    if (rvcModel) {
      setRenderStatus('converting');
      try {
        const form = new FormData();
        form.append('file', new File([renderedBlob], 'rendered.wav', { type: 'audio/wav' }));
        form.append('model', rvcModel);
        Object.entries(rvcSettings).forEach(([k, v]) => form.append(k, String(v)));
        const res = await fetch('/api/synth/convert', { method: 'POST', body: form, signal: ac.signal });
        if (!res.ok) throw new Error(await res.text());
        const prevFinalUrl = useSynthStore.getState().finalUrl;
        if (prevFinalUrl) URL.revokeObjectURL(prevFinalUrl);
        setFinalUrl(URL.createObjectURL(await res.blob()));
      } catch (err: any) {
        if (err.name === 'AbortError') return;
        setRenderError(`${t('synth.convertFailed')}: ${err.message || 'Unknown error'}`);
        setRenderStatus('error');
        return;
      }
    }

    abortRef.current = null;
    markClean();
    setRenderStatus('done');
  };

  const busy = renderStatus === 'rendering' || renderStatus === 'converting';

  return (
    <div>
      <div style={{ display: 'flex', gap: 6 }}>
        <button
          className={`utai-synth-render-btn ${dirty ? 'utai-synth-render-btn--dirty' : ''}`}
          onClick={handleRender}
          disabled={busy}
          style={{ flex: 1 }}
        >
          {renderStatus === 'rendering' ? t('synth.rendering')
            : renderStatus === 'converting' ? t('synth.converting')
            : dirty ? `${t('synth.runPipeline')} *`
            : t('synth.runPipeline')}
        </button>
        {busy && (
          <button
            className="utai-synth-render-btn"
            onClick={handleStop}
            style={{ flex: 'none', width: 36, padding: 0, background: 'rgba(255,80,80,0.15)', color: 'var(--accent-red)' }}
            title={t('synth.stop')}
          >
            ■
          </button>
        )}
      </div>
      {rvcModel && <small style={{ color: 'var(--text-muted)', fontSize: 10, display: 'block', marginTop: 4 }}>+ RVC: {rvcModel}</small>}
      {renderError && (
        <div style={{ color: 'var(--accent-red)', fontSize: 11, marginTop: 4, fontFamily: 'var(--font-pixel)' }}>
          {renderError}
        </div>
      )}
    </div>
  );
}
