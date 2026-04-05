import { useTranslation } from 'react-i18next';
import { useSynthStore } from '../../store/synthStore';
import { serializeUstx } from '../../lib/synth/ustx';
import type { RVCSettings } from './RVCControls';

interface Props {
  rvcSettings: RVCSettings;
}

export default function RenderButton({ rvcSettings }: Props) {
  const { t } = useTranslation();
  const dirty = useSynthStore(s => s.dirty);
  const project = useSynthStore(s => s.project);
  const singer = useSynthStore(s => s.singer);
  const rvcModel = useSynthStore(s => s.rvcModel);
  const renderStatus = useSynthStore(s => s.renderStatus);
  const setRenderStatus = useSynthStore(s => s.setRenderStatus);
  const setRenderedUrl = useSynthStore(s => s.setRenderedUrl);
  const setFinalUrl = useSynthStore(s => s.setFinalUrl);
  const setRenderError = useSynthStore(s => s.setRenderError);
  const markClean = useSynthStore(s => s.markClean);

  const hasNotes = project.voice_parts.some(p => p.notes.length > 0);

  const handleRender = async () => {
    setRenderError('');
    if (!singer) { setRenderError(t('synth.selectSingerFirst')); return; }
    if (!hasNotes) { setRenderError(t('synth.noNotes')); return; }

    const yaml = serializeUstx(project);

    // Step 1: DiffSinger render
    setRenderStatus('rendering');
    let renderedBlob: Blob;
    try {
      const form = new FormData();
      form.append('content', yaml);
      form.append('filename', 'project.ustx');
      const res = await fetch('/api/synth/render/content', { method: 'POST', body: form });
      if (!res.ok) throw new Error(await res.text());
      renderedBlob = await res.blob();
      if (renderedBlob.size < 100) {
        setRenderError(t('synth.renderEmpty'));
        setRenderStatus('error');
        return;
      }
      setRenderedUrl(URL.createObjectURL(renderedBlob));
    } catch (err: any) {
      const msg = await extractError(err);
      setRenderError(`${t('synth.renderFailed')}: ${msg}`);
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
        const res = await fetch('/api/synth/convert', { method: 'POST', body: form });
        if (!res.ok) throw new Error(await res.text());
        setFinalUrl(URL.createObjectURL(await res.blob()));
      } catch (err: any) {
        const msg = await extractError(err);
        setRenderError(`${t('synth.convertFailed')}: ${msg}`);
        setRenderStatus('error');
        return;
      }
    }

    markClean();
    setRenderStatus('done');
  };

  const busy = renderStatus === 'rendering' || renderStatus === 'converting';

  return (
    <div>
      <button
        className={`utai-synth-render-btn ${dirty ? 'utai-synth-render-btn--dirty' : ''}`}
        onClick={handleRender}
        disabled={busy}
      >
        {renderStatus === 'rendering' ? t('synth.rendering')
          : renderStatus === 'converting' ? t('synth.converting')
          : dirty ? `${t('synth.runPipeline')} *`
          : t('synth.runPipeline')}
      </button>
      {rvcModel && <small style={{ color: 'var(--text-muted)', fontSize: 10, display: 'block', marginTop: 4 }}>+ RVC: {rvcModel}</small>}
      {useSynthStore.getState().renderError && (
        <div style={{ color: 'var(--accent-red)', fontSize: 11, marginTop: 4, fontFamily: 'var(--font-pixel)' }}>
          {useSynthStore.getState().renderError}
        </div>
      )}
    </div>
  );
}

async function extractError(err: any): Promise<string> {
  return err.message || 'Unknown error';
}
