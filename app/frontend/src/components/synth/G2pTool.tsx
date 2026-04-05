import { useState } from 'react';
import { useTranslation } from 'react-i18next';
interface G2pWord { phonemes: string; }

export default function G2pTool() {
  const { t } = useTranslation();
  const [open, setOpen] = useState(false);
  const [text, setText] = useState('');
  const [lang, setLang] = useState('en');
  const [words, setWords] = useState<G2pWord[]>([]);
  const [loading, setLoading] = useState(false);

  const handleConvert = async () => {
    if (!text.trim()) return;
    setLoading(true);
    try {
      const form = new FormData();
      form.append('text', text);
      form.append('lang', lang);
      const res = await fetch('/api/synth/g2p', { method: 'POST', body: form });
      const data = await res.json();
      setWords(data.words || []);
    } catch { setWords([]); }
    setLoading(false);
  };

  return (
    <div className="utai-synth-g2p">
      <button className="utai-synth-logs__toggle" onClick={() => setOpen(!open)}>
        {open ? '▼' : '▶'} {t('synth.g2p')}
      </button>
      {open && (
        <div style={{ padding: 8 }}>
          <div style={{ display: 'flex', gap: 4, marginBottom: 6 }}>
            <select value={lang} onChange={e => setLang(e.target.value)} style={{ fontSize: 11, background: 'var(--bg-deep)', color: 'var(--text-primary)', border: '1px solid var(--border-mid)', fontFamily: 'var(--font-pixel)' }}>
              <option value="en">EN</option>
              <option value="zh">ZH</option>
              <option value="ja">JA</option>
            </select>
            <input
              value={text} onChange={e => setText(e.target.value)}
              onKeyDown={e => e.key === 'Enter' && handleConvert()}
              placeholder={t('synth.g2pPlaceholder')}
              style={{ flex: 1, fontSize: 11, padding: '2px 6px', background: 'var(--bg-deep)', color: 'var(--text-primary)', border: '1px solid var(--border-mid)', fontFamily: 'var(--font-pixel)' }}
            />
            <button className="utai-btn" style={{ fontSize: 10, padding: '2px 6px' }} onClick={handleConvert} disabled={loading}>
              {loading ? '...' : t('synth.g2pConvert')}
            </button>
          </div>
          {words.length > 0 && (
            <div>
              <div style={{ color: 'var(--text-muted)', fontSize: 10, marginBottom: 4 }}>{t('synth.g2pClickToCopy')}</div>
              <div className="utai-synth-g2p__results">
                {words.map((w, i) => (
                  <span key={i} className="utai-synth-g2p__phoneme" onClick={() => navigator.clipboard.writeText(w.phonemes)} title="Click to copy">
                    {w.phonemes}
                  </span>
                ))}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
