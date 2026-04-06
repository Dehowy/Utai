import { useState } from 'react';
import { useTranslation } from 'react-i18next';
interface G2pWord { phonemes: string; }

const G2P_LANGS = [
  { code: 'en', label: 'EN' },
  { code: 'zh', label: 'ZH' },
  { code: 'ja', label: 'JA' },
];

export default function G2pTool() {
  const { t } = useTranslation();
  const [open, setOpen] = useState(true);
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
      {/* Header: toggle + title + lang switcher */}
      <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
        <button className="utai-synth-logs__toggle" style={{ flex: 1, textAlign: 'left' }} onClick={() => setOpen(!open)}>
          {open ? '▼' : '▶'} {t('synth.g2p')}
        </button>
        <div className="utai-lang-switcher" style={{ height: 18 }}>
          {G2P_LANGS.map(l => (
            <button
              key={l.code}
              className={`utai-lang-switcher__item ${lang === l.code ? 'utai-lang-switcher__item--active' : ''}`}
              style={{ lineHeight: '16px', fontSize: 9, padding: '0 6px' }}
              onClick={() => setLang(l.code)}
            >
              {l.label}
            </button>
          ))}
        </div>
      </div>
      {open && (
        <div style={{ padding: '6px 8px 8px' }}>
            <textarea
            value={text}
            onChange={e => setText(e.target.value)}
            onKeyDown={e => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); handleConvert(); } }}
            placeholder={t('synth.g2pPlaceholder')}
            rows={3}
            style={{
              width: '100%', boxSizing: 'border-box',
              fontSize: 11, padding: '4px 6px', marginBottom: 4,
              background: 'var(--bg-deep)', color: 'var(--text-primary)',
              border: '1px solid var(--border-mid)', fontFamily: 'var(--font-pixel)',
              resize: 'vertical', minHeight: 36,
            }}
          />
          <div style={{ display: 'flex', justifyContent: 'flex-end', marginBottom: 6 }}>
            <button
              className="utai-btn"
              style={{ fontSize: 10, padding: '2px 10px' }}
              onClick={handleConvert}
              disabled={loading}
            >
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
