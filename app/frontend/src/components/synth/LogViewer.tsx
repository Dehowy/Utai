import { useState, useEffect, useRef } from 'react';
import { useTranslation } from 'react-i18next';
interface LogEntry { level: string; name: string; message: string; time?: string; }

export default function LogViewer() {
  const { t } = useTranslation();
  const [open, setOpen] = useState(false);
  const [logs, setLogs] = useState<LogEntry[]>([]);
  const [autoRefresh, setAutoRefresh] = useState(false);
  const [copied, setCopied] = useState(false);
  const bottomRef = useRef<HTMLDivElement>(null);
  const userScrolledRef = useRef(false);
  const containerRef = useRef<HTMLDivElement>(null);

  const fetchLogs = async () => {
    try { const res = await fetch('/api/synth/logs?limit=100'); const data = await res.json(); setLogs(data.logs); } catch {}
  };

  useEffect(() => { if (open) fetchLogs(); }, [open]);
  useEffect(() => {
    if (!autoRefresh || !open) return;
    const interval = setInterval(fetchLogs, 2000);
    return () => clearInterval(interval);
  }, [autoRefresh, open]);

  useEffect(() => {
    if (open && !userScrolledRef.current) {
      bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
    }
  }, [logs, open]);

  const handleScroll = () => {
    const el = containerRef.current;
    if (!el) return;
    const atBottom = el.scrollHeight - el.scrollTop - el.clientHeight < 30;
    userScrolledRef.current = !atBottom;
  };

  const handleCopy = () => {
    const text = logs.map(log =>
      `${formatTime(log.time)} [${log.level}] ${log.name}: ${log.message}`
    ).join('\n');
    navigator.clipboard.writeText(text).then(() => {
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    }).catch(() => {});
  };

  const formatTime = (time?: string) => {
    if (!time) return '';
    const match = time.match(/(\d{2}:\d{2}:\d{2})/);
    return match ? match[1] : time.slice(-8);
  };

  const levelColor = (level: string) => {
    switch (level) {
      case 'ERROR': return 'var(--accent-red)';
      case 'WARNING': return 'var(--accent-orange)';
      case 'INFO': return 'var(--accent-blue)';
      case 'DEBUG': return 'var(--text-muted)';
      default: return 'var(--text-primary)';
    }
  };

  return (
    <div className="utai-synth-logs">
      <button className="utai-synth-logs__toggle" onClick={() => setOpen(!open)}>
        {open ? '▼' : '▶'} {t('synth.logs')}
      </button>
      {open && (
        <div>
          <div style={{ display: 'flex', gap: 6, padding: '4px 8px', fontSize: 10 }}>
            <button className="utai-btn" style={{ fontSize: 10, padding: '1px 6px' }} onClick={() => { userScrolledRef.current = false; fetchLogs(); }}>{t('synth.refresh')}</button>
            <button className="utai-btn" style={{ fontSize: 10, padding: '1px 6px', borderColor: copied ? 'var(--accent-green)' : undefined, color: copied ? 'var(--accent-green)' : undefined }} onClick={handleCopy}>
              {copied ? t('synth.copied', 'Copied!') : t('synth.copyLogs', 'Copy')}
            </button>
            <label style={{ display: 'flex', alignItems: 'center', gap: 4, color: 'var(--text-secondary)' }}>
              <input type="checkbox" checked={autoRefresh} onChange={e => setAutoRefresh(e.target.checked)} />
              Auto (2s)
            </label>
          </div>
          <div className="utai-synth-logs__content" ref={containerRef} onScroll={handleScroll}>
            {logs.length === 0 && <div style={{ color: 'var(--text-muted)' }}>No logs</div>}
            {logs.map((log, i) => (
              <div key={i} style={{ borderBottom: '1px solid var(--border-dim)', padding: '2px 0' }}>
                <span style={{ color: 'var(--text-muted)', fontSize: '0.85em', fontFamily: 'var(--font-mono)' }}>{formatTime(log.time)}</span>{' '}
                <span style={{ color: levelColor(log.level) }}>[{log.level}]</span>{' '}
                <span style={{ color: 'var(--text-muted)' }}>{log.name}:</span>{' '}
                {log.message}
              </div>
            ))}
            <div ref={bottomRef} />
          </div>
        </div>
      )}
    </div>
  );
}
