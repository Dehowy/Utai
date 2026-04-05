import { useState, useEffect, useRef } from 'react';
import { useTranslation } from 'react-i18next';
interface LogEntry { level: string; name: string; message: string; }

export default function LogViewer() {
  const { t } = useTranslation();
  const [open, setOpen] = useState(false);
  const [logs, setLogs] = useState<LogEntry[]>([]);
  const [autoRefresh, setAutoRefresh] = useState(false);
  const bottomRef = useRef<HTMLDivElement>(null);

  const fetchLogs = async () => {
    try { const res = await fetch('/api/synth/logs?limit=100'); const data = await res.json(); setLogs(data.logs); } catch {}
  };

  useEffect(() => { if (open) fetchLogs(); }, [open]);
  useEffect(() => {
    if (!autoRefresh || !open) return;
    const interval = setInterval(fetchLogs, 2000);
    return () => clearInterval(interval);
  }, [autoRefresh, open]);
  useEffect(() => { if (open) bottomRef.current?.scrollIntoView({ behavior: 'smooth' }); }, [logs, open]);

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
            <button className="utai-btn" style={{ fontSize: 10, padding: '1px 6px' }} onClick={fetchLogs}>{t('synth.refresh')}</button>
            <label style={{ display: 'flex', alignItems: 'center', gap: 4, color: 'var(--text-secondary)' }}>
              <input type="checkbox" checked={autoRefresh} onChange={e => setAutoRefresh(e.target.checked)} />
              Auto (2s)
            </label>
          </div>
          <div className="utai-synth-logs__content">
            {logs.length === 0 && <div style={{ color: 'var(--text-muted)' }}>No logs</div>}
            {logs.map((log, i) => (
              <div key={i} style={{ borderBottom: '1px solid var(--border-dim)', padding: '2px 0' }}>
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
