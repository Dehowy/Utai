import { useCallback, useEffect, useRef, useState } from 'react'
import { useTranslation } from 'react-i18next'
import { useExecutionStore } from '../store/executionStore'

export function WorkflowLogPanel() {
  const { t } = useTranslation()
  const logs = useExecutionStore((s) => s.logs)
  const clearLogs = useExecutionStore((s) => s.clearLogs)
  const logPanelOpen = useExecutionStore((s) => s.logPanelOpen)
  const setLogPanelOpen = useExecutionStore((s) => s.setLogPanelOpen)
  const logsEndRef = useRef<HTMLDivElement>(null)
  const [width, setWidth] = useState(320)

  useEffect(() => {
    logsEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [logs])

  const handleResizeStart = useCallback((e: React.MouseEvent) => {
    e.preventDefault()
    const startX = e.clientX
    const startW = width

    const onMove = (ev: MouseEvent) => {
      const newW = Math.max(180, Math.min(window.innerWidth * 0.5, startW + (startX - ev.clientX)))
      setWidth(newW)
    }
    const onUp = () => {
      document.removeEventListener('mousemove', onMove)
      document.removeEventListener('mouseup', onUp)
      document.body.style.cursor = ''
      document.body.style.userSelect = ''
    }
    document.body.style.cursor = 'col-resize'
    document.body.style.userSelect = 'none'
    document.addEventListener('mousemove', onMove)
    document.addEventListener('mouseup', onUp)
  }, [width])

  const handleSaveLogs = useCallback(() => {
    if (logs.length === 0) return
    const text = logs.join('\n')
    const blob = new Blob([text], { type: 'text/plain;charset=utf-8' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `utai_exec_log_${new Date().toISOString().replace(/[:.]/g, '-').slice(0, 19)}.txt`
    document.body.appendChild(a)
    a.click()
    document.body.removeChild(a)
    URL.revokeObjectURL(url)
  }, [logs])

  if (!logPanelOpen) return null

  return (
    <div className="utai-train-logs" style={{ width, flex: 'none' }}>
      {/* Resize handle */}
      <div className="utai-train-logs__resize" onMouseDown={handleResizeStart} />

      <div className="utai-train-logs__header">
        <span>{t('workflow.logs')}</span>
        <div style={{ display: 'flex', gap: 4 }}>
          <button
            className="utai-btn utai-btn--clear"
            style={{ fontSize: 10, padding: '2px 6px' }}
            onClick={clearLogs}
          >
            {t('toolbar.clear')}
          </button>
          <button
            className="utai-btn utai-btn--save"
            style={{ fontSize: 10, padding: '2px 6px' }}
            onClick={handleSaveLogs}
            disabled={logs.length === 0}
          >
            {t('train.saveLogs')}
          </button>
          <button
            className="utai-btn utai-btn--lang"
            style={{ fontSize: 10, padding: '2px 6px' }}
            onClick={() => setLogPanelOpen(false)}
          >
            ✕
          </button>
        </div>
      </div>
      <div className="utai-train-logs__content">
        {logs.map((log, i) => (
          <div key={i} className={`utai-train-logs__line ${
            log.includes('ERROR') || log.includes('[error]')
              ? 'utai-train-logs__line--error'
              : log.startsWith('[exec]')
                ? 'utai-train-logs__line--header'
                : ''
          }`}>
            {log}
          </div>
        ))}
        <div ref={logsEndRef} />
      </div>
    </div>
  )
}
