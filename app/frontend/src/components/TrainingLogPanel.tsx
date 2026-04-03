import { useCallback, useEffect, useRef, useState } from 'react'
import { useTranslation } from 'react-i18next'
import { useTrainingStore } from '../store/trainingStore'

interface LossPoint {
  epoch: number
  step: number
  loss_disc: number
  loss_gen: number
  loss_fm: number
  loss_mel: number
  loss_kl: number
}

type MetricKey = 'loss_mel' | 'loss_disc' | 'loss_gen' | 'loss_fm' | 'loss_kl'

const METRICS: Array<{ key: MetricKey; label: string; color: string }> = [
  { key: 'loss_mel', label: 'Mel', color: '#ff6b9d' },
  { key: 'loss_disc', label: 'Disc', color: '#6bffb8' },
  { key: 'loss_gen', label: 'Gen', color: '#6bb8ff' },
  { key: 'loss_fm', label: 'FM', color: '#ffb86b' },
  { key: 'loss_kl', label: 'KL', color: '#b86bff' },
]

function InlineChart({ points, width }: { points: LossPoint[]; width: number }) {
  const { t } = useTranslation()
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const [metric, setMetric] = useState<MetricKey>('loss_mel')

  const handleExport = useCallback(() => {
    if (!canvasRef.current) return
    const link = document.createElement('a')
    link.download = `utai_training_${metric}_${new Date().toISOString().slice(0, 10)}.png`
    link.href = canvasRef.current.toDataURL('image/png')
    link.click()
  }, [metric])

  useEffect(() => {
    if (!canvasRef.current || points.length === 0) return
    const canvas = canvasRef.current
    const ctx = canvas.getContext('2d')
    if (!ctx) return

    const dpr = window.devicePixelRatio || 1
    const w = Math.max(280, width - 24)
    const h = 200
    canvas.width = w * dpr
    canvas.height = h * dpr
    canvas.style.width = w + 'px'
    canvas.style.height = h + 'px'
    ctx.scale(dpr, dpr)

    ctx.fillStyle = '#0a0a1a'
    ctx.fillRect(0, 0, w, h)

    const pad = { top: 24, right: 12, bottom: 30, left: 45 }
    const plotW = w - pad.left - pad.right
    const plotH = h - pad.top - pad.bottom

    const values = points.map((p) => p[metric])
    const minV = Math.min(...values) * 0.9
    const maxV = Math.max(...values) * 1.05

    // Grid
    ctx.strokeStyle = '#1a1a35'
    ctx.lineWidth = 0.5
    for (let i = 0; i <= 4; i++) {
      const y = pad.top + (plotH / 4) * i
      ctx.beginPath()
      ctx.moveTo(pad.left, y)
      ctx.lineTo(w - pad.right, y)
      ctx.stroke()
    }

    // Y axis labels
    ctx.fillStyle = '#6a6a8a'
    ctx.font = '9px monospace'
    ctx.textAlign = 'right'
    for (let i = 0; i <= 4; i++) {
      const y = pad.top + (plotH / 4) * i
      const val = maxV - ((maxV - minV) / 4) * i
      ctx.fillText(val.toFixed(1), pad.left - 4, y + 3)
    }

    // X axis
    ctx.textAlign = 'center'
    const epochs = points.map((p) => p.epoch)
    const minE = Math.min(...epochs)
    const maxE = Math.max(...epochs)
    for (let i = 0; i <= 4; i++) {
      const x = pad.left + (plotW / 4) * i
      const epoch = Math.round(minE + ((maxE - minE) / 4) * i)
      ctx.fillText(`E${epoch}`, x, h - pad.bottom + 12)
    }

    // Title
    ctx.fillStyle = '#e0e0ff'
    ctx.font = '10px monospace'
    ctx.textAlign = 'center'
    ctx.fillText(`${metric} — ${t('train.chartTitle')}`, w / 2, 14)

    // Line
    const color = METRICS.find((m) => m.key === metric)?.color || '#ff6b9d'
    ctx.strokeStyle = color
    ctx.lineWidth = 1.5
    ctx.beginPath()
    for (let i = 0; i < points.length; i++) {
      const x = pad.left + ((points[i].epoch - minE) / (maxE - minE || 1)) * plotW
      const y = pad.top + ((maxV - points[i][metric]) / (maxV - minV || 1)) * plotH
      if (i === 0) ctx.moveTo(x, y)
      else ctx.lineTo(x, y)
    }
    ctx.stroke()

    // Dots
    ctx.fillStyle = color
    for (let i = 0; i < points.length; i++) {
      const x = pad.left + ((points[i].epoch - minE) / (maxE - minE || 1)) * plotW
      const y = pad.top + ((maxV - points[i][metric]) / (maxV - minV || 1)) * plotH
      ctx.beginPath()
      ctx.arc(x, y, 2, 0, Math.PI * 2)
      ctx.fill()
    }

    // Best point
    let minIdx = 0
    for (let i = 1; i < values.length; i++) {
      if (values[i] < values[minIdx]) minIdx = i
    }
    const best = points[minIdx]
    const bx = pad.left + ((best.epoch - minE) / (maxE - minE || 1)) * plotW
    const by = pad.top + ((maxV - best[metric]) / (maxV - minV || 1)) * plotH
    ctx.strokeStyle = '#ffff00'
    ctx.lineWidth = 1
    ctx.beginPath()
    ctx.arc(bx, by, 5, 0, Math.PI * 2)
    ctx.stroke()
    ctx.fillStyle = '#ffff00'
    ctx.font = '8px monospace'
    ctx.textAlign = 'left'
    ctx.fillText(`best: E${best.epoch} (${best[metric].toFixed(2)})`, bx + 8, by - 4)
  }, [points, metric, width, t])

  return (
    <div style={{ padding: '6px 8px', borderBottom: '1px solid var(--border-dim)' }}>
      {/* Metric selector + export */}
      <div style={{ display: 'flex', gap: 3, marginBottom: 4, flexWrap: 'wrap', alignItems: 'center' }}>
        {METRICS.map((m) => (
          <button
            key={m.key}
            className="utai-btn"
            onClick={() => setMetric(m.key)}
            style={{
              fontSize: 8,
              padding: '1px 6px',
              color: metric === m.key ? '#0a0a1a' : m.color,
              borderColor: m.color,
              background: metric === m.key ? m.color : 'transparent',
            }}
          >
            {m.label}
          </button>
        ))}
        <div style={{ flex: 1 }} />
        <button
          className="utai-btn utai-btn--save"
          style={{ fontSize: 8, padding: '1px 6px' }}
          onClick={handleExport}
        >
          {t('train.exportChart')}
        </button>
      </div>
      <canvas ref={canvasRef} style={{ width: '100%', borderRadius: 3, border: '1px solid var(--border-dim)' }} />
    </div>
  )
}

export function TrainingLogPanel() {
  const { t } = useTranslation()
  const logs = useTrainingStore((s) => s.logs)
  const clearLogs = useTrainingStore((s) => s.clearLogs)
  const logPanelOpen = useTrainingStore((s) => s.logPanelOpen)
  const setLogPanelOpen = useTrainingStore((s) => s.setLogPanelOpen)
  const logsEndRef = useRef<HTMLDivElement>(null)
  const panelRef = useRef<HTMLDivElement>(null)
  const [width, setWidth] = useState(380)

  // Parse loss points from logs
  const [points, setPoints] = useState<LossPoint[]>([])
  useEffect(() => {
    const parsed: LossPoint[] = []
    let currentEpoch = 0
    let currentStep = 0
    for (const line of logs) {
      const epochMatch = line.match(/Train Epoch:\s*(\d+)/)
      if (epochMatch) currentEpoch = parseInt(epochMatch[1], 10)
      const stepMatch = line.match(/^\[?INFO.*?\[(\d+),\s*[\d.e-]+\]/)
      if (!stepMatch) {
        const s = line.match(/^\[(\d+),\s*[\d.e-]+\]/)
        if (s) currentStep = parseInt(s[1], 10)
      } else {
        currentStep = parseInt(stepMatch[1], 10)
      }
      const lossMatch = line.match(/loss_disc=([\d.]+),\s*loss_gen=([\d.]+),\s*loss_fm=([\d.]+),\s*loss_mel=([\d.]+),\s*loss_kl=([\d.]+)/)
      if (lossMatch) {
        parsed.push({
          epoch: currentEpoch,
          step: currentStep,
          loss_disc: parseFloat(lossMatch[1]),
          loss_gen: parseFloat(lossMatch[2]),
          loss_fm: parseFloat(lossMatch[3]),
          loss_mel: parseFloat(lossMatch[4]),
          loss_kl: parseFloat(lossMatch[5]),
        })
      }
    }
    setPoints(parsed)
  }, [logs])

  useEffect(() => {
    logsEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [logs])

  const handleResizeStart = useCallback((e: React.MouseEvent) => {
    e.preventDefault()
    const startX = e.clientX
    const startW = width

    const onMove = (ev: MouseEvent) => {
      const newW = Math.max(200, Math.min(window.innerWidth * 0.7, startW + (startX - ev.clientX)))
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
    a.download = `utai_training_log_${new Date().toISOString().replace(/[:.]/g, '-').slice(0, 19)}.txt`
    document.body.appendChild(a)
    a.click()
    document.body.removeChild(a)
    URL.revokeObjectURL(url)
  }, [logs])

  if (!logPanelOpen) return null

  const hasChart = points.length > 0

  return (
    <div ref={panelRef} className="utai-train-logs" style={{ width, flex: 'none' }}>
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

      {/* Chart section (top) — only when training data exists */}
      {hasChart && <InlineChart points={points} width={width} />}

      {/* Logs section (bottom) — scrollable */}
      <div className="utai-train-logs__content">
        {logs.map((log, i) => (
          <div key={i} className={`utai-train-logs__line ${
            log.includes('[error]') || log.includes('ERROR') || log.includes('[fatal]')
              ? 'utai-train-logs__line--error'
              : log.startsWith('===') || log.startsWith('STEP')
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
