import { useCallback, useEffect, useRef, useState } from 'react'
import { useTranslation } from 'react-i18next'
import { useTrainingStore } from '../store/trainingStore'
import { cleanupTrainingData } from '../api/client'

interface LossPoint {
  epoch: number
  step: number
  loss_mel: number
  loss_disc: number
  loss_gen: number
  loss_fm: number
  loss_kl: number
}

function parseLossPoints(logs: string[]): LossPoint[] {
  const parsed: LossPoint[] = []
  let currentEpoch = 0
  let currentStep = 0

  for (const line of logs) {
    const epochMatch = line.match(/Train Epoch:\s*(\d+)/)
    if (epochMatch) currentEpoch = parseInt(epochMatch[1], 10)

    const stepMatch = line.match(/^\[?INFO.*?\[(\d+),\s*[\d.e-]+\]/) || line.match(/^\[(\d+),\s*[\d.e-]+\]/)
    if (stepMatch) currentStep = parseInt(stepMatch[1], 10)

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
  return parsed
}

interface TrainingReportModalProps {
  open: boolean
  onClose: () => void
  expName: string
  success: boolean
  startTime: number  // Date.now() when training started
}

export function TrainingReportModal({ open, onClose, expName, success, startTime }: TrainingReportModalProps) {
  const { t } = useTranslation()
  const logs = useTrainingStore((s) => s.logs)
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const [cleanResult, setCleanResult] = useState('')
  const [metric, setMetric] = useState<'loss_mel' | 'loss_disc' | 'loss_gen' | 'loss_fm' | 'loss_kl'>('loss_mel')

  const points = parseLossPoints(logs)
  const duration = Math.round((Date.now() - startTime) / 1000)
  const hours = Math.floor(duration / 3600)
  const mins = Math.floor((duration % 3600) / 60)
  const secs = duration % 60
  const durationStr = hours > 0 ? `${hours}h ${mins}m ${secs}s` : mins > 0 ? `${mins}m ${secs}s` : `${secs}s`

  // Find best epoch (lowest mel loss)
  let bestEpoch = 0
  let bestMelLoss = Infinity
  for (const p of points) {
    if (p.loss_mel < bestMelLoss) {
      bestMelLoss = p.loss_mel
      bestEpoch = p.epoch
    }
  }

  // Parse total epochs from logs
  let totalEpochs = 0
  for (const line of logs) {
    const m = line.match(/Train Epoch:\s*(\d+)/)
    if (m) totalEpochs = Math.max(totalEpochs, parseInt(m[1], 10))
  }

  // Auto-cleanup training data when report opens
  useEffect(() => {
    if (!open || !expName) return
    let cancelled = false
    cleanupTrainingData(expName)
      .then((result) => {
        if (!cancelled) setCleanResult(result.freed_formatted || '0 B')
      })
      .catch(() => {
        if (!cancelled) setCleanResult('')
      })
    return () => { cancelled = true }
  }, [open, expName])

  // Draw chart
  useEffect(() => {
    if (!open || !canvasRef.current || points.length === 0) return
    const canvas = canvasRef.current
    const ctx = canvas.getContext('2d')
    if (!ctx) return

    const dpr = window.devicePixelRatio || 1
    const w = 520
    const h = 240
    canvas.width = w * dpr
    canvas.height = h * dpr
    canvas.style.width = w + 'px'
    canvas.style.height = h + 'px'
    ctx.scale(dpr, dpr)

    ctx.fillStyle = '#0a0a1a'
    ctx.fillRect(0, 0, w, h)

    const pad = { top: 25, right: 15, bottom: 35, left: 50 }
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

    // Y labels
    ctx.fillStyle = '#6a6a8a'
    ctx.font = '9px monospace'
    ctx.textAlign = 'right'
    for (let i = 0; i <= 4; i++) {
      const y = pad.top + (plotH / 4) * i
      const val = maxV - ((maxV - minV) / 4) * i
      ctx.fillText(val.toFixed(1), pad.left - 4, y + 3)
    }

    // X labels
    ctx.textAlign = 'center'
    const epochs = points.map((p) => p.epoch)
    const minE = Math.min(...epochs)
    const maxE = Math.max(...epochs)
    for (let i = 0; i <= 4; i++) {
      const x = pad.left + (plotW / 4) * i
      const epoch = Math.round(minE + ((maxE - minE) / 4) * i)
      ctx.fillText(`E${epoch}`, x, h - pad.bottom + 12)
    }

    // Colors
    const colors: Record<string, string> = {
      loss_mel: '#ff6b9d',
      loss_disc: '#6bffb8',
      loss_gen: '#6bb8ff',
      loss_fm: '#ffb86b',
      loss_kl: '#b86bff',
    }

    // Line
    ctx.strokeStyle = colors[metric] || '#ff6b9d'
    ctx.lineWidth = 1.5
    ctx.beginPath()
    for (let i = 0; i < points.length; i++) {
      const x = pad.left + ((points[i].epoch - minE) / (maxE - minE || 1)) * plotW
      const y = pad.top + ((maxV - points[i][metric]) / (maxV - minV || 1)) * plotH
      if (i === 0) ctx.moveTo(x, y)
      else ctx.lineTo(x, y)
    }
    ctx.stroke()

    // Title
    ctx.fillStyle = '#e0e0ff'
    ctx.font = '10px monospace'
    ctx.textAlign = 'center'
    ctx.fillText(metric, w / 2, 14)
  }, [open, points, metric])

  const handleExportChart = useCallback(() => {
    if (!canvasRef.current) return
    const link = document.createElement('a')
    link.download = `utai_report_${expName}_${metric}.png`
    link.href = canvasRef.current.toDataURL('image/png')
    link.click()
  }, [expName, metric])

  if (!open) return null

  const metrics: Array<{ key: typeof metric; label: string; color: string }> = [
    { key: 'loss_mel', label: 'Mel', color: '#ff6b9d' },
    { key: 'loss_disc', label: 'Disc', color: '#6bffb8' },
    { key: 'loss_gen', label: 'Gen', color: '#6bb8ff' },
    { key: 'loss_fm', label: 'FM', color: '#ffb86b' },
    { key: 'loss_kl', label: 'KL', color: '#b86bff' },
  ]

  return (
    <>
      <div className="utai-modal-backdrop" onClick={onClose} style={{ zIndex: 3000 }} />
      <div
        className="utai-confirm"
        style={{
          zIndex: 3001,
          maxWidth: 580,
          width: '95vw',
          border: success ? '2px solid var(--accent-green)' : '2px solid var(--accent-pink)',
          boxShadow: success
            ? '0 0 40px rgba(0, 255, 136, 0.15)'
            : '0 0 40px rgba(255, 107, 157, 0.15)',
        }}
      >
        {/* Header */}
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 12 }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
            <span style={{
              fontSize: 20,
              filter: 'grayscale(0)',
            }}>
              {success ? '\u2713' : '\u2717'}
            </span>
            <div>
              <div style={{
                color: success ? 'var(--accent-green)' : 'var(--accent-pink)',
                fontFamily: 'var(--font-pixel)',
                fontSize: 14,
              }}>
                {t('train.report.title')}
              </div>
              <div style={{ color: 'var(--text-dim)', fontSize: 10 }}>
                {expName}
              </div>
            </div>
          </div>
          <button className="utai-btn utai-btn--lang" style={{ fontSize: 10, padding: '2px 6px' }} onClick={onClose}>{'\u2715'}</button>
        </div>

        {/* Summary grid */}
        <div style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(3, 1fr)',
          gap: 8,
          marginBottom: 12,
          padding: '8px 0',
          borderTop: '1px solid var(--border-dim)',
          borderBottom: '1px solid var(--border-dim)',
        }}>
          <div style={{ textAlign: 'center' }}>
            <div style={{ color: 'var(--text-dim)', fontSize: 9 }}>{t('train.report.duration')}</div>
            <div style={{ color: 'var(--accent-cyan)', fontSize: 13, fontFamily: 'var(--font-pixel)' }}>{durationStr}</div>
          </div>
          <div style={{ textAlign: 'center' }}>
            <div style={{ color: 'var(--text-dim)', fontSize: 9 }}>{t('train.report.epochs')}</div>
            <div style={{ color: 'var(--accent-cyan)', fontSize: 13, fontFamily: 'var(--font-pixel)' }}>{totalEpochs}</div>
          </div>
          <div style={{ textAlign: 'center' }}>
            <div style={{ color: 'var(--text-dim)', fontSize: 9 }}>{t('train.report.bestEpoch')}</div>
            <div style={{ color: '#ffff00', fontSize: 13, fontFamily: 'var(--font-pixel)' }}>
              {points.length > 0 ? `E${bestEpoch} (${bestMelLoss.toFixed(1)})` : '\u2014'}
            </div>
          </div>
        </div>

        {/* Chart */}
        {points.length > 0 && (
          <>
            <div style={{ display: 'flex', gap: 3, marginBottom: 6, flexWrap: 'wrap' }}>
              {metrics.map((m) => (
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
            </div>
            <canvas
              ref={canvasRef}
              style={{ width: '100%', borderRadius: 4, border: '1px solid var(--border-dim)' }}
            />
          </>
        )}

        {/* Auto-cleanup status + actions */}
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginTop: 10, gap: 6 }}>
          <div style={{ display: 'flex', gap: 6, alignItems: 'center' }}>
            {points.length > 0 && (
              <button
                className="utai-btn utai-btn--save"
                style={{ fontSize: 9, padding: '3px 10px' }}
                onClick={handleExportChart}
              >
                {t('train.exportChart')}
              </button>
            )}
            {cleanResult && (
              <span style={{ fontSize: 9, color: 'var(--text-dim)' }}>
                {t('train.report.autoClean', { size: cleanResult })}
              </span>
            )}
          </div>
          <button
            className="utai-btn"
            style={{
              fontSize: 9,
              padding: '3px 10px',
              borderColor: 'var(--accent-cyan)',
              color: 'var(--accent-cyan)',
            }}
            onClick={onClose}
          >
            {t('dialog.confirm')}
          </button>
        </div>
      </div>
    </>
  )
}
