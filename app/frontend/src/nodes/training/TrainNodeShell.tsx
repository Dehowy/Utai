import { type ReactNode } from 'react'
import { useTranslation } from 'react-i18next'
import { useTrainingStore, type TrainNodeStatus } from '../../store/trainingStore'

interface TrainNodeShellProps {
  id: string
  title: string
  color: string
  children: ReactNode
  onRun?: () => void
  onStop?: () => void
  running?: boolean
}

const STATUS_LABEL: Record<TrainNodeStatus, string> = {
  idle: '',
  running: '>>>',
  completed: 'OK',
  error: 'ERR',
}

export function TrainNodeShell({ id, title, color, children, onRun, onStop, running }: TrainNodeShellProps) {
  const { t } = useTranslation()
  const nodeState = useTrainingStore((s) => s.nodeStates[id])
  const isRunningAll = useTrainingStore((s) => s.isRunningAll)
  const status: TrainNodeStatus = nodeState?.status || 'idle'

  const statusClass =
    status === 'running' ? 'utai-node--running' :
    status === 'completed' ? 'utai-node--completed' :
    status === 'error' ? 'utai-node--error' : ''

  return (
    <div className={`utai-node ${statusClass}`} style={{ minWidth: 280 }}>
      <div className="utai-node__header" style={{ borderBottomColor: color + '44' }}>
        <span style={{ color }}>{title}</span>
        <div style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
          {status !== 'idle' && (
            <span
              className="utai-node__header-status"
              style={{ color: status === 'error' ? 'var(--accent-red)' : 'var(--accent-cyan)' }}
            >
              {STATUS_LABEL[status]}
            </span>
          )}
          {onStop && status === 'running' && (
            <button
              className="utai-train-node-run utai-train-node-stop"
              onClick={(e) => { e.stopPropagation(); onStop() }}
              title={t('status.stop')}
            >
              ■
            </button>
          )}
          {onRun && (
            <button
              className="utai-train-node-run"
              disabled={running || status === 'running' || isRunningAll}
              onClick={(e) => { e.stopPropagation(); onRun() }}
              style={{ borderColor: color }}
            >
              ▶
            </button>
          )}
        </div>
      </div>

      {status === 'running' && (
        <div className="utai-node__progress-bar">
          <div className="utai-node__progress-fill--running" />
        </div>
      )}

      <div className="nodrag nopan utai-node__body">{children}</div>

      {nodeState?.message && (
        <div className={`utai-node__message ${status === 'error' ? 'utai-node__message--error' : ''}`}>
          &gt; {nodeState.message}
        </div>
      )}
    </div>
  )
}
