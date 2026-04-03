import { type ReactNode } from 'react'
import { useExecutionStore } from '../store/executionStore'

interface NodeShellProps {
  id: string
  title: string
  color: string
  children: ReactNode
}

const STATUS_MAP: Record<string, { label: string; className: string }> = {
  idle: { label: '', className: '' },
  pending: { label: '...', className: 'utai-node--pending' },
  running: { label: '>>>',  className: 'utai-node--running' },
  completed: { label: 'OK', className: 'utai-node--completed' },
  error: { label: 'ERR', className: 'utai-node--error' },
}

export function NodeShell({ id, title, color, children }: NodeShellProps) {
  const nodeState = useExecutionStore((s) => s.nodeStates[id])
  const status = nodeState?.status || 'idle'
  const statusInfo = STATUS_MAP[status] || STATUS_MAP.idle

  return (
    <div className={`utai-node ${statusInfo.className}`}>
      {/* Header — retro window title bar */}
      <div
        className="utai-node__header"
        style={{ borderBottomColor: color + '44' }}
      >
        <span style={{ color }}>{title}</span>
        {status !== 'idle' && (
          <span
            className="utai-node__header-status"
            style={{ color: status === 'error' ? 'var(--accent-red)' : 'var(--accent-cyan)' }}
          >
            {statusInfo.label}
          </span>
        )}
      </div>

      {/* Progress indicators */}
      {status === 'running' && (
        <div className="utai-node__progress-bar">
          <div className="utai-node__progress-fill--running" />
        </div>
      )}
      {status === 'pending' && (
        <div className="utai-node__progress-bar">
          <div className="utai-node__progress-fill--pending" />
        </div>
      )}

      {/* Body */}
      <div className="nodrag nopan utai-node__body">{children}</div>

      {/* Status message */}
      {nodeState?.message && (
        <div className={`utai-node__message ${status === 'error' ? 'utai-node__message--error' : ''}`}>
          &gt; {nodeState.message}
        </div>
      )}
    </div>
  )
}
