import { Handle, Position, type NodeProps } from '@xyflow/react'
import { useTranslation } from 'react-i18next'
import { NodeShell } from './NodeShell'
import { useExecutionStore } from '../store/executionStore'
import { getVideoURL } from '../api/client'

export function VideoOutputNode({ id, data }: NodeProps) {
  const { t } = useTranslation()
  const nodeState = useExecutionStore((s) => s.nodeStates[id])

  const videoPath = nodeState?.outputs?.video_path
  const videoName = nodeState?.outputs?.video_name || videoPath?.split(/[/\\]/).pop() || ''

  return (
    <NodeShell id={id} title={t('node.videoOutput')} color="var(--accent-orange)">
      <Handle
        type="target"
        position={Position.Left}
        id="video_path"
        className="utai-handle"
        style={{ background: 'var(--accent-orange)' }}
      />

      {nodeState?.status === 'completed' && videoName ? (
        <div style={{ fontSize: 10, color: 'var(--text-body)' }}>
          <div style={{ marginBottom: 4, color: 'var(--accent-orange)' }}>{'\u25B6'} {videoName}</div>
          {videoPath && (
            <a
              href={getVideoURL(videoName)}
              download
              className="utai-btn"
              style={{ fontSize: 9, padding: '2px 8px', textDecoration: 'none' }}
            >
              {t('player.download')}
            </a>
          )}
        </div>
      ) : (
        <div style={{
          color: 'var(--text-muted)',
          fontFamily: 'var(--font-mono)',
          fontSize: 11,
          textAlign: 'center',
          padding: 8,
        }}>
          {t('status.idle')}
        </div>
      )}
    </NodeShell>
  )
}
