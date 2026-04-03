import { Handle, Position, type NodeProps } from '@xyflow/react'
import { useTranslation } from 'react-i18next'
import { NodeShell } from './NodeShell'

export function VideoInputNode({ id, data }: NodeProps) {
  const { t } = useTranslation()
  const params = (data as any)?.params || {}
  const videoName = params.video_name || params.video_path?.split(/[/\\]/).pop() || ''

  return (
    <NodeShell id={id} title={t('node.videoInput')} color="var(--accent-orange)">
      <div style={{
        fontFamily: 'var(--font-pixel)',
        fontSize: 11,
        color: videoName ? 'var(--accent-orange)' : 'var(--text-muted)',
        textAlign: 'center',
        padding: 8,
      }}>
        {videoName || t('status.idle')}
      </div>
      <Handle type="source" position={Position.Right} id="video_path"
        className="utai-handle" style={{ background: 'var(--accent-orange)' }} />
    </NodeShell>
  )
}
