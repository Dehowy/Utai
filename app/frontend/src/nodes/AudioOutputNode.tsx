import { Handle, Position, type NodeProps } from '@xyflow/react'
import { useTranslation } from 'react-i18next'
import { NodeShell } from './NodeShell'
import { useExecutionStore } from '../store/executionStore'
import { AudioPlayer } from '../components/AudioPlayer'
import { useWorkflowStore } from '../store/workflowStore'
import { UtaiSelect } from '../components/UtaiSelect'

export function AudioOutputNode({ id, data }: NodeProps) {
  const { t } = useTranslation()
  const nodeState = useExecutionStore((s) => s.nodeStates[id])
  const updateNodeData = useWorkflowStore((s) => s.updateNodeData)
  const params = (data as any)?.params || {}

  const outputBufferId = nodeState?.outputs?.audio

  return (
    <NodeShell id={id} title={t('node.audioOutput')} color="var(--accent-orange)">
      <Handle type="target" position={Position.Left} id="audio"
        className="utai-handle" style={{ background: 'var(--accent-orange)' }} />

      <div style={{ marginBottom: 6 }}>
        <label className="utai-label">Format</label>
        <UtaiSelect
          value={params.format || 'wav'}
          onChange={(v) => updateNodeData(id, { params: { ...params, format: v } })}
          options={[
            { value: 'wav', label: 'WAV' },
            { value: 'flac', label: 'FLAC' },
            { value: 'mp3', label: 'MP3' },
          ]}
        />
      </div>

      {outputBufferId ? (
        <AudioPlayer bufferId={outputBufferId} />
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
