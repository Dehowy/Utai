import { useCallback } from 'react'
import { Handle, Position, type NodeProps } from '@xyflow/react'
import { useTranslation } from 'react-i18next'
import { NodeShell } from './NodeShell'
import { useWorkflowStore } from '../store/workflowStore'
import { useExecutionStore } from '../store/executionStore'
import { AudioPlayer } from '../components/AudioPlayer'

const TRACK_COLORS = [
  'var(--accent-green)',
  'var(--accent-blue)',
  'var(--accent-pink)',
  'var(--accent-orange)',
  'var(--accent-purple)',
  'var(--accent-cyan)',
  'var(--accent-gold)',
  'var(--accent-red)',
]

export function AudioMergeNode({ id, data }: NodeProps) {
  const { t } = useTranslation()
  const updateNodeData = useWorkflowStore((s) => s.updateNodeData)
  const nodeState = useExecutionStore((s) => s.nodeStates[id])
  const outputBufferId = nodeState?.outputs?.audio

  const params = (data as any)?.params || {}
  const trackCount = Math.max(2, Math.min(8, parseInt(params.track_count) || 2))

  const setParam = useCallback((key: string, value: any) => {
    updateNodeData(id, { params: { ...params, [key]: value } })
  }, [id, params, updateNodeData])

  return (
    <NodeShell id={id} title={t('node.audioMerge')} color="var(--accent-gold)">
      {/* Track count control */}
      <div style={{
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        padding: '4px 0',
        marginBottom: 4,
      }}>
        <span style={{ fontFamily: 'var(--font-pixel)', fontSize: 10, color: 'var(--text-secondary)' }}>
          {t('node.trackCount')}
        </span>
        <div style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
          <button
            className="utai-btn"
            style={{ padding: '1px 6px', fontSize: 10, minWidth: 20, color: 'var(--accent-gold)', borderColor: 'var(--accent-gold)' }}
            onClick={() => { if (trackCount > 2) setParam('track_count', trackCount - 1) }}
            disabled={trackCount <= 2}
          >
            -
          </button>
          <span style={{
            fontFamily: 'var(--font-mono)',
            fontSize: 13,
            color: 'var(--accent-gold)',
            minWidth: 16,
            textAlign: 'center',
          }}>
            {trackCount}
          </span>
          <button
            className="utai-btn"
            style={{ padding: '1px 6px', fontSize: 10, minWidth: 20, color: 'var(--accent-gold)', borderColor: 'var(--accent-gold)' }}
            onClick={() => { if (trackCount < 8) setParam('track_count', trackCount + 1) }}
            disabled={trackCount >= 8}
          >
            +
          </button>
        </div>
      </div>

      {/* Track input handles */}
      <div style={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
        {Array.from({ length: trackCount }, (_, i) => (
          <div
            key={i}
            className="utai-stem-row"
            style={{
              position: 'relative',
              display: 'flex',
              alignItems: 'center',
              padding: '3px 8px',
              borderLeft: `2px solid ${TRACK_COLORS[i % TRACK_COLORS.length]}`,
              background: 'rgba(255,255,255,0.02)',
            }}
          >
            <Handle
              type="target"
              position={Position.Left}
              id={`audio_${i + 1}`}
              className="utai-handle"
              style={{ background: TRACK_COLORS[i % TRACK_COLORS.length] }}
            />
            <span style={{
              fontFamily: 'var(--font-pixel)',
              fontSize: 10,
              color: TRACK_COLORS[i % TRACK_COLORS.length],
            }}>
              Track {i + 1}
            </span>
          </div>
        ))}
      </div>

      {/* Output */}
      {outputBufferId && (
        <div style={{ marginTop: 6 }}>
          <AudioPlayer bufferId={outputBufferId} />
        </div>
      )}

      {/* Output handle */}
      <Handle
        type="source"
        position={Position.Right}
        id="audio"
        className="utai-handle"
        style={{ background: 'var(--accent-gold)' }}
      />
    </NodeShell>
  )
}
