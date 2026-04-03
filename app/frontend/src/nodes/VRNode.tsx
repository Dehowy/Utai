import { useCallback, useEffect, useState } from 'react'
import { Handle, Position, type NodeProps } from '@xyflow/react'
import { useTranslation } from 'react-i18next'
import { NodeShell } from './NodeShell'
import { useWorkflowStore } from '../store/workflowStore'
import { useExecutionStore } from '../store/executionStore'
import { getVRModels } from '../api/client'
import { AudioPlayer } from '../components/AudioPlayer'
import { UtaiSelect } from '../components/UtaiSelect'
import { onModelsChanged } from '../utils/modelEvents'

export function VRNode({ id, data }: NodeProps) {
  const { t } = useTranslation()
  const updateNodeData = useWorkflowStore((s) => s.updateNodeData)
  const nodeState = useExecutionStore((s) => s.nodeStates[id])
  const params = (data as any)?.params || {}

  const [models, setModels] = useState<any[]>([])

  const fetchModels = useCallback(() => {
    getVRModels().then((all) => setModels(all.filter((m: any) => m.installed))).catch(console.error)
  }, [])

  useEffect(() => {
    fetchModels()
    return onModelsChanged(fetchModels)
  }, [fetchModels])

  const setParam = (key: string, value: unknown) =>
    updateNodeData(id, { params: { ...params, [key]: value } })

  return (
    <NodeShell id={id} title={t('node.vrSeparation')} color="var(--accent-purple)">
      <Handle type="target" position={Position.Left} id="audio"
        className="utai-handle" style={{ background: 'var(--accent-purple)' }} />

      <div style={{ marginBottom: 6 }}>
        <label className="utai-label">Model</label>
        <UtaiSelect
          value={params.model_name || ''}
          onChange={(v) => setParam('model_name', v)}
          options={[{ value: '', label: t('node.noModel') }, ...models.map((m) => ({ value: m.name, label: m.name }))]}
        />
      </div>

      <div style={{ marginBottom: 4 }}>
        <label className="utai-label">Window Size</label>
        <UtaiSelect
          value={params.window_size || '512'}
          onChange={(v) => setParam('window_size', v)}
          options={[
            { value: '320', label: '320' },
            { value: '512', label: '512' },
            { value: '1024', label: '1024' },
          ]}
        />
      </div>

      <div style={{ marginBottom: 4 }}>
        <label className="utai-label">Aggression: {params.aggression ?? 5}</label>
        <input
          type="range" min={0} max={50} step={1}
          value={params.aggression ?? 5}
          onChange={(e) => setParam('aggression', Number(e.target.value))}
          style={{ width: '100%' }}
        />
      </div>

      <div style={{ display: 'flex', gap: 8, marginBottom: 6 }}>
        <label className="utai-label" style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
          <input type="checkbox" checked={params.use_tta || false} onChange={(e) => setParam('use_tta', e.target.checked)} />
          TTA
        </label>
        <label className="utai-label" style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
          <input type="checkbox" checked={params.post_process || false} onChange={(e) => setParam('post_process', e.target.checked)} />
          Post Process
        </label>
      </div>

      {['primary', 'secondary'].map((stem, i) => {
        const bufferId = nodeState?.outputs?.[stem]
        const color = i === 0 ? 'var(--accent-green)' : 'var(--accent-orange)'
        return (
          <div key={stem} className="utai-stem-row" style={{ position: 'relative', marginTop: 2 }}>
            <div style={{ display: 'flex', justifyContent: 'flex-end', alignItems: 'center', gap: 4, paddingRight: 6 }}>
              {bufferId && <AudioPlayer bufferId={bufferId} compact />}
              <span className="utai-stem-label" style={{ color }}>{stem}</span>
            </div>
            <Handle type="source" position={Position.Right} id={stem}
              className="utai-handle" style={{ background: color }} />
          </div>
        )
      })}
    </NodeShell>
  )
}
