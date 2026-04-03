import { useCallback, useEffect, useState } from 'react'
import { Handle, Position, useUpdateNodeInternals, type NodeProps } from '@xyflow/react'
import { useTranslation } from 'react-i18next'
import { NodeShell } from './NodeShell'
import { useWorkflowStore } from '../store/workflowStore'
import { useExecutionStore } from '../store/executionStore'
import { getMSSTModels, getMSSTModelStems } from '../api/client'
import { AudioPlayer } from '../components/AudioPlayer'
import { UtaiSelect } from '../components/UtaiSelect'
import { onModelsChanged } from '../utils/modelEvents'

const STEM_COLORS = [
  'var(--accent-green)', 'var(--accent-orange)', 'var(--accent-pink)',
  'var(--accent-cyan)', 'var(--accent-purple)', 'var(--accent-gold)',
]

export function MSSTNode({ id, data }: NodeProps) {
  const { t } = useTranslation()
  const updateNodeData = useWorkflowStore((s) => s.updateNodeData)
  const nodeState = useExecutionStore((s) => s.nodeStates[id])
  const updateNodeInternals = useUpdateNodeInternals()
  const params = (data as any)?.params || {}

  const [models, setModels] = useState<any[]>([])
  const [stems, setStems] = useState<string[]>(['vocals', 'instrumental'])

  const fetchModels = useCallback(() => {
    getMSSTModels().then((all) => setModels(all.filter((m: any) => m.installed))).catch(console.error)
  }, [])

  useEffect(() => {
    fetchModels()
    return onModelsChanged(fetchModels)
  }, [fetchModels])

  const removeEdgesFromNode = useWorkflowStore((s) => s.removeEdgesFromSource)

  useEffect(() => {
    if (params.model_name) {
      getMSSTModelStems(params.model_name).then((r) => {
        const newStems = r.stems
        const currentEdges = useWorkflowStore.getState().edges
        const staleEdges = currentEdges.filter(
          (e) => e.source === id && e.sourceHandle && !newStems.includes(e.sourceHandle)
        )
        if (staleEdges.length > 0) {
          removeEdgesFromNode(staleEdges.map((e) => e.id))
        }
        setStems(newStems)
      }).catch(console.error)
    }
  }, [params.model_name, id, removeEdgesFromNode])

  useEffect(() => {
    updateNodeInternals(id)
  }, [stems, id, updateNodeInternals])

  return (
    <NodeShell id={id} title={t('node.msstSeparation')} color="var(--accent-blue)">
      <Handle type="target" position={Position.Left} id="audio"
        className="utai-handle" style={{ background: 'var(--accent-blue)' }} />

      <div style={{ marginBottom: 6 }}>
        <label className="utai-label">Model</label>
        <UtaiSelect
          value={params.model_name || ''}
          onChange={(v) => updateNodeData(id, { params: { ...params, model_name: v } })}
          options={[{ value: '', label: t('node.noModel') }, ...models.map((m) => ({ value: m.name, label: m.name }))]}
        />
      </div>

      <div style={{ marginBottom: 6 }}>
        <label className="utai-label" style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
          <input
            type="checkbox"
            checked={params.use_tta || false}
            onChange={(e) =>
              updateNodeData(id, { params: { ...params, use_tta: e.target.checked } })
            }
          />
          TTA
        </label>
      </div>

      <div style={{
        fontFamily: 'var(--font-mono)',
        fontSize: 10,
        color: 'var(--text-muted)',
        marginTop: 4,
        borderTop: '1px dotted var(--border-dim)',
        paddingTop: 4,
      }}>
        outputs: {stems.join(', ')}
      </div>

      {stems.map((stem, i) => {
        const bufferId = nodeState?.outputs?.[stem]
        return (
          <div key={stem} className="utai-stem-row" style={{ position: 'relative', marginTop: 4 }}>
            <div style={{ display: 'flex', justifyContent: 'flex-end', alignItems: 'center', gap: 4, paddingRight: 6 }}>
              {bufferId && <AudioPlayer bufferId={bufferId} compact />}
              <span className="utai-stem-label" style={{ color: STEM_COLORS[i % STEM_COLORS.length] }}>
                {stem}
              </span>
            </div>
            <Handle
              type="source"
              position={Position.Right}
              id={stem}
              className="utai-handle"
              style={{ background: STEM_COLORS[i % STEM_COLORS.length] }}
            />
          </div>
        )
      })}
    </NodeShell>
  )
}
