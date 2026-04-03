import { useCallback, useEffect, useState } from 'react'
import { Handle, Position, type NodeProps } from '@xyflow/react'
import { useTranslation } from 'react-i18next'
import { NodeShell } from './NodeShell'
import { useWorkflowStore } from '../store/workflowStore'
import { useExecutionStore } from '../store/executionStore'
import { getRVCModels } from '../api/client'
import { AudioPlayer } from '../components/AudioPlayer'
import { UtaiSelect } from '../components/UtaiSelect'
import { onModelsChanged } from '../utils/modelEvents'

export function RVCNode({ id, data }: NodeProps) {
  const { t } = useTranslation()
  const updateNodeData = useWorkflowStore((s) => s.updateNodeData)
  const nodeState = useExecutionStore((s) => s.nodeStates[id])
  const params = (data as any)?.params || {}

  const [models, setModels] = useState<any[]>([])

  const fetchModels = useCallback(() => {
    getRVCModels().then(setModels).catch(console.error)
  }, [])

  useEffect(() => {
    fetchModels()
    return onModelsChanged(fetchModels)
  }, [fetchModels])

  const setParam = (key: string, value: unknown) =>
    updateNodeData(id, { params: { ...params, [key]: value } })

  const selectedModel = models.find((m) => m.name === params.model_name)
  const outputBufferId = nodeState?.outputs?.audio

  return (
    <NodeShell id={id} title={t('node.rvcConversion')} color="var(--accent-pink)">
      <Handle type="target" position={Position.Left} id="audio"
        className="utai-handle" style={{ background: 'var(--accent-pink)' }} />

      <div style={{ marginBottom: 6 }}>
        <label className="utai-label">Voice Model</label>
        <UtaiSelect
          value={params.model_name || ''}
          onChange={(v) => setParam('model_name', v)}
          options={[{ value: '', label: t('node.noModel') }, ...models.map((m) => ({ value: m.name, label: m.name }))]}
        />
      </div>

      {selectedModel?.index_files?.length > 0 && (
        <div style={{ marginBottom: 6 }}>
          <label className="utai-label">Index</label>
          <UtaiSelect
            value={params.index_file || ''}
            onChange={(v) => setParam('index_file', v)}
            options={[{ value: '', label: 'None' }, ...selectedModel.index_files.map((f: string) => ({ value: f, label: f.split(/[/\\]/).pop() || f }))]}
          />
        </div>
      )}

      <div style={{ marginBottom: 4 }}>
        <label className="utai-label">Pitch: {params.f0_up_key ?? 0} st</label>
        <input
          type="range" min={-24} max={24} step={1}
          value={params.f0_up_key ?? 0}
          onChange={(e) => setParam('f0_up_key', Number(e.target.value))}
          style={{ width: '100%' }}
        />
      </div>

      <div style={{ marginBottom: 4 }}>
        <label className="utai-label">F0 Method</label>
        <UtaiSelect
          value={params.f0_method || 'rmvpe'}
          onChange={(v) => setParam('f0_method', v)}
          options={[
            { value: 'rmvpe', label: 'RMVPE' },
            { value: 'harvest', label: 'Harvest' },
            { value: 'crepe', label: 'Crepe' },
            { value: 'pm', label: 'PM' },
          ]}
        />
      </div>

      <div style={{ marginBottom: 4 }}>
        <label className="utai-label">Index Rate: {(params.index_rate ?? 0.75).toFixed(2)}</label>
        <input
          type="range" min={0} max={1} step={0.05}
          value={params.index_rate ?? 0.75}
          onChange={(e) => setParam('index_rate', Number(e.target.value))}
          style={{ width: '100%' }}
        />
      </div>

      <div style={{ marginBottom: 4 }}>
        <label className="utai-label">RMS Mix: {(params.rms_mix_rate ?? 0.25).toFixed(2)}</label>
        <input
          type="range" min={0} max={1} step={0.05}
          value={params.rms_mix_rate ?? 0.25}
          onChange={(e) => setParam('rms_mix_rate', Number(e.target.value))}
          style={{ width: '100%' }}
        />
      </div>

      <div style={{ marginBottom: 4 }}>
        <label className="utai-label">Protect: {(params.protect ?? 0.33).toFixed(2)}</label>
        <input
          type="range" min={0} max={0.5} step={0.01}
          value={params.protect ?? 0.33}
          onChange={(e) => setParam('protect', Number(e.target.value))}
          style={{ width: '100%' }}
        />
      </div>

      {outputBufferId && <AudioPlayer bufferId={outputBufferId} />}

      <Handle type="source" position={Position.Right} id="audio"
        className="utai-handle" style={{ background: 'var(--accent-pink)' }} />
    </NodeShell>
  )
}
