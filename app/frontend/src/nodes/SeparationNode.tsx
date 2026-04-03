import { useEffect, useState, useCallback } from 'react'
import { Handle, Position, useUpdateNodeInternals, type NodeProps } from '@xyflow/react'
import { useTranslation } from 'react-i18next'
import { NodeShell } from './NodeShell'
import { useWorkflowStore } from '../store/workflowStore'
import { useExecutionStore } from '../store/executionStore'
import { getModelsByCategory, getMSSTModelStems } from '../api/client'
import { AudioPlayer } from '../components/AudioPlayer'
import { UtaiSelect } from '../components/UtaiSelect'
import { onModelsChanged } from '../utils/modelEvents'

const STEM_COLORS = [
  'var(--accent-green)', 'var(--accent-orange)', 'var(--accent-pink)',
  'var(--accent-cyan)', 'var(--accent-purple)', 'var(--accent-gold)',
]

/** Category display labels — must match backend CATEGORIES */
const CATEGORY_LABELS: Record<string, { zh: string; ja: string; en: string; color: string }> = {
  vocal:        { zh: '提取人声', ja: 'ボーカル抽出', en: 'Extract Vocals', color: 'var(--accent-green)' },
  instrumental: { zh: '提取伴奏', ja: '伴奏抽出',     en: 'Extract Instrumental', color: 'var(--accent-blue)' },
  denoise:      { zh: '去噪',     ja: 'ノイズ除去',   en: 'Denoise', color: 'var(--accent-cyan)' },
  dereverb:     { zh: '去混响',   ja: 'リバーブ除去', en: 'Dereverb', color: 'var(--accent-purple)' },
  karaoke:      { zh: '卡拉OK',   ja: 'カラオケ',     en: 'Karaoke', color: 'var(--accent-pink)' },
  multi_stem:   { zh: '多轨道',   ja: 'マルチトラック', en: 'Multi-stem', color: 'var(--accent-orange)' },
  other:        { zh: '其他',     ja: 'その他',       en: 'Other', color: 'var(--accent-gold)' },
}

export function SeparationNode({ id, data }: NodeProps) {
  const { t, i18n } = useTranslation()
  const updateNodeData = useWorkflowStore((s) => s.updateNodeData)
  const removeEdgesFromNode = useWorkflowStore((s) => s.removeEdgesFromSource)
  const nodeState = useExecutionStore((s) => s.nodeStates[id])
  const updateNodeInternals = useUpdateNodeInternals()

  const params = (data as any)?.params || {}
  const category: string = params.category || 'vocal'

  const [models, setModels] = useState<any[]>([])
  const [stems, setStems] = useState<string[]>(['vocals', 'instrumental'])

  // Fetch models for this category (and refresh when models change)
  const fetchModels = useCallback(() => {
    getModelsByCategory(category)
      .then((all) => setModels(all.filter((m: any) => m.installed)))
      .catch(console.error)
  }, [category])

  useEffect(() => {
    fetchModels()
    return onModelsChanged(fetchModels)
  }, [fetchModels])

  // Fetch stems when model changes
  useEffect(() => {
    if (!params.model_name) return
    const model = models.find((m) => m.name === params.model_name)
    if (model?.model_class === 'VR_Models') {
      // VR always has primary/secondary
      const s = [model.primary_stem || 'Vocals', model.secondary_stem || 'Instrumental']
      setStems(s)
    } else if (params.model_name) {
      getMSSTModelStems(params.model_name)
        .then((r) => {
          const newStems = r.stems
          const currentEdges = useWorkflowStore.getState().edges
          const staleEdges = currentEdges.filter(
            (e) => e.source === id && e.sourceHandle && !newStems.includes(e.sourceHandle)
          )
          if (staleEdges.length > 0) removeEdgesFromNode(staleEdges.map((e) => e.id))
          setStems(newStems)
        })
        .catch(console.error)
    }
  }, [params.model_name, id, models, removeEdgesFromNode])

  useEffect(() => {
    updateNodeInternals(id)
  }, [stems, id, updateNodeInternals])

  const setParam = useCallback((key: string, value: unknown) => {
    updateNodeData(id, { params: { ...params, [key]: value } })
  }, [id, params, updateNodeData])

  const catInfo = CATEGORY_LABELS[category] || CATEGORY_LABELS.vocal
  const lang = i18n.language as 'zh' | 'ja' | 'en'
  const catLabel = catInfo[lang] || catInfo.zh
  const catColor = catInfo.color

  const selectedModel = models.find((m) => m.name === params.model_name)
  const isVR = selectedModel?.model_class === 'VR_Models'

  return (
    <NodeShell id={id} title={catLabel} color={catColor}>
      <Handle type="target" position={Position.Left} id="audio"
        className="utai-handle" style={{ background: catColor }} />

      <div style={{ marginBottom: 6 }}>
        <label className="utai-label">Model</label>
        <UtaiSelect
          value={params.model_name || ''}
          onChange={(v) => setParam('model_name', v)}
          options={[{ value: '', label: models.length > 0 ? t('node.noModel') : t('models.noInstalled') }, ...models.map((m) => ({ value: m.name, label: m.name }))]}
        />
      </div>

      {/* Common: TTA */}
      <label className="utai-label" style={{ display: 'flex', alignItems: 'center', gap: 6, marginBottom: 4 }}>
        <input
          type="checkbox"
          checked={params.use_tta || false}
          onChange={(e) => setParam('use_tta', e.target.checked)}
        />
        TTA
      </label>

      {/* VR-specific controls */}
      {isVR && (
        <>
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
            <input type="range" min={0} max={20} step={1}
              value={params.aggression ?? 5}
              onChange={(e) => setParam('aggression', Number(e.target.value))}
              style={{ width: '100%' }} />
          </div>
          <label className="utai-label" style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
            <input type="checkbox" checked={params.post_process || false}
              onChange={(e) => setParam('post_process', e.target.checked)} />
            Post Process
          </label>
        </>
      )}

      {/* Output stems */}
      <div style={{
        fontFamily: 'var(--font-mono)', fontSize: 10,
        color: 'var(--text-muted)', marginTop: 4,
        borderTop: '1px dotted var(--border-dim)', paddingTop: 4,
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
            <Handle type="source" position={Position.Right} id={stem}
              className="utai-handle" style={{ background: STEM_COLORS[i % STEM_COLORS.length] }} />
          </div>
        )
      })}
    </NodeShell>
  )
}
