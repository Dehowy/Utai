import { useCallback, useEffect, useMemo, useState } from 'react'
import { Handle, Position, useUpdateNodeInternals, useEdges, type NodeProps } from '@xyflow/react'
import { useTranslation } from 'react-i18next'
import { TrainNodeShell } from './TrainNodeShell'
import { useTrainingStore } from '../../store/trainingStore'
import { getModelCategories, getModelsByCategory, getMSSTModelStems, startBatchSeparation, stopTraining, connectTraining } from '../../api/client'
import { onModelsChanged } from '../../utils/modelEvents'
import { FileAudioPlayer } from '../../components/FileAudioPlayer'
import { UtaiSelect } from '../../components/UtaiSelect'

const STEM_COLORS = [
  'var(--accent-green)', 'var(--accent-orange)', 'var(--accent-pink)',
  'var(--accent-cyan)', 'var(--accent-purple)', 'var(--accent-gold)',
]

export function TrainMSSTBatchNode({ id, data }: NodeProps) {
  const { t, i18n } = useTranslation()
  const updateNodeData = useTrainingStore((s) => s.updateNodeData)
  const setNodeState = useTrainingStore((s) => s.setNodeState)
  const addLog = useTrainingStore((s) => s.addLog)
  const setLogPanelOpen = useTrainingStore((s) => s.setLogPanelOpen)
  const updateNodeInternals = useUpdateNodeInternals()

  const nodeState = useTrainingStore((s) => s.nodeStates[id])

  const params = (data as any)?.params || {}
  const [categories, setCategories] = useState<any[]>([])
  const [selectedCat, setSelectedCat] = useState(params.category || '')
  const [models, setModels] = useState<any[]>([])
  const [stems, setStems] = useState<string[]>([])
  const [taskId, setTaskId] = useState<string | null>(null)
  const allEdges = useEdges()

  const lang = i18n.language as 'zh' | 'ja' | 'en'

  // Get completed output files per stem
  const stemFiles: Record<string, string[]> = nodeState?.outputData?.stemFiles || {}
  const connectedHandles = useMemo(() => new Set(
    allEdges.filter((e) => e.source === id).map((e) => e.sourceHandle)
  ), [allEdges, id])

  const truncName = (name: string, max = 20) => {
    if (name.length <= max) return name
    const ext = name.lastIndexOf('.')
    if (ext > 0 && name.length - ext <= 5) {
      const extStr = name.slice(ext)
      const keep = max - extStr.length - 2
      return keep > 0 ? name.slice(0, keep) + '..' + extStr : name.slice(0, max - 2) + '..'
    }
    return name.slice(0, max - 2) + '..'
  }

  // Fetch categories + models (and refresh when models change)
  const fetchAll = useCallback(() => {
    getModelCategories().then((cats) => {
      setCategories(cats.filter((c: any) => c.installed_count > 0))
    }).catch(console.error)
    const cat = selectedCat || (params as any).category || ''
    if (cat) {
      getModelsByCategory(cat).then((all) => {
        setModels(all.filter((m: any) => m.installed))
      }).catch(console.error)
    }
  }, [selectedCat, params])

  useEffect(() => {
    fetchAll()
    return onModelsChanged(fetchAll)
  }, [fetchAll])

  // Fetch stems for selected model
  useEffect(() => {
    if (!params.model_name) {
      setStems([])
      return
    }
    const model = models.find((m) => m.name === params.model_name)
    if (model?.model_class === 'VR_Models') {
      setStems([model.primary_stem || 'primary', model.secondary_stem || 'secondary'])
    } else {
      getMSSTModelStems(params.model_name)
        .then((r) => setStems(r.stems || []))
        .catch(() => setStems([]))
    }
  }, [params.model_name, models])

  // Update node internals when stems or output files change (so handles appear)
  useEffect(() => {
    updateNodeInternals(id)
  }, [stems, stemFiles, id, updateNodeInternals])

  const setParam = (key: string, value: unknown) =>
    updateNodeData(id, { params: { ...params, [key]: value } })

  const getCatLabel = (cat: any) => cat[`label_${lang}`] || cat.label_zh || cat.id

  const handleCatChange = (catId: string) => {
    setSelectedCat(catId)
    setParam('category', catId)
    setParam('model_name', '')
  }

  const handleRun = useCallback(async () => {
    const { edges, nodeStates } = useTrainingStore.getState()
    const incomingEdges = edges.filter((e) => e.target === id && e.targetHandle === 'audio_files')
    if (incomingEdges.length === 0) {
      setNodeState(id, { status: 'error', message: t('train.noInput') })
      return
    }

    let inputDir = ''
    const resolvedFiles: string[] = []

    for (const inEdge of incomingEdges) {
      const upState = nodeStates[inEdge.source]
      if (!upState || upState.status !== 'completed' || !upState.outputData) {
        setNodeState(id, { status: 'error', message: t('train.upstreamNotReady') })
        return
      }
      inputDir = inputDir || upState.outputData.dir || ''
      const srcHandle = inEdge.sourceHandle || 'audio_files'
      const stemFileMatch = srcHandle.match(/^stem_(.+)_file_(\d+)$/)
      if (stemFileMatch) {
        const stemName = stemFileMatch[1]
        const fileIdx = parseInt(stemFileMatch[2], 10)
        const upStemFiles: Record<string, string[]> = upState.outputData.stemFiles || {}
        const stemList = upStemFiles[stemName] || []
        if (fileIdx >= 0 && fileIdx < stemList.length) resolvedFiles.push(stemList[fileIdx])
      } else if (srcHandle.startsWith('stem_')) {
        const stemName = srcHandle.replace('stem_', '')
        const upStemFiles: Record<string, string[]> = upState.outputData.stemFiles || {}
        const stemList = upStemFiles[stemName] || []
        resolvedFiles.push(...stemList)
      } else if (srcHandle.startsWith('file_')) {
        const idx = parseInt(srcHandle.replace('file_', ''), 10)
        const allFiles: string[] = upState.outputData.files || []
        if (idx >= 0 && idx < allFiles.length) resolvedFiles.push(allFiles[idx])
      } else {
        const files = upState.outputData.unroutedFiles || upState.outputData.files || []
        resolvedFiles.push(...files)
      }
    }

    if (!params.model_name) {
      setNodeState(id, { status: 'error', message: t('node.noModel') })
      return
    }

    // Resolve to full paths (upstream outputs filenames only, not full paths)
    const fullPaths = resolvedFiles.map((f) =>
      f.includes('/') || f.includes('\\') ? f : `${inputDir}/${f}`.replace(/\//g, '\\')
    )

    setNodeState(id, { status: 'running', message: t('train.batchProcessing') })
    setLogPanelOpen(true)
    addLog(`[msst-batch] Starting batch separation with ${params.model_name}`)
    addLog(`[msst-batch] Input: ${fullPaths.length} files from ${inputDir}`)

    try {
      const { task_id } = await startBatchSeparation({
        model_name: params.model_name,
        input_dir: inputDir,
        files: fullPaths,
        use_tta: params.use_tta || false,
        window_size: params.window_size || 512,
        aggression: params.aggression || 5,
        post_process: params.post_process || false,
      })
      setTaskId(task_id)

      await new Promise<void>((resolve, reject) => {
        const ws = connectTraining(task_id, (msg) => {
          if (msg.type === 'log') {
            addLog(msg.message)
          } else if (msg.type === 'status') {
            setNodeState(id, {
              status: 'running',
              message: msg.step || 'Separating...',
              progress: msg.progress || 0,
            })
          } else if (msg.type === 'complete') {
            if (msg.success) {
              setNodeState(id, {
                status: 'completed',
                message: `${resolvedFiles.length} files × ${(msg.stems || []).length} stems`,
                outputData: {
                  dir: msg.output_dir || inputDir,
                  files: resolvedFiles,
                  stemFiles: msg.stem_files || {},
                },
              })
              resolve()
            } else {
              setNodeState(id, { status: 'error', message: 'Separation failed' })
              reject(new Error('Batch separation failed'))
            }
            setTaskId(null)
            ws.close()
          }
        })
      })
    } catch (err) {
      addLog(`[msst-batch] Error: ${err}`)
      setNodeState(id, { status: 'error', message: String(err) })
      setTaskId(null)
    }
  }, [id, params, stems])

  const handleStop = useCallback(async () => {
    if (!taskId) return
    addLog('[msst-batch] Stopping...')
    await stopTraining(taskId).catch(console.error)
  }, [taskId])

  return (
    <TrainNodeShell id={id} title={t('train.node.msstBatch')} color="var(--accent-blue)" onRun={handleRun} onStop={handleStop}>
      <Handle type="target" position={Position.Left} id="audio_files"
        className="utai-handle" style={{ background: 'var(--accent-blue)' }} />

      {/* Category selector */}
      <div style={{ marginBottom: 6 }}>
        <label className="utai-label">{t('palette.separation')}</label>
        <UtaiSelect
          value={selectedCat}
          onChange={(v) => handleCatChange(v)}
          options={[{ value: '', label: '---' }, ...categories.map((cat) => ({ value: cat.id, label: getCatLabel(cat) }))]}
        />
      </div>

      {/* Model selector */}
      <div style={{ marginBottom: 6 }}>
        <label className="utai-label">{t('train.model')}</label>
        <UtaiSelect
          value={params.model_name || ''}
          onChange={(v) => setParam('model_name', v)}
          options={[{ value: '', label: t('node.noModel') }, ...models.map((m) => ({ value: m.name, label: m.name }))]}
        />
      </div>

      <label className="utai-label" style={{ display: 'flex', alignItems: 'center', gap: 6, marginBottom: 4 }}>
        <input
          type="checkbox"
          checked={params.use_tta || false}
          onChange={(e) => setParam('use_tta', e.target.checked)}
        />
        TTA
      </label>

      {/* Dynamic output stems with per-stem handles */}
      {stems.length > 0 && (
        <div style={{
          borderTop: '1px dotted var(--border-dim)',
          paddingTop: 4,
          marginTop: 4,
        }}>
          <div style={{ fontFamily: 'var(--font-mono)', fontSize: 9, color: 'var(--text-muted)', marginBottom: 4 }}>
            outputs:
          </div>

          {/* Main "all stems" handle */}
          <div className="utai-stem-row" style={{ position: 'relative', marginBottom: 2 }}>
            <div style={{ display: 'flex', justifyContent: 'flex-end', alignItems: 'center', paddingRight: 6 }}>
              <span className="utai-stem-label" style={{ color: 'var(--accent-blue)', fontSize: 9 }}>
                {t('train.allStems')}
              </span>
            </div>
            <Handle type="source" position={Position.Right} id="audio_files"
              className="utai-handle" style={{ background: 'var(--accent-blue)' }} />
          </div>

          {/* Per-stem handles + per-file sub-handles */}
          {stems.map((stem, i) => {
            const stemColor = STEM_COLORS[i % STEM_COLORS.length]
            const files = stemFiles[stem] || []
            return (
              <div key={stem}>
                {/* Stem group handle */}
                <div className="utai-stem-row" style={{ position: 'relative', marginBottom: 2 }}>
                  <div style={{ display: 'flex', justifyContent: 'flex-end', alignItems: 'center', paddingRight: 6 }}>
                    <span className="utai-stem-label" style={{ color: stemColor, fontSize: 9 }}>
                      {stem} {files.length > 0 && `(${files.length})`}
                    </span>
                  </div>
                  <Handle type="source" position={Position.Right} id={`stem_${stem}`}
                    className="utai-handle" style={{ background: stemColor }} />
                </div>

                {/* Per-file handles within this stem */}
                {files.map((filePath, fi) => {
                  const handleId = `stem_${stem}_file_${fi}`
                  const fileName = filePath.split(/[/\\]/).pop() || filePath
                  const isConnected = connectedHandles.has(handleId)
                  return (
                    <div key={handleId} className="utai-stem-row" style={{
                      position: 'relative',
                      marginBottom: 1,
                      paddingLeft: 12,
                      opacity: isConnected ? 1 : 0.6,
                    }}>
                      <div style={{ display: 'flex', justifyContent: 'flex-end', alignItems: 'center', paddingRight: 6 }}>
                        <span className="utai-stem-label" style={{
                          color: 'var(--text-primary, #e8e8f0)',
                          fontSize: 8,
                          maxWidth: 160,
                          overflow: 'hidden',
                          textOverflow: 'ellipsis',
                          whiteSpace: 'nowrap',
                        }} title={fileName}>
                          {truncName(fileName, 24)}
                        </span>
                      </div>
                      <Handle type="source" position={Position.Right} id={handleId}
                        className="utai-handle utai-handle--file"
                        style={{ background: isConnected ? 'var(--accent-cyan)' : stemColor }} />
                    </div>
                  )
                })}

                {/* Audio preview for each file of this stem */}
                {files.map((fp, pi) => (
                  <div key={`preview_${stem}_${pi}`} style={{ paddingLeft: 12, marginTop: 1, marginBottom: 1 }}>
                    <FileAudioPlayer filePath={fp} />
                  </div>
                ))}
              </div>
            )
          })}
        </div>
      )}

      {/* Fallback: single output if no model selected */}
      {stems.length === 0 && (
        <Handle type="source" position={Position.Right} id="audio_files"
          className="utai-handle" style={{ background: 'var(--accent-blue)' }} />
      )}
    </TrainNodeShell>
  )
}
