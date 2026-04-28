import { useCallback, useEffect, useState, useRef } from 'react'
import { Handle, Position, type NodeProps } from '@xyflow/react'
import { useTranslation } from 'react-i18next'
import { TrainNodeShell } from './TrainNodeShell'
import { useTrainingStore } from '../../store/trainingStore'
import {
  startFullTraining,
  stopTraining,
  connectTraining,
  getPretrainedModels,
  getExperiments,
  browseFolderAuto,
} from '../../api/client'
import { UtaiSelect } from '../../components/UtaiSelect'
import { ConfirmDialog } from '../../components/ConfirmDialog'

export function TrainRVCNode({ id, data }: NodeProps) {
  const { t } = useTranslation()
  const updateNodeData = useTrainingStore((s) => s.updateNodeData)
  const nodeState = useTrainingStore((s) => s.nodeStates[id])
  const setNodeState = useTrainingStore((s) => s.setNodeState)
  const addLog = useTrainingStore((s) => s.addLog)
  const setLogPanelOpen = useTrainingStore((s) => s.setLogPanelOpen)
  const setActiveTaskId = useTrainingStore((s) => s.setActiveTaskId)
  const activeTaskId = useTrainingStore((s) => s.activeTaskId)

  const params = (data as any)?.params || {}
  const [pretrained, setPretrained] = useState<{ G: string[]; D: string[] }>({ G: [], D: [] })
  const [expanded, setExpanded] = useState(false)
  const [confirmReplace, setConfirmReplace] = useState<((ok: boolean) => void) | null>(null)
  const wsRef = useRef<WebSocket | null>(null)

  useEffect(() => {
    getPretrainedModels().then(setPretrained).catch(console.error)
  }, [])

  // Clean up WebSocket on unmount
  useEffect(() => {
    return () => { wsRef.current?.close() }
  }, [])

  const setParam = (key: string, value: unknown) =>
    updateNodeData(id, { params: { ...params, [key]: value } })

  /** Resolve input directory and specific files from all incoming edges */
  const resolveInput = useCallback(() => {
    const { edges, nodeStates } = useTrainingStore.getState()
    const incomingEdges = edges.filter((e) => e.target === id && e.targetHandle === 'audio_files')
    if (incomingEdges.length === 0) return { dir: '', files: [] as string[] }

    let inputDir = ''
    const resolvedFiles: string[] = []

    for (const inEdge of incomingEdges) {
      const upState = nodeStates[inEdge.source]
      if (!upState?.outputData) continue
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
        if (idx >= 0 && idx < allFiles.length) {
          resolvedFiles.push(allFiles[idx])
        }
      } else {
        const files = upState.outputData.unroutedFiles || upState.outputData.files || []
        resolvedFiles.push(...files)
      }
    }
    return { dir: inputDir, files: resolvedFiles }
  }, [id])

  const handleBrowseExportDir = useCallback(async () => {
    try {
      const result = await browseFolderAuto(params.export_dir || '')
      const path = result?.path
      if (path) setParam('export_dir', path)
    } catch (err) {
      console.error('Browse folder failed:', err)
    }
  }, [params.export_dir])

  const handleRun = useCallback(async () => {
    const modelName = params.model_name
    if (!modelName) {
      setNodeState(id, { status: 'error', message: t('train.noModelName') })
      return
    }

    const { dir: inputDir, files: inputFiles } = resolveInput()
    if (!inputDir || inputFiles.length === 0) {
      const { edges, nodeStates } = useTrainingStore.getState()
      const inEdge = edges.find((e) => e.target === id)
      if (inEdge) {
        const upState = nodeStates[inEdge.source]
        if (!upState || upState.status !== 'completed') {
          setNodeState(id, { status: 'error', message: t('train.upstreamNotReady') })
          return
        }
      }
      setNodeState(id, { status: 'error', message: t('train.noInput') })
      return
    }

    addLog(`[rvc-train] Input: ${inputFiles.length} files from ${inputDir}`)
    const trainingStart = Date.now()

    const experiments = await getExperiments().catch(() => [])
    if (experiments.some((e: any) => e.name === modelName)) {
      const confirmed = await new Promise<boolean>((resolve) => {
        setConfirmReplace(() => resolve)
      })
      setConfirmReplace(null)
      if (!confirmed) {
        addLog(`[rvc-train] Cancelled: user chose not to replace "${modelName}"`)
        return
      }
    }

    setNodeState(id, { status: 'running', progress: 0, message: t('train.starting') })
    setLogPanelOpen(true)

    // Close any previous WebSocket
    wsRef.current?.close()

    const trainParams = {
      exp_name: modelName,
      input_dir: inputDir,
      files: inputFiles,
      sr: params.sr || '40k',
      version: params.version || 'v2',
      f0: params.f0 !== false,
      f0_method: params.f0_method || 'rmvpe',
      batch_size: params.batch_size || 4,
      total_epoch: params.total_epoch || 100,
      save_epoch: params.save_epoch || 10,
      gpu_ids: params.gpu_ids || '0',
      pretrained_g: params.pretrained_g || '',
      pretrained_d: params.pretrained_d || '',
      save_latest: params.save_latest !== false,
      cache_gpu: params.cache_gpu || false,
      save_weights: params.save_weights !== false,
      export_dir: params.export_dir || '',
    }

    try {
      addLog(`[rvc-train] Starting full pipeline for "${modelName}"`)
      const { task_id } = await startFullTraining(trainParams)
      setActiveTaskId(task_id)

      const ws = connectTraining(task_id, (msg) => {
        if (msg.type === 'log') {
          addLog(msg.message)
        } else if (msg.type === 'status') {
          setNodeState(id, {
            status: 'running',
            message: msg.step || t('train.running'),
            progress: msg.progress || 0,
          })
        } else if (msg.type === 'complete') {
          setNodeState(id, {
            status: msg.success ? 'completed' : 'error',
            message: msg.success ? t('train.done') : t('train.failed'),
          })
          setActiveTaskId(null)
          wsRef.current = null
          ws.close()
          useTrainingStore.getState().openReport({
            expName: params.model_name || '',
            success: msg.success,
            startTime: trainingStart,
          })
        }
      })

      // Handle unexpected disconnect
      ws.onclose = () => {
        wsRef.current = null
        const currentState = useTrainingStore.getState().nodeStates[id]
        if (currentState?.status === 'running') {
          setNodeState(id, { status: 'error', message: 'Connection lost' })
          setActiveTaskId(null)
        }
      }

      wsRef.current = ws
    } catch (err) {
      setNodeState(id, { status: 'error', message: String(err) })
      addLog(`[rvc-train] Error: ${err}`)
    }
  }, [id, params])

  const handleStop = useCallback(async () => {
    if (activeTaskId) {
      await stopTraining(activeTaskId).catch(console.error)
      addLog('[rvc-train] Stopping...')
    }
  }, [activeTaskId])

  const isTraining = nodeState?.status === 'running'

  return (
    <TrainNodeShell
      id={id}
      title={t('train.node.rvcTrain')}
      color="var(--accent-pink)"
      onRun={handleRun}
      onStop={handleStop}
      running={isTraining}
    >
      <Handle type="target" position={Position.Left} id="audio_files"
        className="utai-handle" style={{ background: 'var(--accent-pink)' }} />

      {/* --- Model Name (was "实验名称") --- */}
      <div style={{ marginBottom: 4 }}>
        <label className="utai-label">{t('train.modelName')}</label>
        <input
          className="utai-input"
          value={params.model_name || ''}
          onChange={(e) => setParam('model_name', e.target.value)}
          placeholder={t('train.modelNameHint')}
        />
      </div>

      <div style={{ display: 'flex', gap: 4, marginBottom: 4 }}>
        <div style={{ flex: 1 }}>
          <label className="utai-label">{t('train.sr')}</label>
          <UtaiSelect
            value={params.sr || '40k'}
            onChange={(v) => setParam('sr', v)}
            options={[
              { value: '32k', label: '32k' },
              { value: '40k', label: '40k' },
              { value: '48k', label: '48k' },
            ]}
          />
        </div>
        <div style={{ flex: 1 }}>
          <label className="utai-label">{t('train.version')}</label>
          <UtaiSelect
            value={params.version || 'v2'}
            onChange={(v) => setParam('version', v)}
            options={[
              { value: 'v2', label: 'v2' },
              { value: 'v1', label: 'v1' },
            ]}
          />
        </div>
      </div>

      <label className="utai-label" style={{ display: 'flex', alignItems: 'center', gap: 6, marginBottom: 4 }}>
        <input type="checkbox" checked={params.f0 !== false} onChange={(e) => setParam('f0', e.target.checked)} />
        {t('train.f0Guidance')}
      </label>

      <div style={{ marginBottom: 4 }}>
        <label className="utai-label">{t('train.f0Method')}</label>
        <UtaiSelect
          value={params.f0_method || 'rmvpe'}
          onChange={(v) => setParam('f0_method', v)}
          options={[
            { value: 'rmvpe', label: 'RMVPE' },
            { value: 'harvest', label: 'Harvest' },
            { value: 'pm', label: 'PM' },
            { value: 'dio', label: 'DIO' },
          ]}
        />
      </div>

      <div style={{ marginBottom: 4 }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
          <label className="utai-label" style={{ flex: 1 }}>{t('train.epochs')}</label>
          <input type="number" min={1} max={2000} step={1}
            value={params.total_epoch || 100}
            onChange={(e) => {
              const v = Math.max(1, Math.min(2000, Number(e.target.value) || 1))
              setParam('total_epoch', v)
            }}
            className="utai-input"
            style={{ width: 64, textAlign: 'center', padding: '1px 4px', fontSize: 11 }}
          />
        </div>
        <input type="range" min={1} max={2000} step={1}
          value={params.total_epoch || 100}
          onChange={(e) => setParam('total_epoch', Number(e.target.value))}
          style={{ width: '100%' }}
        />
      </div>

      <div style={{ marginBottom: 4 }}>
        <label className="utai-label">{t('train.batchSize')}: {params.batch_size || 4}</label>
        <input type="range" min={1} max={32} step={1}
          value={params.batch_size || 4}
          onChange={(e) => setParam('batch_size', Number(e.target.value))}
          style={{ width: '100%' }}
        />
      </div>

      {/* --- Advanced settings (expandable) --- */}
      <button className="utai-train-expand-btn" onClick={() => setExpanded(!expanded)}>
        {expanded ? '▾' : '▸'} {t('train.advanced')}
      </button>

      {expanded && (
        <div className="utai-train-advanced">
          <div style={{ marginBottom: 4 }}>
            <label className="utai-label">{t('train.saveEvery')}: {params.save_epoch || 10}</label>
            <input type="range" min={1} max={100} step={1}
              value={params.save_epoch || 10}
              onChange={(e) => setParam('save_epoch', Number(e.target.value))}
              style={{ width: '100%' }}
            />
          </div>

          <div style={{ marginBottom: 4 }}>
            <label className="utai-label">{t('train.gpu')}</label>
            <input className="utai-input" value={params.gpu_ids || '0'}
              onChange={(e) => setParam('gpu_ids', e.target.value)} placeholder="0" />
          </div>

          <div style={{ marginBottom: 4 }}>
            <label className="utai-label">Pretrained G</label>
            <UtaiSelect
              value={params.pretrained_g || ''}
              onChange={(v) => setParam('pretrained_g', v)}
              options={[{ value: '', label: t('train.autoPretrained') }, ...pretrained.G.map((p) => ({ value: p, label: p.split(/[/\\]/).pop() || p }))]}
            />
          </div>

          <div style={{ marginBottom: 4 }}>
            <label className="utai-label">Pretrained D</label>
            <UtaiSelect
              value={params.pretrained_d || ''}
              onChange={(v) => setParam('pretrained_d', v)}
              options={[{ value: '', label: t('train.autoPretrained') }, ...pretrained.D.map((p) => ({ value: p, label: p.split(/[/\\]/).pop() || p }))]}
            />
          </div>

          <label className="utai-label" style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
            <input type="checkbox" checked={params.save_latest !== false}
              onChange={(e) => setParam('save_latest', e.target.checked)} />
            {t('train.saveLatest')}
          </label>

          <label className="utai-label" style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
            <input type="checkbox" checked={params.cache_gpu || false}
              onChange={(e) => setParam('cache_gpu', e.target.checked)} />
            {t('train.cacheGpu')}
          </label>

          <label className="utai-label" style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
            <input type="checkbox" checked={params.save_weights !== false}
              onChange={(e) => setParam('save_weights', e.target.checked)} />
            {t('train.saveWeights')}
          </label>

          <div style={{ marginTop: 4 }}>
            <label className="utai-label">{t('train.exportDir')}</label>
            <div style={{ display: 'flex', gap: 2, marginTop: 2 }}>
              <input
                className="utai-input"
                style={{ flex: 1, margin: 0 }}
                value={params.export_dir || ''}
                onChange={(e) => setParam('export_dir', e.target.value)}
                placeholder={t('train.exportDirHint')}
              />
              <button
                className="utai-train-browse-btn"
                onClick={handleBrowseExportDir}
                title={t('train.browse')}
              >
                ...
              </button>
            </div>
          </div>
        </div>
      )}

      {isTraining && (
        <button
          className="utai-btn utai-btn--clear"
          style={{ width: '100%', marginTop: 4, fontSize: 11 }}
          onClick={handleStop}
        >
          {t('train.stop')}
        </button>
      )}

      <Handle type="source" position={Position.Right} id="model"
        className="utai-handle" style={{ background: 'var(--accent-pink)' }} />

      <ConfirmDialog
        open={!!confirmReplace}
        message={`"${params.model_name || ''}" ${t('train.alreadyExists') || 'already exists. Replace existing model?'}`}
        onConfirm={() => { if (confirmReplace) confirmReplace(true) }}
        onCancel={() => { if (confirmReplace) confirmReplace(false) }}
      />
    </TrainNodeShell>
  )
}
