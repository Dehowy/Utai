import { useCallback, useEffect, useState, type DragEvent } from 'react'
import { useTranslation } from 'react-i18next'
import { Link } from 'react-router-dom'
import { useAppStore } from '../store/appStore'
import {
  ReactFlow,
  Background,
  Controls,
  useReactFlow,
  type NodeTypes,
  type Connection,
  type Edge,
  ReactFlowProvider,
} from '@xyflow/react'
import '@xyflow/react/dist/style.css'

import { useTrainingStore } from '../store/trainingStore'
import { TrainAudioImportNode } from '../nodes/training/TrainAudioImportNode'
import { TrainMSSTBatchNode } from '../nodes/training/TrainMSSTBatchNode'
import { TrainRVCNode } from '../nodes/training/TrainRVCNode'
import { TrainingLogPanel } from '../components/TrainingLogPanel'
// TrainingChart is now embedded in TrainingLogPanel
import { TrainingReportModal } from '../components/TrainingReportModal'
import { TrainContextMenu } from '../components/TrainContextMenu'
import { LangSwitcher } from '../components/LangSwitcher'
import { ModelManager } from '../components/ModelManager'
import { ConfirmDialog } from '../components/ConfirmDialog'
import { TrainingProjectMenu } from '../components/TrainingProjectMenu'
import { ToastContainer } from '../components/Toast'
import { startFullTraining, startBatchSeparation, connectTraining, stopTraining, cleanupTemp, cleanupBuffers, cleanupAllTrainingData } from '../api/client'

const trainNodeTypes: NodeTypes = {
  train_audio_import: TrainAudioImportNode,
  train_msst_batch: TrainMSSTBatchNode,
  train_rvc: TrainRVCNode,
}

interface PaletteItem {
  type: string
  labelKey: string
  color: string
}

const TRAIN_PALETTE: PaletteItem[] = [
  { type: 'train_audio_import', labelKey: 'train.node.audioImport', color: 'var(--accent-gold)' },
  { type: 'train_msst_batch', labelKey: 'train.node.msstBatch', color: 'var(--accent-blue)' },
  { type: 'train_rvc', labelKey: 'train.node.rvcTrain', color: 'var(--accent-pink)' },
]

let trainNodeCounter = 0

function TrainFlow() {
  const { t } = useTranslation()
  const hasSynth = useAppStore(s => s.modules?.synth) !== false
  const {
    nodes, edges, onNodesChange, onEdgesChange, onConnect,
    addNode, removeNode, removeEdge,
    isRunningAll, setRunningAll,
    nodeStates, setNodeState, resetAllNodeStates,
    addLog, clearLogs, setLogPanelOpen, logPanelOpen,
    setActiveTaskId,
  } = useTrainingStore()

  const { screenToFlowPosition } = useReactFlow()

  const [contextMenu, setContextMenu] = useState<{
    x: number; y: number; nodeId: string | null; edgeId: string | null
  } | null>(null)
  const [modelManagerOpen, setModelManagerOpen] = useState(false)
  const [reportOpen, setReportOpen] = useState(false)
  const [reportData, setReportData] = useState({ expName: '', success: false, startTime: 0 })
  const [confirmClear, setConfirmClear] = useState(false)
  const activeTaskId = useTrainingStore((s) => s.activeTaskId)
  const logs = useTrainingStore((s) => s.logs)
  const clearTraining = useTrainingStore((s) => s.clearTraining)

  // Auto-clear training page when buffers/datasets have been cleaned (e.g. app restart)
  useEffect(() => {
    const importNodes = nodes.filter((n) => n.type === 'train_audio_import')
    const hasStalePaths = importNodes.some((n) => {
      const dir = (n.data as any)?.params?.dataset_dir
      return dir && typeof dir === 'string' && dir.includes('/temp/')
    })
    if (hasStalePaths && importNodes.length > 0) {
      // Check if the dataset dir still exists by trying to fetch it
      const firstDir = (importNodes[0].data as any)?.params?.dataset_dir
      if (firstDir) {
        fetch(`/api/train/dataset-dir?exp_name=_check_stale`)
          .then((r) => r.json())
          .catch(() => {
            // Server temp was cleaned — auto-reset training page
            clearTraining()
          })
      }
    }
  }, []) // only on mount

  const handleClearConfirmed = useCallback(async () => {
    clearTraining()
    await cleanupTemp().catch(console.error)
    await cleanupBuffers().catch(console.error)
    await cleanupAllTrainingData().catch(console.error)
    setConfirmClear(false)
  }, [clearTraining])

  const handleStopAll = useCallback(async () => {
    const tid = useTrainingStore.getState().activeTaskId
    if (tid) {
      addLog('[execute-all] Stopping current task...')
      await stopTraining(tid).catch(console.error)
    }
  }, [addLog])

  const handleContextMenu = useCallback((event: React.MouseEvent) => {
    event.preventDefault()
    const target = event.target as HTMLElement
    const nodeEl = target.closest('.react-flow__node')
    const nodeId = nodeEl?.getAttribute('data-id') || null
    const edgeEl = target.closest('.react-flow__edge')
    const edgeId = nodeId ? null : (edgeEl?.getAttribute('data-id') || null)
    setContextMenu({ x: event.clientX, y: event.clientY, nodeId, edgeId })
  }, [])

  const handlePaneClick = useCallback(() => {
    setContextMenu(null)
  }, [])

  const getTopZIndex = useCallback(() => {
    // Read directly from store to avoid stale closure
    const currentNodes = useTrainingStore.getState().nodes
    let max = 0
    for (const n of currentNodes) {
      const z = (n.style?.zIndex as number) || (n.zIndex as number) || 0
      if (z > max) max = z
    }
    return max + 1
  }, [])

  const handleAddNode = useCallback((type: string) => {
    trainNodeCounter++
    addNode({
      id: `train_${Date.now()}_${trainNodeCounter}`,
      type,
      position: { x: 200 + trainNodeCounter * 40, y: 150 + trainNodeCounter * 30 },
      data: { params: {} },
      zIndex: getTopZIndex() + 1000,
    })
  }, [addNode, getTopZIndex])

  const handleDrop = useCallback((e: DragEvent) => {
    e.preventDefault()
    const type = e.dataTransfer.getData('application/utai-train-node-type')
    if (!type) return
    trainNodeCounter++
    const position = screenToFlowPosition({ x: e.clientX, y: e.clientY })
    addNode({
      id: `train_${Date.now()}_${trainNodeCounter}`,
      type,
      position,
      data: { params: {} },
      zIndex: getTopZIndex() + 1000,
    })
  }, [screenToFlowPosition, addNode, getTopZIndex])

  const handleDragOver = useCallback((e: DragEvent) => {
    e.preventDefault()
    e.dataTransfer.dropEffect = 'move'
  }, [])

  const handleDragStart = (e: React.DragEvent, type: string) => {
    e.dataTransfer.setData('application/utai-train-node-type', type)
    e.dataTransfer.effectAllowed = 'move'
  }

  /** Execute all nodes in topological order with smart skip */
  const handleExecuteAll = useCallback(async () => {
    if (isRunningAll) return
    setRunningAll(true)
    clearLogs()
    setLogPanelOpen(true)
    addLog('[execute-all] Starting full training pipeline...')

    // Simple topological sort: find nodes with no incoming edges first
    const nodeIds = nodes.map((n) => n.id)
    const incoming = new Map<string, string[]>()
    for (const nid of nodeIds) incoming.set(nid, [])
    for (const e of edges) {
      incoming.get(e.target)?.push(e.source)
    }

    // Kahn's algorithm with cycle detection
    const order: string[] = []
    const inDegree = new Map<string, number>()
    for (const nid of nodeIds) inDegree.set(nid, 0)
    for (const e of edges) {
      inDegree.set(e.target, (inDegree.get(e.target) || 0) + 1)
    }

    const queue = nodeIds.filter((nid) => (inDegree.get(nid) || 0) === 0)

    while (queue.length > 0) {
      const nid = queue.shift()!
      order.push(nid)
      for (const e of edges) {
        if (e.source === nid) {
          const deg = (inDegree.get(e.target) || 1) - 1
          inDegree.set(e.target, deg)
          if (deg === 0) queue.push(e.target)
        }
      }
    }

    if (order.length !== nodeIds.length) {
      addLog('[execute-all] Error: cycle detected in workflow')
      setRunningAll(false)
      return
    }

    // Execute each node in order
    for (const nodeId of order) {
      const node = nodes.find((n) => n.id === nodeId)
      if (!node) continue

      // Skip already-completed nodes
      const state = useTrainingStore.getState().nodeStates[nodeId]
      if (state?.status === 'completed') {
        addLog(`[execute-all] Skipping ${node.type} (already completed)`)
        continue
      }

      addLog(`[execute-all] Running ${node.type}...`)

      if (node.type === 'train_audio_import') {
        // Import node: check if it has files loaded, compute routed vs unrouted
        const params = (node.data as any)?.params || {}
        if (params.files?.length > 0) {
          // Determine which files have individual handles connected
          const perFileEdges = edges.filter(
            (e) => e.source === nodeId && e.sourceHandle?.startsWith('file_')
          )
          const connectedIndices = new Set(
            perFileEdges.map((e) => parseInt(e.sourceHandle!.replace('file_', ''), 10))
          )
          // Main handle gets all files whose individual handles are NOT connected
          const unroutedFiles = params.files.filter((_: string, i: number) => !connectedIndices.has(i))

          setNodeState(nodeId, {
            status: 'completed',
            message: `${params.files.length} files`,
            outputData: {
              dir: params.dataset_dir,
              files: params.files,
              unroutedFiles,
              connectedIndices: Array.from(connectedIndices),
            },
          })
        } else {
          setNodeState(nodeId, { status: 'error', message: 'No files uploaded' })
          addLog('[execute-all] Import node has no files - stopping')
          break
        }
      } else if (node.type === 'train_msst_batch' || node.type === 'train_rvc') {
        // Resolve input files from all incoming edges with smart routing
        const incomingEdges = edges.filter((e) => e.target === nodeId && e.targetHandle === 'audio_files')
        let inputDir = ''
        const resolvedFiles: string[] = []

        for (const inEdge of incomingEdges) {
          const upState = useTrainingStore.getState().nodeStates[inEdge.source]
          if (!upState?.outputData) continue
          inputDir = inputDir || upState.outputData.dir || ''
          const srcHandle = inEdge.sourceHandle || 'audio_files'

          // Handle MSST stem sub-routes: stem_{stemName}_file_{index}
          const stemFileMatch = srcHandle.match(/^stem_(.+)_file_(\d+)$/)
          if (stemFileMatch) {
            const stemName = stemFileMatch[1]
            const fileIdx = parseInt(stemFileMatch[2], 10)
            const stemFiles: Record<string, string[]> = upState.outputData.stemFiles || {}
            const stemList = stemFiles[stemName] || []
            if (fileIdx >= 0 && fileIdx < stemList.length) {
              resolvedFiles.push(stemList[fileIdx]) // full path from separation output
            }
          } else if (srcHandle.startsWith('stem_')) {
            // Stem group handle: stem_{stemName} — only push files for this specific stem
            const stemName = srcHandle.replace('stem_', '')
            const stemFiles: Record<string, string[]> = upState.outputData.stemFiles || {}
            const stemList = stemFiles[stemName] || []
            resolvedFiles.push(...stemList)
          } else if (srcHandle.startsWith('file_')) {
            const idx = parseInt(srcHandle.replace('file_', ''), 10)
            const allFiles: string[] = upState.outputData.files || []
            if (idx >= 0 && idx < allFiles.length) {
              resolvedFiles.push(allFiles[idx])
            }
          } else {
            // Main handle (audio_files): if upstream has stemFiles, push all stem files; otherwise use files
            const stemFiles: Record<string, string[]> = upState.outputData.stemFiles || {}
            const allStemPaths = Object.values(stemFiles).flat()
            if (allStemPaths.length > 0) {
              resolvedFiles.push(...allStemPaths)
            } else {
              const files = upState.outputData.unroutedFiles || upState.outputData.files || []
              resolvedFiles.push(...files)
            }
          }
        }

        if (node.type === 'train_msst_batch') {
          const batchParams = (node.data as any)?.params || {}
          if (!batchParams.model_name) {
            setNodeState(nodeId, { status: 'error', message: 'No model selected' })
            addLog('[execute-all] MSST batch node missing model - stopping')
            break
          }

          // Resolve to full paths (upstream outputs filenames only)
          const fullPaths = resolvedFiles.map((f: string) =>
            f.includes('/') || f.includes('\\') ? f : `${inputDir}/${f}`
          )

          addLog(`[execute-all] MSST batch: ${fullPaths.length} files with ${batchParams.model_name}`)
          setNodeState(nodeId, { status: 'running', progress: 0, message: 'Starting separation...' })

          try {
            const { task_id } = await startBatchSeparation({
              model_name: batchParams.model_name,
              input_dir: inputDir,
              files: fullPaths,
              use_tta: batchParams.use_tta || false,
              window_size: batchParams.window_size || 512,
              aggression: batchParams.aggression || 5,
              post_process: batchParams.post_process || false,
            })
            setActiveTaskId(task_id)

            await new Promise<void>((resolve, reject) => {
              const ws = connectTraining(task_id, (msg) => {
                if (msg.type === 'log') {
                  addLog(msg.message)
                } else if (msg.type === 'status') {
                  setNodeState(nodeId, {
                    status: 'running',
                    message: msg.step || 'Separating...',
                    progress: msg.progress || 0,
                  })
                } else if (msg.type === 'complete') {
                  setActiveTaskId(null)
                  if (msg.success) {
                    setNodeState(nodeId, {
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
                    setNodeState(nodeId, { status: 'error', message: 'Separation failed' })
                    reject(new Error('Batch separation failed'))
                  }
                  ws.close()
                }
              })
            })
          } catch (err) {
            setNodeState(nodeId, { status: 'error', message: String(err) })
            addLog(`[execute-all] MSST batch failed: ${err}`)
            break
          }
        } else {
          // train_rvc — full training pipeline
          const params = (node.data as any)?.params || {}
          const modelName = params.model_name || ''

          if (!modelName || (!inputDir && resolvedFiles.length === 0)) {
            setNodeState(nodeId, { status: 'error', message: !modelName ? 'Missing model name' : 'No input data' })
            addLog(`[execute-all] RVC node missing ${!modelName ? 'model name' : 'input data'} - stopping`)
            break
          }

          // Resolve to full paths (upstream may output filenames only)
          const fullPaths = resolvedFiles.map((f: string) =>
            f.includes('/') || f.includes('\\') ? f : `${inputDir}/${f}`
          )

          addLog(`[execute-all] RVC input: ${fullPaths.length} files from ${inputDir}`)
          setNodeState(nodeId, { status: 'running', progress: 0, message: 'Starting...' })

          const trainingStartTime = Date.now()

          try {
            const trainParams = {
              exp_name: modelName,
              input_dir: inputDir,
              files: fullPaths, // pass explicit file paths for flat-dir preparation
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

            const { task_id } = await startFullTraining(trainParams)
            setActiveTaskId(task_id)

            // Wait for training to complete via WebSocket
            await new Promise<void>((resolve, reject) => {
              const ws = connectTraining(task_id, (msg) => {
                if (msg.type === 'log') {
                  addLog(msg.message)
                } else if (msg.type === 'status') {
                  setNodeState(nodeId, {
                    status: 'running',
                    message: msg.step || 'Training...',
                    progress: msg.progress || 0,
                  })
                } else if (msg.type === 'complete') {
                  setNodeState(nodeId, {
                    status: msg.success ? 'completed' : 'error',
                    message: msg.success ? 'Training complete' : 'Training failed',
                  })
                  setActiveTaskId(null)
                  ws.close()
                  // Show training report modal
                  setReportData({ expName: modelName, success: msg.success, startTime: trainingStartTime })
                  setReportOpen(true)
                  if (msg.success) resolve()
                  else reject(new Error('Training failed'))
                }
              })
            })
          } catch (err) {
            setNodeState(nodeId, { status: 'error', message: String(err) })
            addLog(`[execute-all] RVC training failed: ${err}`)
            break
          }
        }
      }
    }

    setRunningAll(false)
    addLog('[execute-all] Pipeline finished')
  }, [nodes, edges, isRunningAll])

  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if (window.location.pathname !== '/train') return
      if ((e.ctrlKey || e.metaKey) && e.key === 's') {
        e.preventDefault()
        import('../components/TrainingProjectMenu').then(m => m.saveTraining())
      }
    }
    window.addEventListener('keydown', handler, true)
    return () => window.removeEventListener('keydown', handler, true)
  }, [])

  return (
    <div style={{ width: '100%', height: '100%', display: 'flex', flexDirection: 'column' }}>
      {/* Toolbar */}
      <div className="utai-toolbar">
        <Link to="/" className="utai-logo" style={{ textDecoration: 'none' }}>
          <span className="utai-logo-bracket">「</span>utai<span className="utai-logo-bracket">」</span>
        </Link>

        <span className="utai-toolbar-divider">|</span>
        <span style={{ color: 'var(--accent-pink)', fontFamily: 'var(--font-pixel)', fontSize: 12 }}>
          {t('train.title')}
        </span>

        <TrainingProjectMenu />

        <button className="utai-btn utai-btn--load" onClick={() => setModelManagerOpen(true)}>
          {t('toolbar.models')}
        </button>

        <button className="utai-btn utai-btn--clear" onClick={() => setConfirmClear(true)}>
          {t('toolbar.clear')}
        </button>

        <button
          className="utai-btn utai-btn--execute"
          disabled={isRunningAll || nodes.length === 0}
          onClick={handleExecuteAll}
        >
          {isRunningAll ? t('status.running') + '...' : t('toolbar.executeAll')}
        </button>

        {isRunningAll && activeTaskId && (
          <button
            className="utai-btn utai-train-node-stop"
            style={{ padding: '4px 10px', fontSize: 12 }}
            onClick={handleStopAll}
            title={t('status.stop')}
          >
            ■ {t('status.stop')}
          </button>
        )}

        <button
          className="utai-btn utai-btn--logs"
          onClick={() => { setLogPanelOpen(!logPanelOpen) }}
        >
          {t('workflow.logs')}
        </button>

        <div style={{ flex: 1 }} />

        {hasSynth && (
          <Link to="/synth" className="utai-btn utai-btn--synth" style={{ textDecoration: 'none' }}>
            {t('toolbar.synth')}
          </Link>
        )}

        <Link to="/" className="utai-btn utai-btn--load" style={{ textDecoration: 'none' }}>
          {t('train.backToWorkflow')}
        </Link>

        <LangSwitcher />
      </div>

      <ModelManager open={modelManagerOpen} onClose={() => setModelManagerOpen(false)} />
      <TrainingReportModal
        open={reportOpen}
        onClose={() => setReportOpen(false)}
        expName={reportData.expName}
        success={reportData.success}
        startTime={reportData.startTime}
      />

      <div style={{ flex: 1, display: 'flex', overflow: 'hidden', minHeight: 0 }}>
        {/* Palette */}
        <div className="utai-palette">
          <div className="utai-palette__title">{t('train.paletteTitle')}</div>

          {TRAIN_PALETTE.map((item) => (
            <button
              key={item.type}
              className="utai-palette__item"
              onClick={() => handleAddNode(item.type)}
              draggable
              onDragStart={(e) => handleDragStart(e, item.type)}
              style={{ color: item.color, borderColor: item.color }}
            >
              {t(item.labelKey)}
            </button>
          ))}
        </div>

        {/* Flow canvas */}
        <div className="utai-flow-area" style={{ flex: 1 }} onContextMenu={handleContextMenu}>
          <ReactFlow
            nodes={nodes}
            edges={edges}
            onNodesChange={onNodesChange}
            onEdgesChange={onEdgesChange}
            onConnect={onConnect}
            isValidConnection={(connection: Edge | Connection) => {
              if (connection.source === connection.target) return false
              const visited = new Set<string>()
              const queue = [connection.target!]
              while (queue.length > 0) {
                const nid = queue.shift()!
                if (nid === connection.source) return false
                if (visited.has(nid)) continue
                visited.add(nid)
                for (const e of edges) {
                  if (e.source === nid) queue.push(e.target)
                }
              }
              return true
            }}
            onPaneClick={handlePaneClick}
            onDrop={handleDrop}
            onDragOver={handleDragOver}
            nodeTypes={trainNodeTypes}
            fitView
            snapToGrid
            snapGrid={[16, 16]}
            deleteKeyCode={['Backspace', 'Delete']}
            edgesFocusable
            defaultEdgeOptions={{
              selectable: true,
              style: { strokeWidth: 1.5, stroke: '#3a3a5a' },
              focusable: true,
            }}
            style={{ background: 'var(--bg-deep)' }}
          >
            <Background color="#1a1a35" gap={20} size={1} />
            <Controls />
          </ReactFlow>

          {contextMenu && (
            <TrainContextMenu
              x={contextMenu.x}
              y={contextMenu.y}
              nodeId={contextMenu.nodeId}
              edgeId={contextMenu.edgeId}
              onClose={() => setContextMenu(null)}
              onDeleteNode={(id) => { removeNode(id); setContextMenu(null) }}
              onDeleteEdge={(id) => { removeEdge(id); setContextMenu(null) }}
              onAddNode={(type, sx, sy) => {
                trainNodeCounter++
                const position = screenToFlowPosition({ x: sx, y: sy })
                addNode({
                  id: `train_${Date.now()}_${trainNodeCounter}`,
                  type,
                  position,
                  data: { params: {} },
                  zIndex: getTopZIndex() + 1000,
                })
                setContextMenu(null)
              }}
            />
          )}

          <div className="utai-counter">★ utai v0.1.0 — training</div>
        </div>

        {/* Log panel (right) */}
        <TrainingLogPanel />
      </div>

      <ConfirmDialog
        open={confirmClear}
        message={t('dialog.confirmClearTraining')}
        onConfirm={handleClearConfirmed}
        onCancel={() => setConfirmClear(false)}
      />
    </div>
  )
}

export default function TrainingPage() {
  return (
    <ReactFlowProvider>
      <TrainFlow />
      <ToastContainer />
    </ReactFlowProvider>
  )
}
