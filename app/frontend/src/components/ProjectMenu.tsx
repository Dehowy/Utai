import { useState, useRef, useEffect, useCallback } from 'react'
import { useTranslation } from 'react-i18next'
import { useNavigate } from 'react-router-dom'
import { useWorkflowStore } from '../store/workflowStore'
import { useExecutionStore } from '../store/executionStore'
import { useTrainingStore } from '../store/trainingStore'
import { exportWorkflowAuto, importWorkflowAuto, cleanupBuffers } from '../api/client'

interface ProjectMenuProps {
  onOpenManager?: () => void
}

export function ProjectMenu({ onOpenManager }: ProjectMenuProps) {
  const { t } = useTranslation()
  const navigate = useNavigate()
  const [open, setOpen] = useState(false)
  const [showSaveAs, setShowSaveAs] = useState(false)
  const [saveAsName, setSaveAsName] = useState('')
  const menuRef = useRef<HTMLDivElement>(null)
  const { nodes, edges, workflowName, setWorkflow } = useWorkflowStore()

  useEffect(() => {
    if (!open) return
    const handle = (e: MouseEvent) => {
      if (menuRef.current && !menuRef.current.contains(e.target as Node)) {
        setOpen(false)
        setShowSaveAs(false)
      }
    }
    document.addEventListener('mousedown', handle)
    return () => document.removeEventListener('mousedown', handle)
  }, [open])

  const buildWorkflowData = useCallback((name: string) => {
    const execStates = useExecutionStore.getState().nodeStates
    return {
      version: '1.0',
      name,
      project_type: 'workflow',
      nodes: nodes.map((n) => {
        let params = n.data?.params || {}
        // For yt-dlp nodes with completed execution, inject buffer_id so
        // backend can convert them to audio_input nodes with bundled audio
        if (n.type === 'ytdlp_download') {
          const execState = execStates[n.id]
          if (execState?.status === 'completed' && execState.outputs?.audio) {
            params = { ...params, buffer_id: execState.outputs.audio }
          }
        }
        return {
          id: n.id,
          type: n.type!,
          position: n.position,
          params,
        }
      }),
      connections: edges.map((e) => ({
        from_node: e.source,
        from_port: e.sourceHandle || 'audio',
        to_node: e.target,
        to_port: e.targetHandle || 'audio',
      })),
    }
  }, [nodes, edges])

  const handleExport = useCallback(async () => {
    const wf = buildWorkflowData(workflowName)
    try {
      await exportWorkflowAuto(wf)
    } catch (err) {
      console.error('Export failed:', err)
    }
    setOpen(false)
  }, [workflowName, buildWorkflowData])

  const handleExportAs = useCallback(async () => {
    setSaveAsName(workflowName === 'Untitled' ? '' : workflowName)
    setShowSaveAs(true)
  }, [workflowName])

  const handleExportAsConfirm = useCallback(async () => {
    const name = saveAsName.trim()
    if (!name) return
    const wf = buildWorkflowData(name)
    try {
      await exportWorkflowAuto(wf)
      // Update workflow name in store
      setWorkflow(name, nodes, edges)
    } catch (err) {
      console.error('Export failed:', err)
    }
    setShowSaveAs(false)
    setOpen(false)
  }, [saveAsName, buildWorkflowData, setWorkflow, nodes, edges])

  const handleImport = useCallback(async () => {
    try {
      const data = await importWorkflowAuto()
      if (data.status === 'cancelled') {
        setOpen(false)
        return
      }
      // Note: do NOT cleanupBuffers here — the import already restored buffers
      // from the .utai file. Old buffers are cleared by startup or 清空 button.

      const projectType = data.project_type || 'workflow'

      const rfNodes = (data.nodes || []).map((n: any) => ({
        id: n.id,
        type: n.type,
        position: n.position,
        data: { params: n.params || {} },
      }))
      const rfEdges = (data.connections || []).map((c: any, i: number) => ({
        id: `e-${i}`,
        source: c.from_node,
        sourceHandle: c.from_port,
        target: c.to_node,
        targetHandle: c.to_port,
        type: 'smoothstep',
        animated: true,
      }))

      if (projectType === 'training') {
        // It's a training project — load into training store and navigate
        useTrainingStore.getState().setWorkflow(data.name || 'Imported', rfNodes, rfEdges)
        navigate('/train')
      } else {
        // Workflow project — load into current page
        setWorkflow(data.name || 'Imported', rfNodes, rfEdges)
      }
    } catch (err) {
      console.error('Import failed:', err)
    }
    setOpen(false)
  }, [setWorkflow, navigate])

  return (
    <div ref={menuRef} style={{ position: 'relative' }}>
      <button
        className="utai-btn utai-btn--project"
        onClick={() => { setOpen(!open); setShowSaveAs(false) }}
      >
        {t('toolbar.project')}
      </button>

      {open && (
        <div className="utai-dropdown">
          <button className="utai-dropdown__item" onClick={handleExportAs}>
            {t('project.exportAs')}
          </button>
          <div className="utai-dropdown__divider" />
          <button className="utai-dropdown__item" onClick={handleImport}>
            {t('project.import')}
          </button>

          {showSaveAs && (
            <div className="utai-dropdown__input-row">
              <input
                className="utai-dropdown__input"
                placeholder={t('project.namePlaceholder')}
                value={saveAsName}
                onChange={(e) => setSaveAsName(e.target.value)}
                onKeyDown={(e) => e.key === 'Enter' && handleExportAsConfirm()}
                autoFocus
              />
              <button className="utai-btn utai-dropdown__confirm" onClick={handleExportAsConfirm}>
                OK
              </button>
            </div>
          )}
        </div>
      )}
    </div>
  )
}
