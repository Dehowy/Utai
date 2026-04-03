import { useState, useRef, useEffect, useCallback } from 'react'
import { useTranslation } from 'react-i18next'
import { useNavigate } from 'react-router-dom'
import { useTrainingStore } from '../store/trainingStore'
import { useWorkflowStore } from '../store/workflowStore'
import { exportWorkflowAuto, importWorkflowAuto } from '../api/client'

export function TrainingProjectMenu() {
  const { t } = useTranslation()
  const navigate = useNavigate()
  const [open, setOpen] = useState(false)
  const [showSaveAs, setShowSaveAs] = useState(false)
  const [saveAsName, setSaveAsName] = useState('')
  const menuRef = useRef<HTMLDivElement>(null)
  const { nodes, edges, workflowName, setWorkflow } = useTrainingStore()

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

  const buildWorkflowData = useCallback((name: string) => ({
    version: '1.0',
    name,
    project_type: 'training',
    nodes: nodes.map((n) => ({
      id: n.id,
      type: n.type!,
      position: n.position,
      params: n.data?.params || {},
    })),
    connections: edges.map((e) => ({
      from_node: e.source,
      from_port: e.sourceHandle || 'audio_files',
      to_node: e.target,
      to_port: e.targetHandle || 'audio_files',
    })),
  }), [nodes, edges])

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
        // Load into training store (current page)
        setWorkflow(data.name || 'Imported', rfNodes, rfEdges)
      } else {
        // It's a workflow project — load into workflow store and navigate
        useWorkflowStore.getState().setWorkflow(data.name || 'Imported', rfNodes, rfEdges)
        navigate('/')
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
