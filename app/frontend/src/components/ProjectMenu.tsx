import { useState, useRef, useEffect, useCallback } from 'react'
import { useTranslation } from 'react-i18next'
import { useNavigate } from 'react-router-dom'
import { useWorkflowStore } from '../store/workflowStore'
import { useExecutionStore } from '../store/executionStore'
import { useTrainingStore } from '../store/trainingStore'
import { exportWorkflowAuto, importWorkflowAuto } from '../api/client'
import { showToast } from './Toast'
import { utaiConfirm } from './UtaiAlert'

interface ProjectMenuProps {
  onOpenManager?: () => void
}

function buildWorkflowData(name: string) {
  const { nodes, edges } = useWorkflowStore.getState()
  const execStates = useExecutionStore.getState().nodeStates
  return {
    version: '1.0',
    name,
    project_type: 'workflow',
    nodes: nodes.map((n) => {
      let params = n.data?.params || {}
      if (n.type === 'ytdlp_download') {
        const execState = execStates[n.id]
        if (execState?.status === 'completed' && execState.outputs?.audio) {
          params = { ...params, buffer_id: execState.outputs.audio }
        }
      }
      return { id: n.id, type: n.type!, position: n.position, params }
    }),
    connections: edges.map((e) => ({
      from_node: e.source, from_port: e.sourceHandle || 'audio',
      to_node: e.target, to_port: e.targetHandle || 'audio',
    })),
  }
}

export async function saveWorkflow(saveAs = false) {
  const store = useWorkflowStore.getState()
  if (store.isSaving) { showToast('Save in progress...', 'info'); return }
  store.setIsSaving(true)
  try {
    const wf = buildWorkflowData(store.workflowName)
    const path = saveAs ? '' : (store.projectPath || '')

    if (path) {
      const res = await fetch('/api/workflows/export-to-path', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ path, workflow: wf }),
      })
      if (!res.ok) throw new Error(await res.text())
      store.markClean()
      showToast(`Saved to ${path}`, 'success')
    } else {
      const result = await exportWorkflowAuto(wf)
      if (result && (result as any).path) {
        if (!saveAs) store.setProjectPath((result as any).path)
        store.markClean()
        showToast(`Saved to ${(result as any).path}`, 'success')
      } else if (result && (result as any).status !== 'cancelled') {
        store.markClean()
        showToast('Project saved', 'success')
      }
    }
  } catch (err: any) {
    showToast(`Save failed: ${err.message}`, 'error')
  } finally {
    store.setIsSaving(false)
  }
}

export function ProjectMenu({ onOpenManager }: ProjectMenuProps) {
  const { t } = useTranslation()
  const navigate = useNavigate()
  const [open, setOpen] = useState(false)
  const menuRef = useRef<HTMLDivElement>(null)
  const dirty = useWorkflowStore(s => s.dirty)
  const projectPath = useWorkflowStore(s => s.projectPath)

  useEffect(() => {
    if (!open) return
    const handle = (e: MouseEvent) => {
      if (menuRef.current && !menuRef.current.contains(e.target as Node)) setOpen(false)
    }
    document.addEventListener('mousedown', handle)
    return () => document.removeEventListener('mousedown', handle)
  }, [open])

  const handleNew = async () => {
    setOpen(false)
    if (dirty && !(await utaiConfirm(t('synth.discardChanges')))) return
    useWorkflowStore.getState().clearWorkflow()
    useExecutionStore.getState().reset()
    showToast('New project', 'info')
  }

  const handleOpen = async () => {
    setOpen(false)
    if (dirty && !(await utaiConfirm(t('synth.discardChanges')))) return
    try {
      const data = await importWorkflowAuto()
      if (data.status === 'cancelled') return
      const projectType = data.project_type || 'workflow'
      if (projectType === 'synth') {
        const { applyOpenedProject } = await import('./synth/SynthProjectMenu')
        applyOpenedProject(data)
        navigate('/synth')
        showToast('Synth project opened', 'success')
        return
      }
      const rfNodes = (data.nodes || []).map((n: any) => ({
        id: n.id, type: n.type, position: n.position, data: { params: n.params || {} },
      }))
      const rfEdges = (data.connections || []).map((c: any, i: number) => ({
        id: `e-${i}`, source: c.from_node, sourceHandle: c.from_port,
        target: c.to_node, targetHandle: c.to_port, type: 'smoothstep', animated: true,
      }))
      if (projectType === 'training') {
        useTrainingStore.getState().setWorkflow(data.name || 'Imported', rfNodes, rfEdges)
        if (data.path) useTrainingStore.getState().setProjectPath(data.path)
        navigate('/train')
      } else {
        useWorkflowStore.getState().setWorkflow(data.name || 'Imported', rfNodes, rfEdges)
        if (data.path) useWorkflowStore.getState().setProjectPath(data.path)
      }
      showToast('Project opened', 'success')
    } catch (err: any) {
      showToast(`Open failed: ${err.message}`, 'error')
    }
  }

  const handleSave = () => { setOpen(false); saveWorkflow() }
  const handleSaveAs = () => { setOpen(false); saveWorkflow(true) }

  return (
    <div ref={menuRef} style={{ position: 'relative' }}>
      <button className="utai-btn utai-btn--project" onClick={() => setOpen(!open)}
        style={{ borderColor: dirty ? 'var(--accent-pink)' : undefined, color: dirty ? 'var(--accent-pink)' : undefined }}>
        {t('toolbar.project')}
      </button>

      {open && (
        <div className="utai-dropdown">
          <button className="utai-dropdown__item" onClick={handleNew}>{t('project.new', 'New')}</button>
          <button className="utai-dropdown__item" onClick={handleOpen}>{t('project.open', 'Open...')}</button>
          <div className="utai-dropdown__divider" />
          <button className="utai-dropdown__item" onClick={handleSave} style={{ display: 'flex', justifyContent: 'space-between' }}>
            <span>{t('project.save', 'Save')}</span>
            <span style={{ fontSize: 9, color: 'var(--text-muted)', marginLeft: 16 }}>Ctrl+S</span>
          </button>
          <button className="utai-dropdown__item" onClick={handleSaveAs}>{t('project.saveAs', 'Save As...')}</button>
        </div>
      )}
    </div>
  )
}
