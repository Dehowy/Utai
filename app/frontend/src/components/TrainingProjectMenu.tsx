import { useState, useRef, useEffect, useCallback } from 'react'
import { useTranslation } from 'react-i18next'
import { useNavigate } from 'react-router-dom'
import { useTrainingStore } from '../store/trainingStore'
import { useWorkflowStore } from '../store/workflowStore'
import { exportWorkflowAuto, importWorkflowAuto } from '../api/client'
import { showToast } from './Toast'
import { utaiConfirm } from './UtaiAlert'

function buildTrainingData(name: string) {
  const { nodes, edges } = useTrainingStore.getState()
  return {
    version: '1.0',
    name,
    project_type: 'training',
    nodes: nodes.map((n) => ({
      id: n.id, type: n.type!, position: n.position, params: n.data?.params || {},
    })),
    connections: edges.map((e) => ({
      from_node: e.source, from_port: e.sourceHandle || 'audio_files',
      to_node: e.target, to_port: e.targetHandle || 'audio_files',
    })),
  }
}

export async function saveTraining(saveAs = false) {
  const store = useTrainingStore.getState()
  if (store.isSaving) { showToast('Save in progress...', 'info'); return }
  store.setIsSaving(true)
  try {
    const wf = buildTrainingData(store.workflowName)
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

export function TrainingProjectMenu() {
  const { t } = useTranslation()
  const navigate = useNavigate()
  const [open, setOpen] = useState(false)
  const menuRef = useRef<HTMLDivElement>(null)
  const dirty = useTrainingStore(s => s.dirty)

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
    useTrainingStore.getState().clearTraining()
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
      } else {
        useWorkflowStore.getState().setWorkflow(data.name || 'Imported', rfNodes, rfEdges)
        if (data.path) useWorkflowStore.getState().setProjectPath(data.path)
        navigate('/')
      }
      showToast('Project opened', 'success')
    } catch (err: any) {
      showToast(`Open failed: ${err.message}`, 'error')
    }
  }

  const handleSave = () => { setOpen(false); saveTraining() }
  const handleSaveAs = () => { setOpen(false); saveTraining(true) }

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
