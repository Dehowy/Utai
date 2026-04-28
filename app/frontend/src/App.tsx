import { useEffect, useState } from 'react'
import { BrowserRouter, useNavigate, useLocation } from 'react-router-dom'
import WorkflowPage from './pages/WorkflowPage'
import TrainingPage from './pages/TrainingPage'
import SynthPage from './pages/SynthPage'
import { importWorkflowFromPath } from './api/client'
import { useWorkflowStore } from './store/workflowStore'
import { useTrainingStore } from './store/trainingStore'
import { useAppStore } from './store/appStore'
import { CloseDialog } from './components/CloseDialog'

function AppContent() {
  const navigate = useNavigate()
  const location = useLocation()
  const modules = useAppStore(s => s.modules)
  const defaultPage = useAppStore(s => s.defaultPage)

  const [visited, setVisited] = useState({ workflow: true, train: false, synth: false })
  const [didInitNav, setDidInitNav] = useState(false)
  const [closeDialog, setCloseDialog] = useState<{ open: boolean; fromTray: boolean }>({ open: false, fromTray: false })

  // Load modules on mount
  useEffect(() => {
    useAppStore.getState().loadPreferences()
    useAppStore.getState().loadModules()
  }, [])

  // Track visited pages
  useEffect(() => {
    if (location.pathname === '/train' && !visited.train) setVisited(v => ({ ...v, train: true }))
    if (location.pathname === '/synth' && !visited.synth) setVisited(v => ({ ...v, synth: true }))
  }, [location.pathname])

  // Navigate to default page once modules loaded
  useEffect(() => {
    if (modules && !didInitNav) {
      setDidInitNav(true)
      if (location.pathname === '/' && defaultPage !== '/' && defaultPage) {
        navigate(defaultPage, { replace: true })
      }
    }
  }, [modules, didInitNav])

  // Block browser context menus
  useEffect(() => {
    const handler = (e: MouseEvent) => {
      const t = e.target as HTMLElement;
      if (t.tagName === 'INPUT' || t.tagName === 'TEXTAREA' || t.isContentEditable) return;
      e.preventDefault();
    };
    document.addEventListener('contextmenu', handler);
    return () => document.removeEventListener('contextmenu', handler);
  }, []);

  // Block browser shortcuts
  useEffect(() => {
    const onKeyDown = (e: KeyboardEvent) => {
      const tag = (e.target as HTMLElement)?.tagName;
      const isInput = tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT';
      if (e.key === 'F5') { e.preventDefault(); return; }
      if (e.key === 'F12') { e.preventDefault(); return; }
      if (e.ctrlKey || e.metaKey) {
        const k = e.key.toLowerCase();
        if (isInput && ['a', 'c', 'v', 'x'].includes(k)) return;
        e.preventDefault(); return;
      }
      if (e.altKey) { e.preventDefault(); return; }
      if (e.key === ' ' && !isInput) { e.preventDefault(); return; }
    };
    const onKeyUp = (e: KeyboardEvent) => { if (e.key === 'Alt') e.preventDefault(); };
    document.addEventListener('keydown', onKeyDown);
    document.addEventListener('keyup', onKeyUp);
    return () => { document.removeEventListener('keydown', onKeyDown); document.removeEventListener('keyup', onKeyUp); };
  }, []);

  // Handle .utai / .utaiSynth file opening
  useEffect(() => {
    const api = window.electronAPI
    if (!api?.onOpenUtaiFile) return
    const cleanup = api.onOpenUtaiFile(async (filePath: string) => {
      try {
        if (filePath.toLowerCase().endsWith('.utaisynth')) {
          const res = await fetch('/api/synth/project/open-path', {
            method: 'POST', headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ path: filePath }),
          })
          if (!res.ok) throw new Error(await res.text())
          const data = await res.json()
          const { applyOpenedProject } = await import('./components/synth/SynthProjectMenu')
          applyOpenedProject(data)
          navigate('/synth')
          return
        }
        const data = await importWorkflowFromPath(filePath)
        if (!data || data.status === 'cancelled') return
        const projectType = data.project_type || 'workflow'
        const rfNodes = (data.nodes || []).map((n: any) => ({
          id: n.id, type: n.type, position: n.position, data: { params: n.params || {} },
        }))
        const rfEdges = (data.connections || []).map((c: any, i: number) => ({
          id: `e-${i}`, source: c.from_node, sourceHandle: c.from_port,
          target: c.to_node, targetHandle: c.to_port, type: 'smoothstep', animated: true,
        }))
        if (projectType === 'training') {
          useTrainingStore.getState().setWorkflow(data.name || 'Imported', rfNodes, rfEdges)
          navigate('/train')
        } else {
          useWorkflowStore.getState().setWorkflow(data.name || 'Imported', rfNodes, rfEdges)
          navigate('/')
        }
      } catch (err) {
        console.error('[utai] Failed to open file:', err)
      }
    })
    return cleanup || undefined
  }, [navigate])

  // Handle close-requested from Electron
  useEffect(() => {
    const api = window.electronAPI
    if (!api?.onCloseRequested) return
    const cleanup = api.onCloseRequested((data) => {
      setCloseDialog({ open: true, fromTray: data.fromTray })
    })
    return cleanup || undefined
  }, [])

  // === ALL HOOKS ABOVE THIS LINE === //

  // Wait for modules before rendering pages
  if (!modules) return <div style={{ background: '#08080f', width: '100vw', height: '100vh' }} />

  const hasSynth = modules.synth !== false
  const hasAudio = modules.audio_tools !== false
  const isActive = (p: string) => location.pathname === p

  return (
    <>
      {hasAudio && (
        <div key="page-workflow" style={{ display: isActive('/') ? 'contents' : 'none' }}>
          <WorkflowPage />
        </div>
      )}
      {hasAudio && visited.train && (
        <div key="page-train" style={{ display: isActive('/train') ? 'contents' : 'none' }}>
          <TrainingPage />
        </div>
      )}
      {hasSynth && visited.synth && (
        <div key="page-synth" style={{ display: isActive('/synth') ? 'contents' : 'none' }}>
          <SynthPage />
        </div>
      )}
      {hasSynth && !hasAudio && isActive('/') && !visited.synth && (
        <div key="page-synth-fallback" style={{ display: 'contents' }}>
          <SynthPage />
        </div>
      )}

      <CloseDialog
        key="close-dialog"
        open={closeDialog.open}
        fromTray={closeDialog.fromTray}
        onDone={() => setCloseDialog({ open: false, fromTray: false })}
      />
    </>
  )
}

export default function App() {
  return (
    <BrowserRouter>
      <AppContent />
    </BrowserRouter>
  )
}
