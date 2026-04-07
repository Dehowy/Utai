import { useEffect, useState } from 'react'
import { BrowserRouter, useNavigate, useLocation } from 'react-router-dom'
import WorkflowPage from './pages/WorkflowPage'
import TrainingPage from './pages/TrainingPage'
import SynthPage from './pages/SynthPage'
import { importWorkflowFromPath } from './api/client'
import { useWorkflowStore } from './store/workflowStore'
import { useTrainingStore } from './store/trainingStore'
import { CloseDialog } from './components/CloseDialog'

function AppContent() {
  const navigate = useNavigate()
  const location = useLocation()

  // Keep-alive: track which pages have been visited (lazy-mount, never unmount)
  const [visited, setVisited] = useState({ workflow: true, train: false, synth: false })

  useEffect(() => {
    if (location.pathname === '/train' && !visited.train) {
      setVisited(v => ({ ...v, train: true }))
    }
    if (location.pathname === '/synth' && !visited.synth) {
      setVisited(v => ({ ...v, synth: true }))
    }
  }, [location.pathname])

  // Close dialog state
  const [closeDialog, setCloseDialog] = useState<{ open: boolean; fromTray: boolean }>({
    open: false, fromTray: false,
  })

  // Block all default browser context menus globally
  useEffect(() => {
    const handler = (e: MouseEvent) => {
      // Allow context menu on text inputs/textareas for copy/paste
      const t = e.target as HTMLElement;
      if (t.tagName === 'INPUT' || t.tagName === 'TEXTAREA' || t.isContentEditable) return;
      e.preventDefault();
    };
    document.addEventListener('contextmenu', handler);
    return () => document.removeEventListener('contextmenu', handler);
  }, []);

  // Block browser default behaviors for modifier keys and shortcuts
  useEffect(() => {
    const onKeyDown = (e: KeyboardEvent) => {
      const tag = (e.target as HTMLElement)?.tagName;
      const isInput = tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT';

      // Block F-keys
      if (e.key === 'F5') { e.preventDefault(); return; }
      if (e.key === 'F12') { e.preventDefault(); return; }

      // Block Ctrl combos (except cut/copy/paste/select-all in inputs)
      if (e.ctrlKey || e.metaKey) {
        const k = e.key.toLowerCase();
        // Allow Ctrl+A/C/V/X in text inputs
        if (isInput && ['a', 'c', 'v', 'x'].includes(k)) return;
        // Block everything else: Ctrl+R, Ctrl+S, Ctrl+P, Ctrl+G, Ctrl+U, etc.
        e.preventDefault();
        return;
      }

      // Block Alt key alone (browser menu bar activation) and Alt combos
      if (e.altKey) {
        e.preventDefault();
        return;
      }

      // Block space on buttons/links to prevent native click + scroll
      // (synth page has its own space handler for play/pause)
      if (e.key === ' ' && !isInput) {
        e.preventDefault();
        return;
      }
    };

    // Block Alt keyup too (some browsers activate menu bar on Alt release)
    const onKeyUp = (e: KeyboardEvent) => {
      if (e.key === 'Alt') e.preventDefault();
    };

    document.addEventListener('keydown', onKeyDown);
    document.addEventListener('keyup', onKeyUp);
    return () => {
      document.removeEventListener('keydown', onKeyDown);
      document.removeEventListener('keyup', onKeyUp);
    };
  }, []);

  // Handle .utai file opening
  useEffect(() => {
    const api = window.electronAPI
    if (!api?.onOpenUtaiFile) return

    const cleanup = api.onOpenUtaiFile(async (filePath: string) => {
      console.log('[utai] Opening .utai file:', filePath)
      try {
        const data = await importWorkflowFromPath(filePath)
        if (!data || data.status === 'cancelled') return

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
          useTrainingStore.getState().setWorkflow(data.name || 'Imported', rfNodes, rfEdges)
          navigate('/train')
        } else {
          useWorkflowStore.getState().setWorkflow(data.name || 'Imported', rfNodes, rfEdges)
          navigate('/')
        }
      } catch (err) {
        console.error('[utai] Failed to open .utai file:', err)
      }
    })
    return cleanup || undefined
  }, [navigate])

  // Handle close-requested from Electron main process
  useEffect(() => {
    const api = window.electronAPI
    if (!api?.onCloseRequested) return

    const cleanup = api.onCloseRequested((data) => {
      setCloseDialog({ open: true, fromTray: data.fromTray })
    })
    return cleanup || undefined
  }, [])

  const isActive = (p: string) => location.pathname === p

  return (
    <>
      {/* Keep-alive page containers: mount on first visit, never unmount */}
      <div key="page-workflow" style={{ display: isActive('/') ? 'contents' : 'none' }}>
        <WorkflowPage />
      </div>
      {visited.train && (
        <div key="page-train" style={{ display: isActive('/train') ? 'contents' : 'none' }}>
          <TrainingPage />
        </div>
      )}
      {visited.synth && (
        <div key="page-synth" style={{ display: isActive('/synth') ? 'contents' : 'none' }}>
          <SynthPage />
        </div>
      )}

      {/* Close / minimize-to-tray dialog */}
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
