import { useEffect } from 'react'
import { BrowserRouter, Routes, Route, useNavigate } from 'react-router-dom'
import WorkflowPage from './pages/WorkflowPage'
import TrainingPage from './pages/TrainingPage'
import { importWorkflowFromPath } from './api/client'
import { useWorkflowStore } from './store/workflowStore'
import { useTrainingStore } from './store/trainingStore'

function AppContent() {
  const navigate = useNavigate()

  useEffect(() => {
    const api = window.electronAPI
    if (!api?.onOpenUtaiFile) return

    api.onOpenUtaiFile(async (filePath: string) => {
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
  }, [navigate])

  return (
    <Routes>
      <Route path="/" element={<WorkflowPage />} />
      <Route path="/train" element={<TrainingPage />} />
    </Routes>
  )
}

export default function App() {
  return (
    <BrowserRouter>
      <AppContent />
    </BrowserRouter>
  )
}
