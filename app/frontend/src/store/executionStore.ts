import { create } from 'zustand'
import type { NodeStatusUpdate } from '../types'

interface NodeExecState {
  status: 'idle' | 'pending' | 'running' | 'completed' | 'error'
  progress: number
  message: string
  outputs?: Record<string, string>
}

interface ExecutionState {
  executionId: string | null
  isRunning: boolean
  nodeStates: Record<string, NodeExecState>

  // Execution logs
  logs: string[]
  logPanelOpen: boolean

  setExecutionId: (id: string | null) => void
  setRunning: (running: boolean) => void
  updateNodeState: (update: NodeStatusUpdate) => void
  setAllPending: (nodeIds: string[]) => void
  addLog: (log: string) => void
  clearLogs: () => void
  setLogPanelOpen: (v: boolean) => void
  reset: () => void
}

export const useExecutionStore = create<ExecutionState>((set, get) => ({
  executionId: null,
  isRunning: false,
  nodeStates: {},
  logs: [],
  logPanelOpen: false,

  setExecutionId: (id) => set({ executionId: id }),

  setRunning: (running) => set({ isRunning: running }),

  updateNodeState: (update) =>
    set({
      nodeStates: {
        ...get().nodeStates,
        [update.node_id]: {
          status: update.status,
          progress: update.progress,
          message: update.message,
          outputs: update.outputs ?? get().nodeStates[update.node_id]?.outputs,
        },
      },
    }),

  setAllPending: (nodeIds) => {
    const states: Record<string, NodeExecState> = {}
    for (const id of nodeIds) {
      states[id] = { status: 'pending', progress: 0, message: '' }
    }
    set({ nodeStates: states })
  },

  addLog: (log) => set({ logs: [...get().logs, log] }),
  clearLogs: () => set({ logs: [] }),
  setLogPanelOpen: (v) => set({ logPanelOpen: v }),

  reset: () => set({ executionId: null, isRunning: false, nodeStates: {} }),
}))
