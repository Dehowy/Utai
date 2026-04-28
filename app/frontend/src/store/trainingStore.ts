import { create } from 'zustand'
import {
  type Node,
  type Edge,
  type Connection,
  type NodeChange,
  type EdgeChange,
  applyNodeChanges,
  applyEdgeChanges,
  addEdge,
} from '@xyflow/react'

export type TrainNodeStatus = 'idle' | 'running' | 'completed' | 'error'

export interface TrainNodeState {
  status: TrainNodeStatus
  progress: number
  message: string
  /** File paths or other output data */
  outputData?: any
}

interface TrainingState {
  // Graph
  nodes: Node[]
  edges: Edge[]
  workflowName: string
  dirty: boolean
  projectPath: string | null
  isSaving: boolean
  markClean: () => void
  setProjectPath: (path: string | null) => void
  setIsSaving: (saving: boolean) => void

  // Graph actions
  onNodesChange: (changes: NodeChange[]) => void
  onEdgesChange: (changes: EdgeChange[]) => void
  onConnect: (connection: Connection) => void
  addNode: (node: Node) => void
  removeNode: (id: string) => void
  removeEdge: (id: string) => void
  updateNodeData: (id: string, data: Record<string, unknown>) => void
  setWorkflow: (name: string, nodes: Node[], edges: Edge[]) => void

  // Execution state per node
  nodeStates: Record<string, TrainNodeState>
  setNodeState: (nodeId: string, state: Partial<TrainNodeState>) => void
  resetNodeState: (nodeId: string) => void
  resetAllNodeStates: () => void

  // Global execution
  isRunningAll: boolean
  setRunningAll: (v: boolean) => void

  // Training logs (right panel)
  logs: string[]
  addLog: (log: string) => void
  clearLogs: () => void
  logPanelOpen: boolean
  setLogPanelOpen: (v: boolean) => void

  // Active task tracking
  activeTaskId: string | null
  setActiveTaskId: (id: string | null) => void

  // Training report modal (shared between TrainRVCNode and TrainingPage)
  reportOpen: boolean
  reportData: { expName: string; success: boolean; startTime: number }
  openReport: (data: { expName: string; success: boolean; startTime: number }) => void
  closeReport: () => void

  // Reset everything
  clearTraining: () => void
}

const DEFAULT_TRAIN_NODES: Node[] = [
  {
    id: 'train_default_import',
    type: 'train_audio_import',
    position: { x: 200, y: 200 },
    data: { params: {} },
  },
]

export const useTrainingStore = create<TrainingState>((set, get) => ({
  nodes: [...DEFAULT_TRAIN_NODES],
  edges: [],
  workflowName: 'Untitled',
  dirty: false,
  projectPath: null,
  isSaving: false,
  markClean: () => set({ dirty: false }),
  setProjectPath: (path) => set({ projectPath: path }),
  setIsSaving: (saving) => set({ isSaving: saving }),

  onNodesChange: (changes) =>
    set({ nodes: applyNodeChanges(changes, get().nodes), dirty: true }),

  onEdgesChange: (changes) =>
    set({ edges: applyEdgeChanges(changes, get().edges), dirty: true }),

  onConnect: (connection) => {
    set({ edges: addEdge({ ...connection, type: 'smoothstep', animated: true }, get().edges), dirty: true })
  },

  addNode: (node) =>
    set({ nodes: [...get().nodes, node], dirty: true }),

  removeNode: (id) =>
    set({
      nodes: get().nodes.filter((n) => n.id !== id),
      edges: get().edges.filter((e) => e.source !== id && e.target !== id),
      dirty: true,
    }),

  removeEdge: (id) =>
    set({ edges: get().edges.filter((e) => e.id !== id), dirty: true }),

  updateNodeData: (id, data) =>
    set({
      nodes: get().nodes.map((n) =>
        n.id === id ? { ...n, data: { ...n.data, ...data } } : n,
      ),
      dirty: true,
    }),

  setWorkflow: (name, nodes, edges) =>
    set({ workflowName: name, nodes, edges, nodeStates: {}, dirty: false, projectPath: null }),

  // Node execution states
  nodeStates: {},
  setNodeState: (nodeId, state) =>
    set({
      nodeStates: {
        ...get().nodeStates,
        [nodeId]: { ...( get().nodeStates[nodeId] || { status: 'idle', progress: 0, message: '' }), ...state },
      },
    }),
  resetNodeState: (nodeId) =>
    set({
      nodeStates: {
        ...get().nodeStates,
        [nodeId]: { status: 'idle', progress: 0, message: '' },
      },
    }),
  resetAllNodeStates: () => set({ nodeStates: {} }),

  isRunningAll: false,
  setRunningAll: (v) => set({ isRunningAll: v }),

  logs: [],
  addLog: (log) => set({ logs: [...get().logs, log] }),
  clearLogs: () => set({ logs: [] }),
  logPanelOpen: false,
  setLogPanelOpen: (v) => set({ logPanelOpen: v }),

  activeTaskId: null,
  setActiveTaskId: (id) => set({ activeTaskId: id }),

  reportOpen: false,
  reportData: { expName: '', success: false, startTime: 0 },
  openReport: (data) => set({ reportData: data, reportOpen: true }),
  closeReport: () => set({ reportOpen: false }),

  clearTraining: () => set({
    nodes: [...DEFAULT_TRAIN_NODES],
    edges: [],
    workflowName: 'Untitled',
    dirty: false,
    projectPath: null,
    nodeStates: {},
    isRunningAll: false,
    logs: [],
    activeTaskId: null,
  }),
}))
