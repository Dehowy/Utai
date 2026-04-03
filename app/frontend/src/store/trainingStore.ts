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

  onNodesChange: (changes) =>
    set({ nodes: applyNodeChanges(changes, get().nodes) }),

  onEdgesChange: (changes) =>
    set({ edges: applyEdgeChanges(changes, get().edges) }),

  onConnect: (connection) => {
    // Training page: allow multiple connections per input port
    // (e.g. multiple audio sources → MSST batch node)
    set({ edges: addEdge({ ...connection, type: 'smoothstep', animated: true }, get().edges) })
  },

  addNode: (node) =>
    set({ nodes: [...get().nodes, node] }),

  removeNode: (id) =>
    set({
      nodes: get().nodes.filter((n) => n.id !== id),
      edges: get().edges.filter((e) => e.source !== id && e.target !== id),
    }),

  removeEdge: (id) =>
    set({ edges: get().edges.filter((e) => e.id !== id) }),

  updateNodeData: (id, data) =>
    set({
      nodes: get().nodes.map((n) =>
        n.id === id ? { ...n, data: { ...n.data, ...data } } : n,
      ),
    }),

  setWorkflow: (name, nodes, edges) =>
    set({ workflowName: name, nodes, edges, nodeStates: {} }),

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

  clearTraining: () => set({
    nodes: [...DEFAULT_TRAIN_NODES],
    edges: [],
    workflowName: 'Untitled',
    nodeStates: {},
    isRunningAll: false,
    logs: [],
    activeTaskId: null,
  }),
}))
