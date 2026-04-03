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

interface WorkflowState {
  nodes: Node[]
  edges: Edge[]
  workflowName: string

  // Actions
  onNodesChange: (changes: NodeChange[]) => void
  onEdgesChange: (changes: EdgeChange[]) => void
  onConnect: (connection: Connection) => void
  addNode: (node: Node) => void
  removeNode: (id: string) => void
  removeEdge: (id: string) => void
  removeEdgesFromSource: (edgeIds: string[]) => void
  updateNodeData: (id: string, data: Record<string, unknown>) => void
  setWorkflow: (name: string, nodes: Node[], edges: Edge[]) => void
  clearWorkflow: () => void
}

const DEFAULT_NODES: Node[] = [
  {
    id: 'node_default_input',
    type: 'audio_input',
    position: { x: 100, y: 200 },
    data: { params: {} },
  },
]

export const useWorkflowStore = create<WorkflowState>((set, get) => ({
  nodes: DEFAULT_NODES,
  edges: [],
  workflowName: 'Untitled',

  onNodesChange: (changes) =>
    set({ nodes: applyNodeChanges(changes, get().nodes) }),

  onEdgesChange: (changes) =>
    set({ edges: applyEdgeChanges(changes, get().edges) }),

  onConnect: (connection) => {
    // Enforce single-connection per input port: remove any existing edge
    // targeting the same (target, targetHandle) before adding the new one.
    const filtered = get().edges.filter(
      (e) => !(e.target === connection.target && e.targetHandle === connection.targetHandle),
    )
    set({ edges: addEdge({ ...connection, type: 'smoothstep', animated: true }, filtered) })
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

  removeEdgesFromSource: (edgeIds) =>
    set({ edges: get().edges.filter((e) => !edgeIds.includes(e.id)) }),

  updateNodeData: (id, data) =>
    set({
      nodes: get().nodes.map((n) =>
        n.id === id ? { ...n, data: { ...n.data, ...data } } : n,
      ),
    }),

  setWorkflow: (name, nodes, edges) =>
    set({ workflowName: name, nodes, edges }),

  clearWorkflow: () => set({
    nodes: [{ ...DEFAULT_NODES[0], data: { params: {} } }],
    edges: [],
    workflowName: 'Untitled',
  }),
}))
