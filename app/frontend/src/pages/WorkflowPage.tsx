import { useCallback, useState, type DragEvent } from 'react'
import { useTranslation } from 'react-i18next'
import {
  ReactFlow,
  Background,
  Controls,
  useReactFlow,
  type NodeTypes,
  type Connection,
  type Edge,
  ReactFlowProvider,
} from '@xyflow/react'
import '@xyflow/react/dist/style.css'

import { useWorkflowStore } from '../store/workflowStore'
import { useExecutionStore } from '../store/executionStore'
import { Toolbar } from '../components/Toolbar'
import { NodePalette } from '../components/NodePalette'
import { ContextMenu } from '../components/ContextMenu'
import { AudioInputNode } from '../nodes/AudioInputNode'
import { AudioOutputNode } from '../nodes/AudioOutputNode'
import { MSSTNode } from '../nodes/MSSTNode'
import { VRNode } from '../nodes/VRNode'
import { RVCNode } from '../nodes/RVCNode'
import { SeparationNode } from '../nodes/SeparationNode'
import { YtdlpNode } from '../nodes/YtdlpNode'
import { VideoInputNode } from '../nodes/VideoInputNode'
import { VideoOutputNode } from '../nodes/VideoOutputNode'
import { MediaOutputNode } from '../nodes/MediaOutputNode'
import { AudioMergeNode } from '../nodes/AudioMergeNode'
import { ModelManager } from '../components/ModelManager'
import { WorkflowLogPanel } from '../components/WorkflowLogPanel'
import { startExecution, startPartialExecution, connectExecution, removeBuffers } from '../api/client'
import { getNextNodeId } from '../components/NodePalette'

const nodeTypes: NodeTypes = {
  audio_input: AudioInputNode,
  audio_output: AudioOutputNode,   // backward compat for old projects
  msst_separation: MSSTNode,       // backward compat
  vr_separation: VRNode,           // backward compat
  separation: SeparationNode,      // new unified node
  rvc_conversion: RVCNode,
  ytdlp_download: YtdlpNode,
  video_input: VideoInputNode,
  video_output: VideoOutputNode,   // backward compat for old projects
  media_output: MediaOutputNode,
  audio_merge: AudioMergeNode,
}

interface ContextMenuState {
  x: number
  y: number
  nodeId: string | null
  edgeId: string | null
  edgeSourceId?: string | null
  edgeTargetId?: string | null
}

let addNodeCounter = 0

function Flow() {
  useTranslation() // ensure re-render on lang change
  const { nodes, edges, onNodesChange, onEdgesChange, onConnect, addNode, removeNode, removeEdge } = useWorkflowStore()
  const { isRunning, setRunning, setExecutionId, updateNodeState, setAllPending, addLog, setLogPanelOpen, reset } = useExecutionStore()
  const [contextMenu, setContextMenu] = useState<ContextMenuState | null>(null)
  const [modelManagerOpen, setModelManagerOpen] = useState(false)
  const { screenToFlowPosition } = useReactFlow()

  const getTopZIndex = useCallback(() => {
    const currentNodes = useWorkflowStore.getState().nodes
    let max = 0
    for (const n of currentNodes) {
      const z = (n.style?.zIndex as number) || (n.zIndex as number) || 0
      if (z > max) max = z
    }
    return max + 1
  }, [])

  const handleAddNode = useCallback((type: string, screenX: number, screenY: number, category?: string) => {
    addNodeCounter++
    const position = screenToFlowPosition({ x: screenX, y: screenY })
    addNode({
      id: `node_${Date.now()}_${addNodeCounter}`,
      type,
      position,
      data: { params: category ? { category } : {} },
      zIndex: getTopZIndex() + 1000,
    })
  }, [screenToFlowPosition, addNode, getTopZIndex])

  const handleDrop = useCallback((e: DragEvent) => {
    e.preventDefault()
    const raw = e.dataTransfer.getData('application/utai-node-type')
    if (!raw) return
    const position = screenToFlowPosition({ x: e.clientX, y: e.clientY })

    // Parse: could be plain type string or JSON with { type, category }
    let type = raw
    let params: Record<string, any> = {}
    try {
      const parsed = JSON.parse(raw)
      if (parsed.type) {
        type = parsed.type
        if (parsed.category) params.category = parsed.category
      }
    } catch {
      // plain string, use as-is
    }

    addNode({
      id: getNextNodeId(),
      type,
      position,
      data: { params },
      zIndex: getTopZIndex() + 1000,
    })
  }, [screenToFlowPosition, addNode, getTopZIndex])

  const handleDragOver = useCallback((e: DragEvent) => {
    e.preventDefault()
    e.dataTransfer.dropEffect = 'move'
  }, [])

  /** Clean up audio buffers and execution state for deleted nodes. */
  const cleanupNodeBuffers = useCallback((nodeIds: string[]) => {
    const states = useExecutionStore.getState().nodeStates
    const bufferIds: string[] = []
    for (const nid of nodeIds) {
      const outputs = states[nid]?.outputs
      if (outputs) {
        bufferIds.push(...Object.values(outputs))
      }
    }
    if (bufferIds.length > 0) {
      removeBuffers(bufferIds).catch((err) => {
        console.warn('Failed to cleanup buffers:', err)
      })
    }
  }, [])

  const handleDeleteNode = useCallback((id: string) => {
    cleanupNodeBuffers([id])
    removeNode(id)
  }, [removeNode, cleanupNodeBuffers])

  /** Check if a node is currently busy (running or pending). */
  const isNodeBusy = useCallback((nodeId: string) => {
    const st = useExecutionStore.getState().nodeStates[nodeId]
    return !!st && (st.status === 'running' || st.status === 'pending')
  }, [])

  const handleNodesChange = useCallback((changes: import('@xyflow/react').NodeChange[]) => {
    // Block deletion of busy nodes
    const filtered = changes.filter((c) => {
      if (c.type === 'remove' && isNodeBusy(c.id)) return false
      return true
    })
    const removedIds = filtered
      .filter((c): c is import('@xyflow/react').NodeRemoveChange => c.type === 'remove')
      .map((c) => c.id)
    if (removedIds.length > 0) {
      cleanupNodeBuffers(removedIds)
    }
    onNodesChange(filtered)
  }, [onNodesChange, cleanupNodeBuffers, isNodeBusy])

  /** Block edge changes (remove/reconnect) if either endpoint is busy. */
  const handleEdgesChange = useCallback((changes: import('@xyflow/react').EdgeChange[]) => {
    const currentEdges = useWorkflowStore.getState().edges
    const filtered = changes.filter((c) => {
      if (c.type === 'remove') {
        const edge = currentEdges.find((e) => e.id === c.id)
        if (edge && (isNodeBusy(edge.source) || isNodeBusy(edge.target))) return false
      }
      return true
    })
    onEdgesChange(filtered)
  }, [onEdgesChange, isNodeBusy])

  /** Block new connections TO busy nodes. */
  const handleConnect = useCallback((connection: import('@xyflow/react').Connection) => {
    if (connection.target && isNodeBusy(connection.target)) return
    onConnect(connection)
  }, [onConnect, isNodeBusy])

  const handleExecute = useCallback(async () => {
    if (isRunning) return
    reset()
    setRunning(true)
    setAllPending(nodes.map((n) => n.id))

    const workflowNodes = nodes.map((n) => ({
      id: n.id,
      type: n.type!,
      position: n.position,
      params: n.data?.params || {},
    }))

    const connections = edges.map((e) => ({
      from_node: e.source,
      from_port: e.sourceHandle || 'audio',
      to_node: e.target,
      to_port: e.targetHandle || 'audio',
    }))

    setLogPanelOpen(true)

    try {
      const { execution_id } = await startExecution({
        version: '1.0',
        name: 'workflow',
        nodes: workflowNodes,
        connections,
      })

      setExecutionId(execution_id)

      const ws = connectExecution(execution_id, (msg) => {
        if (msg.type === 'node_status') {
          updateNodeState(msg)
        } else if (msg.type === 'log') {
          addLog(msg.message)
        } else if (msg.type === 'execution_complete') {
          setRunning(false)
          ws.close()
        }
      })
      ws.onclose = () => {
        // Ensure UI resets if connection drops unexpectedly
        if (useExecutionStore.getState().isRunning) {
          setRunning(false)
        }
      }
    } catch (e) {
      console.error('Execution failed:', e)
      setRunning(false)
    }
  }, [nodes, edges, isRunning])

  const handleRerunFromNode = useCallback(async (fromNodeId: string) => {
    if (isRunning) return
    setLogPanelOpen(true)

    // 1. Snapshot cached outputs from completed nodes BEFORE resetting
    const { nodeStates: prevStates } = useExecutionStore.getState()

    // 2. Build forward and reverse adjacency
    const fwdAdj: Record<string, string[]> = {}
    const revAdj: Record<string, string[]> = {}
    for (const e of edges) {
      if (!fwdAdj[e.source]) fwdAdj[e.source] = []
      fwdAdj[e.source].push(e.target)
      if (!revAdj[e.target]) revAdj[e.target] = []
      revAdj[e.target].push(e.source)
    }

    // 3. Collect downstream (fromNodeId + all children)
    const downstream = new Set<string>()
    const dQueue = [fromNodeId]
    while (dQueue.length > 0) {
      const nid = dQueue.shift()!
      if (downstream.has(nid)) continue
      downstream.add(nid)
      for (const child of (fwdAdj[nid] || [])) dQueue.push(child)
    }

    // 4. Walk upstream from fromNodeId to find contiguous unexecuted predecessors
    //    Stop at nodes that were already completed (don't re-run them)
    const unexecutedUpstream = new Set<string>()
    const uQueue = [...(revAdj[fromNodeId] || [])]
    while (uQueue.length > 0) {
      const nid = uQueue.shift()!
      if (unexecutedUpstream.has(nid) || downstream.has(nid)) continue
      const st = prevStates[nid]
      if (st?.status === 'completed' && st.outputs) continue // already executed — stop here
      unexecutedUpstream.add(nid)
      for (const parent of (revAdj[nid] || [])) uQueue.push(parent)
    }

    // 5. Nodes to re-execute = unexecutedUpstream + downstream
    const toExecute = new Set([...unexecutedUpstream, ...downstream])

    // 6. Build cached outputs for completed nodes NOT in toExecute
    const cachedOutputs: Record<string, Record<string, string>> = {}
    for (const [nodeId, state] of Object.entries(prevStates)) {
      if (state.status === 'completed' && state.outputs && !toExecute.has(nodeId)) {
        cachedOutputs[nodeId] = state.outputs
      }
    }

    // 7. Set states: toExecute = pending, rest = keep
    const newStates: Record<string, { status: string; progress: number; message: string; outputs?: Record<string, string> }> = {}
    for (const [nodeId, state] of Object.entries(prevStates)) {
      if (toExecute.has(nodeId)) {
        newStates[nodeId] = { status: 'pending', progress: 0, message: '' }
      } else {
        newStates[nodeId] = { ...state }
      }
    }
    // Also add nodes in toExecute that weren't in prevStates (first-time execution)
    for (const nodeId of toExecute) {
      if (!newStates[nodeId]) {
        newStates[nodeId] = { status: 'pending', progress: 0, message: '' }
      }
    }

    useExecutionStore.setState({
      executionId: null,
      isRunning: true,
      nodeStates: newStates as any,
    })

    const workflowNodes = nodes.map((n) => ({
      id: n.id,
      type: n.type!,
      position: n.position,
      params: n.data?.params || {},
    }))

    const connections = edges.map((e) => ({
      from_node: e.source,
      from_port: e.sourceHandle || 'audio',
      to_node: e.target,
      to_port: e.targetHandle || 'audio',
    }))

    // Find the topologically earliest node that needs execution.
    // If there are unexecuted upstream nodes, start from the earliest one
    // so the backend's downstream traversal includes everything.
    let actualStartNode = fromNodeId
    if (unexecutedUpstream.size > 0) {
      // Find nodes in unexecutedUpstream that have no parents in unexecutedUpstream
      const roots = [...unexecutedUpstream].filter((nid) =>
        !(revAdj[nid] || []).some((p) => unexecutedUpstream.has(p))
      )
      if (roots.length > 0) actualStartNode = roots[0]
    }

    try {
      const { execution_id } = await startPartialExecution(
        { version: '1.0', name: 'workflow', nodes: workflowNodes, connections },
        actualStartNode,
        cachedOutputs,
      )

      setExecutionId(execution_id)

      const ws = connectExecution(execution_id, (msg) => {
        if (msg.type === 'node_status') {
          updateNodeState(msg)
        } else if (msg.type === 'log') {
          addLog(msg.message)
        } else if (msg.type === 'execution_complete') {
          setRunning(false)
          ws.close()
        }
      })
      ws.onclose = () => {
        if (useExecutionStore.getState().isRunning) {
          setRunning(false)
        }
      }
    } catch (e) {
      console.error('Partial execution failed:', e)
      setRunning(false)
    }
  }, [nodes, edges, isRunning])

  const handleContextMenu = useCallback((event: React.MouseEvent) => {
    event.preventDefault()
    const target = event.target as HTMLElement

    const nodeEl = target.closest('.react-flow__node')
    const nodeId = nodeEl?.getAttribute('data-id') || null

    const edgeEl = target.closest('.react-flow__edge')
    const edgeId = edgeEl?.getAttribute('data-id') || null
    const edge = edgeId ? edges.find((e) => e.id === edgeId) : null

    setContextMenu({
      x: event.clientX,
      y: event.clientY,
      nodeId,
      edgeId: nodeId ? null : edgeId,
      edgeSourceId: edge?.source || null,
      edgeTargetId: edge?.target || null,
    })
  }, [edges])

  /** Port type validation + cycle detection. */
  const isValidConnection = useCallback((connection: Edge | Connection) => {
    const srcHandle = connection.sourceHandle || 'audio'
    const tgtHandle = connection.targetHandle || 'audio'
    const srcIsVideo = srcHandle === 'video_path'
    const tgtIsVideo = tgtHandle === 'video_path'
    // Both must be same type
    if (srcIsVideo !== tgtIsVideo) return false
    // No self-loops
    if (connection.source === connection.target) return false
    // Cycle detection: would adding target→...→source create a cycle?
    // i.e. can we reach source from target following existing edges?
    const currentEdges = useWorkflowStore.getState().edges
    const visited = new Set<string>()
    const queue = [connection.target!]
    while (queue.length > 0) {
      const nid = queue.shift()!
      if (nid === connection.source) return false // cycle!
      if (visited.has(nid)) continue
      visited.add(nid)
      for (const e of currentEdges) {
        if (e.source === nid) queue.push(e.target)
      }
    }
    return true
  }, [])

  /** Run only a single node (no downstream). */
  const handleRerunSingleNode = useCallback(async (nodeId: string) => {
    if (isRunning) return
    setLogPanelOpen(true)
    // Same logic as handleRerunFromNode but we filter downstream to only the target node
    const { nodeStates: prevStates } = useExecutionStore.getState()
    const cachedOutputs: Record<string, Record<string, string>> = {}
    for (const [nid, state] of Object.entries(prevStates)) {
      if (state.status === 'completed' && state.outputs && nid !== nodeId) {
        cachedOutputs[nid] = state.outputs
      }
    }

    // Only mark this single node as pending
    const newStates: Record<string, any> = {}
    for (const [nid, state] of Object.entries(prevStates)) {
      if (nid === nodeId) {
        newStates[nid] = { status: 'pending', progress: 0, message: '' }
      } else {
        newStates[nid] = { ...state }
      }
    }
    // Ensure node appears even if never executed before
    if (!newStates[nodeId]) {
      newStates[nodeId] = { status: 'pending', progress: 0, message: '' }
    }
    useExecutionStore.setState({ executionId: null, isRunning: true, nodeStates: newStates as any })

    const workflowNodes = nodes.map((n) => ({
      id: n.id, type: n.type!, position: n.position, params: n.data?.params || {},
    }))
    // Remove outgoing edges FROM this node so backend doesn't execute downstream
    const connections = edges
      .filter((e) => e.source !== nodeId)
      .map((e) => ({
        from_node: e.source, from_port: e.sourceHandle || 'audio',
        to_node: e.target, to_port: e.targetHandle || 'audio',
      }))

    try {
      const { execution_id } = await startPartialExecution(
        { version: '1.0', name: 'workflow', nodes: workflowNodes, connections },
        nodeId, cachedOutputs,
      )
      setExecutionId(execution_id)
      const ws = connectExecution(execution_id, (msg) => {
        if (msg.type === 'node_status') updateNodeState(msg)
        else if (msg.type === 'log') addLog(msg.message)
        else if (msg.type === 'execution_complete') { setRunning(false); ws.close() }
      })
      ws.onclose = () => { if (useExecutionStore.getState().isRunning) setRunning(false) }
    } catch (e) {
      console.error('Single-node execution failed:', e)
      setRunning(false)
    }
  }, [nodes, edges, isRunning])

  const handlePaneClick = useCallback(() => {
    setContextMenu(null)
  }, [])

  return (
    <div style={{ width: '100%', height: '100%', display: 'flex', flexDirection: 'column' }}>
      <Toolbar
        onExecute={handleExecute}
        onOpenModelManager={() => setModelManagerOpen(true)}
      />
      <ModelManager open={modelManagerOpen} onClose={() => setModelManagerOpen(false)} />
      <div style={{ flex: 1, display: 'flex', overflow: 'hidden', minHeight: 0 }}>
        <NodePalette />
        <div className="utai-flow-area" onContextMenu={handleContextMenu}>
          <ReactFlow
            nodes={nodes}
            edges={edges}
            onNodesChange={handleNodesChange}
            onEdgesChange={handleEdgesChange}
            onConnect={handleConnect}
            isValidConnection={isValidConnection}
            onPaneClick={handlePaneClick}
            onDrop={handleDrop}
            onDragOver={handleDragOver}
            nodeTypes={nodeTypes}
            fitView
            snapToGrid
            snapGrid={[16, 16]}
            deleteKeyCode={['Backspace', 'Delete']}
            edgesFocusable
            edgesReconnectable
            nodesConnectable
            defaultEdgeOptions={{
              selectable: true,
              style: { strokeWidth: 1.5, stroke: '#3a3a5a' },
              focusable: true,
            }}
            style={{ background: 'var(--bg-deep)' }}
          >
            <Background color="#1a1a35" gap={20} size={1} />
            <Controls />
          </ReactFlow>

          {contextMenu && (
            <ContextMenu
              x={contextMenu.x}
              y={contextMenu.y}
              nodeId={contextMenu.nodeId}
              edgeId={contextMenu.edgeId}
              edgeSourceId={contextMenu.edgeSourceId}
              edgeTargetId={contextMenu.edgeTargetId}
              isRunning={isRunning}
              onClose={() => setContextMenu(null)}
              onDeleteNode={handleDeleteNode}
              onDeleteEdge={(id) => removeEdge(id)}
              onRerunFromNode={handleRerunFromNode}
              onRerunSingleNode={handleRerunSingleNode}
              onAddNode={handleAddNode}
            />
          )}

          {/* Retro visitor counter easter egg */}
          <div className="utai-counter">
            ★ utai v0.1.0
          </div>
        </div>
        <WorkflowLogPanel />
      </div>
    </div>
  )
}

export default function WorkflowPage() {
  return (
    <ReactFlowProvider>
      <Flow />
    </ReactFlowProvider>
  )
}
