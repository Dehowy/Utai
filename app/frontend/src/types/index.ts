// Port and Node definition types (mirroring backend schemas)

export interface PortDefinition {
  name: string
  port_type: 'audio' | 'string'
  label: string
  optional?: boolean
}

export interface ParameterOption {
  value: string
  label: string
}

export interface ParameterDefinition {
  name: string
  label: string
  type: 'select' | 'slider' | 'checkbox' | 'text' | 'int' | 'float'
  default: unknown
  options?: ParameterOption[]
  min?: number
  max?: number
  step?: number
}

export interface NodeDefinition {
  node_type: string
  label: string
  category: string
  description: string
  inputs: PortDefinition[]
  outputs: PortDefinition[]
  parameters: ParameterDefinition[]
  dynamic_outputs: boolean
}

// Workflow types
export interface WorkflowNode {
  id: string
  type: string
  position: { x: number; y: number }
  params: Record<string, unknown>
}

export interface WorkflowConnection {
  from_node: string
  from_port: string
  to_node: string
  to_port: string
}

export interface Workflow {
  version: string
  name: string
  nodes: WorkflowNode[]
  connections: WorkflowConnection[]
}

// Execution types
export interface NodeStatusUpdate {
  type: 'node_status'
  node_id: string
  status: 'pending' | 'running' | 'completed' | 'error'
  progress: number
  message: string
  outputs?: Record<string, string> // port_name -> buffer_id
}

export interface ExecutionComplete {
  type: 'execution_complete'
  execution_id: string
  success: boolean
  error?: string
}

export type WSMessage = NodeStatusUpdate | ExecutionComplete

// Audio upload response
export interface AudioUploadResponse {
  buffer_id: string
  filename: string
  sample_rate: number
  duration: number
  channels: number
}
