const API_BASE = import.meta.env.VITE_API_URL || ''

// Electron integration
declare global {
  interface Window {
    electronAPI?: {
      isElectron: boolean
      openFile: (options?: any) => Promise<string | null>
      saveFile: (options?: any) => Promise<string | null>
      openFolder: (options?: any) => Promise<string | null>
      getBackendPort: () => Promise<number>
      onOpenUtaiFile: (callback: (filePath: string) => void) => void
    }
  }
}

export const isElectron = () => !!(window as any).electronAPI?.isElectron

async function fetchJSON<T>(url: string, options?: RequestInit): Promise<T> {
  const res = await fetch(`${API_BASE}${url}`, {
    ...options,
    headers: { 'Content-Type': 'application/json', ...options?.headers },
  })
  if (!res.ok) {
    const text = await res.text()
    throw new Error(`API error ${res.status}: ${text}`)
  }
  return res.json()
}

// Node definitions
export const getNodeDefinitions = () => fetchJSON<any[]>('/api/nodes/definitions')

// Models
export const getMSSTModels = () => fetchJSON<any[]>('/api/models/msst')
export const getVRModels = () => fetchJSON<any[]>('/api/models/vr')
export const getRVCModels = () => fetchJSON<any[]>('/api/models/rvc')
export const getMSSTModelStems = (name: string) =>
  fetchJSON<{ model_name: string; stems: string[] }>(`/api/models/msst/stems?model_name=${encodeURIComponent(name)}`)

// Category-based model API
export const getModelCategories = () => fetchJSON<any[]>('/api/models/categories')
export const getModelsByCategory = (category: string) =>
  fetchJSON<any[]>(`/api/models/by-category/${encodeURIComponent(category)}`)
export const getAllModels = () => fetchJSON<any[]>('/api/models/all')
export const downloadModel = (modelName: string) =>
  fetchJSON<any>('/api/models/download', {
    method: 'POST',
    body: JSON.stringify({ model_name: modelName }),
  })
export const getDownloadStatus = (modelName: string) =>
  fetchJSON<any>(`/api/models/download-status/${encodeURIComponent(modelName)}`)
export const deleteModel = (modelName: string) =>
  fetchJSON<any>(`/api/models/${encodeURIComponent(modelName)}`, { method: 'DELETE' })

// RVC model management
export async function importRVCModel(pthFile: File, indexFile?: File) {
  const formData = new FormData()
  formData.append('pth_file', pthFile)
  if (indexFile) formData.append('index_file', indexFile)
  const res = await fetch(`${API_BASE}/api/models/rvc/import`, { method: 'POST', body: formData })
  if (!res.ok) { const t = await res.text(); throw new Error(`Import failed: ${res.status}: ${t}`) }
  return res.json()
}
export const archiveRVCModel = (name: string) =>
  fetchJSON<any>(`/api/models/rvc/archive/${encodeURIComponent(name)}`, { method: 'POST' })
export const getArchivedRVCModels = () =>
  fetchJSON<any[]>('/api/models/rvc/archived')
export const restoreRVCModel = (archivedName: string) =>
  fetchJSON<any>(`/api/models/rvc/restore/${encodeURIComponent(archivedName)}`, { method: 'POST' })
export const deleteArchivedRVCModel = (archivedName: string) =>
  fetchJSON<any>(`/api/models/rvc/archived/${encodeURIComponent(archivedName)}`, { method: 'DELETE' })

// DiffSinger models
export const getDiffSingerModels = () =>
  fetchJSON<any[]>('/api/models/diffsinger')
export const getDiffSingerCatalog = () =>
  fetchJSON<any>('/api/models/diffsinger/catalog')
export const downloadDiffSingerModel = (modelId: string) =>
  fetchJSON<any>('/api/models/diffsinger/download', {
    method: 'POST',
    body: JSON.stringify({ model_id: modelId }),
  })
export const getDiffSingerDownloadStatus = (modelId: string) =>
  fetchJSON<any>(`/api/models/diffsinger/download-status/${encodeURIComponent(modelId)}`)
export async function importDiffSingerModel(zipFile: File) {
  const formData = new FormData()
  formData.append('file', zipFile)
  const res = await fetch(`${API_BASE}/api/models/diffsinger/import`, { method: 'POST', body: formData })
  if (!res.ok) {
    const text = await res.text()
    throw new Error(`Import failed: ${res.status}: ${text}`)
  }
  return res.json()
}
export const archiveDiffSinger = (singerId: string) =>
  fetchJSON<any>(`/api/models/diffsinger/archive/${encodeURIComponent(singerId)}`, { method: 'POST' })
export const getArchivedDiffSinger = () =>
  fetchJSON<any[]>('/api/models/diffsinger/archived')
export const restoreDiffSinger = (archivedName: string) =>
  fetchJSON<any>(`/api/models/diffsinger/restore/${encodeURIComponent(archivedName)}`, { method: 'POST' })
export const deleteDiffSinger = (singerId: string) =>
  fetchJSON<any>(`/api/models/diffsinger/${encodeURIComponent(singerId)}`, { method: 'DELETE' })
export const deleteArchivedDiffSinger = (archivedName: string) =>
  fetchJSON<any>(`/api/models/diffsinger/archived/${encodeURIComponent(archivedName)}`, { method: 'DELETE' })

// Audio
export async function uploadAudio(file: File) {
  console.log('uploadAudio called:', file.name, file.size, file.type)
  const formData = new FormData()
  formData.append('file', file)
  const res = await fetch(`${API_BASE}/api/audio/upload`, { method: 'POST', body: formData })
  if (!res.ok) {
    const text = await res.text()
    console.error('Upload error response:', res.status, text)
    throw new Error(`Upload failed: ${res.status}: ${text}`)
  }
  return res.json()
}

export const getAudioURL = (bufferId: string) => `${API_BASE}/api/audio/${bufferId}`
export const getVideoURL = (filename: string) =>
  `${API_BASE}/api/audio/video/${encodeURIComponent(filename)}`
export const getDownloadURL = (bufferId: string, format = 'wav') =>
  `${API_BASE}/api/audio/${bufferId}/download?format=${format}`

// Execution
export async function startExecution(workflow: any) {
  return fetchJSON<{ execution_id: string }>('/api/execute', {
    method: 'POST',
    body: JSON.stringify({ workflow }),
  })
}

export const stopExecution = (executionId: string) =>
  fetchJSON<any>(`/api/execute/stop/${executionId}`, { method: 'POST' })

export async function startPartialExecution(
  workflow: any,
  fromNodeId: string,
  cachedOutputs: Record<string, Record<string, string>>,
) {
  return fetchJSON<{ execution_id: string }>('/api/execute/from-node', {
    method: 'POST',
    body: JSON.stringify({ workflow, from_node_id: fromNodeId, cached_outputs: cachedOutputs }),
  })
}

// WebSocket
export function connectExecution(executionId: string, onMessage: (msg: any) => void) {
  const wsBase = API_BASE.replace(/^http/, 'ws') || `ws://${window.location.host}`
  const ws = new WebSocket(`${wsBase}/ws/execution/${executionId}`)
  ws.onmessage = (e) => {
    try {
      onMessage(JSON.parse(e.data))
    } catch (err) {
      console.error('WS message parse error:', err)
    }
  }
  ws.onerror = (e) => console.error('WS error:', e)
  return ws
}

// Workflows (.utai)
export const exportWorkflow = (workflow: any) =>
  fetchJSON<{ status: string; path?: string }>('/api/workflows/export', {
    method: 'POST',
    body: JSON.stringify(workflow),
  })

export const importWorkflowDialog = () =>
  fetchJSON<any>('/api/workflows/import-dialog', { method: 'POST' })

// System
export const getSystemInfo = () => fetchJSON<any>('/api/system/info')

// Training
export const startBatchSeparation = (params: any) =>
  fetchJSON<{ task_id: string; status: string }>('/api/train/batch-separate', {
    method: 'POST',
    body: JSON.stringify(params),
  })

export const startFullTraining = (params: any) =>
  fetchJSON<{ task_id: string; status: string }>('/api/train/start', {
    method: 'POST',
    body: JSON.stringify(params),
  })

export const startTrainingStep = (params: any) =>
  fetchJSON<{ task_id: string; status: string }>('/api/train/step', {
    method: 'POST',
    body: JSON.stringify(params),
  })

export const stopTraining = (taskId: string) =>
  fetchJSON<any>(`/api/train/stop/${taskId}`, { method: 'POST' })

export const getTrainingStatus = (taskId: string) =>
  fetchJSON<any>(`/api/train/status/${taskId}`)

export const getExperiments = () =>
  fetchJSON<any[]>('/api/train/experiments')

export const getPretrainedModels = () =>
  fetchJSON<{ G: string[]; D: string[] }>('/api/train/pretrained')

export const getDatasetDir = (expName: string) =>
  fetchJSON<{ dataset_dir: string; files: string[]; count: number }>(
    `/api/train/dataset-dir?exp_name=${encodeURIComponent(expName)}`
  )

export const getDatasetAudioURL = (filePath: string) =>
  `${API_BASE}/api/train/dataset-audio?path=${encodeURIComponent(filePath)}`

export const browseFolder = (initialDir = '') =>
  fetchJSON<{ path: string }>('/api/train/browse-folder', {
    method: 'POST',
    body: JSON.stringify({ initial_dir: initialDir }),
  })

export async function uploadDataset(files: File[], expName: string) {
  const formData = new FormData()
  for (const f of files) {
    formData.append('files', f)
  }
  formData.append('exp_name', expName)
  const res = await fetch(`${API_BASE}/api/train/upload-dataset`, {
    method: 'POST',
    body: formData,
  })
  if (!res.ok) {
    const text = await res.text()
    throw new Error(`Upload failed: ${res.status}: ${text}`)
  }
  return res.json()
}

export function connectTraining(taskId: string, onMessage: (msg: any) => void) {
  const wsBase = API_BASE.replace(/^http/, 'ws') || `ws://${window.location.host}`
  const ws = new WebSocket(`${wsBase}/api/train/ws/${taskId}`)
  ws.onmessage = (e) => onMessage(JSON.parse(e.data))
  ws.onerror = (e) => console.error('Training WS error:', e)
  return ws
}

// Cleanup
export const getCleanupStatus = () =>
  fetchJSON<any>('/api/cleanup/status')

export const cleanupTemp = () =>
  fetchJSON<any>('/api/cleanup/temp', { method: 'POST' })

export const cleanupBuffers = () =>
  fetchJSON<any>('/api/cleanup/buffers', { method: 'POST' })

export const removeBuffers = (bufferIds: string[]) =>
  fetchJSON<any>('/api/cleanup/buffers/remove', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(bufferIds),
  })

export const cleanupTrainingData = (expName: string) =>
  fetchJSON<any>(`/api/cleanup/training/${encodeURIComponent(expName)}`, { method: 'POST' })

export const cleanupAllTrainingData = () =>
  fetchJSON<any>('/api/cleanup/training-all', { method: 'POST' })

// ── Electron-aware dialog functions ──

export async function importWorkflowAuto(): Promise<any> {
  if (isElectron()) {
    const filePath = await window.electronAPI!.openFile({
      title: 'Import Project',
      filters: [
        { name: 'Utai Project', extensions: ['utai', 'utaiTr'] },
      ],
    })
    if (!filePath) return { status: 'cancelled' }
    return fetchJSON<any>('/api/workflows/import-path', {
      method: 'POST',
      body: JSON.stringify({ path: filePath }),
    })
  }
  return importWorkflowDialog()
}

export async function exportWorkflowAuto(workflow: any): Promise<any> {
  const isTraining = workflow.project_type === 'training'
  const ext = isTraining ? 'utaiTr' : 'utai'
  const filterName = isTraining ? 'Utai Training Project' : 'Utai Project'

  if (isElectron()) {
    const savePath = await window.electronAPI!.saveFile({
      title: 'Export Project',
      defaultPath: `${workflow.name || 'untitled'}.${ext}`,
      filters: [{ name: filterName, extensions: [ext] }],
    })
    if (!savePath) return { status: 'cancelled' }
    return fetchJSON<any>('/api/workflows/export-to-path', {
      method: 'POST',
      body: JSON.stringify({ path: savePath, workflow }),
    })
  }
  return exportWorkflow(workflow)
}

export async function importWorkflowFromPath(filePath: string): Promise<any> {
  return fetchJSON<any>('/api/workflows/import-path', {
    method: 'POST',
    body: JSON.stringify({ path: filePath }),
  })
}

export async function browseFolderAuto(initialDir = ''): Promise<{ path: string } | null> {
  if (isElectron()) {
    const folderPath = await window.electronAPI!.openFolder({
      title: 'Select Folder',
      defaultPath: initialDir,
    })
    if (!folderPath) return null
    return { path: folderPath }
  }
  return browseFolder(initialDir)
}
