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
      onOpenUtaiFile: (callback: (filePath: string) => void) => (() => void) | void
      onCloseRequested: (callback: (data: { fromTray: boolean }) => void) => (() => void) | void
      sendCloseAction: (action: 'minimize' | 'quit' | 'cancel') => void
      onForceAutosave: (callback: () => void) => (() => void) | void
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

// RVC range detection
export async function detectRVCRange(modelName: string, referencePitch = 60) {
  const form = new FormData()
  form.append('model_name', modelName)
  form.append('reference_pitch', String(referencePitch))
  const res = await fetch(`${API_BASE}/api/models/rvc/detect-range`, { method: 'POST', body: form })
  if (!res.ok) throw new Error(`Range detection failed: ${res.status}`)
  return res.json()
}
export const getRVCRangeStatus = (modelName: string) =>
  fetchJSON<any>(`/api/models/rvc/range-status/${encodeURIComponent(modelName)}`)
export const getRVCRange = (modelName: string) =>
  fetchJSON<any>(`/api/models/rvc/range/${encodeURIComponent(modelName)}`)
export const updateRVCRange = (modelName: string, comfortLow: number, comfortHigh: number) =>
  fetchJSON<any>(`/api/models/rvc/range/${encodeURIComponent(modelName)}`, {
    method: 'PUT',
    body: JSON.stringify({ comfort_low: comfortLow, comfort_high: comfortHigh }),
  })
export const resetRVCRange = (modelName: string) =>
  fetchJSON<any>(`/api/models/rvc/reset-range/${encodeURIComponent(modelName)}`, { method: 'POST' })

// DiffSinger removed — replaced by Score2HuBERT. Stubs kept for compile compat.
export const getDiffSingerModels = () => Promise.resolve([])
export const getDiffSingerCatalog = () => Promise.resolve({ vocoders: [], singers: [] })
export const downloadDiffSingerModel = (_id: string) => Promise.resolve({ status: 'removed' })
export const getDiffSingerDownloadStatus = (_id: string) => Promise.resolve({ status: 'not_found', progress: 0 } as any)
export async function importDiffSingerModel(_f: File) { return { status: 'removed' } }
export const archiveDiffSinger = (_id: string) => Promise.resolve({ status: 'removed' })
export const getArchivedDiffSinger = () => Promise.resolve([])
export const restoreDiffSinger = (_n: string) => Promise.resolve({ status: 'removed' })
export const deleteDiffSinger = (_id: string) => Promise.resolve({ status: 'removed' })
export const deleteArchivedDiffSinger = (_n: string) => Promise.resolve({ status: 'removed' })

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
  ws.onmessage = (e) => {
    try {
      onMessage(JSON.parse(e.data))
    } catch (err) {
      console.error('Training WS parse error:', err)
    }
  }
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

// VST plugins
export const getVSTPlugins = () =>
  fetchJSON<any[]>('/api/vst/list')

export const importVSTPlugin = (path: string, name?: string) =>
  fetchJSON<any>('/api/vst/import', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ path, name }),
  })

export const deleteVSTPlugin = (name: string) =>
  fetchJSON<any>(`/api/vst/${encodeURIComponent(name)}`, { method: 'DELETE' })

export const getVSTParams = (name: string) =>
  fetchJSON<any[]>(`/api/vst/params/${encodeURIComponent(name)}`)

export const openVSTEditor = (name: string) =>
  fetchJSON<any>(`/api/vst/editor/open/${encodeURIComponent(name)}`, { method: 'POST' })

export const closeVSTEditor = (name: string) =>
  fetchJSON<any>(`/api/vst/editor/close/${encodeURIComponent(name)}`, { method: 'POST' })

export const previewVSTNote = (pluginName: string, tone: number, velocity = 100, duration = 0.5) =>
  fetch(`${API_BASE}/api/vst/preview`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ plugin_name: pluginName, tone, velocity, duration }),
  }).then(r => {
    if (!r.ok) throw new Error(`VST preview failed: ${r.status}`)
    return r.blob()
  })

export const renderVST = (data: {
  plugin_name: string;
  notes: { tone: number; velocity: number; start: number; duration: number }[];
  bpm: number;
  duration?: number;
  sample_rate?: number;
  params?: Record<number, number>;
}) =>
  fetch(`${API_BASE}/api/vst/render`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(data),
  }).then(r => {
    if (!r.ok) throw new Error(`VST render failed: ${r.status}`)
    return r.blob()
  })

// Effects
export const getEffectTypes = () =>
  fetchJSON<Record<string, any>>('/api/effects/types')

export const applyEffects = (audioPath: string, effects: any[]) =>
  fetch(`${API_BASE}/api/effects/apply`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ audio_path: audioPath, effects }),
  }).then(r => {
    if (!r.ok) throw new Error(`Effects failed: ${r.status}`)
    return r.blob()
  })

export const applyEffectsUpload = (file: Blob, effects: any[]) => {
  const form = new FormData()
  form.append('file', file, 'audio.wav')
  form.append('effects_json', JSON.stringify(effects))
  return fetch(`${API_BASE}/api/effects/apply-upload`, { method: 'POST', body: form })
    .then(r => {
      if (!r.ok) throw new Error(`Effects failed: ${r.status}`)
      return r.blob()
    })
}

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
        { name: 'Utai Project', extensions: ['utai', 'utaiTr', 'utaiSynth'] },
      ],
    })
    if (!filePath) return { status: 'cancelled' }
    if (filePath.endsWith('.utaiSynth')) {
      const data = await fetchJSON<any>('/api/synth/project/open-path', { method: 'POST', body: JSON.stringify({ path: filePath }) })
      return { ...data, project_type: 'synth', path: filePath }
    }
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
