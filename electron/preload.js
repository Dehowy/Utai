const { contextBridge, ipcRenderer } = require('electron')

// Ack close-requested immediately so main knows the renderer process is alive.
// This runs before React mounts, giving the app more time to show its dialog.
ipcRenderer.on('close-requested', () => {
  ipcRenderer.send('close-ack')
})

contextBridge.exposeInMainWorld('electronAPI', {
  isElectron: true,

  openFile: (options) => ipcRenderer.invoke('dialog:open-file', options),
  saveFile: (options) => ipcRenderer.invoke('dialog:save-file', options),
  openFolder: (options) => ipcRenderer.invoke('dialog:open-folder', options),
  getBackendPort: () => ipcRenderer.invoke('get-backend-port'),

  onOpenUtaiFile: (callback) => {
    const handler = (_event, filePath) => callback(filePath)
    ipcRenderer.on('open-utai-file', handler)
    return () => ipcRenderer.removeListener('open-utai-file', handler)
  },

  // Close / minimize-to-tray
  onCloseRequested: (callback) => {
    const handler = (_event, data) => callback(data)
    ipcRenderer.on('close-requested', handler)
    return () => ipcRenderer.removeListener('close-requested', handler)
  },
  sendCloseAction: (action) => {
    ipcRenderer.send('close-action', action)
  },

  onForceAutosave: (callback) => {
    const handler = () => callback()
    ipcRenderer.on('force-autosave', handler)
    return () => ipcRenderer.removeListener('force-autosave', handler)
  },
})
