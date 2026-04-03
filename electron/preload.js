const { contextBridge, ipcRenderer } = require('electron')

contextBridge.exposeInMainWorld('electronAPI', {
  isElectron: true,

  openFile: (options) => ipcRenderer.invoke('dialog:open-file', options),
  saveFile: (options) => ipcRenderer.invoke('dialog:save-file', options),
  openFolder: (options) => ipcRenderer.invoke('dialog:open-folder', options),
  getBackendPort: () => ipcRenderer.invoke('get-backend-port'),

  onOpenUtaiFile: (callback) => {
    ipcRenderer.on('open-utai-file', (_event, filePath) => callback(filePath))
  },
})
