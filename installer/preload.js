const { contextBridge, ipcRenderer } = require('electron');

contextBridge.exposeInMainWorld('installer', {
  close: () => ipcRenderer.invoke('close'),
  getDefaultPath: () => ipcRenderer.invoke('get-default-path'),
  checkExistingInstall: () => ipcRenderer.invoke('check-existing-install'),
  uninstall: (path) => ipcRenderer.invoke('uninstall', path),
  launchUtai: (installPath) => ipcRenderer.invoke('launch-utai', installPath),
  minimize: () => ipcRenderer.invoke('minimize'),
  browseFolder: (defaultPath) => ipcRenderer.invoke('browse-folder', defaultPath),
  detectGpu: () => ipcRenderer.invoke('detect-gpu'),
  checkDiskSpace: (path) => ipcRenderer.invoke('check-disk-space', path),
  testUrl: (url) => ipcRenderer.invoke('test-url', url),
  startInstall: (config) => ipcRenderer.invoke('start-install', config),
  saveLog: (logText) => ipcRenderer.invoke('save-log', logText),
  onInstallLog: (cb) => {
    ipcRenderer.removeAllListeners('install-log');
    ipcRenderer.on('install-log', (_, msg) => cb(msg));
  },
  onInstallDone: (cb) => {
    ipcRenderer.removeAllListeners('install-done');
    ipcRenderer.on('install-done', (_, success) => cb(success));
  },
});
