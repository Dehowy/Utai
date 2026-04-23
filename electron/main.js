const { app, BrowserWindow, ipcMain, dialog, Tray, Menu, nativeImage } = require('electron')
const path = require('path')
const net = require('net')
const { spawn } = require('child_process')
const http = require('http')

// Suppress EPIPE / broken pipe errors during shutdown
process.on('uncaughtException', (err) => {
  if (err.code === 'EPIPE' || err.code === 'ERR_STREAM_DESTROYED') return
  console.error('Uncaught exception:', err)
})

// Prevent multiple instances — forward file args to the existing one
const gotLock = app.requestSingleInstanceLock()
if (!gotLock) {
  app.quit()
  process.exit(0)
}

let mainWindow = null
let pythonProcess = null
let backendPort = 8765
let pendingUtaiFile = null // .utai path queued before renderer is ready
let tray = null
let closeFallbackTimer = null
app.isQuitting = false

// ── Path helpers ──

function getResourcePath() {
  if (app.isPackaged) {
    // In packaged mode, resourcesPath = <install>/resources
    // But app files (python/, app/, synth/) are at <install>/ (parent)
    return path.join(process.resourcesPath, '..')
  }
  return path.join(__dirname, '..')
}

function getPythonPath() {
  const resPath = getResourcePath()
  const fs = require('fs')
  // Dev: venv, Installed: python/  (prefer pythonw for windowsHide)
  for (const p of ['venv/Scripts/pythonw.exe', 'python/pythonw.exe', 'venv/Scripts/python.exe', 'python/python.exe']) {
    const full = path.join(resPath, p)
    if (fs.existsSync(full)) { console.log(`[utai] Python found: ${full}`); return full }
  }
  const fallback = path.join(resPath, 'python', 'python.exe')
  console.error(`[utai] WARNING: No Python found in ${resPath}, using fallback: ${fallback}`)
  return fallback
}

// ── Port management ──

function findFreePort(startPort) {
  return new Promise((resolve, reject) => {
    const server = net.createServer()
    server.listen(startPort, '127.0.0.1', () => {
      const port = server.address().port
      server.close(() => resolve(port))
    })
    server.on('error', () => {
      if (startPort < 9000) {
        resolve(findFreePort(startPort + 1))
      } else {
        reject(new Error('No free port found'))
      }
    })
  })
}

// ── Backend spawning ──

function startBackend(port) {
  const resPath = getResourcePath()
  const pythonExe = getPythonPath()
  const binDir = path.join(resPath, 'bin')

  // Build PATH with our bin/ dir prepended
  const envPath = process.env.PATH || ''
  const newPath = binDir + path.delimiter + envPath

  const pythonDir = path.join(resPath, 'python')
  const sitePackages = path.join(pythonDir, 'Lib', 'site-packages')

  const env = {
    ...process.env,
    PATH: [binDir, pythonDir, path.join(pythonDir, 'Scripts'), newPath].join(path.delimiter),
    PORT: String(port),
    HOST: '127.0.0.1',
    PYTHONPATH: [resPath, sitePackages].join(path.delimiter),
    PYTHONUNBUFFERED: '1',
  }

  // Tell backend to serve static frontend (Electron production mode)
  env.UTAI_SERVE_STATIC = '1'
  delete env.UTAI_DEV

  const args = [
    '-m', 'uvicorn',
    'app.backend.main:app',
    '--host', '127.0.0.1',
    '--port', String(port),
  ]

  console.log(`[utai] Starting backend: ${pythonExe} ${args.join(' ')}`)
  console.log(`[utai] CWD: ${resPath}`)

  pythonProcess = spawn(pythonExe, args, {
    cwd: resPath,
    env,
    stdio: ['ignore', 'pipe', 'pipe'],
    windowsHide: true,
  })

  // Suppress EPIPE errors when pipe breaks during shutdown
  pythonProcess.stdout.on('data', (data) => {
    try { console.log(`[backend] ${data.toString().trim()}`) } catch {}
  })
  pythonProcess.stdout.on('error', () => {})

  let lastStderr = []
  pythonProcess.stderr.on('data', (data) => {
    const line = data.toString().trim()
    try { console.log(`[backend] ${line}`) } catch {}
    lastStderr.push(line)
    if (lastStderr.length > 20) lastStderr.shift()
  })
  pythonProcess.stderr.on('error', () => {})

  pythonProcess.on('exit', (code) => {
    const wasExpected = pythonProcess === null || app.isQuitting
    pythonProcess = null
    if (!wasExpected && mainWindow && !mainWindow.isDestroyed()) {
      const errDetail = lastStderr.join('\n')
      dialog.showErrorBox('Utai', `Backend exited (code ${code}).\n\n${errDetail}`)
      app.quit()
    }
  })

  pythonProcess.on('error', (err) => {
    console.error(`[utai] Failed to start backend:`, err)
    dialog.showErrorBox('Utai', `Failed to start Python backend:\n${err.message}`)
    app.quit()
  })
}

// ── Health check ──

function waitForBackend(port, timeoutMs = 60000) {
  const startTime = Date.now()
  return new Promise((resolve, reject) => {
    function checkApi() {
      if (Date.now() - startTime > timeoutMs) {
        reject(new Error('Backend startup timed out'))
        return
      }
      const req = http.get(`http://127.0.0.1:${port}/api/system/info`, (res) => {
        if (res.statusCode === 200) {
          checkRoot()
        } else {
          setTimeout(checkApi, 500)
        }
      })
      req.on('error', () => setTimeout(checkApi, 500))
      req.setTimeout(2000, () => { req.destroy(); setTimeout(checkApi, 500) })
    }
    function checkRoot() {
      if (Date.now() - startTime > timeoutMs) {
        console.warn('[utai] Timed out waiting for root page — loading anyway')
        resolve()
        return
      }
      const req = http.get(`http://127.0.0.1:${port}/`, (res) => {
        let body = ''
        res.on('data', d => { body += d.toString().slice(0, 200) })
        res.on('end', () => {
          if (res.statusCode === 200 && body.includes('<!DOCTYPE')) {
            resolve()
          } else {
            console.log(`[utai] Root: status=${res.statusCode} html=${body.includes('<!DOCTYPE')}`)
            setTimeout(checkRoot, 500)
          }
        })
      })
      req.on('error', () => setTimeout(checkRoot, 500))
      req.setTimeout(2000, () => { req.destroy(); setTimeout(checkRoot, 500) })
    }
    checkApi()
  })
}

// ── Window creation ──

function createWindow() {
  mainWindow = new BrowserWindow({
    width: 1400,
    height: 900,
    minWidth: 1000,
    minHeight: 600,
    title: 'Utai',
    icon: path.join(__dirname, 'build', 'icon.png'),
    autoHideMenuBar: true,
    webPreferences: {
      preload: path.join(__dirname, 'preload.js'),
      contextIsolation: true,
      nodeIntegration: false,
    },
    show: false,
    backgroundColor: '#0a0a1a',
  })

  // Remove the application menu entirely (no File/Edit/View/Window/Help)
  mainWindow.setMenu(null)

  // Block Chromium-level shortcuts that bypass JavaScript preventDefault
  mainWindow.webContents.on('before-input-event', (event, input) => {
    if (input.type !== 'keyDown') return
    const k = input.key.toLowerCase()
    if ((input.control || input.meta) && ['r', 'w', 'n'].includes(k)) {
      event.preventDefault()
    }
  })

  // Disable HTTP cache so CSS/JS updates take effect immediately
  mainWindow.webContents.session.clearCache()

  // Prevent navigation to external URLs
  mainWindow.webContents.on('will-navigate', (event, url) => {
    try {
      const parsed = new URL(url)
      if (parsed.hostname !== '127.0.0.1' && parsed.hostname !== 'localhost' && parsed.protocol !== 'file:') {
        event.preventDefault()
      }
    } catch { /* allow */ }
  })

  // Prevent opening new windows
  mainWindow.webContents.setWindowOpenHandler(() => ({ action: 'deny' }))

  // Load the splash screen first
  mainWindow.loadFile(path.join(__dirname, 'loading.html'))
  mainWindow.once('ready-to-show', () => {
    mainWindow.show()
  })

  // Intercept close — ask renderer for minimize-to-tray / quit dialog.
  // If renderer can't respond (React failed to load), fall back to native dialog.
  mainWindow.on('close', (event) => {
    if (app.isQuitting) return
    event.preventDefault()
    if (mainWindow && mainWindow.webContents) {
      mainWindow.webContents.send('close-requested', { fromTray: false })
      if (closeFallbackTimer) clearTimeout(closeFallbackTimer)
      closeFallbackTimer = setTimeout(() => {
        nativeFallbackQuit()
      }, 2000)
    }
  })

  mainWindow.on('closed', () => {
    mainWindow = null
  })
}

// ── .utai file handling ──

function extractUtaiPath(argv) {
  for (const arg of argv) {
    const lower = arg.toLowerCase()
    if ((lower.endsWith('.utai') || lower.endsWith('.utaitr')) && !arg.startsWith('--')) {
      return arg
    }
  }
  return null
}

function sendUtaiFileToRenderer(filePath) {
  if (mainWindow && mainWindow.webContents) {
    mainWindow.webContents.send('open-utai-file', filePath)
  }
}

// ── IPC handlers ──

ipcMain.handle('dialog:open-file', async (_event, options) => {
  const result = await dialog.showOpenDialog(mainWindow, {
    title: options?.title || 'Open File',
    filters: options?.filters || [{ name: 'All Files', extensions: ['*'] }],
    properties: ['openFile'],
  })
  if (result.canceled || result.filePaths.length === 0) return null
  return result.filePaths[0]
})

ipcMain.handle('dialog:save-file', async (_event, options) => {
  const result = await dialog.showSaveDialog(mainWindow, {
    title: options?.title || 'Save File',
    defaultPath: options?.defaultPath || '',
    filters: options?.filters || [{ name: 'All Files', extensions: ['*'] }],
  })
  if (result.canceled) return null
  return result.filePath
})

ipcMain.handle('dialog:open-folder', async (_event, options) => {
  const result = await dialog.showOpenDialog(mainWindow, {
    title: options?.title || 'Select Folder',
    defaultPath: options?.defaultPath || '',
    properties: ['openDirectory'],
  })
  if (result.canceled || result.filePaths.length === 0) return null
  return result.filePaths[0]
})

ipcMain.handle('get-backend-port', () => backendPort)

// Close / minimize-to-tray IPC
ipcMain.on('close-action', (_event, action) => {
  if (closeFallbackTimer) { clearTimeout(closeFallbackTimer); closeFallbackTimer = null }
  if (action === 'minimize') {
    if (mainWindow && !mainWindow.isDestroyed()) {
      mainWindow.hide()
    }
  } else if (action === 'quit') {
    app.isQuitting = true
    app.quit()
  }
  // 'cancel' → do nothing
})

// Renderer ack — preload.js sends this immediately on close-requested,
// before React mounts.  Extend the fallback timer to give React time to load.
ipcMain.on('close-ack', () => {
  if (!closeFallbackTimer) return
  clearTimeout(closeFallbackTimer)
  closeFallbackTimer = setTimeout(() => {
    if (!mainWindow || mainWindow.isDestroyed()) return
    nativeFallbackQuit()
  }, 12000)
})

// ── App lifecycle ──

app.on('second-instance', (_event, argv) => {
  // Another instance tried to start — check if it has a .utai file
  const filePath = extractUtaiPath(argv)
  if (filePath) {
    sendUtaiFileToRenderer(filePath)
  }
  if (mainWindow) {
    if (mainWindow.isMinimized()) mainWindow.restore()
    mainWindow.show() // Ensure window is visible (may be hidden to tray)
    mainWindow.focus()
  }
})

app.whenReady().then(async () => {
  // Check for .utai file in command-line args
  pendingUtaiFile = extractUtaiPath(process.argv)

  createWindow()

  // ── System tray ──
  try {
    const iconPath = path.join(__dirname, 'build', 'icon.ico')
    tray = new Tray(iconPath)
    tray.setToolTip('Utai')
    const trayMenu = Menu.buildFromTemplate([
      { label: '「Utai」', enabled: false },
      { type: 'separator' },
      { label: 'Show', click: () => { if (mainWindow && !mainWindow.isDestroyed()) { mainWindow.show(); mainWindow.focus() } } },
      { type: 'separator' },
      { label: 'Quit', click: () => {
        if (mainWindow && !mainWindow.isDestroyed()) {
          mainWindow.show()
          mainWindow.webContents.send('close-requested', { fromTray: true })
          if (closeFallbackTimer) clearTimeout(closeFallbackTimer)
          closeFallbackTimer = setTimeout(() => {
            app.isQuitting = true
            app.quit()
          }, 2000)
        } else {
          app.isQuitting = true
          app.quit()
        }
      }},
    ])
    tray.setContextMenu(trayMenu)
    tray.on('click', () => {
      if (mainWindow && !mainWindow.isDestroyed()) {
        mainWindow.show()
        mainWindow.focus()
      }
    })
  } catch (err) {
    console.log('[utai] Tray creation skipped:', err.message)
  }

  try {
    backendPort = await findFreePort(8765)
    console.log(`[utai] Using port ${backendPort}`)

    startBackend(backendPort)
    await waitForBackend(backendPort)

    console.log('[utai] Backend is ready')
    mainWindow.loadURL(`http://127.0.0.1:${backendPort}`)

    if (!app.isPackaged) {
      mainWindow.webContents.openDevTools({ mode: 'detach' });
    }

    mainWindow.webContents.on('did-finish-load', () => {
      // Send any pending .utai file to the renderer
      if (pendingUtaiFile) {
        // Small delay to ensure React app is initialized
        setTimeout(() => {
          sendUtaiFileToRenderer(pendingUtaiFile)
          pendingUtaiFile = null
        }, 1000)
      }
    })
  } catch (err) {
    console.error('[utai] Startup failed:', err)
    dialog.showErrorBox('Utai', `Failed to start:\n${err.message}`)
    app.quit()
  }
})

app.on('window-all-closed', () => {
  // Only quit if explicitly requested (not just hidden to tray)
  if (app.isQuitting) {
    killBackend()
    app.quit()
  }
})

app.on('before-quit', () => {
  app.isQuitting = true
  killBackend()
})

function nativeFallbackQuit() {
  if (!mainWindow || mainWindow.isDestroyed()) return
  const choice = dialog.showMessageBoxSync(mainWindow, {
    type: 'question',
    buttons: ['Quit', 'Cancel'],
    defaultId: 0,
    title: 'Utai',
    message: 'Quit Utai?',
  })
  if (choice === 0) {
    // Try to trigger autosave via the renderer before quitting.
    // Send force-autosave and give React 1.5s to flush.
    if (mainWindow && mainWindow.webContents) {
      mainWindow.webContents.send('force-autosave')
    }
    setTimeout(() => {
      app.isQuitting = true
      app.quit()
    }, 1500)
  }
}

function killBackend() {
  if (pythonProcess) {
    console.log('[utai] Stopping backend...')
    const pid = pythonProcess.pid
    try {
      // On Windows, kill the entire process tree silently
      const { execFileSync } = require('child_process')
      execFileSync('taskkill', ['/pid', String(pid), '/f', '/t'], {
        windowsHide: true,
        stdio: 'ignore',
      })
    } catch {
      try { pythonProcess.kill() } catch { /* ignore */ }
    }
    pythonProcess = null
  }
}
