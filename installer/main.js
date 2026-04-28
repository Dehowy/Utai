const { app, BrowserWindow, ipcMain, dialog } = require('electron');
const path = require('path');
const fs = require('fs');
const fsP = require('fs/promises');
const { execSync, spawn } = require('child_process');

let mainWindow;
let activeChildren = [];

const gotLock = app.requestSingleInstanceLock();
if (!gotLock) {
  app.quit();
  process.exit(0);
}

app.on('second-instance', () => {
  if (mainWindow) {
    if (mainWindow.isMinimized()) mainWindow.restore();
    mainWindow.focus();
  }
});

app.whenReady().then(() => {
  mainWindow = new BrowserWindow({
    width: 680,
    height: 520,
    resizable: false,
    frame: false,
    transparent: false,
    backgroundColor: '#08080f',
    webPreferences: {
      preload: path.join(__dirname, 'preload.js'),
      contextIsolation: true,
      nodeIntegration: false,
    },
    icon: path.join(__dirname, 'icon.ico'),
  });
  mainWindow.loadFile('index.html');
});

app.on('window-all-closed', () => app.quit());

app.on('before-quit', () => {
  for (const child of activeChildren) {
    try { child.kill(); } catch {}
  }
  activeChildren = [];
});

// ── IPC: Window controls ──
ipcMain.handle('close', () => app.quit());
ipcMain.handle('minimize', () => mainWindow.minimize());

// Persistent install record (survives portable app cleanup)
function getInstallRecordPath() {
  const appData = process.env.APPDATA || app.getPath('userData');
  const dir = path.join(appData, 'Utai');
  fs.mkdirSync(dir, { recursive: true });
  return path.join(dir, 'install_info.json');
}

function readInstallRecords() {
  const recordPath = getInstallRecordPath();
  if (!fs.existsSync(recordPath)) return [];
  try {
    const data = JSON.parse(fs.readFileSync(recordPath, 'utf-8'));
    if (Array.isArray(data.installations)) return data.installations;
    if (data.installPath) return [data];
    return [];
  } catch { return []; }
}

function writeInstallRecords(records) {
  const recordPath = getInstallRecordPath();
  fs.writeFileSync(recordPath, JSON.stringify({ installations: records }, null, 2), 'utf-8');
}

// ── IPC: Existing install detection ──
ipcMain.handle('check-existing-install', () => {
  const records = readInstallRecords();
  const valid = records.filter(r => r.installPath && fs.existsSync(r.installPath));
  if (valid.length !== records.length) writeInstallRecords(valid);
  return { installations: valid };
});

// ── IPC: Uninstall ──
ipcMain.handle('uninstall', async (_, installPath) => {
  const log = (msg) => mainWindow.webContents.send('install-log', msg);
  try {
    log('[STAGE] Uninstalling Utai...');
    log('[PROGRESS] 10');

    try {
      const targetExe = path.join(installPath, 'Utai.exe').replace(/'/g, "''");
      const psCmd = `$desktop = [Environment]::GetFolderPath('Desktop'); $lnk = Join-Path $desktop 'Utai.lnk'; if (Test-Path $lnk) { $ws = New-Object -ComObject WScript.Shell; $sc = $ws.CreateShortcut($lnk); if ($sc.TargetPath -eq '${targetExe}') { Remove-Item $lnk -Force; Write-Output 'removed' } }`;
      const out = execSync(`C:\\Windows\\System32\\WindowsPowerShell\\v1.0\\powershell.exe -NoProfile -Command "${psCmd}"`, { encoding: 'utf-8', timeout: 10000 }).trim();
      if (out === 'removed') log('  Desktop shortcut removed');
    } catch { log('  [!] Could not check desktop shortcut'); }
    log('[PROGRESS] 20');

    log(`  Removing: ${installPath}`);
    if (fs.existsSync(installPath)) {
      await fsP.rm(installPath, { recursive: true, force: true });
      log('  Installation directory removed');
    }
    log('[PROGRESS] 80');

    const records = readInstallRecords();
    const updated = records.filter(r => r.installPath !== installPath);
    if (updated.length > 0) {
      writeInstallRecords(updated);
    } else {
      const installRecord = getInstallRecordPath();
      if (fs.existsSync(installRecord)) fs.unlinkSync(installRecord);
    }
    log('  Install record cleaned');

    log('[PROGRESS] 100');
    log('[STAGE] Uninstall complete!');
    return { success: true };
  } catch (err) {
    log(`[!] Uninstall error: ${err.message}`);
    return { success: false, error: err.message };
  }
});

// ── IPC: Launch installed app ──
ipcMain.handle('launch-utai', (_, installPath) => {
  const utaiExe = path.join(installPath, 'Utai.exe');
  if (!fs.existsSync(utaiExe)) return { ok: false, error: 'Utai.exe not found' };
  try {
    spawn(utaiExe, [], { cwd: installPath, detached: true, stdio: 'ignore' }).unref();
    return { ok: true };
  } catch (e) {
    return { ok: false, error: e.message };
  }
});

// ── IPC: Default install path ──
ipcMain.handle('get-default-path', () => {
  try {
    const output = execSync(
      'C:\\Windows\\System32\\WindowsPowerShell\\v1.0\\powershell.exe -NoProfile -Command "Get-WmiObject Win32_LogicalDisk | Where-Object { $_.DriveType -eq 3 } | Select-Object -ExpandProperty DeviceID"',
      { encoding: 'utf-8', timeout: 5000 }
    );
    const drives = output.trim().split(/\r?\n/).map(d => d.trim()).filter(Boolean).sort();
    const nonSystem = drives.filter(d => d.toUpperCase() !== 'C:');
    const drive = nonSystem.length > 0 ? nonSystem[0] : (drives[0] || 'C:');
    return drive + '\\Utai';
  } catch {
    return 'D:\\Utai';
  }
});

// ── IPC: Browse folder ──
ipcMain.handle('browse-folder', async (_, defaultPath) => {
  const result = await dialog.showOpenDialog(mainWindow, {
    properties: ['openDirectory', 'createDirectory'],
    defaultPath: defaultPath || 'D:\\',
  });
  return result.canceled ? null : result.filePaths[0];
});

// ── IPC: GPU detection ──
ipcMain.handle('detect-gpu', () => {
  try {
    const nvsmi = 'C:\\Windows\\System32\\nvidia-smi.exe';
    const output = execSync(`"${nvsmi}" --query-gpu=name --format=csv,noheader`, { encoding: 'utf-8', timeout: 5000 });
    const name = output.trim().split('\n')[0];
    const is50 = /RTX\s*50\d0/i.test(name);
    return { detected: true, name, recommended: is50 ? 'cu128' : 'cu121' };
  } catch {
    return { detected: false, name: null, recommended: 'cpu' };
  }
});

// ── IPC: Disk space ──
ipcMain.handle('check-disk-space', (_, targetPath) => {
  try {
    const drive = targetPath.charAt(0).toUpperCase();
    const output = execSync(
      `C:\\Windows\\System32\\WindowsPowerShell\\v1.0\\powershell.exe -NoProfile -Command "(Get-PSDrive ${drive}).Free"`,
      { encoding: 'utf-8', timeout: 5000 }
    );
    const bytes = parseInt(output.trim());
    return isNaN(bytes) ? -1 : bytes;
  } catch { return -1; }
});

// ── IPC: Network test ──
ipcMain.handle('test-url', async (_, url) => {
  const https = require('https');
  const http = require('http');
  const mod = url.startsWith('https') ? https : http;
  return new Promise(resolve => {
    const start = Date.now();
    const req = mod.request(url, { method: 'HEAD', timeout: 8000 }, res => {
      resolve({ reachable: res.statusCode < 400, latency: Date.now() - start });
    });
    req.on('error', () => resolve({ reachable: false }));
    req.on('timeout', () => { req.destroy(); resolve({ reachable: false }); });
    req.end();
  });
});

// ── Helpers ──

function resolveBundlePath() {
  const fromResources = path.join(process.resourcesPath, 'bundle');
  if (fs.existsSync(fromResources)) return fromResources;
  const fromDir = path.join(__dirname, 'bundle');
  if (fs.existsSync(fromDir)) return fromDir;
  return null;
}

async function copyDir(src, dest) {
  await fsP.cp(src, dest, { recursive: true, force: true });
}

function trackedSpawn(...args) {
  const child = spawn(...args);
  activeChildren.push(child);
  child.on('close', () => {
    const idx = activeChildren.indexOf(child);
    if (idx >= 0) activeChildren.splice(idx, 1);
  });
  return child;
}

// ── IPC: Save installation log ──
ipcMain.handle('save-log', async (_, logText) => {
  const downloadsPath = app.getPath('downloads');
  const result = await dialog.showSaveDialog(mainWindow, {
    title: 'Save Installation Log',
    defaultPath: path.join(downloadsPath, 'utai_install_log.txt'),
    filters: [{ name: 'Text files', extensions: ['txt'] }],
  });
  if (result.canceled || !result.filePath) return { ok: false };
  try {
    fs.writeFileSync(result.filePath, logText, 'utf-8');
    return { ok: true, path: result.filePath };
  } catch (e) {
    return { ok: false, error: e.message };
  }
});

// ── IPC: Main install handler ──

ipcMain.handle('start-install', (_, config) => {
  const { installPath, components, gpu, mirrors } = config;

  const bundlePath = resolveBundlePath();

  if (!bundlePath) {
    const log = (msg) => { try { mainWindow.webContents.send('install-log', msg); } catch {} };
    log('[!] FATAL: Installation bundle not found.');
    log('[!] This may happen if another instance of the installer is running.');
    log('[!] Please close all other Utai Installer windows and try again.');
    try { mainWindow.webContents.send('install-done', false); } catch {}
    return true;
  }

  const log = (msg) => { try { mainWindow.webContents.send('install-log', msg); } catch {} };
  const stage = (pct, msg) => { log(`[STAGE] ${msg}`); log(`[PROGRESS] ${pct}`); };
  const fail = (reason) => {
    log(`[!] FATAL: ${reason}`);
    try { mainWindow.webContents.send('install-done', false); } catch {}
  };

  (async () => {
    try {
      // ── 1. Create install directory ──
      stage(2, 'Creating directories...');
      await fsP.mkdir(installPath, { recursive: true });

      // ── 2. Copy Electron app (Utai.exe + runtime) ──
      stage(5, 'Copying Utai application...');
      const utaiAppSrc = path.join(bundlePath, 'utai-app', 'Utai-win32-x64');
      if (fs.existsSync(utaiAppSrc)) {
        await copyDir(utaiAppSrc, installPath);
        log('  Utai.exe + runtime copied');
      } else {
        log('  [!] WARNING: utai-app/Utai-win32-x64 not in bundle — Utai.exe will be missing!');
      }

      // ── 3. Copy Python backend + built frontend ──
      stage(10, 'Copying backend...');
      const appSrc = path.join(bundlePath, 'app');
      if (!fs.existsSync(appSrc)) { fail('Bundle missing: app/ directory'); return; }
      await copyDir(appSrc, path.join(installPath, 'app'));
      log('  app/ copied');

      // ── 4. Copy synth components (if selected) ──
      if (components.synth) {
        stage(14, 'Copying synth model...');
        const s2hSrc = path.join(bundlePath, 'score2hubert_v2');
        if (fs.existsSync(s2hSrc)) {
          await copyDir(s2hSrc, path.join(installPath, 'score2hubert_v2'));
          log('  score2hubert_v2/ copied');
        } else {
          log('  [!] WARNING: score2hubert_v2/ not in bundle');
        }
        const synthSrc = path.join(bundlePath, 'synth');
        if (fs.existsSync(synthSrc)) {
          await copyDir(synthSrc, path.join(installPath, 'synth'));
          log('  synth/ copied');
        } else {
          log('  [!] WARNING: synth/ not in bundle');
        }
        const vstHostSrc = path.join(bundlePath, 'VST', 'host');
        if (fs.existsSync(vstHostSrc)) {
          await copyDir(vstHostSrc, path.join(installPath, 'VST', 'host'));
          log('  VST host copied');
        }
      }

      // ── 5. Download RVC / MSST (if needed) ──
      if (components.audio || components.synth) {
        const ghMirror = mirrors.github || '';
        const hasGit = (() => { try { execSync('git --version', { stdio: 'ignore', timeout: 5000 }); return true; } catch { return false; } })();

        const download = async (repoOwner, repoName, dest, label) => {
          if (fs.existsSync(dest)) { log(`  ${label} already exists, skipping`); return; }
          if (hasGit) {
            log(`  Cloning ${label} via git...`);
            const gitUrl = ghMirror
              ? `${ghMirror}/https://github.com/${repoOwner}/${repoName}.git`
              : `https://github.com/${repoOwner}/${repoName}.git`;
            const proc = trackedSpawn('git', ['clone', '--depth', '1', gitUrl, dest]);
            proc.stdout.on('data', d => log('  ' + d.toString().trim()));
            proc.stderr.on('data', d => log('  ' + d.toString().trim()));
            const code = await new Promise(resolve => proc.on('close', resolve));
            if (code === 0) { log(`  ${label} cloned`); return; }
            log(`  [!] git clone failed (code ${code}), trying zip download...`);
          }
          log(`  Downloading ${label} as zip...`);
          const zipUrl = ghMirror
            ? `${ghMirror}/https://github.com/${repoOwner}/${repoName}/archive/refs/heads/main.zip`
            : `https://github.com/${repoOwner}/${repoName}/archive/refs/heads/main.zip`;
          const https = require('https');
          const http = require('http');
          const parentDir = path.dirname(dest);
          fs.mkdirSync(parentDir, { recursive: true });
          const zipPath = path.join(parentDir, `${repoName}.zip`);
          try {
            await new Promise((resolve, reject) => {
              const file = fs.createWriteStream(zipPath);
              const get = (url, redirects = 0) => {
                if (redirects > 5) { reject(new Error('Too many redirects')); return; }
                const mod = url.startsWith('https') ? https : http;
                mod.get(url, { headers: { 'User-Agent': 'Utai-Installer' }, timeout: 120000 }, res => {
                  if (res.statusCode >= 300 && res.statusCode < 400 && res.headers.location) {
                    get(res.headers.location, redirects + 1); return;
                  }
                  if (res.statusCode !== 200) { file.close(); reject(new Error(`HTTP ${res.statusCode} for ${label}`)); return; }
                  res.pipe(file);
                  file.on('finish', () => { file.close(resolve); });
                }).on('error', err => { file.close(); reject(err); })
                  .on('timeout', function() { this.destroy(); file.close(); reject(new Error(`Timeout downloading ${label}`)); });
              };
              get(zipUrl);
            });
            log(`  Extracting ${label}...`);
            const psZip = zipPath.replace(/'/g, "''");
            const psOut = parentDir.replace(/'/g, "''");
            execSync(`C:\\Windows\\System32\\WindowsPowerShell\\v1.0\\powershell.exe -NoProfile -Command "Expand-Archive -LiteralPath '${psZip}' -DestinationPath '${psOut}' -Force"`, { timeout: 300000 });
            const extracted = path.join(parentDir, `${repoName}-main`);
            if (fs.existsSync(extracted)) {
              if (fs.existsSync(dest)) fs.rmSync(dest, { recursive: true, force: true });
              fs.renameSync(extracted, dest);
              log(`  ${label} downloaded`);
            } else {
              log(`  [!] ${label} extraction produced unexpected structure`);
            }
          } catch (e) {
            log(`  [!] ${label} download/extract failed: ${e.message}`);
          } finally {
            try { fs.unlinkSync(zipPath); } catch {}
          }
        };

        const cpRecursive = (src, dest) => {
          fs.mkdirSync(dest, { recursive: true });
          for (const entry of fs.readdirSync(src, { withFileTypes: true })) {
            const s = path.join(src, entry.name), d = path.join(dest, entry.name);
            if (entry.isDirectory()) cpRecursive(s, d);
            else fs.copyFileSync(s, d);
          }
        };

        if (components.audio) {
          stage(18, 'Downloading audio tools (MSST)...');
          await download('ZFTurbo', 'Music-Source-Separation-Training', path.join(installPath, 'MSST'), 'MSST');
          // Upstream repo lacks data/models_info.json — copy from bundle
          const msstDataSrc = path.join(bundlePath, 'msst-data');
          if (fs.existsSync(msstDataSrc)) {
            const msstDataDest = path.join(installPath, 'MSST', 'data');
            fs.mkdirSync(msstDataDest, { recursive: true });
            for (const f of fs.readdirSync(msstDataSrc)) {
              fs.copyFileSync(path.join(msstDataSrc, f), path.join(msstDataDest, f));
            }
            log('  MSST data files installed');
          }
          // Copy bundled yt-dlp source (has local patches for YouTube 412 fix)
          const ytdlpSrc = path.join(bundlePath, 'yt-dlp');
          if (fs.existsSync(ytdlpSrc)) {
            cpRecursive(ytdlpSrc, path.join(installPath, 'yt-dlp'));
            log('  yt-dlp source installed');
          }

          // Overlay modified MSST code — upstream structure differs from what our wrappers expect
          const patchSrc = path.join(bundlePath, 'msst-patches');
          if (fs.existsSync(patchSrc)) {
            const msstRoot = path.join(installPath, 'MSST');
            // Remove upstream inference.py file — replaced by inference/ package
            const upstreamInf = path.join(msstRoot, 'inference.py');
            if (fs.existsSync(upstreamInf)) { try { fs.unlinkSync(upstreamInf); } catch {} }
            // Copy inference/ package and modules/
            if (fs.existsSync(path.join(patchSrc, 'inference')))
              cpRecursive(path.join(patchSrc, 'inference'), path.join(msstRoot, 'inference'));
            if (fs.existsSync(path.join(patchSrc, 'modules')))
              cpRecursive(path.join(patchSrc, 'modules'), path.join(msstRoot, 'modules'));
            // Per-model configs (upstream repo doesn't have these)
            if (fs.existsSync(path.join(patchSrc, 'configs')))
              cpRecursive(path.join(patchSrc, 'configs'), path.join(msstRoot, 'configs'));
            // Patch utils/ — upstream has different files, our inference code needs ours
            if (fs.existsSync(path.join(patchSrc, 'utils')))
              cpRecursive(path.join(patchSrc, 'utils'), path.join(msstRoot, 'utils'));
            log('  MSST code patches applied');
          }
        }

        stage(25, 'Downloading voice conversion (RVC)...');
        await download('RVC-Project', 'Retrieval-based-Voice-Conversion-WebUI', path.join(installPath, 'RVC', 'RVC20240604Nvidia'), 'RVC');

        // Download RVC model files (not included in git repo)
        const hfMirror = mirrors.huggingface || '';
        const hfBase = hfMirror || 'https://huggingface.co';
        const rvcAssets = path.join(installPath, 'RVC', 'RVC20240604Nvidia', 'assets');
        const downloadFile = async (url, dest, label) => {
          if (fs.existsSync(dest)) { log(`  ${label} already exists`); return; }
          fs.mkdirSync(path.dirname(dest), { recursive: true });
          log(`  Downloading ${label}...`);
          const https = require('https');
          const http = require('http');
          await new Promise((resolve, reject) => {
            const file = fs.createWriteStream(dest);
            const get = (u, redirects = 0) => {
              if (redirects > 5) { reject(new Error('Too many redirects')); return; }
              const mod = u.startsWith('https') ? https : http;
              mod.get(u, { headers: { 'User-Agent': 'Utai-Installer' }, timeout: 300000 }, res => {
                if (res.statusCode >= 300 && res.statusCode < 400 && res.headers.location) {
                  get(res.headers.location, redirects + 1); return;
                }
                if (res.statusCode !== 200) { file.close(); try { fs.unlinkSync(dest); } catch {} reject(new Error(`HTTP ${res.statusCode}`)); return; }
                res.pipe(file);
                file.on('finish', () => { file.close(resolve); });
              }).on('error', err => { file.close(); try { fs.unlinkSync(dest); } catch {} reject(err); })
                .on('timeout', function() { this.destroy(); file.close(); try { fs.unlinkSync(dest); } catch {} reject(new Error('Timeout')); });
            };
            get(url);
          });
          log(`  ${label} downloaded`);
        };
        try {
          await downloadFile(
            `${hfBase}/lj1995/VoiceConversionWebUI/resolve/main/hubert_base.pt`,
            path.join(rvcAssets, 'hubert', 'hubert_base.pt'), 'HuBERT model (189MB)');
          await downloadFile(
            `${hfBase}/lj1995/VoiceConversionWebUI/resolve/main/rmvpe.pt`,
            path.join(rvcAssets, 'rmvpe', 'rmvpe.pt'), 'RMVPE model (181MB)');
        } catch (e) {
          log(`  [!] RVC model download failed: ${e.message}`);
          log('  [!] Voice conversion will not work without these models.');
        }

        // ── 5a-2. Download pretrained_v2 models (gitignored in upstream repo) ──
        stage(22, 'Downloading RVC pretrained models...');
        const pretrainedDir = path.join(rvcAssets, 'pretrained_v2');
        fs.mkdirSync(pretrainedDir, { recursive: true });
        const pretrainedBase = `${hfBase}/lj1995/VoiceConversionWebUI/resolve/main/pretrained_v2`;
        const pretrainedFiles = [
          'f0G32k.pth', 'f0G40k.pth', 'f0G48k.pth',
          'f0D32k.pth', 'f0D40k.pth', 'f0D48k.pth',
          'G32k.pth', 'G40k.pth', 'G48k.pth',
          'D32k.pth', 'D40k.pth', 'D48k.pth',
        ];
        try {
          for (const pf of pretrainedFiles) {
            const dest = path.join(pretrainedDir, pf);
            if (fs.existsSync(dest)) { continue; }
            await downloadFile(`${pretrainedBase}/${pf}`, dest, `pretrained_v2/${pf}`);
          }
          log('  Pretrained v2 models downloaded');
        } catch (e) {
          log(`  [!] Pretrained model download failed: ${e.message}`);
          log('  [!] RVC training will not work without pretrained models.');
        }

        // ── 5b. Apply RVC patches ──
        const rvcPatchSrc = path.join(bundlePath, 'rvc-patches');
        if (fs.existsSync(rvcPatchSrc)) {
          const rvcRoot = path.join(installPath, 'RVC', 'RVC20240604Nvidia');
          if (fs.existsSync(rvcRoot)) {
            cpRecursive(rvcPatchSrc, rvcRoot);
            log('  RVC patches applied (audio, process_ckpt, train, utils)');
          }
        }

        // ── 6. Cleanup unnecessary files ──
        // Keep only what our wrappers use:
        //   MSST: inference/, modules/, models/, configs/, utils/, data/
        //   RVC:  infer/, configs/, assets/
        stage(30, 'Cleaning up downloads...');
        const rmPaths = [
          // MSST — native webUI, training, tests
          'MSST/.git', 'MSST/.github', 'MSST/docs',
          'MSST/gui', 'MSST/scripts', 'MSST/tests',
          'MSST/configs_backup', 'MSST/data_backup',
          'MSST/logs', 'MSST/tmpdir', 'MSST/pretrain',
          // RVC — native webUI, docker, notebooks, launchers
          'RVC/RVC20240604Nvidia/.git', 'RVC/RVC20240604Nvidia/.github',
          'RVC/RVC20240604Nvidia/docs', 'RVC/RVC20240604Nvidia/i18n',
          'RVC/RVC20240604Nvidia/tools', 'RVC/RVC20240604Nvidia/MIT协议暨相关引用库协议',
          'RVC/RVC20240604Nvidia/assets/pretrained',
          // pretrained_v2 is KEPT — needed for RVC model training
          'RVC/RVC20240604Nvidia/assets/uvr5_weights',
          'RVC/RVC20240604Nvidia/logs',
        ];
        const rmFiles = [
          // MSST — training scripts
          'MSST/train.py', 'MSST/train_accelerate.py', 'MSST/train_ddp.py',
          'MSST/valid.py', 'MSST/valid_ddp.py', 'MSST/ensemble.py',
          'MSST/requirements.txt',
          // RVC — webUI, APIs, launchers, docker, notebooks
          'RVC/RVC20240604Nvidia/infer-web.py', 'RVC/RVC20240604Nvidia/gui_v1.py',
          'RVC/RVC20240604Nvidia/api_231006.py', 'RVC/RVC20240604Nvidia/api_240604.py',
          'RVC/RVC20240604Nvidia/run.sh', 'RVC/RVC20240604Nvidia/docker-compose.yml',
          'RVC/RVC20240604Nvidia/Dockerfile', 'RVC/RVC20240604Nvidia/environment_dml.yaml',
          'RVC/RVC20240604Nvidia/poetry.lock', 'RVC/RVC20240604Nvidia/pyproject.toml',
        ];
        for (const d of rmPaths) {
          const p = path.join(installPath, d);
          if (fs.existsSync(p)) { try { fs.rmSync(p, { recursive: true, force: true }); } catch {} }
        }
        for (const f of rmFiles) {
          const p = path.join(installPath, f);
          if (fs.existsSync(p)) { try { fs.unlinkSync(p); } catch {} }
        }
        // Remove .md, .bat, .sh, requirements*, .ipynb from top-level of each
        for (const dir of ['MSST', 'RVC/RVC20240604Nvidia']) {
          const dp = path.join(installPath, dir);
          if (fs.existsSync(dp)) {
            for (const f of fs.readdirSync(dp).filter(n =>
              n.endsWith('.md') || n.endsWith('.bat') || n.endsWith('.sh') ||
              n.startsWith('requirements') || n.endsWith('.ipynb') || n.endsWith('.yaml')
            )) {
              try { fs.unlinkSync(path.join(dp, f)); } catch {}
            }
          }
        }
        const rmCache = (dir) => {
          if (!fs.existsSync(dir)) return;
          for (const entry of fs.readdirSync(dir, { withFileTypes: true })) {
            const full = path.join(dir, entry.name);
            if (entry.isDirectory()) {
              if (entry.name === '__pycache__') { try { fs.rmSync(full, { recursive: true, force: true }); } catch {} }
              else rmCache(full);
            }
          }
        };
        rmCache(path.join(installPath, 'MSST'));
        rmCache(path.join(installPath, 'RVC'));
        for (const d of ['RVC/RVC20240604Nvidia/assets/weights', 'RVC/RVC20240604Nvidia/assets/indices', 'RVC/RVC20240604Nvidia/logs']) {
          await fsP.mkdir(path.join(installPath, d), { recursive: true });
        }
        // Recreate MSST pretrain dirs (deleted by cleanup above, needed for model downloads)
        for (const d of ['MSST/pretrain/vocal_models', 'MSST/pretrain/single_stem_models', 'MSST/pretrain/multi_stem_models', 'MSST/pretrain/VR_Models']) {
          await fsP.mkdir(path.join(installPath, d), { recursive: true });
        }
        log('  Cleanup done');
      }

      // ── 7. Copy requirements + icon ──
      stage(34, 'Copying requirements...');
      for (const f of fs.readdirSync(bundlePath).filter(n => n.startsWith('requirements'))) {
        await fsP.copyFile(path.join(bundlePath, f), path.join(installPath, f));
      }
      if (fs.existsSync(path.join(bundlePath, 'icon.png'))) {
        await fsP.copyFile(path.join(bundlePath, 'icon.png'), path.join(installPath, 'icon.png'));
      }
      const wheelsSrc = path.join(bundlePath, 'wheels');
      if (fs.existsSync(wheelsSrc)) {
        await copyDir(wheelsSrc, path.join(installPath, 'wheels'));
        log('  Pre-built wheels copied');
      }
      const constraintsFile = path.join(installPath, 'pip-constraints.txt');
      fs.writeFileSync(constraintsFile, 'numpy>=1.23.0,<2.0\n', 'utf-8');

      // ── 8. Copy embedded Python ──
      stage(37, 'Setting up Python...');
      const pythonSrc = path.join(bundlePath, 'python');
      if (!fs.existsSync(pythonSrc)) { fail('Bundle missing: python/ directory'); return; }
      const pythonDir = path.join(installPath, 'python');
      await copyDir(pythonSrc, pythonDir);

      // Find a working Python executable
      let pythonExe = path.join(pythonDir, 'python.exe');
      if (!fs.existsSync(pythonExe)) {
        const pythonw = path.join(pythonDir, 'pythonw.exe');
        if (fs.existsSync(pythonw)) {
          pythonExe = pythonw;
        } else {
          fail(`Python executable not found after copy! Checked: ${pythonDir}`);
          return;
        }
      }
      log(`  Python at: ${pythonExe}`);

      // ── 9. Bootstrap pip ──
      stage(40, 'Installing pip...');
      const getPip = path.join(bundlePath, 'get-pip.py');
      if (!fs.existsSync(getPip)) { fail('Bundle missing: get-pip.py'); return; }

      log(`  Running: python get-pip.py`);
      const pipProc0 = trackedSpawn(pythonExe, [getPip, '--no-warn-script-location']);
      pipProc0.stdout.on('data', d => log('  ' + d.toString().trim()));
      pipProc0.stderr.on('data', d => log('  [pip] ' + d.toString().trim()));
      const pipBootCode = await new Promise(resolve => pipProc0.on('close', resolve));
      log(`  get-pip exit code: ${pipBootCode}`);
      if (pipBootCode !== 0) {
        fail(`pip bootstrap failed (exit code ${pipBootCode}). Check Python embed integrity.`);
        return;
      }

      // Verify pip works
      const sitePackages = path.join(pythonDir, 'Lib', 'site-packages');
      fs.mkdirSync(sitePackages, { recursive: true });
      try {
        const verPip = trackedSpawn(pythonExe, ['-m', 'pip', '--version']);
        let pipVer = '';
        verPip.stdout.on('data', d => { pipVer += d.toString(); });
        const vpCode = await new Promise(resolve => verPip.on('close', resolve));
        if (vpCode !== 0) { fail('pip installed but cannot run. Check python310._pth includes "import site"'); return; }
        log(`  ${pipVer.trim()}`);
      } catch (e) { fail(`pip verification failed: ${e.message}`); return; }

      // ── 10. Install build dependencies (numpy + Cython) ──
      // pesq has C extensions that need both numpy headers and Cython at build time
      stage(46, 'Installing build dependencies...');
      const pipMirrorArgs = [];
      if (mirrors.pypi) {
        try {
          pipMirrorArgs.push('--index-url', mirrors.pypi, '--trusted-host', new URL(mirrors.pypi).hostname);
        } catch { pipMirrorArgs.push('--index-url', mirrors.pypi); }
      }
      const bdArgs = ['-m', 'pip', 'install', '--no-warn-script-location', '--no-cache-dir',
        'numpy>=1.23.0,<2.0', 'Cython', 'setuptools<71', ...pipMirrorArgs];
      log('  pip install numpy Cython setuptools<71');
      const bdProc = trackedSpawn(pythonExe, bdArgs, { cwd: installPath });
      bdProc.stdout.on('data', d => log('  ' + d.toString().trim()));
      bdProc.stderr.on('data', d => log('  [pip] ' + d.toString().trim()));
      const bdCode = await new Promise(resolve => bdProc.on('close', resolve));
      log(`  Build deps exit code: ${bdCode}`);
      if (bdCode !== 0) { fail(`Build dependencies failed to install (pip exit code ${bdCode})`); return; }

      // ── 11. Install PyTorch ──
      // Install torch BEFORE requirements.txt — avoids double-download (torch as transitive dep)
      // and ensures the CUDA/CPU wheel is present when other packages resolve torch>=2.0.0
      stage(52, `Installing PyTorch (${gpu})...`);
      const torchReq = gpu === 'cu128' ? 'requirements-torch-cu128.txt'
        : gpu === 'cpu' ? 'requirements-torch-cpu.txt' : 'requirements-torch-cu121.txt';
      const torchIndex = gpu === 'cpu' ? 'https://download.pytorch.org/whl/cpu'
        : gpu === 'cu128' ? 'https://download.pytorch.org/whl/cu128' : 'https://download.pytorch.org/whl/cu121';
      const actualIndex = mirrors.pytorch ? `${mirrors.pytorch}/${gpu === 'cpu' ? 'cpu' : gpu}` : torchIndex;
      // --index-url = torch wheel source; --extra-index-url = PyPI for sub-deps (sympy etc.)
      const torchArgs = ['-m', 'pip', 'install', '--no-warn-script-location', '--no-cache-dir',
        '-r', path.join(installPath, torchReq),
        '--index-url', actualIndex];
      if (mirrors.pypi) {
        try {
          torchArgs.push('--extra-index-url', mirrors.pypi, '--trusted-host', new URL(mirrors.pypi).hostname);
        } catch { torchArgs.push('--extra-index-url', mirrors.pypi); }
      } else {
        torchArgs.push('--extra-index-url', 'https://pypi.org/simple');
      }

      torchArgs.push('--constraint', constraintsFile);
      const torchProc = trackedSpawn(pythonExe, torchArgs, { cwd: installPath });
      torchProc.stdout.on('data', d => log('  ' + d.toString().trim()));
      torchProc.stderr.on('data', d => log('  [pip] ' + d.toString().trim()));
      const torchCode = await new Promise(resolve => torchProc.on('close', resolve));
      log(`  PyTorch exit code: ${torchCode}`);
      if (torchCode !== 0) { fail(`PyTorch failed to install (pip exit code ${torchCode})`); return; }
      log('  PyTorch installed');

      // ── 12. Install Python dependencies ──
      // torch already present → pip skips it; numpy+Cython available → pesq can build
      stage(68, 'Installing Python dependencies...');
      const wheelsDir = path.join(installPath, 'wheels');
      const pipArgs = ['-m', 'pip', 'install', '--no-warn-script-location', '--no-cache-dir',
        '--no-build-isolation', '-r', path.join(installPath, 'requirements.txt'),
        '--constraint', constraintsFile, ...pipMirrorArgs];
      if (fs.existsSync(wheelsDir)) pipArgs.push('--find-links', wheelsDir);

      log(`  pip install -r requirements.txt`);
      const pipProc = trackedSpawn(pythonExe, pipArgs, { cwd: installPath });
      pipProc.stdout.on('data', d => log('  ' + d.toString().trim()));
      pipProc.stderr.on('data', d => log('  [pip] ' + d.toString().trim()));
      const depCode = await new Promise(resolve => pipProc.on('close', resolve));
      log(`  Dependencies exit code: ${depCode}`);
      if (depCode !== 0) { fail(`Dependencies failed to install (pip exit code ${depCode})`); return; }

      // ── 12b. Install fairseq (--no-deps, from bundled wheel for reliability) ──
      stage(75, 'Installing fairseq...');
      const fsWheel = path.join(bundlePath, 'wheels', 'fairseq-0.12.2-cp310-cp310-win_amd64.whl');
      const fsArgs = fs.existsSync(fsWheel)
        ? ['-m', 'pip', 'install', '--no-warn-script-location', '--no-deps', fsWheel]
        : ['-m', 'pip', 'install', '--no-warn-script-location', '--no-deps', 'fairseq>=0.12.2', ...pipMirrorArgs];
      log(`  pip install fairseq --no-deps ${fs.existsSync(fsWheel) ? '(bundled wheel)' : '(network)'}`);
      const fsProc = trackedSpawn(pythonExe, fsArgs, { cwd: installPath });
      fsProc.stdout.on('data', d => log('  ' + d.toString().trim()));
      fsProc.stderr.on('data', d => log('  [pip] ' + d.toString().trim()));
      const fsCode = await new Promise(resolve => fsProc.on('close', resolve));
      if (fsCode !== 0) {
        log('  [!] WARNING: fairseq install failed — RVC voice conversion may not work');
      } else {
        log('  fairseq installed');
      }

      // ── 13. Patch librosa for numpy >=1.24 compatibility ──
      // librosa 0.9.2 uses np.float/np.int/etc. which were removed in numpy 1.24
      stage(80, 'Patching libraries...');
      try {
        const librosaUtils = path.join(sitePackages, 'librosa', 'util', 'utils.py');
        if (fs.existsSync(librosaUtils)) {
          let lrContent = fs.readFileSync(librosaUtils, 'utf-8');
          const lrPatched = lrContent
            .replace(/\bnp\.float\b/g, 'float')
            .replace(/\bnp\.int\b/g, 'int')
            .replace(/\bnp\.complex\b/g, 'complex')
            .replace(/\bnp\.object\b/g, 'object')
            .replace(/\bnp\.bool\b/g, 'bool');
          if (lrPatched !== lrContent) {
            fs.writeFileSync(librosaUtils, lrPatched, 'utf-8');
            log('  librosa numpy compatibility patch applied');
          } else {
            log('  librosa: no deprecated numpy aliases found');
          }
        }
      } catch (e) { log(`  [!] librosa patch skipped: ${e.message}`); }

      // ── 13b. Install tensorboard stub (sitecustomize.py) ──
      // RVC training imports torch.utils.tensorboard but the real package is too heavy.
      // Our backend ships a no-op stub in app/backend/stubs/sitecustomize.py.
      try {
        const stubSrc = path.join(installPath, 'app', 'backend', 'stubs', 'sitecustomize.py');
        const stubDst = path.join(sitePackages, 'sitecustomize.py');
        if (fs.existsSync(stubSrc)) {
          fs.copyFileSync(stubSrc, stubDst);
          log('  tensorboard stub installed (sitecustomize.py)');
        }
      } catch (e) { log(`  [!] tensorboard stub skipped: ${e.message}`); }

      // ── 14. Save mirror config ──
      stage(88, 'Saving configuration...');
      const configDir = path.join(installPath, 'config');
      fs.mkdirSync(configDir, { recursive: true });
      fs.writeFileSync(path.join(configDir, 'mirrors.json'), JSON.stringify(mirrors, null, 2), 'utf-8');

      // ── 15. Verify torch import ──
      stage(92, 'Verifying...');
      try {
        const verProc = trackedSpawn(pythonExe, ['-c', 'import torch; print(f"PyTorch {torch.__version__} CUDA: {torch.cuda.is_available()}")']);
        let verOutput = '';
        verProc.stdout.on('data', d => { const s = d.toString().trim(); verOutput += s; log('  ' + s); });
        verProc.stderr.on('data', d => log('  [verify] ' + d.toString().trim()));
        const verCode = await new Promise(resolve => verProc.on('close', resolve));
        if (verCode !== 0) {
          log('  [!] WARNING: PyTorch verification failed — torch may not import correctly');
          log('  [!] The app may still work if torch loads from a different path');
        }
      } catch (e) { log(`  [!] Verification error: ${e.message}`); }

      // ── 16. Desktop shortcut ──
      if (config.createShortcut) {
        stage(96, 'Creating desktop shortcut...');
        try {
          const targetExe = path.join(installPath, 'Utai.exe');
          const psCmd = `$desktop = [Environment]::GetFolderPath('Desktop'); $ws = New-Object -ComObject WScript.Shell; $sc = $ws.CreateShortcut((Join-Path $desktop 'Utai.lnk')); $sc.TargetPath = '${targetExe.replace(/'/g, "''")}'; $sc.WorkingDirectory = '${installPath.replace(/'/g, "''")}'; $sc.IconLocation = '${targetExe.replace(/'/g, "''")}'; $sc.Save()`;
          execSync(`C:\\Windows\\System32\\WindowsPowerShell\\v1.0\\powershell.exe -NoProfile -Command "${psCmd}"`, { timeout: 10000 });
          log('  Desktop shortcut created');
        } catch (e) { log(`  [!] Could not create desktop shortcut: ${e.message}`); }
      }

      // ── 17. Save install record ──
      const records = readInstallRecords();
      const updatedRecords = records.filter(r => r.installPath !== installPath);
      updatedRecords.push({ installPath, components, gpu, installedAt: new Date().toISOString() });
      writeInstallRecords(updatedRecords);

      stage(100, 'Installation complete!');
      mainWindow.webContents.send('install-done', true);
    } catch (err) {
      fail(err.message);
    }
  })();

  return true;
});
