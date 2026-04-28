// ── i18n ──
const I18N = {
  zh: {
    'welcome.subtitle': 'AI 音频工作流 + 合成器',
    'welcome.start': '开始安装',
    'network.title': '网络设置',
    'network.desc': '配置下载源。中国大陆用户建议选择镜像源。',
    'network.test': '测试连接',
    'path.title': '安装路径',
    'path.desc': '选择安装位置。此文件夹安装后即为可移植的程序目录。',
    'path.browse': '浏览',
    'path.shortcut': '创建桌面快捷方式',
    'comp.title': '组件选择',
    'comp.desc': '选择要安装的功能。至少选择一项。',
    'comp.synth': '合成器',
    'comp.synthDesc': '可以把 RVC 模型变成虚拟歌姬的音乐合成器！',
    'comp.audio': '音频工具',
    'comp.audioDesc': '从歌曲里提取人声、去除伴奏、变换音色，还可以训练属于你自己的 AI 声音模型。',
    'comp.warn': '请至少选择一项组件。',
    'gpu.title': '显卡选择',
    'gpu.desc': '选择您的显卡以启用 AI 加速。',
    'gpu.cpu': '仅 CPU（无显卡加速）',
    'confirm.title': '确认安装',
    'confirm.install': '安装',
    'installing.title': '正在安装...',
    'done.success': '安装完成！',
    'done.launch': '启动 Utai',
    'done.close': '关闭',
    'btn.back': '上一步',
    'btn.next': '下一步',
    'mirror.default': '默认（直连）',
    'mirror.tsinghua': '清华源',
    'mirror.sjtu': '上海交大',
    'mirror.aliyun': '阿里云',
    'welcome.uninstall': '卸载现有安装',
    'done.retry': '重试',
    'network.skip_warn': '尚未测试网络连接，确定继续吗？',
    'done.saveLog': '保存日志',
    'done.logSaved': '日志已保存到：',
    'done.failed': '安装失败',
    'done.failedDesc': '请查看下方日志了解详情。',
  },
  ja: {
    'welcome.subtitle': 'AI オーディオワークフロー + シンセサイザー',
    'welcome.start': 'インストール開始',
    'network.title': 'ネットワーク設定',
    'network.desc': 'ダウンロードソースを設定します。',
    'network.test': '接続テスト',
    'path.title': 'インストール先',
    'path.desc': 'インストール先を選択してください。このフォルダがポータブルプログラムになります。',
    'path.browse': '参照',
    'path.shortcut': 'デスクトップショートカットを作成',
    'comp.title': 'コンポーネント',
    'comp.desc': 'インストールする機能を選択してください。',
    'comp.synth': 'シンセサイザー',
    'comp.synthDesc': 'RVCモデルをバーチャル歌姫に変える音楽シンセサイザー！',
    'comp.audio': 'オーディオツール',
    'comp.audioDesc': '楽曲からボーカル抽出・伴奏除去・音色変換。自分だけのAI音声モデルもトレーニングできます。',
    'comp.warn': '少なくとも1つ選択してください。',
    'gpu.title': 'GPU選択',
    'gpu.desc': 'AI高速化のためGPUを選択してください。',
    'gpu.cpu': 'CPUのみ（GPU加速なし）',
    'confirm.title': 'インストール確認',
    'confirm.install': 'インストール',
    'installing.title': 'インストール中...',
    'done.success': 'インストール完了！',
    'done.launch': 'Utaiを起動',
    'done.close': '閉じる',
    'btn.back': '戻る',
    'btn.next': '次へ',
    'mirror.default': 'デフォルト',
    'mirror.tsinghua': '清華ミラー',
    'mirror.sjtu': '上海交通大学',
    'mirror.aliyun': 'Aliyun',
    'welcome.uninstall': '既存のインストールを削除',
    'done.retry': 'リトライ',
    'network.skip_warn': 'ネットワークテストがまだ完了していません。続行しますか？',
    'done.saveLog': 'ログを保存',
    'done.logSaved': 'ログの保存先：',
    'done.failed': 'インストール失敗',
    'done.failedDesc': '詳細は下のログをご確認ください。',
  },
  en: {},
};

function escapeHtml(s) {
  return s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
}

let currentLang = 'en';

function setLang(lang, btn) {
  currentLang = lang;
  document.querySelectorAll('.lang-btn').forEach(b => b.classList.remove('active'));
  btn.classList.add('active');
  document.querySelectorAll('[data-i18n]').forEach(el => {
    const key = el.getAttribute('data-i18n');
    if (!el.getAttribute('data-i18n-en')) el.setAttribute('data-i18n-en', el.textContent);
    const text = (I18N[lang] || {})[key];
    let display = text || el.getAttribute('data-i18n-en') || el.textContent;
    if (el.dataset.uninstallPath) display += ' (' + el.dataset.uninstallPath + ')';
    el.textContent = display;
  });
}

// ── State ──
let mirrors = { pypi: '', github: '', huggingface: '', pytorch: '' };
let gpuInfo = { detected: false, name: null, recommended: 'cpu' };
let networkTested = false;
let installLogText = '';

// ── Init ──
installer.getDefaultPath().then(p => {
  document.getElementById('install-path').value = p;
});

// Check for existing installations → show uninstall option(s)
installer.checkExistingInstall().then(info => {
  const installs = info.installations || [];
  if (installs.length === 0) return;
  const section = document.getElementById('uninstall-section');
  if (!section) return;

  if (installs.length === 1) {
    const displayPath = installs[0].installPath.replace(/\\/g, '/');
    const btn = document.createElement('button');
    btn.className = 'btn-secondary';
    btn.style.cssText = 'margin-top:8px; color:#ff6b9d; border-color:#ff6b9d;';
    btn.dataset.uninstallPath = displayPath;
    btn.setAttribute('data-i18n', 'welcome.uninstall');
    const label = (I18N[currentLang] || {})['welcome.uninstall'] || 'Uninstall existing';
    btn.textContent = label + ' (' + displayPath + ')';
    btn.onclick = () => doUninstall(installs[0].installPath);
    section.appendChild(btn);
  } else {
    const container = document.createElement('div');
    container.className = 'uninstall-multi';
    const select = document.createElement('select');
    select.id = 'uninstall-select';
    for (const inst of installs) {
      const opt = document.createElement('option');
      opt.value = inst.installPath;
      opt.textContent = inst.installPath.replace(/\\/g, '/');
      select.appendChild(opt);
    }
    const btn = document.createElement('button');
    btn.className = 'btn-secondary';
    btn.style.cssText = 'color:#ff6b9d; border-color:#ff6b9d; white-space:nowrap;';
    btn.setAttribute('data-i18n', 'welcome.uninstall');
    btn.textContent = (I18N[currentLang] || {})['welcome.uninstall'] || 'Uninstall existing';
    btn.onclick = () => { if (select.value) doUninstall(select.value); };
    container.appendChild(select);
    container.appendChild(btn);
    section.appendChild(container);
  }
});

async function doUninstall(installPath) {
  goTo('installing');
  document.getElementById('install-status').textContent = 'Uninstalling...';
  installer.onInstallLog(msg => {
    const logBox = document.getElementById('install-log');
    const div = document.createElement('div');
    div.className = 'log-line' + (msg.startsWith('[!]') ? ' log-error' : '');
    div.textContent = msg;
    logBox.appendChild(div);
    logBox.scrollTop = logBox.scrollHeight;
    const statusEl = document.getElementById('install-status');
    const progressEl = document.getElementById('install-progress');
    if (msg.includes('[STAGE]')) statusEl.textContent = msg.split('[STAGE]')[1].trim();
    if (msg.includes('[PROGRESS]')) {
      const pct = parseInt(msg.split('[PROGRESS]')[1]);
      if (!isNaN(pct)) progressEl.style.width = pct + '%';
    }
  });
  const result = await installer.uninstall(installPath);
  goTo('done');
  if (result.success) {
    document.getElementById('done-success').style.display = 'flex';
    document.getElementById('done-failed').style.display = 'none';
    document.querySelector('#done-success .subtitle').textContent = 'Uninstall complete!';
    document.getElementById('btn-launch').style.display = 'none';
  } else {
    document.getElementById('done-success').style.display = 'none';
    document.getElementById('done-failed').style.display = 'flex';
    document.getElementById('fail-log').innerHTML = document.getElementById('install-log').innerHTML;
  }
}

// ── Navigation ──
function goTo(stepId, skipCheck) {
  if (stepId === 'path' && !skipCheck && !networkTested) {
    const msg = (I18N[currentLang] || {})['network.skip_warn'] || 'Network connection not tested. Continue anyway?';
    if (!confirm(msg)) return;
  }
  document.querySelectorAll('.step').forEach(el => el.classList.add('hidden'));
  document.getElementById('step-' + stepId).classList.remove('hidden');
  if (stepId === 'gpu') detectGpu();
  if (stepId === 'confirm') buildSummary();
  if (stepId === 'path') checkDiskSpace();
  if (stepId === 'network') updateNetworkNext();
}

// ── Mirror presets ──
function setMirror(key, value, btn) {
  mirrors[key] = value;
  document.getElementById('mirror-' + key).value = value;
  const row = btn.parentElement;
  row.querySelectorAll('.preset').forEach(b => b.classList.remove('active'));
  btn.classList.add('active');
  networkTested = false;
  updateNetworkNext();
}

function updateNetworkNext() {
  networkTested = false;
  document.getElementById('network-test-result').textContent = '';
}

// ── Network test ──
async function testNetwork() {
  const el = document.getElementById('network-test-result');
  const btn = document.getElementById('btn-test');
  btn.disabled = true;
  el.textContent = 'Testing...';
  el.className = 'info-text';

  const pypi = mirrors.pypi || 'https://pypi.org';
  const hf = mirrors.huggingface || 'https://huggingface.co';
  const gh = mirrors.github || 'https://github.com';
  const pt = mirrors.pytorch || 'https://download.pytorch.org/whl/torch_stable.html';
  const results = [];

  for (const [name, url] of [['PyPI', pypi], ['HuggingFace', hf], ['GitHub', gh], ['PyTorch', pt]]) {
    const r = await installer.testUrl(url);
    results.push({ name, ok: r.reachable, latency: r.latency });
  }

  const allOk = results.every(r => r.ok);
  el.innerHTML = results.map(r =>
    `<span style="color:${r.ok ? '#5dba5d' : '#ff6b9d'}">${r.name}: ${r.ok ? 'OK (' + r.latency + 'ms)' : 'FAILED'}</span>`
  ).join('&nbsp;&nbsp;|&nbsp;&nbsp;');
  el.className = 'info-text';

  networkTested = allOk;
  btn.disabled = false;
}

// ── Path ──
async function browsePath() {
  const cur = document.getElementById('install-path').value;
  const parent = cur.replace(/\\Utai$/, '');
  const chosen = await installer.browseFolder(parent);
  if (chosen) {
    const p = chosen.endsWith('\\Utai') || chosen.endsWith('/Utai') ? chosen : chosen + '\\Utai';
    document.getElementById('install-path').value = p;
    checkDiskSpace();
  }
}

async function checkDiskSpace() {
  const path = document.getElementById('install-path').value;
  const free = await installer.checkDiskSpace(path);
  const el = document.getElementById('disk-space');
  if (free < 0) {
    el.textContent = 'Could not check disk space';
    el.className = 'info-text warn';
  } else {
    const freeGB = (free / 1073741824).toFixed(1);
    const enough = free > 5368709120;
    el.textContent = `Free space: ${freeGB} GB` + (enough ? '' : ' (need at least 5 GB)');
    el.className = 'info-text ' + (enough ? 'ok' : 'error');
  }
}

function checkPathAndGo() {
  const path = document.getElementById('install-path').value;
  if (!path || path.length < 3) { alert('Please select a valid installation path'); return; }
  goTo('components');
}

// ── Components ──
function validateComponents() {
  const synth = document.getElementById('comp-synth').checked;
  const audio = document.getElementById('comp-audio').checked;
  const valid = synth || audio;
  document.getElementById('comp-warn').style.display = valid ? 'none' : 'block';
  document.getElementById('btn-comp-next').disabled = !valid;
  document.getElementById('btn-comp-next').style.opacity = valid ? '1' : '0.3';
}

// ── GPU ──
async function detectGpu() {
  const el = document.getElementById('gpu-detect');
  gpuInfo = await installer.detectGpu();
  if (gpuInfo.detected) {
    el.textContent = `Detected: ${gpuInfo.name}`;
    el.className = 'info-text ok';
  } else {
    el.textContent = 'No NVIDIA GPU detected';
    el.className = 'info-text warn';
  }
  const radio = document.querySelector(`input[name="gpu"][value="${gpuInfo.recommended}"]`);
  if (radio) radio.checked = true;
}

// ── Summary ──
function buildSummary() {
  const path = document.getElementById('install-path').value;
  const synth = document.getElementById('comp-synth').checked;
  const audio = document.getElementById('comp-audio').checked;
  const gpu = document.querySelector('input[name="gpu"]:checked')?.value || 'cpu';
  const gpuLabels = { cu121: 'NVIDIA CUDA 12.1 (RTX 20/30/40)', cu128: 'NVIDIA CUDA 12.8 (RTX 50)', cpu: 'CPU Only' };
  const comps = [synth ? 'Synthesizer' : '', audio ? 'Audio Tools' : ''].filter(Boolean).join(', ');

  document.getElementById('confirm-summary').innerHTML = `
    <div><strong>Install Path:</strong> ${escapeHtml(path)}</div>
    <div><strong>Components:</strong> ${escapeHtml(comps)}</div>
    <div><strong>GPU:</strong> ${escapeHtml(gpuLabels[gpu] || gpu)}</div>
    <div><strong>PyPI Mirror:</strong> ${escapeHtml(mirrors.pypi || 'Default')}</div>
    <div><strong>Estimated size:</strong> ~${(synth && audio) ? '4-5' : '3-4'} GB</div>
  `;
}

// ── Install ──
function startInstall() {
  const config = {
    installPath: document.getElementById('install-path').value,
    components: {
      synth: document.getElementById('comp-synth').checked,
      audio: document.getElementById('comp-audio').checked,
    },
    gpu: document.querySelector('input[name="gpu"]:checked')?.value || 'cpu',
    mirrors,
    createShortcut: document.getElementById('create-shortcut')?.checked ?? true,
  };

  goTo('installing');

  installLogText = '';
  installer.onInstallLog(msg => {
    installLogText += msg + '\n';
    const logBox = document.getElementById('install-log');
    const div = document.createElement('div');
    div.className = 'log-line' + (msg.startsWith('[!]') ? ' log-error' : '');
    div.textContent = msg;
    logBox.appendChild(div);
    logBox.scrollTop = logBox.scrollHeight;

    const statusEl = document.getElementById('install-status');
    const progressEl = document.getElementById('install-progress');
    if (msg.includes('[STAGE]')) statusEl.textContent = msg.split('[STAGE]')[1].trim();
    if (msg.includes('[PROGRESS]')) {
      const pct = parseInt(msg.split('[PROGRESS]')[1]);
      if (!isNaN(pct)) progressEl.style.width = pct + '%';
    }
  });

  installer.onInstallDone(success => {
    goTo('done');
    if (success) {
      document.getElementById('done-success').style.display = '';
      document.getElementById('done-failed').style.display = 'none';
    } else {
      document.getElementById('done-success').style.display = 'none';
      document.getElementById('done-failed').style.display = 'flex';
      document.getElementById('done-failed').style.flexDirection = 'column';
      document.getElementById('done-failed').style.height = '100%';
      // Copy log to fail page
      document.getElementById('fail-log').innerHTML = document.getElementById('install-log').innerHTML;
    }
  });

  installer.startInstall(config);
}

function retryInstall() {
  document.getElementById('install-log').innerHTML = '';
  document.getElementById('install-progress').style.width = '0%';
  goTo('confirm', true);
}

function launchAndClose() {
  const installPath = document.getElementById('install-path').value;
  installer.launchUtai(installPath);
  installer.close();
}

async function saveInstallLog() {
  if (!installLogText) { alert('No log to save'); return; }
  const result = await installer.saveLog(installLogText);
  if (result.ok) {
    const msg = (I18N[currentLang] || {})['done.logSaved'] || 'Log saved to:';
    alert(`${msg} ${result.path}`);
  }
}
