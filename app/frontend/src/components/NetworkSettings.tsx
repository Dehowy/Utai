/** Network mirror settings panel — matches installer behavior. */

import React, { useState, useEffect } from 'react';
import { useTranslation } from 'react-i18next';
import { showToast } from './Toast';
import { useAppStore } from '../store/appStore';

interface Mirrors { pypi: string; github: string; huggingface: string; pytorch: string; }

const PRESETS: Record<keyof Mirrors, { label: string; labelZh: string; value: string }[]> = {
  pypi: [
    { label: 'Default', labelZh: '默认（直连）', value: '' },
    { label: 'Tsinghua', labelZh: '清华源', value: 'https://pypi.tuna.tsinghua.edu.cn/simple' },
    { label: 'Aliyun', labelZh: '阿里云', value: 'https://mirrors.aliyun.com/pypi/simple' },
    { label: 'USTC', labelZh: '中科大', value: 'https://pypi.mirrors.ustc.edu.cn/simple' },
  ],
  huggingface: [
    { label: 'Default', labelZh: '默认（直连）', value: '' },
    { label: 'hf-mirror', labelZh: 'hf-mirror', value: 'https://hf-mirror.com' },
  ],
  github: [
    { label: 'Default', labelZh: '默认（直连）', value: '' },
    { label: 'ghproxy', labelZh: 'ghproxy', value: 'https://ghproxy.com' },
    { label: 'gh-proxy', labelZh: 'gh-proxy', value: 'https://gh-proxy.com' },
  ],
  pytorch: [
    { label: 'Default', labelZh: '默认（直连）', value: '' },
    { label: 'SJTU', labelZh: '上海交大', value: 'https://mirror.sjtu.edu.cn/pytorch-wheels' },
  ],
};

const CUSTOM_KEY = '__custom__';

function DefaultPagePicker({ presetStyle }: { presetStyle: (active: boolean) => React.CSSProperties }) {
  const { t } = useTranslation();
  const dp = useAppStore(s => s.defaultPage);
  const setDp = useAppStore(s => s.setDefaultPage);
  const modules = useAppStore(s => s.modules);
  const options = [
    ...(modules?.synth !== false ? [{ value: '/synth', label: t('settings.pageSynth', 'Synthesizer') }] : []),
    ...(modules?.audio_tools !== false ? [
      { value: '/', label: t('settings.pageAudio', 'Audio Tools') },
      { value: '/train', label: t('settings.pageTrain', 'Training') },
    ] : []),
  ];
  return (
    <div style={{ marginTop: 12, paddingTop: 8, borderTop: '1px solid var(--border-dim)' }}>
      <span style={{ fontSize: 11, color: 'var(--accent-cyan)', fontFamily: 'var(--font-pixel)' }}>{t('settings.defaultPage', 'Default Page')}</span>
      <div style={{ display: 'flex', gap: 4, marginTop: 3 }}>
        {options.map(o => (
          <button key={o.value} onClick={() => setDp(o.value)} style={presetStyle(dp === o.value)}>
            {o.label}
          </button>
        ))}
      </div>
    </div>
  );
}

export default function NetworkSettings({ open, onClose }: { open: boolean; onClose: () => void }) {
  const [mirrors, setMirrors] = useState<Mirrors>({ pypi: '', github: '', huggingface: '', pytorch: '' });
  const [customValues, setCustomValues] = useState<Mirrors>({ pypi: '', github: '', huggingface: '', pytorch: '' });
  const [selected, setSelected] = useState<Record<keyof Mirrors, string>>({ pypi: '', github: '', huggingface: '', pytorch: '' });
  const [testResult, setTestResult] = useState<{ status: string; details: string } | null>(null);
  const [testing, setTesting] = useState(false);
  const [tested, setTested] = useState(false);
  const [networkDirty, setNetworkDirty] = useState(false);
  const [origMirrors, setOrigMirrors] = useState<Mirrors>({ pypi: '', github: '', huggingface: '', pytorch: '' });

  useEffect(() => {
    if (!open) return;
    setTested(false);
    setTestResult(null);
    setNetworkDirty(false);
    fetch('/api/system/mirrors').then(r => r.json()).then((m: Mirrors) => {
      setMirrors(m);
      setOrigMirrors({ ...m });
      for (const key of Object.keys(m) as (keyof Mirrors)[]) {
        const preset = PRESETS[key].find(p => p.value === m[key]);
        if (preset) { setSelected(s => ({ ...s, [key]: m[key] })); }
        else if (m[key]) { setSelected(s => ({ ...s, [key]: CUSTOM_KEY })); setCustomValues(c => ({ ...c, [key]: m[key] })); }
        else { setSelected(s => ({ ...s, [key]: '' })); }
      }
    }).catch(() => {});
  }, [open]);

  const getMirrorValue = (key: keyof Mirrors) => selected[key] === CUSTOM_KEY ? customValues[key] : selected[key];

  const handleSelect = (key: keyof Mirrors, value: string) => {
    setSelected(s => ({ ...s, [key]: value }));
    const newVal = value === CUSTOM_KEY ? customValues[key] : value;
    setMirrors(m => ({ ...m, [key]: newVal }));
    setNetworkDirty(true);
    setTested(false);
    setTestResult(null);
  };

  const handleCustomChange = (key: keyof Mirrors, value: string) => {
    setCustomValues(c => ({ ...c, [key]: value }));
    setMirrors(m => ({ ...m, [key]: value }));
    setNetworkDirty(true);
    setTested(false);
    setTestResult(null);
  };

  const handleTest = async () => {
    setTesting(true);
    setTestResult(null);
    const tests: { name: string; url: string }[] = [
      { name: 'PyPI', url: getMirrorValue('pypi') || 'https://pypi.org' },
      { name: 'HuggingFace', url: getMirrorValue('huggingface') || 'https://huggingface.co' },
      { name: 'GitHub', url: getMirrorValue('github') || 'https://github.com' },
      { name: 'PyTorch', url: getMirrorValue('pytorch') || 'https://download.pytorch.org/whl/torch_stable.html' },
    ];
    const results: string[] = [];
    let allOk = true;
    for (const { name, url } of tests) {
      try {
        const r = await (await fetch('/api/system/mirrors/test?url=' + encodeURIComponent(url))).json();
        results.push(`${name}: ${r.reachable ? 'OK' : 'FAILED'}`);
        if (!r.reachable) allOk = false;
      } catch {
        results.push(`${name}: FAILED`);
        allOk = false;
      }
    }
    setTestResult({ status: allOk ? 'ok' : 'error', details: results.join('  |  ') });
    setTested(allOk);
    setTesting(false);
  };

  const handleSave = async () => {
    try {
      const res = await fetch('/api/system/mirrors', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(mirrors),
      });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      showToast('Settings saved', 'success');
      onClose();
    } catch (e: any) { showToast(`Failed: ${e.message}`, 'error'); }
  };

  if (!open) return null;

  const presetStyle = (active: boolean) => ({
    fontSize: 11, padding: '3px 10px', cursor: 'pointer',
    fontFamily: 'var(--font-pixel)',
    background: active ? 'rgba(92,224,216,0.1)' : '#0d0d1a',
    border: `1px solid ${active ? 'var(--accent-cyan)' : '#2a2a4a'}`,
    color: active ? 'var(--accent-cyan)' : '#7a7aa0',
    transition: 'all 0.15s',
  });

  const renderField = (key: keyof Mirrors, label: string) => (
    <div style={{ marginBottom: 10 }}>
      <span style={{ fontSize: 11, color: 'var(--accent-cyan)', fontFamily: 'var(--font-pixel)' }}>{label}</span>
      <div style={{ display: 'flex', gap: 4, flexWrap: 'wrap', margin: '3px 0' }}>
        {PRESETS[key].map(p => (
          <button key={p.value} style={presetStyle(selected[key] === p.value)} onClick={() => handleSelect(key, p.value)}>
            {p.labelZh}
          </button>
        ))}
        <button style={presetStyle(selected[key] === CUSTOM_KEY)} onClick={() => handleSelect(key, CUSTOM_KEY)}>
          自定义
        </button>
      </div>
      {selected[key] === CUSTOM_KEY && (
        <input value={customValues[key]} onChange={e => handleCustomChange(key, e.target.value)}
          placeholder="https://..."
          style={{ width: '100%', fontSize: 11, padding: '3px 6px', background: 'var(--bg-deep)', border: '1px solid var(--border-dim)', color: 'var(--text-primary)', fontFamily: 'var(--font-mono)', outline: 'none' }}
        />
      )}
    </div>
  );

  return (
    <>
      <div className="utai-modal-backdrop" onClick={onClose} style={{ zIndex: 1100 }} />
      <div style={{
        position: 'fixed', top: '50%', left: '50%', transform: 'translate(-50%,-50%)',
        zIndex: 1101, background: 'var(--bg-panel)', border: '1px solid var(--accent-cyan)',
        padding: 16, minWidth: 420, maxWidth: 520,
      }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 12 }}>
          <span style={{ fontSize: 13, color: 'var(--accent-cyan)', fontFamily: 'var(--font-pixel)' }}>Settings</span>
          <button onClick={onClose} style={{ background: 'none', border: 'none', color: 'var(--text-dim)', fontSize: 16, cursor: 'pointer' }}>x</button>
        </div>
        {renderField('pypi', 'PyPI')}
        {renderField('huggingface', 'HuggingFace')}
        {renderField('github', 'GitHub')}
        {renderField('pytorch', 'PyTorch')}

        {/* Default page preference */}
        <DefaultPagePicker presetStyle={presetStyle} />

        {testResult && (
          <div style={{ fontSize: 11, padding: '4px 0', color: testResult.status === 'ok' ? '#5dba5d' : '#ff6b9d', fontFamily: 'var(--font-mono)' }}>
            {testResult.details}
          </div>
        )}

        <div style={{ display: 'flex', justifyContent: 'flex-end', gap: 6, marginTop: 8 }}>
          <button onClick={() => { setMirrors({ pypi: '', github: '', huggingface: '', pytorch: '' }); setSelected({ pypi: '', github: '', huggingface: '', pytorch: '' }); setCustomValues({ pypi: '', github: '', huggingface: '', pytorch: '' }); setNetworkDirty(true); setTested(false); setTestResult(null); }}
            style={{ fontSize: 10, padding: '4px 14px', background: 'none', border: '1px solid var(--border-dim)', color: 'var(--text-dim)', cursor: 'pointer', fontFamily: 'var(--font-pixel)' }}>Reset</button>
          <button onClick={handleTest} disabled={testing}
            style={{ fontSize: 10, padding: '4px 14px', background: 'none', border: '2px solid #e8b84b', color: '#e8b84b', cursor: 'pointer', fontFamily: 'var(--font-pixel)', fontWeight: 'bold' }}>
            {testing ? 'Testing...' : 'Test Connection'}
          </button>
          <button onClick={handleSave} disabled={networkDirty && !tested}
            style={{ fontSize: 10, padding: '4px 14px', background: 'none', border: `1px solid ${(!networkDirty || tested) ? 'var(--accent-cyan)' : 'var(--border-dim)'}`, color: (!networkDirty || tested) ? 'var(--accent-cyan)' : 'var(--text-dim)', cursor: (!networkDirty || tested) ? 'pointer' : 'not-allowed', fontFamily: 'var(--font-pixel)', opacity: (!networkDirty || tested) ? 1 : 0.4 }}>Save</button>
        </div>
      </div>
    </>
  );
}
