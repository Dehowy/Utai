/** Synth page — AI singing voice synthesis with split-pane piano roll editor. */

import { useState, useEffect, useRef, useCallback } from 'react'
import { useTranslation } from 'react-i18next'
import { Link } from 'react-router-dom'
import { LangSwitcher } from '../components/LangSwitcher'
import { ModelManager } from '../components/ModelManager'
import SplitPane from '../components/synth/SplitPane'
import NetworkSettings from '../components/NetworkSettings'
import { useAppStore } from '../store/appStore'
import { getSegmentClipboard, setSegmentClipboard, pasteSegment, isDragging } from '../components/synth/ArrangementView'
import { saveProject } from '../components/synth/SynthProjectMenu'
import { ToastContainer } from '../components/Toast'
import PianoRollPanel from '../components/synth/PianoRollPanel'
import ControlPanel from '../components/synth/ControlPanel'
import SynthProjectMenu from '../components/synth/SynthProjectMenu'
import { useSynthStore } from '../store/synthStore'

function useAutoRenderEffects() {
  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const abortRef = useRef<AbortController | null>(null);
  const trackEffects = useSynthStore(s => s.trackEffects);
  const trackFingerprint = useSynthStore(s =>
    s.project.tracks.map(t => `${t.volume}|${t.pan}`).join(',')
  );

  const doRender = useCallback(async () => {
    if (isDragging()) return;

    abortRef.current?.abort();
    const ctrl = new AbortController();
    abortRef.current = ctrl;

    const state = useSynthStore.getState();
    for (const track of state.project.tracks) {
      const trackIdx = state.project.tracks.indexOf(track);
      if (ctrl.signal.aborted) break;

      if (track.track_type === 'audio') {
        const waveParts = state.project.wave_parts.filter(wp => wp.track_no === trackIdx);
        if (waveParts.length === 0) continue;

        const effects = (state.trackEffects[trackIdx] || [])
          .filter(fx => fx.enabled)
          .map(fx => ({ type: fx.type, params: fx.params, enabled: true }));

        for (const wp of waveParts) {
          if (!wp.buffer_id || ctrl.signal.aborted) continue;
          try {
            const res = await fetch(`/api/audio/${wp.buffer_id}/apply-effects`, {
              method: 'POST',
              headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify({ effects, volume_db: track.volume, pan: track.pan }),
              signal: ctrl.signal,
            });
            if (!res.ok || ctrl.signal.aborted) continue;
            const data = await res.json();
            if (data.buffer_id && !ctrl.signal.aborted) {
              const { getAudioURL } = await import('../api/client');
              useSynthStore.getState().setTrackAudioUrl(trackIdx, getAudioURL(data.buffer_id));
            }
          } catch { /* aborted or failed */ }
        }
      } else if (track.track_type === 'vocal') {
        const rawUrl = state.trackRawAudioUrls[trackIdx];
        if (!rawUrl) continue;

        const effects = (state.trackEffects[trackIdx] || [])
          .filter(fx => fx.enabled)
          .map(fx => ({ type: fx.type, params: fx.params, enabled: true }));
        if (effects.length === 0) {
          useSynthStore.getState().setTrackAudioUrl(trackIdx, rawUrl);
          continue;
        }

        try {
          const rawRes = await fetch(rawUrl, { signal: ctrl.signal });
          if (!rawRes.ok || ctrl.signal.aborted) continue;
          const rawBlob = await rawRes.blob();
          const { applyEffectsUpload } = await import('../api/client');
          const fxBlob = await applyEffectsUpload(rawBlob, effects);
          if (!ctrl.signal.aborted) {
            useSynthStore.getState().setTrackAudioUrl(trackIdx, URL.createObjectURL(fxBlob));
          }
        } catch { /* aborted or failed */ }
      }
    }
  }, []);

  useEffect(() => {
    if (timerRef.current) clearTimeout(timerRef.current);
    timerRef.current = setTimeout(doRender, 600);
    return () => { if (timerRef.current) clearTimeout(timerRef.current); };
  }, [trackEffects, trackFingerprint, doRender]);

  useEffect(() => {
    const onDragEnd = () => {
      if (timerRef.current) clearTimeout(timerRef.current);
      timerRef.current = setTimeout(doRender, 300);
    };
    window.addEventListener('utai-drag-end', onDragEnd);
    return () => window.removeEventListener('utai-drag-end', onDragEnd);
  }, [doRender]);
}

function flushAutosave() {
  const s = useSynthStore.getState();
  return fetch('/api/synth/project/autosave', {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ project: s.project, trackEffects: s.trackEffects, settings: { language: s.language, trackLanguage: s.trackLanguage, rvcModel: s.rvcModel, trackRvcModel: s.trackRvcModel, muteStrategy: s.muteStrategy, trackMuteStrategy: s.trackMuteStrategy, globalF0Offset: s.globalF0Offset } }),
  }).catch(() => {});
}

function useAutoSave() {
  const dirty = useSynthStore(s => s.dirty);
  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const saveGenRef = useRef(0);

  useEffect(() => {
    if (!dirty) return;
    saveGenRef.current++;
    const gen = saveGenRef.current;
    if (timerRef.current) clearTimeout(timerRef.current);
    timerRef.current = setTimeout(async () => {
      if (gen !== saveGenRef.current) return;
      try { await flushAutosave(); } catch { /* silent */ }
    }, 10000);
    return () => { if (timerRef.current) clearTimeout(timerRef.current); };
  }, [dirty]);

  // Listen for force-autosave from Electron main (native fallback quit path)
  useEffect(() => {
    const api = window.electronAPI;
    if (!api?.onForceAutosave) return;
    return api.onForceAutosave(() => { flushAutosave(); }) || undefined;
  }, []);
}

let _autoRecoverDone = false;
function useAutoRecover() {
  useEffect(() => {
    if (_autoRecoverDone) return;
    _autoRecoverDone = true;
    (async () => {
      try {
        const res = await fetch('/api/synth/project/autosave');
        const data = await res.json();
        if (data.available && data.project) {
          const { utaiConfirm } = await import('../components/UtaiAlert');
          const { showToast } = await import('../components/Toast');
          const recover = await utaiConfirm('Recovered unsaved project from last session. Restore?');
          if (recover) {
            const { applyOpenedProject } = await import('../components/synth/SynthProjectMenu');
            applyOpenedProject(data);
            useSynthStore.setState({ dirty: true });
            showToast('Project recovered', 'success');
          }
          await fetch('/api/synth/project/autosave', { method: 'DELETE' });
        }
      } catch { /* no autosave available */ }
    })();
  }, []);
}

export default function SynthPage() {
  const { t } = useTranslation()
  const [modelManagerOpen, setModelManagerOpen] = useState(false)
  const [networkSettingsOpen, setNetworkSettingsOpen] = useState(false)
  const viewMode = useSynthStore(s => s.viewMode)
  const appModules = useAppStore(s => s.modules)
  const hasAudio = appModules !== null && appModules.audio_tools !== false

  useAutoRenderEffects();
  useAutoSave();
  useAutoRecover();

  // Global keyboard shortcuts + block browser defaults on synth page
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      const synthEl = document.querySelector('.utai-synth-page');
      if (!synthEl || (synthEl.closest('[style]') as HTMLElement)?.style.display === 'none') return
      // Don't capture keys when typing in inputs
      const tag = (e.target as HTMLElement)?.tagName
      if (tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT') return

      // Ctrl+Z / Ctrl+Y — undo/redo (global, works in both pianoroll and arrangement)
      if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 'z' && !e.shiftKey) {
        e.preventDefault()
        useSynthStore.getState().undo()
      }
      if ((e.ctrlKey || e.metaKey) && (e.key === 'y' || (e.key.toLowerCase() === 'z' && e.shiftKey))) {
        e.preventDefault()
        useSynthStore.getState().redo()
      }
      if ((e.ctrlKey || e.metaKey) && e.key === 's') {
        e.preventDefault()
        // Render if possible, AND save project
        const renderBtn = document.querySelector('.utai-synth-render-btn') as HTMLButtonElement | null
        if (renderBtn && !renderBtn.disabled) renderBtn.click()
        saveProject()
      }
      if (e.key === ' ') {
        e.preventDefault()
        const playBtn = document.querySelector('.utai-synth-transport__btn') as HTMLButtonElement | null
        if (playBtn) playBtn.click()
      }
      // Ctrl+C — copy selected segment
      if ((e.ctrlKey || e.metaKey) && e.key === 'c') {
        const s = useSynthStore.getState()
        const seg = s.selectedSegment
        if (seg) {
          e.preventDefault()
          const parts = seg.type === 'voice' ? s.project.voice_parts
            : seg.type === 'wave' ? s.project.wave_parts : s.project.instrument_parts
          const part = parts[seg.index]
          if (part) setSegmentClipboard({ type: seg.type, data: JSON.parse(JSON.stringify(part)) })
        }
      }
      // Ctrl+V — paste segment at playhead on selected track
      if ((e.ctrlKey || e.metaKey) && e.key === 'v') {
        const clip = getSegmentClipboard()
        if (clip) {
          e.preventDefault()
          const s = useSynthStore.getState()
          pasteSegment(clip, s.playbackPositionTicks, s.selectedTrackIndex)
        }
      }
      // Ctrl+K — split segment at playhead
      if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 'k') {
        e.preventDefault()
        const s = useSynthStore.getState()
        const ts = s.project.time_signatures[0]
        const ticksPerBar = 480 * (ts?.beat_per_bar ?? 4)
        if (s.activeWavePartIndex >= 0) {
          // Audio track: split at exact playhead position (no bar snap)
          s.splitWavePart(s.activeWavePartIndex, s.playbackPositionTicks)
        } else {
          const splitTick = Math.round(s.playbackPositionTicks / ticksPerBar) * ticksPerBar
          if (s.viewMode === 'pianoroll') {
            s.splitVoicePart(s.activePartIndex, splitTick)
          } else {
            const seg = s.selectedSegment
            if (seg?.type === 'voice') s.splitVoicePart(seg.index, splitTick)
            else if (seg?.type === 'instrument') s.splitInstrumentPart(seg.index, splitTick)
            else if (seg?.type === 'wave') s.splitWavePart(seg.index, s.playbackPositionTicks)
          }
        }
      }
      // Block browser shortcuts that interfere with DAW
      if (e.ctrlKey || e.metaKey) {
        if ('=-+dfghkp'.includes(e.key.toLowerCase())) e.preventDefault()
      }
      if (e.key === 'Delete' || e.key === 'Backspace') {
        e.preventDefault()
        const s = useSynthStore.getState()
        if (s.viewMode === 'arrangement' && s.selectedSegment) {
          const seg = s.selectedSegment
          if (seg.type === 'voice') s.deleteVoicePart(seg.index)
          else if (seg.type === 'wave') s.deleteWavePart(seg.index)
          else if (seg.type === 'instrument') s.deleteInstrumentPart(seg.index)
        }
      }
      if (e.key === 'Tab') e.preventDefault()
    }
    // Block browser zoom on Ctrl/Alt+wheel anywhere on synth page
    const wheelHandler = (e: WheelEvent) => {
      const synthEl = document.querySelector('.utai-synth-page');
      if (!synthEl || (synthEl.closest('[style]') as HTMLElement)?.style.display === 'none') return
      // Only block browser zoom on non-canvas elements (canvas handlers do their own preventDefault)
      const tag = (e.target as HTMLElement)?.tagName;
      if (tag === 'CANVAS') return;
      if (e.ctrlKey || e.metaKey || e.altKey) e.preventDefault()
    }
    const blockSpace = (e: KeyboardEvent) => { if (e.key === ' ' && window.location.pathname === '/synth') e.preventDefault(); };
    window.addEventListener('keydown', handler, true) // capture phase for Tab
    window.addEventListener('keyup', blockSpace, true)
    window.addEventListener('wheel', wheelHandler, { passive: false })
    return () => {
      window.removeEventListener('keydown', handler, true)
      window.removeEventListener('keyup', blockSpace, true)
      window.removeEventListener('wheel', wheelHandler)
    }
  }, [])

  return (
    <div className="utai-synth-page"
      onContextMenu={e => e.preventDefault()}
      onDragStart={e => e.preventDefault()}
      onMouseDown={e => { if (e.button === 1) e.preventDefault(); }}
      style={{ userSelect: 'none' }}
    >
      {/* Toolbar */}
      <div className="utai-synth-toolbar">
        <span className="utai-logo">
          <span className="utai-logo-bracket">「</span>
          {t('app.title')}
          <span className="utai-logo-bracket">」</span>
        </span>

        <span style={{
          fontFamily: 'var(--font-pixel)',
          fontSize: 11,
          color: 'var(--accent-purple)',
          letterSpacing: 1,
        }}>
          {t('synth.title')}
        </span>

        <SynthProjectMenu />

        <button className="utai-btn" onClick={() => setModelManagerOpen(true)}
          style={{ borderColor: 'var(--accent-gold)', color: 'var(--accent-gold)', fontSize: 11 }}>
          {t('toolbar.models')}
        </button>

        <div style={{ flex: 1 }} />

        {hasAudio && <Link to="/" className="utai-btn" style={{ textDecoration: 'none', fontSize: 11, borderColor: 'var(--accent-green)', color: 'var(--accent-green)' }}>
          {t('synth.backToWorkflow')}
        </Link>}
        {hasAudio && <Link to="/train" className="utai-btn" style={{ textDecoration: 'none', fontSize: 11, borderColor: 'var(--accent-pink)', color: 'var(--accent-pink)' }}>
          {t('toolbar.training')}
        </Link>}

        <button className="utai-btn" onClick={() => setNetworkSettingsOpen(true)}
          style={{ fontSize: 11, borderColor: 'var(--border-dim)', color: 'var(--text-dim)', padding: '2px 6px' }}
          title="Network Settings">
          &#9881;
        </button>
        <LangSwitcher />
      </div>

      {/* Split pane: Piano Roll (left) + Controls (right) */}
      <SplitPane
        left={<PianoRollPanel />}
        right={<ControlPanel />}
        defaultLeftWidth={65}
        rightCollapsed={false}
      />

      <ModelManager
        open={modelManagerOpen}
        onClose={() => {
          setModelManagerOpen(false);
          useSynthStore.getState().bumpRvcModelList();
        }}
        initialTab="rvc"
      />
      <NetworkSettings open={networkSettingsOpen} onClose={() => setNetworkSettingsOpen(false)} />
      <ToastContainer />
    </div>
  )
}
