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
import { saveProject, getSynthSettings } from '../components/synth/SynthProjectMenu'
import { ToastContainer } from '../components/Toast'
import PianoRollPanel from '../components/synth/PianoRollPanel'
import ControlPanel from '../components/synth/ControlPanel'
import SynthProjectMenu from '../components/synth/SynthProjectMenu'
import { useSynthStore } from '../store/synthStore'

function useAutoRenderEffects() {
  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const abortRef = useRef<AbortController | null>(null);
  const runningRef = useRef(false);
  const pendingRef = useRef(false);
  const prevFxRef = useRef<string>('');

  const doRender = useCallback(async (ctrl: AbortController) => {
    if (isDragging() || runningRef.current) { pendingRef.current = true; return; }
    runningRef.current = true;
    pendingRef.current = false;

    try {
      const state = useSynthStore.getState();
      const { applyEffectsUpload } = await import('../api/client');

      for (let trackIdx = 0; trackIdx < state.project.tracks.length; trackIdx++) {
        const track = state.project.tracks[trackIdx];
        if (ctrl.signal.aborted) break;

        const effects = (state.trackEffects[trackIdx] || [])
          .filter(fx => fx.enabled)
          .map(fx => ({ type: fx.type, params: fx.params, enabled: true }));

        if (track.track_type === 'audio') {
          const waveParts = state.project.wave_parts.filter(wp => wp.track_no === trackIdx && wp.buffer_id);
          if (effects.length === 0) {
            let needsProjectUpdate = false;
            const proj = JSON.parse(JSON.stringify(useSynthStore.getState().project));
            for (const wp of waveParts) {
              const bid = wp.buffer_id!;
              const prev = useSynthStore.getState().waveFxUrls[bid];
              if (prev) {
                URL.revokeObjectURL(prev);
                const next = { ...useSynthStore.getState().waveFxUrls };
                delete next[bid];
                const peakNext = { ...useSynthStore.getState().peakData };
                delete peakNext[`fx:${bid}`];
                useSynthStore.setState({ waveFxUrls: next, peakData: peakNext });
              }
              const wpIdx = proj.wave_parts.findIndex((w2: any) => w2.buffer_id === bid);
              if (wpIdx >= 0 && proj.wave_parts[wpIdx]._origDurationMs) {
                proj.wave_parts[wpIdx].file_duration_ms = proj.wave_parts[wpIdx]._origDurationMs;
                delete proj.wave_parts[wpIdx]._origDurationMs;
                needsProjectUpdate = true;
              }
            }
            if (needsProjectUpdate) useSynthStore.setState({ project: proj });
            continue;
          }
          useSynthStore.setState({ audioFxRendering: { ...useSynthStore.getState().audioFxRendering, [trackIdx]: true } });
          for (const wp of waveParts) {
            const bid = wp.buffer_id!;
            const prevFxUrl = useSynthStore.getState().waveFxUrls[bid];
            const origDurationMs = wp._origDurationMs ?? wp.file_duration_ms;
            try {
              const rawBlob = await fetch(`/api/audio/${bid}`).then(r => r.blob());
              const fxBlob = await applyEffectsUpload(rawBlob, effects);
              if (prevFxUrl) URL.revokeObjectURL(prevFxUrl);
              const fxUrl = URL.createObjectURL(fxBlob);
              useSynthStore.setState({ waveFxUrls: { ...useSynthStore.getState().waveFxUrls, [bid]: fxUrl } });
              try {
                const { loadAudioBuffer } = await import('../lib/synth/audioEngine');
                const audioBuf = await loadAudioBuffer(fxUrl);
                const newDurationMs = audioBuf.duration * 1000;
                const ch = audioBuf.getChannelData(0);
                const w = 4000;
                const peaks = new Float32Array(w);
                const step = Math.max(1, Math.floor(ch.length / w));
                for (let i = 0; i < w; i++) {
                  let max = 0;
                  const off = i * step;
                  for (let j = 0; j < step && off + j < ch.length; j++) max = Math.max(max, Math.abs(ch[off + j]));
                  peaks[i] = max;
                }
                useSynthStore.getState().setPeakData(`fx:${bid}`, peaks);
                const proj = useSynthStore.getState().project;
                const wpIdx = proj.wave_parts.findIndex(w2 => w2.buffer_id === bid);
                if (wpIdx >= 0) {
                  const updated = JSON.parse(JSON.stringify(proj));
                  updated.wave_parts[wpIdx]._origDurationMs = origDurationMs;
                  updated.wave_parts[wpIdx].file_duration_ms = newDurationMs;
                  useSynthStore.setState({ project: updated });
                }
              } catch { /* peak/duration update failed */ }
            } catch { /* fx processing failed */ }
          }
          useSynthStore.setState({ audioFxRendering: { ...useSynthStore.getState().audioFxRendering, [trackIdx]: false } });
          continue;
        }

        const rawUrl = state.trackRawAudioUrls[trackIdx];
        if (!rawUrl) continue;

        const prevUrl = state.trackAudioUrls[trackIdx];

        if (effects.length === 0) {
          if (prevUrl && prevUrl !== rawUrl) URL.revokeObjectURL(prevUrl);
          useSynthStore.getState().setTrackAudioUrl(trackIdx, rawUrl);
          if (track.track_type === 'vocal') useSynthStore.getState().setFinalUrl(rawUrl);
          continue;
        }

        try {
          const rawBlob = await fetch(rawUrl).then(r => r.blob());
          if (ctrl.signal.aborted) continue;
          const fxBlob = await applyEffectsUpload(rawBlob, effects);
          if (ctrl.signal.aborted) continue;
          if (prevUrl && prevUrl !== rawUrl) URL.revokeObjectURL(prevUrl);
          const fxUrl = URL.createObjectURL(fxBlob);
          useSynthStore.getState().setTrackAudioUrl(trackIdx, fxUrl);
          if (track.track_type === 'vocal') useSynthStore.getState().setFinalUrl(fxUrl);
        } catch { /* aborted or failed */ }
      }
    } finally {
      runningRef.current = false;
      if (pendingRef.current && !ctrl.signal.aborted) {
        pendingRef.current = false;
        const next = new AbortController();
        abortRef.current = next;
        setTimeout(() => doRender(next), 300);
      }
    }
  }, []);

  useEffect(() => {
    const unsub = useSynthStore.subscribe((state) => {
      const snap = JSON.stringify(state.trackEffects);
      if (snap === prevFxRef.current) return;
      prevFxRef.current = snap;

      if (timerRef.current) clearTimeout(timerRef.current);
      abortRef.current?.abort();

      const ctrl = new AbortController();
      abortRef.current = ctrl;
      timerRef.current = setTimeout(() => {
        if (!runningRef.current) doRender(ctrl);
      }, 600);
    });
    return () => { unsub(); if (timerRef.current) clearTimeout(timerRef.current); };
  }, [doRender]);
}

function flushAutosave() {
  const s = useSynthStore.getState();
  return fetch('/api/synth/project/autosave', {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ project: s.project, trackEffects: s.trackEffects, settings: getSynthSettings() }),
  }).catch(() => {});
}

function useAutoSave() {
  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  // Debounce-save on ANY state change (500ms). No dirty gate — every change is
  // worth preserving so recovery never loses settings, RVC models, or toggles.
  useEffect(() => {
    const unsub = useSynthStore.subscribe(() => {
      if (timerRef.current) clearTimeout(timerRef.current);
      timerRef.current = setTimeout(() => {
        const s = useSynthStore.getState();
        if (s.project.voice_parts.length > 0 || s.project.wave_parts.length > 0 || s.project.instrument_parts.length > 0) {
          flushAutosave().catch(() => {});
        }
      }, 500);
    });
    return () => { unsub(); if (timerRef.current) clearTimeout(timerRef.current); };
  }, []);

  // Save on page unload (F5, tab close, navigation)
  useEffect(() => {
    const onUnload = () => {
      const s = useSynthStore.getState();
      if (s.project.voice_parts.length > 0 || s.project.wave_parts.length > 0 || s.project.instrument_parts.length > 0) {
        const blob = new Blob([JSON.stringify({ project: s.project, trackEffects: s.trackEffects, settings: getSynthSettings() })], { type: 'application/json' });
        navigator.sendBeacon('/api/synth/project/autosave', blob);
      }
    };
    window.addEventListener('beforeunload', onUnload);
    return () => window.removeEventListener('beforeunload', onUnload);
  }, []);

  // Listen for force-autosave from Electron main (native fallback quit path)
  useEffect(() => {
    const api = window.electronAPI;
    if (!api?.onForceAutosave) return;
    return api.onForceAutosave(() => { flushAutosave(); }) || undefined;
  }, []);
}

function useAutoRecover() {
  const doneRef = useRef(false);
  useEffect(() => {
    if (doneRef.current) return;
    doneRef.current = true;
    (async () => {
      try {
        const res = await fetch('/api/synth/project/autosave');
        const data = await res.json();
        if (data.available && data.project) {
          const proj = data.project;
          const hasContent =
            (proj.voice_parts || []).some((vp: any) => vp.notes?.length > 0) ||
            (proj.wave_parts || []).length > 0 ||
            (proj.instrument_parts || []).some((ip: any) => ip.notes?.length > 0);
          if (!hasContent) {
            await fetch('/api/synth/project/autosave', { method: 'DELETE' });
          } else {
            const { utaiConfirm } = await import('../components/UtaiAlert');
            const { showToast } = await import('../components/Toast');
            const i18n = (await import('../i18n/i18n')).default;
            const recover = await utaiConfirm(i18n.t('synth.recoverProject'));
            if (recover) {
              const { applyOpenedProject } = await import('../components/synth/SynthProjectMenu');
              applyOpenedProject(data);
              showToast('Project recovered', 'success');
            } else {
              await fetch('/api/synth/project/autosave', { method: 'DELETE' });
            }
          }
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
