/** Synth page — AI singing voice synthesis with split-pane piano roll editor. */

import { useState, useEffect } from 'react'
import { useTranslation } from 'react-i18next'
import { Link } from 'react-router-dom'
import { LangSwitcher } from '../components/LangSwitcher'
import { ModelManager } from '../components/ModelManager'
import SplitPane from '../components/synth/SplitPane'
import PianoRollPanel from '../components/synth/PianoRollPanel'
import ControlPanel from '../components/synth/ControlPanel'
import SynthProjectMenu from '../components/synth/SynthProjectMenu'
import { useSynthStore } from '../store/synthStore'

export default function SynthPage() {
  const { t } = useTranslation()
  const [modelManagerOpen, setModelManagerOpen] = useState(false)
  const viewMode = useSynthStore(s => s.viewMode)

  // Global keyboard shortcuts for synth page (gated on pathname for keep-alive)
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if (window.location.pathname !== '/synth') return
      // Don't capture keys when typing in inputs
      const tag = (e.target as HTMLElement)?.tagName
      if (tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT') return

      if ((e.ctrlKey || e.metaKey) && e.key === 's') {
        e.preventDefault()
        const renderBtn = document.querySelector('.utai-synth-render-btn') as HTMLButtonElement | null
        if (renderBtn && !renderBtn.disabled) renderBtn.click()
      }
      if (e.key === ' ') {
        e.preventDefault()
        const playBtn = document.querySelector('.utai-synth-transport__btn') as HTMLButtonElement | null
        if (playBtn) playBtn.click()
      }
    }
    window.addEventListener('keydown', handler)
    return () => window.removeEventListener('keydown', handler)
  }, [])

  return (
    <div className="utai-synth-page">
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

        <Link to="/" className="utai-btn" style={{ textDecoration: 'none', fontSize: 11, borderColor: 'var(--accent-green)', color: 'var(--accent-green)' }}>
          {t('synth.backToWorkflow')}
        </Link>
        <Link to="/train" className="utai-btn" style={{ textDecoration: 'none', fontSize: 11, borderColor: 'var(--accent-pink)', color: 'var(--accent-pink)' }}>
          {t('toolbar.training')}
        </Link>

        <LangSwitcher />
      </div>

      {/* Split pane: Piano Roll (left) + Controls (right) */}
      <SplitPane
        left={<PianoRollPanel />}
        right={<ControlPanel />}
        defaultLeftWidth={65}
        rightCollapsed={viewMode === 'arrangement'}
      />

      <ModelManager
        open={modelManagerOpen}
        onClose={() => setModelManagerOpen(false)}
        initialTab="diffsinger"
      />
    </div>
  )
}
