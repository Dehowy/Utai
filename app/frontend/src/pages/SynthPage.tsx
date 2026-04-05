/** Synth page — AI singing voice synthesis with split-pane piano roll editor. */

import { useTranslation } from 'react-i18next'
import { Link } from 'react-router-dom'
import { LangSwitcher } from '../components/LangSwitcher'
import SplitPane from '../components/synth/SplitPane'
import PianoRollPanel from '../components/synth/PianoRollPanel'
import ControlPanel from '../components/synth/ControlPanel'
import SynthProjectMenu from '../components/synth/SynthProjectMenu'

export default function SynthPage() {
  const { t } = useTranslation()

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

        <div style={{ flex: 1 }} />

        <Link to="/" className="utai-btn utai-btn--load" style={{ textDecoration: 'none', fontSize: 11 }}>
          {t('synth.backToWorkflow')}
        </Link>
        <Link to="/train" className="utai-btn utai-btn--save" style={{ textDecoration: 'none', fontSize: 11 }}>
          {t('toolbar.training')}
        </Link>

        <LangSwitcher />
      </div>

      {/* Split pane: Piano Roll (left) + Controls (right) */}
      <SplitPane
        left={<PianoRollPanel />}
        right={<ControlPanel />}
        defaultLeftWidth={65}
      />
    </div>
  )
}
