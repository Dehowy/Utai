/**
 * CloseDialog — shown when user clicks window close button.
 * Offers: minimize to tray, exit program, cancel.
 * If exit is chosen and there's unsaved work or running tasks, shows a confirmation.
 */

import { useState, useCallback, useEffect, useRef } from 'react'
import { useTranslation } from 'react-i18next'
import { useExecutionStore } from '../store/executionStore'
import { useTrainingStore } from '../store/trainingStore'
import { useSynthStore } from '../store/synthStore'
import { useWorkflowStore } from '../store/workflowStore'
import { cleanupTemp, cleanupBuffers } from '../api/client'
import { getSynthSettings } from './synth/SynthProjectMenu'

interface CloseDialogProps {
  open: boolean
  fromTray: boolean
  onDone: () => void
}

export function CloseDialog({ open, fromTray, onDone }: CloseDialogProps) {
  const { t } = useTranslation()
  const [confirmQuit, setConfirmQuit] = useState(false)
  const [quitting, setQuitting] = useState(false)
  const fromTrayHandled = useRef(false)

  // Reset state when dialog is closed
  useEffect(() => {
    if (!open) {
      setConfirmQuit(false)
      setQuitting(false)
      fromTrayHandled.current = false
    }
  }, [open])

  const getWarnings = useCallback((): string[] => {
    const warnings: string[] = []
    // Check running tasks
    if (useExecutionStore.getState().isRunning) {
      warnings.push(t('close.warnRunningWorkflow'))
    }
    if (useTrainingStore.getState().isRunningAll) {
      warnings.push(t('close.warnRunningTraining'))
    }
    const renderStatus = useSynthStore.getState().renderStatus
    if (renderStatus === 'rendering' || renderStatus === 'converting') {
      warnings.push(t('close.warnRendering'))
    }
    // Check unsaved work
    if (useSynthStore.getState().dirty) {
      warnings.push(t('close.warnSynthUnsaved'))
    }
    const wf = useWorkflowStore.getState()
    if (wf.nodes.length > 1 || wf.edges.length > 0) {
      warnings.push(t('close.warnWorkflowData'))
    }
    return warnings
  }, [t])

  const doQuit = useCallback(async () => {
    setQuitting(true)
    try {
      const s = useSynthStore.getState()
      await fetch('/api/synth/project/autosave', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ project: s.project, trackEffects: s.trackEffects, settings: getSynthSettings() }),
      }).catch(() => {})
      await cleanupTemp().catch(() => {})
      await cleanupBuffers().catch(() => {})
    } catch {}
    window.electronAPI?.sendCloseAction('quit')
  }, [])

  const handleQuitRequest = useCallback(() => {
    const warnings = getWarnings()
    if (warnings.length > 0) {
      setConfirmQuit(true)
    } else {
      doQuit()
    }
  }, [getWarnings, doQuit])

  const handleMinimize = useCallback(() => {
    window.electronAPI?.sendCloseAction('minimize')
    onDone()
  }, [onDone])

  const handleCancel = useCallback(() => {
    setConfirmQuit(false)
    window.electronAPI?.sendCloseAction('cancel')
    onDone()
  }, [onDone])

  // When opened from tray "Quit", skip minimize dialog — auto-trigger quit flow
  useEffect(() => {
    if (open && fromTray && !fromTrayHandled.current) {
      fromTrayHandled.current = true
      handleQuitRequest()
    }
  }, [open, fromTray, handleQuitRequest])

  if (!open) return null

  // Quitting spinner
  if (quitting) {
    return (
      <>
        <div className="utai-modal-backdrop" style={{ zIndex: 9500 }} />
        <div className="utai-confirm" style={{ zIndex: 9501, textAlign: 'center' }}>
          <div style={{ color: 'var(--accent-cyan)', fontFamily: 'var(--font-pixel)', fontSize: 13 }}>
            {t('close.quitting')}
          </div>
        </div>
      </>
    )
  }

  // Second dialog: confirm quit with warnings
  if (confirmQuit) {
    const warnings = getWarnings()
    return (
      <>
        <div className="utai-modal-backdrop" onClick={handleCancel} style={{ zIndex: 9500 }} />
        <div className="utai-confirm" style={{ zIndex: 9501, maxWidth: 420 }}>
          <div style={{
            color: 'var(--accent-orange)',
            fontFamily: 'var(--font-pixel)',
            fontSize: 13,
            marginBottom: 10,
          }}>
            {t('close.confirmTitle')}
          </div>
          <div style={{
            fontSize: 11,
            color: 'var(--text-dim)',
            marginBottom: 10,
            lineHeight: 1.8,
          }}>
            {warnings.map((w, i) => (
              <div key={i} style={{ display: 'flex', gap: 6, alignItems: 'baseline' }}>
                <span style={{ color: 'var(--accent-orange)' }}>*</span>
                <span>{w}</span>
              </div>
            ))}
          </div>
          <div className="utai-confirm__actions">
            <button
              className="utai-btn utai-confirm__btn utai-confirm__btn--cancel"
              onClick={() => setConfirmQuit(false)}
            >
              {t('dialog.cancel')}
            </button>
            <button
              className="utai-btn utai-confirm__btn utai-confirm__btn--ok"
              style={{ borderColor: 'var(--accent-pink)', color: 'var(--accent-pink)' }}
              onClick={doQuit}
            >
              {t('close.confirmQuit')}
            </button>
          </div>
        </div>
      </>
    )
  }

  // From tray: don't show minimize/quit dialog, let useEffect handle it
  if (fromTray) return null

  // First dialog: minimize or quit
  return (
    <>
      <div className="utai-modal-backdrop" onClick={handleCancel} style={{ zIndex: 9500 }} />
      <div className="utai-confirm" style={{ zIndex: 9501, maxWidth: 380 }}>
        <div style={{
          color: 'var(--accent-cyan)',
          fontFamily: 'var(--font-pixel)',
          fontSize: 13,
          marginBottom: 12,
          textAlign: 'center',
        }}>
          {t('close.title')}
        </div>

        <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
          {/* Minimize to tray */}
          <button
            className="utai-btn utai-close-option"
            onClick={handleMinimize}
          >
            <div style={{ fontFamily: 'var(--font-pixel)', fontSize: 12, color: 'var(--accent-cyan)' }}>
              {t('close.minimize')}
            </div>
            <div style={{ fontSize: 10, color: 'var(--text-dim)', marginTop: 2 }}>
              {t('close.minimizeDesc')}
            </div>
          </button>

          {/* Exit program */}
          <button
            className="utai-btn utai-close-option"
            onClick={handleQuitRequest}
          >
            <div style={{ fontFamily: 'var(--font-pixel)', fontSize: 12, color: 'var(--accent-pink)' }}>
              {t('close.quit')}
            </div>
            <div style={{ fontSize: 10, color: 'var(--text-dim)', marginTop: 2 }}>
              {t('close.quitDesc')}
            </div>
          </button>
        </div>

        <div style={{ textAlign: 'center', marginTop: 10 }}>
          <button
            className="utai-btn utai-confirm__btn utai-confirm__btn--cancel"
            onClick={handleCancel}
            style={{ fontSize: 10 }}
          >
            {t('dialog.cancel')}
          </button>
        </div>
      </div>
    </>
  )
}
