import { useState, useCallback } from 'react'
import { useTranslation } from 'react-i18next'
import { Link } from 'react-router-dom'
import { useWorkflowStore } from '../store/workflowStore'
import { useAppStore } from '../store/appStore'
import { useExecutionStore } from '../store/executionStore'
import { stopExecution, cleanupTemp, cleanupBuffers } from '../api/client'
import { LangSwitcher } from './LangSwitcher'
import { ProjectMenu } from './ProjectMenu'
import { ConfirmDialog } from './ConfirmDialog'

interface ToolbarProps {
  onExecute: () => void
  onOpenModelManager?: () => void
}

export function Toolbar({ onExecute, onOpenModelManager }: ToolbarProps) {
  const { t } = useTranslation()
  const isRunning = useExecutionStore((s) => s.isRunning)
  const hasSynth = useAppStore(s => s.modules?.synth) !== false
  const executionId = useExecutionStore((s) => s.executionId)
  const { clearWorkflow } = useWorkflowStore()
  const [confirmClear, setConfirmClear] = useState(false)

  const handleStop = useCallback(async () => {
    if (executionId) {
      await stopExecution(executionId).catch(console.error)
      // Give the backend a moment to cancel, then force-reset UI if still running
      setTimeout(() => {
        if (useExecutionStore.getState().isRunning) {
          useExecutionStore.getState().setRunning(false)
        }
      }, 3000)
    }
  }, [executionId])

  const handleClearConfirmed = useCallback(async () => {
    // Reset workflow canvas
    clearWorkflow()
    // Reset execution state
    useExecutionStore.getState().reset()
    useExecutionStore.getState().clearLogs()
    // Clean server-side data
    await cleanupTemp().catch(console.error)
    await cleanupBuffers().catch(console.error)
    setConfirmClear(false)
  }, [clearWorkflow])

  return (
    <div className="utai-toolbar">
      <span className="utai-logo">
        <span className="utai-logo-bracket">「</span>
        {t('app.title')}
        <span className="utai-logo-bracket">」</span>
      </span>

      {/* Left section: 项目 模型 清空 执行全部 */}
      <ProjectMenu />

      <button className="utai-btn utai-btn--load" onClick={onOpenModelManager}>
        {t('toolbar.models')}
      </button>

      <button className="utai-btn utai-btn--clear" onClick={() => setConfirmClear(true)}>
        {t('toolbar.clear')}
      </button>

      <button
        className="utai-btn utai-btn--execute"
        onClick={onExecute}
        disabled={isRunning}
      >
        {isRunning ? t('status.running') + '...' : t('toolbar.executeAll')}
      </button>

      <button
        className="utai-btn utai-btn--logs"
        onClick={() => {
          const store = useExecutionStore.getState()
          store.setLogPanelOpen(!store.logPanelOpen)
        }}
      >
        {t('workflow.logs')}
      </button>

      {isRunning && executionId && (
        <button
          className="utai-btn utai-train-node-stop"
          style={{ padding: '4px 10px', fontSize: 12 }}
          onClick={handleStop}
          title={t('status.stop')}
        >
          ■ {t('status.stop')}
        </button>
      )}

      <div style={{ flex: 1 }} />

      {hasSynth && (
        <Link to="/synth" className="utai-btn utai-btn--synth" style={{ textDecoration: 'none' }}>
          {t('toolbar.synth')}
        </Link>
      )}

      <Link to="/train" className="utai-btn utai-btn--save" style={{ textDecoration: 'none' }}>
        {t('toolbar.training')}
      </Link>

      <LangSwitcher />

      <ConfirmDialog
        open={confirmClear}
        message={t('dialog.confirmClear')}
        onConfirm={handleClearConfirmed}
        onCancel={() => setConfirmClear(false)}
      />
    </div>
  )
}
