import { useTranslation } from 'react-i18next'

interface ConfirmDialogProps {
  open: boolean
  message: string
  onConfirm: () => void
  onCancel: () => void
}

export function ConfirmDialog({ open, message, onConfirm, onCancel }: ConfirmDialogProps) {
  const { t } = useTranslation()
  if (!open) return null

  return (
    <>
      <div className="utai-modal-backdrop" onClick={onCancel} style={{ zIndex: 2000 }} />
      <div className="utai-confirm" style={{ zIndex: 2001 }}>
        <div className="utai-confirm__msg">{message}</div>
        <div className="utai-confirm__actions">
          <button className="utai-btn utai-confirm__btn utai-confirm__btn--cancel" onClick={onCancel}>
            {t('dialog.cancel')}
          </button>
          <button className="utai-btn utai-confirm__btn utai-confirm__btn--ok" onClick={onConfirm}>
            {t('dialog.confirm')}
          </button>
        </div>
      </div>
    </>
  )
}
