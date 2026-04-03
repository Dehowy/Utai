import { useEffect, useState, useCallback, useRef } from 'react'
import { useTranslation } from 'react-i18next'
import { ConfirmDialog } from './ConfirmDialog'
import {
  getModelCategories,
  getModelsByCategory,
  downloadModel,
  getDownloadStatus,
  deleteModel,
  getRVCModels,
  importRVCModel,
  archiveRVCModel,
  getArchivedRVCModels,
  restoreRVCModel,
  deleteArchivedRVCModel,
} from '../api/client'
import { emitModelsChanged } from '../utils/modelEvents'

interface ModelManagerProps {
  open: boolean
  onClose: () => void
  initialCategory?: string
  /** Start on RVC tab */
  initialTab?: 'separation' | 'rvc'
}

function formatSize(bytes: number): string {
  if (!bytes) return ''
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(0)} KB`
  if (bytes < 1024 * 1024 * 1024) return `${(bytes / (1024 * 1024)).toFixed(1)} MB`
  return `${(bytes / (1024 * 1024 * 1024)).toFixed(2)} GB`
}

export function ModelManager({ open, onClose, initialCategory, initialTab }: ModelManagerProps) {
  const { t, i18n } = useTranslation()
  // Top-level section: 'separation' or 'rvc'
  const [section, setSection] = useState<'separation' | 'rvc'>(initialTab || 'separation')
  const [categories, setCategories] = useState<any[]>([])
  const [selectedCat, setSelectedCat] = useState(initialCategory || '')
  const [models, setModels] = useState<any[]>([])
  const [rvcModels, setRvcModels] = useState<any[]>([])
  const [archivedModels, setArchivedModels] = useState<any[]>([])
  const [rvcSubTab, setRvcSubTab] = useState<'active' | 'archived'>('active')
  const [downloading, setDownloading] = useState<Record<string, number>>({})
  const [deleteMode, setDeleteMode] = useState(false)
  const [deleting, setDeleting] = useState<Record<string, boolean>>({})
  const [confirmDelete, setConfirmDelete] = useState<{ name: string; action: () => void } | null>(null)
  const [dragOffset, setDragOffset] = useState<{ x: number; y: number } | null>(null)
  const [modalPos, setModalPos] = useState<{ x: number; y: number } | null>(null)
  const [scale, setScale] = useState(1)
  const [resizing, setResizing] = useState(false)
  const resizeRef = useRef<{ startY: number; startScale: number } | null>(null)
  const wrapperRef = useRef<HTMLDivElement>(null)
  const pollRef = useRef<Record<string, ReturnType<typeof setInterval>>>({})

  const refreshSepData = useCallback(() => {
    getModelCategories().then(setCategories).catch(console.error)
    if (selectedCat) {
      getModelsByCategory(selectedCat).then(setModels).catch(console.error)
    }
  }, [selectedCat])

  const refreshRvcData = useCallback(() => {
    getRVCModels().then(setRvcModels).catch(console.error)
    getArchivedRVCModels().then(setArchivedModels).catch(console.error)
  }, [])

  // Load data on open
  useEffect(() => {
    if (!open) return
    getModelCategories().then((cats) => {
      setCategories(cats)
      if (!selectedCat && cats.length > 0) {
        setSelectedCat(initialCategory || cats[0].id)
      }
    }).catch(console.error)
    refreshRvcData()
  }, [open])

  // Load models for selected separation category
  useEffect(() => {
    if (!selectedCat || !open || section !== 'separation') return
    getModelsByCategory(selectedCat).then(setModels).catch(console.error)
  }, [selectedCat, open, section])

  // Reset on close
  useEffect(() => {
    if (!open) {
      setDeleteMode(false)
      setModalPos(null)
      setScale(1)
      setImportOpen(false)
      setImportPthFile(null)
      setImportIdxFile(null)
      setImportPthPath('')
      setImportIdxPath('')
    }
  }, [open])

  // Exit delete mode when switching section
  useEffect(() => {
    setDeleteMode(false)
  }, [section])

  // ── Drag ──
  const handleHeaderMouseDown = useCallback((e: React.MouseEvent) => {
    if ((e.target as HTMLElement).closest('button')) return
    e.preventDefault()
    const wrapper = wrapperRef.current
    if (!wrapper) return
    const rect = wrapper.getBoundingClientRect()
    setDragOffset({ x: e.clientX - rect.left, y: e.clientY - rect.top })
  }, [])

  useEffect(() => {
    if (!dragOffset) return
    const move = (e: MouseEvent) => setModalPos({ x: e.clientX - dragOffset.x, y: e.clientY - dragOffset.y })
    const up = () => setDragOffset(null)
    window.addEventListener('mousemove', move)
    window.addEventListener('mouseup', up)
    return () => { window.removeEventListener('mousemove', move); window.removeEventListener('mouseup', up) }
  }, [dragOffset])

  // ── Resize ──
  const handleResizeMouseDown = useCallback((e: React.MouseEvent) => {
    e.preventDefault()
    resizeRef.current = { startY: e.clientY, startScale: scale }
    setResizing(true)
  }, [scale])

  useEffect(() => {
    if (!resizing) return
    const start = resizeRef.current!
    const move = (e: MouseEvent) => setScale(Math.min(2.5, Math.max(0.7, start.startScale + (e.clientY - start.startY) / 300)))
    const up = () => { resizeRef.current = null; setResizing(false) }
    document.addEventListener('mousemove', move)
    document.addEventListener('mouseup', up)
    return () => { document.removeEventListener('mousemove', move); document.removeEventListener('mouseup', up) }
  }, [resizing])

  const getCatLabel = (cat: any) => {
    const lang = i18n.language
    return cat[`label_${lang}`] || cat.label_zh || cat.id
  }

  // ── Separation download ──
  const handleDownload = useCallback(async (modelName: string) => {
    try {
      setDownloading((prev) => ({ ...prev, [modelName]: 0 }))
      await downloadModel(modelName)
      const interval = setInterval(async () => {
        try {
          const status = await getDownloadStatus(modelName)
          if (status.status === 'completed') {
            clearInterval(interval)
            delete pollRef.current[modelName]
            setDownloading((prev) => { const n = { ...prev }; delete n[modelName]; return n })
            refreshSepData()
            emitModelsChanged()
          } else if (status.status === 'error') {
            clearInterval(interval)
            delete pollRef.current[modelName]
            setDownloading((prev) => { const n = { ...prev }; delete n[modelName]; return n })
          } else {
            setDownloading((prev) => ({ ...prev, [modelName]: status.progress || 0 }))
          }
        } catch { /* ignore */ }
      }, 1000)
      pollRef.current[modelName] = interval
    } catch { /* ignore */ }
  }, [refreshSepData])

  // ── Separation delete ──
  const handleDeleteSep = useCallback(async (modelName: string) => {
    setConfirmDelete({
      name: modelName,
      action: async () => {
        setDeleting((prev) => ({ ...prev, [modelName]: true }))
        try {
          await deleteModel(modelName)
          refreshSepData()
          emitModelsChanged()
        } catch { /* ignore */ }
        finally {
          setDeleting((prev) => { const n = { ...prev }; delete n[modelName]; return n })
        }
      },
    })
  }, [refreshSepData])

  // ── RVC delete ──
  const handleDeleteRvc = useCallback(async (modelName: string) => {
    setConfirmDelete({
      name: modelName,
      action: async () => {
        setDeleting((prev) => ({ ...prev, [modelName]: true }))
        try {
          // RVC models: DELETE via the /api/models/rvc/{name} or we can add a dedicated endpoint
          // For now, use a simple fetch
          const API_BASE = (import.meta as any).env?.VITE_API_URL || ''
          await fetch(`${API_BASE}/api/models/rvc/${encodeURIComponent(modelName)}`, { method: 'DELETE' })
          refreshRvcData()
          emitModelsChanged()
        } catch { /* ignore */ }
        finally {
          setDeleting((prev) => { const n = { ...prev }; delete n[modelName]; return n })
        }
      },
    })
  }, [refreshRvcData])

  // ── RVC import panel ──
  const [importOpen, setImportOpen] = useState(false)
  const [importPthFile, setImportPthFile] = useState<File | null>(null)
  const [importIdxFile, setImportIdxFile] = useState<File | null>(null)
  const [importPthPath, setImportPthPath] = useState('')
  const [importIdxPath, setImportIdxPath] = useState('')
  const [importing, setImporting] = useState(false)
  const [importError, setImportError] = useState('')
  const pthInputRef = useRef<HTMLInputElement>(null)
  const idxInputRef = useRef<HTMLInputElement>(null)

  const handleImportSubmit = useCallback(async () => {
    setImportError('')
    if (!importPthFile) {
      setImportError(t('models.rvcImportNoPth'))
      return
    }
    if (!importPthFile.name.endsWith('.pth')) {
      setImportError(t('models.rvcImportBadPth'))
      return
    }
    setImporting(true)
    try {
      await importRVCModel(importPthFile, importIdxFile || undefined)
      refreshRvcData()
      emitModelsChanged()
      setImportOpen(false)
      setImportPthFile(null)
      setImportIdxFile(null)
      setImportPthPath('')
      setImportIdxPath('')
    } catch (err: any) {
      const msg = err?.message || String(err)
      if (msg.includes('409')) {
        setImportError(t('models.rvcImportExists'))
      } else {
        setImportError(msg)
      }
    }
    setImporting(false)
  }, [importPthFile, importIdxFile, refreshRvcData, t])

  // ── RVC archive ──
  const handleArchiveRvc = useCallback(async (modelName: string) => {
    setDeleting((prev) => ({ ...prev, [modelName]: true }))
    try {
      await archiveRVCModel(modelName)
      refreshRvcData()
      emitModelsChanged()
    } catch { /* ignore */ }
    finally {
      setDeleting((prev) => { const n = { ...prev }; delete n[modelName]; return n })
    }
  }, [refreshRvcData])

  // ── RVC restore ──
  const handleRestoreRvc = useCallback(async (archivedName: string) => {
    setDeleting((prev) => ({ ...prev, [archivedName]: true }))
    try {
      await restoreRVCModel(archivedName)
      refreshRvcData()
      emitModelsChanged()
    } catch { /* ignore */ }
    finally {
      setDeleting((prev) => { const n = { ...prev }; delete n[archivedName]; return n })
    }
  }, [refreshRvcData])

  // ── RVC delete archived ──
  const handleDeleteArchived = useCallback(async (archivedName: string) => {
    setConfirmDelete({
      name: archivedName,
      action: async () => {
        setDeleting((prev) => ({ ...prev, [archivedName]: true }))
        try {
          await deleteArchivedRVCModel(archivedName)
          refreshRvcData()
        } catch { /* ignore */ }
        finally {
          setDeleting((prev) => { const n = { ...prev }; delete n[archivedName]; return n })
        }
      },
    })
  }, [refreshRvcData])

  // Cleanup
  useEffect(() => {
    return () => { Object.values(pollRef.current).forEach(clearInterval) }
  }, [])

  if (!open) return null

  const displayModels = deleteMode ? models.filter((m) => m.installed) : models

  return (
    <>
      <div className="utai-modal-backdrop" onClick={onClose} />
      <div
        ref={wrapperRef}
        className="utai-modal-positioner"
        style={modalPos ? { left: modalPos.x, top: modalPos.y, transform: 'none' } : undefined}
      >
        <div
          className="utai-modal utai-model-manager"
          style={{ transform: `scale(${scale})`, transformOrigin: 'top left' }}
        >
          <div className="utai-modal__header" onMouseDown={handleHeaderMouseDown} style={{ cursor: 'grab' }}>
            <span>{t('models.title')}</span>
            <div style={{ display: 'flex', gap: 6, alignItems: 'center' }}>
              <button
                className={`utai-btn utai-model-manager__delete-mode-btn ${deleteMode ? 'utai-model-manager__delete-mode-btn--active' : ''}`}
                onClick={() => setDeleteMode(!deleteMode)}
              >
                {deleteMode ? t('models.exitDelete') : t('models.manage')}
              </button>
              <button className="utai-modal__close" onClick={onClose}>x</button>
            </div>
          </div>

          {/* Top-level section tabs: Separation | RVC */}
          <div className="utai-model-manager__section-tabs">
            <button
              className={`utai-model-manager__section-tab ${section === 'separation' ? 'utai-model-manager__section-tab--active' : ''}`}
              onClick={() => setSection('separation')}
            >
              {t('models.tabSeparation')}
            </button>
            <button
              className={`utai-model-manager__section-tab ${section === 'rvc' ? 'utai-model-manager__section-tab--active' : ''}`}
              onClick={() => setSection('rvc')}
            >
              {t('models.tabRVC')}
            </button>
          </div>

          {section === 'separation' ? (
            <div className="utai-model-manager__body">
              {/* Category sub-tabs */}
              <div className="utai-model-manager__tabs">
                {categories.map((cat) => (
                  <button
                    key={cat.id}
                    className={`utai-model-manager__tab ${selectedCat === cat.id ? 'utai-model-manager__tab--active' : ''}`}
                    onClick={() => setSelectedCat(cat.id)}
                    style={selectedCat === cat.id ? { borderColor: cat.color, color: cat.color } : undefined}
                  >
                    {getCatLabel(cat)}
                    <span className="utai-model-manager__tab-count">
                      {cat.installed_count}/{cat.total_count}
                    </span>
                  </button>
                ))}
              </div>

              {/* Model list grouped by architecture */}
              <div className="utai-model-manager__list">
                {(() => {
                  // Group models by arch_detail
                  const groups: Record<string, any[]> = {}
                  for (const m of displayModels) {
                    const arch = m.arch_detail || m.arch || 'Other'
                    if (!groups[arch]) groups[arch] = []
                    groups[arch].push(m)
                  }
                  const archOrder = ['VR', 'MDX-Net', 'Demucs', 'MDX23C', 'BS-Roformer', 'Mel-Roformer', 'Swin', 'SCNet', 'SegM', 'Bandit', 'Apollo', 'MDXC', 'Other']
                  const sortedKeys = Object.keys(groups).sort((a, b) => {
                    const ai = archOrder.indexOf(a)
                    const bi = archOrder.indexOf(b)
                    return (ai === -1 ? 99 : ai) - (bi === -1 ? 99 : bi)
                  })

                  const lang = i18n.language as 'zh' | 'ja' | 'en'
                  const descKey = `desc_${lang}` as const

                  return sortedKeys.map((archKey) => (
                    <div key={archKey}>
                      {sortedKeys.length > 1 && (
                        <div style={{
                          padding: '4px 10px 2px',
                          fontSize: 10,
                          color: 'var(--accent-cyan)',
                          fontFamily: 'var(--font-pixel)',
                          borderBottom: '1px solid var(--border-dim)',
                          background: 'rgba(92, 224, 216, 0.04)',
                        }}>
                          [{archKey}] — {groups[archKey].length}
                        </div>
                      )}
                      {groups[archKey].map((m: any) => {
                        const isDownloading = m.name in downloading
                        const isDeleting = m.name in deleting
                        const progress = downloading[m.name] || 0
                        const desc = m[descKey] || m.desc_zh || ''
                        return (
                          <div key={m.name} className={`utai-model-manager__item ${deleteMode ? 'utai-model-manager__item--delete-mode' : ''}`}>
                            <div className="utai-model-manager__item-info">
                              <span className={`utai-model-manager__item-name ${m.installed ? '' : 'utai-model-manager__item-name--dim'}`}>
                                {!deleteMode && m.installed && <span className="utai-model-manager__check">&#10003; </span>}
                                {m.name}
                                {!m.is_official && <span style={{ color: 'var(--accent-red)', fontSize: 10, marginLeft: 2 }}>*</span>}
                              </span>
                              {desc && (
                                <span style={{ fontSize: 9, color: 'var(--accent-cyan)', opacity: 0.55, marginLeft: 6, whiteSpace: 'nowrap' }}>
                                  {desc}
                                </span>
                              )}
                              <span className="utai-model-manager__item-size">{formatSize(m.size)}</span>
                            </div>
                            <div className="utai-model-manager__item-actions">
                              {deleteMode ? (
                                m.installed ? (
                                  <button className="utai-btn utai-model-manager__delete-btn" onClick={() => handleDeleteSep(m.name)} disabled={isDeleting}>
                                    {isDeleting ? '...' : t('models.delete')}
                                  </button>
                                ) : null
                              ) : m.installed ? (
                                <span className="utai-model-manager__installed">{t('models.installed')}</span>
                              ) : isDownloading ? (
                                <div className="utai-model-manager__progress">
                                  <div className="utai-model-manager__progress-bar">
                                    <div className="utai-model-manager__progress-fill" style={{ width: `${progress}%` }} />
                                  </div>
                                  <span className="utai-model-manager__progress-text">{progress.toFixed(0)}%</span>
                                </div>
                              ) : (
                                <button className="utai-btn utai-model-manager__download-btn" onClick={() => handleDownload(m.name)}>
                                  {t('models.download')}
                                </button>
                              )}
                            </div>
                          </div>
                        )
                      })}
                    </div>
                  ))
                })()}
                {displayModels.length === 0 && (
                  <div style={{ textAlign: 'center', color: 'var(--text-dim)', padding: 20, fontSize: 12 }}>
                    {deleteMode ? t('models.noInstalled') : t('models.empty')}
                  </div>
                )}
              </div>
            </div>
          ) : (
            /* ── RVC Section ── */
            <div className="utai-model-manager__body">
              {/* Hidden file inputs */}
              <input ref={pthInputRef} type="file" accept=".pth" style={{ display: 'none' }}
                onChange={(e) => {
                  const f = e.target.files?.[0]
                  if (f) { setImportPthFile(f); setImportPthPath(f.name) }
                  e.target.value = ''
                }} />
              <input ref={idxInputRef} type="file" accept=".index" style={{ display: 'none' }}
                onChange={(e) => {
                  const f = e.target.files?.[0]
                  if (f) { setImportIdxFile(f); setImportIdxPath(f.name) }
                  e.target.value = ''
                }} />

              {/* Import button + sub-tabs */}
              <div style={{ display: 'flex', alignItems: 'center', gap: 6, padding: '6px 10px', borderBottom: '1px solid var(--border-dim)' }}>
                <button
                  className="utai-btn utai-model-manager__download-btn"
                  onClick={() => setImportOpen(!importOpen)}
                  style={{ fontSize: 10, padding: '2px 8px' }}
                >
                  {t('models.rvcImport')}
                </button>
                <div style={{ flex: 1 }} />
                <button
                  className={`utai-model-manager__section-tab ${rvcSubTab === 'active' ? 'utai-model-manager__section-tab--active' : ''}`}
                  style={{ fontSize: 10, padding: '2px 8px' }}
                  onClick={() => setRvcSubTab('active')}
                >
                  {t('models.rvcActive')}
                </button>
                <button
                  className={`utai-model-manager__section-tab ${rvcSubTab === 'archived' ? 'utai-model-manager__section-tab--active' : ''}`}
                  style={{ fontSize: 10, padding: '2px 8px' }}
                  onClick={() => setRvcSubTab('archived')}
                >
                  {t('models.rvcArchived')} {archivedModels.length > 0 && `(${archivedModels.length})`}
                </button>
              </div>

              {/* Import panel */}
              {importOpen && (
                <div className="utai-rvc-import-panel">
                  {/* .pth file row */}
                  <div className="utai-rvc-import-panel__row">
                    <label className="utai-rvc-import-panel__label">{t('models.rvcImportPth')}</label>
                    <div className="utai-rvc-import-panel__input-row">
                      <input
                        className="utai-input"
                        type="text"
                        readOnly
                        value={importPthPath}
                        placeholder="model.pth"
                        style={{ flex: 1 }}
                      />
                      <button
                        className="utai-btn utai-btn--browse"
                        onClick={() => pthInputRef.current?.click()}
                      >
                        {t('models.rvcImportBrowse')}
                      </button>
                    </div>
                  </div>
                  {/* .index file row */}
                  <div className="utai-rvc-import-panel__row">
                    <label className="utai-rvc-import-panel__label">{t('models.rvcImportIdx')}</label>
                    <div className="utai-rvc-import-panel__input-row">
                      <input
                        className="utai-input"
                        type="text"
                        readOnly
                        value={importIdxPath}
                        placeholder="model.index"
                        style={{ flex: 1 }}
                      />
                      <button
                        className="utai-btn utai-btn--browse"
                        onClick={() => idxInputRef.current?.click()}
                      >
                        {t('models.rvcImportBrowse')}
                      </button>
                    </div>
                  </div>
                  {/* Error message */}
                  {importError && (
                    <div style={{ fontSize: 10, color: 'var(--accent-red)', marginTop: 2, textAlign: 'center' }}>
                      {importError}
                    </div>
                  )}
                  {/* Submit — centered */}
                  <div style={{ display: 'flex', justifyContent: 'center', marginTop: 6 }}>
                    <button
                      className="utai-btn utai-model-manager__download-btn"
                      disabled={importing}
                      onClick={handleImportSubmit}
                      style={{ fontSize: 10, padding: '3px 16px' }}
                    >
                      {importing ? t('models.rvcImportUploading') : t('models.rvcImportSubmit')}
                    </button>
                  </div>
                </div>
              )}

              <div className="utai-model-manager__list">
                {rvcSubTab === 'active' ? (
                  <>
                    {rvcModels.map((m) => {
                      const isDeleting = m.name in deleting
                      return (
                        <div key={m.name} className={`utai-model-manager__item ${deleteMode ? 'utai-model-manager__item--delete-mode' : ''}`}>
                          <div className="utai-model-manager__item-info">
                            <span className="utai-model-manager__item-name">
                              {m.name}
                            </span>
                            <span className="utai-model-manager__item-size" style={{ color: 'var(--text-muted)', fontSize: 9 }}>
                              {m.index_files?.length ? `${m.index_files.length} index` : ''}
                            </span>
                          </div>
                          <div className="utai-model-manager__item-actions">
                            {deleteMode ? (
                              <div style={{ display: 'flex', gap: 4 }}>
                                <button className="utai-btn" onClick={() => handleArchiveRvc(m.name)} disabled={isDeleting}
                                  style={{ fontSize: 9, padding: '1px 6px', color: 'var(--accent-orange)', borderColor: 'var(--accent-orange)' }}>
                                  {t('models.archive')}
                                </button>
                                <button className="utai-btn utai-model-manager__delete-btn" onClick={() => handleDeleteRvc(m.name)} disabled={isDeleting}>
                                  {isDeleting ? '...' : t('models.delete')}
                                </button>
                              </div>
                            ) : (
                              <span className="utai-model-manager__installed">{t('models.installed')}</span>
                            )}
                          </div>
                        </div>
                      )
                    })}
                    {rvcModels.length === 0 && (
                      <div style={{ textAlign: 'center', color: 'var(--text-dim)', padding: 20, fontSize: 12 }}>
                        {t('models.rvcEmpty')}
                      </div>
                    )}
                  </>
                ) : (
                  <>
                    {archivedModels.map((m) => {
                      const isProcessing = m.name in deleting
                      return (
                        <div key={m.name} className="utai-model-manager__item">
                          <div className="utai-model-manager__item-info">
                            <span className="utai-model-manager__item-name" style={{ opacity: 0.6 }}>
                              {m.name}
                            </span>
                            <span className="utai-model-manager__item-size" style={{ color: 'var(--text-muted)', fontSize: 9 }}>
                              {formatSize(m.size)}
                            </span>
                          </div>
                          <div className="utai-model-manager__item-actions" style={{ display: 'flex', gap: 4 }}>
                            <button className="utai-btn" onClick={() => handleRestoreRvc(m.name)} disabled={isProcessing}
                              style={{ fontSize: 9, padding: '1px 6px', color: 'var(--accent-green)', borderColor: 'var(--accent-green)' }}>
                              {t('models.restore')}
                            </button>
                            {deleteMode && (
                              <button className="utai-btn utai-model-manager__delete-btn" onClick={() => handleDeleteArchived(m.name)} disabled={isProcessing}>
                                {isProcessing ? '...' : t('models.delete')}
                              </button>
                            )}
                          </div>
                        </div>
                      )
                    })}
                    {archivedModels.length === 0 && (
                      <div style={{ textAlign: 'center', color: 'var(--text-dim)', padding: 20, fontSize: 12 }}>
                        {t('models.archiveEmpty')}
                      </div>
                    )}
                  </>
                )}
              </div>
            </div>
          )}

          <div className="utai-modal__resize-handle" onMouseDown={handleResizeMouseDown} />
        </div>
      </div>
      <ConfirmDialog
        open={!!confirmDelete}
        message={t('dialog.confirmDelete', { name: confirmDelete?.name || '' })}
        onConfirm={() => { confirmDelete?.action(); setConfirmDelete(null) }}
        onCancel={() => setConfirmDelete(null)}
      />
    </>
  )
}
