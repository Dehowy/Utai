import { useEffect, useState, useCallback } from 'react'
import { useTranslation } from 'react-i18next'
import { useWorkflowStore } from '../store/workflowStore'
import { getModelCategories } from '../api/client'
import { onModelsChanged } from '../utils/modelEvents'

interface PaletteItem {
  type: string
  label: string
  category: string
  color: string
  /** For separation nodes, the model category */
  sepCategory?: string
}

let nodeIdCounter = 0

export function getNextNodeId() {
  nodeIdCounter++
  return `node_${Date.now()}_${nodeIdCounter}`
}

export function NodePalette() {
  const { t, i18n } = useTranslation()
  const addNode = useWorkflowStore((s) => s.addNode)
  const nodes = useWorkflowStore((s) => s.nodes)
  const [modelCategories, setModelCategories] = useState<any[]>([])

  const fetchCategories = useCallback(() => {
    getModelCategories().then(setModelCategories).catch(console.error)
  }, [])

  useEffect(() => {
    fetchCategories()
    // Re-fetch when models are downloaded/deleted
    return onModelsChanged(fetchCategories)
  }, [fetchCategories])

  const getTopZIndex = () => {
    // Read directly from store to avoid stale closures
    const currentNodes = useWorkflowStore.getState().nodes
    let max = 0
    for (const n of currentNodes) {
      const z = (n.style?.zIndex as number) || (n.zIndex as number) || 0
      if (z > max) max = z
    }
    return max + 1
  }

  const handleAdd = (item: PaletteItem) => {
    const zIndex = getTopZIndex() + 1000
    addNode({
      id: getNextNodeId(),
      type: item.type,
      position: { x: 200 + nodeIdCounter * 30, y: 100 + nodeIdCounter * 30 },
      data: { params: item.sepCategory ? { category: item.sepCategory } : {} },
      zIndex,
    })
  }

  const handleDragStart = (e: React.DragEvent, item: PaletteItem) => {
    const dragData = item.sepCategory
      ? JSON.stringify({ type: item.type, category: item.sepCategory })
      : item.type
    e.dataTransfer.setData('application/utai-node-type', dragData)
    e.dataTransfer.effectAllowed = 'move'
  }

  const lang = i18n.language as 'zh' | 'ja' | 'en'

  // Build separation items from dynamic categories (only show if installed > 0)
  const sepItems: PaletteItem[] = modelCategories
    .filter((cat) => cat.installed_count > 0)
    .map((cat) => ({
      type: 'separation',
      label: cat[`label_${lang}`] || cat.label_zh,
      category: 'separation',
      color: cat.color,
      sepCategory: cat.id,
    }))

  const hasNoModels = modelCategories.length > 0 && sepItems.length === 0

  return (
    <div className="utai-palette">
      <div className="utai-palette__title">
        {t('palette.title')}
      </div>

      {/* Input */}
      <div className="utai-palette__category">
        <div className="utai-palette__category-label">{t('palette.input')}</div>
        <button
          className="utai-palette__item"
          onClick={() => handleAdd({ type: 'audio_input', label: '', category: 'input', color: 'var(--accent-green)' })}
          draggable
          onDragStart={(e) => handleDragStart(e, { type: 'audio_input', label: '', category: 'input', color: 'var(--accent-green)' })}
          style={{ color: 'var(--accent-green)', borderColor: 'var(--accent-green)' }}
        >
          {t('node.audioInput')}
        </button>
      </div>

      {/* Download */}
      <div className="utai-palette__category">
        <div className="utai-palette__category-label">{t('palette.download')}</div>
        <button
          className="utai-palette__item"
          onClick={() => handleAdd({ type: 'ytdlp_download', label: '', category: 'input', color: 'var(--accent-orange)' })}
          draggable
          onDragStart={(e) => handleDragStart(e, { type: 'ytdlp_download', label: '', category: 'input', color: 'var(--accent-orange)' })}
          style={{ color: 'var(--accent-orange)', borderColor: 'var(--accent-orange)' }}
        >
          {t('node.ytdlpDownload')}
        </button>
      </div>

      {/* Separation — dynamic categories */}
      <div className="utai-palette__category">
        <div className="utai-palette__category-label">{t('palette.separation')}</div>
        {sepItems.map((item) => (
          <button
            key={item.sepCategory}
            className="utai-palette__item"
            onClick={() => handleAdd(item)}
            draggable
            onDragStart={(e) => handleDragStart(e, item)}
            style={{ color: item.color, borderColor: item.color }}
          >
            {item.label}
          </button>
        ))}
        {hasNoModels && (
          <div className="utai-palette__hint">
            {t('models.noInstalledHint')}
          </div>
        )}
      </div>

      {/* Conversion */}
      <div className="utai-palette__category">
        <div className="utai-palette__category-label">{t('palette.conversion')}</div>
        <button
          className="utai-palette__item"
          onClick={() => handleAdd({ type: 'rvc_conversion', label: '', category: 'conversion', color: 'var(--accent-pink)' })}
          draggable
          onDragStart={(e) => handleDragStart(e, { type: 'rvc_conversion', label: '', category: 'conversion', color: 'var(--accent-pink)' })}
          style={{ color: 'var(--accent-pink)', borderColor: 'var(--accent-pink)' }}
        >
          {t('node.rvcConversion')}
        </button>
      </div>

      {/* Processing */}
      <div className="utai-palette__category">
        <div className="utai-palette__category-label">{t('palette.processing')}</div>
        <button
          className="utai-palette__item"
          onClick={() => handleAdd({ type: 'audio_merge', label: '', category: 'processing', color: 'var(--accent-gold)' })}
          draggable
          onDragStart={(e) => handleDragStart(e, { type: 'audio_merge', label: '', category: 'processing', color: 'var(--accent-gold)' })}
          style={{ color: 'var(--accent-gold)', borderColor: 'var(--accent-gold)' }}
        >
          {t('node.audioMerge')}
        </button>
        <button
          className="utai-palette__item"
          onClick={() => handleAdd({ type: 'effects', label: '', category: 'processing', color: 'var(--accent-cyan)' })}
          draggable
          onDragStart={(e) => handleDragStart(e, { type: 'effects', label: '', category: 'processing', color: 'var(--accent-cyan)' })}
          style={{ color: 'var(--accent-cyan)', borderColor: 'var(--accent-cyan)' }}
        >
          {t('node.effects')}
        </button>
      </div>

      {/* Output */}
      <div className="utai-palette__category">
        <div className="utai-palette__category-label">{t('palette.output')}</div>
        <button
          className="utai-palette__item"
          onClick={() => handleAdd({ type: 'media_output', label: '', category: 'output', color: 'var(--accent-orange)' })}
          draggable
          onDragStart={(e) => handleDragStart(e, { type: 'media_output', label: '', category: 'output', color: 'var(--accent-orange)' })}
          style={{ color: 'var(--accent-orange)', borderColor: 'var(--accent-orange)' }}
        >
          {t('node.mediaOutput')}
        </button>
      </div>
    </div>
  )
}

// Export for backward compat
export const PALETTE_ITEMS = [
  { type: 'audio_input', labelKey: 'node.audioInput', category: 'input', color: 'var(--accent-green)' },
  { type: 'separation', labelKey: 'node.separation', category: 'separation', color: 'var(--accent-blue)' },
  { type: 'rvc_conversion', labelKey: 'node.rvcConversion', category: 'conversion', color: 'var(--accent-pink)' },
  { type: 'media_output', labelKey: 'node.mediaOutput', category: 'output', color: 'var(--accent-orange)' },
]
